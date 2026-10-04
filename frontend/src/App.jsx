import React, { useEffect, useRef, useState } from "react";
import {
  Sparkles,
  ArrowUp,
  ArrowUpRight,
  Plus,
  SlidersHorizontal,
  Box,
  ChevronDown,
  ChevronRight,
  Check,
  CheckCircle2,
  LoaderCircle,
  Download,
  ImagePlus,
  X,
  Command,
  Layers3,
  Move3D,
  Triangle,
  Palette,
  Clock3,
  FolderOpen,
  RefreshCw,
  Square,
  Settings2,
  ExternalLink,
  AlertCircle,
  FileBox,
  Copy,
  Activity,
  Maximize2,
} from "lucide-react";
import Viewer from "./Viewer.jsx";

const running = ["queued", "running", "cancelling"];
const delivered = ["completed", "needs_review"];
const names = {
  queued: "排队中",
  running: "生成中",
  cancelling: "停止中",
  completed: "已完成",
  needs_review: "已生成 · 待人工复核",
  failed: "生成失败",
  cancelled: "已停止",
  interrupted: "待恢复",
};
const stages = [
  ["analyze", "理解需求"],
  ["references", "准备参考图"],
  ["reference_review", "参考图复核"],
  ["reconstruct", "图像生成 3D"],
  ["generate", "Blender 整理"],
  ["validate", "网格检查"],
  ["visual_review", "外观复核"],
  ["finish", "交付资产"],
];
const engineNames = {
  neural: "Hunyuan3D 多视图重建",
  procedural: "语言模型 + Blender",
  hunyuan: "Hunyuan3D 2.1",
  triposr: "TripoSR",
};
const assetUrl = (task, type) => `/api/tasks/${task.id}/files/${type}`;
const clearSecrets = (value) => ({ ...value, api_key: "", generative_api_key: "", image_api_key: "", tencent_cos_secret_id: "", tencent_cos_secret_key: "" });
const imageProviderDefaults = {
  fal: { image_model: "fal-ai/qwen-image-2512", image_base_url: "" },
  "aliyun-bailian": { image_model: "wan2.2-t2i-plus", image_base_url: "https://dashscope.aliyuncs.com/api/v1" },
  "tencent-tokenhub": { image_model: "hy-image-v3", image_base_url: "https://tokenhub.tencentmaas.com" },
  siliconflow: { image_model: "", image_base_url: "https://api.siliconflow.cn/v1" },
  "volcengine-ark": { image_model: "doubao-seedream-5-0-lite-260128", image_base_url: "https://ark.cn-beijing.volces.com/api/v3" },
  "openai-compatible": { image_model: "", image_base_url: "" },
};
const reconstructionProviderDefaults = {
  "fal-hunyuan": { reconstruction_model: "fal-ai/hunyuan3d/v2", reconstruction_base_url: "" },
  "aliyun-bailian": { reconstruction_model: "Tripo/Tripo-P1.0", reconstruction_base_url: "" },
  "tencent-tokenhub": { reconstruction_model: "hy-3d-3.1", reconstruction_base_url: "https://tokenhub.tencentmaas.com" },
  "tripo-official": { reconstruction_model: "v3.1-20260211", reconstruction_base_url: "https://openapi.tripo3d.com/v3" },
};
const providerLinks = {
  fal: ["https://fal.ai/dashboard/keys", "fal"],
  "fal-hunyuan": ["https://fal.ai/dashboard/keys", "fal"],
  "aliyun-bailian": ["https://bailian.console.aliyun.com/", "阿里云百炼"],
  "tencent-tokenhub": ["https://console.cloud.tencent.com/tokenhub/keys", "腾讯云 TokenHub"],
  siliconflow: ["https://cloud.siliconflow.cn/account/ak", "硅基流动"],
  "volcengine-ark": ["https://ark.volcengine.com/region:cn-beijing/apikey", "火山引擎方舟"],
  "tripo-official": ["https://developers.tripo3d.com/zh/keys", "Tripo 官方"],
};
const routeName = (task) => task?.route === "image-to-3d"
  ? "参考图片 → 3D"
  : task?.engine === "neural" || task?.route === "text-to-image-to-3d"
    ? "文字 → 三视图 → 3D"
    : "历史建模路线";
const size = (n) =>
  !n
    ? "—"
    : n > 1048576
      ? `${(n / 1048576).toFixed(1)} MB`
      : `${(n / 1024).toFixed(0)} KB`;
async function api(path, options = {}) {
  const { timeout = 15000, ...request } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  let response, result;
  try {
    response = await fetch("/api" + path, {
      ...request,
      signal: controller.signal,
      headers: {
        ...(request.body && !(request.body instanceof FormData)
          ? { "Content-Type": "application/json" }
          : {}),
        ...request.headers,
      },
    });
    result = await response.json();
  } catch (cause) {
    const error = new Error(
      cause.name === "AbortError"
        ? "请求超时。请检查本地服务是否仍在运行，然后重试。"
        : response
          ? "本地服务返回了无效响应，请重新连接后重试。"
          : "无法连接本地服务。请双击项目目录中的 Start-Studio.cmd 启动服务，然后点击重新连接。",
    );
    error.code = "SERVICE_UNAVAILABLE";
    throw error;
  } finally {
    clearTimeout(timer);
  }
  if (!response.ok)
    throw new Error(
      typeof result.detail === "string"
        ? result.detail
        : "请求未通过校验，请检查输入。",
    );
  return result;
}

function Brand() {
  return (
    <div className="brand">
      <span className="brand-symbol">
        <Box size={23} strokeWidth={1.6} />
      </span>
      <span>
        FORMA<span className="brand-dot">.</span>
      </span>
    </div>
  );
}

function Settings({ onClose, onSaved }) {
  const [value, setValue] = useState(null),
    [imageParameters, setImageParameters] = useState("{}"),
    [reconstructionParameters, setReconstructionParameters] = useState("{}"),
    [message, setMessage] = useState(""),
    [messageSection, setMessageSection] = useState(""),
    [loading, setLoading] = useState(true),
    [busy, setBusy] = useState(false);
  async function load() {
    setLoading(true);
    setMessage("");
    try {
      const loaded = await api("/settings");
      setValue(clearSecrets(loaded));
      setImageParameters(JSON.stringify(loaded.image_parameters || {}, null, 2));
      setReconstructionParameters(JSON.stringify(loaded.reconstruction_parameters || {}, null, 2));
      onSaved(loaded);
    } catch (e) {
      setMessage(e.message);
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => { load(); }, []);
  const field = (key, label, placeholder, type = "text") => (
    <label className="field">
      <span>{label}</span>
      <input
        type={type}
        aria-label={label}
        disabled={busy}
        value={value?.[key] || ""}
        placeholder={placeholder}
        onChange={(e) => setValue({ ...value, [key]: e.target.value })}
        autoComplete="off"
      />
    </label>
  );
  function switchImageProvider(provider) {
    setValue({ ...value, image_provider: provider, image_api_key: "", has_image_api_key: false,
      image_generation_available: false, ...imageProviderDefaults[provider] });
    setImageParameters("{}");
  }
  function switchReconstructionProvider(provider) {
    setValue({ ...value, reconstruction_provider: provider, generative_api_key: "", has_generative_api_key: false,
      generation_available: false, ...reconstructionProviderDefaults[provider] });
    setReconstructionParameters("{}");
  }
  async function save(test = "") {
    setBusy(true);
    setMessageSection(test);
    setMessage("");
    let saved = false;
    try {
      let parameters;
      try { parameters = JSON.parse(imageParameters); } catch { throw new Error("文生图额外参数必须是有效的 JSON 对象。"); }
      if (!parameters || Array.isArray(parameters) || typeof parameters !== "object") throw new Error("文生图额外参数必须是 JSON 对象，例如 {}。");
      let threeDParameters;
      try { threeDParameters = JSON.parse(reconstructionParameters); } catch { throw new Error("3D 额外参数必须是有效的 JSON 对象。"); }
      if (!threeDParameters || Array.isArray(threeDParameters) || typeof threeDParameters !== "object") throw new Error("3D 额外参数必须是 JSON 对象，例如 {}。");
      const updated = await api("/settings", {
        method: "PUT",
        body: JSON.stringify({ ...value, image_parameters: parameters, reconstruction_parameters: threeDParameters }),
      });
      onSaved(updated);
      setValue(clearSecrets(updated));
      saved = true;
      if (test) {
        setMessage(test === "language" ? "设置已保存，正在测试语言模型连接…" : "设置已保存，正在检查生成服务配置与凭据…");
        const testPath = { generation: "/settings/test-generation", image: "/settings/test-image", language: "/settings/test" }[test];
        const result = await api(testPath, { method: "POST", timeout: 300000 });
        setMessage(result.message + (result.model ? " · " + result.model : ""));
        const checked = await api("/settings");
        onSaved(checked);
        setValue(clearSecrets(checked));
      } else {
        setMessage("设置已保存");
      }
    } catch (e) {
      setMessage(saved ? "设置已保存，但连接测试未通过：" + e.message : e.message);
      if (test) {
        try { onSaved(await api("/settings")); } catch {}
      }
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <section
        className="settings-modal"
        role="dialog"
        aria-modal="true"
        aria-label="模型与服务设置"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-heading">
          <div>
            <div className="eyebrow">CONNECTIONS</div>
            <h2>模型与服务设置</h2>
          </div>
          <button
            className="icon-button"
            aria-label="关闭设置"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        {value ? (
          <>
            <p className="settings-note">
              配置生成服务与设计助手。密钥由本地后端保存，页面不会读取已保存的密钥。
            </p>
            <section className="settings-group">
              <div className="settings-section-title"><ImagePlus size={16} /><h3>文字生成三视图</h3><span>可更换模型</span></div>
              <div className="model-route-summary">
                <strong>{value.image_model || "fal-ai/qwen-image-2512"}</strong>
                <span>文字生成正面、左侧与背面参考图</span>
              </div>
              <p className="settings-note">可选 fal、阿里云百炼、腾讯云 TokenHub、硅基流动、火山引擎方舟或其他 Images 兼容服务。三视图是概念参考，视角一致性仍需视觉复核。</p>
              <label className="field"><span>文生图平台</span><select aria-label="文生图平台" disabled={busy} value={value.image_provider || "fal"} onChange={(e) => switchImageProvider(e.target.value)}><option value="fal">fal</option><option value="aliyun-bailian">阿里云百炼</option><option value="tencent-tokenhub">腾讯云 TokenHub</option><option value="siliconflow">硅基流动</option><option value="volcengine-ark">火山引擎方舟</option><option value="openai-compatible">其他 OpenAI Images 兼容接口</option></select></label>
              {field("image_model", "文生图模型 ID", value.image_provider === "siliconflow" ? "从硅基流动模型广场复制当前模型 ID" : value.image_provider === "openai-compatible" ? "服务商提供的图像模型 ID" : imageProviderDefaults[value.image_provider]?.image_model || "模型 ID")}
              {value.image_provider !== "fal" && field("image_base_url", "文生图 API 地址", imageProviderDefaults[value.image_provider]?.image_base_url || "https://你的服务地址/v1")}
              {field("image_api_key", "文生图 API Key", value.has_image_api_key ? "当前平台已有密钥；留空保留，切换平台时需重新填写" : "填写当前文生图平台的 API Key", "password")}
              {providerLinks[value.image_provider] && <div className="settings-provider-link"><a href={providerLinks[value.image_provider][0]} target="_blank" rel="noreferrer">前往 {providerLinks[value.image_provider][1]} 管理密钥 <ExternalLink size={11} /></a><span>{value.has_image_api_key ? "当前平台密钥已保存" : "尚未配置"}</span></div>}
              <label className="toggle-field"><input type="checkbox" disabled={busy} checked={value.web_reference_enabled !== false} onChange={(e) => setValue({ ...value, web_reference_enabled: e.target.checked })} /><span>生成前检索公开图片参考</span></label>
              <p className="settings-note compact-note">启用后，应用会在 Wikimedia Commons 检索多个公开候选，再通过视觉模型核验是否确为用户指定的真实对象。只有至少三张候选支持同一对象、且核验置信度达到 80 时才采用。当前腾讯 Hy-Image 3.0 / 3.5 会收到选中的图片；其他文生图平台会保守地退回纯文字生成。</p>
              <details className="advanced-settings"><summary>模型额外参数 <ChevronDown size={14} /></summary><p>按服务商文档填写 size、steps 等参数，必须是 JSON 对象；不同模型支持的字段不同。</p><label className="field"><span>额外参数 JSON</span><textarea aria-label="额外参数 JSON" className="json-parameters" rows={5} disabled={busy} value={imageParameters} spellCheck={false} onChange={(e) => setImageParameters(e.target.value)} /></label></details>
              <button className="secondary-button inline-test" disabled={busy} onClick={() => save("image")}><Activity size={14} />保存并检查文生图配置</button>
              {messageSection === "image" && message && <div className="form-message" role="status" aria-live="polite">{message}</div>}
              <p className="settings-note compact-note">请选择能生成横向三栏图片的模型，并按其文档设置横向尺寸（宽高比至少 1.35）。配置检查不会生成图片，实际三视图能力需在任务中验证。</p>
            </section>
            <section className="settings-group">
              <div className="settings-section-title"><Box size={16} /><h3>图片生成 3D</h3><span>可更换平台</span></div>
              <div className="model-route-summary"><strong>{value.reconstruction_model || "3D 重建模型"}</strong><span>上传图片走单图重建；文字入口生成三视图后走多图重建</span></div>
              <label className="field"><span>图生 3D 平台</span><select aria-label="图生 3D 平台" disabled={busy} value={value.reconstruction_provider || "fal-hunyuan"} onChange={(e) => switchReconstructionProvider(e.target.value)}><option value="fal-hunyuan">fal · Hunyuan3D</option><option value="aliyun-bailian">阿里云百炼 · Tripo</option><option value="tencent-tokenhub">腾讯云 TokenHub · HY-3D</option><option value="tripo-official">Tripo 官方 API</option></select></label>
              {field("reconstruction_model", "3D 模型 ID", reconstructionProviderDefaults[value.reconstruction_provider]?.reconstruction_model || "模型 ID")}
              {value.reconstruction_provider !== "fal-hunyuan" && field("reconstruction_base_url", "3D API 地址", value.reconstruction_provider === "aliyun-bailian" ? "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1" : reconstructionProviderDefaults[value.reconstruction_provider]?.reconstruction_base_url || "https://服务地址")}
              {field("generative_api_key", "图生 3D API Key", value.has_generative_api_key ? "当前平台已有密钥；留空保留，切换平台时需重新填写" : "填写当前 3D 平台的 API Key", "password")}
              <div className="settings-provider-link"><a href={providerLinks[value.reconstruction_provider][0]} target="_blank" rel="noreferrer">前往 {providerLinks[value.reconstruction_provider][1]} 管理密钥 <ExternalLink size={11} /></a><span>{value.has_generative_api_key ? "当前平台密钥已保存" : "尚未配置"}</span></div>
              <label className="toggle-field"><input type="checkbox" disabled={busy} checked={value.reconstruction_textured !== false} onChange={(e) => setValue({ ...value, reconstruction_textured: e.target.checked })} /><span>生成纹理（增加云端用量）</span></label>
              {value.reconstruction_provider === "tencent-tokenhub" && <div className="advanced-settings">
                <h4>腾讯多视图 · COS 临时图片</h4>
                <p className="settings-note">文字生成的左视图与背视图上传到你的私有 COS，通过两小时有效的签名链接交给 HY-3D，云端生成结束后自动清理。单张上传图无需配置。COS 使用独立的腾讯云凭据，不是 TokenHub API Key。</p>
                {field("tencent_cos_bucket", "COS 存储桶", "例如 studio-1250000000（包含 APPID）")}
                {field("tencent_cos_region", "COS 地域", "例如 ap-guangzhou")}
                {field("tencent_cos_secret_id", "COS SecretId", value.has_tencent_cos_credentials ? "已保存；留空保留" : "腾讯云 CAM SecretId", "password")}
                {field("tencent_cos_secret_key", "COS SecretKey", value.has_tencent_cos_credentials ? "已保存；留空保留" : "腾讯云 CAM SecretKey", "password")}
                <p className="settings-note">凭据需有该桶 forma-references/ 前缀的 PutObject、GetObject、DeleteObject 权限。建议为此前缀设置一天后删除的生命周期，清理中断任务残留；COS 可能产生少量存储与流量费用。</p>
                <a href="https://console.cloud.tencent.com/cos/bucket" target="_blank" rel="noreferrer">打开腾讯 COS 控制台</a>
              </div>}
              <details className="advanced-settings"><summary>3D 模型额外参数 <ChevronDown size={14} /></summary><p>{value.reconstruction_provider === "aliyun-bailian" ? "百炼 Tripo 仅支持 texture_quality（standard 或 detailed）；Tripo-H3.1 还支持 geometry_quality（standard 或 ultra）。" : "按平台文档填写 geometry_quality、texture_quality、face_limit、auto_size 等参数；模型、图片、纹理开关和密钥由上方字段管理。"}</p><label className="field"><span>3D 额外参数 JSON</span><textarea aria-label="3D 额外参数 JSON" className="json-parameters" rows={5} disabled={busy} value={reconstructionParameters} spellCheck={false} onChange={(e) => setReconstructionParameters(e.target.value)} /></label></details>
              <button className="secondary-button inline-test" disabled={busy} onClick={() => save("generation")}><Activity size={14} />保存并检查生成服务</button>
              {messageSection === "generation" && message && <div className="form-message" role="status" aria-live="polite">{message}</div>}
              <p className="settings-note compact-note">Tripo 官方支持本地单图和三视图上传，默认使用中国站 v3 API；国际账号可改为 openapi.tripo3d.ai/v3。阿里 Tripo 需填写含 WorkspaceId 的北京地域地址。腾讯单图可直接提交，三视图需填写上方 COS 配置。配置检查不执行付费生成，也不验证 COS 实际上传权限。</p>
            </section>
            <section className="settings-group">
            <div className="settings-section-title"><Sparkles size={16} /><h3>需求理解与视觉复核</h3></div>
            <p className="settings-note">理解文字、整理结构要求，并比较参考图与渲染结果。开启视觉复核时，需要支持图片输入的模型。</p>
            <label className="field">
              <span>接口协议</span>
              <select
                value={value.provider}
                disabled={busy}
                onChange={(e) =>
                  setValue({ ...value, provider: e.target.value })
                }
              >
                <option value="anthropic">Anthropic Messages</option>
                <option value="openai-compatible">OpenAI 兼容接口</option>
              </select>
            </label>
            {field("base_url", "API 地址", "https://…")}
            {field("model", "语言 / 视觉模型名称", "服务商提供的模型 ID")}
            {field(
              "api_key",
              "语言模型 API Key",
              value.has_api_key ? "已配置，留空保留当前密钥" : "输入 API Key",
              "password",
            )}
            <label className="toggle-field"><input type="checkbox" disabled={busy} checked={value.visual_review_enabled !== false} onChange={(e) => setValue({ ...value, visual_review_enabled: e.target.checked })} /><span>检查参考图一致性与模型外观</span></label>
            <label className="field"><span>每阶段自动修正上限</span><select disabled={busy} value={value.max_quality_retries ?? 1} onChange={(e) => setValue({ ...value, max_quality_retries: Number(e.target.value) })}><option value={0}>不自动重试</option><option value={1}>参考图与 3D 各最多重试一次（增加用量）</option></select></label>
            <p className="settings-note compact-note">参考图明确未通过时停止重建；模型外观未通过时保留可用模型，标记待人工复核。</p>
            <button className="secondary-button inline-test" disabled={busy} onClick={() => save("language")}><Activity size={14} />保存并测试语言模型</button>
            {messageSection === "language" && message && <div className="form-message" role="status" aria-live="polite">{message}</div>}
            </section>
            <section className="settings-group">
              <div className="settings-section-title"><Box size={16} /><h3>本地处理</h3></div>
              {field("blender_path", "Blender 路径", "D:\\blender\\blender.exe")}
              <p className="settings-note compact-note">Blender 整理网格、保留材质、渲染检查视图并导出模型。</p>
            </section>
            {message && !messageSection && (
              <div className="form-message" role="status">
                {message}
              </div>
            )}
            <div className="modal-actions">
              <button
                className="primary-button"
                disabled={busy}
                onClick={() => save()}
              >
                {busy ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />} 保存设置
              </button>
            </div>
          </>
        ) : loading ? (
          <div className="loading-section" role="status">
            <LoaderCircle className="spin" />
            <p>正在读取 API 设置…</p>
          </div>
        ) : (
          <div className="settings-unavailable" role="alert">
            <AlertCircle size={28} />
            <strong>暂时无法读取 API 设置</strong>
            <p>{message}</p>
            <button className="primary-button" onClick={load}>
              <RefreshCw size={15} />重新连接
            </button>
          </div>
        )}
      </section>
    </div>
  );
}

function AssetCard({ task, selected, onClick }) {
  return (
    <button
      className={"asset-card " + (selected ? "selected" : "")}
      onClick={onClick}
    >
      <div className="asset-thumb">
        {task.has_preview ? (
          <img src={assetUrl(task, "preview")} alt={task.title} />
        ) : running.includes(task.status) ? (
          <LoaderCircle className="spin" size={27} />
        ) : (
          <Box size={30} />
        )}
        <span className="asset-type">
          {task.is_example
            ? "EXAMPLE"
            : task.parent_id
              ? "VERSION"
              : "GENERATED"}
        </span>
      </div>
      <div className="asset-card-text">
        <strong>{task.title}</strong>
        <span>
          {task.status === "completed"
            ? `${task.report?.triangles?.toLocaleString() || "—"} tris`
            : names[task.status]}
          <ArrowUpRight size={13} />
        </span>
      </div>
    </button>
  );
}

function ReviewResult({ review, title }) {
  if (!review) return null;
  const labels = { passed: "已通过", needs_review: "待人工复核", skipped: "未启用", unavailable: "未能完成" };
  const passed = review.status === "passed";
  return (
    <section className={"review-result " + (passed ? "review-passed" : "review-pending")} aria-label={title}>
      <div className="review-heading">{passed ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}<strong>{title}</strong><span>{labels[review.status] || "检查中"}</span></div>
      {review.summary && <p>{review.summary}</p>}
      {review.issues?.length > 0 && <ul>{review.issues.map((issue, index) => <li key={index}>{typeof issue === "string" ? issue : issue.message || issue.description || JSON.stringify(issue)}</li>)}</ul>}
    </section>
  );
}

function ReferenceViews({ task }) {
  const views = task?.reference_views || [];
  if (!views.length && !task?.reference_sheet_url) return null;
  const labels = { front: "正面", left: "左侧", back: "背面" };
  return (
    <section className="reference-views" aria-label="生成参考图">
      <div className="reference-views-heading"><ImagePlus size={14} /><strong>{task.route === "image-to-3d" ? "输入参考图" : "三视图参考"}</strong>{task.reference_sheet_url && <a href={task.reference_sheet_url} target="_blank" rel="noreferrer">查看全图 <ArrowUpRight size={11} /></a>}</div>
      <div className={"reference-views-grid " + (views.length === 1 ? "single-view" : "")}>
        {views.map((view) => <a key={view.view} href={view.url} target="_blank" rel="noreferrer"><img src={view.url} alt={view.label || labels[view.view] || "参考图"} loading="lazy" /><span>{view.label || labels[view.view]}</span></a>)}
      </div>
      {task.web_reference && <p className="settings-note compact-note">{task.web_reference.used ? <>已采用公开图片参考：{task.web_reference.candidate?.title || "Wikimedia Commons"}。<a href={task.web_reference.candidate?.page_url} target="_blank" rel="noreferrer">查看来源 <ArrowUpRight size={11} /></a></> : <>公开图片参考未采用：{task.web_reference.reason || "未找到足够准确的候选图。"}</>}</p>}
      <p>{task.route === "image-to-3d" ? "用上传图片约束外形与材质，未展示的面由模型推断。" : "生成的概念参考图；不同视角不等于精确工程投影。"}</p>
    </section>
  );
}

export default function App() {
  const [tasks, setTasks] = useState([]),
    [selectedId, setSelectedId] = useState(null),
    [task, setTask] = useState(null),
    [events, setEvents] = useState([]);
  const [prompt, setPrompt] = useState(""),
    [quality, setQuality] = useState("standard"),
    [style, setStyle] = useState("industrial");
  const [image, setImage] = useState(null),
    [sending, setSending] = useState(false),
    [uploading, setUploading] = useState(false),
    [error, setError] = useState("");
  const [showSettings, setShowSettings] = useState(() => new URLSearchParams(location.search).get("settings") === "1"),
    [config, setConfig] = useState(null),
    [backendError, setBackendError] = useState(""),
    [wireframe, setWireframe] = useState(false),
    [page, setPage] = useState("studio"),
    [downloadOpen, setDownloadOpen] = useState(false);
  const [search, setSearch] = useState(""),
    [filter, setFilter] = useState("all"),
    [downloading, setDownloading] = useState(false);
  const fileInput = useRef(null),
    promptInput = useRef(null),
    backendConnected = useRef(false),
    initialized = useRef(false);
  async function loadTasks() {
    try {
      const data = await api("/tasks");
      setTasks(data);
      if (!backendConnected.current) {
        setConfig(await api("/settings"));
        backendConnected.current = true;
      }
      setBackendError("");
      if (!initialized.current && data.length) {
        initialized.current = true;
        setSelectedId(
          data.find((t) => !t.is_example)?.id || data[0].id,
        );
      }
    } catch (e) {
      if (e.code === "SERVICE_UNAVAILABLE") {
        backendConnected.current = false;
        setBackendError(e.message);
      } else {
        setError(e.message);
      }
    }
  }
  useEffect(() => {
    loadTasks();
    const interval = setInterval(loadTasks, 4000);
    return () => clearInterval(interval);
  }, []);
  useEffect(() => {
    if (!selectedId) {
      setTask(null);
      setEvents([]);
      return;
    }
    let active = true,
      source;
    setEvents([]);
    api("/tasks/" + selectedId)
      .then((result) => {
        if (!active) return;
        setTask(result);
        setEvents(result.events || []);
        if (running.includes(result.status)) {
          source = new EventSource(`/api/tasks/${selectedId}/events`);
          source.addEventListener("progress", (e) => {
            const entry = JSON.parse(e.data);
            setEvents((previous) =>
              previous.some((p) => p.seq === entry.seq)
                ? previous
                : [...previous, entry],
            );
          });
          source.addEventListener("task", (e) => {
            const value = JSON.parse(e.data);
            setTask(value);
            setTasks((prev) =>
              prev.map((p) => (p.id === value.id ? value : p)),
            );
            if (!running.includes(value.status)) {
              source.close();
              loadTasks();
            }
          });
        }
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
      source?.close();
    };
  }, [selectedId]);
  const current =
    task?.id === selectedId ? task : tasks.find((t) => t.id === selectedId);
  const active = current && running.includes(current.status);
  const ready = delivered.includes(current?.status);
  const report = current?.report;
  const completeCount = tasks.filter((t) => delivered.includes(t.status)).length;
  const reconstructionProvider = current?.pipeline_options?.reconstruction_provider || config?.reconstruction_provider || "fal-hunyuan";
  const reconstructionName = { "fal-hunyuan": "fal · Hunyuan3D", "aliyun-bailian": "阿里云 · Tripo", "tencent-tokenhub": "腾讯云 · HY-3D", "tripo-official": "Tripo 官方" }[reconstructionProvider] || "图生 3D";
  const setupIssue = !config ? "正在连接本地服务…"
    : !config.generation_available ? "先配置图片生成 3D 服务"
      : !image && !config.image_generation_available ? "先配置文生图服务"
        : !config.has_api_key ? "先配置需求分析模型"
          : !config.blender_available ? "先配置本机 Blender" : "";
  const taskStages = current?.engine === "neural" ? stages : [
    ["analyze", "理解需求"], ["design", "设计方案"], ["generate", "构建模型"], ["validate", "检查与交付"],
  ];
  function choose(t) {
    setTask(t);
    setSelectedId(t.id);
    setPage("studio");
    setError("");
    setDownloadOpen(false);
  }
  const [deletingId, setDeletingId] = useState(null);
  async function deleteAsset(asset) {
    if (running.includes(asset.status)) {
      window.alert('请先取消任务并等待执行停止，再删除资产。');
      return;
    }
    if (!window.confirm(`删除资产“${asset.title}”？\n将永久删除该资产的模型、材质、参考图、渲染图和执行记录，无法撤销。`)) return;
    setDeletingId(asset.id);
    try {
      await api('/tasks/' + asset.id, { method: 'DELETE' });
      setTasks(previous => previous.filter(item => item.id !== asset.id));
      if (selectedId === asset.id) {
        setSelectedId(null);
        setTask(null);
        setDownloadOpen(false);
      }
    } catch (e) {
      window.alert('删除失败：' + e.message);
    } finally {
      setDeletingId(null);
    }
  }
  function fresh() {
    initialized.current = true;
    setSelectedId(null);
    setTask(null);
    setPrompt("");
    setImage(null);
    setPage("studio");
    setError("");
    setTimeout(() => promptInput.current?.focus(), 50);
  }
  async function upload(file) {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const data = new FormData();
      data.append("file", file);
      setImage(await api("/uploads", { method: "POST", body: data }));
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }
  async function submit() {
    if ((!prompt.trim() && !image) || sending || uploading) return;
    if (setupIssue) { setShowSettings(true); return; }
    setSending(true);
    setError("");
    try {
      const next = await api("/tasks", {
        method: "POST",
        headers: { "Idempotency-Key": crypto.randomUUID() },
        body: JSON.stringify({
          prompt: prompt.trim() || "根据参考图片生成 3D 模型",
          quality,
          style,
          engine: "neural",
          image_id: image?.id || null,
        }),
      });
      initialized.current = true;
      setTasks((prev) => [next, ...prev]);
      setTask(next);
      setSelectedId(next.id);
      setPrompt("");
      setImage(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setSending(false);
    }
  }
  async function action(name) {
    try {
      await api(`/tasks/${current.id}/${name}`, { method: "POST" });
      const id = current.id;
      setSelectedId(null);
      await loadTasks();
      setTimeout(() => setSelectedId(id), 30);
    } catch (e) {
      setError(e.message);
    }
  }
  async function downloadAsset(event, url, filename) {
    event.preventDefault();
    if (downloading) return;
    setDownloading(true);
    setError("");
    try {
      // Fetch a JSON envelope so browser download-manager integrations cannot
      // intercept the response body before it becomes a local Blob.
      const response = await fetch(url + (url.includes("?") ? "&" : "?") + "transport=json");
      if (!response.ok) throw new Error("下载失败，请重试");
      const payload = await response.json();
      const bytes = Uint8Array.from(atob(payload.base64), c => c.charCodeAt(0));
      if (!bytes.length) throw new Error("下载内容为空，请重试");
      const blob = new Blob([bytes], {type: payload.mime});
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 60000);
    } catch (e) {
      setError(e.message);
    } finally {
      setDownloading(false);
      setDownloadOpen(false);
    }
  }
  const materials = Object.entries(current?.materials || {});
  return (
    <div className="app-shell">
      <header className="topbar">
        <Brand />
        <div className="brand-divider" />
        <span className="workspace-name">
          个人工作空间
          <ChevronDown size={12} />
        </span>
        <nav>
          <button
            className={page === "studio" ? "nav-active" : ""}
            onClick={() => setPage("studio")}
          >
            设计工作台
          </button>
          <button
            className={page === "library" ? "nav-active" : ""}
            onClick={() => setPage("library")}
          >
            资产库<span>{completeCount}</span>
          </button>
        </nav>
        <div className="top-actions">
          <span className="connection-state">
            <span
              className={
                !backendError && config?.generation_available && config?.image_generation_available && config?.has_api_key && config?.blender_available
                  ? "live-dot"
                  : "offline-dot"
              }
            />
            {backendError ? "本地服务未连接" : config?.generation_available && config?.image_generation_available ? "生成服务已配置" : "生成服务待配置"}
          </span>
          <button
            className="icon-button settings-trigger"
            aria-label="模型设置"
            title="API 设置"
            onClick={() => setShowSettings(true)}
          >
            <SlidersHorizontal size={18} />
            <span>API 设置</span>
          </button>
          <div className="avatar">F</div>
        </div>
      </header>
      {backendError && (
        <div className="service-banner" role="alert">
          <AlertCircle size={17} />
          <span>{backendError}</span>
          <button className="secondary-button" onClick={loadTasks}>重新连接</button>
        </div>
      )}
      {page === "library" ? (
        <main className="library">
          <div className="library-heading">
            <div>
              <div className="eyebrow">YOUR COLLECTION</div>
              <h1>让想法，有迹可循。</h1>
              <p>所有设计、生成结果与迭代版本，都保存在这里。</p>
            </div>
            <button className="primary-button" onClick={fresh}>
              <Plus size={17} />
              创建新资产
            </button>
          </div>
          <div className="library-filters">
            <div>
              <button
                className={filter === "all" ? "chosen" : ""}
                onClick={() => setFilter("all")}
              >
                全部资产
              </button>
              <button
                className={filter === "generated" ? "chosen" : ""}
                onClick={() => setFilter("generated")}
              >
                我的生成
              </button>
              <button
                className={filter === "example" ? "chosen" : ""}
                onClick={() => setFilter("example")}
              >
                示例设计
              </button>
            </div>
            <input
              aria-label="搜索资产"
              placeholder="搜索你的设计…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <div className="library-grid">
            {tasks
              .filter(
                (t) =>
                  (filter === "all" ||
                    (filter === "example" ? t.is_example : !t.is_example)) &&
                  t.title.toLowerCase().includes(search.toLowerCase()),
              )
              .map((t) => (
                <div key={t.id} className="library-asset-item">
                  <AssetCard task={t} onClick={() => choose(t)} />
                  <button className="secondary-button asset-delete-button" disabled={!!deletingId} onClick={() => deleteAsset(t)} aria-label={`删除资产：${t.title}`}>{deletingId === t.id ? '正在删除…' : '删除资产'}</button>
                </div>
              ))}
          </div>
          {!tasks.length && (
            <div className="empty-library">
              还没有资产，创建你的第一个 3D 设计。
            </div>
          )}
        </main>
      ) : (
        <>
          <main className="workspace">
            <aside className="design-panel">
              <div className="panel-top">
                <span>
                  <Sparkles size={15} />
                  设计助手
                </span>
                <button
                  className="icon-button"
                  title="新建设计"
                  aria-label="新建设计"
                  onClick={fresh}
                >
                  <Plus size={18} />
                </button>
              </div>
              <div className="conversation">
                <div className="intro">
                  <div className="eyebrow">
                    <span />
                    IDEA INTO FORM
                  </div>
                  <h1>
                    从一个想法，
                    <br />
                    到一个立体世界<span>。</span>
                  </h1>
                  <p>
                    文字先生成三视图，图片直接进入重建。
                    <br />
                    从参考图到可探索、可带走的 3D 资产。
                  </p>
                </div>
                {current ? (
                  <div className="task-conversation">
                    <div className="conversation-label">
                      <span className="mini-avatar">
                        {current.is_example ? (
                          <Box size={12} />
                        ) : (
                          <Command size={12} />
                        )}
                      </span>
                      {current.is_example ? "示例设计简报" : "你的设计需求"}
                      <span className="time-stamp">
                        {new Date(current.created_at * 1000).toLocaleTimeString(
                          "zh-CN",
                          { hour: "2-digit", minute: "2-digit" },
                        )}
                      </span>
                    </div>
                    <div className="user-bubble">{current.prompt}</div>
                    <div className="task-route"><ImagePlus size={12} />{routeName(current)}</div>
                    <div className="assistant-heading">
                      <Sparkles size={14} />
                      <span>FORMA</span>
                      <span className="assistant-tag">DESIGN AGENT</span>
                    </div>
                    {current.analysis && (
                      <>
                        <p className="assistant-summary">
                          {current.analysis.summary}
                        </p>
                        <div className="requirement-tags">
                          {current.analysis.requirements?.map((r, i) => (
                            <span key={i}>{r}</span>
                          ))}
                        </div>
                      </>
                    )}
                    <div
                      className={
                        "task-state " +
                        (current.status === "failed" ? "failed" : current.status === "needs_review" ? "review-pending" : "")
                      }
                    >
                      <div>
                        {active ? (
                          <LoaderCircle className="spin" size={15} />
                        ) : ready && current.status !== "needs_review" ? (
                          <CheckCircle2 size={15} />
                        ) : (
                          <AlertCircle size={15} />
                        )}
                        <strong>
                          {current.is_example
                            ? "可交互示例已就绪"
                            : names[current.status]}
                        </strong>
                        {ready && (
                          <span>
                            {Math.round(current.elapsed_seconds || 0)}s
                          </span>
                        )}
                      </div>
                      {active && (
                        <>
                          <p>
                            {events.at(-1)?.message || "任务已进入生成队列"}
                          </p>
                          <div className="progress-track">
                            <i style={{ width: `${current.progress || 3}%` }} />
                          </div>
                          <button
                            className="text-button stop-button"
                            onClick={() => action("cancel")}
                            disabled={current.status === "cancelling"}
                          >
                            <Square size={10} />
                            停止任务
                          </button>
                        </>
                      )}
                      {current.error && <p>{current.error}</p>}
                      {current.status === "failed" && current.can_proceed_unverified_reference && (
                        <div className="reference-override-actions">
                          <p>三视图未通过自动校验。继续生成可能得到结构错误、视角混乱或缺失部件的模型，最终结果会标记为待人工复核。</p>
                          <button
                            className="secondary-button"
                            onClick={() => action("proceed-unverified-reference")}
                          >
                            <Move3D size={13} />
                            仍然生成 3D
                          </button>
                        </div>
                      )}
                      {current.status === "needs_review" && <p>模型已生成，可预览与下载；外观一致性尚未确认，请查看下方复核结果。</p>}
                      {["failed", "cancelled", "interrupted"].includes(
                        current.status,
                      ) && (
                        <button
                          className="text-button"
                          onClick={() => action("retry")}
                        >
                          <RefreshCw size={12} />
                          继续 / 重试
                        </button>
                      )}
                    </div>
                    <ReferenceViews task={current} />
                    <ReviewResult title="参考图复核" review={current.reference_review} />
                    <ReviewResult title="模型外观复核" review={current.visual_review} />
                    {events.length > 0 && (
                      <details className="trace">
                        <summary>
                          执行轨迹<span>{events.length} 个事件</span>
                          <ChevronDown size={12} />
                        </summary>
                        <div className="trace-list">
                          {events.map((e) => (
                            <div key={e.seq}>
                              <i />
                              <span>{e.message}</span>
                              <time>
                                {new Date(e.time * 1000).toLocaleTimeString(
                                  "zh-CN",
                                  {
                                    hour: "2-digit",
                                    minute: "2-digit",
                                    second: "2-digit",
                                  },
                                )}
                              </time>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                    {ready && current.engine !== "neural" && <p className="legacy-note">历史资产可继续预览与下载。新任务使用参考图重建路线。</p>}
                  </div>
                ) : (
                  <div className="starter">
                    <div className="small-label">从这里开始</div>
                    {[
                      "一个复古橙色太空探测车，带六个轮子",
                      "一把胡桃木与奶油色软垫休闲椅",
                      "一盏极简金属蘑菇台灯",
                    ].map((text) => (
                      <button
                        key={text}
                        onClick={() => {
                          setPrompt(text);
                          promptInput.current?.focus();
                        }}
                      >
                        <span>{text}</span>
                        <ArrowUpRight size={13} />
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div className="composer-area">
                <div className="route-guide" aria-label="当前生成路线"><span>{image ? "图片生成 3D" : "文字生成 3D"}</span><p>{image ? "参考图片 → 图像重建 → 检查与交付" : "文字 → 三视图 → 图像重建 → 检查与交付"}</p></div>
                {setupIssue && <button className="setup-prompt" onClick={() => setShowSettings(true)}><SlidersHorizontal size={14} /><span>{setupIssue}</span><ArrowUpRight size={13} /></button>}
                {image && (
                  <div className="reference-image">
                    <img src={image.url} alt="参考图片" />
                    <span>参考图片已添加</span>
                    <button
                      aria-label="移除参考图片"
                      onClick={() => setImage(null)}
                    >
                      <X size={13} />
                    </button>
                  </div>
                )}
                {error && (
                  <div className="error-message" role="alert">
                    <AlertCircle size={14} />
                    <span>{error}</span>
                    <button aria-label="关闭错误" onClick={() => setError("")}>
                      <X size={12} />
                    </button>
                  </div>
                )}
                <div className="composer">
                  <textarea
                    ref={promptInput}
                    aria-label="设计描述"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder={image ? "可补充外形、材质要求，也可直接生成…" : "描述物体的外形、材质和风格，AI 将先生成三视图…"}
                    maxLength={5000}
                    onKeyDown={(e) => {
                      if ((e.ctrlKey || e.metaKey) && e.key === "Enter")
                        submit();
                    }}
                  />
                  <div className="composer-bottom">
                    <input
                      type="file"
                      accept="image/png,image/jpeg,image/webp"
                      hidden
                      ref={fileInput}
                      onChange={(e) => upload(e.target.files[0])}
                    />
                    <button
                      className="icon-button"
                      aria-label="上传参考图"
                      title="上传参考图"
                      disabled={uploading}
                      onClick={() => fileInput.current.click()}
                    >
                      {uploading ? (
                        <LoaderCircle className="spin" size={17} />
                      ) : (
                        <ImagePlus size={18} />
                      )}
                    </button>
                    <span>{image ? "图片重建" : "三视图驱动"}</span>
                    <button
                      className="send-button"
                      aria-label="生成模型"
                      disabled={(!prompt.trim() && !image) || sending || uploading || !!backendError || !config}
                      title={setupIssue || "开始生成"}
                      onClick={submit}
                    >
                      {sending ? (
                        <LoaderCircle className="spin" size={17} />
                      ) : (
                        <ArrowUp size={20} />
                      )}
                    </button>
                  </div>
                </div>
                <div className="generation-controls">
                  <span className="neural-engine"><Layers3 size={12} />{reconstructionName}</span>
                  <select
                    aria-label="预览渲染质量"
                    value={quality}
                    onChange={(e) => setQuality(e.target.value)}
                  >
                    <option value="draft">快速预览</option>
                    <option value="standard">标准预览</option>
                    <option value="high">精细预览</option>
                  </select>
                  <span title="Ctrl + Enter 生成">⌘ ↵</span>
                </div>
                <div className="composer-footnote">
                  云端生成 · Blender 整理
                  <span>本地保存</span>
                </div>
              </div>
            </aside>
            <section className="center-panel">
              <div className="viewport-header">
                <div>
                  <span className="breadcrumb">工作空间 / </span>
                  <strong>{current?.title || "未命名设计"}</strong>
                  {current?.is_example && (
                    <span className="example-badge">示例</span>
                  )}
                </div>
                <span className="viewport-mode">
                  <span />
                  3D 视图
                </span>
                {ready && (
                  <a
                    className="compact-download"
                    href={`/api/tasks/${current.id}/download`}
                    onClick={(e) =>
                      downloadAsset(
                        e,
                        `/api/tasks/${current.id}/download`,
                        `forma-${current.id.slice(0, 8)}.zip`,
                      )
                    }
                    aria-label="下载当前资产"
                  >
                    <Download size={16} />
                  </a>
                )}
              </div>
              <div className="viewport-area">
                <Viewer
                  url={ready ? assetUrl(current, "glb") : null}
                  wireframe={wireframe}
                  setWireframe={setWireframe}
                />
                {!ready && (
                  <div className="viewport-empty">
                    <div className={"empty-orb " + (active ? "breathing" : "")}>
                      <Box size={48} strokeWidth={0.9} />
                    </div>
                    <div className="eyebrow">
                      {active ? "SHAPING YOUR IDEA" : "A NEW DIMENSION"}
                    </div>
                    <h2>{active ? "灵感正在成形" : "每个形体，始于想象。"}</h2>
                    <p>
                      {active
                        ? "正在生成参考、重建模型并检查结果"
                        : "在左侧描述你的想法，或打开下方示例。"}
                    </p>
                    {active && (
                      <div className="stage-pills">
                        {taskStages.map(([key, label], i) => (
                          <span
                            className={
                              current.stage === key
                                ? "current"
                                : taskStages.findIndex(
                                      (s) => s[0] === current.stage,
                                    ) > i
                                  ? "done"
                                  : ""
                            }
                            key={key}
                          >
                            {taskStages.findIndex((s) => s[0] === current.stage) >
                            i ? (
                              <Check size={11} />
                            ) : (
                              <i />
                            )}
                            {label}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                {ready && (
                  <div className="model-caption">
                    <div className="eyebrow">
                      {current.is_example ? "CURATED EXAMPLE" : "YOUR CREATION"}
                    </div>
                    <h2>{current.title.split("·")[0]}</h2>
                    <span>
                      {current.is_example
                        ? "真实建模示例 · 可预览与下载"
                        : routeName(current)}
                    </span>
                  </div>
                )}
              </div>
              <div className="asset-strip">
                <div className="strip-heading">
                  <span>
                    <Layers3 size={14} />
                    最近的资产 <b>{tasks.length.toString().padStart(2, "0")}</b>
                  </span>
                  <button onClick={() => setPage("library")}>
                    查看全部
                    <ArrowUpRight size={12} />
                  </button>
                </div>
                <div className="asset-row">
                  {tasks.slice(0, 8).map((t) => (
                    <AssetCard
                      key={t.id}
                      task={t}
                      selected={selectedId === t.id}
                      onClick={() => choose(t)}
                    />
                  ))}
                  <button className="new-asset-card" onClick={fresh}>
                    <Plus size={23} />
                    <span>新的可能</span>
                  </button>
                </div>
              </div>
            </section>
            <aside className="inspector">
              <div className="panel-top">
                <span>资产详情</span>
                <Settings2 size={15} />
              </div>
              <div className="inspector-scroll">
                <div className="object-heading">
                  <div className="object-icon">
                    <Box size={22} />
                  </div>
                  <div>
                    <h3>{current?.title || "等待创建"}</h3>
                    <span>
                      {current?.id
                        ? `ASSET / ${current.id.slice(0, 8).toUpperCase()}`
                        : "YOUR NEXT CREATION"}
                    </span>
                  </div>
                </div>
                <div className="detail-section">
                  <div className="detail-label">生成配置</div>
                  <div className="detail-pair">
                    <span>引擎</span>
                    <strong>
                      {current?.engine === "neural" ? `${reconstructionName} · ${current.route === "image-to-3d" ? "单图重建" : "多视图重建"}` : current ? engineNames[current.engine] || current.engine : reconstructionName}
                    </strong>
                  </div>
                  <div className="detail-pair">
                    <span>{current?.engine === "neural" || !current ? "预览质量" : "质量"}</span>
                    <strong>
                      {
                        { draft: "快速草稿", standard: "标准", high: "精细" }[
                          current?.quality || quality
                        ]
                      }
                    </strong>
                  </div>
                  <div className="detail-pair">
                    <span>状态</span>
                    <strong className={current?.status === "completed" ? "green-text" : ""}>
                      {current ? names[current.status] : "准备就绪"}
                    </strong>
                  </div>
                  <label className="style-label">
                    设计风格
                    <select
                      aria-label="设计风格"
                      value={style}
                      onChange={(e) => setStyle(e.target.value)}
                    >
                      <option value="industrial">未来工业</option>
                      <option value="minimal">现代极简</option>
                      <option value="playful">趣味卡通</option>
                      <option value="realistic">写实产品</option>
                    </select>
                  </label>
                </div>
                <div className="detail-section">
                  <div className="detail-label">
                    模型数据 <Triangle size={12} />
                  </div>
                  <div className="metrics-grid">
                    <div>
                      <span>三角面</span>
                      <strong>
                        {report?.triangles?.toLocaleString() || "—"}
                      </strong>
                    </div>
                    <div>
                      <span>顶点</span>
                      <strong>
                        {report?.vertices?.toLocaleString() || "—"}
                      </strong>
                    </div>
                    <div>
                      <span>部件</span>
                      <strong>{report?.objects || "—"}</strong>
                    </div>
                    <div>
                      <span>文件大小</span>
                      <strong>{size(report?.file_bytes)}</strong>
                    </div>
                  </div>
                  {report?.dimensions && (
                    <div className="dimensions">
                      <Move3D size={13} />
                      {report.dimensions.map((v) => v.toFixed(2)).join(" × ")} {current?.engine === "neural" ? "场景单位" : "m"}
                    </div>
                  )}
                </div>
                <div className="detail-section">
                  <div className="detail-label">
                    材质调色板 <span>{materials.length || "—"}</span>
                  </div>
                  <div className="material-palette">
                    {materials.length ? (
                      materials.map(([name, m]) => (
                        <div key={name} title={`${name} ${m.color}`}>
                          <i style={{ background: m.color }} />
                          <span>{name.replaceAll("_", " ")}</span>
                        </div>
                      ))
                    ) : (
                      <p className="muted small">生成后显示材质信息</p>
                    )}
                  </div>
                </div>
                {report && (
                  <div className="detail-section">
                    <div className="detail-label">
                      网格检查 <span className={report.checks?.every((check) => check.passed) ? "green-text" : ""}>{report.checks?.every((check) => check.passed) ? "已通过" : "查看报告"}</span>
                    </div>
                    <div className="checks">
                      {report.checks?.map((c, i) => (
                        <div key={i} title={c.detail}>
                          {c.passed ? (
                            <CheckCircle2 size={13} />
                          ) : (
                            <AlertCircle size={13} />
                          )}
                          <span>{c.label}</span>
                        </div>
                      ))}
                    </div>
                    <details className="quality-details">
                      <summary>
                        查看报告与说明
                        <ChevronDown size={12} />
                      </summary>
                      {report.warnings?.map((w, i) => (
                        <p key={i}>{w}</p>
                      ))}
                      <a
                        href={assetUrl(current, "report")}
                        target="_blank"
                        rel="noreferrer"
                      >
                        打开完整报告
                        <ExternalLink size={11} />
                      </a>
                    </details>
                  </div>
                )}
                {current?.parent_id && (
                  <button
                    className="parent-link"
                    onClick={() => {
                      const parent = tasks.find(
                        (t) => t.id === current.parent_id,
                      );
                      if (parent) choose(parent);
                    }}
                  >
                    查看上一个版本 <ArrowUpRight size={12} />
                  </button>
                )}
              </div>
              <div className="export-area">
                {current && <button className="secondary-button asset-delete-button" disabled={!!deletingId} onClick={() => deleteAsset(current)}>{deletingId === current.id ? '正在删除…' : '删除当前资产'}</button>}
                <div className="export-formats">
                  <span>交付格式</span>
                  <strong>
                    {current?.formats?.length
                      ? current.formats.slice(0, 3).join(" / ").toUpperCase()
                      : "GLB / OBJ / STL"}
                  </strong>
                </div>
                <div className="export-buttons">
                  <a
                    className={
                      "primary-button download-main " +
                      (!ready ? "disabled" : "")
                    }
                    href={
                      ready ? `/api/tasks/${current.id}/download` : undefined
                    }
                    aria-disabled={!ready || downloading}
                    onClick={(e) =>
                      ready &&
                      downloadAsset(
                        e,
                        `/api/tasks/${current.id}/download`,
                        `forma-${current.id.slice(0, 8)}.zip`,
                      )
                    }
                  >
                    {downloading ? (
                      <LoaderCircle size={16} className="spin" />
                    ) : (
                      <Download size={16} />
                    )}{" "}
                    {downloading ? "准备下载…" : "下载资产包"}
                  </a>
                  <button
                    aria-label="选择下载格式"
                    className="download-options"
                    disabled={!ready}
                    onClick={() => setDownloadOpen(!downloadOpen)}
                  >
                    <ChevronDown size={16} />
                  </button>
                </div>
                {downloadOpen && ready && (
                  <div className="download-menu">
                    {current.formats?.map((format) => (
                      <a
                        key={format}
                        href={assetUrl(current, format) + "?download=true"}
                        onClick={(e) =>
                          downloadAsset(
                            e,
                            assetUrl(current, format) + "?download=true",
                            `forma-${current.id.slice(0, 8)}.${format}`,
                          )
                        }
                      >
                        <FileBox size={14} />
                        {format.toUpperCase()}
                        <Download size={12} />
                      </a>
                    ))}
                  </div>
                )}
                <p>包含模型、材质与检查报告</p>
              </div>
            </aside>
          </main>
        </>
      )}
      <footer className="statusbar">
        <span>
          <span className="live-dot" />
          所有资产保存在本机
        </span>
        <span>
          FORMA STUDIO <i /> 让想象触手可及
        </span>
        <span>
          {tasks.filter((t) => running.includes(t.status)).length} 个进行中{" "}
          <i /> {completeCount} 个已完成
        </span>
      </footer>
      {showSettings && (
        <Settings onClose={() => setShowSettings(false)} onSaved={setConfig} />
      )}
    </div>
  );
}
