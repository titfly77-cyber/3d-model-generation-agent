// UI contract tests only. Every API response is mocked; no paid generation occurs.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '../frontend/node_modules/playwright/index.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dist = path.join(root, 'frontend', 'dist');
const out = path.join(root, 'data', 'verification', 'neural-ui');
fs.mkdirSync(out, { recursive: true });
fs.mkdirSync(path.join(root, 'tmp'), { recursive: true });
fs.mkdirSync(path.join(root, 'cache'), { recursive: true });
for (const name of ['TEMP', 'TMP', 'TMPDIR']) process.env[name] = path.join(root, 'tmp');
const server = http.createServer((request, response) => {
  const pathname = new URL(request.url, 'http://localhost').pathname;
  const target = path.resolve(dist, '.' + (pathname === '/' ? '/index.html' : pathname));
  if (!target.startsWith(dist + path.sep) || !fs.existsSync(target) || !fs.statSync(target).isFile()) {
    response.writeHead(404).end(); return;
  }
  const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
  response.writeHead(200, { 'Content-Type': mime[path.extname(target)] || 'application/octet-stream' });
  fs.createReadStream(target).pipe(response);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
const errors = [];
const submitted = [];
const settingsWrites = [];
const checks = [];
const overrideCalls = [];
let settings = {
  provider: 'openai-compatible', base_url: 'https://example.invalid/v1', model: 'mock-vision', has_api_key: true,
  connection_status: 'connected', blender_path: 'D:\\blender\\blender.exe', blender_available: true,
  image_provider: 'fal', image_model: 'fal-ai/qwen-image-2512', image_base_url: '', image_parameters: {},
  has_image_api_key: false, image_generation_available: false,
  has_generative_api_key: true, generation_available: true, reconstruction_provider: 'fal-hunyuan',
  reconstruction_textured: true, visual_review_enabled: true, max_quality_retries: 1, pipeline_version: 2,
};
let tasks = [];
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII=', 'base64');
// An existing local sample only exercises the viewer, never represents cloud output.
const sampleGlb = path.join(root, 'data', 'assets', 'f950d435a0435bd1ad92e6205d22e883', 'model.glb');
function resultFor(input) {
  const id = String(tasks.length + 1).padStart(32, '0');
  const imageRoute = !!input.image_id;
  return {
    ...input, id, title: imageRoute ? '图片路线 UI 测试资产' : '文字路线 UI 测试资产',
    created_at: Date.now() / 1000, elapsed_seconds: 1, status: 'needs_review', stage: 'finish', progress: 100,
    route: imageRoute ? 'image-to-3d' : 'text-to-image-to-3d', has_preview: true, is_example: false,
    formats: ['glb'], report: { triangles: 12, vertices: 8, objects: 1, checks: [{ label: '测试网格检查', passed: true }] },
    reference_views: (imageRoute ? ['front'] : ['front', 'left', 'back']).map(view => ({
      view, label: imageRoute ? '输入图片' : { front: '正面', left: '左侧', back: '背面' }[view], url: `/api/tasks/${id}/files/reference_${view}`,
    })),
    reference_sheet_url: imageRoute ? null : `/api/tasks/${id}/files/reference_sheet`,
    reference_review: { status: 'passed', summary: '这是模拟的参考图复核响应。', issues: [] },
    visual_review: { status: 'needs_review', summary: '这是模拟的外观复核响应。', issues: ['背面细节需要人工检查'] }, events: [],
  };
}
try {
  browser = await chromium.launchPersistentContext(fs.mkdtempSync(path.join(root, 'cache', 'neural-ui-')), {
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
    viewport: { width: 1600, height: 1050 }, deviceScaleFactor: 1, acceptDownloads: true, downloadsPath: path.join(root, 'tmp'),
    args: ['--no-first-run', '--disable-extensions', '--disable-background-networking', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });
  const page = await browser.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const pathname = url.pathname;
    const json = value => route.fulfill({ json: value });
    if (pathname === '/api/settings' && request.method() === 'GET') return json(settings);
    if (pathname === '/api/settings' && request.method() === 'PUT') {
      const body = request.postDataJSON(); settingsWrites.push(body);
      const { api_key, image_api_key, generative_api_key, tencent_cos_secret_id, tencent_cos_secret_key, ...publicValues } = body;
      settings = { ...settings, ...publicValues, has_image_api_key: !!image_api_key || settings.has_image_api_key, image_generation_available: !!image_api_key || settings.image_generation_available };
      return json(settings);
    }
    if (pathname.startsWith('/api/settings/test')) return json({ message: '模拟配置检查通过，未调用真实服务' });
    if (pathname === '/api/uploads') return json({ id: 'upload-fixture', url: '/api/uploads/upload-fixture' });
    if (pathname.startsWith('/api/uploads/')) return route.fulfill({ contentType: 'image/png', body: png });
    if (pathname === '/api/tasks' && request.method() === 'GET') return json(tasks);
    if (pathname === '/api/tasks' && request.method() === 'POST') {
      const body = request.postDataJSON(); submitted.push(body);
      const result = resultFor(body); tasks = [result, ...tasks]; return json(result);
    }
    const overridePath = pathname.match(/^\/api\/tasks\/([^/]+)\/proceed-unverified-reference$/);
    if (overridePath && request.method() === 'POST') {
      overrideCalls.push(overridePath[1]);
      tasks = tasks.map(task => task.id === overridePath[1]
        ? { ...task, can_proceed_unverified_reference: false }
        : task);
      return json(tasks.find(task => task.id === overridePath[1]));
    }
    const taskPath = pathname.match(/^\/api\/tasks\/([^/]+)$/);
    if (taskPath && request.method() === 'DELETE') {
      tasks = tasks.filter(task => task.id !== taskPath[1]);
      return json({ deleted: true, id: taskPath[1] });
    }
    if (taskPath) return json(tasks.find(task => task.id === taskPath[1]));
    if (pathname.endsWith('/files/glb')) return route.fulfill({ contentType: 'model/gltf-binary', body: fs.readFileSync(sampleGlb) });
    if (pathname.includes('/files/')) return route.fulfill({ contentType: 'image/png', body: png });
    if (pathname.endsWith('/download')) return json({ mime: 'application/zip', base64: Buffer.from('UI-DOWNLOAD-FIXTURE').toString('base64') });
    throw new Error(`Unexpected API request: ${request.method()} ${pathname}`);
  });
  await page.goto(`http://127.0.0.1:${server.address().port}`, { waitUntil: 'networkidle' });
  await page.getByRole('textbox', { name: '设计描述', exact: true }).fill('一台橙色太空车');
  await page.getByRole('button', { name: '生成模型', exact: true }).click();
  await page.getByRole('dialog').waitFor();
  assert.equal(submitted.length, 0, 'Missing image service must open settings before generation');
  checks.push('missing text-to-image service opens settings');

  const passwords = page.locator('input[type=password]');
  for (const input of await passwords.all()) assert.equal(await input.inputValue(), '');
  await page.getByLabel('文生图平台', { exact: true }).waitFor();
  const imageProviderValues = await page.getByLabel('文生图平台', { exact: true }).locator('option').evaluateAll(options => options.map(option => option.value));
  assert.ok(imageProviderValues.includes('siliconflow') && imageProviderValues.includes('volcengine-ark'));
  assert.ok(imageProviderValues.includes('aliyun-bailian') && imageProviderValues.includes('tencent-tokenhub'));
  const reconstructionProviderValues = await page.getByLabel('图生 3D 平台', { exact: true }).locator('option').evaluateAll(options => options.map(option => option.value));
  assert.ok(reconstructionProviderValues.includes('tripo-official'));
  await page.getByLabel('文生图平台', { exact: true }).selectOption('openai-compatible');
  await page.getByLabel('文生图模型 ID', { exact: true }).fill('custom-image-model');
  await page.getByLabel('文生图 API 地址', { exact: true }).fill('https://image.example.invalid/v1');
  await page.getByLabel('文生图 API Key', { exact: true }).fill('ui-test-secret-not-real');
  await page.getByText('模型额外参数', { exact: true }).click();
  await page.getByLabel('额外参数 JSON', { exact: true }).fill('[]');
  await page.getByRole('button', { name: '保存设置', exact: true }).click();
  await page.getByText('文生图额外参数必须是 JSON 对象，例如 {}。', { exact: true }).waitFor();
  assert.equal(settingsWrites.length, 0);
  await page.getByLabel('额外参数 JSON', { exact: true }).fill('{"size":"1536x1024"}');
  await page.getByRole('button', { name: '保存并检查文生图配置', exact: true }).click();
  await page.getByText('模拟配置检查通过，未调用真实服务', { exact: true }).waitFor();
  const imageGroup = page.locator('.settings-group').filter({ has: page.getByLabel('文生图平台', { exact: true }) });
  assert.ok((await imageGroup.getByRole('status').innerText()).includes('模拟配置检查通过'));
  assert.equal(settingsWrites.at(-1).image_model, 'custom-image-model');
  assert.deepEqual(settingsWrites.at(-1).image_parameters, { size: '1536x1024' });
  for (const input of await passwords.all()) assert.equal(await input.inputValue(), '');
  assert.ok(!(await page.locator('body').innerText()).includes('ui-test-secret-not-real'));
  await page.screenshot({ path: path.join(out, 'mock-settings-desktop.png'), fullPage: true });
  checks.push('custom provider/model/JSON parameters save; secrets cleared; invalid JSON blocked');
  await page.getByRole('button', { name: '关闭设置', exact: true }).click();

  await page.getByRole('button', { name: '生成模型', exact: true }).click();
  await page.locator('.reference-views-grid img').nth(2).waitFor();
  assert.equal(submitted[0].engine, 'neural');
  assert.equal(submitted[0].image_id, null);
  assert.ok(!submitted[0].parent_id);
  await page.getByText('背面细节需要人工检查', { exact: true }).waitFor();
  assert.equal(await page.locator('.reference-views-grid img').count(), 3);
  const downloadLink = page.getByRole('link', { name: '下载资产包', exact: true });
  assert.equal(await downloadLink.getAttribute('aria-disabled'), 'false');
  const downloadPromise = page.waitForEvent('download');
  await downloadLink.click();
  const download = await downloadPromise;
  await download.saveAs(path.join(out, 'mock-download.zip'));
  assert.equal(fs.readFileSync(path.join(out, 'mock-download.zip'), 'utf8'), 'UI-DOWNLOAD-FIXTURE');
  await page.screenshot({ path: path.join(out, 'mock-reference-review-desktop.png'), fullPage: true });
  checks.push('text sends neural; three views and review issues shown; needs_review remains downloadable');

  await page.getByRole('button', { name: '新建设计', exact: true }).click();
  await page.locator('input[type=file]').setInputFiles({ name: 'fixture.png', mimeType: 'image/png', buffer: png });
  await page.getByText('参考图片已添加', { exact: true }).waitFor();
  assert.equal(await page.getByRole('textbox', { name: '设计描述', exact: true }).inputValue(), '');
  assert.ok(await page.getByRole('button', { name: '生成模型', exact: true }).isEnabled());
  await page.getByRole('button', { name: '生成模型', exact: true }).click();
  await page.locator('.reference-views-grid.single-view img').waitFor();
  assert.equal(submitted[1].engine, 'neural');
  assert.equal(submitted[1].image_id, 'upload-fixture');
  assert.equal(submitted[1].prompt, '根据参考图片生成 3D 模型');
  assert.equal(await page.locator('.reference-views-grid img').count(), 1);
  assert.equal(await page.getByRole('button', { name: '继续修改这个设计', exact: true }).count(), 0);
  checks.push('image upload submits empty prompt as neural image-to-3D; legacy edit unavailable');

  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
  assert.ok(await page.getByRole('link', { name: '下载当前资产', exact: true }).isVisible());
  await page.screenshot({ path: path.join(out, 'mock-mobile.png'), fullPage: true });
  await page.getByRole('button', { name: '模型设置', exact: true }).click();
  await page.getByRole('dialog').waitFor();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
  await page.screenshot({ path: path.join(out, 'mock-settings-mobile.png'), fullPage: true });
  checks.push('390px mobile layout and download; mobile settings no horizontal overflow');
  await page.getByLabel('图生 3D 平台', { exact: true }).selectOption('tencent-tokenhub');
  await page.getByLabel('COS 存储桶', { exact: true }).fill('studio-1250000000');
  await page.getByLabel('COS 地域', { exact: true }).fill('ap-guangzhou');
  await page.getByLabel('COS SecretId', { exact: true }).fill('ui-cos-id');
  await page.getByLabel('COS SecretKey', { exact: true }).fill('ui-cos-secret');
  await page.getByRole('button', { name: '保存并检查生成服务', exact: true }).click();
  const generationGroup = page.locator('.settings-group').filter({ has: page.getByLabel('图生 3D 平台', { exact: true }) });
  await generationGroup.getByRole('status').filter({ hasText: '模拟配置检查通过' }).waitFor();
  assert.equal(settingsWrites.at(-1).tencent_cos_bucket, 'studio-1250000000');
  assert.equal(settingsWrites.at(-1).tencent_cos_secret_key, 'ui-cos-secret');
  assert.equal(await page.getByLabel('COS SecretId', { exact: true }).inputValue(), '');
  assert.equal(await page.getByLabel('COS SecretKey', { exact: true }).inputValue(), '');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
  checks.push('Tencent COS configuration saves on mobile; secrets cleared; result remains beside generation check');
  await page.getByRole('button', { name: '关闭设置', exact: true }).click();
  await page.setViewportSize({ width: 1600, height: 1050 });
  const beforeDelete = tasks.length;
  page.once('dialog', dialog => dialog.dismiss());
  await page.getByRole('button', { name: '删除当前资产', exact: true }).click();
  assert.equal(tasks.length, beforeDelete);
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: '删除当前资产', exact: true }).click();
  await page.getByRole('button', { name: '删除当前资产', exact: true }).waitFor({ state: 'hidden' });
  assert.equal(tasks.length, beforeDelete - 1);
  checks.push('asset deletion requires confirmation and clears selected preview after success');
  const failedId = '9'.repeat(32);
  tasks = [{ id: failedId, title: '三视图待确认', prompt: '测试未通过参考图', created_at: Date.now() / 1000,
    status: 'failed', stage: 'reference_review', progress: 33, engine: 'neural', route: 'text-to-image-to-3d',
    can_proceed_unverified_reference: true, error: '参考图检查未通过，已暂停后续 3D 生成。',
    reference_review: { status: 'needs_review', summary: '视角不一致', issues: ['视角不一致'] }, events: [] }, ...tasks];
  await page.reload({ waitUntil: 'networkidle' });
  const proceedButton = page.getByRole('button', { name: '仍然生成 3D', exact: true });
  await proceedButton.waitFor();
  await proceedButton.click();
  assert.deepEqual(overrideCalls, [failedId]);
  await proceedButton.waitFor({ state: 'hidden' });
  checks.push('failed reference review offers an explicit user-controlled continue-to-3D action');
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(out, 'result.json'), JSON.stringify({ mode: 'mock-api-no-paid-generation', checks, errors, submitted: submitted.map(({ engine, image_id }) => ({ engine, image_id })) }, null, 2));
  console.log(JSON.stringify({ checks, errors }));
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
