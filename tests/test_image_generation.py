import base64
import io
import json

import httpx
import pytest
from PIL import Image

from backend import image_generation as generation


PROMPT = "Same object, three panels: front, left, back."
KEY = "unit-test-private-key"
FAL = "https://queue.fal.run/fal-ai/qwen-image-2512"
REQUEST_ID = "reference-job-001"
CDN = "https://v3.fal.media/files/reference.png"


def png_bytes(size=(768, 256), image_format="PNG"):
    image = Image.new("RGB", size, "white")
    image.paste("orange", (40, 40, 210, 220))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture
def fal_cfg():
    return {"image_provider": "fal", "image_model": generation.DEFAULT_MODEL,
            "image_api_key": KEY, "image_parameters": {}}


@pytest.fixture
def openai_cfg():
    return {"image_provider": "openai-compatible", "image_model": "my-image-model-v2",
            "image_base_url": "https://image.example/v1", "image_api_key": KEY,
            "image_parameters": {"size": "1536x1024", "quality": "high"}}


@pytest.fixture
def install_http(monkeypatch):
    original = httpx.Client
    monkeypatch.setattr(generation, "POLL_SECONDS", 0)

    def install(handler):
        calls = []

        def record(request):
            calls.append(request)
            return handler(request)

        monkeypatch.setattr(generation.httpx, "Client", lambda **kwargs: original(
            **kwargs, transport=httpx.MockTransport(record)))
        return calls

    return install


def fal_response(request, *, result=None, endpoint=FAL):
    url = str(request.url)
    if request.method == "POST" and url == endpoint:
        return httpx.Response(200, json={"request_id": REQUEST_ID})
    if url == endpoint + "/requests/" + REQUEST_ID + "/status":
        return httpx.Response(200, json={"status": "COMPLETED"})
    if url == endpoint + "/requests/" + REQUEST_ID:
        return httpx.Response(200, json=result or {"images": [{"url": CDN}], "seed": 42})
    if url == CDN:
        return httpx.Response(200, content=png_bytes())
    raise AssertionError(f"Unexpected mocked request {request.method} {url}")


def test_fal_default_generates_sheet_and_reuses_saved_image(tmp_path, fal_cfg, install_http):
    calls = install_http(fal_response)
    output, job = tmp_path / "sheet.png", tmp_path / "job.json"
    progress = []
    result = generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None, progress.append, job)
    assert result["model"] == "fal-ai/qwen-image-2512"
    assert (result["width"], result["height"]) == (768, 256)
    assert result["seed"] == 42 and not result["reused"]
    submitted = json.loads(calls[0].content)
    assert submitted["image_size"] == {"width": 1664, "height": 928}
    assert submitted["prompt"] == PROMPT
    assert calls[0].headers["authorization"] == "Key " + KEY
    assert "authorization" not in calls[-1].headers
    assert KEY not in job.read_text(encoding="utf-8")
    with Image.open(output) as image:
        assert image.format == "PNG" and image.mode == "RGB"
    previous_calls = len(calls)
    reused = generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None, job_path=job)
    assert reused["reused"] and len(calls) == previous_calls


def test_fal_can_change_model_without_qwen_specific_parameters(tmp_path, fal_cfg, install_http):
    endpoint = "https://queue.fal.run/example-owner/other-model/fast"
    model = "example-owner/other-model/fast"
    fal_cfg.update(image_model=model, image_parameters={"steps": 8, "prompt": "wrong"})
    calls = install_http(lambda r: fal_response(r, endpoint=endpoint))
    progress = []
    result = generation.generate_sheet(PROMPT, tmp_path / "sheet.png", fal_cfg, lambda: None, progress.append)
    assert result["model"] == model
    assert json.loads(calls[0].content) == {"steps": 8, "prompt": PROMPT}
    assert all("Qwen" not in text for text in progress)


def test_fal_accepts_app_root_queue_urls_for_nested_models(tmp_path, fal_cfg, install_http):
    fal_cfg["image_model"] = "example/model/version"
    base = "https://queue.fal.run/example/model/requests/" + REQUEST_ID

    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"request_id": REQUEST_ID, "status_url": base + "/status",
                                            "response_url": base + "/response", "cancel_url": base + "/cancel"})
        if str(request.url) == base + "/status":
            return httpx.Response(200, json={"status": "COMPLETED"})
        if str(request.url) == base + "/response":
            return httpx.Response(200, json={"images": [{"url": CDN}]})
        assert str(request.url) == CDN
        return httpx.Response(200, content=png_bytes())

    install_http(handler)
    assert generation.generate_sheet(PROMPT, tmp_path / "sheet.png", fal_cfg, lambda: None)["request_id"] == REQUEST_ID


def test_fal_resume_after_poll_network_failure_does_not_resubmit(tmp_path, fal_cfg, install_http):
    failures = [True]

    def handler(request):
        if request.url.path.endswith("/status") and failures[0]:
            failures[0] = False
            raise httpx.ConnectError("upstream might echo " + KEY, request=request)
        return fal_response(request)

    calls = install_http(handler)
    output = tmp_path / "sheet.png"
    with pytest.raises(generation.ImageGenerationError) as error:
        generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None)
    assert KEY not in str(error.value)
    generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None)
    assert len([c for c in calls if c.method == "POST"]) == 1


def test_ambiguous_submission_is_not_repeated(tmp_path, fal_cfg, install_http):
    def handler(request):
        raise httpx.ReadTimeout("request may have succeeded: " + KEY, request=request)

    calls = install_http(handler)
    output = tmp_path / "sheet.png"
    with pytest.raises(generation.ImageGenerationError, match="连接中断"):
        generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None)
    with pytest.raises(generation.ImageGenerationError, match="提交结果不明"):
        generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None)
    assert len(calls) == 1


def test_provider_timeout_status_is_not_automatically_resubmitted(tmp_path, fal_cfg, install_http):
    calls = install_http(lambda r: httpx.Response(408, json={"error": "late response"}))
    output = tmp_path / "sheet.png"
    with pytest.raises(generation.ImageGenerationError, match="408"):
        generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None)
    with pytest.raises(generation.ImageGenerationError, match="提交结果不明"):
        generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None)
    assert len(calls) == 1


def test_auth_error_redacted_and_can_retry_after_key_correction(tmp_path, fal_cfg, install_http):
    denied = [True]

    def handler(request):
        if denied[0]:
            return httpx.Response(401, json={"detail": "upstream echoed " + KEY})
        return fal_response(request)

    calls = install_http(handler)
    output = tmp_path / "sheet.png"
    with pytest.raises(generation.ImageGenerationError) as error:
        generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None)
    assert "401" in str(error.value) and KEY not in str(error.value)
    denied[0] = False
    fal_cfg["image_api_key"] = "replacement-key"
    generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None)
    assert len([c for c in calls if c.method == "POST"]) == 2


def test_fal_cancellation_exception_is_preserved_and_remote_cancelled(tmp_path, fal_cfg, install_http):
    cancellation = httpx.ConnectError("callback cancellation sentinel")
    submitted = [False]

    def handler(request):
        if request.method == "POST":
            submitted[0] = True
            return httpx.Response(200, json={"request_id": REQUEST_ID})
        assert request.method == "PUT" and request.url.path.endswith("/cancel")
        return httpx.Response(202, json={"status": "CANCELLATION_REQUESTED"})

    def check():
        if submitted[0]:
            raise cancellation

    calls = install_http(handler)
    job = tmp_path / "job.json"
    with pytest.raises(httpx.ConnectError) as error:
        generation.generate_sheet(PROMPT, tmp_path / "sheet.png", fal_cfg, check, job_path=job)
    assert error.value is cancellation
    assert [c.method for c in calls] == ["POST", "PUT"]
    assert json.loads(job.read_text(encoding="utf-8"))["state"] == "cancelled"


def test_progress_callback_error_propagates_and_existing_job_resumes(tmp_path, fal_cfg, install_http):
    first_poll = [True]

    def handler(request):
        if request.url.path.endswith("/status") and first_poll[0]:
            first_poll[0] = False
            return httpx.Response(200, json={"status": "IN_PROGRESS"})
        return fal_response(request)

    failure = ValueError("progress sink failed")

    def progress(message):
        if "正在生成三视图" in message:
            raise failure

    calls = install_http(handler)
    output = tmp_path / "sheet.png"
    with pytest.raises(ValueError) as error:
        generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None, progress)
    assert error.value is failure
    generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None)
    assert len([c for c in calls if c.method == "POST"]) == 1


def test_queue_timeout_is_bounded_and_retains_job(tmp_path, fal_cfg, install_http, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(generation, "DEADLINE_SECONDS", 3.0)
    monkeypatch.setattr(generation, "POLL_SECONDS", 2.0)
    monkeypatch.setattr(generation.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(generation.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def handler(request):
        if request.method == "POST":
            return fal_response(request)
        return httpx.Response(200, json={"status": "IN_QUEUE"})

    calls = install_http(handler)
    job = tmp_path / "job.json"
    with pytest.raises(generation.ImageGenerationError, match="超时"):
        generation.generate_sheet(PROMPT, tmp_path / "sheet.png", fal_cfg, lambda: None, job_path=job)
    assert json.loads(job.read_text(encoding="utf-8"))["request_id"] == REQUEST_ID
    assert len([c for c in calls if c.method == "POST"]) == 1


def test_openai_compatible_b64_model_and_parameters(tmp_path, openai_cfg, install_http):
    openai_cfg["image_parameters"].update(prompt="must not override", model="must-not-override")
    calls = install_http(lambda r: httpx.Response(200, json={"data": [{
        "b64_json": base64.b64encode(png_bytes(image_format="JPEG")).decode()}]}))
    result = generation.generate_sheet(PROMPT, tmp_path / "sheet.png", openai_cfg, lambda: None)
    assert len(calls) == 1
    assert str(calls[0].url) == "https://image.example/v1/images/generations"
    assert calls[0].headers["authorization"] == "Bearer " + KEY
    assert json.loads(calls[0].content) == {"model": "my-image-model-v2", "prompt": PROMPT,
                                          "quality": "high", "size": "1536x1024"}
    assert result["provider"] == "openai-compatible" and result["model"] == "my-image-model-v2"
    with Image.open(tmp_path / "sheet.png") as image:
        assert image.format == "PNG"


def test_openai_url_download_can_resume_without_duplicate_generation(tmp_path, openai_cfg, install_http, monkeypatch):
    monkeypatch.setattr(generation.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("1.1.1.1", 443))])
    fail = [True]
    url = "https://images.example/generated.png"

    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"data": [{"url": url}]})
        assert str(request.url) == url and "authorization" not in request.headers
        if fail[0]:
            fail[0] = False
            raise httpx.ConnectError("download failed", request=request)
        return httpx.Response(200, content=png_bytes())

    calls = install_http(handler)
    output = tmp_path / "sheet.png"
    with pytest.raises(generation.ImageGenerationError):
        generation.generate_sheet(PROMPT, output, openai_cfg, lambda: None)
    generation.generate_sheet(PROMPT, output, openai_cfg, lambda: None)
    assert len([c for c in calls if c.method == "POST"]) == 1


@pytest.mark.parametrize("change", [
    {"image_model": "new-model"},
    {"image_base_url": "https://different.example/v1"},
    {"image_parameters": {"size": "1024x1024"}},
    {"image_provider": "fal", "image_model": "fal-ai/other"},
])
def test_job_identity_prevents_cross_model_provider_and_parameter_reuse(tmp_path, openai_cfg, install_http, change):
    calls = install_http(lambda r: httpx.Response(200, json={"data": [{
        "b64_json": base64.b64encode(png_bytes()).decode()}]}))
    output = tmp_path / "sheet.png"
    generation.generate_sheet(PROMPT, output, openai_cfg, lambda: None)
    openai_cfg.update(change)
    with pytest.raises(generation.ImageGenerationError, match="不同"):
        generation.generate_sheet(PROMPT, output, openai_cfg, lambda: None)
    assert len(calls) == 1


@pytest.mark.parametrize("model", ["https://evil.example/model", "owner/../model", "owner/model?key=x"])
def test_invalid_fal_endpoint_is_rejected_before_request(tmp_path, fal_cfg, install_http, model):
    calls = install_http(lambda r: pytest.fail("No HTTP request expected"))
    fal_cfg["image_model"] = model
    with pytest.raises(generation.ImageGenerationError, match="模型名称"):
        generation.generate_sheet(PROMPT, tmp_path / "sheet.png", fal_cfg, lambda: None)
    assert not calls


def test_provider_cannot_exfiltrate_key_with_queue_url(tmp_path, fal_cfg, install_http):
    calls = install_http(lambda r: httpx.Response(200, json={"request_id": REQUEST_ID,
        "status_url": "https://evil.example/steal"}))
    with pytest.raises(generation.ImageGenerationError, match="任务地址"):
        generation.generate_sheet(PROMPT, tmp_path / "sheet.png", fal_cfg, lambda: None)
    assert len(calls) == 1


def test_private_image_download_is_rejected(tmp_path, openai_cfg, install_http, monkeypatch):
    monkeypatch.setattr(generation.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 443))])
    calls = install_http(lambda r: httpx.Response(200, json={"data": [{"url": "https://local.example/secret"}]}))
    with pytest.raises(generation.ImageGenerationError, match="内网"):
        generation.generate_sheet(PROMPT, tmp_path / "sheet.png", openai_cfg, lambda: None)
    assert len(calls) == 1


@pytest.mark.parametrize("image_bytes", [b"not an image", png_bytes((128, 128))])
def test_unusable_image_not_saved(tmp_path, openai_cfg, install_http, image_bytes):
    install_http(lambda r: httpx.Response(200, json={"data": [{
        "b64_json": base64.b64encode(image_bytes).decode()}]}))
    output = tmp_path / "sheet.png"
    with pytest.raises(generation.ImageGenerationError):
        generation.generate_sheet(PROMPT, output, openai_cfg, lambda: None)
    assert not output.exists()


def test_oversized_download_rejected_without_reading_body(tmp_path, fal_cfg, install_http):
    def handler(request):
        if str(request.url) == CDN:
            return httpx.Response(200, headers={"Content-Length": str(generation.MAX_IMAGE_BYTES + 1)})
        return fal_response(request)

    install_http(handler)
    output = tmp_path / "sheet.png"
    with pytest.raises(generation.ImageGenerationError, match="32 MB"):
        generation.generate_sheet(PROMPT, output, fal_cfg, lambda: None)
    assert not output.exists()


def test_transparent_reference_uses_white_background():
    buffer = io.BytesIO()
    Image.new("RGBA", (768, 256), (0, 0, 0, 0)).save(buffer, format="PNG")
    normalized = generation._normalized_image(buffer.getvalue())
    assert normalized.mode == "RGB" and normalized.getpixel((0, 0)) == (255, 255, 255)
