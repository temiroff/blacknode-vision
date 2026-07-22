"""Generic vision nodes for Blacknode.

This package stays camera- and robot-neutral. ROS 2 transport is handled by
blacknode-ros2; these nodes add reusable vision prompts, status views, and an
optional OpenAI-compatible VLM call for one captured frame.
"""
from __future__ import annotations

import base64
import html
import json
import mimetypes
import os
from pathlib import Path
import textwrap
import urllib.error
import urllib.request
from typing import Any

from blacknode.pkg.blacknode_perception import cv2_runtime

from blacknode import streams as bn_streams
from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Image, Int, List, Text, node

_CATEGORY = "Perception"
_OPENAI_COMPATIBLE_DEFAULT_ENDPOINT = "https://api.openai.com/v1"
_OPENAI_COMPATIBLE_DEFAULT_MODEL = "gpt-4o-mini"
# Nemotron Nano VL is NVIDIA's own open-weight vision-language model -- the
# default for provider=nvidia, distinct from the generic OpenAI-compatible
# fallback. (Cosmos Reason1/Reason2 are also NVIDIA open models built
# specifically for physical AI/robot reasoning, but as of this writing NIM
# hosts them as gated "functions" that 404 for accounts without separate
# access approval; Nemotron Nano VL works out of the box with just an
# NVIDIA_API_KEY.)
_NVIDIA_NIM_DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1"
_NVIDIA_NIM_DEFAULT_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
_PROVIDER_DEFAULT_ENDPOINTS = {
    "ollama": "http://127.0.0.1:11434",
    "anthropic": "https://api.anthropic.com/v1",
    "nvidia": _NVIDIA_NIM_DEFAULT_ENDPOINT,
    "openai-compatible": _OPENAI_COMPATIBLE_DEFAULT_ENDPOINT,
}
_PROVIDER_DEFAULT_MODELS = {
    "ollama": "qwen3-vl:4b",
    "anthropic": "claude-sonnet-4-5",
    "nvidia": _NVIDIA_NIM_DEFAULT_MODEL,
    "openai-compatible": _OPENAI_COMPATIBLE_DEFAULT_MODEL,
}


def _image_kind(value: str) -> str:
    if not value:
        return "empty"
    if value.startswith("data:image/"):
        return "data-url"
    if value.startswith(("http://", "https://")):
        return "url"
    return "path-or-text"


def _clip(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _wrap_text(value: Any, width: int = 68, max_lines: int = 3) -> list[str]:
    text = " ".join(str(value or "").split())
    if not text:
        return [""]
    lines = textwrap.wrap(
        text,
        width=max(12, width),
        break_long_words=True,
        break_on_hyphens=False,
    ) or [text]
    if len(lines) <= max_lines:
        return lines
    kept = lines[:max_lines]
    kept[-1] = _clip(kept[-1], max(8, width - 3))
    if not kept[-1].endswith("..."):
        kept[-1] = kept[-1][: max(0, width - 3)].rstrip() + "..."
    return kept


def _svg_data(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _image_data_url(image: str) -> tuple[str, str]:
    """Return (image_data_url, error) for dashboard-safe embedding."""
    value = image.strip()
    if not value:
        return "", ""
    if value.startswith("data:image/"):
        return value, ""
    try:
        media_type, data, _kind = _image_data_parts(value)
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}"
    return f"data:{media_type};base64,{data}", ""


def _normal_provider(value: Any, endpoint_url: str = "") -> str:
    raw = str(value or "openai-compatible").strip().lower().replace("_", "-")
    if raw in {"nvidia", "nim"}:
        return "nvidia"
    if raw in {"openai", "openai-compatible", "openrouter"}:
        return "openai-compatible"
    if raw in {"anthropic", "claude"}:
        return "anthropic"
    if raw in {"ollama", "local", "local-ollama"}:
        return "ollama"
    if raw == "auto":
        endpoint = endpoint_url.lower()
        if "11434" in endpoint or "ollama" in endpoint:
            return "ollama"
        if "anthropic.com" in endpoint:
            return "anthropic"
        if "integrate.api.nvidia.com" in endpoint:
            return "nvidia"
        return "openai-compatible"
    return "openai-compatible"


def _default_model(provider: str, model: str) -> str:
    raw = model.strip()
    if raw and raw not in _PROVIDER_DEFAULT_MODELS.values():
        return raw
    return _PROVIDER_DEFAULT_MODELS.get(provider, _OPENAI_COMPATIBLE_DEFAULT_MODEL)


def _default_endpoint(provider: str, endpoint_url: str) -> str:
    endpoint = endpoint_url.strip().rstrip("/")
    if endpoint and endpoint not in _PROVIDER_DEFAULT_ENDPOINTS.values():
        return endpoint
    return _PROVIDER_DEFAULT_ENDPOINTS.get(provider, _OPENAI_COMPATIBLE_DEFAULT_ENDPOINT)


def _read_url_bytes(url: str) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "BlacknodeVision/0.1"})
    with urllib.request.urlopen(req, timeout=30) as response:
        media_type = response.headers.get_content_type() or "image/jpeg"
        if media_type.startswith("multipart/"):
            raise ValueError(
                f"{url} is a live MJPEG stream ({media_type}), not a single frame, "
                "and would never finish downloading. Use the stream's snapshot_url instead."
            )
        return response.read(), media_type


def _image_data_parts(image: str) -> tuple[str, str, str]:
    """Return (media_type, base64_data, source_kind) for data URL, URL, or path."""
    value = image.strip()
    if value.startswith("data:"):
        header, data = value.split(",", 1)
        media_type = header[5:].split(";", 1)[0] or "image/jpeg"
        return media_type, data, "data-url"
    if value.startswith(("http://", "https://")):
        raw, media_type = _read_url_bytes(value)
        return media_type, base64.b64encode(raw).decode("ascii"), "url"

    path = Path(value).expanduser()
    raw = path.read_bytes()
    media_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return media_type, base64.b64encode(raw).decode("ascii"), "path"


def _anthropic_image_source(image: str) -> dict[str, Any]:
    value = image.strip()
    if value.startswith(("http://", "https://")):
        return {"type": "url", "url": value}
    media_type, data, _kind = _image_data_parts(value)
    return {"type": "base64", "media_type": media_type, "data": data}


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in payload.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float = 90.0) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_http_error(exc: urllib.error.HTTPError, limit: int = 300) -> str:
    raw = exc.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw)
        error = payload.get("error") if isinstance(payload, dict) else None
        message = str((error or {}).get("message") or "").strip() if isinstance(error, dict) else ""
        error_type = str((error or {}).get("type") or (error or {}).get("code") or "").strip() if isinstance(error, dict) else ""
    except (json.JSONDecodeError, TypeError):
        message, error_type = "", ""

    if exc.code == 429 and error_type in {"insufficient_quota", "insufficient_quota_error"}:
        return (
            "HTTP 429: out of API quota/billing on this provider. "
            "Add credits at your provider's billing page, or switch the "
            "provider input to ollama for a free local model."
        )
    if exc.code == 429:
        return f"HTTP 429: rate limited{f' ({_clip(message, limit)})' if message else ''}. Wait a moment and retry."
    if exc.code == 401:
        return f"HTTP 401: invalid or missing API key{f' ({_clip(message, limit)})' if message else ''}."
    if exc.code == 403:
        return f"HTTP 403: API key lacks permission for this model/endpoint{f' ({_clip(message, limit)})' if message else ''}."
    if message:
        return f"HTTP {exc.code}: {_clip(message, limit)}"
    return f"HTTP {exc.code}: {_clip(raw, limit)}"


def _format_connection_error(exc: urllib.error.URLError, provider: str, endpoint: str, model: str) -> str:
    if provider == "ollama":
        return (
            f"could not reach Ollama at {endpoint} ({exc.reason}). "
            f"Install it from https://ollama.com, run `ollama pull {model}`, "
            "and make sure `ollama serve` is running, or switch the provider input."
        )
    return f"could not reach {endpoint} ({exc.reason})"


@node(
    name="FramePrompt",
    category=_CATEGORY,
    description="Build a concise VLM prompt for a camera frame and robot task.",
    inputs={
        "image": Image(default=""),
        "question": Text(default="What is visible in this camera frame?"),
        "context": Text(default=""),
        "robot_task": Text(default=""),
        "include_safety_checks": Bool(default=True),
    },
    outputs={"prompt": Text, "summary": Dict},
)
def vision_frame_prompt(ctx: dict) -> dict:
    image = str(ctx.get("image") or "").strip()
    question = str(ctx.get("question") or "What is visible in this camera frame?").strip()
    context = str(ctx.get("context") or "").strip()
    robot_task = str(ctx.get("robot_task") or "").strip()
    include_safety = bool(ctx.get("include_safety_checks", True))

    parts = [
        "You are inspecting one robot camera frame.",
        "Answer with concrete visual observations, not guesses.",
    ]
    if context:
        parts.append(f"Scene/context: {context}")
    if robot_task:
        parts.append(f"Robot task: {robot_task}")
    if include_safety:
        parts.append("Call out obstacles, people, cables, glass, liquids, unstable objects, and any uncertainty.")
    parts.append(f"Question: {question}")
    parts.append("Return: short summary, visible evidence, uncertainty, and next useful robot action.")

    kind = _image_kind(image)
    return {
        "prompt": "\n".join(parts),
        "summary": {
            "has_image": kind != "empty",
            "image_kind": kind,
            "question": question,
            "context": context,
            "robot_task": robot_task,
            "safety_checks": include_safety,
        },
    }


@node(
    name="DetectionPrompt",
    category=_CATEGORY,
    description="Build an LLM prompt from CV2 detections so local text models can reason about robot actions.",
    inputs={
        "detection": Dict(default={}),
        "detections": List(default=[]),
        "question": Text(default="What should the robot do next?"),
        "context": Text(default=""),
        "robot_task": Text(default="track and approach the visible target"),
    },
    outputs={"prompt": Text, "summary": Dict},
)
def vision_detection_prompt(ctx: dict) -> dict:
    detection = ctx.get("detection") if isinstance(ctx.get("detection"), dict) else {}
    detections = ctx.get("detections") if isinstance(ctx.get("detections"), list) else []
    question = str(ctx.get("question") or "What should the robot do next?").strip()
    context = str(ctx.get("context") or "").strip()
    robot_task = str(ctx.get("robot_task") or "").strip()
    payload = {
        "primary_detection": detection,
        "detections": detections,
    }
    parts = [
        "You are a robot vision assistant using an attached camera frame plus structured CV2 detections.",
        "First describe what is visible in the camera frame in plain language.",
        "Then use the CV2 detections to identify the tracked target, confidence, and next robot action.",
        "Do not invent objects that are not present in the detections.",
    ]
    if context:
        parts.append(f"Scene/context: {context}")
    if robot_task:
        parts.append(f"Robot task: {robot_task}")
    parts.append("Detection data:")
    parts.append(json.dumps(payload, indent=2, sort_keys=True))
    parts.append(f"Question: {question}")
    parts.append("Return: visible scene, tracked target state, confidence, uncertainty, and next robot action.")
    return {
        "prompt": "\n".join(parts),
        "summary": {
            "found": bool(detection.get("found")) if isinstance(detection, dict) else False,
            "detection_count": len(detections),
            "question": question,
            "robot_task": robot_task,
        },
    }


@node(
    name="CameraDashboard",
    category=_CATEGORY,
    description="Render camera stream readiness as a dashboard image.",
    inputs={
        "frame_stream": Dict(default={}),
        "camera_topic": Text(default="/camera/image_raw"),
        "stream_url": Text(default=""),
        "streaming": Bool(default=False),
        "run_report": Text(default=""),
        "stream_report": Text(default=""),
    },
    outputs={"dashboard": Image, "ready": Bool, "report": Text},
)
def vision_stream_status(ctx: dict) -> dict:
    topic = str(ctx.get("camera_topic") or "/camera/image_raw")
    stream_url = bn_streams.source_url(ctx.get("frame_stream"), str(ctx.get("stream_url") or ""))
    streaming = bool(ctx.get("streaming", False))
    run_report = str(ctx.get("run_report") or "")
    stream_report = str(ctx.get("stream_report") or "")
    ready = streaming and bool(stream_url)

    color = "#18a058" if ready else "#f59e0b"
    status = "LIVE" if ready else "WAITING"
    report = f"{status}: {topic}" + (f" -> {stream_url}" if stream_url else "")
    rows = [
        ("topic", topic),
        ("stream", stream_url or "not available"),
        ("run", run_report or "no run report"),
        ("image", stream_report or "no stream report"),
    ]
    row_parts = []
    y = 154
    for label, value in rows:
        lines = _wrap_text(value, width=66, max_lines=3)
        row_parts.append(
            f'<text x="36" y="{y}" fill="#9aa4b2" font-size="18" font-family="Inter, Arial">'
            f"{html.escape(label)}</text>"
        )
        tspans = "".join(
            f'<tspan x="150" dy="{0 if index == 0 else 22}">{html.escape(line)}</tspan>'
            for index, line in enumerate(lines)
        )
        row_parts.append(
            f'<text x="150" y="{y}" fill="#e5edf7" font-size="17" font-family="Inter, Arial">{tspans}</text>'
        )
        y += max(46, 24 * len(lines) + 18)
    height = max(380, y + 42)
    inner_height = height - 48
    row_svg = "\n".join(row_parts)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">
<rect width="900" height="{height}" rx="18" fill="#111827"/>
<rect x="24" y="24" width="852" height="{inner_height}" rx="14" fill="#162033" stroke="#263449"/>
<circle cx="58" cy="72" r="12" fill="{color}"/>
<text x="82" y="79" fill="{color}" font-size="24" font-weight="800" font-family="Inter, Arial">{status}</text>
<text x="36" y="118" fill="#e5edf7" font-size="30" font-weight="800" font-family="Inter, Arial">Camera</text>
{row_svg}
</svg>"""
    return {"dashboard": _svg_data(svg), "ready": ready, "report": report}


@node(
    name="VLM",
    category=_CATEGORY,
    description="Describe one image or detection prompt with OpenAI-compatible, Anthropic, or local Ollama chat.",
    inputs={
        "image": Image(default=""),
        "question": Text(default="What do you see?"),
        "system": Text(default="You are a precise robot vision assistant. Describe only what is visible."),
        "provider": Enum(["openai-compatible", "nvidia", "anthropic", "ollama", "auto"], default="ollama"),
        "model": Text(default="qwen3-vl:4b"),
        "endpoint_url": Text(default="http://127.0.0.1:11434"),
        "api_key": Text(default=""),
        "max_tokens": Int(default=512),
        "temperature": Float(default=0.2),
        "allow_text_only": Bool(default=False),
    },
    outputs={"text": Text, "report": Text, "raw": Dict},
)
def vision_vlm_describe(ctx: dict) -> dict:
    image = str(ctx.get("image") or "").strip()
    question = str(ctx.get("question") or "What do you see?").strip()
    system = str(ctx.get("system") or "").strip()
    provider = _normal_provider(ctx.get("provider"), str(ctx.get("endpoint_url") or ""))
    endpoint = _default_endpoint(provider, str(ctx.get("endpoint_url") or ""))
    model = _default_model(provider, str(ctx.get("model") or ""))
    max_tokens = max(1, min(int(ctx.get("max_tokens") or 512), 8192))
    temperature = float(ctx.get("temperature") or 0.2)
    allow_text_only = bool(ctx.get("allow_text_only", False))
    image_kind = _image_kind(image)
    has_image = image_kind in {"data-url", "url", "path-or-text"}

    if not image and not allow_text_only:
        return {
            "text": "",
            "report": "VLM describe FAILED: provide an image or enable allow_text_only for LLM-only reasoning",
            "raw": {},
        }

    try:
        if provider == "anthropic":
            api_key = (
                str(ctx.get("api_key") or "").strip()
                or os.environ.get("ANTHROPIC_API_KEY", "").strip()
                or os.environ.get("VISION_API_KEY", "").strip()
            )
            if not api_key:
                return {
                    "text": "",
                    "report": "VLM describe FAILED: set api_key or ANTHROPIC_API_KEY/VISION_API_KEY for Anthropic",
                    "raw": {},
                }
            content: list[dict[str, Any]] = []
            if image:
                content.append({"type": "image", "source": _anthropic_image_source(image)})
            content.append({"type": "text", "text": question})
            body: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": content}],
            }
            if system:
                body["system"] = system
            payload = _post_json(
                endpoint + "/messages",
                body,
                {
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            text = _extract_anthropic_text(payload)
            return {"text": text, "report": f"VLM describe OK via anthropic/{model}", "raw": payload}

        if provider == "ollama":
            if "qwen3" in model.lower() and max_tokens < 4096:
                max_tokens = 4096
            message: dict[str, Any] = {"role": "user", "content": question}
            if image:
                _media_type, image_data, _source_kind = _image_data_parts(image)
                message["images"] = [image_data]
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append(message)
            body = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            headers = {"Content-Type": "application/json"}
            api_key = str(ctx.get("api_key") or "").strip() or os.environ.get("OLLAMA_API_KEY", "").strip()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = _post_json(endpoint + "/api/chat", body, headers, timeout=180.0)

            def extract_ollama_text(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                message_data = data.get("message") if isinstance(data.get("message"), dict) else {}
                content_data = message_data.get("content")
                if isinstance(content_data, list):
                    extracted = "\n".join(
                        str(item.get("text", ""))
                        for item in content_data
                        if isinstance(item, dict)
                    ).strip()
                else:
                    extracted = str(content_data or "").strip()
                return extracted, message_data

            if payload.get("error"):
                return {"text": "", "report": f"VLM describe FAILED: ollama/{model}: {payload['error']}", "raw": payload}
            text, message_payload = extract_ollama_text(payload)
            retried_for_qwen3 = False
            if (
                not text
                and "qwen3" in model.lower()
                and str(payload.get("done_reason") or "").lower() == "length"
                and max_tokens < 8192
            ):
                retry_body = {
                    **body,
                    "options": {
                        **body["options"],
                        "num_predict": 8192,
                    },
                }
                payload = _post_json(endpoint + "/api/chat", retry_body, headers, timeout=240.0)
                retried_for_qwen3 = True
                if payload.get("error"):
                    return {"text": "", "report": f"VLM describe FAILED: ollama/{model}: {payload['error']}", "raw": payload}
                text, message_payload = extract_ollama_text(payload)
            if not text:
                message_keys = ", ".join(sorted(str(key) for key in message_payload)) or "none"
                thinking_note = "; thinking field was present but is hidden" if message_payload.get("thinking") else ""
                return {
                    "text": "",
                    "report": (
                        f"VLM describe FAILED: ollama/{model} returned empty final content"
                        f"{thinking_note}; message keys: {message_keys}"
                    ),
                    "raw": payload,
                }
            retry_note = " after Qwen3 length retry" if retried_for_qwen3 else ""
            return {"text": text, "report": f"VLM describe OK via ollama/{model}{retry_note}", "raw": payload}

        if has_image and image_kind not in {"data-url", "url"}:
            media_type, image_data, _source_kind = _image_data_parts(image)
            image_for_request = f"data:{media_type};base64,{image_data}"
        else:
            image_for_request = image
        api_key = (
            str(ctx.get("api_key") or "").strip()
            or os.environ.get("VISION_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
            or os.environ.get("NVIDIA_API_KEY", "").strip()
        )
        local_endpoint = endpoint.startswith(("http://127.0.0.1", "http://localhost"))
        if not api_key and not local_endpoint:
            return {
                "text": "",
                "report": "VLM describe FAILED: set api_key or VISION_API_KEY/OPENAI_API_KEY/NVIDIA_API_KEY",
                "raw": {},
            }
        user_content: Any
        if image_for_request:
            user_content = [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_for_request}},
            ]
        else:
            user_content = question
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = _post_json(endpoint + "/chat/completions", body, headers)
    except urllib.error.HTTPError as exc:
        return {"text": "", "report": f"VLM describe FAILED: {_format_http_error(exc)}", "raw": {}}
    except urllib.error.URLError as exc:
        return {"text": "", "report": f"VLM describe FAILED: {_format_connection_error(exc, provider, endpoint, model)}", "raw": {}}
    except Exception as exc:  # noqa: BLE001
        return {"text": "", "report": f"VLM describe FAILED: {type(exc).__name__}: {exc}", "raw": {}}

    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        text = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
    else:
        text = str(content or "").strip()
    return {"text": text, "report": f"VLM describe OK via openai-compatible/{model}", "raw": payload}


def _svg_multiline_text(lines: list[str], *, x: int, y: int, fill: str, size: int = 18, weight: int = 500) -> str:
    tspans = "".join(
        f'<tspan x="{x}" dy="{0 if index == 0 else size + 6}">{html.escape(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
        f'font-family="Inter, Arial">{tspans}</text>'
    )


@node(
    name="ReasoningDashboard",
    category=_CATEGORY,
    description="Render a captured camera frame with the VLM's visible observations, evidence, uncertainty, and action.",
    inputs={
        "image": Image(default=""),
        "answer": Text(default=""),
        "prompt": Text(default=""),
        "report": Text(default=""),
        "title": Text(default="Reasoning"),
    },
    outputs={"dashboard": Image, "ready": Bool, "summary": Dict},
)
def vision_reasoning_dashboard(ctx: dict) -> dict:
    image = str(ctx.get("image") or "").strip()
    answer = str(ctx.get("answer") or "").strip()
    prompt = str(ctx.get("prompt") or "").strip()
    report = str(ctx.get("report") or "").strip()
    title = str(ctx.get("title") or "Reasoning").strip()
    image_kind = _image_kind(image)
    image_data_url, image_error = _image_data_url(image)
    ready = bool(answer) and "FAILED" not in report.upper()
    status = "VLM READY" if ready else "WAITING FOR VLM"
    color = "#18a058" if ready else "#f59e0b"

    prompt_lines = _wrap_text(prompt or "No prompt yet.", width=70, max_lines=4)
    answer_lines = _wrap_text(answer or "Cook the VLM node after the camera frame is captured.", width=70, max_lines=12)
    report_lines = _wrap_text(report or "No VLM report yet.", width=70, max_lines=3)

    if image_data_url:
        image_svg = (
            f'<image x="36" y="132" width="390" height="292" preserveAspectRatio="xMidYMid meet" '
            f'href="{html.escape(image_data_url, quote=True)}"/>'
        )
    else:
        placeholder = "No captured frame yet" if not image_error else f"Frame unavailable: {_clip(image_error, 44)}"
        image_svg = (
            '<rect x="36" y="132" width="390" height="292" rx="10" fill="#0f172a" stroke="#334155"/>'
            f'<text x="58" y="284" fill="#94a3b8" font-size="18" font-family="Inter, Arial">{html.escape(placeholder)}</text>'
        )

    prompt_y = 174
    answer_y = prompt_y + 54 + len(prompt_lines) * 26
    report_y = answer_y + 62 + len(answer_lines) * 26
    height = max(620, report_y + max(1, len(report_lines)) * 24 + 56)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}">
<rect width="1120" height="{height}" rx="18" fill="#111827"/>
<rect x="24" y="24" width="1072" height="{height - 48}" rx="14" fill="#162033" stroke="#263449"/>
<circle cx="58" cy="72" r="12" fill="{color}"/>
<text x="82" y="79" fill="{color}" font-size="24" font-weight="800" font-family="Inter, Arial">{status}</text>
<text x="36" y="116" fill="#e5edf7" font-size="30" font-weight="800" font-family="Inter, Arial">{html.escape(title)}</text>
<rect x="36" y="132" width="390" height="292" rx="10" fill="#0b1020" stroke="#334155"/>
{image_svg}
<text x="36" y="462" fill="#94a3b8" font-size="16" font-family="Inter, Arial">captured frame: {html.escape(image_kind if not image_error else 'unavailable')}</text>
<text x="460" y="150" fill="#94a3b8" font-size="16" font-weight="800" font-family="Inter, Arial">PROMPT</text>
{_svg_multiline_text(prompt_lines, x=460, y=prompt_y, fill="#dbeafe", size=17, weight=500)}
<text x="460" y="{answer_y - 24}" fill="#94a3b8" font-size="16" font-weight="800" font-family="Inter, Arial">VISIBLE REASONING</text>
{_svg_multiline_text(answer_lines, x=460, y=answer_y, fill="#e5edf7", size=18, weight=600)}
<text x="460" y="{report_y - 24}" fill="#94a3b8" font-size="16" font-weight="800" font-family="Inter, Arial">MODEL REPORT</text>
{_svg_multiline_text(report_lines, x=460, y=report_y, fill="#cbd5e1", size=16, weight=500)}
</svg>"""
    return {
        "dashboard": _svg_data(svg),
        "ready": ready,
        "summary": {
            "ready": ready,
            "image_kind": image_kind,
            "image_embedded": bool(image_data_url),
            "image_error": image_error,
            "answer_chars": len(answer),
            "report": report,
        },
    }


@node(
    name="ReasoningStream",
    live=True,
    category=_CATEGORY,
    description="Start or stop a live MJPEG dashboard that periodically describes a camera image (local Ollama or NVIDIA NIM).",
    inputs={
        "trigger": AnyPort,
        "action": Enum(["start", "stop"], default="start"),
        "stream_id": Text(default="vision_reasoning"),
        "image_url": Text(default=""),
        "detection_url": Text(default=""),
        "prompt": Text(default="Describe what you see in this camera frame. If a colored cube is visible, mention its color and approximate location. Then give one useful next robot action."),
        "system": Text(default="You are a robot vision assistant. Describe only visible evidence from the image, then give a concise next action."),
        "provider": Enum(["ollama", "nvidia"], default="ollama"),
        "model": Text(default="qwen3-vl:4b"),
        "endpoint_url": Text(default="http://127.0.0.1:11434"),
        "api_key": Text(default=""),
        "temperature": Float(default=0.2),
        "max_tokens": Int(default=4096),
        "interval_seconds": Float(default=8.0),
        "host": Text(default="127.0.0.1"),
        "port": Int(default=0),
        "max_fps": Float(default=2.0),
        "max_width": Int(default=960),
        "title": Text(default="Live Reasoning"),
    },
    outputs={
        "dashboard": Image,
        "streaming": Bool,
        "stream_url": Text,
        "snapshot_url": Text,
        "state_url": Text,
        "stream_id": Text,
        "report": Text,
        "frame_stream": Dict,
    },
)
def vision_reasoning_stream(ctx: dict) -> dict:
    stream_id = str(ctx.get("stream_id") or "vision_reasoning").strip() or "vision_reasoning"
    action = str(ctx.get("action") or "start").strip().lower()
    empty = {
        "dashboard": "",
        "streaming": False,
        "stream_url": "",
        "snapshot_url": "",
        "state_url": "",
        "stream_id": stream_id,
    }
    if action == "stop":
        result = cv2_runtime.stop_reasoning_stream(stream_id)
        return {**empty, "report": f"stopped {result.get('stopped', 0)} reasoning stream(s)"}

    image_url = str(ctx.get("image_url") or "").strip()
    if not image_url:
        return {**empty, "report": "reasoning stream FAILED: connect image_url to a camera snapshot URL"}

    provider = _normal_provider(ctx.get("provider"), str(ctx.get("endpoint_url") or ""))
    if provider not in {"ollama", "nvidia"}:
        return {**empty, "report": "reasoning stream FAILED: only provider=ollama or provider=nvidia is supported for live streaming"}

    model = _default_model(provider, str(ctx.get("model") or ""))
    max_tokens = max(1, min(int(ctx.get("max_tokens") or 4096), 8192))
    if "qwen3" in model.lower() and max_tokens < 4096:
        max_tokens = 4096
    host = str(ctx.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    api_key = str(ctx.get("api_key") or "").strip()
    if not api_key:
        api_key = os.environ.get("NVIDIA_API_KEY", "").strip() if provider == "nvidia" else os.environ.get("OLLAMA_API_KEY", "").strip()
    if provider == "nvidia" and not api_key:
        return {**empty, "report": "reasoning stream FAILED: set api_key or NVIDIA_API_KEY for provider=nvidia"}
    result = cv2_runtime.start_reasoning_stream(
        stream_id=stream_id,
        image_url=image_url,
        detection_url=str(ctx.get("detection_url") or "").strip(),
        prompt=str(ctx.get("prompt") or "").strip(),
        system=str(ctx.get("system") or "").strip(),
        provider=provider,
        model=model,
        endpoint_url=_default_endpoint(provider, str(ctx.get("endpoint_url") or "")),
        api_key=api_key,
        temperature=float(ctx.get("temperature") or 0.2),
        max_tokens=max_tokens,
        interval_seconds=max(1.0, float(ctx.get("interval_seconds") or 8.0)),
        host=host,
        port=max(0, int(ctx.get("port") or 0)),
        max_fps=max(0.1, min(10.0, float(ctx.get("max_fps") or 2.0))),
        max_width=max(0, int(ctx.get("max_width") or 960)),
        title=str(ctx.get("title") or "Live Reasoning").strip() or "Live Reasoning",
    )
    if not result.get("ok"):
        return {**empty, "report": f"reasoning stream FAILED: {result.get('error', 'unknown error')}"}

    stream_url = str(result.get("stream_url") or "")
    snapshot_url = str(result.get("snapshot_url") or "")
    state_url = str(result.get("state_url") or "")
    model_label = model if model.lower().startswith(f"{provider.lower()}/") else f"{provider}/{model}"
    return {
        "dashboard": stream_url,
        "streaming": True,
        "stream_url": stream_url,
        "snapshot_url": snapshot_url,
        "state_url": state_url,
        "stream_id": stream_id,
        "report": f"LIVE REASONING STREAM running on {stream_url} from {image_url} with {model_label}",
    }
