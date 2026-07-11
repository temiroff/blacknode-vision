"""Generic vision nodes for Blacknode.

This package stays camera- and robot-neutral. ROS 2 transport is handled by
blacknode-ros2; these nodes add reusable vision prompts, status views, and an
optional OpenAI-compatible VLM call for one captured frame.
"""
from __future__ import annotations

import base64
import html
import json
import os
import textwrap
import urllib.error
import urllib.request
from typing import Any

from blacknode.node import Bool, Dict, Float, Image, Int, Text, node

_CATEGORY = "Vision"


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


@node(
    name="VisionFramePrompt",
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
    name="VisionStreamStatus",
    category=_CATEGORY,
    description="Render camera stream readiness as a dashboard image.",
    inputs={
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
    stream_url = str(ctx.get("stream_url") or "")
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
<text x="36" y="118" fill="#e5edf7" font-size="30" font-weight="800" font-family="Inter, Arial">Blacknode Vision Stream</text>
{row_svg}
</svg>"""
    return {"dashboard": _svg_data(svg), "ready": ready, "report": report}


@node(
    name="VisionVLMDescribe",
    category=_CATEGORY,
    description="Describe one image with an OpenAI-compatible vision chat endpoint.",
    inputs={
        "image": Image(default=""),
        "question": Text(default="What do you see?"),
        "system": Text(default="You are a precise robot vision assistant. Describe only what is visible."),
        "model": Text(default="gpt-4o-mini"),
        "endpoint_url": Text(default="https://api.openai.com/v1"),
        "api_key": Text(default=""),
        "max_tokens": Int(default=512),
        "temperature": Float(default=0.2),
    },
    outputs={"text": Text, "report": Text, "raw": Dict},
)
def vision_vlm_describe(ctx: dict) -> dict:
    image = str(ctx.get("image") or "").strip()
    if _image_kind(image) not in {"data-url", "url"}:
        return {"text": "", "report": "VLM describe FAILED: provide a data:image or http(s) image URL", "raw": {}}

    endpoint = str(ctx.get("endpoint_url") or "https://api.openai.com/v1").rstrip("/")
    url = endpoint + "/chat/completions"
    model = str(ctx.get("model") or "gpt-4o-mini").strip()
    question = str(ctx.get("question") or "What do you see?").strip()
    system = str(ctx.get("system") or "").strip()
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

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            },
        ],
        "max_tokens": max(1, min(int(ctx.get("max_tokens") or 512), 4096)),
        "temperature": float(ctx.get("temperature") or 0.2),
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"text": "", "report": f"VLM describe FAILED: HTTP {exc.code}: {_clip(detail, 240)}", "raw": {}}
    except Exception as exc:  # noqa: BLE001
        return {"text": "", "report": f"VLM describe FAILED: {type(exc).__name__}: {exc}", "raw": {}}

    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        text = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
    else:
        text = str(content or "").strip()
    return {"text": text, "report": f"VLM describe OK via {model}", "raw": payload}
