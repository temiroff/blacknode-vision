"""Deterministic OpenCV colour-object tracking nodes."""
from __future__ import annotations

import base64
import json
import time
from typing import Any

try:
    import cv2
    import numpy as np

    _CV2_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - machines without OpenCV
    cv2 = None
    np = None
    _CV2_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from blacknode import streams as bn_streams
from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Color, Dict, Enum, Float, Image, Int, List, Text, node
from blacknode.pkg.blacknode_perception import cv2_runtime
from blacknode.pkg.blacknode_perception.image_ops import (
    _HSV_COLOR_RANGES,
    _bool_value,
    _decode_image_bgr,
    _find_color,
    _find_object_label,
    _format_hsv,
    _missing_cv2_outputs,
    _parse_hsv,
    _read_reasoning_state_answer,
)

_CATEGORY = "Tracking"


def _encode_bgr(image: Any, *, image_format: str = "jpeg", jpeg_quality: int = 86) -> str:
    fmt = image_format.lower()
    if fmt in {"jpg", "jpeg"}:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        mime = "image/jpeg"
    else:
        ok, encoded = cv2.imencode(".png", image)
        mime = "image/png"
    if not ok:
        raise ValueError("OpenCV image encoding failed")
    return f"data:{mime};base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _prepare_mask(
    image_bgr: Any,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
    blur: int,
    morphology_iters: int,
) -> Any:
    blur = max(0, int(blur))
    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        image_bgr = cv2.GaussianBlur(image_bgr, (blur, blur), 0)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    if lower_hsv[0] <= upper_hsv[0]:
        mask = cv2.inRange(hsv, lower, upper)
    else:
        low_a = np.array([lower_hsv[0], lower_hsv[1], lower_hsv[2]], dtype=np.uint8)
        high_a = np.array([179, upper_hsv[1], upper_hsv[2]], dtype=np.uint8)
        low_b = np.array([0, lower_hsv[1], lower_hsv[2]], dtype=np.uint8)
        high_b = np.array([upper_hsv[0], upper_hsv[1], upper_hsv[2]], dtype=np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, low_a, high_a), cv2.inRange(hsv, low_b, high_b))

    iters = max(0, int(morphology_iters))
    if iters:
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return mask


@node(
    name="TrackingColorHint",
    hidden=True,
    category="Tracking",
    description="Resolve target or reasoning text such as 'track the red cube' into HSV settings for CV2 tracking.",
    inputs={
        "target": Text(default="track green cube"),
        "reasoning": Text(default=""),
        "reasoning_state_url": Text(default=""),
        "reasoning_wait_seconds": Float(default=0.0),
        "fallback_color": Enum(sorted(_HSV_COLOR_RANGES), default="green"),
        "fallback_label": Text(default="cube"),
    },
    outputs={
        "target": Text,
        "color": Text,
        "label": Text,
        "lower_hsv": Text,
        "upper_hsv": Text,
        "found": Bool,
        "source": Text,
        "metadata": Dict,
        "report": Text,
    },
)
def cv2_color_target_hint(ctx: dict) -> dict:
    target = str(ctx.get("target") or "").strip()
    reasoning = str(ctx.get("reasoning") or "").strip()
    reasoning_state_url = str(ctx.get("reasoning_state_url") or "").strip()
    wait_seconds = max(0.0, float(ctx.get("reasoning_wait_seconds") or 0.0))
    fallback_color = _find_color(ctx.get("fallback_color")) or "green"
    fallback_label = str(ctx.get("fallback_label") or "cube").strip() or "cube"

    target_color = _find_color(target)
    reasoning_color = _find_color(reasoning)
    state_answer = ""
    state_error = ""
    if not target_color and not reasoning_color and reasoning_state_url:
        state_answer, state_error = _read_reasoning_state_answer(reasoning_state_url, wait_seconds)
        if state_answer:
            reasoning = "\n".join(part for part in (reasoning, state_answer) if part)
            reasoning_color = _find_color(reasoning)

    if target_color:
        color = target_color
        source = "target"
        found = True
        source_text = target
    elif reasoning_color:
        color = reasoning_color
        source = "reasoning"
        found = True
        source_text = reasoning
    else:
        color = fallback_color if fallback_color in _HSV_COLOR_RANGES else "green"
        source = "fallback"
        found = False
        source_text = target or reasoning

    label_word = _find_object_label(target, "") or _find_object_label(reasoning, "") or fallback_label
    label = f"{color} {label_word}".strip()
    lower, upper = _HSV_COLOR_RANGES[color]
    resolved_target = target or reasoning or label
    lower_text = _format_hsv(lower)
    upper_text = _format_hsv(upper)
    report_source = f" from {source} text" if found else f" using fallback {fallback_color}"
    if source == "reasoning" and state_answer:
        report_source = f" from reasoning state {reasoning_state_url}"
    elif source == "fallback" and state_error:
        report_source += f"; reasoning state unavailable: {state_error}"
    report = f"CV2 target hint OK: tracking {label} with HSV {lower_text}-{upper_text}{report_source}"
    return {
        "target": resolved_target,
        "color": color,
        "label": label,
        "lower_hsv": lower_text,
        "upper_hsv": upper_text,
        "found": found,
        "source": source,
        "metadata": {
            "target": target,
            "reasoning": reasoning,
            "reasoning_state_url": reasoning_state_url,
            "reasoning_state_error": state_error,
            "source_text": source_text,
            "color": color,
            "label": label,
            "lower_hsv": lower,
            "upper_hsv": upper,
            "source": source,
            "explicit_color_found": bool(target_color),
            "reasoning_color_found": bool(reasoning_color),
            "reasoning_state_used": bool(state_answer),
        },
        "report": report,
    }


@node(
    name="TrackingColorMask",
    hidden=True,
    category="Tracking",
    description="Create an HSV color mask from a Blacknode image using OpenCV.",
    inputs={
        "image": Image(default=""),
        "lower_hsv": Text(default="35,60,60"),
        "upper_hsv": Text(default="85,255,255"),
        "blur": Int(default=5),
        "morphology_iters": Int(default=1),
    },
    outputs={"mask": Image, "preview": Image, "metadata": Dict, "report": Text},
)
def cv2_hsv_mask(ctx: dict) -> dict:
    if cv2 is None or np is None:
        missing = _missing_cv2_outputs()
        return {"mask": "", "preview": "", "metadata": {}, "report": missing["report"]}

    image, error = _decode_image_bgr(ctx.get("image"))
    if error:
        return {"mask": "", "preview": "", "metadata": {}, "report": error}

    lower = _parse_hsv(ctx.get("lower_hsv"), (35, 60, 60))
    upper = _parse_hsv(ctx.get("upper_hsv"), (85, 255, 255))
    mask = _prepare_mask(
        image,
        lower,
        upper,
        blur=int(ctx.get("blur") or 0),
        morphology_iters=int(ctx.get("morphology_iters") or 0),
    )
    preview = cv2.bitwise_and(image, image, mask=mask)
    h, w = mask.shape[:2]
    pixels = int(cv2.countNonZero(mask))
    metadata = {
        "width": int(w),
        "height": int(h),
        "selected_pixels": pixels,
        "selected_ratio": pixels / max(1, w * h),
        "lower_hsv": lower,
        "upper_hsv": upper,
    }
    report = f"CV2 HSV mask OK: {pixels} pixel(s) selected from {w}x{h}"
    return {
        "mask": _encode_bgr(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), image_format="png"),
        "preview": _encode_bgr(preview, image_format="jpeg"),
        "metadata": metadata,
        "report": report,
    }


@node(
    name="TrackingObject",
    live=True,
    category="Tracking",
    primary_inputs=["frame_stream"],
    description="Live object tracking on a wired camera stream: draws boxes around the tracked colour object and serves annotated MJPEG plus detection JSON. Wire a Camera's frame_stream in and press Go Live.",
    inputs={
        "trigger": AnyPort,
        "action": Enum(["start", "stop"], default="start"),
        "stream_id": Text(default="cube_tracker"),
        "frame_stream": Dict(default={}),
        "source_url": Text(default=""),
        "object_color": Color(default="#22c55e"),
        "use_reasoning_color": Bool(default=True),
        "target": Text(default=""),
        "reasoning_state_url": Text(default=""),
        "target_update_seconds": Float(default=2.0),
        "show_follow_guides": Bool(default=True),
        "follow_target_x": Float(default=0.4),
        "follow_deadband": Float(default=0.12),
        "label": Text(default="cube"),
        "min_area": Int(default=300),
        "max_detections": Int(default=3),
        "blur": Int(default=5),
        "morphology_iters": Int(default=1),
        "host": Text(default="127.0.0.1"),
        "port": Int(default=0),
        "max_fps": Float(default=10.0),
        "max_width": Int(default=960),
        "jpeg_quality": Int(default=82),
    },
    outputs={
        "preview": Image,
        "snapshot": Image,
        "mask": Image,
        "streaming": Bool,
        "stream_url": Text,
        "snapshot_url": Text,
        "mask_stream_url": Text,
        "mask_url": Text,
        "detection_stream": Dict,
        "detection_url": Text,
        "stream_id": Text,
        "found": Bool,
        "detection": Dict,
        "detections": List,
        "report": Text,
        "frame_stream": Dict,
    },
)
def cv2_color_object_stream(ctx: dict) -> dict:
    stream_id = str(ctx.get("stream_id") or "cube_tracker").strip() or "cube_tracker"
    action = str(ctx.get("action") or "start").strip().lower()
    empty = {
        "preview": "",
        "snapshot": "",
        "mask": "",
        "streaming": False,
        "stream_url": "",
        "snapshot_url": "",
        "mask_stream_url": "",
        "mask_url": "",
        "detection_stream": {},
        "detection_url": "",
        "stream_id": stream_id,
        "found": False,
        "detection": {},
        "detections": [],
    }
    if action == "stop":
        result = cv2_runtime.stop_color_stream(stream_id)
        return {**empty, "report": f"stopped {result.get('stopped', 0)} CV2 stream(s)"}

    if cv2 is None or np is None:
        return {**empty, "report": _missing_cv2_outputs()["report"]}

    # Track on a per-frame snapshot, not the continuous MJPEG: the stream server
    # fetches the source with urlopen().read(), which blocks forever on a
    # multipart /stream.mjpg (the "waiting for first frame" hang). Prefer the
    # frame_stream's snapshot URL, or derive it from the stream URL.
    frame_stream = ctx.get("frame_stream") if isinstance(ctx.get("frame_stream"), dict) else {}
    stream_url = bn_streams.source_url(frame_stream, str(ctx.get("source_url") or ""))
    source_url = str(frame_stream.get("snapshot_url") or "")
    if not source_url and stream_url.endswith("/stream.mjpg"):
        source_url = stream_url[: -len("/stream.mjpg")] + "/snapshot.jpg"
    if not source_url:
        source_url = stream_url
    if not source_url:
        return {**empty, "report": (
            "TrackingObject FAILED: nothing wired to 'frame_stream'.\n"
            "CHECK: connect a Camera node's frame_stream output and cook it first."
        )}

    label = str(ctx.get("label") or "object").strip() or "object"
    object_color = (
        str(ctx.get("object_color") or ctx.get("manual_color") or ctx.get("fallback_color") or "#22c55e").strip()
        or "#22c55e"
    )
    host = str(ctx.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    result = cv2_runtime.start_color_stream(
        stream_id=stream_id,
        source_url=source_url,
        object_color=object_color,
        use_reasoning_color=_bool_value(ctx.get("use_reasoning_color"), True),
        label=label,
        target_text=str(ctx.get("target") or "").strip(),
        reasoning_state_url=str(ctx.get("reasoning_state_url") or "").strip(),
        target_update_seconds=max(0.25, float(ctx.get("target_update_seconds") or 2.0)),
        show_follow_guides=_bool_value(ctx.get("show_follow_guides"), True),
        follow_target_x=max(0.0, min(1.0, float(ctx.get("follow_target_x") if ctx.get("follow_target_x") is not None else 0.4))),
        follow_deadband=max(0.0, min(0.5, float(ctx.get("follow_deadband") if ctx.get("follow_deadband") is not None else 0.12))),
        min_area=max(0, int(ctx.get("min_area") or 0)),
        max_detections=max(1, int(ctx.get("max_detections") or 1)),
        blur=max(0, int(ctx.get("blur") or 0)),
        morphology_iters=max(0, int(ctx.get("morphology_iters") or 0)),
        host=host,
        port=max(0, int(ctx.get("port") or 0)),
        max_fps=max(0.1, min(60.0, float(ctx.get("max_fps") or 10.0))),
        max_width=max(0, int(ctx.get("max_width") or 960)),
        jpeg_quality=max(1, min(100, int(ctx.get("jpeg_quality") or 82))),
    )
    if not result.get("ok"):
        return {**empty, "report": f"CV2 stream FAILED: {result.get('error', 'unknown error')}"}

    payload = result.get("detection") if isinstance(result.get("detection"), dict) else {}
    detection = payload.get("detection") if isinstance(payload.get("detection"), dict) else {}
    detections = payload.get("detections") if isinstance(payload.get("detections"), list) else []
    found = bool(payload.get("found") or detection.get("found"))
    stream_url = str(result.get("stream_url") or "")
    snapshot_url = str(result.get("snapshot_url") or "")
    mask_stream_url = str(result.get("mask_stream_url") or "")
    mask_url = str(result.get("mask_url") or "")
    detection_url = str(result.get("detection_url") or "")
    detection_stream = {
        "kind": "blacknode.latest-value-stream",
        "stream_id": stream_id,
        "url": detection_url,
        "media_type": "application/json",
    }
    report = (
        f"LIVE CV2 STREAM running on {stream_url} from {source_url}; "
        f"{str(payload.get('report') or 'waiting for detections')}"
    )
    return {
        "preview": stream_url,
        "snapshot": snapshot_url,
        "mask": mask_stream_url or mask_url,
        "streaming": True,
        "stream_url": stream_url,
        "snapshot_url": snapshot_url,
        "mask_stream_url": mask_stream_url,
        "mask_url": mask_url,
        "detection_stream": detection_stream,
        "detection_url": detection_url,
        "stream_id": stream_id,
        "found": found,
        "detection": detection or {"found": False, "label": label},
        "detections": detections,
        "report": report,
    }
