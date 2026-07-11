"""OpenCV nodes for Blacknode Vision."""
from __future__ import annotations

import base64
import json
import textwrap
import urllib.request
from pathlib import Path
from typing import Any

from . import cv2_runtime

try:
    import cv2
    import numpy as np

    _CV2_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - exercised on machines without OpenCV
    cv2 = None
    np = None
    _CV2_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Image, Int, List, Text, node

_CATEGORY = "CV2"


def _missing_cv2_outputs() -> dict[str, Any]:
    report = (
        "CV2 node FAILED: OpenCV is not installed in this Blacknode Python environment. "
        "Run: blacknode packages setup blacknode-vision"
    )
    if _CV2_IMPORT_ERROR:
        report += f" ({_CV2_IMPORT_ERROR})"
    return {
        "mask": "",
        "preview": "",
        "overlay": "",
        "found": False,
        "center_x": 0,
        "center_y": 0,
        "area": 0.0,
        "metadata": {},
        "detection": {},
        "detections": [],
        "report": report,
    }


def _parse_hsv(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    raw = value if value not in (None, "") else default
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            parts = list(default)
        else:
            try:
                decoded = json.loads(text)
                parts = list(decoded) if isinstance(decoded, (list, tuple)) else []
            except json.JSONDecodeError:
                parts = [part for part in text.replace(";", ",").replace(" ", ",").split(",") if part]
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        parts = list(default)
    values = list(default)
    for index, part in enumerate(parts[:3]):
        try:
            values[index] = int(float(part))
        except (TypeError, ValueError):
            values[index] = default[index]
    return (
        max(0, min(179, values[0])),
        max(0, min(255, values[1])),
        max(0, min(255, values[2])),
    )


def _decode_image_bgr(source: Any) -> tuple[Any, str]:
    if cv2 is None or np is None:
        return None, _missing_cv2_outputs()["report"]

    text = str(source or "").strip()
    if not text:
        return None, "CV2 node FAILED: no image provided"

    try:
        if text.startswith("data:"):
            if "," not in text:
                return None, "CV2 node FAILED: invalid image data URL"
            raw = base64.b64decode(text.split(",", 1)[1])
            data = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif text.startswith(("http://", "https://")):
            with urllib.request.urlopen(text, timeout=20) as response:
                raw = response.read()
            data = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            path = Path(text).expanduser()
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        return None, f"CV2 node FAILED: could not decode image: {type(exc).__name__}: {exc}"

    if image is None:
        return None, "CV2 node FAILED: could not decode image; use PNG/JPEG data URL, URL, or file path"
    return image, ""


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


def _find_detections(mask: Any, *, label: str, min_area: float, max_detections: int) -> list[dict[str, Any]]:
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0] if len(result) == 2 else result[1]
    image_h, image_w = mask.shape[:2]
    detections: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments.get("m00"):
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
        bbox_area = max(1, w * h)
        detections.append(
            {
                "label": label,
                "center": {"x": cx, "y": cy},
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "area": area,
                "area_ratio": area / max(1, image_w * image_h),
                "bbox_fill": area / bbox_area,
                "aspect_ratio": w / max(1, h),
            }
        )
    detections.sort(key=lambda item: float(item["area"]), reverse=True)
    return detections[: max(1, int(max_detections))]


def _draw_detections(image_bgr: Any, detections: list[dict[str, Any]], *, label: str) -> Any:
    overlay = image_bgr.copy()
    for index, detection in enumerate(detections, start=1):
        bbox = detection["bbox"]
        center = detection["center"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        cx, cy = center["x"], center["y"]
        color = (22, 163, 74) if index == 1 else (37, 99, 235)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.drawMarker(overlay, (cx, cy), color, cv2.MARKER_CROSS, 18, 2)
        caption = f"{label} {index}: ({cx},{cy}) area={int(detection['area'])}"
        cv2.putText(
            overlay,
            caption,
            (max(4, x), max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


@node(
    name="CV2HSVMask",
    category=_CATEGORY,
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
    name="CV2ColorObjectTracker",
    category=_CATEGORY,
    description="Track the largest object in an HSV color range and draw an overlay.",
    inputs={
        "image": Image(default=""),
        "label": Text(default="cube"),
        "lower_hsv": Text(default="35,60,60"),
        "upper_hsv": Text(default="85,255,255"),
        "min_area": Int(default=300),
        "max_detections": Int(default=3),
        "blur": Int(default=5),
        "morphology_iters": Int(default=1),
    },
    outputs={
        "overlay": Image,
        "mask": Image,
        "found": Bool,
        "center_x": Int,
        "center_y": Int,
        "area": Float,
        "detection": Dict,
        "detections": List,
        "report": Text,
    },
)
def cv2_color_object_tracker(ctx: dict) -> dict:
    if cv2 is None or np is None:
        missing = _missing_cv2_outputs()
        return {
            "overlay": "",
            "mask": "",
            "found": False,
            "center_x": 0,
            "center_y": 0,
            "area": 0.0,
            "detection": {},
            "detections": [],
            "report": missing["report"],
        }

    image, error = _decode_image_bgr(ctx.get("image"))
    if error:
        return {
            "overlay": "",
            "mask": "",
            "found": False,
            "center_x": 0,
            "center_y": 0,
            "area": 0.0,
            "detection": {},
            "detections": [],
            "report": error,
        }

    label = str(ctx.get("label") or "object").strip() or "object"
    lower = _parse_hsv(ctx.get("lower_hsv"), (35, 60, 60))
    upper = _parse_hsv(ctx.get("upper_hsv"), (85, 255, 255))
    min_area = max(0.0, float(ctx.get("min_area") or 0))
    mask = _prepare_mask(
        image,
        lower,
        upper,
        blur=int(ctx.get("blur") or 0),
        morphology_iters=int(ctx.get("morphology_iters") or 0),
    )
    detections = _find_detections(
        mask,
        label=label,
        min_area=min_area,
        max_detections=int(ctx.get("max_detections") or 1),
    )
    overlay = _draw_detections(image, detections, label=label)
    detection = detections[0] if detections else {}
    center = detection.get("center") or {}
    found = bool(detection)
    report = (
        f"CV2 tracker OK: found {len(detections)} {label} candidate(s); "
        f"largest center=({center.get('x', 0)}, {center.get('y', 0)})"
        if found
        else f"CV2 tracker OK: no {label} found in HSV {lower}-{upper} above area {int(min_area)}"
    )
    return {
        "overlay": _encode_bgr(overlay, image_format="jpeg"),
        "mask": _encode_bgr(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), image_format="png"),
        "found": found,
        "center_x": int(center.get("x") or 0),
        "center_y": int(center.get("y") or 0),
        "area": float(detection.get("area") or 0.0),
        "detection": {
            **detection,
            "found": found,
            "lower_hsv": lower,
            "upper_hsv": upper,
        }
        if found
        else {"found": False, "label": label, "lower_hsv": lower, "upper_hsv": upper},
        "detections": detections,
        "report": report,
    }


@node(
    name="CV2ColorObjectStream",
    category=_CATEGORY,
    description="Start or stop a live MJPEG stream with OpenCV color tracking overlay and detection JSON.",
    inputs={
        "trigger": AnyPort,
        "action": Enum(["start", "stop"], default="start"),
        "stream_id": Text(default="cube_tracker"),
        "source_url": Text(default=""),
        "label": Text(default="cube"),
        "lower_hsv": Text(default="35,60,60"),
        "upper_hsv": Text(default="85,255,255"),
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
        "streaming": Bool,
        "stream_url": Text,
        "snapshot_url": Text,
        "detection_url": Text,
        "stream_id": Text,
        "found": Bool,
        "detection": Dict,
        "detections": List,
        "report": Text,
    },
)
def cv2_color_object_stream(ctx: dict) -> dict:
    stream_id = str(ctx.get("stream_id") or "cube_tracker").strip() or "cube_tracker"
    action = str(ctx.get("action") or "start").strip().lower()
    empty = {
        "preview": "",
        "snapshot": "",
        "streaming": False,
        "stream_url": "",
        "snapshot_url": "",
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

    source_url = str(ctx.get("source_url") or "").strip()
    if not source_url:
        return {**empty, "report": "CV2 stream FAILED: connect source_url to a camera snapshot URL"}

    label = str(ctx.get("label") or "object").strip() or "object"
    lower_hsv = ",".join(str(value) for value in _parse_hsv(ctx.get("lower_hsv"), (35, 60, 60)))
    upper_hsv = ",".join(str(value) for value in _parse_hsv(ctx.get("upper_hsv"), (85, 255, 255)))
    host = str(ctx.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    result = cv2_runtime.start_color_stream(
        stream_id=stream_id,
        source_url=source_url,
        label=label,
        lower_hsv=lower_hsv,
        upper_hsv=upper_hsv,
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
    detection_url = str(result.get("detection_url") or "")
    report = (
        f"LIVE CV2 STREAM running on {stream_url} from {source_url}; "
        f"{str(payload.get('report') or 'waiting for detections')}"
    )
    return {
        "preview": stream_url,
        "snapshot": snapshot_url,
        "streaming": True,
        "stream_url": stream_url,
        "snapshot_url": snapshot_url,
        "detection_url": detection_url,
        "stream_id": stream_id,
        "found": found,
        "detection": detection or {"found": False, "label": label},
        "detections": detections,
        "report": report,
    }


@node(
    name="CV2TrackerPythonExport",
    category=_CATEGORY,
    description="Generate a standalone OpenCV color tracker script for robot deployment experiments.",
    inputs={
        "label": Text(default="cube"),
        "lower_hsv": Text(default="35,60,60"),
        "upper_hsv": Text(default="85,255,255"),
        "min_area": Int(default=300),
        "camera_device": Int(default=0),
        "width": Int(default=640),
        "height": Int(default=480),
        "show_preview": Bool(default=True),
    },
    outputs={"source": Text, "report": Text},
)
def cv2_tracker_python_export(ctx: dict) -> dict:
    label = str(ctx.get("label") or "cube").strip() or "cube"
    lower = _parse_hsv(ctx.get("lower_hsv"), (35, 60, 60))
    upper = _parse_hsv(ctx.get("upper_hsv"), (85, 255, 255))
    min_area = max(0, int(ctx.get("min_area") or 0))
    camera_device = max(0, int(ctx.get("camera_device") or 0))
    width = max(1, int(ctx.get("width") or 640))
    height = max(1, int(ctx.get("height") or 480))
    show_preview = bool(ctx.get("show_preview", True))

    source = f'''#!/usr/bin/env python3
"""Standalone OpenCV color tracker generated by Blacknode."""
from __future__ import annotations

import json
import time

import cv2
import numpy as np

LABEL = {label!r}
LOWER_HSV = np.array({list(lower)!r}, dtype=np.uint8)
UPPER_HSV = np.array({list(upper)!r}, dtype=np.uint8)
MIN_AREA = {min_area}
CAMERA_DEVICE = {camera_device}
WIDTH = {width}
HEIGHT = {height}
SHOW_PREVIEW = {show_preview!r}


def build_mask(frame):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    if int(LOWER_HSV[0]) <= int(UPPER_HSV[0]):
        return cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    high_a = np.array([179, UPPER_HSV[1], UPPER_HSV[2]], dtype=np.uint8)
    low_b = np.array([0, LOWER_HSV[1], LOWER_HSV[2]], dtype=np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, LOWER_HSV, high_a), cv2.inRange(hsv, low_b, UPPER_HSV))


def detect(frame):
    mask = build_mask(frame)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        cx = int(moments["m10"] / moments["m00"]) if moments["m00"] else int(x + w / 2)
        cy = int(moments["m01"] / moments["m00"]) if moments["m00"] else int(y + h / 2)
        item = {{"found": True, "label": LABEL, "center": {{"x": cx, "y": cy}}, "bbox": {{"x": x, "y": y, "width": w, "height": h}}, "area": area}}
        if best is None or item["area"] > best["area"]:
            best = item
    return best or {{"found": False, "label": LABEL}}


def main():
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {{CAMERA_DEVICE}}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            detection = detect(frame)
            print(json.dumps(detection), flush=True)
            if SHOW_PREVIEW:
                if detection.get("found"):
                    box = detection["bbox"]
                    center = detection["center"]
                    cv2.rectangle(frame, (box["x"], box["y"]), (box["x"] + box["width"], box["y"] + box["height"]), (22, 163, 74), 2)
                    cv2.drawMarker(frame, (center["x"], center["y"]), (22, 163, 74), cv2.MARKER_CROSS, 18, 2)
                cv2.imshow("Blacknode CV2 Tracker", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''
    source = textwrap.dedent(source).strip() + "\n"
    report = f"standalone CV2 tracker script generated for {label} on camera {camera_device}"
    return {"source": source, "report": report}
