"""Camera, calibration, and deterministic OpenCV nodes for Blacknode Perception."""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
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
from blacknode.node import Bool, Color, Dict, Enum, Float, Image, Int, List, Text, node

_CATEGORY = "CV2"
_HSV_COLOR_RANGES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "red": ((170, 80, 60), (10, 255, 255)),
    "orange": ((5, 80, 60), (25, 255, 255)),
    "yellow": ((20, 80, 80), (35, 255, 255)),
    "green": ((35, 60, 60), (85, 255, 255)),
    "cyan": ((85, 60, 60), (100, 255, 255)),
    "blue": ((100, 60, 50), (130, 255, 255)),
    "purple": ((130, 50, 50), (160, 255, 255)),
    "pink": ((145, 50, 80), (175, 255, 255)),
    "white": ((0, 0, 180), (179, 60, 255)),
    "black": ((0, 0, 0), (179, 255, 70)),
}

_CALIBRATION_SAMPLES: dict[str, dict[str, Any]] = {}


@node(
    name="CameraCalibration",
    category="Camera",
    description="Capture checkerboard views, solve camera intrinsics/FOV, and attach them to a frame stream.",
    primary_inputs=["trigger", "image", "frame_stream"],
    primary_outputs=["calibrated_stream", "calibration", "report"],
    inputs={
        "trigger": AnyPort,
        "action": Enum(["capture", "reset", "solve"], default="capture"),
        "stream_id": Text(default="camera_0"),
        "image": Image(default=""),
        "frame_stream": Dict(default={}),
        "frames": List(default=[]),
        "board_columns": Int(default=9),
        "board_rows": Int(default=6),
        "square_size": Float(default=0.025),
        "min_samples": Int(default=12),
    },
    outputs={"calibration": Dict, "calibrated_stream": Dict, "samples": Int, "ready": Bool, "report": Text},
)
def camera_calibration(ctx: dict) -> dict:
    """Collect checkerboard observations and solve camera intrinsics/FOV."""
    frame_stream = dict(ctx.get("frame_stream") or {})
    empty = {"calibration": {}, "calibrated_stream": frame_stream, "samples": 0, "ready": False}
    if cv2 is None or np is None:
        return {**empty, "report": f"camera calibration unavailable: {_CV2_IMPORT_ERROR}"}
    key = str(ctx.get("stream_id") or frame_stream.get("stream_id") or "camera_0")
    action = str(ctx.get("action") or "capture").lower()
    cols = max(2, int(ctx.get("board_columns") or 9))
    rows = max(2, int(ctx.get("board_rows") or 6))
    square = max(1e-6, float(ctx.get("square_size") or 0.025))
    board = (cols, rows, square)
    if action == "reset":
        _CALIBRATION_SAMPLES.pop(key, None)
        return {**empty, "report": f"camera calibration reset for {key}"}
    state = _CALIBRATION_SAMPLES.get(key)
    if state is None or tuple(state.get("board") or ()) != board:
        state = {"board": board, "samples": []}
        _CALIBRATION_SAMPLES[key] = state
    samples = state["samples"]
    if action == "capture":
        sources = []
        if ctx.get("image") not in (None, ""):
            sources.append(ctx.get("image"))
        sources.extend(list(ctx.get("frames") or []))
        if not sources and frame_stream.get("snapshot_url"):
            sources.append(frame_stream["snapshot_url"])
        captured = 0
        rejected = 0
        for frame in sources:
            image = frame
            if isinstance(frame, str):
                image, _ = _decode_image_bgr(frame)
            if image is None or not hasattr(image, "shape"):
                rejected += 1
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            size = tuple(int(value) for value in gray.shape[::-1])
            if samples and tuple(samples[0][1]) != size:
                rejected += 1
                continue
            found, corners = cv2.findChessboardCorners(
                gray,
                (cols, rows),
                getattr(cv2, "CALIB_CB_ADAPTIVE_THRESH", 0) | getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0),
            )
            if found:
                criteria = (
                    getattr(cv2, "TERM_CRITERIA_EPS", 2) + getattr(cv2, "TERM_CRITERIA_MAX_ITER", 1),
                    30,
                    0.001,
                )
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                samples.append((corners, size))
                captured += 1
            else:
                rejected += 1
        required = max(3, int(ctx.get("min_samples") or 12))
        return {
            **empty,
            "samples": len(samples),
            "report": (
                f"captured {captured} checkerboard view(s) for {key}; {len(samples)}/{required} stored"
                + (f"; rejected {rejected}" if rejected else "")
            ),
        }
    required = max(3, int(ctx.get("min_samples") or 12))
    if action == "solve" and len(samples) >= required:
        object_points = np.zeros((rows * cols, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
        rms, matrix, distortion, _, _ = cv2.calibrateCamera(
            [object_points.copy() for _ in samples],
            [item[0] for item in samples], samples[0][1], None)
        width, height = samples[0][1]
        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        fov_horizontal = float(2 * np.degrees(np.arctan(width / (2 * fx))))
        fov_vertical = float(2 * np.degrees(np.arctan(height / (2 * fy))))
        calibration = {
            "kind": "blacknode.camera-calibration",
            "schema_version": 1,
            "stream_id": key,
            "camera_model": "opencv-pinhole",
            "camera_matrix": matrix.tolist(),
            "distortion": distortion.reshape(-1).tolist(),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "rms_error": float(rms),
            "fov_horizontal": fov_horizontal,
            "fov_vertical": fov_vertical,
            "width": int(width), "height": int(height),
            "board_columns": cols, "board_rows": rows,
            "square_size": square, "samples": len(samples),
        }
        calibrated_stream = {
            **frame_stream,
            "stream_id": str(frame_stream.get("stream_id") or key),
            "camera_model": calibration["camera_model"],
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "fov_horizontal": fov_horizontal,
            "fov_vertical": fov_vertical,
            "distortion": calibration["distortion"],
            "calibration": calibration,
        }
        return {"calibration": calibration, "calibrated_stream": calibrated_stream,
                "samples": len(samples), "ready": True,
                "report": f"camera calibration solved for {key} with RMS error {float(rms):.5f}"}
    return {**empty, "samples": len(samples),
            "report": f"checkerboard samples: {len(samples)}; need {required} to solve"}
_COLOR_ALIASES: dict[str, str] = {
    "red": "red",
    "orange": "orange",
    "yellow": "yellow",
    "green": "green",
    "lime": "green",
    "cyan": "cyan",
    "turquoise": "cyan",
    "teal": "cyan",
    "blue": "blue",
    "purple": "purple",
    "violet": "purple",
    "magenta": "pink",
    "pink": "pink",
    "white": "white",
    "black": "black",
}
_OBJECT_WORDS = (
    "cube",
    "block",
    "box",
    "ball",
    "bottle",
    "cup",
    "marker",
    "object",
    "target",
)


def _camera_candidates(max_devices: int) -> list[tuple[int | str, str]]:
    if sys.platform.startswith("linux"):
        candidates = []
        for path in sorted(Path("/dev").glob("video*"))[:max_devices]:
            label_path = Path("/sys/class/video4linux") / path.name / "name"
            label = label_path.read_text(encoding="utf-8", errors="replace").strip() if label_path.exists() else path.name
            candidates.append((str(path), label))
        return candidates
    labels: list[str] = []
    if os.name == "nt":
        try:
            command = "Get-PnpDevice -PresentOnly | Where-Object {$_.Class -in @('Camera','Image')} | Select-Object -ExpandProperty FriendlyName"
            result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                labels = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except Exception:
            pass
    return [(index, labels[index] if index < len(labels) else f"Camera {index}") for index in range(max_devices)]


def _probe_backend(name: str) -> int | None:
    if cv2 is None or name == "auto":
        return None
    mapping = {
        "dshow": getattr(cv2, "CAP_DSHOW", None), "msmf": getattr(cv2, "CAP_MSMF", None),
        "v4l2": getattr(cv2, "CAP_V4L2", None), "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", None),
        "any": getattr(cv2, "CAP_ANY", None),
    }
    return mapping.get(name)


@node(name="CameraDiscovery", category="Camera", hidden=True,
      description="Discover and probe connected cameras so a specific device can be selected before streaming.",
      inputs={"refresh": AnyPort, "backend": Enum(["auto", "dshow", "msmf", "v4l2", "avfoundation", "any"], default="auto"),
              "max_devices": Int(default=8)},
      outputs={"found": Bool, "count": Int, "devices": List, "recommended": Dict, "discovery": Dict, "report": Text})
def cv2_camera_discovery(ctx: dict) -> dict:
    if cv2 is None:
        return {"found": False, "count": 0, "devices": [], "recommended": {}, "discovery": {},
                "report": _missing_cv2_outputs()["report"]}
    backend = str(ctx.get("backend") or "auto").lower()
    api = _probe_backend(backend)
    devices: list[dict[str, Any]] = []
    for hardware_index, (device, label) in enumerate(
        _camera_candidates(max(1, min(32, int(ctx.get("max_devices") or 8))))
    ):
        capture = None
        try:
            capture = cv2.VideoCapture(device) if api is None else cv2.VideoCapture(device, api)
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            devices.append({
                "kind": "blacknode.camera-device", "schema_version": 1,
                "index": hardware_index, "device": str(device), "label": label,
                "backend": backend, "width": int(width), "height": int(height),
            })
        except Exception:
            continue
        finally:
            if capture is not None:
                capture.release()
    discovery = {"kind": "blacknode.camera-discovery", "schema_version": 1, "devices": devices, "backend": backend}
    return {"found": bool(devices), "count": len(devices), "devices": devices,
            "recommended": dict(devices[0]) if devices else {}, "discovery": discovery,
            "report": f"found {len(devices)} camera(s)" if devices else "no usable cameras found"}


@node(name="CV2CameraDiscovery", category="Camera", hidden=True,
      description="Compatibility alias for CameraDiscovery.",
      inputs={"refresh": AnyPort, "backend": Enum(["auto", "dshow", "msmf", "v4l2", "avfoundation", "any"], default="auto"),
              "max_devices": Int(default=8)},
      outputs={"found": Bool, "count": Int, "devices": List, "recommended": Dict, "discovery": Dict, "report": Text})
def cv2_camera_discovery_compat(ctx: dict) -> dict:
    return cv2_camera_discovery(ctx)


@node(name="CameraSelect", category="Camera", hidden=True,
      description="Select one discovered camera by stable hardware index and emit its descriptor.",
      inputs={"trigger": AnyPort, "discovery": Dict(default={}), "selection": Int(default=0)},
      outputs={"selected": Bool, "camera": Dict, "device": Text, "backend": Text, "label": Text, "report": Text})
def cv2_camera_select(ctx: dict) -> dict:
    discovery = dict(ctx.get("discovery") or {})
    devices = [item for item in discovery.get("devices", []) if isinstance(item, dict)]
    selection = int(ctx.get("selection") or 0)
    camera = next(
        (
            item for item in devices
            if item.get("index") == selection or str(item.get("device", "")) == str(selection)
        ),
        None,
    )
    if camera is None:
        return {"selected": False, "camera": {}, "device": "", "backend": "", "label": "",
                "report": f"camera hardware index {selection} is unavailable; discovered {len(devices)} camera(s)"}
    camera = dict(camera)
    return {"selected": True, "camera": camera, "device": str(camera.get("device") or ""),
            "backend": str(camera.get("backend") or "auto"), "label": str(camera.get("label") or f"Camera {selection}"),
            "report": f"selected {camera.get('label') or camera.get('device')}"}


@node(name="CV2CameraSelect", category="Camera", hidden=True,
      description="Compatibility alias for CameraSelect.",
      inputs={"trigger": AnyPort, "discovery": Dict(default={}), "selection": Int(default=0)},
      outputs={"selected": Bool, "camera": Dict, "device": Text, "backend": Text, "label": Text, "report": Text})
def cv2_camera_select_compat(ctx: dict) -> dict:
    return cv2_camera_select(ctx)


def _missing_cv2_outputs() -> dict[str, Any]:
    report = (
        "CV2 node FAILED: OpenCV is not installed in this Blacknode Python environment. "
        "Run: blacknode packages setup blacknode-perception"
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


def _format_hsv(value: tuple[int, int, int]) -> str:
    return ",".join(str(int(part)) for part in value)


def _normalize_words(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _find_color(value: Any) -> str:
    text = _normalize_words(value)
    if not text:
        return ""
    words = set(text.split())
    for alias, color in _COLOR_ALIASES.items():
        if alias in words:
            return color
    return ""


def _find_object_label(value: Any, fallback: str) -> str:
    text = _normalize_words(value)
    if text:
        words = set(text.split())
        for word in _OBJECT_WORDS:
            if word in words:
                return word
    return fallback.strip() or "object"


def _read_reasoning_state_answer(state_url: str, wait_seconds: float) -> tuple[str, str]:
    url = state_url.strip()
    if not url:
        return "", ""
    deadline = time.monotonic() + max(0.0, wait_seconds)
    last_error = ""
    while True:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BlacknodeCV2TargetHint/0.1"})
            with urllib.request.urlopen(req, timeout=1.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            answer = str(payload.get("answer") or "").strip()
            report = str(payload.get("report") or "").strip()
            if answer:
                return answer, ""
            last_error = report or "reasoning state has no answer yet"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if time.monotonic() >= deadline:
            break
        time.sleep(0.35)
    return "", last_error


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@node(
    name="CV2ColorTargetHint",
    category=_CATEGORY,
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
    name="CameraStream",
    live=True,
    category="Camera",
    hidden=True,
    description="Open a local camera directly and serve live MJPEG and snapshot endpoints on Windows, Linux, or macOS.",
    inputs={
        "trigger": AnyPort,
        "action": Enum(["start", "stop"], default="start"),
        "stream_id": Text(default="local_camera"),
        "camera": Dict(default={}),
        "device": Text(default="0"),
        "backend": Enum(["auto", "dshow", "msmf", "v4l2", "avfoundation", "any"], default="auto"),
        "width": Int(default=640),
        "height": Int(default=480),
        "host": Text(default="127.0.0.1"),
        "port": Int(default=0),
        "max_fps": Float(default=15.0),
        "max_width": Int(default=960),
        "jpeg_quality": Int(default=82),
    },
    outputs={
        "preview": Image,
        "streaming": Bool,
        "stream_url": Text,
        "snapshot_url": Text,
        "health_url": Text,
        "frame_stream": Dict,
        "stream_id": Text,
        "report": Text,
    },
)
def cv2_camera_stream(ctx: dict) -> dict:
    camera = dict(ctx.get("camera") or {})
    selected_device = str(camera.get("device") or "").strip()
    stream_id = str(ctx.get("stream_id") or "").strip()
    if not stream_id:
        suffix = re.sub(r"[^a-zA-Z0-9_-]+", "_", selected_device).strip("_")
        stream_id = f"camera_{suffix}" if suffix else "local_camera"
    empty = {
        "preview": "",
        "streaming": False,
        "stream_url": "",
        "snapshot_url": "",
        "health_url": "",
        "frame_stream": {},
        "stream_id": stream_id,
    }
    if str(ctx.get("action") or "start").strip().lower() == "stop":
        result = cv2_runtime.stop_camera_stream(stream_id)
        return {**empty, "report": f"stopped {result.get('stopped', 0)} camera stream(s)"}
    if cv2 is None or np is None:
        return {**empty, "report": _missing_cv2_outputs()["report"]}
    device = str(camera.get("device") if camera.get("device") is not None else ctx.get("device") if ctx.get("device") is not None else "0").strip() or "0"
    backend = str(camera.get("backend") or ctx.get("backend") or "auto").strip().lower() or "auto"
    result = cv2_runtime.start_camera_stream(
        stream_id=stream_id,
        device=device,
        backend=backend,
        width=max(0, int(ctx.get("width") or 0)),
        height=max(0, int(ctx.get("height") or 0)),
        host=str(ctx.get("host") or "127.0.0.1").strip() or "127.0.0.1",
        port=max(0, int(ctx.get("port") or 0)),
        max_fps=max(0.1, min(60.0, float(ctx.get("max_fps") or 15.0))),
        max_width=max(0, int(ctx.get("max_width") or 960)),
        jpeg_quality=max(1, min(100, int(ctx.get("jpeg_quality") or 82))),
    )
    if not result.get("ok"):
        return {**empty, "report": f"camera stream FAILED: {result.get('error', 'unknown error')}"}
    stream_url = str(result.get("stream_url") or "")
    snapshot_url = str(result.get("snapshot_url") or "")
    health_url = str(result.get("health_url") or "")
    frame_stream = {
        "kind": "blacknode.frame-stream",
        "schema_version": 1,
        "stream_id": stream_id,
        "snapshot_url": snapshot_url,
        "health_url": health_url,
        "media_type": "image/jpeg",
        "mode": "latest",
        "clock": "unix_ns",
    }
    health = result.get("health") if isinstance(result.get("health"), dict) else {}
    return {
        "preview": stream_url,
        "streaming": True,
        "stream_url": stream_url,
        "snapshot_url": snapshot_url,
        "health_url": health_url,
        "frame_stream": frame_stream,
        "stream_id": stream_id,
        "report": f"LIVE CAMERA STREAM running on {stream_url}; {health.get('report', 'camera ready')}",
    }


_CAMERA_STREAM_INPUTS = {
    "trigger": AnyPort,
    "action": Enum(["start", "stop"], default="start"),
    "stream_id": Text(default="local_camera"),
    "camera": Dict(default={}),
    "device": Text(default="0"),
    "backend": Enum(["auto", "dshow", "msmf", "v4l2", "avfoundation", "any"], default="auto"),
    "width": Int(default=640),
    "height": Int(default=480),
    "host": Text(default="127.0.0.1"),
    "port": Int(default=0),
    "max_fps": Float(default=15.0),
    "max_width": Int(default=960),
    "jpeg_quality": Int(default=82),
}
_CAMERA_STREAM_OUTPUTS = {
    "preview": Image,
    "streaming": Bool,
    "stream_url": Text,
    "snapshot_url": Text,
    "health_url": Text,
    "frame_stream": Dict,
    "stream_id": Text,
    "report": Text,
}


@node(name="CV2CameraStream", live=True, category="Camera", hidden=True,
      description="Compatibility alias for CameraStream.",
      inputs=_CAMERA_STREAM_INPUTS, outputs=_CAMERA_STREAM_OUTPUTS)
def cv2_camera_stream_compat(ctx: dict) -> dict:
    return cv2_camera_stream(ctx)


@node(
    name="Camera",
    live=True,
    category="Camera",
    description="One easy camera node: discover connected cameras, select one by number, and show its live preview.",
    primary_inputs=["trigger"],
    primary_outputs=["preview", "frame_stream", "report"],
    inputs={
        "trigger": AnyPort,
        "action": Enum(["start", "stop"], default="start"),
        "selection": Int(default=0),
        "stream_id": Text(default=""),
        "camera": Dict(default={}),
        "device": Text(default=""),
        "backend": Enum(["auto", "dshow", "msmf", "v4l2", "avfoundation", "any"], default="auto"),
        "max_devices": Int(default=8),
        "width": Int(default=640),
        "height": Int(default=480),
        "host": Text(default="127.0.0.1"),
        "port": Int(default=0),
        "max_fps": Float(default=15.0),
        "max_width": Int(default=960),
        "jpeg_quality": Int(default=82),
    },
    outputs={
        "found": Bool,
        "count": Int,
        "devices": List,
        "camera": Dict,
        "label": Text,
        **_CAMERA_STREAM_OUTPUTS,
    },
)
def camera(ctx: dict) -> dict:
    if str(ctx.get("action") or "start").strip().lower() == "stop" and str(ctx.get("stream_id") or "").strip():
        streamed = cv2_camera_stream(ctx)
        return {"found": False, "count": 0, "devices": [], "camera": {}, "label": "", **streamed}
    supplied_camera = dict(ctx.get("camera") or {})
    supplied_device = str(ctx.get("device") or "").strip()
    if supplied_camera or supplied_device:
        selected_camera = supplied_camera or {
            "kind": "blacknode.camera-device", "schema_version": 1,
            "device": supplied_device, "label": f"Camera {supplied_device}",
            "backend": str(ctx.get("backend") or "auto"),
        }
        discovery = {"found": True, "count": 1, "devices": [selected_camera], "report": "using configured camera"}
        selected = {"selected": True, "camera": selected_camera, "label": str(selected_camera.get("label") or supplied_device)}
    else:
        discovery = cv2_camera_discovery(ctx)
        selected = cv2_camera_select({"discovery": discovery.get("discovery", {}), "selection": ctx.get("selection", 0)})
        if not selected.get("selected"):
            selection = int(ctx.get("selection") or 0)
            candidates = _camera_candidates(max(1, min(32, int(ctx.get("max_devices") or 8))))
            if 0 <= selection < len(candidates):
                device, label = candidates[selection]
                selected_camera = {
                    "kind": "blacknode.camera-device",
                    "schema_version": 1,
                    "index": selection,
                    "device": str(device),
                    "label": label,
                    "backend": str(ctx.get("backend") or "auto"),
                }
                selected = {
                    "selected": True,
                    "camera": selected_camera,
                    "device": str(device),
                    "backend": selected_camera["backend"],
                    "label": label,
                    "report": f"selected hardware camera {selection}",
                }
    base = {
        "found": bool(discovery.get("found")),
        "count": int(discovery.get("count") or 0),
        "devices": list(discovery.get("devices") or []),
        "camera": dict(selected.get("camera") or {}),
        "label": str(selected.get("label") or ""),
    }
    if not selected.get("selected"):
        return {
            **base,
            "preview": "", "streaming": False, "stream_url": "", "snapshot_url": "", "health_url": "",
            "frame_stream": {}, "stream_id": str(ctx.get("stream_id") or ""),
            "report": f"{discovery.get('report', '')}; {selected.get('report', '')}".strip("; "),
        }
    streamed = cv2_camera_stream({**ctx, "camera": selected["camera"]})
    return {**base, **streamed}


@node(
    name="CV2ColorObjectStream",
    live=True,
    category=_CATEGORY,
    description="Start or stop a live MJPEG stream with OpenCV color tracking overlay and detection JSON.",
    inputs={
        "trigger": AnyPort,
        "action": Enum(["start", "stop"], default="start"),
        "stream_id": Text(default="cube_tracker"),
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

    source_url = str(ctx.get("source_url") or "").strip()
    if not source_url:
        return {**empty, "report": "CV2 stream FAILED: connect source_url to a camera snapshot URL"}

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
