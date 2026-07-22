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
from .image_ops import _decode_image_bgr, _missing_cv2_outputs

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
        # Lets the editor reattach this stream to the node that owns it after a
        # tab switch; stream_id is derived here, so the graph never sees it.
        node_id=str(ctx.get("__node_id__") or ""),
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


