"""Runtime helpers for live OpenCV streams started by blacknode-vision."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

_STREAMS: dict[str, dict[str, Any]] = {}
_CAMERA_STREAMS: dict[str, dict[str, Any]] = {}
_REASONING_STREAMS: dict[str, dict[str, Any]] = {}


def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "cv2_color_stream_server.py"


def _reasoning_script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "vision_reasoning_stream_server.py"


def _camera_script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "cv2_camera_stream_server.py"


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _port_open(host: str, port: int, timeout: float = 0.15) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _read_json(url: str, timeout: float = 1.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "BlacknodeCV2Runtime/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any], timeout: float = 1.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "BlacknodeCV2Runtime/0.1"},
        method="PATCH",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _terminate_process(proc: subprocess.Popen) -> bool:
    if proc.poll() is not None:
        return False
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
    return True


def start_color_stream(
    *,
    stream_id: str,
    source_url: str,
    object_color: str,
    use_reasoning_color: bool,
    label: str,
    target_text: str,
    reasoning_state_url: str,
    target_update_seconds: float,
    show_follow_guides: bool,
    follow_target_x: float,
    follow_deadband: float,
    min_area: int,
    max_detections: int,
    blur: int,
    morphology_iters: int,
    host: str,
    port: int,
    max_fps: float,
    max_width: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    existing = _STREAMS.get(stream_id)
    if existing and existing.get("proc") is not None and existing["proc"].poll() is None:
        update_result = update_color_stream_config(stream_id, {
            "source_url": source_url,
            "object_color": object_color,
            "use_reasoning_color": _bool_value(use_reasoning_color, True),
            "label": label,
            "target_text": target_text,
            "reasoning_state_url": reasoning_state_url,
            "target_update_seconds": target_update_seconds,
            "show_follow_guides": bool(show_follow_guides),
            "follow_target_x": follow_target_x,
            "follow_deadband": follow_deadband,
            "min_area": min_area,
            "max_detections": max_detections,
            "blur": blur,
            "morphology_iters": morphology_iters,
            "max_fps": max_fps,
            "max_width": max_width,
            "jpeg_quality": jpeg_quality,
        })
        detection: dict[str, Any] = {}
        try:
            detection = _read_json(str(existing.get("detection_url") or ""), timeout=0.5)
        except Exception:
            pass
        return {
            "ok": bool(update_result.get("ok", True)),
            "stream_id": stream_id,
            "stream_url": existing.get("stream_url", ""),
            "snapshot_url": existing.get("snapshot_url", ""),
            "mask_stream_url": existing.get("mask_stream_url", ""),
            "mask_url": existing.get("mask_url", ""),
            "detection_url": existing.get("detection_url", ""),
            "detection": detection,
            "port": int(str(existing.get("stream_url", "")).rsplit(":", 1)[-1].split("/", 1)[0] or 0) if ":" in str(existing.get("stream_url", "")) else 0,
            "updated": update_result.get("updated", []),
        }
    stop_color_stream(stream_id)
    script = _script_path()
    if not script.exists():
        return {"ok": False, "error": f"CV2 stream helper not found: {script}"}
    selected_port = int(port) if int(port) > 0 else _free_port(host)
    args = [
        sys.executable,
        str(script),
        "--source-url",
        source_url,
        "--object-color",
        object_color,
        "--use-reasoning-color",
        "true" if _bool_value(use_reasoning_color, True) else "false",
        "--label",
        label,
        "--target-text",
        target_text,
        "--reasoning-state-url",
        reasoning_state_url,
        "--target-update-seconds",
        str(target_update_seconds),
        "--show-follow-guides",
        "true" if show_follow_guides else "false",
        "--follow-target-x",
        str(follow_target_x),
        "--follow-deadband",
        str(follow_deadband),
        "--min-area",
        str(min_area),
        "--max-detections",
        str(max_detections),
        "--blur",
        str(blur),
        "--morphology-iters",
        str(morphology_iters),
        "--host",
        host,
        "--port",
        str(selected_port),
        "--max-fps",
        str(max_fps),
        "--max-width",
        str(max_width),
        "--jpeg-quality",
        str(jpeg_quality),
    ]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return {"ok": False, "error": "CV2 stream helper exited before opening its HTTP port"}
        if _port_open(host, selected_port):
            break
        time.sleep(0.05)
    else:
        _terminate_process(proc)
        return {"ok": False, "error": f"CV2 stream helper did not open http://{host}:{selected_port}"}

    base_url = f"http://{host}:{selected_port}"
    detection_url = f"{base_url}/detection.json"
    detection: dict[str, Any] = {}
    wait_until = time.monotonic() + 3.0
    while time.monotonic() < wait_until:
        try:
            detection = _read_json(detection_url, timeout=0.5)
            if detection.get("updated_at"):
                break
        except Exception:
            pass
        time.sleep(0.1)

    _STREAMS[stream_id] = {
        "proc": proc,
        "source_url": source_url,
        "config_url": f"{base_url}/config.json",
        "stream_url": f"{base_url}/stream.mjpg",
        "snapshot_url": f"{base_url}/snapshot.jpg",
        "mask_stream_url": f"{base_url}/mask.mjpg",
        "mask_url": f"{base_url}/mask.png",
        "detection_url": detection_url,
        "object_color": object_color,
        "use_reasoning_color": _bool_value(use_reasoning_color, True),
        "label": label,
        "target_text": target_text,
        "reasoning_state_url": reasoning_state_url,
    }
    return {
        "ok": True,
        "stream_id": stream_id,
        "stream_url": _STREAMS[stream_id]["stream_url"],
        "snapshot_url": _STREAMS[stream_id]["snapshot_url"],
        "mask_stream_url": _STREAMS[stream_id]["mask_stream_url"],
        "mask_url": _STREAMS[stream_id]["mask_url"],
        "detection_url": detection_url,
        "detection": detection,
        "port": selected_port,
    }


def start_camera_stream(
    *,
    stream_id: str,
    device: str,
    backend: str,
    width: int,
    height: int,
    host: str,
    port: int,
    max_fps: float,
    max_width: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    existing = _CAMERA_STREAMS.get(stream_id)
    if existing and existing.get("proc") is not None and existing["proc"].poll() is None:
        return {"ok": True, "stream_id": stream_id, **{key: existing[key] for key in ("stream_url", "snapshot_url", "health_url")}}
    stop_camera_stream(stream_id)
    script = _camera_script_path()
    if not script.exists():
        return {"ok": False, "error": f"camera stream helper not found: {script}"}
    selected_port = int(port) if int(port) > 0 else _free_port(host)
    args = [
        sys.executable, str(script),
        "--device", device,
        "--backend", backend,
        "--width", str(width),
        "--height", str(height),
        "--host", host,
        "--port", str(selected_port),
        "--max-fps", str(max_fps),
        "--max-width", str(max_width),
        "--jpeg-quality", str(jpeg_quality),
    ]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    base_url = f"http://{host}:{selected_port}"
    health_url = f"{base_url}/health.json"
    deadline = time.monotonic() + 8.0
    health: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return {"ok": False, "error": "camera stream helper exited before opening its HTTP port"}
        try:
            health = _read_json(health_url, timeout=0.5)
            if health.get("streaming"):
                break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        _terminate_process(proc)
        return {"ok": False, "error": str(health.get("report") or f"camera {device!r} did not produce a frame")}
    item = {
        "proc": proc,
        "device": device,
        "backend": backend,
        "stream_url": f"{base_url}/stream.mjpg",
        "snapshot_url": f"{base_url}/snapshot.jpg",
        "health_url": health_url,
        "health": health,
    }
    _CAMERA_STREAMS[stream_id] = item
    return {"ok": True, "stream_id": stream_id, **{key: item[key] for key in ("stream_url", "snapshot_url", "health_url")}, "health": health}


def start_reasoning_stream(
    *,
    stream_id: str,
    image_url: str,
    detection_url: str,
    prompt: str,
    system: str,
    provider: str,
    model: str,
    endpoint_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    interval_seconds: float,
    host: str,
    port: int,
    max_fps: float,
    max_width: int,
    title: str,
) -> dict[str, Any]:
    existing = _REASONING_STREAMS.get(stream_id)
    if existing and existing.get("proc") is not None and existing["proc"].poll() is None:
        update_result = update_reasoning_stream_config(stream_id, {
            "image_url": image_url,
            "detection_url": detection_url,
            "prompt": prompt,
            "system": system,
            "provider": provider,
            "model": model,
            "endpoint_url": endpoint_url,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "interval_seconds": interval_seconds,
        })
        state: dict[str, Any] = {}
        try:
            state = _read_json(str(existing.get("state_url") or ""), timeout=0.5)
        except Exception:
            pass
        return {
            "ok": bool(update_result.get("ok", True)),
            "stream_id": stream_id,
            "stream_url": existing.get("stream_url", ""),
            "snapshot_url": existing.get("snapshot_url", ""),
            "state_url": existing.get("state_url", ""),
            "state": state,
            "updated": update_result.get("updated", []),
        }

    stop_reasoning_stream(stream_id)
    script = _reasoning_script_path()
    if not script.exists():
        return {"ok": False, "error": f"reasoning stream helper not found: {script}"}
    selected_port = int(port) if int(port) > 0 else _free_port(host)
    args = [
        sys.executable,
        str(script),
        "--image-url",
        image_url,
        "--detection-url",
        detection_url,
        "--prompt",
        prompt,
        "--system",
        system,
        "--provider",
        provider,
        "--model",
        model,
        "--endpoint-url",
        endpoint_url,
        "--api-key",
        api_key,
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--interval-seconds",
        str(interval_seconds),
        "--host",
        host,
        "--port",
        str(selected_port),
        "--max-fps",
        str(max_fps),
        "--max-width",
        str(max_width),
        "--title",
        title,
    ]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return {"ok": False, "error": "reasoning stream helper exited before opening its HTTP port"}
        if _port_open(host, selected_port):
            break
        time.sleep(0.05)
    else:
        _terminate_process(proc)
        return {"ok": False, "error": f"reasoning stream helper did not open http://{host}:{selected_port}"}

    base_url = f"http://{host}:{selected_port}"
    state_url = f"{base_url}/state.json"
    state: dict[str, Any] = {}
    wait_until = time.monotonic() + 2.0
    while time.monotonic() < wait_until:
        try:
            state = _read_json(state_url, timeout=0.5)
            if state.get("updated_at"):
                break
        except Exception:
            pass
        time.sleep(0.1)

    _REASONING_STREAMS[stream_id] = {
        "proc": proc,
        "image_url": image_url,
        "detection_url": detection_url,
        "stream_url": f"{base_url}/dashboard.mjpg",
        "snapshot_url": f"{base_url}/dashboard.jpg",
        "state_url": state_url,
        "config_url": f"{base_url}/config.json",
        "model": model,
    }
    return {
        "ok": True,
        "stream_id": stream_id,
        "stream_url": _REASONING_STREAMS[stream_id]["stream_url"],
        "snapshot_url": _REASONING_STREAMS[stream_id]["snapshot_url"],
        "state_url": state_url,
        "state": state,
        "port": selected_port,
    }


def stop_color_stream(stream_id: str = "") -> dict[str, Any]:
    ids = [stream_id] if stream_id else list(_STREAMS)
    stopped = 0
    for sid in ids:
        item = _STREAMS.pop(sid, None)
        if not item:
            continue
        if _terminate_process(item["proc"]):
            stopped += 1
    return {"ok": True, "stopped": stopped}


def stop_camera_stream(stream_id: str = "") -> dict[str, Any]:
    ids = [stream_id] if stream_id else list(_CAMERA_STREAMS)
    stopped = 0
    for sid in ids:
        item = _CAMERA_STREAMS.pop(sid, None)
        if item and _terminate_process(item["proc"]):
            stopped += 1
    return {"ok": True, "stopped": stopped}


def update_reasoning_stream_config(stream_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    item = _REASONING_STREAMS.get(stream_id)
    if not item:
        return {"ok": True, "active": False, "updated": [], "report": f"reasoning stream '{stream_id}' is not running"}
    proc = item.get("proc")
    if proc is None or proc.poll() is not None:
        _REASONING_STREAMS.pop(stream_id, None)
        return {"ok": True, "active": False, "updated": [], "report": f"reasoning stream '{stream_id}' has stopped"}
    config_url = str(item.get("config_url") or "")
    if not config_url:
        return {"ok": False, "active": True, "error": f"reasoning stream '{stream_id}' has no config endpoint"}
    try:
        result = _post_json(config_url, updates, timeout=1.0)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "active": True, "error": f"{type(exc).__name__}: {exc}"}
    if "image_url" in updates:
        item["image_url"] = str(updates.get("image_url") or "")
    if "detection_url" in updates:
        item["detection_url"] = str(updates.get("detection_url") or "")
    if "model" in updates:
        item["model"] = str(updates.get("model") or "")
    return {"ok": True, "active": True, **result}


def stop_reasoning_stream(stream_id: str = "") -> dict[str, Any]:
    ids = [stream_id] if stream_id else list(_REASONING_STREAMS)
    stopped = 0
    for sid in ids:
        item = _REASONING_STREAMS.pop(sid, None)
        if not item:
            continue
        if _terminate_process(item["proc"]):
            stopped += 1
    return {"ok": True, "stopped": stopped}


def runtime_status() -> dict[str, Any]:
    streams: list[dict[str, Any]] = []
    for stream_id, item in list(_STREAMS.items()):
        proc = item.get("proc")
        if proc is None or proc.poll() is not None:
            _STREAMS.pop(stream_id, None)
            continue
        streams.append({
            "stream_id": stream_id,
            "source_url": item.get("source_url", ""),
            "stream_url": item.get("stream_url", ""),
            "snapshot_url": item.get("snapshot_url", ""),
            "mask_stream_url": item.get("mask_stream_url", ""),
            "mask_url": item.get("mask_url", ""),
            "detection_url": item.get("detection_url", ""),
            "label": item.get("label", ""),
            "object_color": item.get("object_color", ""),
            "use_reasoning_color": _bool_value(item.get("use_reasoning_color"), True),
        })
    reasoning_streams: list[dict[str, Any]] = []
    for stream_id, item in list(_REASONING_STREAMS.items()):
        proc = item.get("proc")
        if proc is None or proc.poll() is not None:
            _REASONING_STREAMS.pop(stream_id, None)
            continue
        reasoning_streams.append({
            "stream_id": stream_id,
            "image_url": item.get("image_url", ""),
            "detection_url": item.get("detection_url", ""),
            "stream_url": item.get("stream_url", ""),
            "snapshot_url": item.get("snapshot_url", ""),
            "state_url": item.get("state_url", ""),
            "model": item.get("model", ""),
        })
    camera_streams: list[dict[str, Any]] = []
    for stream_id, item in list(_CAMERA_STREAMS.items()):
        proc = item.get("proc")
        if proc is None or proc.poll() is not None:
            _CAMERA_STREAMS.pop(stream_id, None)
            continue
        camera_streams.append({
            "stream_id": stream_id,
            "device": item.get("device", ""),
            "backend": item.get("backend", ""),
            "stream_url": item.get("stream_url", ""),
            "snapshot_url": item.get("snapshot_url", ""),
            "health_url": item.get("health_url", ""),
        })
    return {
        "ok": True,
        "active": bool(camera_streams or streams or reasoning_streams),
        "cv2_streams": camera_streams + streams,
        "reasoning_streams": reasoning_streams,
    }


def update_color_stream_config(stream_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    item = _STREAMS.get(stream_id)
    if not item:
        return {"ok": True, "active": False, "updated": [], "report": f"CV2 stream '{stream_id}' is not running"}
    proc = item.get("proc")
    if proc is None or proc.poll() is not None:
        _STREAMS.pop(stream_id, None)
        return {"ok": True, "active": False, "updated": [], "report": f"CV2 stream '{stream_id}' has stopped"}
    config_url = str(item.get("config_url") or "")
    if not config_url:
        return {"ok": False, "active": True, "error": f"CV2 stream '{stream_id}' has no config endpoint"}
    try:
        result = _post_json(config_url, updates, timeout=1.0)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "active": True, "error": f"{type(exc).__name__}: {exc}"}
    if "target_text" in updates:
        item["target_text"] = str(updates.get("target_text") or "")
    if "reasoning_state_url" in updates:
        item["reasoning_state_url"] = str(updates.get("reasoning_state_url") or "")
    if "source_url" in updates:
        item["source_url"] = str(updates.get("source_url") or "")
    if "object_color" in updates:
        item["object_color"] = str(updates.get("object_color") or "")
    if "use_reasoning_color" in updates:
        item["use_reasoning_color"] = _bool_value(updates.get("use_reasoning_color"), True)
    return {"ok": True, "active": True, **result}


def stop_runtime_services() -> dict[str, Any]:
    before = runtime_status()
    camera_result = stop_camera_stream("")
    color_result = stop_color_stream("")
    reasoning_result = stop_reasoning_stream("")
    stopped = {
        "cv2_streams": int(camera_result.get("stopped") or 0) + int(color_result.get("stopped") or 0),
        "reasoning_streams": int(reasoning_result.get("stopped") or 0),
    }
    return {
        "ok": True,
        "active_before": before,
        "stopped": stopped,
        "report": f"stopped {stopped['cv2_streams']} CV2 stream(s), {stopped['reasoning_streams']} reasoning stream(s)",
    }
