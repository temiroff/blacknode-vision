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
_REASONING_STREAMS: dict[str, dict[str, Any]] = {}


def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "cv2_color_stream_server.py"


def _reasoning_script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "vision_reasoning_stream_server.py"


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
    label: str,
    lower_hsv: str,
    upper_hsv: str,
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
        "--label",
        label,
        "--lower-hsv",
        lower_hsv,
        "--upper-hsv",
        upper_hsv,
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
        "stream_url": f"{base_url}/stream.mjpg",
        "snapshot_url": f"{base_url}/snapshot.jpg",
        "mask_stream_url": f"{base_url}/mask.mjpg",
        "mask_url": f"{base_url}/mask.png",
        "detection_url": detection_url,
        "label": label,
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
    return {
        "ok": True,
        "active": bool(streams or reasoning_streams),
        "cv2_streams": streams,
        "reasoning_streams": reasoning_streams,
    }


def stop_runtime_services() -> dict[str, Any]:
    before = runtime_status()
    color_result = stop_color_stream("")
    reasoning_result = stop_reasoning_stream("")
    stopped = {
        "cv2_streams": int(color_result.get("stopped") or 0),
        "reasoning_streams": int(reasoning_result.get("stopped") or 0),
    }
    return {
        "ok": True,
        "active_before": before,
        "stopped": stopped,
        "report": f"stopped {stopped['cv2_streams']} CV2 stream(s), {stopped['reasoning_streams']} reasoning stream(s)",
    }
