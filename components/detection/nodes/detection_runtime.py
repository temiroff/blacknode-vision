"""Lifecycle for the detection MJPEG stream subprocess.

Mirrors the camera/colour-tracker runtimes: spawn the annotated-stream server
detached, key it by stream_id, and stop it through the shared cross-platform
terminator so a stopped stream releases its port. See the memory rules on
stream-server lifecycle.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from blacknode.process import terminate_tree

_STREAMS: dict[str, dict[str, Any]] = {}


def _server_script() -> Path:
    return Path(__file__).resolve().parents[1] / "runtime" / "detection_stream_server.py"


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


def start_detection_stream(
    *,
    stream_id: str,
    source_url: str,
    mode: str,
    model: str = "yolov8n.pt",
    conf: float = 0.35,
    node_id: str = "",
    host: str = "127.0.0.1",
    port: int = 0,
    max_fps: float = 10.0,
    max_width: int = 960,
    jpeg_quality: int = 80,
) -> dict[str, Any]:
    existing = _STREAMS.get(stream_id)
    if existing and existing.get("proc") is not None and existing["proc"].poll() is None:
        # Same stream already running; a re-cook reuses it rather than leaking a
        # second server on the same source.
        return {"ok": True, "stream_id": stream_id, **{k: existing[k] for k in ("stream_url", "snapshot_url")}}

    stop_detection_stream(stream_id)
    script = _server_script()
    if not script.exists():
        return {"ok": False, "error": f"detection stream server not found: {script}"}
    selected_port = int(port) if int(port) > 0 else _free_port(host)
    args = [
        sys.executable, str(script),
        "--source-url", source_url,
        "--mode", str(mode or "motion"),
        "--model", str(model or "yolov8n.pt"),
        "--conf", str(conf),
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

    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return {"ok": False, "error": "detection stream server exited before opening its port"}
        if _port_open(host, selected_port):
            break
        time.sleep(0.05)
    else:
        terminate_tree(proc)
        return {"ok": False, "error": f"detection stream server did not open http://{host}:{selected_port}"}

    base = f"http://{host}:{selected_port}"
    item = {
        "proc": proc, "node_id": node_id, "mode": str(mode or "motion"),
        "stream_url": f"{base}/stream.mjpg",
        "snapshot_url": f"{base}/snapshot.jpg",
        "detection_url": f"{base}/detection.json",
    }
    _STREAMS[stream_id] = item
    return {"ok": True, "stream_id": stream_id,
            **{k: item[k] for k in ("stream_url", "snapshot_url", "detection_url")}}


def stop_detection_stream(stream_id: str = "") -> dict[str, Any]:
    ids = [stream_id] if stream_id else list(_STREAMS)
    stopped = 0
    for sid in ids:
        item = _STREAMS.pop(sid, None)
        if item and terminate_tree(item.get("proc")):
            stopped += 1
    return {"ok": True, "stopped": stopped}


def runtime_status() -> dict[str, Any]:
    streams = []
    for stream_id, item in list(_STREAMS.items()):
        proc = item.get("proc")
        if proc is None or proc.poll() is not None:
            _STREAMS.pop(stream_id, None)
            continue
        streams.append({
            "stream_id": stream_id, "node_id": item.get("node_id", ""),
            "mode": item.get("mode", ""), "stream_url": item.get("stream_url", ""),
            "snapshot_url": item.get("snapshot_url", ""),
        })
    return {"ok": True, "active": bool(streams), "streams": streams}


def stop_runtime_services() -> dict[str, Any]:
    result = stop_detection_stream("")
    return {"ok": True, "stopped": {"streams": int(result.get("stopped") or 0)},
            "report": f"stopped {result.get('stopped', 0)} detection stream(s)"}
