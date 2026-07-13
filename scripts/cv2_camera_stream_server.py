"""Serve a native OpenCV camera as snapshot and MJPEG endpoints."""
from __future__ import annotations

import argparse
import json
import platform
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.jpeg = b""
        self.health: dict[str, Any] = {
            "ok": False,
            "streaming": False,
            "frames": 0,
            "report": "opening camera",
        }

    def update(self, jpeg: bytes, **health: Any) -> None:
        with self.lock:
            if jpeg:
                self.jpeg = jpeg
            self.health.update(health)

    def snapshot(self) -> tuple[bytes, dict[str, Any]]:
        with self.lock:
            return self.jpeg, dict(self.health)


def _backend_candidates(name: str) -> list[tuple[str, int]]:
    normalized = str(name or "auto").strip().lower()
    explicit = {
        "any": cv2.CAP_ANY,
        "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
        "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
    }
    if normalized != "auto":
        return [(normalized, explicit.get(normalized, cv2.CAP_ANY))]
    system = platform.system().lower()
    if system == "windows":
        return [("dshow", explicit["dshow"]), ("msmf", explicit["msmf"]), ("any", cv2.CAP_ANY)]
    if system == "darwin":
        return [("avfoundation", explicit["avfoundation"]), ("any", cv2.CAP_ANY)]
    return [("v4l2", explicit["v4l2"]), ("any", cv2.CAP_ANY)]


def open_camera(device: int | str, backend: str, width: int, height: int) -> tuple[Any, str]:
    errors: list[str] = []
    for label, api in _backend_candidates(backend):
        capture = cv2.VideoCapture(device, api)
        if width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if capture.isOpened():
            return capture, label
        capture.release()
        errors.append(label)
    raise RuntimeError(f"could not open camera {device!r} with backend(s): {', '.join(errors)}")


def capture_loop(args: argparse.Namespace, state: SharedState) -> None:
    device: int | str = int(args.device) if str(args.device).isdigit() else str(args.device)
    capture = None
    try:
        capture, backend = open_camera(device, args.backend, args.width, args.height)
        frames = 0
        failures = 0
        while not state.stop.is_set():
            started = time.monotonic()
            ok, frame = capture.read()
            if not ok or frame is None:
                failures += 1
                state.update(b"", ok=False, streaming=False, report=f"camera frame read failed ({failures})")
                time.sleep(0.1)
                continue
            failures = 0
            frames += 1
            if args.max_width > 0 and frame.shape[1] > args.max_width:
                scale = args.max_width / float(frame.shape[1])
                frame = cv2.resize(frame, (args.max_width, max(1, int(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            encoded_ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, args.jpeg_quality))]
            )
            if not encoded_ok:
                state.update(b"", ok=False, streaming=False, report="OpenCV JPEG encode failed")
                continue
            state.update(
                encoded.tobytes(),
                ok=True,
                streaming=True,
                frames=frames,
                backend=backend,
                device=device,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                updated_at=time.time(),
                report=f"camera {device!r} streaming via {backend}",
            )
            delay = max(0.0, (1.0 / max(0.1, args.max_fps)) - (time.monotonic() - started))
            state.stop.wait(delay)
    except Exception as exc:  # noqa: BLE001
        state.update(b"", ok=False, streaming=False, report=f"{type(exc).__name__}: {exc}")
    finally:
        if capture is not None:
            capture.release()


def make_handler(state: SharedState, max_fps: float):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            jpeg, health = state.snapshot()
            if self.path.startswith("/health.json"):
                body = json.dumps(health).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/snapshot.jpg"):
                if not jpeg:
                    self.send_error(503, str(health.get("report") or "camera has no frame"))
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while not state.stop.is_set():
                    jpeg, _health = state.snapshot()
                    if not jpeg:
                        time.sleep(0.05)
                        continue
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg + b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    time.sleep(1.0 / max(0.1, max_fps))
                return
            self.send_error(404, "not found")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-fps", type=float, default=15.0)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    shared = SharedState()
    signal.signal(signal.SIGTERM, lambda _sig, _frame: shared.stop.set())
    thread = threading.Thread(target=capture_loop, args=(args, shared), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(shared, args.max_fps))
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        shared.stop.set()
        server.server_close()
