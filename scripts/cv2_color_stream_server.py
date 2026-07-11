#!/usr/bin/env python3
"""HTTP MJPEG server that overlays CV2 color-object tracking on a snapshot URL."""
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: bytes = b""
        self.detection: dict[str, Any] = {
            "ok": False,
            "found": False,
            "detections": [],
            "report": "waiting for first frame",
            "updated_at": 0.0,
        }
        self.stop = threading.Event()

    def jpeg_snapshot(self) -> bytes:
        with self.lock:
            return self.jpeg

    def detection_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.detection)


def parse_hsv(value: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    parts = [part for part in value.replace(";", ",").replace(" ", ",").split(",") if part]
    values = list(default)
    for index, part in enumerate(parts[:3]):
        try:
            values[index] = int(float(part))
        except ValueError:
            values[index] = default[index]
    return (
        max(0, min(179, values[0])),
        max(0, min(255, values[1])),
        max(0, min(255, values[2])),
    )


def fetch_frame(source_url: str, timeout: float) -> Any:
    req = urllib.request.Request(source_url, headers={"User-Agent": "BlacknodeCV2Stream/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    data = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("source did not return a decodable image")
    return frame


def build_mask(
    frame: Any,
    *,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
    blur: int,
    morphology_iters: int,
) -> Any:
    blur = max(0, int(blur))
    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        frame = cv2.GaussianBlur(frame, (blur, blur), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if lower_hsv[0] <= upper_hsv[0]:
        mask = cv2.inRange(hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8))
    else:
        low_a = np.array([lower_hsv[0], lower_hsv[1], lower_hsv[2]], dtype=np.uint8)
        high_a = np.array([179, upper_hsv[1], upper_hsv[2]], dtype=np.uint8)
        low_b = np.array([0, lower_hsv[1], lower_hsv[2]], dtype=np.uint8)
        high_b = np.array([upper_hsv[0], upper_hsv[1], upper_hsv[2]], dtype=np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, low_a, high_a), cv2.inRange(hsv, low_b, high_b))
    if morphology_iters > 0:
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morphology_iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morphology_iters)
    return mask


def find_detections(mask: Any, *, label: str, min_area: float, max_detections: int) -> list[dict[str, Any]]:
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
        detections.append({
            "found": True,
            "label": label,
            "center": {"x": cx, "y": cy},
            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "area": area,
            "area_ratio": area / max(1, image_w * image_h),
            "bbox_fill": area / max(1, w * h),
            "aspect_ratio": w / max(1, h),
        })
    detections.sort(key=lambda item: float(item["area"]), reverse=True)
    return detections[: max(1, int(max_detections))]


def draw_overlay(frame: Any, detections: list[dict[str, Any]], label: str) -> Any:
    overlay = frame.copy()
    for index, detection in enumerate(detections, start=1):
        box = detection["bbox"]
        center = detection["center"]
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cx, cy = center["x"], center["y"]
        color = (22, 163, 74) if index == 1 else (37, 99, 235)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.drawMarker(overlay, (cx, cy), color, cv2.MARKER_CROSS, 18, 2)
        cv2.putText(
            overlay,
            f"{label} {index}: ({cx},{cy}) area={int(detection['area'])}",
            (max(4, x), max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    if not detections:
        cv2.putText(
            overlay,
            f"tracking {label}: no target",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (245, 158, 11),
            2,
            cv2.LINE_AA,
        )
    return overlay


def capture_loop(args: argparse.Namespace, state: SharedState) -> None:
    lower = parse_hsv(args.lower_hsv, (35, 60, 60))
    upper = parse_hsv(args.upper_hsv, (85, 255, 255))
    period = 1.0 / max(0.1, float(args.max_fps))
    while not state.stop.is_set():
        started = time.monotonic()
        try:
            frame = fetch_frame(args.source_url, args.source_timeout)
            if args.max_width and frame.shape[1] > args.max_width:
                scale = args.max_width / float(frame.shape[1])
                frame = cv2.resize(frame, (args.max_width, max(1, int(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            mask = build_mask(
                frame,
                lower_hsv=lower,
                upper_hsv=upper,
                blur=args.blur,
                morphology_iters=args.morphology_iters,
            )
            detections = find_detections(
                mask,
                label=args.label,
                min_area=float(args.min_area),
                max_detections=int(args.max_detections),
            )
            overlay = draw_overlay(frame, detections, args.label)
            ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
            if not ok:
                raise RuntimeError("OpenCV JPEG encode failed")
            detection = detections[0] if detections else {"found": False, "label": args.label}
            report = (
                f"tracking {args.label}: found {len(detections)} candidate(s)"
                if detections
                else f"tracking {args.label}: no candidate above area {args.min_area}"
            )
            with state.lock:
                state.jpeg = encoded.tobytes()
                state.detection = {
                    "ok": True,
                    "found": bool(detections),
                    "detection": detection,
                    "detections": detections,
                    "lower_hsv": lower,
                    "upper_hsv": upper,
                    "report": report,
                    "updated_at": time.time(),
                }
        except Exception as exc:  # noqa: BLE001
            with state.lock:
                state.detection = {
                    "ok": False,
                    "found": False,
                    "detection": {"found": False, "label": args.label},
                    "detections": [],
                    "report": f"CV2 stream FAILED: {type(exc).__name__}: {exc}",
                    "updated_at": time.time(),
                }
        elapsed = time.monotonic() - started
        state.stop.wait(max(0.01, period - elapsed))


def make_handler(state: SharedState, *, max_fps: float):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BlacknodeCV2ColorStream/0.1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/health.json"):
                self._send_json({"ok": True, **state.detection_snapshot()})
                return
            if self.path.startswith("/detection.json"):
                self._send_json(state.detection_snapshot())
                return
            if self.path.startswith("/snapshot.jpg"):
                jpeg = state.jpeg_snapshot()
                if not jpeg:
                    self.send_error(503, "no frame yet")
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
                    jpeg = state.jpeg_snapshot()
                    if not jpeg:
                        time.sleep(0.05)
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    time.sleep(1.0 / max(0.1, float(max_fps)))
                return
            self.send_error(404, "not found")

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--label", default="cube")
    parser.add_argument("--lower-hsv", default="35,60,60")
    parser.add_argument("--upper-hsv", default="85,255,255")
    parser.add_argument("--min-area", type=float, default=300)
    parser.add_argument("--max-detections", type=int, default=3)
    parser.add_argument("--blur", type=int, default=5)
    parser.add_argument("--morphology-iters", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-fps", type=float, default=10.0)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--source-timeout", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    shared = SharedState()
    signal.signal(signal.SIGTERM, lambda _sig, _frame: shared.stop.set())
    thread = threading.Thread(target=capture_loop, args=(args, shared), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(shared, max_fps=args.max_fps))
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        shared.stop.set()
