#!/usr/bin/env python3
"""HTTP MJPEG server that overlays CV2 color-object tracking on a snapshot URL."""
from __future__ import annotations

import argparse
import json
import re
import signal
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np

HSV_COLOR_RANGES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
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
COLOR_ALIASES: dict[str, str] = {
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
OBJECT_WORDS = ("cube", "block", "box", "ball", "bottle", "cup", "marker", "object", "target")


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: bytes = b""
        self.mask_png: bytes = b""
        self.mask_jpeg: bytes = b""
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

    def mask_snapshot(self) -> bytes:
        with self.lock:
            return self.mask_png

    def mask_stream_snapshot(self) -> bytes:
        with self.lock:
            return self.mask_jpeg

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


def normalize_words(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def find_color(value: Any) -> str:
    words = set(normalize_words(value).split())
    for alias, color in COLOR_ALIASES.items():
        if alias in words:
            return color
    return ""


def find_object_label(*values: Any, fallback: str) -> str:
    for value in values:
        words = set(normalize_words(value).split())
        for word in OBJECT_WORDS:
            if word in words:
                return word
    return fallback.strip() or "object"


def fetch_reasoning_answer(state_url: str, timeout: float) -> tuple[str, str]:
    if not state_url:
        return "", ""
    try:
        req = urllib.request.Request(state_url, headers={"User-Agent": "BlacknodeCV2Target/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload.get("answer") or "").strip(), str(payload.get("report") or "").strip()
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}"


def resolve_target(
    args: argparse.Namespace,
    *,
    default_label: str,
    default_lower: tuple[int, int, int],
    default_upper: tuple[int, int, int],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    target_text = args.target_text.strip()
    target_color = find_color(target_text)
    reasoning_answer = ""
    reasoning_report = ""
    reasoning_color = ""
    if not target_color and args.reasoning_state_url.strip():
        reasoning_answer, reasoning_report = fetch_reasoning_answer(args.reasoning_state_url.strip(), timeout=0.5)
        reasoning_color = find_color(reasoning_answer)

    if target_color:
        color = target_color
        source = "target"
        source_text = target_text
    elif reasoning_color:
        color = reasoning_color
        source = "reasoning"
        source_text = reasoning_answer
    elif previous and previous.get("source") == "reasoning":
        kept = dict(previous)
        kept["reasoning_answer"] = reasoning_answer
        kept["reasoning_report"] = reasoning_report
        kept["reasoning_state_url"] = args.reasoning_state_url.strip()
        return kept
    else:
        fallback_color = find_color(args.fallback_color)
        if fallback_color:
            color = fallback_color
            source = "fallback"
            source_text = args.fallback_color
        else:
            label = default_label
            return {
                "color": "",
                "label": label,
                "lower_hsv": default_lower,
                "upper_hsv": default_upper,
                "source": "configured_hsv",
                "target_text": target_text,
                "source_text": target_text,
                "reasoning_state_url": args.reasoning_state_url.strip(),
                "reasoning_answer": reasoning_answer,
                "reasoning_report": reasoning_report,
            }

    lower, upper = HSV_COLOR_RANGES[color]
    label_word = find_object_label(target_text, reasoning_answer, fallback=default_label)
    return {
        "color": color,
        "label": f"{color} {label_word}".strip(),
        "lower_hsv": lower,
        "upper_hsv": upper,
        "source": source,
        "target_text": target_text,
        "source_text": source_text,
        "reasoning_state_url": args.reasoning_state_url.strip(),
        "reasoning_answer": reasoning_answer,
        "reasoning_report": reasoning_report,
    }


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
    default_lower = parse_hsv(args.lower_hsv, (35, 60, 60))
    default_upper = parse_hsv(args.upper_hsv, (85, 255, 255))
    target = resolve_target(
        args,
        default_label=args.label,
        default_lower=default_lower,
        default_upper=default_upper,
        previous=None,
    )
    last_target_update = 0.0
    period = 1.0 / max(0.1, float(args.max_fps))
    while not state.stop.is_set():
        started = time.monotonic()
        try:
            update_period = max(0.25, float(args.target_update_seconds))
            if time.monotonic() - last_target_update >= update_period:
                target = resolve_target(
                    args,
                    default_label=args.label,
                    default_lower=default_lower,
                    default_upper=default_upper,
                    previous=target,
                )
                last_target_update = time.monotonic()
            label = str(target.get("label") or args.label)
            lower = tuple(target.get("lower_hsv") or default_lower)
            upper = tuple(target.get("upper_hsv") or default_upper)
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
                label=label,
                min_area=float(args.min_area),
                max_detections=int(args.max_detections),
            )
            overlay = draw_overlay(frame, detections, label)
            ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
            if not ok:
                raise RuntimeError("OpenCV JPEG encode failed")
            mask_ok, encoded_mask = cv2.imencode(".png", mask)
            if not mask_ok:
                raise RuntimeError("OpenCV mask PNG encode failed")
            mask_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_jpeg_ok, encoded_mask_jpeg = cv2.imencode(
                ".jpg",
                mask_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
            )
            if not mask_jpeg_ok:
                raise RuntimeError("OpenCV mask JPEG encode failed")
            detection = detections[0] if detections else {"found": False, "label": label}
            report = (
                f"tracking {label} from {target.get('source')}: found {len(detections)} candidate(s)"
                if detections
                else f"tracking {label} from {target.get('source')}: no candidate above area {args.min_area}"
            )
            with state.lock:
                state.jpeg = encoded.tobytes()
                state.mask_png = encoded_mask.tobytes()
                state.mask_jpeg = encoded_mask_jpeg.tobytes()
                state.detection = {
                    "ok": True,
                    "found": bool(detections),
                    "detection": detection,
                    "detections": detections,
                    "lower_hsv": lower,
                    "upper_hsv": upper,
                    "target": target,
                    "report": report,
                    "updated_at": time.time(),
                }
        except Exception as exc:  # noqa: BLE001
            with state.lock:
                state.detection = {
                    "ok": False,
                    "found": False,
                    "detection": {"found": False, "label": str(target.get("label") or args.label)},
                    "detections": [],
                    "target": target,
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
            if self.path.startswith("/mask.png"):
                mask_png = state.mask_snapshot()
                if not mask_png:
                    self.send_error(503, "no mask yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(mask_png)))
                self.end_headers()
                self.wfile.write(mask_png)
                return
            if self.path.startswith("/mask.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while not state.stop.is_set():
                    jpeg = state.mask_stream_snapshot()
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
    parser.add_argument("--target-text", default="")
    parser.add_argument("--reasoning-state-url", default="")
    parser.add_argument("--fallback-color", default="")
    parser.add_argument("--target-update-seconds", type=float, default=2.0)
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
