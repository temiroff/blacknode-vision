"""Live object-detection MJPEG server for Blacknode Perception.

Pulls frames from a source MJPEG stream (a Camera's frame_stream), runs an
OpenCV detector, draws boxes, and serves the annotated video plus a detection
JSON - the same shape as the colour tracker, so any frame-stream source can
feed it. Detectors are OpenCV-native (Haar cascades ship with opencv-python),
so nothing is downloaded and it runs anywhere the camera does.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np

_BOX_COLOR = (34, 197, 94)
_TEXT_COLOR = (255, 255, 255)


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: bytes = b""
        self.detection: dict[str, Any] = {
            "ok": False, "found": False, "detections": [],
            "report": "waiting for first frame", "updated_at": 0.0,
        }
        self.stop = threading.Event()

    def jpeg_snapshot(self) -> bytes:
        with self.lock:
            return self.jpeg

    def detection_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.detection)

    def update(self, jpeg: bytes, detection: dict[str, Any]) -> None:
        with self.lock:
            self.jpeg = jpeg
            self.detection = detection


def _fetch_frame(source_url: str, timeout: float) -> Any:
    req = urllib.request.Request(source_url, headers={"User-Agent": "BlacknodeDetection/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("source did not return a decodable image")
    return frame


def _boxes_from_mask(mask: Any, label: str, min_area: int) -> list[dict[str, Any]]:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        out.append({"label": label, "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "area": int(area), "center_x": int(x + w // 2), "center_y": int(y + h // 2)})
    out.sort(key=lambda d: d["area"], reverse=True)
    return out


class _MotionDetector:
    """Moving regions via background subtraction. Detectors are core-OpenCV only
    (cv2 5.0.0 dropped the classic Haar/HOG object detectors) so nothing is
    downloaded and it runs on whatever build ships with the camera."""
    label = "motion"

    def __init__(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        return _boxes_from_mask(self._bg.apply(frame), "motion", 900)


class _ContourDetector:
    """Bright/high-contrast regions via adaptive threshold — a model-free object
    finder that works from the first frame (no warm-up like motion)."""
    label = "object"

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return _boxes_from_mask(mask, "object", 1500)


class _YoloDetector:
    """Real object detection via ultralytics YOLO — the same API the ROSOrin
    robot uses. ultralytics is a heavy optional dependency (pulls torch), so the
    import is deferred here and its absence is reported, not crashed on."""

    def __init__(self, model: str, conf: float) -> None:
        from ultralytics import YOLO  # optional heavy dep; guarded by the caller

        self.label = "yolo"
        self._conf = conf
        self._model = YOLO(model)
        self._names = self._model.names

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        results = self._model(frame, conf=self._conf, verbose=False)
        out: list[dict[str, Any]] = []
        for result in results:
            for box in getattr(result, "boxes", []) or []:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cls = int(box.cls[0])
                name = self._names.get(cls, str(cls)) if isinstance(self._names, dict) else str(cls)
                out.append({"label": name, "confidence": round(float(box.conf[0]), 3),
                            "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                            "area": (x2 - x1) * (y2 - y1),
                            "center_x": (x1 + x2) // 2, "center_y": (y1 + y2) // 2})
        out.sort(key=lambda d: d["confidence"], reverse=True)
        return out


def _make_detector(mode: str, model: str, conf: float):
    mode = str(mode).lower()
    if mode == "yolo":
        return _YoloDetector(model, conf)
    if mode == "object":
        return _ContourDetector()
    return _MotionDetector()


def _annotate(frame: Any, detections: list[dict[str, Any]], mode: str) -> Any:
    overlay = frame.copy()
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), _BOX_COLOR, 2)
        conf = det.get("confidence")
        caption = f"{det['label']} {conf:.2f}" if isinstance(conf, (int, float)) else det["label"]
        cv2.putText(overlay, caption, (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(overlay, f"{mode}: {len(detections)}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _BOX_COLOR, 2, cv2.LINE_AA)
    return overlay


def _capture_loop(args: argparse.Namespace, state: SharedState) -> None:
    try:
        detector = _make_detector(args.mode, args.model, args.conf)
    except ImportError:
        # YOLO needs ultralytics; say so instead of dying silently in a thread.
        state.update(b"", {
            "ok": False, "found": False, "detections": [],
            "report": "YOLO detection needs ultralytics: pip install ultralytics",
            "updated_at": time.time(),
        })
        return
    except Exception as exc:  # noqa: BLE001 - surface any model-load failure
        state.update(b"", {
            "ok": False, "found": False, "detections": [],
            "report": f"detector failed to start: {exc}", "updated_at": time.time(),
        })
        return
    quality = max(1, min(100, int(args.jpeg_quality)))
    interval = 1.0 / max(0.1, float(args.max_fps))
    while not state.stop.is_set():
        started = time.monotonic()
        try:
            frame = _fetch_frame(args.source_url, timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            state.update(state.jpeg_snapshot(), {
                "ok": False, "found": False, "detections": [],
                "report": f"cannot read source: {exc}", "updated_at": time.time(),
            })
            state.stop.wait(0.5)
            continue
        if args.max_width > 0 and frame.shape[1] > args.max_width:
            scale = args.max_width / float(frame.shape[1])
            frame = cv2.resize(frame, (args.max_width, max(1, int(frame.shape[0] * scale))))
        detections = detector.detect(frame)
        annotated = _annotate(frame, detections, detector.label)
        ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            state.update(encoded.tobytes(), {
                "ok": True, "found": bool(detections), "count": len(detections),
                "detections": detections, "mode": detector.label,
                "report": f"{detector.label}: {len(detections)} detected", "updated_at": time.time(),
            })
        state.stop.wait(max(0.0, interval - (time.monotonic() - started)))


def _make_handler(state: SharedState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:  # silence per-request logging
            pass

        def _mjpeg(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while not state.stop.is_set():
                jpeg = state.jpeg_snapshot()
                if jpeg:
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                time.sleep(0.05)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/detection.json"):
                body = json.dumps(state.detection_snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/snapshot.jpg"):
                jpeg = state.jpeg_snapshot()
                self.send_response(200 if jpeg else 503)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path.startswith("/stream.mjpg"):
                self._mjpeg()
                return
            self.send_response(404)
            self.end_headers()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--mode", default="motion")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--max-fps", type=float, default=10.0)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()

    state = SharedState()
    worker = threading.Thread(target=_capture_loop, args=(args, state), daemon=True)
    worker.start()
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(state))
    print(f"detection stream on http://{args.host}:{server.server_address[1]}/stream.mjpg", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()


if __name__ == "__main__":
    main()
