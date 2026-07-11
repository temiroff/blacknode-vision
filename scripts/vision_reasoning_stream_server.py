#!/usr/bin/env python3
"""HTTP MJPEG server that periodically updates a live VLM reasoning dashboard."""
from __future__ import annotations

import argparse
import base64
import json
import signal
import textwrap
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
        self.dashboard_jpeg: bytes = b""
        self.state: dict[str, Any] = {
            "ok": False,
            "answer": "",
            "report": "waiting for first reasoning update",
            "updated_at": 0.0,
        }
        self.stop = threading.Event()

    def dashboard_snapshot(self) -> bytes:
        with self.lock:
            return self.dashboard_jpeg

    def state_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)


def fetch_bytes(url: str, timeout: float) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "BlacknodeVisionReasoning/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        media_type = response.headers.get_content_type() or "image/jpeg"
        return response.read(), media_type


def fetch_frame(url: str, timeout: float) -> tuple[Any, bytes, str]:
    raw, media_type = fetch_bytes(url, timeout)
    data = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("image_url did not return a decodable image")
    return frame, raw, media_type


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    if not url:
        return {}
    raw, _media_type = fetch_bytes(url, timeout)
    return json.loads(raw.decode("utf-8"))


def post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_ollama_text(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, list):
        text = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
    else:
        text = str(content or "").strip()
    return text, message


def build_prompt(args: argparse.Namespace, detection: dict[str, Any]) -> str:
    prompt = args.prompt.strip() or (
        "Describe what you see in the attached camera frame. Then use the CV2 "
        "detection JSON to report target state, confidence, uncertainty, and one safe next action."
    )
    parts = [prompt]
    if detection:
        parts.append("Live CV2 detection JSON:")
        parts.append(json.dumps(detection, indent=2, sort_keys=True))
    parts.append("Return 2-4 short lines: scene, target, confidence/uncertainty, next action.")
    return "\n".join(parts)


def call_ollama(args: argparse.Namespace, image_raw: bytes, media_type: str, detection: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    model = args.model.strip() or "qwen3-vl:4b"
    max_tokens = max(1, min(int(args.max_tokens), 8192))
    if "qwen3" in model.lower() and max_tokens < 4096:
        max_tokens = 4096
    user_message: dict[str, Any] = {
        "role": "user",
        "content": build_prompt(args, detection),
        "images": [base64.b64encode(image_raw).decode("ascii")],
    }
    messages: list[dict[str, Any]] = []
    if args.system.strip():
        messages.append({"role": "system", "content": args.system.strip()})
    messages.append(user_message)
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(args.temperature),
            "num_predict": max_tokens,
        },
    }
    headers = {"Content-Type": "application/json"}
    if args.api_key.strip():
        headers["Authorization"] = f"Bearer {args.api_key.strip()}"
    endpoint = args.endpoint_url.strip().rstrip("/") or "http://127.0.0.1:11434"
    payload = post_json(endpoint + "/api/chat", body, headers, timeout=float(args.request_timeout))
    if payload.get("error"):
        return "", f"ollama/{model} FAILED: {payload['error']}", payload
    text, message = extract_ollama_text(payload)
    retried = False
    if (
        not text
        and "qwen3" in model.lower()
        and str(payload.get("done_reason") or "").lower() == "length"
        and max_tokens < 8192
    ):
        retry_body = {**body, "options": {**body["options"], "num_predict": 8192}}
        payload = post_json(endpoint + "/api/chat", retry_body, headers, timeout=max(float(args.request_timeout), 240.0))
        retried = True
        if payload.get("error"):
            return "", f"ollama/{model} FAILED: {payload['error']}", payload
        text, message = extract_ollama_text(payload)
    if not text:
        keys = ", ".join(sorted(str(key) for key in message)) or "none"
        thinking = "; thinking field was present but is hidden" if message.get("thinking") else ""
        return "", f"ollama/{model} returned empty final content{thinking}; message keys: {keys}", payload
    retry_note = " after Qwen3 length retry" if retried else ""
    return text, f"ollama/{model} OK{retry_note}; media={media_type}", payload


def wrap_lines(text: str, width: int, max_lines: int) -> list[str]:
    lines: list[str] = []
    for raw in str(text or "").splitlines() or [""]:
        lines.extend(textwrap.wrap(raw.strip(), width=width, break_long_words=True) or [""])
    if len(lines) <= max_lines:
        return lines
    kept = lines[:max_lines]
    kept[-1] = kept[-1][: max(0, width - 3)].rstrip() + "..."
    return kept


def draw_text_block(canvas: Any, lines: list[str], *, x: int, y: int, color: tuple[int, int, int], scale: float = 0.55) -> int:
    current_y = y
    for line in lines:
        cv2.putText(canvas, line, (x, current_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        current_y += int(30 * scale) + 10
    return current_y


def render_dashboard(frame: Any | None, answer: str, report: str, detection: dict[str, Any], *, title: str) -> bytes:
    canvas = np.full((720, 1120, 3), (23, 32, 51), dtype=np.uint8)
    cv2.rectangle(canvas, (24, 24), (1096, 696), (17, 24, 39), -1)
    cv2.rectangle(canvas, (24, 24), (1096, 696), (38, 52, 73), 1)
    ready = bool(answer) and "FAILED" not in report.upper()
    status_color = (88, 166, 92) if ready else (11, 145, 245)
    cv2.circle(canvas, (58, 70), 10, status_color, -1)
    cv2.putText(canvas, "LIVE REASONING" if ready else "REASONING UPDATE", (82, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, title[:54], (36, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (247, 250, 252), 2, cv2.LINE_AA)

    x0, y0, w, h = 36, 144, 420, 315
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (15, 23, 42), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (51, 65, 85), 1)
    if frame is not None:
        fh, fw = frame.shape[:2]
        scale = min(w / max(1, fw), h / max(1, fh))
        resized = cv2.resize(frame, (max(1, int(fw * scale)), max(1, int(fh * scale))), interpolation=cv2.INTER_AREA)
        rh, rw = resized.shape[:2]
        ox = x0 + (w - rw) // 2
        oy = y0 + (h - rh) // 2
        canvas[oy:oy + rh, ox:ox + rw] = resized
    else:
        cv2.putText(canvas, "No frame yet", (x0 + 95, y0 + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (148, 163, 184), 2, cv2.LINE_AA)

    target = detection.get("detection") if isinstance(detection.get("detection"), dict) else {}
    found = bool(detection.get("found") or target.get("found"))
    target_text = "target: found" if found else "target: not found"
    if target.get("center"):
        center = target["center"]
        target_text += f" at ({center.get('x', 0)}, {center.get('y', 0)})"
    cv2.putText(canvas, target_text[:64], (36, 498), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (203, 213, 225), 1, cv2.LINE_AA)

    cv2.putText(canvas, "VISIBLE REASONING", (490, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (148, 163, 184), 2, cv2.LINE_AA)
    answer_text = answer or "Waiting for the first VLM update..."
    y = draw_text_block(canvas, wrap_lines(answer_text, 62, 13), x=490, y=182, color=(229, 237, 247), scale=0.56)
    y = max(y + 18, 518)
    cv2.putText(canvas, "MODEL REPORT", (490, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 2, cv2.LINE_AA)
    draw_text_block(canvas, wrap_lines(report or "No report yet.", 68, 4), x=490, y=y + 34, color=(203, 213, 225), scale=0.48)

    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        raise RuntimeError("dashboard JPEG encode failed")
    return encoded.tobytes()


def update_dashboard(state: SharedState, frame: Any | None, answer: str, report: str, detection: dict[str, Any], title: str, raw: dict[str, Any] | None = None) -> None:
    jpeg = render_dashboard(frame, answer, report, detection, title=title)
    with state.lock:
        state.dashboard_jpeg = jpeg
        state.state = {
            "ok": bool(answer) and "FAILED" not in report.upper(),
            "answer": answer,
            "report": report,
            "detection": detection,
            "raw": raw or {},
            "updated_at": time.time(),
        }


def reasoning_loop(args: argparse.Namespace, state: SharedState) -> None:
    interval = max(1.0, float(args.interval_seconds))
    answer = ""
    detection: dict[str, Any] = {}
    frame = None
    while not state.stop.is_set():
        started = time.monotonic()
        raw_payload: dict[str, Any] = {}
        try:
            frame, image_raw, media_type = fetch_frame(args.image_url, float(args.source_timeout))
            if int(args.max_width) > 0 and frame.shape[1] > int(args.max_width):
                scale = int(args.max_width) / float(frame.shape[1])
                frame = cv2.resize(frame, (int(args.max_width), max(1, int(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            detection = fetch_json(args.detection_url, float(args.source_timeout)) if args.detection_url else {}
            update_dashboard(state, frame, answer, "VLM update in progress...", detection, args.title)
            if args.provider != "ollama":
                raise RuntimeError("VisionReasoningStream currently supports provider=ollama")
            answer, report, raw_payload = call_ollama(args, image_raw, media_type, detection)
            update_dashboard(state, frame, answer, report, detection, args.title, raw_payload)
        except Exception as exc:  # noqa: BLE001
            update_dashboard(state, frame, answer, f"reasoning stream FAILED: {type(exc).__name__}: {exc}", detection, args.title, raw_payload)
        elapsed = time.monotonic() - started
        state.stop.wait(max(0.05, interval - elapsed))


def make_handler(state: SharedState, *, max_fps: float):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BlacknodeVisionReasoningStream/0.1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/state.json"):
                self._send_json(state.state_snapshot())
                return
            if self.path.startswith("/dashboard.jpg"):
                jpeg = state.dashboard_snapshot()
                if not jpeg:
                    self.send_error(503, "no dashboard yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path.startswith("/dashboard.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while not state.stop.is_set():
                    jpeg = state.dashboard_snapshot()
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
    parser.add_argument("--image-url", required=True)
    parser.add_argument("--detection-url", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--system", default="You are a robot vision assistant. Describe only what is visible, then give a concise next action.")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen3-vl:4b")
    parser.add_argument("--endpoint-url", default="http://127.0.0.1:11434")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--interval-seconds", type=float, default=8.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-fps", type=float, default=2.0)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--source-timeout", type=float, default=4.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--title", default="Blacknode Live Vision Reasoning")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    shared = SharedState()
    signal.signal(signal.SIGTERM, lambda _sig, _frame: shared.stop.set())
    update_dashboard(shared, None, "", "starting live reasoning stream", {}, args.title)
    thread = threading.Thread(target=reasoning_loop, args=(args, shared), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(shared, max_fps=args.max_fps))
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        shared.stop.set()
