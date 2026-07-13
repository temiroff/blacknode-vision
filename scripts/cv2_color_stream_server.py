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
CONFIG_FIELDS = {
    "source_url",
    "object_color",
    "use_reasoning_color",
    # Legacy fields accepted so older helpers/workflows can still patch a running stream.
    "tracking_mode",
    "label",
    "lower_hsv",
    "upper_hsv",
    "manual_color",
    "target_text",
    "reasoning_state_url",
    "fallback_color",
    "target_update_seconds",
    "show_follow_guides",
    "follow_target_x",
    "follow_deadband",
    "min_area",
    "max_detections",
    "blur",
    "morphology_iters",
    "max_fps",
    "max_width",
    "jpeg_quality",
    "source_timeout",
    "hsv_override",
}


class SharedState:
    def __init__(self, config: dict[str, Any]) -> None:
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
        self.config = dict(config)
        self.config_version = 0
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

    def config_snapshot(self) -> tuple[dict[str, Any], int]:
        with self.lock:
            return dict(self.config), self.config_version

    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        clean = {key: value for key, value in patch.items() if key in CONFIG_FIELDS}
        if not clean:
            config, version = self.config_snapshot()
            return {"ok": True, "updated": [], "ignored": sorted(patch), "version": version, "config": config}
        if "tracking_mode" in clean and "use_reasoning_color" not in clean:
            mode = normalize_tracking_mode(clean["tracking_mode"])
            if mode == "reasoning":
                clean["use_reasoning_color"] = True
            elif mode == "manual_color":
                clean["use_reasoning_color"] = False
        with self.lock:
            if ("lower_hsv" in clean or "upper_hsv" in clean) and "hsv_override" not in clean:
                clean["hsv_override"] = True
            self.config.update(clean)
            self.config_version += 1
            config = dict(self.config)
            version = self.config_version
        return {"ok": True, "updated": sorted(clean), "ignored": sorted(set(patch) - set(clean)), "version": version, "config": config}


def initial_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_url": args.source_url,
        "object_color": args.object_color or args.manual_color or args.fallback_color or "#22c55e",
        "use_reasoning_color": bool_value(args.use_reasoning_color, True),
        "tracking_mode": args.tracking_mode,
        "label": args.label,
        "lower_hsv": args.lower_hsv,
        "upper_hsv": args.upper_hsv,
        "manual_color": args.manual_color,
        "target_text": args.target_text,
        "reasoning_state_url": args.reasoning_state_url,
        "fallback_color": args.fallback_color,
        "target_update_seconds": args.target_update_seconds,
        "show_follow_guides": bool_value(args.show_follow_guides, True),
        "follow_target_x": args.follow_target_x,
        "follow_deadband": args.follow_deadband,
        "min_area": args.min_area,
        "max_detections": args.max_detections,
        "blur": args.blur,
        "morphology_iters": args.morphology_iters,
        "max_fps": args.max_fps,
        "max_width": args.max_width,
        "jpeg_quality": args.jpeg_quality,
        "source_timeout": args.source_timeout,
        "hsv_override": False,
    }


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


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _find_color_in_words(value: Any) -> str:
    words = set(normalize_words(value).split())
    for alias, color in COLOR_ALIASES.items():
        if alias in words:
            return color
    return ""


def find_color(value: Any) -> str:
    text = str(value or "")
    # Reasoning answers are asked to structure a "Target: <color> <object>"
    # line, but the "Scene:" line describing the surroundings often mentions
    # other colors too (e.g. "green and red cubes" when the target is the
    # green one). Prefer whatever color is named right after "target:" so
    # that scene-description colors don't win just by matching earlier in
    # COLOR_ALIASES' fixed iteration order.
    target_match = re.search(r"target\s*:\s*([^\n.]*)", text, flags=re.IGNORECASE)
    if target_match:
        color = _find_color_in_words(target_match.group(1))
        if color:
            return color
    return _find_color_in_words(text)


def parse_hex_color(value: Any) -> tuple[int, int, int] | None:
    text = str(value or "").strip()
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", text)
    if not match:
        return None
    raw = match.group(1)
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def normalize_tracking_mode(value: Any) -> str:
    mode = str(value or "reasoning").strip().lower().replace("-", "_").replace(" ", "_")
    if mode in {"manual", "manual_color", "color"}:
        return "manual_color"
    if mode in {"manual_hsv", "hsv"}:
        return "manual_hsv"
    return "reasoning"


def color_range_from_value(value: Any) -> tuple[str, tuple[int, int, int], tuple[int, int, int]] | None:
    named = find_color(value)
    if named:
        lower, upper = HSV_COLOR_RANGES[named]
        return named, lower, upper

    rgb = parse_hex_color(value)
    if rgb is None:
        return None
    r, g, b = rgb
    sample = np.uint8([[[b, g, r]]])
    h, s, v = (int(part) for part in cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[0][0])
    if v < 80:
        lower, upper = HSV_COLOR_RANGES["black"]
    elif s < 50 and v > 170:
        lower, upper = HSV_COLOR_RANGES["white"]
    else:
        lower = ((h - 10) % 180, max(30, s - 80), max(40, v - 90))
        upper = ((h + 10) % 180, 255, 255)
    return f"#{r:02x}{g:02x}{b:02x}", lower, upper


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
    allow_reasoning = bool_value(getattr(args, "use_reasoning_color", True), True)
    legacy_mode = normalize_tracking_mode(getattr(args, "tracking_mode", "reasoning"))
    tracking_mode = "reasoning" if allow_reasoning else "manual"
    target_text = args.target_text.strip()
    object_color = str(
        getattr(args, "object_color", "")
        or getattr(args, "manual_color", "")
        or getattr(args, "fallback_color", "")
        or "#22c55e"
    ).strip()
    target_color = find_color(target_text)
    reasoning_answer = ""
    reasoning_report = ""
    reasoning_color = ""

    if legacy_mode == "manual_hsv":
        return {
            "color": "",
            "label": default_label,
            "lower_hsv": default_lower,
            "upper_hsv": default_upper,
            "source": "manual_hsv",
            "tracking_mode": "manual_hsv",
            "target_text": target_text,
            "source_text": "",
            "reasoning_state_url": args.reasoning_state_url.strip(),
            "reasoning_answer": "",
            "reasoning_report": "",
        }

    if not allow_reasoning:
        resolved = color_range_from_value(object_color)
        if resolved:
            color, lower, upper = resolved
        else:
            color, lower, upper = "", default_lower, default_upper
        label_word = find_object_label(target_text, object_color, fallback=default_label)
        label = f"{color} {label_word}".strip() if color and not color.startswith("#") else label_word
        return {
            "color": color,
            "label": label,
            "lower_hsv": lower,
            "upper_hsv": upper,
            "source": "object_color",
            "tracking_mode": tracking_mode,
            "target_text": target_text,
            "source_text": object_color,
            "reasoning_state_url": args.reasoning_state_url.strip(),
            "reasoning_answer": "",
            "reasoning_report": "",
        }

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
        resolved = color_range_from_value(object_color)
        if resolved:
            color, lower, upper = resolved
            source = "object_color"
            source_text = object_color
        else:
            label = default_label
            return {
                "color": "",
                "label": label,
                "lower_hsv": default_lower,
                "upper_hsv": default_upper,
                "source": "default_hsv",
                "tracking_mode": tracking_mode,
                "target_text": target_text,
                "source_text": target_text,
                "reasoning_state_url": args.reasoning_state_url.strip(),
                "reasoning_answer": reasoning_answer,
                "reasoning_report": reasoning_report,
            }

    if source not in {"object_color"}:
        lower, upper = HSV_COLOR_RANGES[color]
    label_word = find_object_label(target_text, reasoning_answer, object_color, fallback=default_label)
    label = f"{color} {label_word}".strip() if color and not color.startswith("#") else label_word
    return {
        "color": color,
        "label": label,
        "lower_hsv": lower,
        "upper_hsv": upper,
        "source": source,
        "tracking_mode": tracking_mode,
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
            "frame_width": int(image_w),
            "frame_height": int(image_h),
            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "area": area,
            "area_ratio": area / max(1, image_w * image_h),
            "bbox_fill": area / max(1, w * h),
            "aspect_ratio": w / max(1, h),
        })
    detections.sort(key=lambda item: float(item["area"]), reverse=True)
    return detections[: max(1, int(max_detections))]


def format_hsv(value: tuple[int, int, int]) -> str:
    return ",".join(str(int(part)) for part in value)


def tracking_status_text(target: dict[str, Any], lower: tuple[int, int, int], upper: tuple[int, int, int]) -> str:
    mode = str(target.get("tracking_mode") or "reasoning")
    source = str(target.get("source") or "configured")
    color = str(target.get("color") or "custom")
    return f"mode={mode} source={source} color={color} hsv={format_hsv(lower)}-{format_hsv(upper)}"


def draw_overlay(
    frame: Any,
    detections: list[dict[str, Any]],
    label: str,
    status: str = "",
    *,
    show_follow_guides: bool = True,
    follow_target_x: float = 0.4,
    follow_deadband: float = 0.12,
) -> tuple[Any, dict[str, Any]]:
    overlay = frame.copy()
    image_h, image_w = overlay.shape[:2]
    target_x = max(0.0, min(1.0, float(follow_target_x)))
    deadband = max(0.0, min(0.5, float(follow_deadband)))
    left_fraction = max(0.0, target_x - deadband)
    right_fraction = min(1.0, target_x + deadband)
    left_boundary = int(image_w * left_fraction)
    target_line = int(image_w * target_x)
    right_boundary = int(image_w * right_fraction)
    center_x = detections[0]["center"]["x"] if detections else None
    if center_x is None:
        command = "NO TARGET"
        zone = "NONE"
    elif center_x < left_boundary:
        command = "MOVE LEFT"
        zone = "LEFT"
    elif center_x > right_boundary:
        command = "MOVE RIGHT"
        zone = "RIGHT"
    else:
        command = "HOLD"
        zone = "CENTER"
    guide = {
        "visible": bool(show_follow_guides),
        "target_x": target_x,
        "deadband": deadband,
        "left_x": left_boundary,
        "target_x_pixels": target_line,
        "right_x": right_boundary,
        "zone": zone,
        "command": command,
    }
    if show_follow_guides:
        cv2.line(overlay, (left_boundary, 38), (left_boundary, image_h), (245, 158, 11), 1)
        cv2.line(overlay, (target_line, 38), (target_line, image_h), (229, 237, 247), 2)
        cv2.line(overlay, (right_boundary, 38), (right_boundary, image_h), (245, 158, 11), 1)
        cv2.rectangle(overlay, (0, max(0, image_h - 36)), (image_w, image_h), (15, 23, 42), -1)
        command_color = (34, 197, 94) if command == "HOLD" else (245, 158, 11)
        cv2.putText(
            overlay,
            f"COMMAND: {command}  |  LEFT < {left_fraction:.0%}  HOLD {left_fraction:.0%}-{right_fraction:.0%}  RIGHT > {right_fraction:.0%}",
            (12, image_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            command_color,
            1,
            cv2.LINE_AA,
        )
    if status:
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 38), (15, 23, 42), -1)
        cv2.putText(
            overlay,
            status[:110],
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (229, 237, 247),
            1,
            cv2.LINE_AA,
        )
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
            (12, 64 if status else 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (245, 158, 11),
            2,
            cv2.LINE_AA,
        )
    return overlay, guide


def capture_loop(args: argparse.Namespace, state: SharedState) -> None:
    config, config_version = state.config_snapshot()
    cfg = argparse.Namespace(**config)
    default_lower = parse_hsv(str(cfg.lower_hsv), (35, 60, 60))
    default_upper = parse_hsv(str(cfg.upper_hsv), (85, 255, 255))
    target = resolve_target(
        cfg,
        default_label=str(cfg.label),
        default_lower=default_lower,
        default_upper=default_upper,
        previous=None,
    )
    last_target_update = 0.0
    while not state.stop.is_set():
        started = time.monotonic()
        try:
            config, next_config_version = state.config_snapshot()
            cfg = argparse.Namespace(**config)
            config_changed = next_config_version != config_version
            if config_changed:
                config_version = next_config_version
            default_lower = parse_hsv(str(cfg.lower_hsv), (35, 60, 60))
            default_upper = parse_hsv(str(cfg.upper_hsv), (85, 255, 255))
            update_period = max(0.25, float(cfg.target_update_seconds))
            if config_changed or time.monotonic() - last_target_update >= update_period:
                target = resolve_target(
                    cfg,
                    default_label=str(cfg.label),
                    default_lower=default_lower,
                    default_upper=default_upper,
                    previous=None if config_changed else target,
                )
                last_target_update = time.monotonic()
            if bool(getattr(cfg, "hsv_override", False)):
                target = {
                    **target,
                    "lower_hsv": default_lower,
                    "upper_hsv": default_upper,
                    "source": "manual_hsv",
                    "tracking_mode": "manual_hsv",
                }
            label = str(target.get("label") or cfg.label)
            lower = tuple(target.get("lower_hsv") or default_lower)
            upper = tuple(target.get("upper_hsv") or default_upper)
            frame = fetch_frame(str(cfg.source_url), float(cfg.source_timeout))
            max_width = max(0, int(cfg.max_width))
            if max_width and frame.shape[1] > max_width:
                scale = max_width / float(frame.shape[1])
                frame = cv2.resize(frame, (max_width, max(1, int(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            mask = build_mask(
                frame,
                lower_hsv=lower,
                upper_hsv=upper,
                blur=int(cfg.blur),
                morphology_iters=int(cfg.morphology_iters),
            )
            detections = find_detections(
                mask,
                label=label,
                min_area=float(cfg.min_area),
                max_detections=int(cfg.max_detections),
            )
            status = tracking_status_text(target, lower, upper)
            overlay, follow_guide = draw_overlay(
                frame,
                detections,
                label,
                status=status,
                show_follow_guides=bool_value(cfg.show_follow_guides, True),
                follow_target_x=float(cfg.follow_target_x),
                follow_deadband=float(cfg.follow_deadband),
            )
            jpeg_quality = max(1, min(100, int(cfg.jpeg_quality)))
            ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not ok:
                raise RuntimeError("OpenCV JPEG encode failed")
            mask_ok, encoded_mask = cv2.imencode(".png", mask)
            if not mask_ok:
                raise RuntimeError("OpenCV mask PNG encode failed")
            mask_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_jpeg_ok, encoded_mask_jpeg = cv2.imencode(
                ".jpg",
                mask_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if not mask_jpeg_ok:
                raise RuntimeError("OpenCV mask JPEG encode failed")
            detection = detections[0] if detections else {"found": False, "label": label}
            report = (
                f"tracking {label} ({status}): found {len(detections)} candidate(s)"
                if detections
                else f"tracking {label} ({status}): no candidate above area {cfg.min_area}"
            )
            with state.lock:
                state.jpeg = encoded.tobytes()
                state.mask_png = encoded_mask.tobytes()
                state.mask_jpeg = encoded_mask_jpeg.tobytes()
                state.detection = {
                    "ok": True,
                    "found": bool(detections),
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "follow_guide": follow_guide,
                    "detection": detection,
                    "detections": detections,
                    "lower_hsv": lower,
                    "upper_hsv": upper,
                    "tracking_mode": str(target.get("tracking_mode") or "reasoning"),
                    "active_source": str(target.get("source") or ""),
                    "active_color": str(target.get("color") or ""),
                    "target": target,
                    "report": report,
                    "updated_at": time.time(),
                }
        except Exception as exc:  # noqa: BLE001
            with state.lock:
                state.detection = {
                    "ok": False,
                    "found": False,
                    "detection": {"found": False, "label": str(target.get("label") or getattr(cfg, "label", args.label))},
                    "detections": [],
                    "target": target,
                    "report": f"CV2 stream FAILED: {type(exc).__name__}: {exc}",
                    "updated_at": time.time(),
                }
        elapsed = time.monotonic() - started
        period = 1.0 / max(0.1, float(getattr(cfg, "max_fps", args.max_fps)))
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
            if self.path.startswith("/config.json"):
                config, version = state.config_snapshot()
                self._send_json({"ok": True, "version": version, "config": config})
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
                    config, _version = state.config_snapshot()
                    time.sleep(1.0 / max(0.1, float(config.get("max_fps", max_fps))))
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
                    config, _version = state.config_snapshot()
                    time.sleep(1.0 / max(0.1, float(config.get("max_fps", max_fps))))
                return
            self.send_error(404, "not found")

        def do_PATCH(self) -> None:  # noqa: N802
            self._handle_config_update()

        def do_POST(self) -> None:  # noqa: N802
            self._handle_config_update()

        def _handle_config_update(self) -> None:
            if not self.path.startswith("/config.json"):
                self.send_error(404, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                body = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(body.decode("utf-8") or "{}")
                if not isinstance(payload, dict):
                    raise ValueError("config update must be a JSON object")
            except Exception as exc:  # noqa: BLE001
                self.send_error(400, f"invalid config update: {type(exc).__name__}: {exc}")
                return
            self._send_json(state.update_config(payload))

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
    parser.add_argument("--object-color", default="#22c55e")
    parser.add_argument("--use-reasoning-color", default="true")
    # Legacy arguments remain accepted for older callers and saved stream configs.
    parser.add_argument("--tracking-mode", choices=["reasoning", "manual_color", "manual_hsv"], default="reasoning")
    parser.add_argument("--label", default="cube")
    parser.add_argument("--lower-hsv", default="35,60,60")
    parser.add_argument("--upper-hsv", default="85,255,255")
    parser.add_argument("--manual-color", default="#22c55e")
    parser.add_argument("--target-text", default="")
    parser.add_argument("--reasoning-state-url", default="")
    parser.add_argument("--fallback-color", default="")
    parser.add_argument("--target-update-seconds", type=float, default=2.0)
    parser.add_argument("--show-follow-guides", default="true")
    parser.add_argument("--follow-target-x", type=float, default=0.4)
    parser.add_argument("--follow-deadband", type=float, default=0.12)
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
    shared = SharedState(initial_config(args))
    signal.signal(signal.SIGTERM, lambda _sig, _frame: shared.stop.set())
    thread = threading.Thread(target=capture_loop, args=(args, shared), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(shared, max_fps=args.max_fps))
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        shared.stop.set()
