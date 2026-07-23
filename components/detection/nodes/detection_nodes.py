"""Object-detection nodes: consume a camera frame stream, emit annotated video.

DetectionStream takes any frame_stream (a Camera, a filter, a ROS image
stream) and runs an OpenCV detector on it, so detection composes with the rest
of the stream graph rather than owning the camera itself. It emits its own
frame_stream so the annotated video can be published or recorded downstream.
"""
from __future__ import annotations

from blacknode import streams as bn_streams
from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Image, Int, List, Text, node

from . import detection_runtime as rt

_CATEGORY = "Detection"


@node(
    name="DetectionStream",
    live=True,
    category=_CATEGORY,
    description=(
        "Run OpenCV object detection on a wired camera stream and show the boxes live. "
        "Wire a Camera (or any frame_stream) into frame_stream. Detectors are built in "
        "(no model download): 'motion' or 'object'."
    ),
    primary_inputs=["frame_stream"],
    primary_outputs=["preview", "detection_stream", "report"],
    inputs={
        "trigger": AnyPort,
        "action": Enum(["start", "stop"], default="start"),
        "frame_stream": Dict(default={}),
        "source_url": Text(default=""),
        "mode": Enum(["motion", "object"], default="motion"),
        "stream_id": Text(default="detection"),
        "max_fps": Float(default=10.0),
        "max_width": Int(default=960),
        "jpeg_quality": Int(default=80),
    },
    outputs={
        "preview": Image,
        "streaming": Bool,
        "stream_url": Text,
        "snapshot_url": Text,
        "detection_stream": Dict,
        "detections": List,
        "found": Bool,
        "stream_id": Text,
        "report": Text,
        "frame_stream": Dict,
    },
)
def detection_stream(ctx: dict) -> dict:
    stream_id = str(ctx.get("stream_id") or "detection").strip() or "detection"
    blank = {"preview": "", "streaming": False, "stream_url": "", "snapshot_url": "",
             "detection_stream": {}, "detections": [], "found": False, "stream_id": stream_id}

    if str(ctx.get("action") or "start").strip().lower() == "stop":
        result = rt.stop_detection_stream(stream_id)
        return {**blank, "report": f"stopped {result.get('stopped', 0)} detection stream(s)"}

    # Detect on a per-frame snapshot, not the continuous MJPEG: reading the
    # multipart stream with urlopen().read() blocks forever. The snapshot
    # endpoint returns the latest frame on each fetch, which is what to poll.
    frame_stream = ctx.get("frame_stream") if isinstance(ctx.get("frame_stream"), dict) else {}
    stream_url = bn_streams.source_url(frame_stream, str(ctx.get("source_url") or ""))
    source_url = str(frame_stream.get("snapshot_url") or "")
    if not source_url and stream_url.endswith("/stream.mjpg"):
        source_url = stream_url[: -len("/stream.mjpg")] + "/snapshot.jpg"
    if not source_url:
        source_url = stream_url
    if not ctx.get("frame_stream") and not source_url:
        return {**blank, "report": (
            "detection FAILED: nothing wired to 'frame_stream'.\n"
            "CHECK: connect a Camera node's frame_stream output to this input and cook it "
            "first, so there is live video to detect on."
        )}
    if not source_url:
        return {**blank, "report": (
            "detection FAILED: the wired stream carries no video URL. Cook the upstream "
            "camera so it is streaming before detecting on it."
        )}

    result = rt.start_detection_stream(
        stream_id=stream_id,
        source_url=source_url,
        mode=str(ctx.get("mode") or "motion"),
        node_id=str(ctx.get("__node_id__") or ""),
        max_fps=max(0.1, min(30.0, float(ctx.get("max_fps") or 10.0))),
        max_width=max(0, int(ctx.get("max_width") or 960)),
        jpeg_quality=max(1, min(100, int(ctx.get("jpeg_quality") or 80))),
    )
    if not result.get("ok"):
        return {**blank, "report": f"detection FAILED: {result.get('error', 'unknown error')}"}

    stream_url = str(result.get("stream_url") or "")
    snapshot_url = str(result.get("snapshot_url") or "")
    return {
        "preview": stream_url,
        "streaming": True,
        "stream_url": stream_url,
        "snapshot_url": snapshot_url,
        "detection_stream": {"kind": "blacknode.detection-stream", "schema_version": 1,
                             "stream_id": stream_id, "detection_url": result.get("detection_url", "")},
        "detections": [],
        "found": False,
        "stream_id": stream_id,
        "report": f"LIVE detection ({ctx.get('mode') or 'motion'}) on {stream_url}",
    }


@node(
    name="YoloDetection",
    live=True,
    category=_CATEGORY,
    description=(
        "Real object detection with YOLO (the same ultralytics engine the robot uses) on a "
        "wired camera stream. Wire a Camera (or any frame_stream) into frame_stream. The model "
        "auto-downloads on first use; runs on GPU when torch sees CUDA. Needs: pip install ultralytics."
    ),
    primary_inputs=["frame_stream"],
    primary_outputs=["preview", "detection_stream", "report"],
    inputs={
        "trigger": AnyPort,
        "action": Enum(["start", "stop"], default="start"),
        "frame_stream": Dict(default={}),
        "source_url": Text(default=""),
        "model": Text(default="yolov8n.pt"),
        "conf": Float(default=0.35),
        "stream_id": Text(default="yolo"),
        "max_fps": Float(default=10.0),
        "max_width": Int(default=960),
        "jpeg_quality": Int(default=80),
    },
    outputs={
        "preview": Image,
        "streaming": Bool,
        "stream_url": Text,
        "snapshot_url": Text,
        "detection_stream": Dict,
        "detections": List,
        "found": Bool,
        "stream_id": Text,
        "report": Text,
        "frame_stream": Dict,
    },
)
def yolo_detection(ctx: dict) -> dict:
    stream_id = str(ctx.get("stream_id") or "yolo").strip() or "yolo"
    blank = {"preview": "", "streaming": False, "stream_url": "", "snapshot_url": "",
             "detection_stream": {}, "detections": [], "found": False, "stream_id": stream_id}

    if str(ctx.get("action") or "start").strip().lower() == "stop":
        result = rt.stop_detection_stream(stream_id)
        return {**blank, "report": f"stopped {result.get('stopped', 0)} YOLO stream(s)"}

    frame_stream = ctx.get("frame_stream") if isinstance(ctx.get("frame_stream"), dict) else {}
    stream_url = bn_streams.source_url(frame_stream, str(ctx.get("source_url") or ""))
    source_url = str(frame_stream.get("snapshot_url") or "")
    if not source_url and stream_url.endswith("/stream.mjpg"):
        source_url = stream_url[: -len("/stream.mjpg")] + "/snapshot.jpg"
    if not source_url:
        source_url = stream_url
    if not ctx.get("frame_stream") and not source_url:
        return {**blank, "report": (
            "YOLO FAILED: nothing wired to 'frame_stream'.\n"
            "CHECK: connect a Camera node's frame_stream output and cook it first."
        )}
    if not source_url:
        return {**blank, "report": "YOLO FAILED: the wired stream carries no video URL."}

    model = str(ctx.get("model") or "yolov8n.pt").strip() or "yolov8n.pt"
    result = rt.start_detection_stream(
        stream_id=stream_id,
        source_url=source_url,
        mode="yolo",
        model=model,
        conf=max(0.0, min(1.0, float(ctx.get("conf") or 0.35))),
        node_id=str(ctx.get("__node_id__") or ""),
        max_fps=max(0.1, min(30.0, float(ctx.get("max_fps") or 10.0))),
        max_width=max(0, int(ctx.get("max_width") or 960)),
        jpeg_quality=max(1, min(100, int(ctx.get("jpeg_quality") or 80))),
    )
    if not result.get("ok"):
        return {**blank, "report": f"YOLO FAILED: {result.get('error', 'unknown error')}"}

    stream_url = str(result.get("stream_url") or "")
    snapshot_url = str(result.get("snapshot_url") or "")
    return {
        "preview": stream_url,
        "streaming": True,
        "stream_url": stream_url,
        "snapshot_url": snapshot_url,
        "detection_stream": {"kind": "blacknode.detection-stream", "schema_version": 1,
                             "stream_id": stream_id, "detection_url": result.get("detection_url", "")},
        "detections": [],
        "found": False,
        "stream_id": stream_id,
        "report": f"LIVE YOLO ({model}) on {stream_url} — read boxes from detection_stream",
    }
