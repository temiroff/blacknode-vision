"""Camera capability over ROS 2: image-topic streams, USB camera bridging, and
robot web_video_server viewing.

These adapt the perception camera capability to a ROS 2 graph. The transport
plumbing (running ``ros2``, bridging MJPEG in/out of the graph) is provided by
``blacknode-ros2/core``; every node here returns a structured report instead of
raising, so workflows stay usable on machines without ROS.
"""
from __future__ import annotations

import time
import urllib.parse

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Enum, Float, Image, Int, Text, _NODE_REGISTRY, node


class _LazyRos2Runtime:
    """Delay ROS 2 imports so dependency discovery can load in any folder order."""

    def __getattr__(self, name: str):
        from blacknode.pkg.blacknode_ros2 import ros2_runtime

        return getattr(ros2_runtime, name)


rt = _LazyRos2Runtime()

_CATEGORY = "Perception"


def _resolve_image_message_type(topic: str, requested: str) -> tuple[str, str]:
    value = requested.strip().lower()
    if value in {"raw", "compressed"}:
        return value, ""
    result = rt.run_ros2(["topic", "type", topic], timeout=10)
    if not result.get("ok"):
        # `ros2 topic type` only resolves topics with an active publisher/
        # subscriber right now, and fails with an opaque "exited with code 1"
        # (often no stderr) when nothing is publishing yet. Check topic list
        # membership so the failure explains what to fix instead of just the
        # bare exit code.
        listing = rt.run_ros2(["topic", "list"], timeout=10)
        known_topics = {
            line.strip() for line in listing.get("stdout", "").splitlines() if line.strip()
        } if listing.get("ok") else set()
        if listing.get("ok") and topic not in known_topics:
            return "", (
                f"{topic} has no active publisher right now. Start a camera driver that "
                f"publishes to {topic} (a Camera node, or the 'Camera Livestream' "
                f"template) before starting the stream, or set 'topic' to a topic ros2 "
                f"topic list already shows."
            )
        return "", result.get("error", "could not discover topic type")
    types = [line.strip() for line in result.get("stdout", "").splitlines() if line.strip()]
    if any("sensor_msgs/msg/CompressedImage" in line for line in types):
        return "compressed", ""
    if any("sensor_msgs/msg/Image" in line for line in types):
        return "raw", ""
    return "", f"{topic} is not a sensor_msgs Image topic (types: {', '.join(types) or 'none'})"


@node(
    name="ROS2ImageStream",
    live=True,
    category=_CATEGORY,
    description="Live camera feed for a raw or compressed ROS 2 image topic. Go Live streams continuous MJPEG; a plain one-shot Run captures a single frame instead.",
    inputs={
        "trigger": AnyPort,
        "action": Enum(["start", "stop"], default="start"),
        "stream_id": Text(default="camera"),
        "topic": Text(default="/camera/image_raw"),
        "message_type": Enum(["auto", "raw", "compressed"], default="auto"),
        "host": Text(default="127.0.0.1"),
        "port": Int(default=0),
        "max_fps": Float(default=10.0),
        "max_width": Int(default=960),
        "jpeg_quality": Int(default=80),
    },
    outputs={
        "preview": Image,
        "streaming": Bool,
        "stream_url": Text,
        "snapshot_url": Text,
        "stream_id": Text,
        "report": Text,
    },
)
def ros2_image_stream(ctx: dict) -> dict:
    stream_id = str(ctx.get("stream_id") or "camera").strip() or "camera"
    action = str(ctx.get("action") or "start").strip().lower()
    if action == "stop":
        result = rt.stop_image_stream(stream_id)
        return {
            "preview": "",
            "streaming": False,
            "stream_url": "",
            "snapshot_url": "",
            "stream_id": stream_id,
            "report": f"stopped {result.get('stopped', 0)} image stream(s)",
        }

    topic = str(ctx.get("topic") or "/camera/image_raw").strip()
    message_type, error = _resolve_image_message_type(topic, str(ctx.get("message_type") or "auto"))
    if error:
        return {
            "preview": "",
            "streaming": False,
            "stream_url": "",
            "snapshot_url": "",
            "stream_id": stream_id,
            "report": f"image stream FAILED: {error}",
        }

    host = str(ctx.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    port = max(0, int(ctx.get("port") or 0))
    max_fps = max(0.1, min(60.0, float(ctx.get("max_fps") or 10.0)))
    max_width = max(0, int(ctx.get("max_width") or 960))
    jpeg_quality = max(1, min(100, int(ctx.get("jpeg_quality") or 80)))

    if ctx.get("__run_mode__") == "once":
        # A one-shot Run has nothing to keep watching a persistent MJPEG URL,
        # and leaving a background stream server running after it would leak.
        # Return a single real frame instead; Go Live starts the live stream.
        shot = rt.capture_image_snapshot(
            topic=topic,
            message_type=message_type,
            timeout=max(1.0, 15.0),
            output_format="jpeg",
            jpeg_quality=jpeg_quality,
        )
        if not shot.get("ok"):
            return {
                "preview": "",
                "streaming": False,
                "stream_url": "",
                "snapshot_url": "",
                "stream_id": stream_id,
                "report": f"image frame FAILED: {shot.get('error', 'unknown error')}",
            }
        metadata = dict(shot.get("metadata") or {})
        return {
            "preview": str(shot.get("image") or ""),
            "streaming": False,
            "stream_url": "",
            "snapshot_url": "",
            "stream_id": stream_id,
            "report": (
                f"captured one {metadata.get('width', '?')}x{metadata.get('height', '?')} "
                f"{message_type} frame from {topic} — press Go Live for a continuous stream"
            ),
        }

    result = rt.start_image_stream(
        stream_id=stream_id,
        topic=topic,
        message_type=message_type,
        host=host,
        port=port,
        max_fps=max_fps,
        max_width=max_width,
        jpeg_quality=jpeg_quality,
    )
    if not result.get("ok"):
        return {
            "preview": "",
            "streaming": False,
            "stream_url": "",
            "snapshot_url": "",
            "stream_id": stream_id,
            "report": f"image stream FAILED: {result.get('error', 'unknown error')}",
        }
    stream_url = str(result["stream_url"])
    snapshot_url = str(result["snapshot_url"])
    report = (
        f"LIVE STREAM running on {stream_url} from {topic} "
        f"({message_type}, {max_fps:g} FPS max, width {max_width or 'source'})"
    )
    return {
        "preview": stream_url,
        "streaming": True,
        "stream_url": stream_url,
        "snapshot_url": snapshot_url,
        "stream_id": stream_id,
        "report": report,
    }


@node(
    name="ROS2USBCamera",
    live=True,
    category=_CATEGORY,
    description=(
        "Plug-and-play USB camera on ROS 2: opens a USB camera, publishes it as a real "
        "sensor_msgs/Image topic, and shows the picture read back from ROS. No IP, no wiring, "
        "no camera driver to install."
    ),
    inputs={
        "trigger": AnyPort,
        "action": Enum(["start", "stop"], default="start"),
        "selection": Int(default=0),
        "topic": Text(default="/camera/image_raw"),
        "max_fps": Float(default=15.0),
        "max_width": Int(default=640),
        "jpeg_quality": Int(default=80),
        "wait_seconds": Float(default=25.0),
    },
    outputs={
        "preview": Image,
        "streaming": Bool,
        "topic": Text,
        "camera": Text,
        "stream_url": Text,
        "report": Text,
    },
)
def ros2_usb_camera(ctx: dict) -> dict:
    """Own the whole USB-camera-to-ROS path so a template needs no setup.

    Docker cannot open a USB camera (the helper container has no /dev/video*),
    so the capture happens on this machine and is bridged into the ROS graph.
    The preview deliberately comes back out of ROS rather than from the capture
    directly, so what is shown is proof the topic really carries the camera.
    """
    topic = str(ctx.get("topic") or "/camera/image_raw").strip() or "/camera/image_raw"
    stream_id = "ros2_usb_camera"
    blank = {"preview": "", "streaming": False, "topic": topic, "camera": "", "stream_url": ""}

    camera_node = _NODE_REGISTRY.get("Camera")
    if camera_node is None:
        return {**blank, "report": (
            "USB camera FAILED: the Camera capture node is not installed. "
            "Enable the blacknode-perception camera component, then run this again."
        )}

    if str(ctx.get("action") or "start").strip().lower() == "stop":
        camera_node({"action": "stop", "stream_id": stream_id})
        rt.stop_host_camera_publisher(stream_id)
        rt.stop_image_stream(stream_id)
        return {**blank, "report": "stopped the USB camera, its ROS publisher, and the preview"}

    # 1. open the USB camera here and serve it where the ROS side can read it.
    #    0.0.0.0 matters: on the Docker backend 127.0.0.1 would be the container.
    capture = camera_node({
        "action": "start",
        "selection": int(ctx.get("selection") or 0),
        "stream_id": stream_id,
        "host": "0.0.0.0",
        "port": 0,
        "max_fps": max(0.1, float(ctx.get("max_fps") or 15.0)),
        "max_width": max(0, int(ctx.get("max_width") or 640)),
        "jpeg_quality": max(1, min(100, int(ctx.get("jpeg_quality") or 80))),
    })
    label = str(capture.get("label") or "")
    source_url = str(capture.get("stream_url") or "")
    if not capture.get("streaming") or not source_url:
        return {**blank, "camera": label, "report": (
            f"USB camera FAILED: {capture.get('report') or 'the camera did not start'}\n"
            "CHECK: is the camera plugged in, not already in use by another app "
            "(Teams/Zoom/OBS, or another Blacknode camera node), and allowed under "
            "Windows camera privacy settings? Try a different 'selection' number."
        )}

    # 2. bridge it into the ROS graph as a real image topic
    published = rt.start_host_camera_publisher(
        run_id=stream_id,
        source_url=source_url,
        topic=topic,
        frame_id="camera_frame",
        max_fps=max(0.1, float(ctx.get("max_fps") or 15.0)),
    )
    if not published.get("ok"):
        camera_node({"action": "stop", "stream_id": stream_id})
        return {**blank, "camera": label, "report": (
            f"USB camera FAILED to reach ROS: {published.get('error', 'unknown error')}"
        )}

    wait_seconds = max(0.0, float(ctx.get("wait_seconds") or 25.0))
    deadline = time.time() + wait_seconds
    discovered = False
    while time.time() < deadline:
        check = rt.run_ros2(["topic", "list"], timeout=10)
        topics = {line.strip().split()[0] for line in check.get("stdout", "").splitlines() if line.strip()}
        if check.get("ok") and topic in topics:
            discovered = True
            break
        time.sleep(1)
    if not discovered:
        return {**blank, "camera": label, "report": (
            f"USB camera is running, but {topic} did not appear on the ROS graph within "
            f"{wait_seconds:g}s. The camera itself is fine — this is the ROS side."
        )}

    # 3. read it back *out of ROS* so the picture proves the topic works
    message_type, error = _resolve_image_message_type(topic, "auto")
    if not error:
        shown = rt.start_image_stream(
            stream_id=stream_id,
            topic=topic,
            message_type=message_type,
            host="127.0.0.1",
            port=0,
            max_fps=max(0.1, float(ctx.get("max_fps") or 15.0)),
            max_width=max(0, int(ctx.get("max_width") or 640)),
            jpeg_quality=max(1, min(100, int(ctx.get("jpeg_quality") or 80))),
        )
        if shown.get("ok"):
            url = str(shown["stream_url"])
            return {
                "preview": url,
                "streaming": True,
                "topic": topic,
                "camera": label,
                "stream_url": url,
                "report": (
                    f"LIVE: '{label}' publishing to {topic} on ROS, shown from the ROS topic "
                    f"via the {published['backend']} backend"
                ),
            }
        error = str(shown.get("error", "could not read the topic back"))

    # ROS has the topic but reading it back failed; still report the live topic.
    return {
        "preview": "",
        "streaming": True,
        "topic": topic,
        "camera": label,
        "stream_url": "",
        "report": (
            f"'{label}' is publishing to {topic} on ROS, but the preview could not be "
            f"read back: {error}"
        ),
    }


def _web_video_url(host: str, port: int, topic: str, quality: int, width: int, height: int) -> str:
    params = [f"topic={urllib.parse.quote(topic, safe='/')}", "type=mjpeg"]
    if quality > 0:
        params.append(f"quality={quality}")
    if width > 0:
        params.append(f"width={width}")
    if height > 0:
        params.append(f"height={height}")
    return f"http://{host}:{port}/stream?{'&'.join(params)}"


@node(
    name="ROS2WebVideoStream",
    live=True,
    category=_CATEGORY,
    description=(
        "Watch a camera topic published by a robot running web_video_server. The robot "
        "serves MJPEG over HTTP, so this needs no local ROS graph and works even when DDS "
        "discovery cannot reach the robot."
    ),
    inputs={
        "trigger": AnyPort,
        "host": Text(default="ROBOT_IP"),
        "port": Int(default=8080),
        "topic": Text(default="/camera/image_raw"),
        "quality": Int(default=80),
        "width": Int(default=0),
        "height": Int(default=0),
        "timeout": Float(default=10.0),
    },
    outputs={"preview": Image, "streaming": Bool, "stream_url": Text, "report": Text},
)
def ros2_web_video_stream(ctx: dict) -> dict:
    host = str(ctx.get("host") or "").strip()
    port = int(ctx.get("port") or 8080)
    topic = str(ctx.get("topic") or "/camera/image_raw").strip() or "/camera/image_raw"
    timeout = max(1.0, float(ctx.get("timeout") or 10.0))
    blank = {"preview": "", "streaming": False, "stream_url": ""}

    if not host or host == "ROBOT_IP":
        return {
            **blank,
            "report": (
                "robot camera FAILED: set 'host' to your robot's IP address "
                "(the machine running web_video_server), e.g. 192.168.1.50"
            ),
        }

    url = _web_video_url(
        host, port, topic,
        max(0, min(100, int(ctx.get("quality") or 0))),
        max(0, int(ctx.get("width") or 0)),
        max(0, int(ctx.get("height") or 0)),
    )

    # Probe before handing the URL to the canvas: a broken <img> is never
    # retried, so a silent failure would just show an empty node forever.
    ok, detail = rt.probe_web_video(url, timeout)
    if not ok:
        return {
            **blank,
            "report": (
                f"robot camera FAILED: {detail}\n"
                f"tried {url}\n"
                f"CHECK: is the robot at {host} powered and on this network, is web_video_server "
                f"running on port {port}, and does it publish '{topic}'? "
                f"Open http://{host}:{port}/ in a browser to list the robot's camera topics."
            ),
        }

    return {
        "preview": url,
        "streaming": True,
        "stream_url": url,
        "report": f"LIVE robot camera from {topic} on {host}:{port}",
    }
