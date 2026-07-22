"""blacknode-perception — camera capability over ROS 2 (adapter contracts).

All tests run without ROS: the ros2 runtime helpers are monkeypatched, so the
topic-type resolution, stream lifecycle, and USB bridging logic are exercised
pure. Every node must return a structured report instead of raising.
"""
import json
from pathlib import Path

import pytest

import blacknode  # noqa: F401  triggers package discovery
from blacknode.node import _NODE_REGISTRY
from blacknode.packages import _import_nodes_module, _tag_new_package_nodes
from blacknode.workflow import validate_workflow

_ADAPTER = Path(__file__).resolve().parents[1] / "components" / "camera" / "adapters" / "ros2"
_ADAPTER_NODES = _ADAPTER / "nodes"
_before = dict(_NODE_REGISTRY)
_import_nodes_module("blacknode.pkg.blacknode_perception.camera.adapters.ros2", _ADAPTER_NODES)
_tag_new_package_nodes(_before, "blacknode-perception", _ADAPTER_NODES, "camera", "ros2")

from blacknode.pkg.blacknode_perception.camera.adapters.ros2 import camera_stream as cam
from blacknode.pkg.blacknode_ros2 import ros2_runtime as rt

NEW_NODES = ["ROS2ImageStream", "ROS2USBCamera", "ROS2WebVideoStream"]


def test_new_nodes_registered_with_category_and_package():
    for name in NEW_NODES:
        assert name in _NODE_REGISTRY, name
        assert _NODE_REGISTRY[name]._bn_category == "Perception"
        assert _NODE_REGISTRY[name]._bn_package == "blacknode-perception"


def test_no_backend_is_structured_error(monkeypatch):
    monkeypatch.setattr(rt, "detect_backend", lambda refresh=False: {"backend": "none", "detail": "x"})

    result = _NODE_REGISTRY["ROS2ImageStream"]({"topic": "/camera/image_raw", "message_type": "raw"})

    assert result["preview"] == ""
    assert result["streaming"] is False
    assert "FAILED" in result["report"]


# --- ROS2ImageStream --------------------------------------------------------------

def test_image_stream_starts_with_auto_raw_topic(monkeypatch):
    calls = {}

    def fake_run(args, timeout=15.0):
        assert args == ["topic", "type", "/camera/image_raw"]
        return {"ok": True, "backend": "native", "stdout": "sensor_msgs/msg/Image\n", "stderr": ""}

    def fake_start(**kwargs):
        calls.update(kwargs)
        return {
            "ok": True,
            "backend": "native",
            "stream_url": "http://127.0.0.1:9010/stream.mjpg",
            "snapshot_url": "http://127.0.0.1:9010/snapshot.jpg",
            "health_url": "http://127.0.0.1:9010/health.json",
            "port": 9010,
        }

    monkeypatch.setattr(rt, "run_ros2", fake_run)
    monkeypatch.setattr(rt, "start_image_stream", fake_start)
    result = _NODE_REGISTRY["ROS2ImageStream"]({
        "topic": "/camera/image_raw",
        "message_type": "auto",
        "stream_id": "cam",
        "max_fps": 12.0,
        "max_width": 800,
    })
    assert result["preview"] == "http://127.0.0.1:9010/stream.mjpg"
    assert result["streaming"] is True
    assert result["stream_url"] == result["preview"]
    assert calls["message_type"] == "raw"
    assert calls["topic"] == "/camera/image_raw"
    assert calls["stream_id"] == "cam"
    assert calls["max_fps"] == 12.0
    assert calls["max_width"] == 800


def test_image_stream_auto_detects_compressed_topic(monkeypatch):
    monkeypatch.setattr(rt, "run_ros2", lambda args, timeout=15.0: {
        "ok": True,
        "backend": "native",
        "stdout": "sensor_msgs/msg/CompressedImage\n",
        "stderr": "",
    })
    monkeypatch.setattr(rt, "start_image_stream", lambda **kwargs: {
        "ok": True,
        "backend": "native",
        "stream_url": "http://127.0.0.1:9011/stream.mjpg",
        "snapshot_url": "http://127.0.0.1:9011/snapshot.jpg",
    })
    result = _NODE_REGISTRY["ROS2ImageStream"]({"topic": "/camera/compressed", "message_type": "auto"})
    assert result["preview"].endswith("/stream.mjpg")
    assert result["streaming"] is True
    assert "compressed" in result["report"]


def test_image_stream_run_once_returns_a_single_frame(monkeypatch):
    monkeypatch.setattr(rt, "run_ros2", lambda args, timeout=15.0: {
        "ok": True, "backend": "native", "stdout": "sensor_msgs/msg/Image\n", "stderr": "",
    })
    monkeypatch.setattr(rt, "start_image_stream", lambda **kwargs: pytest.fail("run-once must not start a stream"))
    captured = {}

    def fake_capture(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "backend": "native",
            "image": "data:image/jpeg;base64,/9j/2Q==",
            "metadata": {"width": 640, "height": 480},
        }

    monkeypatch.setattr(rt, "capture_image_snapshot", fake_capture)

    result = _NODE_REGISTRY["ROS2ImageStream"]({
        "topic": "/camera/image_raw",
        "message_type": "auto",
        "__run_mode__": "once",
    })

    assert result["streaming"] is False
    assert result["preview"].startswith("data:image/jpeg;base64,")
    assert result["stream_url"] == ""
    assert captured["message_type"] == "raw"
    assert "captured one 640x480" in result["report"]
    assert "Go Live" in result["report"]


def test_image_stream_live_mode_still_starts_a_stream(monkeypatch):
    monkeypatch.setattr(rt, "run_ros2", lambda args, timeout=15.0: {
        "ok": True, "backend": "native", "stdout": "sensor_msgs/msg/Image\n", "stderr": "",
    })
    monkeypatch.setattr(rt, "capture_image_snapshot", lambda **kwargs: pytest.fail("live mode must not snapshot"))
    monkeypatch.setattr(rt, "start_image_stream", lambda **kwargs: {
        "ok": True,
        "backend": "native",
        "stream_url": "http://127.0.0.1:9012/stream.mjpg",
        "snapshot_url": "http://127.0.0.1:9012/snapshot.jpg",
    })

    result = _NODE_REGISTRY["ROS2ImageStream"]({
        "topic": "/camera/image_raw",
        "message_type": "auto",
        "__run_mode__": "live",
    })

    assert result["streaming"] is True
    assert result["preview"] == "http://127.0.0.1:9012/stream.mjpg"
    assert "LIVE STREAM running" in result["report"]


def test_image_stream_stop_calls_runtime(monkeypatch):
    captured = {}

    def fake_stop(stream_id=""):
        captured["stream_id"] = stream_id
        return {"ok": True, "stopped": 1}

    monkeypatch.setattr(rt, "stop_image_stream", fake_stop)
    result = _NODE_REGISTRY["ROS2ImageStream"]({"action": "stop", "stream_id": "cam"})
    assert captured["stream_id"] == "cam"
    assert result["preview"] == ""
    assert result["streaming"] is False
    assert "stopped 1" in result["report"]


def test_image_stream_explains_a_topic_with_no_publisher(monkeypatch):
    def fake_run(args, timeout=15.0):
        if args[:2] == ["topic", "type"]:
            return {"ok": False, "backend": "native", "stdout": "", "stderr": "", "error": "exited with code 1"}
        return {"ok": True, "backend": "native", "stdout": "/parameter_events\n", "stderr": ""}

    monkeypatch.setattr(rt, "run_ros2", fake_run)

    result = _NODE_REGISTRY["ROS2ImageStream"]({"topic": "/camera/image_raw", "message_type": "auto"})

    assert result["streaming"] is False
    assert "no active publisher" in result["report"]


# --- ROS2USBCamera ----------------------------------------------------------------

def test_usb_camera_binds_all_interfaces_so_the_ros_container_can_reach_it(monkeypatch):
    # The capture must bind 0.0.0.0: on the Docker backend 127.0.0.1 would be
    # the container itself and the bridge would never see a frame.
    seen = {}

    def fake_camera(ctx):
        seen.update(ctx)
        return {"streaming": True, "label": "USB Cam", "stream_url": "http://127.0.0.1:5000/stream.mjpg"}

    monkeypatch.setitem(_NODE_REGISTRY, "Camera", fake_camera)
    monkeypatch.setattr(cam, "_NODE_REGISTRY", _NODE_REGISTRY)
    monkeypatch.setattr(rt, "start_host_camera_publisher", lambda **k: {"ok": True, "backend": "docker"})

    def fake_run(args, timeout=15.0):
        stdout = "sensor_msgs/msg/Image\n" if args[:2] == ["topic", "type"] else "/camera/image_raw\n"
        return {"ok": True, "backend": "docker", "stdout": stdout, "stderr": ""}

    monkeypatch.setattr(rt, "run_ros2", fake_run)
    monkeypatch.setattr(rt, "start_image_stream", lambda **k: {
        "ok": True, "backend": "docker", "stream_url": "http://127.0.0.1:39000/stream.mjpg",
        "snapshot_url": "", "health_url": "",
    })

    result = _NODE_REGISTRY["ROS2USBCamera"]({"action": "start", "selection": 0})

    assert seen["host"] == "0.0.0.0"
    assert result["streaming"] is True
    assert result["camera"] == "USB Cam"
    # the picture must come back out of ROS, not straight from the capture
    assert result["preview"] == "http://127.0.0.1:39000/stream.mjpg"


def test_usb_camera_explains_a_camera_that_will_not_open(monkeypatch):
    monkeypatch.setitem(_NODE_REGISTRY, "Camera", lambda ctx: {
        "streaming": False, "label": "USB Cam", "report": "camera '0' did not produce a frame",
    })
    monkeypatch.setattr(cam, "_NODE_REGISTRY", _NODE_REGISTRY)
    monkeypatch.setattr(rt, "start_host_camera_publisher", lambda **k: pytest.fail("must not bridge a dead camera"))

    result = _NODE_REGISTRY["ROS2USBCamera"]({"action": "start", "selection": 0})

    assert result["streaming"] is False
    assert "already in use" in result["report"]
    assert "selection" in result["report"]


# --- ROS2WebVideoStream -----------------------------------------------------------

def test_web_video_stream_refuses_the_placeholder_host():
    result = _NODE_REGISTRY["ROS2WebVideoStream"]({"host": "ROBOT_IP", "topic": "/camera/image_raw"})

    assert result["streaming"] is False
    assert result["preview"] == ""
    assert "robot's IP address" in result["report"]


def test_web_video_stream_builds_url_and_reports_live(monkeypatch):
    seen = {}

    def fake_probe(url, timeout):
        seen["url"] = url
        return True, "multipart/x-mixed-replace"

    monkeypatch.setattr(rt, "probe_web_video", fake_probe)

    result = _NODE_REGISTRY["ROS2WebVideoStream"]({
        "host": "192.168.1.50",
        "port": 8080,
        "topic": "/depth_cam/rgb0/image_raw",
        "quality": 70,
    })

    assert result["streaming"] is True
    assert seen["url"].startswith("http://192.168.1.50:8080/stream?")
    assert "topic=/depth_cam/rgb0/image_raw" in seen["url"]
    assert "quality=70" in seen["url"]
    assert result["preview"] == seen["url"]
    assert "LIVE robot camera" in result["report"]


def test_web_video_stream_explains_an_unreachable_robot(monkeypatch):
    monkeypatch.setattr(rt, "probe_web_video", lambda url, timeout: (False, "cannot reach the robot"))

    result = _NODE_REGISTRY["ROS2WebVideoStream"]({"host": "192.168.1.50", "topic": "/camera/image_raw"})

    assert result["streaming"] is False
    assert result["preview"] == ""
    assert "cannot reach the robot" in result["report"]
    assert "8080" in result["report"]


def test_templates_validate():
    for path in sorted((_ADAPTER / "templates").glob("*.json")):
        report = validate_workflow(json.loads(path.read_text(encoding="utf-8")))
        assert report.ok, f"{path.name}: {report.to_dict()}"
