"""blacknode-vision package contracts."""
import base64
import json
from pathlib import Path

import blacknode  # noqa: F401  triggers package discovery
from blacknode.node import _NODE_REGISTRY
from blacknode.workflow import validate_workflow

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"

EXPECTED_NODES = [
    "VisionFramePrompt",
    "VisionStreamStatus",
    "VisionVLMDescribe",
]


def test_nodes_registered_with_package_and_category():
    for name in EXPECTED_NODES:
        assert name in _NODE_REGISTRY, name
        assert _NODE_REGISTRY[name]._bn_package == "blacknode-vision"
        assert _NODE_REGISTRY[name]._bn_category == "Vision"


def test_templates_validate():
    for path in sorted(TEMPLATE_DIR.glob("*.json")):
        report = validate_workflow(json.loads(path.read_text(encoding="utf-8")))
        assert report.ok, f"{path.name}: {report.to_dict()}"


def test_camera_console_defaults_to_bundled_usb_camera():
    path = TEMPLATE_DIR / "vision-camera-console.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    params = workflow["node_meta"]["camera_run"]["params"]
    assert params["package"] == "blacknode_usb_camera"
    assert params["executable"] == "usb_camera"
    assert "/camera/image_raw" in params["arguments"]


def test_frame_prompt_summarizes_context():
    result = _NODE_REGISTRY["VisionFramePrompt"]({
        "image": "data:image/png;base64,abc",
        "question": "Is the table clear?",
        "context": "bench camera",
        "robot_task": "pick cube",
    })
    assert "Is the table clear?" in result["prompt"]
    assert "pick cube" in result["prompt"]
    assert result["summary"]["has_image"] is True
    assert result["summary"]["image_kind"] == "data-url"


def test_stream_status_ready_dashboard():
    result = _NODE_REGISTRY["VisionStreamStatus"]({
        "camera_topic": "/camera/image_raw",
        "stream_url": "http://127.0.0.1:9000/stream.mjpg",
        "streaming": True,
    })
    assert result["ready"] is True
    assert result["dashboard"].startswith("data:image/svg+xml;base64,")
    assert "LIVE" in result["report"]


def test_stream_status_wraps_long_dashboard_text():
    long_report = (
        "ROS 2 run process running: blacknode_usb_camera usb_camera; "
        "/camera/image_raw is discoverable via native backend with a long status message"
    )
    result = _NODE_REGISTRY["VisionStreamStatus"]({
        "camera_topic": "/camera/image_raw",
        "stream_url": "http://127.0.0.1:12345/stream.mjpg?with=a-long-query-string-that-would-overflow",
        "streaming": True,
        "run_report": long_report,
        "stream_report": long_report,
    })
    svg = base64.b64decode(result["dashboard"].split(",", 1)[1]).decode("utf-8")
    assert "<tspan" in svg
    assert 'height="380"' not in svg
    assert "/camera/image_raw is discoverable" in svg


def test_vlm_describe_requires_image():
    result = _NODE_REGISTRY["VisionVLMDescribe"]({"image": ""})
    assert result["text"] == ""
    assert "FAILED" in result["report"]


def test_vlm_describe_requires_key_for_remote(monkeypatch):
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    result = _NODE_REGISTRY["VisionVLMDescribe"]({
        "image": "data:image/png;base64,abc",
        "endpoint_url": "https://api.openai.com/v1",
        "api_key": "",
    })
    assert result["text"] == ""
    assert "api_key" in result["report"]
