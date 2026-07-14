"""blacknode-vision package contracts."""
import base64
import importlib.util
import json
from pathlib import Path

import blacknode  # noqa: F401  triggers package discovery
from blacknode.pkg.blacknode_vision import cv2_runtime
from blacknode.node import _NODE_REGISTRY
from blacknode.workflow import validate_workflow

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"

EXPECTED_NODES = {
    "CV2CameraStream": "CV2",
    "CV2ColorObjectStream": "CV2",
    "CV2ColorTargetHint": "CV2",
    "CV2ColorObjectTracker": "CV2",
    "CV2HSVMask": "CV2",
    "VisionDetectionPrompt": "Vision",
    "VisionFramePrompt": "Vision",
    "VisionReasoningDashboard": "Vision",
    "VisionReasoningStream": "Vision",
    "VisionStreamStatus": "Vision",
    "VisionVLMDescribe": "Vision",
}


def test_nodes_registered_with_package_and_category():
    for name, category in EXPECTED_NODES.items():
        assert name in _NODE_REGISTRY, name
        assert _NODE_REGISTRY[name]._bn_package == "blacknode-vision"
        assert _NODE_REGISTRY[name]._bn_category == category


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


def test_detection_prompt_summarizes_cv2_output():
    result = _NODE_REGISTRY["VisionDetectionPrompt"]({
        "detection": {
            "found": True,
            "label": "cube",
            "center": {"x": 320, "y": 240},
            "area": 1200.0,
        },
        "detections": [{"label": "cube"}],
        "question": "Should the robot move left or right?",
    })
    assert "CV2 detections" in result["prompt"]
    assert "First describe what is visible" in result["prompt"]
    assert '"x": 320' in result["prompt"]
    assert result["summary"]["found"] is True


def test_cv2_color_target_hint_uses_explicit_target_color():
    result = _NODE_REGISTRY["CV2ColorTargetHint"]({
        "target": "track the red cube",
        "reasoning": "A blue cube is also visible.",
        "fallback_color": "green",
    })
    assert result["found"] is True
    assert result["source"] == "target"
    assert result["color"] == "red"
    assert result["label"] == "red cube"
    assert result["lower_hsv"] == "170,80,60"
    assert result["upper_hsv"] == "10,255,255"


def test_cv2_color_target_hint_uses_reasoning_when_target_is_vague():
    result = _NODE_REGISTRY["CV2ColorTargetHint"]({
        "target": "track the cube",
        "reasoning": "The most visible object is a blue cube near the center.",
        "fallback_color": "green",
    })
    assert result["found"] is True
    assert result["source"] == "reasoning"
    assert result["color"] == "blue"
    assert result["label"] == "blue cube"
    assert result["lower_hsv"] == "100,60,50"
    assert result["upper_hsv"] == "130,255,255"


def test_cv2_color_target_hint_reads_reasoning_state_url(monkeypatch):
    fn = _NODE_REGISTRY["CV2ColorTargetHint"]

    def fake_read_reasoning_state_answer(state_url, wait_seconds):
        assert state_url == "http://127.0.0.1:9200/state.json"
        assert wait_seconds == 2.5
        return "I see a yellow cube on the table.", ""

    monkeypatch.setitem(fn.__globals__, "_read_reasoning_state_answer", fake_read_reasoning_state_answer)
    result = fn({
        "target": "track the cube",
        "reasoning_state_url": "http://127.0.0.1:9200/state.json",
        "reasoning_wait_seconds": 2.5,
        "fallback_color": "green",
    })
    assert result["source"] == "reasoning"
    assert result["color"] == "yellow"
    assert result["metadata"]["reasoning_state_used"] is True


def test_cv2_color_target_hint_falls_back_without_color():
    result = _NODE_REGISTRY["CV2ColorTargetHint"]({
        "target": "track the cube",
        "reasoning": "A cube is visible, but the color is unclear.",
        "fallback_color": "purple",
    })
    assert result["found"] is False
    assert result["source"] == "fallback"
    assert result["color"] == "purple"
    assert result["label"] == "purple cube"


def test_cv2_stream_runtime_pushes_live_config(monkeypatch):
    calls = []

    class Proc:
        def poll(self):
            return None

    def fake_post_json(url, payload, timeout=1.0):
        calls.append({"url": url, "payload": payload, "timeout": timeout})
        return {"ok": True, "updated": sorted(payload), "version": 2}

    monkeypatch.setitem(cv2_runtime._STREAMS, "cube_tracker", {
        "proc": Proc(),
        "config_url": "http://127.0.0.1:9911/config.json",
        "source_url": "http://127.0.0.1:9900/snapshot.jpg",
        "detection_url": "http://127.0.0.1:9911/detection.json",
    })
    monkeypatch.setattr(cv2_runtime, "_post_json", fake_post_json)

    result = cv2_runtime.update_color_stream_config("cube_tracker", {"object_color": "#22c55e"})

    assert result["ok"] is True
    assert result["active"] is True
    assert result["updated"] == ["object_color"]
    assert calls == [{
        "url": "http://127.0.0.1:9911/config.json",
        "payload": {"object_color": "#22c55e"},
        "timeout": 1.0,
    }]


def test_cv2_camera_stream_starts_native_runtime(monkeypatch):
    fn = _NODE_REGISTRY["CV2CameraStream"]
    calls = []

    def fake_start_camera_stream(**kwargs):
        calls.append(kwargs)
        return {
            "ok": True,
            "stream_url": "http://127.0.0.1:9000/stream.mjpg",
            "snapshot_url": "http://127.0.0.1:9000/snapshot.jpg",
            "health_url": "http://127.0.0.1:9000/health.json",
            "health": {"report": "camera 0 streaming via dshow"},
        }

    monkeypatch.setattr(fn.__globals__["cv2_runtime"], "start_camera_stream", fake_start_camera_stream)
    result = fn({"device": "0", "backend": "auto", "width": 640, "height": 480})

    assert result["streaming"] is True
    assert result["preview"] == "http://127.0.0.1:9000/stream.mjpg"
    assert calls[0]["device"] == "0"
    assert calls[0]["backend"] == "auto"
    assert calls[0]["width"] == 640
    assert calls[0]["height"] == 480


def test_cv2_camera_stream_reports_start_failure(monkeypatch):
    fn = _NODE_REGISTRY["CV2CameraStream"]
    monkeypatch.setattr(
        fn.__globals__["cv2_runtime"],
        "start_camera_stream",
        lambda **_kwargs: {"ok": False, "error": "camera busy"},
    )

    result = fn({"device": "0"})

    assert result["streaming"] is False
    assert "camera busy" in result["report"]


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


def test_reasoning_dashboard_includes_image_and_answer():
    result = _NODE_REGISTRY["VisionReasoningDashboard"]({
        "image": "data:image/jpeg;base64,abc",
        "prompt": "Describe what the robot sees.",
        "answer": "Summary: a workbench is visible. Evidence: flat surface and tools. Next action: wait.",
        "report": "VLM describe OK via test-model",
    })
    svg = base64.b64decode(result["dashboard"].split(",", 1)[1]).decode("utf-8")
    assert result["ready"] is True
    assert "VISIBLE REASONING" in svg
    assert "Summary:" in svg
    assert "data:image/jpeg;base64,abc" in svg


def test_reasoning_dashboard_inlines_url_image(monkeypatch):
    fn = _NODE_REGISTRY["VisionReasoningDashboard"]

    def fake_image_data_parts(image):
        assert image == "http://127.0.0.1:9100/snapshot.jpg"
        return "image/jpeg", "abc123", "url"

    monkeypatch.setitem(fn.__globals__, "_image_data_parts", fake_image_data_parts)
    result = fn({
        "image": "http://127.0.0.1:9100/snapshot.jpg",
        "prompt": "Describe what the robot sees.",
        "answer": "A cube is visible on the table.",
        "report": "VLM describe OK via test-model",
    })
    svg = base64.b64decode(result["dashboard"].split(",", 1)[1]).decode("utf-8")
    assert 'href="data:image/jpeg;base64,abc123"' in svg
    assert "http://127.0.0.1:9100/snapshot.jpg" not in svg
    assert result["summary"]["image_embedded"] is True


def test_vlm_describe_ollama_text_only(monkeypatch):
    calls = []

    def fake_post_json(url, body, headers, timeout=90.0):
        calls.append({"url": url, "body": body, "headers": headers, "timeout": timeout})
        return {"message": {"content": "move slightly left"}}

    fn = _NODE_REGISTRY["VisionVLMDescribe"]
    monkeypatch.setitem(fn.__globals__, "_post_json", fake_post_json)
    result = fn({
        "image": "",
        "question": "Detection center x is 420. What next?",
        "provider": "ollama",
        "model": "qwen2.5vl:7b",
        "endpoint_url": "http://127.0.0.1:11434",
        "allow_text_only": True,
    })
    assert result["text"] == "move slightly left"
    assert result["report"] == "VLM describe OK via ollama/qwen2.5vl:7b"
    assert calls[0]["url"] == "http://127.0.0.1:11434/api/chat"
    assert calls[0]["body"]["stream"] is False
    assert "images" not in calls[0]["body"]["messages"][-1]


def test_vlm_describe_ollama_empty_content_reports_failure(monkeypatch):
    calls = []

    def fake_post_json(url, body, headers, timeout=90.0):
        calls.append({"url": url, "body": body, "headers": headers, "timeout": timeout})
        return {"message": {"content": "", "thinking": "internal reasoning is hidden"}}

    fn = _NODE_REGISTRY["VisionVLMDescribe"]
    monkeypatch.setitem(fn.__globals__, "_post_json", fake_post_json)
    result = fn({
        "image": "",
        "question": "What next?",
        "provider": "ollama",
        "model": "qwen3-vl:4b",
        "endpoint_url": "http://127.0.0.1:11434",
        "allow_text_only": True,
    })
    assert result["text"] == ""
    assert "empty final content" in result["report"]
    assert "thinking field was present but is hidden" in result["report"]
    assert "internal reasoning" not in result["report"]
    assert calls[0]["body"]["options"]["num_predict"] == 4096


def test_vlm_describe_ollama_retries_qwen3_length_stop(monkeypatch):
    calls = []

    def fake_post_json(url, body, headers, timeout=90.0):
        calls.append({"url": url, "body": body, "headers": headers, "timeout": timeout})
        if len(calls) == 1:
            return {"message": {"content": "", "thinking": "long hidden reasoning"}, "done_reason": "length"}
        return {"message": {"content": "Cube centered at (320, 240)."}, "done_reason": "stop"}

    fn = _NODE_REGISTRY["VisionVLMDescribe"]
    monkeypatch.setitem(fn.__globals__, "_post_json", fake_post_json)
    result = fn({
        "image": "",
        "question": "Where is the cube?",
        "provider": "ollama",
        "model": "qwen3-vl:4b",
        "endpoint_url": "http://127.0.0.1:11434",
        "allow_text_only": True,
        "max_tokens": 512,
    })
    assert result["text"] == "Cube centered at (320, 240)."
    assert "length retry" in result["report"]
    assert [call["body"]["options"]["num_predict"] for call in calls] == [4096, 8192]


def test_vlm_describe_anthropic_image(monkeypatch):
    calls = []

    def fake_post_json(url, body, headers, timeout=90.0):
        calls.append({"url": url, "body": body, "headers": headers, "timeout": timeout})
        return {"content": [{"type": "text", "text": "A cube is visible."}]}

    fn = _NODE_REGISTRY["VisionVLMDescribe"]
    monkeypatch.setitem(fn.__globals__, "_post_json", fake_post_json)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    result = fn({
        "image": "data:image/png;base64,abc",
        "question": "What do you see?",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "endpoint_url": "https://api.anthropic.com/v1",
    })
    assert result["text"] == "A cube is visible."
    assert calls[0]["url"] == "https://api.anthropic.com/v1/messages"
    assert calls[0]["headers"]["x-api-key"] == "test-anthropic-key"
    source = calls[0]["body"]["messages"][0]["content"][0]["source"]
    assert source == {"type": "base64", "media_type": "image/png", "data": "abc"}


def test_cv2_tracker_reports_missing_or_detects_green_cube():
    fn = _NODE_REGISTRY["CV2ColorObjectTracker"]
    if fn.__globals__["cv2"] is None:
        result = fn({"image": "data:image/png;base64,abc"})
        assert result["found"] is False
        assert "OpenCV is not installed" in result["report"]
        return

    cv2 = fn.__globals__["cv2"]
    np = fn.__globals__["np"]
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[30:80, 60:110] = (0, 255, 0)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    source = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    result = fn({
        "image": source,
        "label": "cube",
        "lower_hsv": "35,60,60",
        "upper_hsv": "85,255,255",
        "min_area": 100,
    })
    assert result["found"] is True
    assert 75 <= result["center_x"] <= 95
    assert 45 <= result["center_y"] <= 65
    assert result["overlay"].startswith("data:image/jpeg;base64,")


def test_cv2_color_object_stream_starts_runtime(monkeypatch):
    fn = _NODE_REGISTRY["CV2ColorObjectStream"]
    if fn.__globals__["cv2"] is None:
        result = fn({"source_url": "http://127.0.0.1:9000/snapshot.jpg"})
        assert result["streaming"] is False
        assert "OpenCV is not installed" in result["report"]
        return

    calls = []

    def fake_start_color_stream(**kwargs):
        calls.append(kwargs)
        return {
            "ok": True,
            "stream_url": "http://127.0.0.1:9100/stream.mjpg",
            "snapshot_url": "http://127.0.0.1:9100/snapshot.jpg",
            "mask_stream_url": "http://127.0.0.1:9100/mask.mjpg",
            "mask_url": "http://127.0.0.1:9100/mask.png",
            "detection_url": "http://127.0.0.1:9100/detection.json",
            "detection": {
                "found": True,
                "detection": {"found": True, "label": "cube", "center": {"x": 40, "y": 20}},
                "detections": [{"label": "cube"}],
                "report": "tracking cube: found 1 candidate(s)",
            },
        }

    monkeypatch.setattr(fn.__globals__["cv2_runtime"], "start_color_stream", fake_start_color_stream)
    result = fn({
        "stream_id": "cube_tracker",
        "source_url": "http://127.0.0.1:9000/snapshot.jpg",
        "object_color": "#ef4444",
        "use_reasoning_color": False,
        "target": "track the red cube",
        "reasoning_state_url": "http://127.0.0.1:9200/state.json",
        "target_update_seconds": 2.0,
        "label": "cube",
    })
    assert result["streaming"] is True
    assert result["preview"] == "http://127.0.0.1:9100/stream.mjpg"
    assert result["mask"] == "http://127.0.0.1:9100/mask.mjpg"
    assert result["detection_stream"] == {
        "kind": "blacknode.latest-value-stream",
        "stream_id": "cube_tracker",
        "url": "http://127.0.0.1:9100/detection.json",
        "media_type": "application/json",
    }
    assert result["found"] is True
    assert result["detection"]["center"]["x"] == 40
    assert calls[0]["source_url"] == "http://127.0.0.1:9000/snapshot.jpg"
    assert calls[0]["stream_id"] == "cube_tracker"
    assert calls[0]["object_color"] == "#ef4444"
    assert calls[0]["use_reasoning_color"] is False
    assert calls[0]["target_text"] == "track the red cube"
    assert calls[0]["reasoning_state_url"] == "http://127.0.0.1:9200/state.json"
    assert calls[0]["target_update_seconds"] == 2.0
    assert calls[0]["show_follow_guides"] is True
    assert calls[0]["follow_target_x"] == 0.4
    assert calls[0]["follow_deadband"] == 0.12


def test_cv2_follow_guide_reports_visible_direction():
    script = TEMPLATE_DIR.parent / "scripts" / "cv2_color_stream_server.py"
    spec = importlib.util.spec_from_file_location("cv2_color_stream_server_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = module.np.zeros((300, 640, 3), dtype=module.np.uint8)
    detection = {
        "center": {"x": 455, "y": 120},
        "bbox": {"x": 430, "y": 90, "width": 50, "height": 60},
        "area": 2000.0,
    }

    _overlay, guide = module.draw_overlay(
        frame, [detection], "cube", show_follow_guides=True, follow_target_x=0.4, follow_deadband=0.12,
    )

    assert guide["visible"] is True
    assert guide["target_x_pixels"] == 256
    assert guide["zone"] == "RIGHT"
    assert guide["command"] == "MOVE RIGHT"


def test_cv2_color_object_stream_stops_runtime(monkeypatch):
    fn = _NODE_REGISTRY["CV2ColorObjectStream"]

    def fake_stop_color_stream(stream_id):
        assert stream_id == "cube_tracker"
        return {"ok": True, "stopped": 1}

    monkeypatch.setattr(fn.__globals__["cv2_runtime"], "stop_color_stream", fake_stop_color_stream)
    result = fn({"action": "stop", "stream_id": "cube_tracker"})
    assert result["streaming"] is False
    assert "stopped 1 CV2 stream" in result["report"]


def test_vision_reasoning_stream_starts_runtime(monkeypatch):
    fn = _NODE_REGISTRY["VisionReasoningStream"]
    calls = []

    def fake_start_reasoning_stream(**kwargs):
        calls.append(kwargs)
        return {
            "ok": True,
            "stream_url": "http://127.0.0.1:9200/dashboard.mjpg",
            "snapshot_url": "http://127.0.0.1:9200/dashboard.jpg",
            "state_url": "http://127.0.0.1:9200/state.json",
        }

    monkeypatch.setattr(fn.__globals__["cv2_runtime"], "start_reasoning_stream", fake_start_reasoning_stream)
    result = fn({
        "stream_id": "reason",
        "image_url": "http://127.0.0.1:9000/snapshot.jpg",
        "prompt": "Describe what you see.",
        "provider": "ollama",
        "model": "qwen3-vl:4b",
        "max_tokens": 512,
    })
    assert result["streaming"] is True
    assert result["preview"] == "http://127.0.0.1:9200/dashboard.mjpg"
    assert result["state_url"] == "http://127.0.0.1:9200/state.json"
    assert calls[0]["image_url"] == "http://127.0.0.1:9000/snapshot.jpg"
    assert calls[0]["detection_url"] == ""
    assert calls[0]["max_tokens"] == 4096


def test_vision_reasoning_stream_stops_runtime(monkeypatch):
    fn = _NODE_REGISTRY["VisionReasoningStream"]

    def fake_stop_reasoning_stream(stream_id):
        assert stream_id == "reason"
        return {"ok": True, "stopped": 1}

    monkeypatch.setattr(fn.__globals__["cv2_runtime"], "stop_reasoning_stream", fake_stop_reasoning_stream)
    result = fn({"action": "stop", "stream_id": "reason"})
    assert result["streaming"] is False
    assert "stopped 1 reasoning stream" in result["report"]

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


def test_cube_template_uses_live_cv2_stream_and_qwen3():
    path = TEMPLATE_DIR / "vision-cv2-cube-local-reasoning.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert workflow["node_meta"]["stream"]["type"] == "CV2CameraStream"
    assert workflow["node_meta"]["stream"]["params"]["device"] == "0"
    assert workflow["node_meta"]["stream"]["params"]["backend"] == "auto"
    assert "camera_run" not in workflow["node_meta"]
    assert workflow["node_meta"]["cv2_stream"]["type"] == "CV2ColorObjectStream"
    assert workflow["node_meta"]["target_prompt"]["type"] == "Text"
    assert "green cube" not in workflow["node_meta"]["target_prompt"]["params"]["value"].lower()
    assert "target_hint" not in workflow["node_meta"]
    assert "python_export" not in workflow["node_meta"]
    assert workflow["node_meta"]["cv2_stream"]["params"]["object_color"] == "#22c55e"
    assert workflow["node_meta"]["cv2_stream"]["params"]["use_reasoning_color"] is True
    assert "tracking_mode" not in workflow["node_meta"]["cv2_stream"]["params"]
    assert "fallback_color" not in workflow["node_meta"]["cv2_stream"]["params"]
    assert "lower_hsv" not in workflow["node_meta"]["cv2_stream"]["params"]
    assert workflow["node_meta"]["live_reason"]["type"] == "VisionReasoningStream"
    assert workflow["node_meta"]["live_reason"]["params"]["model"] == "qwen3-vl:4b"
    assert workflow["node_meta"]["live_reason"]["params"]["max_tokens"] == 4096
    assert workflow["node_meta"]["live_reason"]["params"]["interval_seconds"] == 3.0
    assert workflow["node_meta"]["live_reason"]["params"]["max_fps"] == 4.0
    assert "Describe what you see" in workflow["node_meta"]["live_reason"]["params"]["prompt"]
    assert "Do not rely on CV2 detections" in workflow["node_meta"]["live_reason"]["params"]["system"]
    edges = {
        (edge["from"], edge["from_port"], edge["to"], edge["to_port"])
        for edge in workflow["edges"]
    }
    assert ("stream", "snapshot_url", "cv2_stream", "source_url") in edges
    assert ("target_prompt", "value", "live_reason", "prompt") in edges
    assert ("live_reason", "state_url", "cv2_stream", "reasoning_state_url") in edges
    assert ("target_prompt", "value", "cv2_stream", "target") not in edges
    assert ("cv2_stream", "preview", "overlay_out", "image") in edges
    assert ("cv2_stream", "mask", "mask_out", "image") in edges
    assert ("stream", "snapshot_url", "live_reason", "image_url") in edges
    assert ("live_reason", "preview", "reason_dashboard_out", "image") in edges
    assert ("cv2_stream", "detection_url", "live_reason", "detection_url") not in edges
    assert workflow["node_meta"]["check"]["type"] == "ROS2RosbridgeServer"
    assert workflow["node_meta"]["preset"]["params"]["transport"] == "rosbridge"
    assert workflow["node_meta"]["joint_state"]["type"] == "ROS2JointState"
    assert workflow["node_meta"]["follow_cube"]["type"] == "ROS2ContinuousFollowDetectionJoint"
    assert workflow["node_meta"]["follow_cube"]["params"]["action"] == "start"
    assert workflow["node_meta"]["follow_cube"]["params"]["loop_hz"] == 2.0
    assert workflow["node_meta"]["follow_cube"]["params"]["joint"] == "shoulder_pan"
    assert workflow["node_meta"]["follow_cube"]["params"]["armed"] is False
    assert workflow["node_meta"]["follow_cube"]["params"]["frame_width"] == 640
    assert workflow["node_meta"]["follow_cube"]["params"]["target_x"] == 0.4
    assert workflow["node_meta"]["follow_cube"]["params"]["deadband"] == 0.12
    assert workflow["node_meta"]["follow_cube"]["params"]["gain"] == 10.0
    assert workflow["node_meta"]["follow_cube"]["params"]["max_step"] == 2.0
    assert workflow["node_meta"]["cv2_stream"]["params"]["show_follow_guides"] is True
    assert workflow["node_meta"]["cv2_stream"]["params"]["follow_target_x"] == 0.4
    assert workflow["node_meta"]["cv2_stream"]["params"]["follow_deadband"] == 0.12
    assert "shoulder_pan_index" not in workflow["node_meta"]
    assert ("joint_state", "names", "shoulder_pan_index", "items") not in edges
    assert ("shoulder_pan_index", "value", "follow_cube", "joint") not in edges


def test_cube_native_ros2_template_keeps_ros_camera_transport():
    path = TEMPLATE_DIR / "vision-cv2-cube-ros2-native-reasoning.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    node_types = {node_id: meta["type"] for node_id, meta in workflow["node_meta"].items()}
    edges = {
        (edge["from"], edge["from_port"], edge["to"], edge["to_port"])
        for edge in workflow["edges"]
    }

    assert node_types["camera_run"] == "ROS2Run"
    assert node_types["stream"] == "ROS2ImageStream"
    assert node_types["follow_cube"] == "ROS2NativeFollowDetectionJoint"
    assert "CV2CameraStream" not in node_types.values()
    assert workflow["node_meta"]["camera_run"]["params"]["package"] == "blacknode_usb_camera"
    assert workflow["node_meta"]["stream"]["params"]["topic"] == "/camera/image_raw"
    assert ("check", "report", "camera_run", "trigger") in edges
    assert ("camera_run", "report", "stream", "trigger") in edges
    assert ("stream", "snapshot_url", "cv2_stream", "source_url") in edges


def test_cube_rosbridge_template_uses_rosbridge_follow_nodes():
    path = TEMPLATE_DIR / "vision-cv2-cube-rosbridge-reasoning.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    node_types = {
        node_id: meta["type"]
        for node_id, meta in workflow["node_meta"].items()
    }
    package_names = {
        package["name"]
        for package in workflow["metadata"]["required_packages"]
    }
    edges = {
        (edge["from"], edge["from_port"], edge["to"], edge["to_port"])
        for edge in workflow["edges"]
    }

    assert {"blacknode-vision", "blacknode-ros2", "blacknode-robot", "blacknode-cuda"} <= package_names
    assert not any(node_type.startswith("ROS2Native") for node_type in node_types.values())
    assert node_types["check"] == "ROS2RosbridgeServer"
    assert node_types["stream"] == "CV2CameraStream"
    assert "camera_run" not in node_types
    assert node_types["preset"] == "RobotDriverPreset"
    assert workflow["node_meta"]["preset"]["params"]["transport"] == "rosbridge"
    assert node_types["robot_bridge"] == "RobotDiscovery"
    assert node_types["joint_state"] == "ROS2JointState"
    assert node_types["follow_cube"] == "ROS2ContinuousFollowDetectionJoint"
    assert workflow["node_meta"]["follow_cube"]["params"]["action"] == "start"
    assert workflow["node_meta"]["follow_cube"]["params"]["loop_hz"] == 10.0
    assert workflow["node_meta"]["follow_cube"]["params"]["armed"] is False
    assert workflow["node_meta"]["follow_cube"]["params"]["host"] == "127.0.0.1"
    assert workflow["node_meta"]["follow_cube"]["params"]["port"] == 9090
    assert workflow["node_meta"]["follow_cube"]["params"]["frame_width"] == 640
    assert workflow["node_meta"]["follow_cube"]["params"]["target_x"] == 0.4
    assert workflow["node_meta"]["follow_cube"]["params"]["deadband"] == 0.03
    assert workflow["node_meta"]["follow_cube"]["params"]["gain"] == 4.0
    assert workflow["node_meta"]["follow_cube"]["params"]["max_step"] == 0.75
    assert workflow["node_meta"]["cv2_stream"]["params"]["show_follow_guides"] is True
    assert workflow["node_meta"]["cv2_stream"]["params"]["follow_target_x"] == 0.4
    assert workflow["node_meta"]["cv2_stream"]["params"]["follow_deadband"] == 0.03
    assert ("stream", "snapshot_url", "cv2_stream", "source_url") in edges
    assert ("cv2_stream", "detection", "follow_cube", "detection") in edges
    assert ("cv2_stream", "detection_stream", "follow_cube", "detection_stream") in edges
    assert ("cv2_stream", "detection_url", "follow_cube", "detection_url") not in edges
    assert ("check", "report", "robot_bridge", "trigger") in edges
    assert ("preset", "driver", "robot_bridge", "driver") in edges
    assert ("robot_bridge", "report", "joint_state", "trigger") in edges
    assert ("robot_bridge", "report", "follow_cube", "trigger") in edges
    assert ("robot_bridge", "robot", "follow_cube", "robot") in edges
    assert "shoulder_pan_index" not in node_types
