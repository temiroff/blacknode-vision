# blacknode-vision

Vision workflows for [Blacknode](https://github.com/temiroff/Blacknode).

This is a separate Blacknode extension package. It does not replace
`blacknode-ros2`; it builds on it. ROS 2 handles camera transport, topic
inspection, snapshots, and streams. `blacknode-vision` adds vision-specific
workflow pieces: a bundled generic USB camera ROS 2 node, camera consoles,
frame prompts, stream dashboards, OpenCV tracking, and optional VLM/LLM
inspection.

## Install

From the Blacknode repo root:

```bash
blacknode packages install https://github.com/temiroff/blacknode-vision.git
```

This package expects `blacknode-ros2` when using the ROS camera templates:

```bash
blacknode packages install https://github.com/temiroff/blacknode-ros2.git
```

Build the bundled ROS 2 camera package:

```bash
blacknode packages setup blacknode-vision
```

If your Blacknode build does not run package setup scripts yet, run
`bash packages/blacknode-vision/scripts/setup.sh` from the Blacknode repo root.

Restart Blacknode, or press **Reload** in the editor's Packages tab.

## Nodes

### ROS 2 executables

| Package | Executable | What it does |
|---|---|---|
| `blacknode_usb_camera` | `usb_camera` | Publishes a local USB camera to `/camera/image_raw` |

The USB camera node accepts ROS parameters:

```text
device:=0
image_topic:=/camera/image_raw
width:=640
height:=480
hz:=30.0
rotation:=0
```

### Blacknode nodes

| Node | What it does |
|---|---|
| `VisionFramePrompt` | Builds a concise robot-vision prompt for one camera frame |
| `VisionDetectionPrompt` | Builds an LLM prompt from CV2 detections for local reasoning |
| `VisionStreamStatus` | Renders live camera stream readiness as a dashboard image |
| `VisionVLMDescribe` | Sends one image frame or text-only detection prompt to OpenAI-compatible, Anthropic, or local Ollama chat |
| `VisionReasoningDashboard` | Shows the captured frame with the VLM's visible observations, evidence, uncertainty, and next action |
| `VisionReasoningStream` | Starts a live MJPEG dashboard that periodically describes a camera image with local Ollama |
| `CV2HSVMask` | Creates an HSV color mask from a Blacknode image |
| `CV2ColorTargetHint` | Converts target/reasoning text like `track red cube` into label and HSV settings for CV2 tracking |
| `CV2ColorObjectTracker` | Tracks colored objects such as cubes and returns overlay, mask, center, area, and detections |
| `CV2ColorObjectStream` | Starts live MJPEG overlay and mask streams from a camera snapshot URL and exposes current snapshot and detection JSON |
| `CV2TrackerPythonExport` | Generates a standalone OpenCV tracker script for robot deployment experiments |

## Templates

- **Blacknode Vision Camera Console** — start `blacknode_usb_camera usb_camera`,
  stream `/camera/image_raw`, and show a live status dashboard.
- **Blacknode Vision Frame VLM** — capture one ROS 2 camera frame, show it on
  the canvas, and send it to a VLM endpoint.
- **Blacknode Vision Live VLM Reasoning** — start the USB camera, keep the live
  stream visible, capture one frame, call the VLM, and render a reasoning
  dashboard beside the image.
- **Blacknode Vision CV2 Cube Local Reasoning** — start the USB camera, stream
  the raw image, run live image-first Ollama/Qwen reasoning from the raw camera
  snapshot, resolve the target color from the reasoning state, and stream a live
  OpenCV tracking overlay and mask.

For the common case, `./start.sh` auto-sources `/opt/ros/jazzy/setup.bash` and
auto-sources a ROS workspace when it finds exactly one `ros2_ws/install/setup.bash`.
If you have multiple ROS workspaces, source the one you want before starting
Blacknode so the overlay order is explicit:

```bash
source /opt/ros/jazzy/setup.bash
source /path/to/ros2_ws/install/setup.bash
./start.sh
```

Then load **Blacknode Vision Camera Console**. It defaults to:

```text
package: blacknode_usb_camera
executable: usb_camera
expected_topic: /camera/image_raw
```

For a different camera index, edit `ROS2Run.arguments`, for example
`-p device:=1`.

## VLM and LLM endpoints

`VisionVLMDescribe` supports three providers:

| Provider | Endpoint | Key |
|---|---|---|
| `openai-compatible` | `/chat/completions` | `VISION_API_KEY`, `OPENAI_API_KEY`, or `NVIDIA_API_KEY` |
| `anthropic` | `/messages` | `ANTHROPIC_API_KEY` or `VISION_API_KEY` |
| `ollama` | `/api/chat` | no key for local Ollama |

For hosted endpoints, set the key before starting Blacknode:

```bash
export VISION_API_KEY=...
# or OPENAI_API_KEY / NVIDIA_API_KEY / ANTHROPIC_API_KEY
```

Local Ollama defaults to:

```text
provider: ollama
endpoint_url: http://127.0.0.1:11434
model: qwen3-vl:4b
max_tokens: 4096
```

Qwen3 models can spend many tokens in Ollama's hidden thinking phase before
returning final `content`, so `VisionVLMDescribe` automatically raises
`num_predict` to at least `4096` for Qwen3 models.

If your installed Ollama model is text-only, keep `allow_text_only` enabled and
feed it a `VisionDetectionPrompt` from CV2 detections. If your model is a true
local VLM, connect the camera snapshot image into `VisionVLMDescribe.image`.

Live reasoning uses a snapshot URL for inference, not the MJPEG stream itself.
`VisionReasoningStream` periodically samples the current snapshot and serves an
MJPEG reasoning dashboard, so the visible panel keeps updating while the camera
and tracker streams run. The CV2 local-reasoning template defaults to
`interval_seconds: 3.0` and dashboard `max_fps: 4.0`; actual reasoning updates
are still limited by how fast the local VLM returns an answer.

Changing `interval_seconds`, model, or FPS on an already-running reasoning
stream requires cooking the node again. The current process is started with the
node settings it had at launch; stop and run the workflow again to apply new
stream settings.

## CV2 tracking

The cube tracker uses HSV thresholds. The default range is tuned for green:

```text
lower_hsv: 35,60,60
upper_hsv: 85,255,255
```

In the live reasoning template, the target prompt goes to the VLM first, not
directly to CV2:

```text
Text target prompt
  -> VisionReasoningStream
  -> CV2ColorObjectStream.reasoning_state_url
```

The model answer chooses the target color, then the CV2 stream updates its HSV
range while it is running. The template leaves `CV2ColorObjectStream.target`
and `fallback_color` empty so the tracker does not silently lock onto a
hard-coded color before the VLM answers. For non-VLM workflows,
`CV2ColorTargetHint.target` or `CV2ColorObjectStream.target` can still accept
direct text such as `track red cube`.

The CV2 tracker is still a fast color-threshold tracker, so it does not detect
every cube automatically by shape. The VLM/reasoning side chooses what color to
track, then CV2 does the live low-latency tracking.

`CV2ColorObjectStream` keeps overlay and mask previews live and exposes the
latest mask stream at `/mask.mjpg`, mask snapshot at `/mask.png`, frame
snapshot at `/snapshot.jpg`, and detection at `/detection.json`;
`CV2ColorObjectTracker` is still useful for single-frame tests and exports.
Both return structured detections, so the same prototype can drive a local LLM,
robot control node, dashboard, or Python export.

## Development

Run tests from the Blacknode repo root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest packages/blacknode-vision/tests
```

Validate templates:

```bash
for f in packages/blacknode-vision/templates/*.json; do .venv/bin/blacknode validate "$f"; done
```

## License

Apache-2.0, same as Blacknode.
