# blacknode-perception

**Robot vision nodes for [Blacknode](https://github.com/temiroff/Blacknode).**

Install this Blacknode **extension package** to add robot vision to the visual
workflow editor: run USB cameras,
stream ROS 2 images, inspect VLM reasoning, track objects with OpenCV, and
drive it all from workflows or AI agents over MCP.

It composes with `blacknode-ros2`: ROS 2 handles camera transport, topic
inspection, snapshots, and streams, while `blacknode-perception` adds
vision-specific workflow pieces: a bundled generic USB camera ROS 2 node,
camera consoles, frame prompts, stream dashboards, OpenCV tracking, and
optional VLM/LLM inspection.

## Install

From the Blacknode repo root:

```bash
blacknode packages install https://github.com/temiroff/blacknode-perception.git
```

This package expects `blacknode-ros2` when using the ROS camera templates:

```bash
blacknode packages install https://github.com/temiroff/blacknode-ros2.git
```

For direct local cameras, add one `Camera` node and set `selection` to `0`.
Duplicate it and set `selection` to `1`, `2`, and so on for more cameras. The
node handles discovery, selection, streaming, and live preview itself.
Discovery, selection, streaming, and the older `CV2Camera*` node types remain
registered internally for saved-workflow compatibility but stay out of the
normal node palette.

Build the bundled ROS 2 camera package:

```bash
blacknode packages setup blacknode-perception
```

If your Blacknode build does not run package setup scripts yet, run
`bash packages/blacknode-perception/scripts/setup.sh` from the Blacknode repo root.

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
| `Camera` | Discovers, selects, and streams one local camera with a live preview; duplicate it for more cameras |
| `CameraCalibration` | Captures checkerboard views, solves intrinsics and field of view, and emits a calibrated camera stream |
| `FramePrompt` | Builds a concise robot-vision prompt for one camera frame |
| `DetectionPrompt` | Builds an LLM prompt from CV2 detections for local reasoning |
| `StreamStatus` | Renders live camera stream readiness as a dashboard image |
| `VLM` | Sends one image frame or text-only detection prompt to OpenAI-compatible, NVIDIA NIM, Anthropic, or local Ollama chat |
| `ReasoningDashboard` | Shows the captured frame with the VLM's visible observations, evidence, uncertainty, and next action |
| `ReasoningStream` | Starts a live MJPEG dashboard that periodically describes a camera image with local Ollama or NVIDIA NIM |
| `CV2HSVMask` | Creates an HSV color mask from a Blacknode image |
| `CV2ColorTargetHint` | Converts target/reasoning text like `track red cube` into label and HSV settings for CV2 tracking |
| `CV2ColorObjectTracker` | Tracks colored objects such as cubes and returns overlay, mask, center, area, and detections |
| `CV2ColorObjectStream` | Starts live MJPEG overlay and mask streams from a camera snapshot URL and exposes current snapshot and detection JSON |

## Camera calibration

Connect `Camera.frame_stream` to `CameraCalibration.frame_stream`. Use a printed
checkerboard and set `board_columns` and `board_rows` to its number of inner
corners, then set `square_size` to the measured edge length of one square in
meters. Keep the camera's resolution and focus fixed while calibrating.

Run the node with `action=capture` for at least `min_samples` varied views. Move
the board across the image, tilt it in different directions, and include both
near and far views. Each cook captures the current `snapshot_url`; an `image`
input or a batch `frames` list can be supplied instead. Views with no complete
checkerboard or a different resolution are rejected. Use `action=reset` to
discard stored observations for the selected `stream_id`.

After collecting the views, run with `action=solve`. The node reports the RMS
reprojection error and emits a versioned `blacknode.camera-calibration` artifact
containing image size, camera matrix, distortion coefficients, `fx`, `fy`, `cx`,
`cy`, horizontal/vertical field of view, board settings, and sample count. Its
`calibrated_stream` output copies the input frame-stream handle and attaches the
same intrinsics and calibration artifact. Connect that output to dataset camera
collection so the recorder stores the calibration alongside the camera stream
metadata for later replay, simulation, and training workflows.

## Templates

- **Camera Console** — start the local camera, stream it live, and show a
  status dashboard.
- **Live VLM Reasoning** — start the local camera, keep the live stream
  visible, send frames to a VLM, and render a reasoning dashboard beside the
  image.

Both default to the native `Camera` node (OpenCV/DirectShow, no ROS 2 or
Docker required). For a different camera index, edit the node's `selection`
input.

CV2 color-object tracking and follow-target templates (cube/target tracking,
ROS 2 or rosbridge robot control) live in `blacknode-skills`' `follow-person`
component now, not here.

## VLM and LLM endpoints

`VLM` and `ReasoningStream` both support these providers:

| Provider | Endpoint | Key | Default model |
|---|---|---|---|
| `openai-compatible` | `/chat/completions` | `VISION_API_KEY`, `OPENAI_API_KEY`, or `NVIDIA_API_KEY` | `gpt-4o-mini` |
| `nvidia` | `https://integrate.api.nvidia.com/v1/chat/completions` | `NVIDIA_API_KEY` | `nvidia/nemotron-nano-12b-v2-vl` |
| `anthropic` | `/messages` | `ANTHROPIC_API_KEY` or `VISION_API_KEY` | `claude-sonnet-4-5` |
| `ollama` | `/api/chat` | no key for local Ollama | `qwen3-vl:4b` |

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

`provider: nvidia` calls NVIDIA's hosted NIM catalog with the same
OpenAI-compatible wire format, using `NVIDIA_API_KEY`. `nvidia/cosmos-reason1-7b`
and `nvidia/cosmos-reason2-8b` are NVIDIA's own open models built specifically
for physical-AI/robot reasoning, but as of this writing NIM hosts them as gated
"functions" that 404 for accounts without separate access approval; the
`nvidia/nemotron-nano-12b-v2-vl` default works out of the box with just an API
key. Switching `provider` on the node's editor panel automatically swaps
`model`/`endpoint_url` to that provider's own default unless you've explicitly
set them yourself. The editor also renders `model` as a dropdown of your
installed models (via `GET /ollama/models`) whenever `provider: ollama`.

Qwen3 models can spend many tokens in Ollama's hidden thinking phase before
returning final `content`, so `VLM` automatically raises
`num_predict` to at least `4096` for Qwen3 models.

If your installed Ollama model is text-only, keep `allow_text_only` enabled and
feed it a `DetectionPrompt` from CV2 detections. If your model is a true
local VLM, connect the camera snapshot image into `VLM.image`.

Live reasoning uses a snapshot URL for inference, not the MJPEG stream itself.
`ReasoningStream` periodically samples the current snapshot and serves an
MJPEG reasoning dashboard, so the visible panel keeps updating while the camera
and tracker streams run. The CV2 local-reasoning template defaults to
`interval_seconds: 3.0` and dashboard `max_fps: 4.0`; actual reasoning updates
are still limited by how fast the local VLM returns an answer.

Changing `interval_seconds`, `model`, `provider`, `prompt`, or similar params
on an already-running reasoning stream takes effect the next time you cook the
node with its per-node Run control — the running
process is patched in place over HTTP rather than restarted, so the dashboard
doesn't drop or reconnect. `image_url`/`detection_url`/`host`/`port` follow the
same live-patch path; only stopping and starting the stream changes those.

## CV2 tracking

The live cube tracker exposes one color picker, `object_color`. Internally it
derives the HSV threshold range needed by OpenCV from that selected color.

In the live reasoning template, the target prompt goes to the VLM first, not
directly to CV2:

```text
Text target prompt
  -> ReasoningStream
  -> CV2ColorObjectStream.reasoning_state_url
```

When `use_reasoning_color` is enabled, the model answer can choose the target
color and the CV2 stream updates its internal HSV range while it is running. If
reasoning does not return a color yet, the stream uses `object_color`. When
`use_reasoning_color` is disabled, `object_color` is the manual tracking color.
Reasoning answers are asked to include a `Target: <color> <object>` line
precisely so color extraction can prefer whatever's named there — a `Scene:`
line describing the surroundings often mentions other colors in frame (e.g.
"green and red cubes" when only the green one is the target), so the color
picker specifically looks after `Target:` first before falling back to
scanning the whole answer.
Most `CV2ColorObjectStream` properties are hot updated from the editor:
changing `object_color`, `use_reasoning_color`, `target`, `min_area`, `blur`,
`morphology_iters`, FPS, width, or JPEG quality updates the running overlay,
mask, and detection JSON without restarting the MJPEG URLs. For non-VLM
workflows, `CV2ColorTargetHint.target` or `CV2ColorObjectStream.target` can
still accept direct text such as `track red cube`.

The CV2 tracker is still a fast color-threshold tracker, so it does not detect
every cube automatically by shape. The VLM/reasoning side chooses what color to
track, then CV2 does the live low-latency tracking.

`CV2ColorObjectStream` keeps overlay and mask previews live and exposes the
latest mask stream at `/mask.mjpg`, mask snapshot at `/mask.png`, frame
snapshot at `/snapshot.jpg`, and detection at `/detection.json`;
it also returns a `detection_stream` handle for persistent controller nodes,
so new detections do not require graph re-cooks.
`CV2ColorObjectTracker` is still useful for single-frame tests. Both return
structured detections, so the same prototype can drive a local LLM, robot
control node, dashboard, or graph export.

## Export

Use the top-bar **Export** dropdown on the actual canvas graph. **Plain
Python** exports the same nodes and edges you built visually, including ROS 2,
camera, reasoning, and CV2 nodes. No special export node is required.

The exported Python keeps Blacknode as the runtime layer. A later robot-deploy
exporter can compile supported graph patterns into smaller standalone scripts,
but it should still be an export target, not a node on the canvas.

## Development

Coding agents should read [`AGENTS.md`](AGENTS.md) before changing this package.
It defines vision ownership, managed-stream behavior, freshness requirements,
and verification commands.

Run tests from the Blacknode repo root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest packages/blacknode-perception/tests
```

Validate templates:

```bash
for f in packages/blacknode-perception/templates/*.json; do .venv/bin/blacknode validate "$f"; done
```

## License

Apache-2.0, same as Blacknode.
