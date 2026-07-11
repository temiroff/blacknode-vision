# blacknode-vision

Vision workflows for [Blacknode](https://github.com/temiroff/Blacknode).

This is a separate Blacknode extension package. It does not replace
`blacknode-ros2`; it builds on it. ROS 2 handles camera transport, topic
inspection, snapshots, and streams. `blacknode-vision` adds vision-specific
workflow pieces: a bundled generic USB camera ROS 2 node, camera consoles,
frame prompts, stream dashboards, and optional VLM inspection.

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
| `VisionStreamStatus` | Renders live camera stream readiness as a dashboard image |
| `VisionVLMDescribe` | Sends one image frame to an OpenAI-compatible vision chat endpoint |

## Templates

- **Blacknode Vision Camera Console** — start `blacknode_usb_camera usb_camera`,
  stream `/camera/image_raw`, and show a live status dashboard.
- **Blacknode Vision Frame VLM** — capture one ROS 2 camera frame, show it on
  the canvas, and send it to a VLM endpoint.

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

## VLM endpoint

`VisionVLMDescribe` calls an OpenAI-compatible `/chat/completions` endpoint with
one image. For hosted endpoints, set one of these environment variables before
starting Blacknode:

```bash
export VISION_API_KEY=...
# or OPENAI_API_KEY / NVIDIA_API_KEY
```

Local OpenAI-compatible endpoints on `localhost` or `127.0.0.1` can run without
an API key.

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
