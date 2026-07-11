# blacknode-vision

Vision workflows for [Blacknode](https://github.com/temiroff/Blacknode).

This is a separate Blacknode extension package. It does not replace
`blacknode-ros2`; it builds on it. ROS 2 handles camera transport, topic
inspection, snapshots, and streams. `blacknode-vision` adds vision-specific
workflow pieces: camera consoles, frame prompts, stream dashboards, and optional
VLM inspection.

## Install

From the Blacknode repo root:

```bash
blacknode packages install https://github.com/temiroff/blacknode-vision.git
```

This package expects `blacknode-ros2` when using the ROS camera templates:

```bash
blacknode packages install https://github.com/temiroff/blacknode-ros2.git
```

Restart Blacknode, or press **Reload** in the editor's Packages tab.

## Nodes

| Node | What it does |
|---|---|
| `VisionFramePrompt` | Builds a concise robot-vision prompt for one camera frame |
| `VisionStreamStatus` | Renders live camera stream readiness as a dashboard image |
| `VisionVLMDescribe` | Sends one image frame to an OpenAI-compatible vision chat endpoint |

## Templates

- **Blacknode Vision Camera Console** — start any ROS 2 camera executable,
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

Then load **Blacknode Vision Camera Console** and fill `ROS2Run`:

```text
package: your_camera_package
executable: your_camera_executable
expected_topic: /camera/image_raw
```

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
