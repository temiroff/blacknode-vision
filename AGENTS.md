# blacknode-vision Agent Instructions

This is an independent extension-package repository. Check and commit its Git
state separately from the Blacknode core checkout that may contain it.

## Scope

Keep camera acquisition, OpenCV tracking, VLM image reasoning, vision stream
servers, and vision templates here. Keep ROS transport in `blacknode-ros2` and
generic graph/runtime behavior in Blacknode core.

## Development rules

- Preserve loadability without a camera, ROS, OpenCV, or a model endpoint.
- Return structured readiness/errors and allow unrelated nodes to keep working.
- Treat camera, tracker, and reasoning streams as managed services. Cooking once
  starts or patches a service; new frames must not require graph re-cooks.
- Preserve stable stream URLs and latest-value handles during hot updates.
- Distinguish worker heartbeat from source-frame freshness. Never present a
  cached frame or detection as live without checking its age.
- Keep CV2 tracking deterministic and testable with generated fixtures. Keep
  provider/network tests optional and credential-gated.
- Declare new imports in `blacknode-package.toml` and `requirements.txt`.
- Mark templates with `metadata.required_packages` for every package they use.

## Verification

From the Blacknode root:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
python -m pytest packages/blacknode-vision/tests
Get-ChildItem packages\blacknode-vision\templates\*.json | ForEach-Object { blacknode validate $_.FullName }
```

Report camera, ROS, GPU, or model paths that were not exercised. See the
Blacknode `docs/packages.md` and `blacknode-development` skill for shared rules.
