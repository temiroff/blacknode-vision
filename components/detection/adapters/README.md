# Adapters

Transport adapters for the `detection` component of `blacknode-perception`.

One folder per transport, each mirroring the component layout:

    adapters/ros2/nodes/
    adapters/ros2/templates/

Declare it in `blacknode-package.toml`:

    [components.detection.adapters.ros2]
    description = "ROS 2 adapter for detection."
    default = false
    capabilities = ["adapter.detection.ros2"]
    nodes = ["components/detection/adapters/ros2/nodes"]

Adapters stay `default = false`: the capability package owns them, and
`blacknode-ros2` provides only the shared transport underneath.
