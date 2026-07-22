# Adapters

Transport adapters for the `slam` component of `blacknode-perception`.

One folder per transport, each mirroring the component layout:

    adapters/ros2/nodes/
    adapters/ros2/templates/

Declare it in `blacknode-package.toml`:

    [components.slam.adapters.ros2]
    description = "ROS 2 adapter for slam."
    default = false
    capabilities = ["adapter.slam.ros2"]
    nodes = ["components/slam/adapters/ros2/nodes"]

Adapters stay `default = false`: the capability package owns them, and
`blacknode-ros2` provides only the shared transport underneath.
