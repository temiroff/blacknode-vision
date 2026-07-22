# Adapters

Transport adapters for the `depth` component of `blacknode-perception`.

One folder per transport, each mirroring the component layout:

    adapters/ros2/nodes/
    adapters/ros2/templates/

Declare it in `blacknode-package.toml`:

    [components.depth.adapters.ros2]
    description = "ROS 2 adapter for depth."
    default = false
    capabilities = ["adapter.depth.ros2"]
    nodes = ["components/depth/adapters/ros2/nodes"]

Adapters stay `default = false`: the capability package owns them, and
`blacknode-ros2` provides only the shared transport underneath.
