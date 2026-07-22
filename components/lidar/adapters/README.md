# Adapters

Transport adapters for the `lidar` component of `blacknode-perception`.

One folder per transport, each mirroring the component layout:

    adapters/ros2/nodes/
    adapters/ros2/templates/

Declare it in `blacknode-package.toml`:

    [components.lidar.adapters.ros2]
    description = "ROS 2 adapter for lidar."
    default = false
    capabilities = ["adapter.lidar.ros2"]
    nodes = ["components/lidar/adapters/ros2/nodes"]

Adapters stay `default = false`: the capability package owns them, and
`blacknode-ros2` provides only the shared transport underneath.
