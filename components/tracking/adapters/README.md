# Adapters

Transport adapters for the `tracking` component of `blacknode-perception`.

One folder per transport, each mirroring the component layout:

    adapters/ros2/nodes/
    adapters/ros2/templates/

Declare it in `blacknode-package.toml`:

    [components.tracking.adapters.ros2]
    description = "ROS 2 adapter for tracking."
    default = false
    capabilities = ["adapter.tracking.ros2"]
    nodes = ["components/tracking/adapters/ros2/nodes"]

Adapters stay `default = false`: the capability package owns them, and
`blacknode-ros2` provides only the shared transport underneath.
