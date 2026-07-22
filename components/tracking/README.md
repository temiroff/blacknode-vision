# Tracking

Component of `blacknode-perception`.

Node sources for this component belong in this folder. Until they move here,
nodes claim the component inline:

    @node(name="MyNode", component="tracking", ...)

Once sources live here, declare the folder in `blacknode-package.toml`:

    [components.tracking]
    nodes = ["components/tracking/nodes"]

and the inline `component=` argument can be dropped — the loader infers it
from the directory.
