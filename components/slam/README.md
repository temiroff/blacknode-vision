# Slam

Component of `blacknode-perception`.

Node sources for this component belong in this folder. Until they move here,
nodes claim the component inline:

    @node(name="MyNode", component="slam", ...)

Once sources live here, declare the folder in `blacknode-package.toml`:

    [components.slam]
    nodes = ["components/slam/nodes"]

and the inline `component=` argument can be dropped — the loader infers it
from the directory.
