# Runtime interfaces

The runtime reads the current robot state, constructs causal observations and requests
a frozen policy reference. A bounded absolute Actor or explicit teleoperator selects
the command. The execution interface applies workspace and rate limits. The recorder
stores the command synchronously sent by the robot adapter, the requested command,
the original reference and the execution period. Background heartbeat sends repeat
that exact endpoint; they do not apply another hidden motion projection.

Episode starts and physical placement are operator-managed. There are no automatic
reset motions or coordinate-based phase transitions. Device addresses and camera
serials belong in local hardware configuration.
