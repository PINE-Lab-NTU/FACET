# Robot backends

Both adapters expose native base-frame Cartesian pose as `[x,y,z,rx,ry,rz]` in metres
and rotation-vector radians, measured gripper opening in [0,1], and measured wrench
in [N,Nm]. The meaning/frame of wrench must match the training data; this package
does not silently rotate, bias or re-zero it.

`policy.position_sign` is an explicit feature-encoding convention. For example,
`[-1,1,1]` supports a corpus that stored a negated position-x channel while keeping
the original base-frame rotation vector. It is not a general rigid frame transform.
`reference_rotvec` optionally selects an equivalent rotation-vector branch; it does
not command an orientation or act as a goal pose.

Workspace bounds always use the native robot frame. The model and teleop proposals
pass through the same fresh-state checks and physical speed/force bounds. Unknown
configuration fields are rejected, so archived per-task settings cannot silently
reenable a fixed grasp/start/reset pose.

## UR

Install the `ur` extra and configure a reachable RTDE address. The arm must already
be in the correct remote/external-control mode. A `probe` constructs only the receive
interface; it does not create an RTDE control program or send servo commands.
Physical `run --execute` constructs the control interface and uses Cartesian servoL.
It does not call moveJ/moveL, forceMode, zeroFtSensor or a home/reset routine.

For an already-active Robotiq UR register server, set `gripper: robotiq` and its port.
Register POS is converted from 0=open/255=closed to the wire's continuous 1=open.
Activation/calibration must be completed separately; the adapter only checks STA.
If `gripper: none`, gripper commands are ignored and the reported placeholder opening
is 1. Do not use that setting when your policy depends on measured gripping state.

RTDE safety mode must be normal and controller timestamps must advance. A dedicated
streamer sends targets at the configured RTDE control frequency. Loss of fresh
commands or invalid measured state latches a software fault and servo-stops.
The UR backend supplies position servoing only: force-compliant insertion requires
your own validated controller/backend. The release does not reuse task-coordinate
force-mode transitions from the experiment code.

## Franka

Start your existing low-level compliant controller and gripper driver. Configure the
robot-server URL. The expected HTTP protocol is:

```text
POST /getstate {}
  -> {pose: [x,y,z,qx,qy,qz,qw], gripper_pos: opening_0_to_1,
      force: [fx,fy,fz], torque: [tx,ty,tz], state_fresh: true,
      state_age_sec: measured_age_seconds}
POST /pose {arr: [x,y,z,qx,qy,qz,qw]}
POST /move_gripper {gripper_pos: integer_0_closed_to_255_open}
```

`/pose` and `/move_gripper` must reject invalid targets in the robot-side controller.
The HTTP interface is not a real-time impedance controller. This client adds a
guarded target stream and freshness/heartbeat checks, but controller safety remains
external. The robot-side controller must also handle command timeouts/disconnections
safely; a client cannot send a hold through a broken network link. A hold resends the
measured pose and never changes gripper opening.
If using a server with another gripper unit/frame convention, write an explicit
adapter rather than relying on the matching endpoint names.

## Cameras and teleoperation

Use exact training roles `view1`, `hand` and `view2`. Set physical serials locally;
aliases explicitly duplicate a physical camera for a corpus trained that way.
No serial is shipped. Images are RGB; the policy's own transforms resize/normalize
them. There is no third-view masking assumption in the robot runtime.

SpaceMouse translation is relative to the measured native pose. Verify the axis
mapping at low speed in your cell. Rotation is passed through in this curated runtime.
Button 0 requests open, button 1 close; buttons never label success or trigger a reset.
