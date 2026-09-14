"""One room, one model, one clock: drive to a table, work there, drive on.

Navigation and manipulation each used to build their own model - one balancing
robot free in the room, one bolted to the floor at a dock - so a demo that did
both jumped between two simulations at the table, resetting the clock and the
objects on the way.  Here both run in the same `MjModel` and the same `MjData`.

What replaces the bolt is a parking brake: a weld equality between the base and
the world, declared when the model is compiled because equalities cannot be
added to a compiled one, inactive while the robot drives, and switched on at
the pose the robot actually arrived at.  The arms then work against a base held
as still as `welded_at` held it, and the brake comes off before the next leg.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .control import DriveController, Gains
from .gripper_pads import add_pads
from .grasp import Rig, hold_everything, read_state, set_const
from .navigate import Navigator, true_pose
from .navmap import OccupancyGrid
from .planner import goal_for
from .robot import ROOM
from .room import TABLES, Table
from .skills import Robot

BRAKE = "park"
COMMAND_HZ = 50            # the bridge ticks its controller at 50 Hz
ARRIVE_XY, ARRIVE_YAW = 0.10, 5.0
FALLEN = math.radians(45)
DRIVE_IMPRATIO = 10        # the room solver stays at 10 to drive,
GRIP_IMPRATIO = 200        # and manipulation wants 200 for a firm pinch
_SPATIAL_EPS = 1e-9


# The pads each table's controller was tuned on.  Anything not listed gets
# the stock pads the skills use.
PADS_AT = {"pick": "padded"}
# Where a controller needs the base closer to its training pose than the
# navigator's 10 cm / 5 deg.  Flybrain reaches the cube through IK from
# wherever the base stopped, and from 9 cm out the carry lands the cube
# 0.38 m from the goal; from 1.4 cm it places it.  Costs ~15 s of settling.
ARRIVE_AT = {"pick": (0.04, 2.0)}
FINGERS = [f"{prefix}{finger}_finger__{finger}_finger"
           for prefix in ("", "l_") for finger in ("left", "right")]
INERTIAL = ("body_mass", "body_inertia", "body_ipos", "body_iquat")
# Everything that makes a pad the pad it is, all of it writable on a compiled
# model: where it sits on the blade, how big, how grippy, how soft.
PAD_FIELDS = ("geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_solref",
              "geom_solimp", "geom_condim", "geom_rbound", "geom_aabb")


def blade_params(model, suffix: str) -> dict:
    """Per blade: its pad's fields, its inertials, and its broadphase box.

    Two hands' worth of rubber have to share one room: the skills were tuned
    on the pads scripts/build_mjcf.py lays on (`*_pad0`), the Flybrain policy
    was trained on the fitted ones (`*_pad`), and neither works on the other's.
    Two pad geoms on one blade does not work either - a second leaf in the
    blade's collision tree, even one that touches nothing, cost the skills
    every grasp (0/16 against 16/16).  So each blade carries one pad, and
    `LiveSim.use_pads` rewrites it into the other one at the table.
    """
    out = {}
    for finger in FINGERS:
        body, geom = model.body(finger).id, model.geom(finger + suffix).id
        assert model.body_bvhnum[body] == 1, f"{finger} should have one colliding geom, its pad"
        out[finger] = dict(
            pad=tuple(getattr(model, field)[geom].copy() for field in PAD_FIELDS),
            inertial=tuple(getattr(model, field)[body].copy() for field in INERTIAL),
            bvh=model.bvh_aabb[model.body_bvhadr[body]].copy())
    return out


def with_brake(path=ROOM):
    """The room with the Flybrain pads fitted to the blades, plus one inactive
    weld between the base and the world.

    No joints are deleted, so the room's keyframes still fit the model and the
    robot can still be dropped at a dock.
    """
    spec = mujoco.MjSpec.from_file(str(path))
    add_pads(spec)
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1, eq.name2, eq.name = "root", "world", BRAKE
    eq.active = False
    return spec.compile()


def furniture_contacts(model, data) -> set:
    """Names of room geoms the robot is touching. Wheels on the floor do not count.

    The same check `scripts/navigate.py` scores routes with, copied rather than
    imported: `src/` does not import from `scripts/`.
    """
    root = model.body("root").id
    names = set()
    for k in range(data.ncon):
        con = data.contact[k]
        roots = [model.body_rootid[model.geom_bodyid[g]] for g in (con.geom1, con.geom2)]
        if (roots[0] == root) == (roots[1] == root):
            continue
        other = con.geom2 if roots[0] == root else con.geom1
        if model.geom_type[other] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        names.add(model.geom(other).name or f"geom{other}")
    return names


class LiveSim:
    """One room, one model, one clock: drive to tables and work at them without reloading."""

    def __init__(self, keyframe: str = "start", on_step=None):
        self.model = with_brake()
        self._blades = {"padded": blade_params(self.model, "_pad"),
                        "stock": blade_params(mujoco.MjModel.from_xml_path(str(ROOM)), "_pad0")}
        self._scratch = mujoco.MjData(self.model)   # mj_setConst writes qpos0 into its data
        self.data = mujoco.MjData(self.model)
        self.use_pads("stock")
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(keyframe).id)
        mujoco.mj_forward(self.model, self.data)
        hold_everything(self.model, self.data)
        self.brake = self.model.equality(BRAKE).id
        self.wheels = [self.model.actuator(f"wheel_{s}").id for s in ("left", "right")]
        self.rig = Rig(self.model, self.data, driver=None, on_step=on_step)
        self.station: str | None = None
        self.parked = False

    @property
    def table(self) -> Table | None:
        return next((t for t in TABLES if t.name.split("_")[1] == self.station), None)

    def pose(self) -> tuple[float, float, float]:
        return true_pose(self.data)

    def drive_to(self, station: str, grid=None, nav_hz: float = 10.0,
                 arrive_xy: float | None = None, arrive_yaw: float | None = None) -> dict:
        """Drive to a table's dock the way `scripts/navigate.py` drives there.

        Same three rates - 500 Hz physics, 50 Hz commands, a 10 Hz navigator
        replanning from the sim's true pose - but stepped through the rig, so a
        viewer or a recorder sees every step of the drive as it sees every step
        of a pick.  The controller runs as the rig's driver for the duration:
        that is the one hook that gets called before each `mj_step`.
        """
        goal = goal_for(station)        # an unknown station is a mistake, not a
                                        # route: fail before the brake comes off
        tol_xy, tol_yaw = ARRIVE_AT.get(station, (ARRIVE_XY, ARRIVE_YAW))
        arrive_xy = tol_xy if arrive_xy is None else arrive_xy
        arrive_yaw = tol_yaw if arrive_yaw is None else arrive_yaw
        if self.parked:
            self.home()                 # an arm left over the table sweeps the
            self.unpark()               # furniture on the way out
        self.model.opt.impratio = DRIVE_IMPRATIO

        radius = float(self.model.geom("wheel_left_collision").size[0])
        control = DriveController(radius, Gains.for_bracketbot())
        nav = Navigator(OccupancyGrid.from_room() if grid is None else grid,
                        arrive_xy=arrive_xy, arrive_yaw=math.radians(arrive_yaw))
        nav.go_to(goal, self.pose(), self.data.time)

        dt = self.model.opt.timestep
        every = max(1, int(round(1 / (COMMAND_HZ * dt))))
        think = max(1, int(round(1 / (nav_hz * dt))))
        sensors = (self.model.sensor("gyro").adr[0],
                   self.model.sensor("vel_left").adr[0],
                   self.model.sensor("vel_right").adr[0])
        command, step = (0.0, 0.0), 0

        def tick(data):
            nonlocal command, step
            # mj_step leaves the body poses a step behind qpos; the bridge reads
            # a settled state, so refresh before the navigator or the balancer
            # looks at one.
            mujoco.mj_forward(self.model, data)
            if step % think == 0:
                command = nav.step(true_pose(data), data.time)
            if step % every == 0:
                control.command(*command, data.time)
            data.ctrl[self.wheels] = control(read_state(self.model, data, sensors),
                                             data.time, dt)
            step += 1

        started = self.data.time
        budget = started + (nav.traj.duration * 5 + 15 if nav.traj is not None else 0.0)
        fell, touched = False, furniture_contacts(self.model, self.data)
        self.rig.driver = tick
        try:
            while (self.data.time < budget and not nav.done and not nav.failed
                   and not fell):
                self.rig.step()
                fell = abs(read_state(self.model, self.data, sensors).pitch) > FALLEN
                touched |= furniture_contacts(self.model, self.data)
        finally:
            self.rig.driver = None

        x, y, yaw = self.pose()
        failed = nav.failed
        if not nav.done and failed is None and not fell:
            failed = f"ran out of time after {budget - started:.0f} s"
        result = dict(
            station=station,
            err_xy=math.hypot(x - goal[0], y - goal[1]),
            err_yaw=math.degrees(abs(math.atan2(math.sin(yaw - goal[2]),
                                                math.cos(yaw - goal[2])))),
            seconds=float(self.data.time - started),
            replans=len(nav.replans),
            fell=fell,
            touched=sorted(touched),
            failed=failed,
        )
        result["ok"] = bool(not fell and not touched and failed is None
                            and result["err_xy"] <= arrive_xy + _SPATIAL_EPS
                            and result["err_yaw"] <= arrive_yaw + math.degrees(_SPATIAL_EPS))
        if result["ok"]:
            self.station = station
            self.park()
        return result

    def park(self) -> None:
        """Hold the base exactly where it stands, so the arms have a fixed dock.

        The weld wants the *world's* pose expressed in the root's frame.  Handing
        it the root's pose in the world instead compiles and runs, and satisfies
        the constraint about two metres away: the robot snaps there on the first
        step.  Written this way it drifts 0.1 mm in a thousand steps.
        """
        self.data.ctrl[self.wheels] = 0.0
        pos = self.data.body("root").xpos.copy()
        quat = self.data.body("root").xquat.copy()
        world = np.zeros(4)
        mujoco.mju_negQuat(world, quat)
        offset = np.zeros(3)
        mujoco.mju_rotVecQuat(offset, -pos, world)
        self.model.eq_data[self.brake, :3] = 0.0
        self.model.eq_data[self.brake, 3:6] = offset
        self.model.eq_data[self.brake, 6:10] = world
        self.model.eq_data[self.brake, 10] = 1.0        # torquescale
        self.data.eq_active[self.brake] = 1
        self.rig.driver = None      # the brake holds it up; nothing balances
        self.model.opt.impratio = GRIP_IMPRATIO
        self.use_pads(PADS_AT.get(self.station, "stock"))
        self.parked = True

    def unpark(self) -> None:
        self.data.eq_active[self.brake] = 0
        self.model.opt.impratio = DRIVE_IMPRATIO
        self.use_pads("stock")
        self.parked = False
        self.station = None

    def use_pads(self, kind: str) -> None:
        """Make each blade's pad the stock one the skills were tuned on, or the
        fitted one the Flybrain policy trained on.

        The pad's fields, the blade's inertials (4 g at the tip is the
        difference between the follower blade rebounding off the cube and
        swinging over it) and its broadphase box all move together.  No pose,
        no reset, no reload: `set_const` runs on a scratch MjData because
        `mj_setConst` writes qpos0 into whatever it is given.
        """
        for finger, blade in self._blades[kind].items():
            body, geom = self.model.body(finger).id, self.model.geom(finger + "_pad").id
            for field, value in zip(PAD_FIELDS, blade["pad"]):
                getattr(self.model, field)[geom] = value
            for field, value in zip(INERTIAL, blade["inertial"]):
                getattr(self.model, field)[body] = value
            self.model.bvh_aabb[self.model.body_bvhadr[body]] = blade["bvh"]
        set_const(self.model, self._scratch)
        self.pads = kind

    def robot(self) -> Robot:
        """The skills API on this model - no second simulation, no reload."""
        if not self.parked:
            raise RuntimeError("park at a table before working there")
        return Robot(self.station, model=self.model, data=self.data,
                     on_step=self.rig.on_step)

    def home(self) -> None:
        self.robot().home()
