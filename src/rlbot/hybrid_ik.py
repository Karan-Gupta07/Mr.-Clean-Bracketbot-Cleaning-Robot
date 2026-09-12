"""ctypes binding for the organisers' arm IK, libhybrid_ik_lib.so.

The library is BracketBot's fork of RangedIK (uwgraphics/relaxed_ik_core,
branch ranged-ik), built for Linux aarch64: it runs on the robot, or on a Mac
inside an arm64 container.  The binding and the defaults below mirror the
sponsors' own `ik.py` / `constants.py` (arm daemon), so a solve here is the
same call the teleop stack makes on hardware.  Verified by running it: the
constructor returns a handle, `forward_kinematics` matches MuJoCo's own
kinematics of this URDF to 0.001 mm, and a solve lands 5-9 mm from the target
with the sponsors' centering weights (that floor is the centering objective
pulling toward the nominal, not the solver failing: with centering off it
reaches 0.2 mm).  This file depends on nothing else in `rlbot` (whose
`__init__` imports mujoco), so on the robot copy it alone or load it by path.

    # on a Mac: the self-check, in a container
    docker run --rm --platform linux/arm64 -v "$PWD":/w -w /w python:3.12-slim \\
        python3 src/rlbot/hybrid_ik.py /path/to/libhybrid_ik_lib.so \\
        models/bracketbot/chopped_urdf_v2.urdf right_eef

How the sponsors drive it (constants.py):

  * one solver per arm: `base_link="arm_base"`, `ee_link="left_eef"` or
    `"right_eef"`; the URDF is passed as *text*;
  * `starting_config` all zeros, `nominal_config` with the elbow at 90 deg
    (`j3 = 1.5708`, their home), `joint_centering_weight = 0.1`, per-joint
    `centering_weights` pulling toward that nominal;
  * `rik_max_iterations = 5`, called every teleop tick (150 Hz), ~1.3 ms a
    call.  It is effectively a one-tick solver: a reachable target 180 mm
    away is within ~8 mm after the first call.  It can also switch branches
    in one tick (rail 0.1 m, joints ~0.9 rad at once), so rate-limit what you
    send to the motors, as the daemon does with clip_target and its EMA;
  * the rail (j0) is not held: it does most of any vertical move and shifts
    even for a pure forward target.  `set_mast_hold` with guessed arguments
    did not pin it;
  * tolerances 5 mm / 0.01 rad set once with `set_tolerances`, then
    `solve_pose(pos, quat)` - not `solve_position`, which takes tolerances
    per call;
  * quaternions are x, y, z, w;
  * `Opt` is {double* data; int32 length}; the memory belongs to the library
    and is valid until the next call, so copy it out;
  * joint order is [rail (m), j1..j6 (rad)]; the gripper is not IK-driven.

Setters whose meanings are not published (`set_mast_hold`,
`set_collision_params`, `set_fresh_seed`) are bound with the right types only;
the sponsors leave them at the Rust defaults and so do we.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import POINTER, c_char_p, c_double, c_int, c_void_p

DBL = POINTER(c_double)

# sponsors' arm daemon settings (constants.py, arm_left / arm_right)
TOLERANCES = [0.005] * 3 + [0.01] * 3          # m, rad
NOMINAL = [0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 0.0]  # elbow at 90 deg: their home
CENTERING_WEIGHTS = [50.0, 25.0, 15.0, 5.0, 10.0, 12.0, 1.5]
JOINT_CENTERING_WEIGHT = 0.1
MAX_ITER = 5


class Opt(ctypes.Structure):
    _fields_ = [("data", DBL), ("length", c_int)]

    def tolist(self) -> list[float]:
        return list(self.data[: self.length])


def load(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)
    lib.relaxed_ik_new.argtypes = [c_char_p, c_char_p, c_char_p,   # urdf text, base link, ee link
                                   DBL, c_int,                     # starting_config
                                   DBL, c_int,                     # nominal_config (NULL, 0 allowed)
                                   c_double,                       # joint_centering_weight
                                   c_int]                          # rik_max_iterations (0 panics)
    lib.relaxed_ik_new.restype = c_void_p
    lib.relaxed_ik_free.argtypes = [c_void_p]
    lib.reset.argtypes = [c_void_p, DBL, c_int]
    for name in ("solve", "solve_position", "solve_velocity"):
        getattr(lib, name).argtypes = [c_void_p, DBL, c_int, DBL, c_int, DBL, c_int]
        getattr(lib, name).restype = Opt
    lib.solve_pose.argtypes = [c_void_p, DBL, c_int, DBL, c_int]
    lib.solve_pose.restype = Opt
    for name in ("forward_kinematics", "get_all_frames"):
        getattr(lib, name).argtypes = [c_void_p, DBL, c_int]
        getattr(lib, name).restype = Opt
    lib.set_tolerances.argtypes = [c_void_p, DBL, c_int]
    # The rest are not in every build of the solver family (the sponsors'
    # ik.py looks these up the same way), so a missing one is not an error.
    optional = {
        "get_ee_positions": ([c_void_p], Opt),
        "num_obstacles": ([c_void_p], c_int),
        "set_nominal_config": ([c_void_p, DBL, c_int], None),
        "set_centering_weights": ([c_void_p, DBL, c_int], None),
        "set_mast_hold": ([c_void_p, c_double, c_double, c_double], None),
        "set_collision_params": ([c_void_p, c_double, c_double, c_double, c_double], None),
        "set_fresh_seed": ([c_void_p, c_double, c_double, c_int], None),
    }
    for name, (argtypes, restype) in optional.items():
        fn = getattr(lib, name, None)
        if fn is not None:
            fn.argtypes = argtypes
            if restype is not None:
                fn.restype = restype
    return lib


def arr(values) -> ctypes.Array:
    return (c_double * len(values))(*values)


class HybridIK:
    """One arm, configured the way the sponsors' daemon configures it."""

    def __init__(self, lib, urdf_text: bytes, ee: str, base: str = "arm_base",
                 starting_config=(0.0,) * 7, nominal=NOMINAL,
                 joint_centering_weight: float = JOINT_CENTERING_WEIGHT,
                 max_iter: int = MAX_ITER, tolerances=TOLERANCES,
                 centering_weights=CENTERING_WEIGHTS):
        self.lib, self.n, self.nominal = lib, len(starting_config), list(nominal)
        self.h = lib.relaxed_ik_new(urdf_text, base.encode(), ee.encode(),
                                    arr(starting_config), self.n,
                                    arr(nominal), len(nominal),
                                    joint_centering_weight, max_iter)
        if not self.h:
            raise RuntimeError("relaxed_ik_new returned NULL")
        lib.set_tolerances(self.h, arr(tolerances), len(tolerances))
        lib.set_centering_weights(self.h, arr(centering_weights), len(centering_weights))

    def __del__(self):
        if getattr(self, "h", None):
            self.lib.relaxed_ik_free(self.h)

    def reset(self, q) -> None:
        """Restart the tracker from a joint state.  This also overwrites the
        nominal (the sponsors note the same), so it is restored here."""
        self.lib.reset(self.h, arr(q), self.n)
        self.set_nominal(self.nominal)

    def set_nominal(self, q) -> None:
        self.lib.set_nominal_config(self.h, arr(q), len(q))

    def fk(self, q) -> tuple[list[float], list[float]]:
        """(xyz, quat xyzw) of the end effector in the base link frame."""
        out = self.lib.forward_kinematics(self.h, arr(q), self.n).tolist()
        return out[:3], out[3:7]

    def solve(self, pos, quat_xyzw) -> list[float]:
        """One tracker step toward the pose, from the last solution."""
        return self.lib.solve_pose(self.h, arr(pos), 3, arr(quat_xyzw), 4).tolist()


def _self_check(lib_path: str, urdf_path: str, ee: str) -> None:
    lib = load(lib_path)
    arm = HybridIK(lib, open(urdf_path, "rb").read(), ee)
    pos0, quat0 = arm.fk(NOMINAL)
    print(f"{ee} at nominal: pos {[round(v, 4) for v in pos0]}  quat xyzw {[round(v, 4) for v in quat0]}")
    arm.reset(NOMINAL)
    goal = [pos0[0] + 0.10, pos0[1] - 0.10, pos0[2] - 0.15]
    # teleop regime: the target walks over 40 ticks, 5 iterations each
    ticks = 40
    for i in range(1, ticks + 1):
        target = [a + (b - a) * i / ticks for a, b in zip(pos0, goal)]
        q = arm.solve(target, quat0)
    got, _ = arm.fk(q)
    err = sum((a - b) ** 2 for a, b in zip(got, goal)) ** 0.5
    print(f"tracked to +(0.10,-0.10,-0.15) in {ticks} ticks: q={[round(v, 3) for v in q]}  error {err * 1000:.1f} mm")
    assert err < 0.010, "tracker did not reach the target within tolerance"
    print("ok")


if __name__ == "__main__":
    _self_check(*sys.argv[1:4])
