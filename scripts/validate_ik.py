"""Round-trip check of ArmIK, and the top-down reachability sweep from build_room.

    .venv/bin/python scripts/validate_ik.py [--n 60]

Part 1 draws random joint vectors inside the limits, runs forward kinematics to
get where the grip site actually is, then asks the solver to find that pose
again from a cold seed.  Every target is reachable by construction, so any
failure is the solver's.  The residual is recomputed from scratch off the
returned joints, not read back from the solver, so it cannot flatter itself.

Part 2 is build_room.check_reach: the same IK aimed at each object in the room
from its docking pose.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rlbot.arm import ArmIK                # noqa: E402
from rlbot.robot import BRACKETBOT, ROOM   # noqa: E402


def fk(model, data, ik, q):
    data.qpos[:] = model.qpos0
    data.qpos[ik.qadr] = q
    mujoco.mj_kinematics(model, data)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[ik.site])
    return data.site_xpos[ik.site].copy(), quat


def residual(pos, quat, got_pos, got_quat):
    neg, diff, w = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(neg, got_quat)
    mujoco.mju_mulQuat(diff, quat, neg)
    mujoco.mju_quat2Vel(w, diff, 1.0)
    return float(np.linalg.norm(pos - got_pos)), float(np.linalg.norm(w))


def round_trip(n: int, seed: int) -> bool:
    model = mujoco.MjModel.from_xml_path(str(BRACKETBOT))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    all_ok = True
    print(f"round trip: {n} random reachable poses per arm, cold seed (qpos0)")
    for side in ("right", "left"):
        ik = ArmIK(model, side)
        pe, re, ok, limit_bad, t = [], [], 0, 0, 0.0
        for i in range(n):
            q_true = rng.uniform(ik.lo, ik.hi)
            pos, quat = fk(model, data, ik, q_true)
            data.qpos[ik.qadr] = model.qpos0[ik.qadr]
            tick = time.perf_counter()
            sol = ik.solve(data, pos, quat, seed=model.qpos0[ik.qadr],
                           rng=np.random.default_rng(i))
            t += time.perf_counter() - tick
            got_pos, got_quat = fk(model, data, ik, sol.qpos)
            p, r = residual(pos, quat, got_pos, got_quat)
            pe.append(p); re.append(r)
            ok += sol.ok
            limit_bad += bool(np.any(sol.qpos < ik.lo - 1e-9) or np.any(sol.qpos > ik.hi + 1e-9))
            # the solver's own residual must agree with the independent one
            assert abs(p - sol.pos_err) < 1e-6 and abs(r - sol.rot_err) < 1e-6, \
                f"solver residual disagrees with FK: {sol.pos_err} vs {p}"
        pe, re = np.array(pe), np.array(re)
        print(f"  {side:>5}: ok {ok}/{n}   pos err median {np.median(pe)*1000:.2f} mm  "
              f"p90 {np.percentile(pe, 90)*1000:.1f} mm  max {pe.max()*1000:.1f} mm   "
              f"rot err median {math.degrees(np.median(re)):.2f} deg  "
              f"max {math.degrees(re.max()):.1f} deg   "
              f"limits violated {limit_bad}   {t/n*1000:.0f} ms/solve")
        # random joint-space targets include contorted poses nobody would
        # command; a stubborn miss or two out of 60 is normal for DLS
        all_ok &= ok >= 0.95 * n and limit_bad == 0
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ok = round_trip(args.n, args.seed)

    from build_room import check_reach     # noqa: E402  (scripts/ is on the path)
    problems = check_reach(mujoco.MjModel.from_xml_path(str(ROOM)))
    for p in problems:
        print("  PROBLEM", p)
    sys.exit(0 if ok and not problems else 1)


if __name__ == "__main__":
    main()
