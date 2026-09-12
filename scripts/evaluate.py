"""Headless check: does the controller keep it up, and for how long?

    .venv/bin/python scripts/evaluate.py                  # the BracketBot
    .venv/bin/python scripts/evaluate.py --robot toy      # the first-principles model
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlbot import Balancer, BalanceController, Gains   # noqa: E402

ROBOTS = {
    "bracketbot": (Balancer.bracketbot, Gains.for_bracketbot),
    "toy": (Balancer, Gains),
}


def rollout(gains: Gains, robot: str = "bracketbot", seconds: float = 20.0,
            verbose: bool = False, push: float = 0.0) -> dict:
    make_bot, _ = ROBOTS[robot]
    bot = make_bot()
    ctrl = BalanceController(gains)
    s = bot.reset()

    steps = int(seconds / bot.dt)
    push_step = int(steps / 2) if push else -1
    max_pitch = 0.0
    for i in range(steps):
        if i == push_step:
            # shove it in +x for 50 ms and see whether it recovers
            bot.data.body(bot.chassis).xfrc_applied[0] = push
        elif i == push_step + 25:
            bot.data.body(bot.chassis).xfrc_applied[0] = 0.0
        s = bot.step(*ctrl(s))
        max_pitch = max(max_pitch, abs(s.pitch))
        if bot.has_fallen(s):
            return {"survived": bot.data.time, "fell": True,
                    "max_pitch_deg": math.degrees(max_pitch), "drift_m": bot.data.qpos[0]}
        if verbose and i % 1000 == 0:
            print(f"  t={bot.data.time:5.1f}  pitch={math.degrees(s.pitch):6.2f}deg  "
                  f"wheel={s.wheel_speed:7.2f}rad/s  x={bot.data.qpos[0]:6.3f}m")

    return {"survived": bot.data.time, "fell": False,
            "max_pitch_deg": math.degrees(max_pitch), "drift_m": float(bot.data.qpos[0])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=sorted(ROBOTS), default="bracketbot")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--push", type=float, default=0.0,
                    help="newtons of forward shove at the halfway mark")
    args = ap.parse_args()

    _, make_gains = ROBOTS[args.robot]
    r = rollout(make_gains(), robot=args.robot, seconds=args.seconds,
                verbose=True, push=args.push)
    verdict = "FELL" if r["fell"] else "balanced"
    print(f"\n{verdict}: {r['survived']:.2f}s, max lean {r['max_pitch_deg']:.2f}deg, "
          f"drift {r['drift_m']:.3f}m")
    sys.exit(1 if r["fell"] else 0)
