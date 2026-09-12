"""Headless check: does the controller keep it up, and for how long?

    .venv/bin/python scripts/evaluate.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlbot import Balancer, BalanceController, Gains   # noqa: E402


def rollout(gains: Gains, seconds: float = 20.0, verbose: bool = False) -> dict:
    bot = Balancer()
    ctrl = BalanceController(gains)
    s = bot.reset()

    steps = int(seconds / bot.dt)
    max_pitch = 0.0
    for i in range(steps):
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
    r = rollout(Gains(), seconds=20.0, verbose=True)
    verdict = "FELL" if r["fell"] else "balanced"
    print(f"\n{verdict}: {r['survived']:.2f}s, max lean {r['max_pitch_deg']:.2f}deg, "
          f"drift {r['drift_m']:.3f}m")
    sys.exit(1 if r["fell"] else 0)
