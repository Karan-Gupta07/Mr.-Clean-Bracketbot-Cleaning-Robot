"""Watch the PD controller balance the robot.

    .venv/bin/mjpython scripts/balance.py                 # the BracketBot
    .venv/bin/mjpython scripts/balance.py --robot toy     # the first-principles model

mjpython, not python: on macOS the window has to be created on the process main
thread.  launch_passive routes it there; mujoco.viewer.launch() does not and
fails with "Caught an unknown exception!".
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco.viewer                                    # noqa: E402
from rlbot import Balancer, BalanceController, Gains    # noqa: E402

ROBOTS = {
    "bracketbot": (Balancer.bracketbot, Gains.for_bracketbot),
    "toy": (Balancer, Gains),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=sorted(ROBOTS), default="bracketbot")
    args = ap.parse_args()

    make_bot, make_gains = ROBOTS[args.robot]
    bot = make_bot()
    ctrl = BalanceController(make_gains())
    state = bot.reset()

    with mujoco.viewer.launch_passive(bot.model, bot.data) as viewer:
        while viewer.is_running():
            tick = time.time()
            state = bot.step(*ctrl(state))
            viewer.sync()
            slack = bot.dt - (time.time() - tick)
            if slack > 0:
                time.sleep(slack)   # real time, not as-fast-as-possible


if __name__ == "__main__":
    main()
