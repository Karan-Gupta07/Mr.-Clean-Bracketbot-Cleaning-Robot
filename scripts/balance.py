"""Watch the PD controller balance the robot.

    .venv/bin/mjpython scripts/balance.py

mjpython, not python: on macOS the window has to be created on the process main
thread.  launch_passive routes it there; mujoco.viewer.launch() does not and
fails with "Caught an unknown exception!".
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco.viewer                                    # noqa: E402
from rlbot import Balancer, BalanceController           # noqa: E402


def main() -> None:
    bot = Balancer()
    ctrl = BalanceController()
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
