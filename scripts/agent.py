"""Let an agent tidy a table: Claude Fable 5.1 calling the robot's skills.

    .venv/bin/python scripts/agent.py --table cubes                 # needs an API key
    .venv/bin/python scripts/agent.py --table cubes --planner sweep # no key needed
    .venv/bin/python scripts/agent.py --table ware --video out/ware.mp4
    .venv/bin/mjpython scripts/agent.py --table ware --view          # watch it live

The robot is docked at one table and does not drive: the base is welded at the
docking pose, so the wheels never turn.  Navigation is a separate problem and
this is not it.

Two planners drive the same skills:

  fable   Claude Fable 5.1 over the Anthropic API, deciding what to do next from
          the scene and the outcome of its last call.  Needs ANTHROPIC_API_KEY.
  sweep   A fixed policy - pick each object, put it in the crate - with no model
          in the loop.  It exists so the skill layer can be tested, timed and
          filmed on a machine with no API access, and so a failure can be
          attributed to the robot rather than the agent.

The harness, not the model, enforces the loop rules that LLM agents are
documented to get wrong: a retry cap per object, a repeated-call detector, and a
step budget.  `give_up` is a real tool, because an agent with no way to stop
keeps calling the one skill that is failing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rlbot.room import TABLES                                        # noqa: E402
from rlbot.skills import RETRY_CAP, Robot                            # noqa: E402

MODEL = "claude-fable-5-1"
STEP_BUDGET = 40             # tool calls before the harness calls it a day

SYSTEM = """You are operating a two-armed robot that is parked at a table. Your
job is to put every loose object on that table into the crate.

How this robot works, so you do not have to guess:

- It cannot drive. The base is fixed. Everything happens within arm's reach.
- You do not choose which arm. Ask for `pick` and the robot picks the hand that
  can reach; `arm` is a hint it may ignore.
- The crate is furniture. Do not try to pick it up.
- `pick` already retries internally with different wrist angles and both hands.
  If it comes back `grasp_failed`, the same call again will not help much - two
  attempts on one object is plenty before you move on.
- Some objects this gripper cannot hold at all. A smooth ball is one. When you
  have evidence something is not going to work, say so with `give_up` and move
  on rather than burning attempts.

Work one object at a time: pick it, then put it in the crate, then the next.
Call `finished` when every object you can move is in the crate."""

TOOLS = [
    {
        "name": "look",
        "description": "Look at the table. Returns every object, where it is, "
                       "and what each hand is holding.",
        "input_schema": {"type": "object", "properties": {},
                         "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "pick",
        "description": "Pick up one object. Retries internally with different "
                       "wrist angles and both hands before reporting failure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object": {"type": "string",
                           "description": "the object's name, exactly as `look` "
                                          "reported it"},
                "arm": {"type": "string", "enum": ["left", "right"],
                        "description": "optional hint; the robot may use the "
                                       "other hand if that one cannot reach"},
            },
            "required": ["object"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "place",
        "description": "Put whatever is in the hand into the crate, or down on "
                       "handover_spot for the other hand to take.",
        "input_schema": {
            "type": "object",
            "properties": {
                "into": {"type": "string",
                         "description": "the crate's name, or handover_spot. "
                                        "Defaults to the crate."},
            },
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "home",
        "description": "Put the arms back at rest, out of the way.",
        "input_schema": {"type": "object", "properties": {},
                         "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "give_up",
        "description": "Declare one object impossible and stop trying it. Use "
                       "this rather than repeating a skill that keeps failing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object": {"type": "string"},
                "why": {"type": "string"},
            },
            "required": ["object", "why"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "finished",
        "description": "Everything that can be moved is in the crate.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class Pacer:
    """Drives a live viewer from inside the simulation loop.

    The rig calls this every step.  Syncing the window 500 times a second is
    wasted work, so it only refreshes at a frame rate, and sleeps the difference
    to keep the run at something like real time - otherwise the arm crosses the
    table in a blink and there is nothing to watch.  Unlike the recorder it
    holds no frames, which matters: filming the cubes table buffers about 3000
    of them before ffmpeg sees any.
    """

    def __init__(self, viewer, model, speed: float = 1.0, fps: int = 60):
        self.viewer, self.model, self.speed = viewer, model, speed
        self.every = max(1, round(1 / (fps * model.opt.timestep)))
        self.count = 0
        self.clock = time.time()

    def __call__(self, data) -> None:
        self.count += 1
        if self.count % self.every:
            return
        self.viewer.sync()
        ahead = self.every * self.model.opt.timestep / self.speed - (
            time.time() - self.clock)
        if ahead > 0:
            time.sleep(ahead)
        self.clock = time.time()


def watch(robot: Robot, run, speed: float) -> None:
    """Open a window on this robot, run the job in it, and leave it open."""
    import mujoco.viewer

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        rot = robot.data.body("root").xmat.reshape(3, 3)
        viewer.cam.lookat[:] = (robot.data.body("root").xpos
                                + rot[:, 0] * 0.45 + np.array([0, 0, 0.3]))
        viewer.cam.distance = 1.8
        viewer.cam.elevation = -20
        viewer.cam.azimuth = math.degrees(math.atan2(rot[1, 0], rot[0, 0])) + 150
        robot.rig.on_step = Pacer(viewer, robot.model, speed)
        viewer.sync()

        run()

        print("\ndone - the window stays open, close it to finish")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


class Harness:
    """Runs the skills, and owns the loop rules the model should not be trusted
    with: how many times one object may be retried, whether the same call is
    coming round again, and when to stop."""

    def __init__(self, robot: Robot, budget: int = STEP_BUDGET, log=print):
        self.robot = robot
        self.budget = budget
        self.log = log
        self.calls = 0
        self.tries: dict[tuple, int] = {}
        self.given_up: set[str] = set()
        self.done = False
        self.summary = ""

    def run(self, name: str, args: dict) -> str:
        self.calls += 1
        key = (name, json.dumps(args, sort_keys=True))
        self.log(f"  -> {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")

        if name == "finished":
            self.done = True
            self.summary = args.get("summary", "")
            return "acknowledged"
        if name == "give_up":
            self.given_up.add(args.get("object", ""))
            return (f"noted, {args.get('object')} left alone.\n"
                    f"{self.robot.scene()}")
        if args.get("object") in self.given_up:
            return (f"you already gave up on {args['object']}; leave it and move "
                    f"on\n{self.robot.scene()}")
        # Count failures, not calls - and only for calls that name an object.
        # `place()` takes no arguments, so keying on the arguments refuses the
        # third object's placement on the grounds that the first two calls
        # looked identical.  Looping is something agents do on one *object*.
        if args.get("object") and self.tries.get(key, 0) >= RETRY_CAP:
            return (f"{name} with those arguments has failed {RETRY_CAP} times "
                    f"already. Try something else, or give_up on it.\n"
                    f"{self.robot.scene()}")

        if name == "look":
            out = self.robot.look()
        elif name == "pick":
            out = self.robot.pick(args["object"], args.get("arm"))
        elif name == "place":
            out = self.robot.place(args.get("into"))
        elif name == "home":
            out = self.robot.home()
        else:
            return f"no tool called {name!r}"

        self.tries[key] = 0 if out.ok else self.tries.get(key, 0) + 1
        self.log(f"     {'ok' if out.ok else 'FAILED'}: {out.note}")
        return str(out)

    @property
    def spent(self) -> bool:
        return self.calls >= self.budget


def sweep_planner(harness: Harness):
    """No model: pick each object, put it in the crate, move on.

    The point of comparison.  Whatever this gets is what the robot can do; what
    an agent adds on top is judgement about what to retry and what to abandon.
    """
    robot = harness.robot
    harness.run("look", {})
    for name, item in robot.items.items():
        if not item.graspable or harness.spent:
            continue
        for _ in range(2):
            if "picked up" in harness.run("pick", {"object": name}):
                harness.run("place", {})
                break
        # Back to a known pose between objects.  Failed attempts leave the arm
        # in whatever configuration it stalled in, and the next placement has to
        # be planned out of that - which is how a crate the arm reached easily a
        # minute ago comes back `unreachable`.
        harness.run("home", {})
    harness.run("finished", {"summary": "swept the table"})


def fable_planner(harness: Harness, table: str, effort: str):
    """Claude Fable 5.1, deciding each next move from what just happened."""
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content":
                 f"Tidy the {table} table. Start by looking at it."}]

    while not harness.done and not harness.spent:
        # Fable 5.1: thinking is always on, so no `thinking` parameter; forced
        # tool choice is rejected, so `auto` plus an instruction; and a refusal
        # is a normal HTTP 200 that the fallback parameter routes around.
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=8000,
            system=SYSTEM,
            tools=TOOLS,
            tool_choice={"type": "auto"},
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=messages,
        ) as stream:
            reply = stream.get_final_message()

        if reply.stop_reason == "refusal":
            harness.log(f"  model declined: {reply.stop_details}")
            return

        messages.append({"role": "assistant", "content": reply.content})
        calls = [b for b in reply.content if b.type == "tool_use"]
        for block in reply.content:
            if block.type == "text" and block.text.strip():
                harness.log(f"  model: {block.text.strip()[:300]}")
        if not calls:
            harness.log("  model stopped without calling a tool")
            return

        results = []
        for call in calls:
            out = harness.run(call.name, dict(call.input))
            results.append({"type": "tool_result", "tool_use_id": call.id,
                            "content": out})
        messages.append({"role": "user", "content": results})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="cubes",
                    choices=[t.name.split("_")[1] for t in TABLES])
    ap.add_argument("--planner", default="fable", choices=["fable", "sweep"])
    ap.add_argument("--effort", default="high",
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--video", help="write an mp4 of the run here")
    ap.add_argument("--view", action="store_true",
                    help="watch it live (run with mjpython, not python)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed for --view; 2 runs twice real time")
    ap.add_argument("--budget", type=int, default=STEP_BUDGET)
    args = ap.parse_args()

    if args.planner == "fable" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Export a key, or run "
                         "with --planner sweep to drive the same skills without "
                         "a model.")

    recorder = None
    if args.video:
        from rlbot.filming import Recorder
        recorder = Recorder(args.video)

    print(f"table {args.table}, planner {args.planner}"
          f"{', recording' if recorder else ''}")
    robot = Robot(args.table, on_step=recorder)
    if recorder:
        recorder.attach(robot.model)
    harness = Harness(robot, budget=args.budget)

    def job():
        if args.planner == "sweep":
            sweep_planner(harness)
        else:
            fable_planner(harness, args.table, args.effort)

    started = time.time()
    if args.view:
        watch(robot, job, args.speed)
    else:
        job()

    loose = [n for n, i in robot.items.items() if i.graspable]
    crated = [n for n in loose if robot.where(n) == "in the crate"]
    print(f"\n{len(crated)}/{len(loose)} in the crate after {harness.calls} "
          f"tool calls, {time.time() - started:.0f}s")
    print(robot.scene())
    if harness.summary:
        print(f"\nagent: {harness.summary}")
    if recorder:
        recorder.close()


if __name__ == "__main__":
    main()
