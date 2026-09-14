"""One robot, one simulation: say what you want, watch it drive there and do it.

    .venv/bin/python scripts/demo.py "tidy the cubes table"                 # needs ANTHROPIC_API_KEY
    .venv/bin/python scripts/demo.py --planner sweep "go to the ACT table"  # no key: keyword routing
    .venv/bin/mjpython scripts/demo.py --view                               # window + prompt loop

Everything happens in one MuJoCo world that stays up between prompts, which is
the whole point.  scripts/agent.py welds the robot to a docking pose and never
drives; scripts/live_demo.py drives, then throws that simulation away and opens
a second, fixed-base one for the arms.  Here the robot balances, drives, parks,
works the table and is still standing there when the next prompt arrives.

A top-level Fable agent gets three tools - go_to, manipulate, finished - and the
room has three tables with a different manipulation tool bolted to each one:

  cubes  coloured cubes and a crate, tidied by the Fable skills agent that
         scripts/agent.py drives: an agent nested inside an agent.
  pick   a blue cube and a rectangle, worked by the Flybrain PPO policy.
  ball   a red ball and a box, put away by the ACT policy trained on the
         teleoperated demonstrations (checkpoints/, PR #10).

`manipulate` dispatches on where the robot is parked, and nothing in here ever
stands in for anything else: if a table's tool is unavailable, the tool result
says so and the agent decides what to do about it.  There is no fallback
controller, because a demo that quietly swaps in a different one is a demo of
the wrong thing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# The shipped Flybrain policy; its FlyWire graph sits next to it as checkpoints/graph_512.npz.
FLYBRAIN_CHECKPOINT = REPO / "checkpoints" / "flybrain_arm_padded_calibrated.zip"

from agent import (MODEL, STEP_BUDGET, Harness, Pacer,                # noqa: E402
                   fable_planner, sweep_planner)
from rlbot.act import (ACTUnavailableError, DEFAULT_CHECKPOINT,         # noqa: E402
                       prepare_act_live)
from rlbot.live import LiveSim                                        # noqa: E402
from rlbot.live_arm import prepare_flybrain_live                      # noqa: E402
from rlbot.orchestration import ToolResult                            # noqa: E402

TOP_BUDGET = 12              # top-level tool calls per prompt

# station -> (the manipulation tool that lives there, words that mean it).
# The aliases are only for the offline planner: with no model in the loop
# something still has to turn "tidy the cubes" into a station.
STATIONS = {
    "cubes": ("run_fable", ("cubes", "cube", "fable", "agent", "colored",
                            "coloured", "crate")),
    "ball": ("run_act", ("ball", "act", "red", "box")),
    "pick": ("run_flybrain", ("pick", "flybrain", "fly", "blue cube", "blue",
                              "rectangle")),
}

ALIAS_HELP = "; ".join(f"{station}: {', '.join(aliases)}"
                       for station, (_, aliases) in STATIONS.items())

# Longest first, so "blue cube" is the pick table rather than the cubes table.
# Python's alternation takes the first branch that matches at a position, so
# the ordering here *is* the tie-break.
_ALIASES = sorted(((alias, station)
                   for station, (_, aliases) in STATIONS.items()
                   for alias in aliases),
                  key=lambda pair: (-len(pair[0]), pair[0]))
_ALIAS_OF = dict(_ALIASES)
_ALIAS_RE = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(a) for a, _ in _ALIASES),
                       re.IGNORECASE)


def route_prompt(text: str) -> list[str]:
    """Stations named in a prompt, in the order they are named.

    Consecutive repeats collapse, because "pick up the blue cube" names the
    pick table twice and is still one trip.
    """
    order: list[str] = []
    for match in _ALIAS_RE.finditer(text):
        station = _ALIAS_OF[match.group(0).lower()]
        if not order or order[-1] != station:
            order.append(station)
    return order


TOP_TOOLS = [
    {
        "name": "go_to",
        "description": "Drive to a table and park at it. Plans a path, balances "
                       "the whole way, folds the arms first if they are out. "
                       "Slow: 30-90 seconds of simulated time per route.",
        "input_schema": {
            "type": "object",
            "properties": {
                "station": {"type": "string", "enum": list(STATIONS),
                            "description": "which table to park at"},
            },
            "required": ["station"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "manipulate",
        "description": "Run the manipulation tool belonging to the table the "
                       "robot is parked at. Only works after go_to has parked "
                       "it. You do not choose the controller; the table does.",
        "input_schema": {"type": "object", "properties": {},
                         "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "finished",
        "description": "The prompt is satisfied, or it is clear it cannot be.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

TOP_SYSTEM = """You are driving a two-armed balancing robot around a room with
three tables in it. The user says what they want; you get the robot there and
have it done.

The room, and what lives at each table:

  cubes  four coloured cubes and a crate. Tidying it hands off to a second
         agent that has the robot's own pick and place skills.
  ball   a red ball and a box. This is the ACT policy's table: `manipulate`
         runs the learned policy once, which tries to put the ball in the box
         and reports whether it did.
  pick   a blue cube and a rectangle. This is the Flybrain policy's table.

So the words in a prompt map onto stations: "the ACT table" or "the ACT
station" means go_to('ball'); "the Flybrain table" or "the blue cube" means
go_to('pick'); "the cubes" or "the crate" means go_to('cubes').

How this robot works, so you do not have to guess:

- `manipulate` only works once `go_to` has parked the robot at a table. It runs
  whatever tool belongs to that table and nothing else.
- Driving is slow: one route across this room is 30 to 90 seconds of simulated
  time. Do not call `go_to` again for a station you have already reached.
- If `go_to` fails, one retry is fair. If it fails twice, stop.
- If a table's tool reports that it is unavailable, that is the answer for
  that table. Nothing else in this room can do that job, so do not try another
  controller there; report it and carry on with whatever else the prompt asks.

Call `finished` as soon as the prompt is satisfied, or as soon as it is clear
that it cannot be."""


class Camera:
    """Points the live window at whatever is happening.

    The whole room while the robot drives - a table close-up shows a wall going
    past - and the table it is working once it parks.
    """

    def __init__(self, viewer, sim):
        self.viewer, self.sim = viewer, sim

    @property
    def running(self) -> bool:
        return self.viewer.is_running()

    def overview(self) -> None:
        self.viewer.cam.lookat[:] = (0, 0, 0.6)
        self.viewer.cam.distance = 7.5
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -65
        self.viewer.opt.geomgroup[3] = 0
        self.viewer.sync()

    def close_up(self) -> None:
        rot = self.sim.data.body("root").xmat.reshape(3, 3)
        self.viewer.cam.lookat[:] = (self.sim.data.body("root").xpos
                                     + rot[:, 0] * 0.45 + np.array([0, 0, 0.3]))
        self.viewer.cam.distance = 1.8
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = math.degrees(math.atan2(rot[1, 0], rot[0, 0])) + 150
        self.viewer.sync()


class Commander:
    """Runs the top-level tools, and owns the rules the model should not be
    trusted with: that manipulation needs a parked robot, that a route already
    driven is not driven again, and when to stop."""

    def __init__(self, sim, *, planner: str = "fable", effort: str = "high",
                 checkpoint=None, act_checkpoint=None, seed: int = 3000,
                 budget: int = TOP_BUDGET, camera: Camera | None = None, log=print):
        self.sim = sim
        self.planner, self.effort = planner, effort
        self.checkpoint, self.act_checkpoint, self.seed = checkpoint, act_checkpoint, seed
        self.budget, self.camera, self.log = budget, camera, log
        self.prompt = ""
        self.calls = 0
        self.budget_from = 0
        self.done = False
        self.summary = ""
        self.visited: list[str] = []
        self.results: list[dict] = []

    def start(self, prompt: str) -> None:
        """Begin a prompt: fresh budget, fresh stopping state.

        The sim is not reset and neither is the robot's position - it stays
        where the last prompt left it, which is what makes this continuous.
        """
        self.prompt = prompt
        self.budget_from = self.calls
        self.done, self.summary = False, ""

    @property
    def spent(self) -> bool:
        return self.calls - self.budget_from >= self.budget

    def run(self, name: str, args: dict) -> str:
        self.calls += 1
        self.log(f"  -> {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
        if name == "go_to":
            return self.go_to(args.get("station", ""))
        if name == "manipulate":
            return self.manipulate()
        if name == "finished":
            self.done = True
            self.summary = args.get("summary", "")
            return "acknowledged"
        return f"no tool called {name!r}"

    def go_to(self, station: str) -> str:
        if station not in STATIONS:
            return (f"no station called {station!r}; the tables are "
                    f"{', '.join(STATIONS)}")
        # A route is a minute of simulation.  Driving to where the robot is
        # already standing is the one mistake worth refusing outright.
        if self.sim.parked and self.sim.station == station:
            return f"already parked at the {station} table, nothing to drive"

        if self.camera:
            self.camera.overview()
        result = self.sim.drive_to(station)
        ok = bool(result.get("ok"))
        self.visited.append(station)
        self.results.append(dict(tool="go_to", station=station, ok=ok, details=result))
        self.log(f"     {'parked at' if ok else 'DID NOT REACH'} the {station} table")
        if ok and self.camera:
            self.camera.close_up()

        head = (f"parked at the {station} table."
                if ok else f"did not reach the {station} table: "
                           f"{result.get('failed') or 'the route failed'}.")
        return f"{head}\n{json.dumps(result, default=str)}"

    def manipulate(self) -> str:
        if not self.sim.parked:
            return ("the robot is not parked at a table, so there is nothing to "
                    "manipulate; call go_to first")
        station = self.sim.station
        tool = STATIONS[station][0]
        if station == "pick" and self.checkpoint is None:
            self.log("     unavailable: no --checkpoint for Flybrain")
            return ("no Flybrain checkpoint was given, so the pick table cannot "
                    "be worked; restart with --checkpoint <policy.zip>. Nothing "
                    "else here does this job.")
        try:
            if station == "cubes":
                result = self.tidy_cubes()
            elif station == "pick":
                result = prepare_flybrain_live(self.sim, self.checkpoint, self.seed)()
            else:
                result = prepare_act_live(self.sim, self.act_checkpoint)()
        except (ACTUnavailableError, RuntimeError, ValueError) as error:
            # A closed viewer arrives as a RuntimeError from the Pacer, from
            # deep inside whichever backend was running.  That is the run
            # ending, not a table refusing, so it goes up rather than back to
            # the model as a tool result.
            if self.camera and not self.camera.running:
                raise
            self.log(f"     unavailable: {error}")
            self.results.append(dict(tool=tool, station=station, ok=False,
                                     details=dict(reason=str(error))))
            return str(error)

        self.results.append(dict(tool=tool, station=station, ok=result.ok,
                                 details=result.details))
        self.log(f"     {'ok' if result.ok else 'FAILED'}: "
                 f"{json.dumps(result.details, default=str)}")
        return json.dumps(asdict(result), default=str)

    def tidy_cubes(self) -> ToolResult:
        """The nested agent: agent.py's harness and planners, on this sim.

        The robot is the one that just drove here, held at the dock by the
        parking brake, so the skills run against the same world the navigation
        left behind rather than a fresh welded copy of it.
        """
        robot = self.sim.robot()
        harness = Harness(robot, budget=STEP_BUDGET, log=self.log)
        if self.planner == "sweep":
            sweep_planner(harness)
        else:
            import anthropic
            try:
                fable_planner(harness, "cubes", self.effort)
            except anthropic.APIError:
                raise RuntimeError("Fable API request failed while tidying the "
                                   "cubes table; verify API/model access. No "
                                   "fallback controller was used.") from None
        loose = [n for n, item in robot.items.items() if item.graspable]
        placed = [n for n in loose if robot.where(n) == "in the crate"]
        return ToolResult(len(placed) == len(loose),
                          dict(tool="run_fable", planner=self.planner, placed=placed,
                               expected=loose, calls=harness.calls))


def fable_commander(commander: Commander, prompt: str, effort: str) -> None:
    """Claude Fable 5.1 deciding where to go and what to do when it gets there."""
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]

    while not commander.done and not commander.spent:
        # Fable 5.1, same as agent.py: thinking is always on so there is no
        # `thinking` parameter, forced tool choice is rejected, and a refusal
        # comes back as a normal 200 that the fallback parameter routes around.
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=8000,
            system=TOP_SYSTEM,
            tools=TOP_TOOLS,
            tool_choice={"type": "auto"},
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=messages,
        ) as stream:
            reply = stream.get_final_message()

        if reply.stop_reason == "refusal":
            commander.log(f"  model declined: {reply.stop_details}")
            return

        messages.append({"role": "assistant", "content": reply.content})
        calls = [b for b in reply.content if b.type == "tool_use"]
        for block in reply.content:
            if block.type == "text" and block.text.strip():
                commander.log(f"  model: {block.text.strip()[:300]}")
        if not calls:
            commander.log("  model stopped without calling a tool")
            return

        results = []
        for call in calls:
            out = commander.run(call.name, dict(call.input))
            results.append({"type": "tool_result", "tool_use_id": call.id,
                            "content": out})
        messages.append({"role": "user", "content": results})


def sweep_commander(commander: Commander, prompt: str) -> None:
    """No model: match table names in the prompt, then visit each one.

    The point of comparison, and the only way to drive the whole loop -
    navigation, docking, the nested table agents - on a machine with no API
    access.  It cannot infer anything the words do not say.
    """
    stations = route_prompt(prompt)
    if not stations:
        commander.log(f"  no table named in that prompt. Words I know - {ALIAS_HELP}")
        commander.run("finished", {"summary": "no table was named, so the robot "
                                              "stayed where it was"})
        return
    for station in stations:
        if commander.spent:
            break
        if "parked at" in commander.run("go_to", {"station": station}):
            commander.run("manipulate", {})
    commander.run("finished", {"summary": f"drove the route: {', '.join(stations)}"})


def banner(args) -> str:
    """What this demo is and is not, said once, before anything moves."""
    pick = ("pick runs Flybrain WITHOUT its validation gate"
            if args.checkpoint else "pick has no --checkpoint, so Flybrain refuses")
    act = Path(args.act_checkpoint or DEFAULT_CHECKPOINT).name
    return ("one continuous sim: navigation uses simulator truth, not SLAM; the base is "
            f"held by a parking brake while the arms work; {pick}; ball runs ACT from "
            f"{act}, one closed-loop episode per visit.")


def save_report(path, **status) -> None:
    """Non-secret run status: what was asked, what was called, what happened."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")


def ask(commander: Commander, prompt: str) -> None:
    commander.start(prompt)
    if commander.planner == "sweep":
        sweep_commander(commander, prompt)
    else:
        fable_commander(commander, prompt, commander.effort)
    if commander.summary:
        print(f"agent: {commander.summary}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="?",
                        help="what to do; omit it for a prompt loop on one sim")
    parser.add_argument("--planner", default="fable", choices=["fable", "sweep"],
                        help="drives both the commander and the cubes agent")
    parser.add_argument("--effort", default="high",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--checkpoint", type=Path,
                        default=FLYBRAIN_CHECKPOINT if FLYBRAIN_CHECKPOINT.exists() else None,
                        help=f"Flybrain policy for the pick table (default {FLYBRAIN_CHECKPOINT.name}); "
                             "without one that table refuses rather than falling back")
    parser.add_argument("--act-checkpoint", type=Path,
                        help=f"ACT checkpoint for the ball table (default {DEFAULT_CHECKPOINT.name})")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--view", action="store_true",
                        help="watch it live (run with mjpython, not python)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed for --view; 2 runs twice real time")
    parser.add_argument("--budget", type=int, default=TOP_BUDGET,
                        help="top-level tool calls per prompt")
    parser.add_argument("--report", type=Path, help="write non-secret run status here")
    parser.add_argument("--video", type=Path,
                        help="record the run to this .mp4/.mov, over the robot's shoulder (needs ffmpeg)")
    args = parser.parse_args(argv)

    if args.planner == "fable" and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.exit(2, "ANTHROPIC_API_KEY is not set. Export a key, or run with "
                       "--planner sweep to drive the same robot without a model.\n")

    print(banner(args), flush=True)
    recorder = None
    if args.video:
        if args.view:
            parser.error("--video records headless; drop --view")
        from rlbot.filming import Recorder
        recorder = Recorder(args.video)
    sim = LiveSim(on_step=recorder)
    if recorder is not None:
        recorder.attach(sim.model)
    commander = Commander(sim, planner=args.planner, effort=args.effort,
                          checkpoint=args.checkpoint, act_checkpoint=args.act_checkpoint,
                          seed=args.seed, budget=args.budget)

    viewer, prompts, code = None, [], 0
    try:
        if args.view:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            sim.rig.on_step = Pacer(viewer, sim.model, args.speed)
            commander.camera = Camera(viewer, sim)
            commander.camera.overview()

        if args.prompt:
            prompts.append(args.prompt)
            ask(commander, args.prompt)
        else:
            print('what should it do? empty line, "quit" or ctrl-D to stop', flush=True)
            while True:
                try:
                    prompt = input("> ").strip()
                except EOFError:
                    print()
                    break
                if not prompt or prompt in ("quit", "exit"):
                    break
                prompts.append(prompt)
                ask(commander, prompt)
        code = 0 if commander.results and all(r["ok"] for r in commander.results) else 1
    except RuntimeError as error:
        # The Pacer, when the window is closed: the run is over, and the report
        # still gets written.
        print(f"\nstopped: {error}", flush=True)
        code = 130
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
        code = 130
    finally:
        save_report(args.report, prompts=prompts, planner=args.planner,
                    calls=commander.calls, stations=commander.visited,
                    results=commander.results)
        if viewer is not None:
            viewer.close()
        if recorder is not None:
            recorder.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
