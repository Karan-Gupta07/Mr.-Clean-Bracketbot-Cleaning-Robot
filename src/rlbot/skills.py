"""The robot as an agent sees it: a few typed skills and a symbolic scene.

The shape of this API is not arbitrary.  What the literature on LLM-driven
manipulation agrees on, and what this follows:

  * **Few verbs, rich observation.**  Code as Policies gives its planner two
    control primitives and eight perception queries.  Actuation verbs are where
    LLMs are weakest and skills are strongest; observation is the opposite.
  * **Object names are an enum, not a string.**  The largest hallucination class
    in embodied LLM agents is a confidently-requested object that is not in the
    scene, and corrective feedback does not reliably fix it.  Unknown names are
    rejected at the tool boundary with the list of real ones.
  * **The code picks the arm, not the model.**  Bimanual results are blunt about
    this: planners that let the LLM assign arms score near zero where the same
    LLM feeding a deterministic assigner scores near the ceiling.  `arm` here is
    a hint, and `choose_arm` is free to ignore it.
  * **Preconditions are checked in code and refused with a reason.**  Reaching
    for something while already holding something is a documented dominant
    failure mode, and it is trivially preventable here.
  * **Recovery variations live in the skill.**  LLMs re-sequence and re-target
    well and invent new low-level motion strategies badly, so "try again
    differently" is a parameter of `pick`, not something the model has to think
    up.

Every skill returns an `Outcome`: a boolean, a cause from a closed vocabulary,
and the scene as it now stands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .arm import Arm, Gripper, down_quat
from .grasp import (APPROACH, LIFT, Rig, StationDriver, hold_everything, in_hand,
                    move, plan_waypoints, squeeze, welded_at)
from .robot import ROOM
from .room import TABLES, grasp_pose

# Closed vocabulary.  An agent that sees a new cause string every failure cannot
# learn anything from it; one that sees the same six can.
CAUSES = ("object_not_found", "hand_full", "no_free_hand", "not_holding",
          "unreachable", "grasp_failed", "place_failed")

DROP_HEIGHT = 0.10     # m above the crate rim to open the fingers
GRASP_TRIES = 4        # wrist angles and hands `pick` works through itself
RETREAT_LIMIT = 1.0    # rad of arm travel allowed when backing out of the crate
REPLAN_TRAVEL = 4.0    # rad of arm travel a from-scratch placement may cost
RELEASE_OPEN = 0.30    # rad the jaws open to let go
RELEASE_SECONDS = 1.0  # s taken to open them
RETRY_CAP = 2          # times the harness will let the agent re-ask for the same thing

# The one spot on the table both arms can reach.  It exists because this robot's
# workspace is genuinely split: the objects sit to the robot's right, the crate
# 0.22-0.27 m to its left, and the arm that can pick most objects up is 54 mm
# short of the crate.  Putting something down here and picking it up with the
# other hand is the handover, and it is the whole reason the robot has two arms.
HANDOVER = "handover_spot"
HANDOVER_REACH = 0.32  # m in front of the mast, in the near row
HANDOVER_DROP = 0.03   # m to drop from when setting something down


@dataclass
class Outcome:
    ok: bool
    note: str
    cause: str | None = None
    scene: str = ""

    def __str__(self) -> str:
        head = self.note if self.ok else f"{self.note} [{self.cause}]"
        return f"{head}\n{self.scene}" if self.scene else head


@dataclass
class Robot:
    """One docked robot, one table, two hands.

    The base is welded at the docking pose by default.  The wheels are not
    driven anywhere and cannot roll: navigation is somebody else's job, and a
    balancing robot that reaches for something has to roll its wheels to stay
    up, which walks it off the docking pose its arms were planned against.
    `balancing=True` swaps in the station keeper instead, which leans to keep
    the mass over the axle - it survives a reach, but it drifts.
    """

    table_name: str
    balancing: bool = False
    on_step: object = None
    model: object = field(init=False)
    data: object = field(init=False)

    def __post_init__(self):
        self.table = next(t for t in TABLES
                          if t.name.split("_")[1] == self.table_name)
        x, y, self.dock_yaw = self.table.dock
        if self.balancing:
            self.model = mujoco.MjModel.from_xml_path(str(ROOM))
            self.data = mujoco.MjData(self.model)
            mujoco.mj_resetDataKeyframe(
                self.model, self.data,
                self.model.key(f"dock_{self.table_name}").id)
        else:
            self.model = welded_at(x, y, self.dock_yaw)
            self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        hold_everything(self.model, self.data)
        driver = StationDriver(self.model, self.data) if self.balancing else None
        self.rig = Rig(self.model, self.data, driver=driver, on_step=self.on_step)
        self.arms = {s: Arm(self.model, self.data, s) for s in ("right", "left")}
        self.hands = {s: Gripper(self.model, s) for s in ("right", "left")}
        self.holding: dict[str, str | None] = {"right": None, "left": None}
        self.grasp_quat: dict[str, object] = {"right": None, "left": None}
        self.items = {i.name: i for i in self.table.items}
        self.crate = next(n for n in self.items if n.startswith("crate"))
        self.rig.seconds(0.4)

    # ---- what the agent is allowed to know ------------------------------
    def where(self, name: str) -> str:
        """Symbolic, not coordinates.  Higher observation fidelity measurably
        hurts embodied LLM problem solving; `in the crate` is the useful fact."""
        for side, held in self.holding.items():
            if held == name:
                return f"held in the {side} hand"
        pos = self.data.body(name).xpos
        if name != self.crate and self.inside_crate(pos):
            return "in the crate"
        if pos[2] < 0.4:
            return "on the floor"
        return "on the table"

    def inside_crate(self, pos) -> bool:
        """Is that point within the crate's own walls?

        Against the crate's footprint, not a radius around its centre: the crate
        is 220 x 160 mm, so a radius tight enough to exclude the table calls
        half the crate's own interior 'outside'.
        """
        crate = self.data.body(self.crate)
        size = self.items[self.crate].size
        local = crate.xmat.reshape(3, 3).T @ (pos - crate.xpos)
        return bool(abs(local[0]) < size["l"] / 2 and abs(local[1]) < size["w"] / 2
                    and 0 <= local[2] < size["h"] + 0.05)

    def scene(self) -> str:
        lines = [f"objects on {self.table.name}:"]
        for name in self.items:
            kind = "crate" if name == self.crate else self.items[name].kind
            lines.append(f"  {name} ({kind}, {self.items[name].width * 1000:.0f} mm "
                         f"across) - {self.where(name)}")
        lines.append("hands: " + ", ".join(
            f"{s} {'holding ' + h if h else 'empty'}"
            for s, h in self.holding.items()))
        return "\n".join(lines)

    def out(self, ok, note, cause=None) -> Outcome:
        return Outcome(ok, note, cause, self.scene())

    # ---- arm assignment, decided here and not by the model ---------------
    def free_arms(self, name: str, hint: str | None) -> list[str]:
        """Free hands, nearest side first - and both, because side is only a
        preference and reachability is the thing that decides.

        `hint` is advisory.  An LLM asked to allocate two arms will either pick
        one and use it for everything or contradict itself across turns; what
        actually decides it - which hand is empty, which one can reach - is all
        here, so it is settled here.  The caller tries them in this order and
        takes the first that plans.
        """
        free = [s for s, held in self.holding.items() if held is None]
        if not free:
            return []
        # Nearest hand first.  The crate sits on the centreline where both arms
        # can reach it, so whichever hand picks something up can also put it
        # away - one arm carries a job from pick to place, and the two arms
        # split the table between them.
        order = ["left", "right"] if self.across(name) > 0 else ["right", "left"]
        if hint in free:
            order.remove(hint)
            order.insert(0, hint)
        return [s for s in order if s in free]

    def across(self, name: str) -> float:
        """How far something sits to the robot's left, in metres."""
        rot = self.data.body("root").xmat.reshape(3, 3)
        return float(rot[:, 1] @ (self.data.body(name).xpos
                                  - self.data.body("root").xpos))

    # ---- the skills ------------------------------------------------------
    def look(self) -> Outcome:
        return self.out(True, "looking at the table")

    def pick(self, obj: str, arm: str | None = None) -> Outcome:
        """Pick something up, trying a few ways before giving up.

        The retries are here rather than in the agent on purpose.  This gripper
        sits right at its reliability threshold - the same cube at the same spot
        grasps or does not depending on which IK branch the arm took - and what
        fixes that is a different wrist angle or the other hand, not a different
        plan.  Models re-sequence and re-target well; they do not invent new
        low-level motion strategies.  So `pick` owns the variations and reports
        one honest outcome.
        """
        if obj not in self.items:
            return self.out(False, f"no object called {obj!r}; there is "
                            f"{', '.join(self.items)}", "object_not_found")
        if obj in self.holding.values():
            return self.out(False, f"{obj} is already in a hand", "hand_full")
        if obj == self.crate:
            return self.out(False, f"the {obj} is furniture, not something to "
                            f"carry", "unreachable")
        sides = self.free_arms(obj, arm)
        if not sides:
            return self.out(False, "both hands are full", "no_free_hand")

        item = self.items[obj]
        forward, reversed_ = item.yaws, tuple(reversed(item.yaws))
        tries = [(side, yaws) for side in sides for yaws in (forward, reversed_)]
        reached = False

        for n, (side, yaws) in enumerate(tries[:GRASP_TRIES]):
            plan = plan_waypoints(self.model, mujoco.MjData(self.model),
                                  self.jaw_target(obj), item.width, yaws,
                                  self.dock_yaw, self.data.qpos.copy(),
                                  sides=(side,))
            if plan is None:
                continue
            reached = True
            _, side, used_yaw, opening, (above, on, up) = plan

            hand = self.arms[side]
            hand.grip(opening)
            self.rig.seconds(0.3)
            move(self.rig, hand, above.qpos, 1.6)
            move(self.rig, hand, on.qpos, 1.0)
            squeeze(self.rig, hand)
            move(self.rig, hand, up.qpos, 1.2)
            self.rig.seconds(0.8)

            if in_hand(self.model, self.data, obj, side, self.hands[side]):
                self.holding[side] = obj
                self.grasp_quat[side] = down_quat(self.dock_yaw + used_yaw)
                extra = "" if n == 0 else f" (on try {n + 1})"
                return self.out(True, f"picked up {obj} with the {side} "
                                f"hand{extra}")
            # Back out the way it came in.  Opening wide and sweeping to the
            # rest pose from down among the objects is how a failed grasp at one
            # cube knocks the next two onto the floor - and then the agent is
            # chasing a table it wrecked itself.
            hand.grip(opening)
            move(self.rig, hand, above.qpos, 1.0)

        if not reached:
            return self.out(False, f"cannot reach {obj} with either free hand",
                            "unreachable")
        return self.out(False, f"closed on nothing at {obj} in "
                        f"{min(len(tries), GRASP_TRIES)} tries", "grasp_failed")

    def place(self, into: str | None = None, arm: str | None = None) -> Outcome:
        into = into or self.crate
        if into not in self.items and into != HANDOVER:
            return self.out(False, f"nowhere called {into!r}; there is "
                            f"{', '.join(self.items)} and {HANDOVER}",
                            "object_not_found")
        side = arm if arm in self.holding and self.holding[arm] else next(
            (s for s, held in self.holding.items() if held), None)
        if side is None:
            return self.out(False, "neither hand is holding anything",
                            "not_holding")

        obj = self.holding[side]
        target = (self.spot() if into == HANDOVER
                  else self.free_spot_in(into))
        if target is None:
            return self.out(False, "there is no clear space to set anything down",
                            "place_failed")
        hand, gripper = self.arms[side], self.hands[side]

        # Note: do not re-squeeze here.  The obvious idea - take a fresh bite
        # before carrying, since the object settles in the jaw after the lift -
        # makes it strictly worse.  The stall detector sees a joint that is
        # already loaded, calls it contact immediately, and bites another
        # 0.20 rad into an object that is already pinched: 3 of 7 placements
        # became 0 of 7.
        opening = float(self.data.ctrl[hand.grip_act])

        # Sweep the wrist the way the pick does.  Holding the hand at one fixed
        # yaw over the crate is a much tighter ask than it looks: the arm is
        # already committed to whatever pose it grabbed the object in.
        # Carry at the angle it was picked up at, and only translate.
        #
        # There is no force closure here worth the name: the object sits in the
        # jaw held by two pads and gravity, and the moment the wrist turns it
        # falls straight out.  Re-solving the drop pose at a fresh wrist angle
        # loses the object every time, at any speed and any grip force - the
        # shortest of those carries still rotates the hand 1.4 rad.  Keeping the
        # grasp orientation and moving the hand in a straight line keeps it.
        #
        # The sweep stays as a fallback for when the crate is not reachable at
        # the angle the object was picked up at; it will often drop it.
        scratch = mujoco.MjData(self.model)
        here = self.data.qpos[hand.ik.qadr].copy()
        held_at = self.grasp_quat[side]
        over_quat = down_quat(self.dock_yaw) if held_at is None else held_at
        # Ordered by how little they disturb the grasp, not by convenience: a
        # wrist that turns 15 degrees on the way to the crate usually keeps the
        # object, one that turns 90 never does.
        angles = [math.radians(a) for a in
                  (0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 180)]
        over = None

        if held_at is not None:
            scratch.qpos[:] = self.data.qpos
            mujoco.mj_kinematics(self.model, scratch)
            got = hand.ik.solve(scratch,
                                gripper.site_target(target, held_at, opening),
                                held_at, seed=here)
            if got.ok:
                over, over_quat = (float(np.abs(got.qpos - here).sum()), got), held_at

        base = held_at if held_at is not None else down_quat(self.dock_yaw)
        for extra in ([] if over else angles):
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, np.array([math.cos(extra / 2), 0, 0,
                                               math.sin(extra / 2)]), base)
            scratch.qpos[:] = self.data.qpos
            mujoco.mj_kinematics(self.model, scratch)
            got = hand.ik.solve(scratch, gripper.site_target(target, quat, opening),
                                quat, seed=here)
            if not got.ok:
                continue
            travel = float(np.abs(got.qpos - here).sum())
            if over is None or travel < over[0]:
                over, over_quat = (travel, got), quat
        # Last resort: re-plan from scratch rather than from where the arm
        # happens to be.  A seeded solve inherits the arm's current
        # configuration, and after a few failed picks that configuration can be
        # one from which no small motion reaches the crate - the same placement
        # that works from a clean start reports `unreachable`.
        if over is None:
            for extra in angles:
                quat = np.zeros(4)
                mujoco.mju_mulQuat(quat, np.array([math.cos(extra / 2), 0, 0,
                                                   math.sin(extra / 2)]), base)
                scratch.qpos[:] = self.data.qpos
                mujoco.mj_kinematics(self.model, scratch)
                got = hand.ik.solve(scratch,
                                    gripper.site_target(target, quat, opening),
                                    quat)
                travel = float(np.abs(got.qpos - here).sum())
                if got.ok and travel < REPLAN_TRAVEL:
                    over, over_quat = (travel, got), quat
                    break

        if over is not None:
            travel, over = over
            move(self.rig, hand, over.qpos, max(3.0, 2.5 * travel))
        if over is None:
            note = f"the {side} arm cannot reach over {into}"
            if into != HANDOVER:
                note += (f"; it can reach {HANDOVER}, and the other hand can "
                         f"reach {into} from there")
            return self.out(False, note, "unreachable")

        # Let go gently.  Commanding the jaws 0.35 rad open in one step is a
        # flick: the blades accelerate off the object and throw it clear of the
        # crate.  Heavy things survive it, 50 g cubes do not.
        for i in range(int(RELEASE_SECONDS / self.model.opt.timestep)):
            hand.grip(opening + RELEASE_OPEN * (i + 1)
                      / int(RELEASE_SECONDS / self.model.opt.timestep))
            self.rig.step()
        self.rig.seconds(1.0)
        self.holding[side] = None
        self.grasp_quat[side] = None

        # Retreat straight up, and only up.  Anything that moves the open hand
        # sideways out of the crate takes what was just put in it with it: the
        # bowl landed cleanly inside, and retracing the path in was enough to
        # flick it onto the floor.  A vertical lift cannot sweep.
        lift = target + np.array([0, 0, 0.15])
        scratch.qpos[:] = self.data.qpos
        mujoco.mj_kinematics(self.model, scratch)
        away = hand.ik.solve(scratch,
                             gripper.site_target(lift, over_quat, opening),
                             over_quat, seed=self.data.qpos[hand.ik.qadr])
        # ...and only if it is the *same* arm pose lifted, not a fresh IK
        # branch.  A 0.15 m target has plenty of solutions, and one of them
        # turns the elbow inside out on the way: the object is in the crate, the
        # arm swings, and the object is 85 cm away on the floor.
        settled = self.data.qpos[hand.ik.qadr].copy()
        if away.ok and float(np.abs(away.qpos - settled).sum()) < RETREAT_LIMIT:
            move(self.rig, hand, away.qpos, 1.6)
        self.rig.seconds(0.5)

        landed = self.where(obj)
        if into == HANDOVER:
            return self.out(True, f"put {obj} down on {HANDOVER}")
        if into == self.crate and landed != "in the crate":
            return self.out(False, f"{obj} did not end up in the {into}; it is "
                            f"{landed}", "place_failed")
        return self.out(True, f"put {obj} {landed}")

    def home(self, arm: str | None = None) -> Outcome:
        """Arms back beside the mast - lifting clear of the table first.

        The rest pose is below and behind the table edge, so going straight to
        it from anywhere over the table drags the whole arm across the work
        surface.  Raise the carriage to the top of the rail before folding, and
        the arm comes back over the top of everything instead of through it.
        """
        for side in ([arm] if arm in self.arms else list(self.arms)):
            if self.holding[side]:
                continue                    # do not fold up with a full hand
            raised = self.data.qpos[self.arms[side].ik.qadr].copy()
            raised[0] = 0.0                 # carriage to the top of the rail
            move(self.rig, self.arms[side], raised, 1.2)
            move(self.rig, self.arms[side], self.tucked(side), 1.4)
        return self.out(True, "arms back at rest")

    # ---- helpers ---------------------------------------------------------
    def free_spot_in(self, crate_name: str) -> np.ndarray:
        """Where in the crate to drop this one, given what is already in it.

        Not always the middle.  Four cubes dropped on the same point land on
        each other and knock the earlier ones back out - the run reports four
        successful placements and the crate ends up with three.  The crate is
        220 x 160 mm and the objects are under 60 mm, so there is room to put
        each one down somewhere of its own.
        """
        crate = self.data.body(crate_name)
        size = self.items[crate_name].size
        rot = crate.xmat.reshape(3, 3)
        inside = [self.data.body(n).xpos for n in self.items
                  if n != crate_name and self.inside_crate(self.data.body(n).xpos)]

        best, best_score = None, -1.0
        span_l, span_w = size["l"] / 2 - 0.075, size["w"] / 2 - 0.055
        for along in np.linspace(-span_l, span_l, 3):
            for across in np.linspace(-span_w, span_w, 3):
                here = crate.xpos + rot[:, 0] * along + rot[:, 1] * across
                clear = min((float(np.linalg.norm(here[:2] - o[:2]))
                             for o in inside), default=9.9)
                # furthest from what is already in there, ties to the middle
                score = clear - 0.05 * (abs(along) + abs(across))
                if score > best_score:
                    best, best_score = here, score
        return best + np.array([0, 0, DROP_HEIGHT])

    def spot(self) -> np.ndarray | None:
        """A clear patch of table both arms can reach, for handing over.

        Searched rather than assumed: a fixed point in front of the mast lands
        either in front of the table edge or on top of whatever is already
        there.  Walk along the near row, keep the candidates that are clear of
        every other object and that both arms can plan a pick at, and take the
        one nearest the middle.
        """
        rot = self.data.body("root").xmat.reshape(3, 3)
        root = self.data.body("root").xpos
        table_z = self.data.body(self.crate).xpos[2]
        others = [self.data.body(n).xpos for n, held in
                  ((n, self.where(n)) for n in self.items)
                  if "hand" not in held]

        best = None
        for across in np.arange(-0.30, 0.301, 0.03):
            here = root + rot[:, 0] * HANDOVER_REACH + rot[:, 1] * across
            here[2] = table_z
            if any(np.linalg.norm(here[:2] - o[:2]) < 0.11 for o in others):
                continue
            if best is None or abs(across) < abs(best[0]):
                best = (across, here)
        if best is None:
            return None
        spot = best[1].copy()
        spot[2] = table_z + HANDOVER_DROP
        return spot

    def jaw_target(self, obj: str) -> np.ndarray:
        """Where the jaws go for this object, from where it is *now*.

        `grasp_pose` works off the table layout, which is only right until
        something gets moved.  Same height rule, live position.
        """
        item = self.items[obj]
        laid_out = grasp_pose(item, self.table)
        here = self.data.body(obj).xpos
        return np.array([here[0], here[1], here[2] + (laid_out[2] - 0.70)])

    def tucked(self, side: str) -> np.ndarray:
        """Arms down by the mast, out of the way of the table."""
        rest = np.zeros(len(self.arms[side].ik.qadr))
        rest[0] = -0.30        # the rail, 0.30 m down from the top
        return rest
