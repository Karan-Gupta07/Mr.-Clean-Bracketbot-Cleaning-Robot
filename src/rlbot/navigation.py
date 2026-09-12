"""Privileged-state point-goal RL pilot with the existing PD balancer.

The learned policy commands speed and yaw, not wheel torque. Position/attitude
come from MuJoCo in this pilot; this is not a SLAM or hardware-ready environment.
"""

from __future__ import annotations

import math
import gymnasium as gym
import mujoco
import numpy as np

from .robot import Balancer, BRACKETBOT, ROOM
from .control import BalanceController, Gains


class NavigationEnv(gym.Env):
    metadata = {"render_modes": []}
    max_speed = 0.35
    max_yaw = 0.8

    def __init__(self, room=False, horizon=300):
        self.bot = Balancer(ROOM if room else BRACKETBOT, chassis="root", keyframe="start" if room else "home")
        self.controller = BalanceController(Gains.for_bracketbot())
        self.radius = float(self.bot.model.geom("wheel_left_collision").size[0])
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(12,), dtype=np.float32)
        self.substeps = 25
        self.horizon = horizon
        self.goal = np.zeros(2)
        self.previous_action = np.zeros(2)
        self.command = np.zeros(2)
        self.steps = 0
        self.stopped = 0
        self.room = room

    @property
    def control_dt(self):
        return self.substeps * self.bot.dt

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.bot.reset()
        yaw = float(self.np_random.uniform(-math.pi, math.pi))
        self.bot.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.bot.model, self.bot.data)
        self.previous_action[:] = 0
        self.command[:] = 0
        self.steps = self.stopped = 0
        angle = yaw + self.np_random.uniform(-1.4, 1.4)
        distance = self.np_random.uniform(0.5, 1.2)
        self.goal = np.array([math.cos(angle), math.sin(angle)]) * distance
        if options and "goal" in options:
            self.goal = np.array(options["goal"], dtype=float)
        if options and "yaw" in options:
            yaw = float(options["yaw"])
            self.bot.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            mujoco.mj_forward(self.bot.model, self.bot.data)
        return self.observe(), self.info()

    def observe(self):
        state = self.bot.state()
        delta = self.goal - self.bot.data.qpos[:2]
        distance = float(np.linalg.norm(delta))
        heading = math.atan2(delta[1], delta[0]) - state.yaw
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        obs = [
            (c * delta[0] + s * delta[1]) / 2, (-s * delta[0] + c * delta[1]) / 2,
            distance / 2, math.sin(heading), math.cos(heading),
            state.pitch / 0.2, state.pitch_rate / 3, state.forward_speed / self.max_speed,
            state.yaw_rate / self.max_yaw, state.wheel_speed / 5,
            *self.previous_action,
        ]
        return np.clip(obs, -10, 10).astype(np.float32)

    def info(self):
        state = self.bot.state()
        return {"distance": float(np.linalg.norm(self.goal - self.bot.data.qpos[:2])),
                "pitch_deg": math.degrees(state.pitch), "speed": state.forward_speed,
                "success": self.stopped >= 10, "fell": self.bot.has_fallen(state),
                "sim_time": float(self.bot.data.time)}

    def teacher(self):
        state = self.bot.state()
        delta = self.goal - self.bot.data.qpos[:2]
        distance = float(np.linalg.norm(delta))
        error = (math.atan2(delta[1], delta[0]) - state.yaw + math.pi) % (2 * math.pi) - math.pi
        if distance < 0.09:
            return np.zeros(2, dtype=np.float32)
        speed = min(self.max_speed, 0.65 * max(0, distance - 0.045)) * max(0, math.cos(error)) ** 2
        yaw = np.clip(1.6 * error, -self.max_yaw, self.max_yaw)
        return np.array([speed / self.max_speed, yaw / self.max_yaw], dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected two finite actions")
        action = np.clip(action, -1, 1)
        old_distance = self.info()["distance"]
        target = action * [self.max_speed, self.max_yaw]
        self.command += np.clip(target - self.command, -np.array([0.5, 1.5]) * self.control_dt,
                                np.array([0.5, 1.5]) * self.control_dt)
        for _ in range(self.substeps):
            state = self.bot.state()
            self.bot.step(*self.controller(state, yaw_rate_ref=float(self.command[1]),
                                           wheel_speed_ref=float(self.command[0] / self.radius)))
            if self.bot.has_fallen():
                break
        self.steps += 1
        info = self.info()
        near_and_still = info["distance"] < 0.12 and abs(info["speed"]) < 0.06 and abs(self.bot.state().yaw_rate) < 0.2
        self.stopped = self.stopped + 1 if near_and_still else 0
        info["success"] = self.stopped >= 10
        progress = 20 * (old_distance - info["distance"])
        tilt_cost = 0.1 * (info["pitch_deg"] / 10) ** 2
        smooth_cost = 0.015 * float(np.square(action - self.previous_action).sum())
        reward = progress - 0.025 - tilt_cost - smooth_cost
        if near_and_still:
            reward += 0.2
        if info["success"]:
            reward += 10
        escaped = np.linalg.norm(self.bot.data.qpos[:2]) > 3.0
        if info["fell"] or escaped:
            reward -= 20
        info["reward_components"] = {"progress": progress, "tilt_cost": tilt_cost, "smooth_cost": smooth_cost}
        self.previous_action[:] = action
        terminated = bool(info["success"] or info["fell"] or escaped)
        truncated = bool(self.steps >= self.horizon and not terminated)
        return self.observe(), float(reward), terminated, truncated, info
