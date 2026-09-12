"""Thin wrapper around the MuJoCo model: load it, read state, apply wheel torques."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "balancer.xml"


@dataclass
class State:
    """Everything a balance controller needs, in SI units / radians."""

    pitch: float          # lean about the wheel axis; +ve = tipped forward (+x)
    pitch_rate: float     # rad/s, from the gyro
    yaw: float            # heading, rad
    yaw_rate: float       # rad/s
    wheel_speed: float    # mean wheel spin, rad/s
    forward_speed: float  # chassis velocity along its own +x, m/s
    height: float         # chassis origin height, m — drops when it falls over


class Balancer:
    def __init__(self, model_path: Path | str = MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._imu = self.model.sensor("gyro").adr[0]
        self._vel_l = self.model.sensor("vel_left").adr[0]
        self._vel_r = self.model.sensor("vel_right").adr[0]
        self.reset()

    def reset(self, keyframe: str | None = "tipped") -> State:
        mujoco.mj_resetData(self.model, self.data)
        if keyframe is not None:
            mujoco.mj_resetDataKeyframe(
                self.model, self.data, self.model.key(keyframe).id
            )
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    @property
    def dt(self) -> float:
        return self.model.opt.timestep

    def state(self) -> State:
        d, chassis = self.data, self.data.body("chassis")
        R = chassis.xmat.reshape(3, 3)

        # body +z tilted within the world xz-plane -> pitch about the wheel axis
        pitch = math.atan2(R[0, 2], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])

        gyro = d.sensordata[self._imu : self._imu + 3]      # body frame
        forward_world = d.qvel[0:3]                          # free joint: world frame
        forward = float(R[:, 0] @ forward_world)

        return State(
            pitch=pitch,
            pitch_rate=float(gyro[1]),
            yaw=yaw,
            yaw_rate=float(gyro[2]),
            wheel_speed=float(
                (d.sensordata[self._vel_l] + d.sensordata[self._vel_r]) / 2
            ),
            forward_speed=forward,
            height=float(chassis.xpos[2]),
        )

    def step(self, torque_left: float, torque_right: float) -> State:
        lo, hi = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = np.clip([torque_left, torque_right], lo, hi)
        mujoco.mj_step(self.model, self.data)
        return self.state()

    def has_fallen(self, state: State | None = None) -> bool:
        s = state or self.state()
        return abs(s.pitch) > math.radians(45)
