"""Hand-tuned PD balance controller.

This is the baseline a learned policy has to beat, and the sanity check that the
model itself is physically sensible.  Cascade:

    wheel speed -> desired pitch -> wheel torque

The outer loop is what keeps it from driving off.  Wheels spinning forward mean
the robot is running away, and the way to stop a two-wheeler is to lean *back*:
the inner loop then rolls the wheels backward until the mass is over them again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .robot import State


@dataclass
class Gains:
    kp_pitch: float = 14.0       # torque per rad of lean
    kd_pitch: float = 0.50      # torque per rad/s of lean rate
    kp_speed: float = 0.010     # desired-pitch per rad/s of wheel speed
    kp_yaw: float = 0.05        # differential torque per rad/s of yaw rate
    max_pitch_ref: float = math.radians(12)

    @staticmethod
    def for_bracketbot() -> "Gains":
        """The real robot is 12 kg with its mass 0.63 m up: m*g*h = 74 N*m/rad of
        destabilising torque, so kp has to be an order of magnitude above the
        toy model's."""
        return Gains(kp_pitch=80.0, kd_pitch=15.0, kp_speed=0.010, kp_yaw=1.0)


class BalanceController:
    def __init__(self, gains: Gains | None = None):
        self.gains = gains or Gains()

    def __call__(self, s: State, yaw_rate_ref: float = 0.0,
                 speed_ref: float = 0.0) -> tuple[float, float]:
        """Wheel torques.  `speed_ref` is a wheel-speed target in rad/s: asking
        for one makes the outer loop hold a lean instead of cancelling it, which
        is how a two-wheeler drives anywhere."""
        g = self.gains

        # outer loop: wheels running faster than asked -> lean back to slow down
        pitch_ref = max(
            -g.max_pitch_ref,
            min(g.max_pitch_ref, -g.kp_speed * (s.wheel_speed - speed_ref)),
        )

        # inner loop: PD on lean.  +ve pitch (tipped toward +x) needs +ve wheel
        # torque, which rolls the wheels toward +x and drives the feet back
        # under the mass.
        torque = g.kp_pitch * (s.pitch - pitch_ref) + g.kd_pitch * s.pitch_rate

        # Positive yaw is counter-clockwise, which for this drive means the
        # right wheel pushing harder than the left.  The error has to be
        # (wanted - actual): the other way round is positive feedback, and with
        # kp_yaw = 1.0 it spun the robot onto the floor from any turn command
        # above about 0.4 rad/s while looking stable standing still.
        steer = g.kp_yaw * (yaw_rate_ref - s.yaw_rate)
        return torque - steer, torque + steer
