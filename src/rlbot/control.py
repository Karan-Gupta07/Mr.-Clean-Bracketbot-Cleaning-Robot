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
        return Gains(kp_pitch=80.0, kd_pitch=15.0, kp_speed=0.010, kp_yaw=5.0)


class BalanceController:
    def __init__(self, gains: Gains | None = None):
        self.gains = gains or Gains()

    def __call__(self, s: State, yaw_rate_ref: float = 0.0,
                 wheel_speed_ref: float = 0.0) -> tuple[float, float]:
        g = self.gains

        # outer loop: wheels running forward -> ask for a backward lean
        pitch_ref = max(
            -g.max_pitch_ref, min(g.max_pitch_ref, -g.kp_speed * (s.wheel_speed - wheel_speed_ref))
        )

        # inner loop: PD on lean.  +ve pitch (tipped toward +x) needs +ve wheel
        # torque, which rolls the wheels toward +x and drives the feet back
        # under the mass.
        torque = g.kp_pitch * (s.pitch - pitch_ref) + g.kd_pitch * s.pitch_rate

        # Left wheel is at +y: more right-wheel forward torque turns +yaw.
        steer = g.kp_yaw * (yaw_rate_ref - s.yaw_rate)
        return torque - steer, torque + steer


class DriveController:
    """Bound and ramp velocity commands; stale commands return to zero-speed balance."""

    def __init__(self, wheel_radius: float, gains: Gains | None = None,
                 max_speed=0.15, max_yaw_rate=0.3, linear_accel=0.1,
                 angular_accel=0.3, timeout=0.5):
        if not all(math.isfinite(v) and v > 0 for v in
                   (wheel_radius, max_speed, max_yaw_rate, linear_accel, angular_accel, timeout)):
            raise ValueError("wheel radius, velocity/acceleration limits and timeout must be positive")
        self.balance = BalanceController(gains)
        self.radius, self.timeout = wheel_radius, timeout
        self.limits, self.acceleration = (max_speed, max_yaw_rate), (linear_accel, angular_accel)
        self.reference = [0.0, 0.0]
        self._target = [0.0, 0.0]
        self._stamp = None

    def command(self, speed: float, yaw_rate: float, stamp: float) -> None:
        if not all(math.isfinite(v) for v in (speed, yaw_rate, stamp)) or stamp < 0:
            self._target, self._stamp = [0.0, 0.0], None
            raise ValueError("velocity commands and timestamps must be finite; time must be nonnegative")
        self._target = [max(-limit, min(limit, value)) for limit, value in
                        zip(self.limits, (speed, yaw_rate))]
        self._stamp = stamp

    def __call__(self, state: State, now: float, dt: float) -> tuple[float, float]:
        if not math.isfinite(now) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("require finite time and a positive step")
        target = self._target if self._stamp is not None and 0 <= now - self._stamp <= self.timeout else (0, 0)
        for i in range(2):
            change = self.acceleration[i] * dt
            self.reference[i] += max(-change, min(change, target[i] - self.reference[i]))
        return self.balance(state, yaw_rate_ref=self.reference[1],
                            wheel_speed_ref=self.reference[0] / self.radius - state.pitch_rate)
