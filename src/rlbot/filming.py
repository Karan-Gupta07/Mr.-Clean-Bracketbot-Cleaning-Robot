"""Record a run to an mp4, by sampling the simulation as it steps."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np

FPS = 30
WIDTH, HEIGHT = 960, 720


class Recorder:
    """A callback the rig runs every step; renders every Nth and pipes to ffmpeg.

    Rendering is the expensive part of filming a MuJoCo run, so this samples at
    the video's own rate rather than the solver's - a 2 ms timestep is 500 frames
    a second and 30 of them are enough.
    """

    def __init__(self, path, fps: int = FPS, camera: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.camera = camera
        self.renderer = None
        self.every = 1
        self.count = 0
        self.frames: list[np.ndarray] = []

    def attach(self, model) -> None:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
        model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
        model.vis.headlight.ambient[:] = 0.55
        model.vis.headlight.diffuse[:] = 0.55
        self.model = model
        self.renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        self.every = max(1, round(1 / (self.fps * model.opt.timestep)))
        self.cam = mujoco.MjvCamera()
        if self.camera is not None:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = model.camera(self.camera).id

    def __call__(self, data) -> None:
        if self.renderer is None:
            return
        self.count += 1
        if self.count % self.every:
            return
        if self.camera is None:
            # Over the robot's shoulder, looking at the table it is working on.
            root = data.body("root")
            rot = root.xmat.reshape(3, 3)
            self.cam.lookat[:] = root.xpos + rot[:, 0] * 0.45 + np.array([0, 0, 0.35])
            self.cam.distance = 1.9
            self.cam.elevation = -22
            self.cam.azimuth = np.degrees(np.arctan2(rot[1, 0], rot[0, 0])) + 145
        self.renderer.update_scene(data, self.cam)
        self.frames.append(self.renderer.render())

    def close(self) -> str | None:
        if not self.frames:
            return None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise SystemExit("no ffmpeg on PATH - cannot write an mp4")
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(self.fps),
             "-i", "-", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "23", str(self.path)],
            stdin=subprocess.PIPE)
        for frame in self.frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
        seconds = len(self.frames) / self.fps
        print(f"wrote {self.path} ({seconds:.0f}s, {len(self.frames)} frames)")
        return str(self.path)
