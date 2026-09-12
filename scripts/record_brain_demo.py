"""Record a synchronized robot/neurons replay as a standalone HTML + GIF.

No external web services or frontend dependencies are required to view it.
Neuron activity is from the policy that actually issued each recorded command.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from stable_baselines3 import PPO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rlbot.navigation import NavigationEnv
from rlbot.connectome import ConnectomeFeatures


def jpeg(pixels):
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("out/rl/connectome_final/policy.zip"))
    parser.add_argument("--graph", type=Path, default=Path("out/flywire/graph_512.npz"))
    parser.add_argument("--output", type=Path, default=Path("out/demo"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--goal", type=float, nargs=2, default=[0.85, -0.55])
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=20.0, help="Demo time limit; scored training evaluations use 15 seconds")
    args = parser.parse_args()
    torch.set_num_threads(2)
    graph_sha = hashlib.sha256(args.graph.read_bytes()).hexdigest()
    graph_meta = json.loads(args.graph.with_suffix(".json").read_text())
    if graph_sha != graph_meta["graph_sha256"]:
        raise ValueError("Graph checksum does not match its manifest")
    model = PPO.load(args.checkpoint, device="cpu")
    features = model.policy.features_extractor
    expected_sha = getattr(model, "graph_sha256", None)
    # Initial pilot checkpoints stored graph provenance in their training report.
    report_path = args.checkpoint.parent / "report.json"
    if expected_sha is None and report_path.exists():
        expected_sha = json.loads(report_path.read_text()).get("graph_sha256")
    if not isinstance(features, ConnectomeFeatures) or expected_sha != graph_sha:
        raise ValueError("Checkpoint and graph do not match")
    with np.load(args.graph, allow_pickle=False) as graph:
        saved_graph = features.adjacency.coalesce()
        np.testing.assert_array_equal(saved_graph.indices().cpu().numpy(), np.stack([graph["post"], graph["pre"]]))
        np.testing.assert_array_equal(saved_graph.values().cpu().numpy(), graph["weight"])
        np.testing.assert_array_equal(features.inputs.cpu().numpy(), graph["inputs"])
        np.testing.assert_array_equal(features.outputs.cpu().numpy(), graph["outputs"])
        positions = graph["positions"].copy()
        ids = graph["ids"].astype(str).tolist()
        categories = graph["super_class"].tolist()
        order = np.argsort(np.abs(graph["weight"]))[-500:]
        edges = np.stack([graph["pre"][order], graph["post"][order]], axis=1).tolist()
    # Center/scale uniformly; preserve relative anatomical geometry.
    positions = (positions - (positions.min(axis=0) + positions.max(axis=0)) / 2) / max(float(np.ptp(positions, axis=0).max()), 1)
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    env = NavigationEnv(room=True, horizon=max(1, round(args.seconds / 0.05)))
    obs, info = env.reset(seed=args.seed, options={"goal": args.goal, "yaw": args.yaw})
    env.bot.model.vis.map.znear = 0.005 / env.bot.model.stat.extent
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0.25, -0.2, 0.65]
    camera.distance, camera.azimuth, camera.elevation = 3.5, -135, -50
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    frames, gif_frames = [], []
    cumulative_reward = 0.0
    terminal = False
    args.output.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(env.bot.model, height=480, width=640) as renderer:
        for step in range(env.horizon + 1):
            action = model.predict(obs, deterministic=True)[0]
            activity = features.last_activity[0].cpu().numpy()
            if step % 2 == 0 or terminal:
                renderer.update_scene(env.bot.data, camera=camera, scene_option=options)
                # A visual marker only: it has no collision geometry.
                scene = renderer.scene
                if scene.ngeom < scene.maxgeom:
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.array([0.06, 0.06, 0.06]), np.array([*env.goal, 0.07]),
                                       np.eye(3).ravel(), np.array([0.1, 1.0, 0.7, 0.9], dtype=np.float32))
                    scene.ngeom += 1
                robot = renderer.render().copy()
                renderer.update_scene(env.bot.data, camera="head_cam", scene_option=options)
                head = renderer.render().copy()
                frames.append({"time": info["sim_time"], "robot": jpeg(robot), "head": jpeg(head),
                               "activity": activity.round(4).tolist(), "action": action.round(4).tolist(),
                               "xy": env.bot.data.qpos[:2].round(4).tolist(), "distance": info["distance"],
                               "speed": info["speed"], "pitch": info["pitch_deg"], "return": cumulative_reward,
                               "qpos": env.bot.data.qpos.tolist(), "terminal": terminal,
                               "action_applied": not terminal})
                sheet = Image.new("RGB", (1120, 560), "#080f1b")
                draw = ImageDraw.Draw(sheet)
                sheet.paste(Image.fromarray(robot), (0, 50))
                draw.text((22, 20), "NEURAL DRIVE  /  BracketBot + FlyWire", fill="#7deac6")
                draw.text((670, 30), f"{len(ids)} real neurons / artificial activity", fill="white")
                brain_panel = Image.new("RGB", (480, 410), "#080f1b")
                brain_draw = ImageDraw.Draw(brain_panel)
                plot_scale = min(430 / max(np.ptp(positions[:, 0]), 0.01),
                                 360 / max(np.ptp(positions[:, 1]), 0.01))
                for a, b in edges[:200]:
                    pa, pb = positions[a], positions[b]
                    brain_draw.line((240 + pa[0] * plot_scale, 205 + pa[1] * plot_scale,
                                     240 + pb[0] * plot_scale, 205 + pb[1] * plot_scale), fill="#143f43")
                for point, value in zip(positions, activity):
                    x, y = 240 + point[0] * plot_scale, 205 + point[1] * plot_scale
                    color = (int(40 + 100 * value), int(90 + 160 * value), int(150 + 95 * value))
                    brain_draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
                sheet.paste(brain_panel, (640, 50))
                draw.text((670, 470), f"t {info['sim_time']:.1f}s   goal {info['distance']:.2f}m   speed {info['speed']:.2f}m/s", fill="white")
                draw.text((670, 495), "Subset / privileged state / PD balance / recorded", fill="#91a4bd")
                gif_frames.append(sheet)
            if terminal:
                break
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            terminal = terminated or truncated
    report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "graph": graph_meta, "seed": args.seed, "goal": args.goal, "yaw": args.yaw,
              "time_limit_seconds": args.seconds,
              "final": info, "return": cumulative_reward, "frames": len(frames),
              "scene_sha256": hashlib.sha256((REPO / "models/room_scene.xml").read_bytes()).hexdigest()}
    payload = {"meta": report, "ids": ids, "positions": positions.round(5).tolist(), "categories": categories,
               "edges": edges, "frames": frames}
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    (args.output / "episode.json").write_text(serialized)
    template = (REPO / "demo/index.html").read_text(encoding="utf-8")
    (args.output / "index.html").write_text(template.replace("__DEMO_DATA__", serialized.replace("</", "<\\/")), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if gif_frames:
        gif_frames[0].save(args.output / "demo.gif", save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
        gif_frames[-1].save(args.output / "preview.png")
    print(f"Recorded {len(frames)} frames, success={info['success']}, distance={info['distance']:.3f} m", flush=True)
    print(f"Open {(args.output / 'index.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
