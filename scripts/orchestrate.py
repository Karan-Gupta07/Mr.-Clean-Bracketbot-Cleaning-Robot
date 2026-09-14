import argparse
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from rlbot.orchestration import Dispatcher, Recognition, Target, ToolResult, route_for, station_at
from rlbot.act import prepare_act


def prepare_fable(planner, view=False):
    if planner not in ('fable', 'sweep'):
        raise ValueError('Fable planner must be fable or explicitly offline sweep')
    if planner == 'fable':
        if importlib.util.find_spec('anthropic') is None:
            raise RuntimeError('Install requirements-agent.txt to run the Fable API planner')
        if not os.environ.get('ANTHROPIC_API_KEY'):
            raise RuntimeError('Set ANTHROPIC_API_KEY locally to run Fable, or explicitly select --planner sweep for offline skill validation')
    from agent import Harness, MODEL, Robot, fable_planner, sweep_planner, watch
    if planner == 'fable':
        import anthropic
        try:
            with anthropic.Anthropic(timeout=30., max_retries=0) as client:
                client.models.retrieve(MODEL)
        except anthropic.APIError as error:
            status = getattr(error, 'status_code', None)
            category = f'HTTP {status}' if isinstance(status, int) else type(error).__name__
            raise RuntimeError(f'Fable API/model preflight failed ({category}; model {MODEL}); verify the locally configured key and model access. No movement started.') from None

    def execute():
        robot = Robot('cubes')
        harness = Harness(robot)

        def job():
            if planner == 'sweep':
                sweep_planner(harness)
            else:
                try:
                    fable_planner(harness, 'cubes', 'high')
                except anthropic.APIError:
                    raise RuntimeError('Fable API request failed; verify API/model access. No fallback controller was used.') from None
        if view:
            watch(robot, job, 1.)
        else:
            job()
        loose = [n for n, item in robot.items.items() if item.graspable]
        placed = [n for n in loose if robot.where(n) == 'in the crate']
        return ToolResult(len(placed) == len(loose), dict(planner=planner, placed=placed,
                                                        expected=loose, calls=harness.calls))
    return execute


def prepare_flybrain(checkpoint, seed, view=False):
    import hashlib
    import torch
    from stable_baselines3 import PPO
    from rlbot.arm_env import ArmEnv, validate_arm_config

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise RuntimeError(f'Flybrain checkpoint missing: {checkpoint}; run scripts/train_arm.py first')
    report_path = checkpoint.parent/'validation/report.json'
    if not report_path.is_file():
        raise RuntimeError('Flybrain has no validation report; run scripts/run_arm.py --episodes 20 --seed 3000 --output out/rl/arm_observable/validation')
    report = json.loads(report_path.read_text())
    if (report.get('episodes',0) < 20 or report.get('successes') != report['episodes']
            or report.get('checkpoint_sha256') != hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
        raise RuntimeError('Flybrain checkpoint has not passed its recorded evaluation; use scripts/run_arm.py for diagnostics')
    torch.set_num_threads(2)
    policy = PPO.load(checkpoint, device='cpu')
    config = getattr(policy, 'arm_config', None)
    if not isinstance(config, dict):
        raise RuntimeError('Flybrain checkpoint is missing environment metadata')
    env = ArmEnv(gripper='padded', station='pick', history=config.get('history_length', 1),
                 motion_deadband=config.get('motion_deadband', 0.))
    validate_arm_config(config, env.configuration)
    validate_arm_config(report.get('environment'), env.configuration)

    def execute():
        import contextlib
        import mujoco.viewer
        try:
            observation, _ = env.reset(seed=seed)
            window = mujoco.viewer.launch_passive(env.model, env.data) if view else contextlib.nullcontext()
            with window as viewer:
                if viewer is not None:
                    viewer.cam.lookat[:] = env.to_world([-.26,-1.60,1.0])
                    viewer.cam.distance = 1.7
                    viewer.cam.azimuth = -60 + math.degrees(env.frame_yaw)
                    viewer.cam.elevation = -25
                    viewer.opt.geomgroup[3] = 0
                    env.model.vis.map.znear = .003 / env.model.stat.extent
                started = time.monotonic()
                for _ in range(env.horizon):
                    if viewer is not None and not viewer.is_running():
                        return ToolResult(False, {'reason':'viewer closed before completion'})
                    action, _ = policy.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = env.step(action)
                    if viewer is not None:
                        viewer.sync()
                        time.sleep(max(0, env.data.time-(time.monotonic()-started)))
                    if terminated or truncated:
                        break
            return ToolResult(info['success'], dict(seed=seed, **info))
        finally:
            env.close()
    return execute


def navigate_to(station, view=False):
    from navigate import drive
    from rlbot.navmap import OccupancyGrid
    result = drive(f'start-{station}', 'start', station, OccupancyGrid.from_room(), 10, view)
    details = asdict(result)
    details['touched'] = sorted(result.touched)
    return ToolResult(result.ok, details)


def main():
    parser = argparse.ArgumentParser(description='Route a supplied recognition to Fable, ACT, or Flybrain. No VLM or camera detector is used here.')
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--station', choices=['ball','cubes','pick'])
    target.add_argument('--spot', type=float, nargs=2, metavar=('X','Y'))
    parser.add_argument('--recognized', required=True, choices=[t.value for t in Target])
    parser.add_argument('--confidence', type=float, default=1.)
    parser.add_argument('--execute', action='store_true', help='Run navigation, then a separate fixed-base manipulation simulation')
    parser.add_argument('--planner', choices=['fable','sweep'], default='fable')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'out/rl/arm_observable/policy.zip')
    parser.add_argument('--act-checkpoint', type=Path,
                        help='ACT checkpoint (default checkpoints/act_ball_run1_noaug.pt)')
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--view', action='store_true')
    args = parser.parse_args()
    try:
        station = args.station or station_at(args.spot)
        now = time.monotonic()
        recognition = Recognition(Target(args.recognized), args.confidence, now)
        call = route_for(station, recognition, now=now)
        print(json.dumps(dict(call=asdict(call), recognition_source='user_provided_label',
                              simulation_mode='navigation_then_separate_fixed_base_manipulation'), indent=2), flush=True)
        if not args.execute:
            return 0
        dispatcher = Dispatcher(lambda station: navigate_to(station, args.view), {
            'run_fable':lambda: prepare_fable(args.planner, args.view),
            'run_flybrain':lambda: prepare_flybrain(args.checkpoint, args.seed, args.view),
            'run_act':lambda: prepare_act(args.act_checkpoint, args.view),
        })
        result = dispatcher.run(station, recognition, now=now)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, f'{error}\n')
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.ok else 1


if __name__ == '__main__':
    sys.exit(main())
