"""Launch existing navigation and manipulation viewers, not a replay or a new controller."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import getpass
import json
import os
from pathlib import Path
import sys
import time

import orchestrate
from rlbot.orchestration import Dispatcher, Recognition, Target, route_for


@contextmanager
def local_fable_key(prompt):
    if not prompt:
        yield
        return
    if not sys.stdin.isatty():
        raise RuntimeError('The secure API key prompt requires an interactive console')
    try:
        key = getpass.getpass('Rotated Anthropic API key (hidden, this process only): ').strip()
    except EOFError:
        raise RuntimeError('API key prompt cancelled; no movement started') from None
    if not key:
        raise RuntimeError('No API key entered; no movement started')
    previous = os.environ.get('ANTHROPIC_API_KEY')
    os.environ['ANTHROPIC_API_KEY'] = key
    try:
        yield
    except Exception as error:
        if key in str(error):
            raise RuntimeError('Live demo failed; credential-bearing error details were redacted') from None
        raise
    finally:
        if previous is None:
            os.environ.pop('ANTHROPIC_API_KEY', None)
        else:
            os.environ['ANTHROPIC_API_KEY'] = previous


def save_report(path, metadata, **result):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(metadata, **result), indent=2), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog='Requires an interactive desktop/OpenGL. Use .venv\\Scripts\\python.exe '
               'on Windows; MuJoCo passive viewers require mjpython on macOS.')
    parser.add_argument('--station', required=True, choices=['cubes', 'pick', 'ball'])
    parser.add_argument('--recognized', required=True, choices=[t.value for t in Target],
                        help='Explicit operator-supplied label, NOT camera recognition')
    parser.add_argument('--confidence', type=float, default=1.)
    parser.add_argument('--planner', choices=['fable', 'sweep'], default='fable',
                        help='Cubes only: Fable API (default), or explicitly offline skill validation')
    parser.add_argument('--checkpoint', type=Path,
                        help='Required for pick; checkpoint and its validation/report.json must pass existing preflight')
    parser.add_argument('--act-checkpoint', type=Path,
                        help='ACT scaffold checkpoint path; unavailable until the real backend is supplied')
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--prompt-api-key', action='store_true',
                        help='Read a replacement Fable API key from a secure local console prompt; never saved')
    parser.add_argument('--report', type=Path, help='Write non-secret demo status and results to JSON')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report the route only: no readiness check, API call, viewer, or movement')
    args = parser.parse_args(argv)
    if args.station == 'pick' and args.checkpoint is None:
        parser.error('pick requires an explicitly selected --checkpoint; no policy is selected automatically')
    if args.station != 'pick' and args.checkpoint is not None:
        parser.error('--checkpoint is only used by the pick/Flybrain route')
    if args.station != 'cubes' and args.planner == 'sweep':
        parser.error('--planner sweep is only offline skill validation for cubes, not a controller fallback')
    if args.station != 'ball' and args.act_checkpoint is not None:
        parser.error('--act-checkpoint is only used by the ball/ACT route')
    if args.prompt_api_key and (args.station != 'cubes' or args.planner != 'fable'):
        parser.error('--prompt-api-key is only used for the live Fable API planner')
    metadata = dict(station=args.station, planner=args.planner if args.station=='cubes' else None,
                    recognition_source='user_provided_label',
                    simulation_mode='navigation_then_separate_fixed_base_manipulation')

    try:
        now = time.monotonic()
        recognition = Recognition(Target(args.recognized), args.confidence, now)
        call = route_for(args.station, recognition, now=now)
        print(json.dumps(dict(call=asdict(call), recognized=args.recognized,
                              recognition_source='user_provided_label',
                              simulation_mode='navigation_then_separate_fixed_base_manipulation',
                              dry_run=args.dry_run), indent=2), flush=True)
        print('Supplied labels are NOT camera recognition; navigation uses simulator truth, not ROS localization.\n'
              'SEPARATE SIMULATION STAGES: navigation, then a new fixed-base manipulation simulation; '
              'NOT a continuous handoff.', flush=True)
        if call.tool == 'run_fable':
            print('Planner: Fable API (requires locally configured ANTHROPIC_API_KEY; no silent fallback).'
                  if args.planner == 'fable' else
                  'Planner: sweep = OFFLINE skill validation; NOT live Fable API.', flush=True)
        elif call.tool == 'run_flybrain':
            print(f'Flybrain checkpoint: {args.checkpoint}; seed: {args.seed}.\n'
                  'Existing preflight requires a checkpoint-bound perfect evaluation of at least 20 episodes '
                  'and matching checkpoint/report environment metadata; no bypass or scripted fallback.', flush=True)
        else:
            print('ACT is unavailable until controller code and a compatible checkpoint are supplied; no fallback.', flush=True)
        if args.dry_run:
            print('DRY RUN: route only, readiness NOT checked; no API call, viewer, or movement.', flush=True)
            save_report(args.report, metadata, status='dry_run', ready='not_checked')
            return 0

        print('LIVE MuJoCo: controller preflight must pass BEFORE navigation or either viewer starts.\n'
              'Stage 1 navigation window closes automatically at completion; closing it early aborts the route.\n'
              'Stage 2 opens a separate fixed-base simulation only after successful navigation.', flush=True)
        if call.tool == 'run_fable':
            print('The cubes window stays open after "done"; close it then to finish and print the result.\n'
                  'Use Ctrl+C in the terminal to interrupt the demo.', flush=True)
        elif call.tool == 'run_flybrain':
            print('The Flybrain window closes automatically when the episode ends; closing it early fails the episode.', flush=True)

        dispatcher = Dispatcher(lambda station: orchestrate.navigate_to(station, view=True), {
            'run_fable': lambda: orchestrate.prepare_fable(args.planner, view=True),
            'run_flybrain': lambda: orchestrate.prepare_flybrain(args.checkpoint, args.seed, view=True),
            'run_act': lambda: orchestrate.prepare_act(args.act_checkpoint, view=True),
        })
        save_report(args.report, metadata, status='awaiting_key' if args.prompt_api_key else 'preflight')
        with local_fable_key(args.prompt_api_key):
            save_report(args.report, metadata, status='running')
            now = time.monotonic()
            recognition = Recognition(Target(args.recognized), args.confidence, now)
            result = dispatcher.run(args.station, recognition, now=now)
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        save_report(args.report, metadata, status='failed', ok=False, reason=str(error))
        parser.exit(2, f'{error}\n')
    except KeyboardInterrupt:
        save_report(args.report, metadata, status='cancelled', ok=False)
        parser.exit(130, 'Live demo interrupted\n')
    save_report(args.report, metadata, status='completed', **asdict(result))
    print(json.dumps(asdict(result), indent=2), flush=True)
    return 0 if result.ok else 1


if __name__ == '__main__':
    sys.exit(main())
