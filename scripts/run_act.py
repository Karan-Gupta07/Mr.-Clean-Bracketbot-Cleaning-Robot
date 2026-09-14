import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rlbot.act import ACT_DESCRIPTION, DEFAULT_CHECKPOINT, prepare_act


def main(argv=None):
    parser = argparse.ArgumentParser(description=ACT_DESCRIPTION)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--describe', action='store_true', help='Describe the tool without loading or running anything')
    mode.add_argument('--check', action='store_true', help='Preflight only: the checkpoint is found, nothing moves')
    parser.add_argument('--checkpoint', type=Path, help=f'ACT checkpoint (default {DEFAULT_CHECKPOINT}); no fallback policy')
    parser.add_argument('--view', action='store_true', help='Watch the episode (run with mjpython)')
    args = parser.parse_args(argv)
    if args.describe:
        checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
        print(json.dumps(dict(tool='run_act', station='ball', description=ACT_DESCRIPTION,
                              status='ready' if checkpoint.is_file() else 'no checkpoint',
                              checkpoint=str(checkpoint), backend_implemented=True,
                              ready=checkpoint.is_file(), fallback=None,
                              execution_started=False), indent=2))
        return 0
    try:
        execute = prepare_act(args.checkpoint, args.view)
        if not callable(execute):
            raise RuntimeError('ACT preflight must return an executable controller')
        if args.check:
            return 0
        result = execute()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, f'{error}\n')
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.ok else 1


if __name__ == '__main__':
    sys.exit(main())
