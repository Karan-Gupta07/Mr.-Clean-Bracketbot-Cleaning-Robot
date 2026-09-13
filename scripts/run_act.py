import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from rlbot.act import ACT_DESCRIPTION, prepare_act


def main(argv=None):
    parser = argparse.ArgumentParser(description=f'{ACT_DESCRIPTION} Scaffold only; backend unavailable.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--describe', action='store_true', help='Describe the scaffold, not a readiness or execution success')
    mode.add_argument('--check', action='store_true', help='Preflight only; exits nonzero while ACT is unavailable')
    parser.add_argument('--checkpoint', type=Path, help='Compatible ACT checkpoint; no default or fallback policy')
    parser.add_argument('--view', action='store_true', help='Reserved for the real ACT execution viewer; preflight opens no viewer')
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(dict(tool='run_act', station='ball', description=ACT_DESCRIPTION,
                              status='scaffold', backend_implemented=False, ready=False,
                              fallback=None, execution_started=False), indent=2))
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
