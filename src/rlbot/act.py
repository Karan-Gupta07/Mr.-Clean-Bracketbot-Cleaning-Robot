from collections.abc import Callable
from os import PathLike
from pathlib import Path

from .orchestration import ToolResult


ACT_DESCRIPTION = (
    "ACT (Action Chunking with Transformers) for the red ball/box, station='ball'; not VLA."
)


class ACTUnavailableError(RuntimeError):
    pass


def prepare_act(checkpoint: str | PathLike[str] | None = None,
                view: bool = False) -> Callable[[], ToolResult]:
    if checkpoint is None:
        checkpoint_status = 'No ACT checkpoint was supplied.'
    elif not Path(checkpoint).is_file():
        checkpoint_status = f'ACT checkpoint unavailable: {checkpoint}.'
    else:
        checkpoint_status = f'Checkpoint {checkpoint} cannot be used without the real ACT backend.'
    raise ACTUnavailableError(
        f'{ACT_DESCRIPTION} Scaffold only: ACT backend is not implemented. '
        f'{checkpoint_status} Supply the real ACT implementation and a compatible checkpoint, '
        'then implement rlbot.act.prepare_act to return a callable producing ToolResult. '
        'Preflight stops before navigation or manipulation; '
        'no VLA, scripted, Fable, or Flybrain fallback is used.'
    )
