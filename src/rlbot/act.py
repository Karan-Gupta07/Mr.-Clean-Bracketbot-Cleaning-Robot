"""ACT - Action Chunking with Transformers - on the recorded demonstrations.

The policy from Zhao et al. 2023, sized for a laptop.  Three cameras go
through a shared ResNet-18, the proprioceptive state through a linear layer,
and a transformer decodes a chunk of future joint commands from a fixed set
of queries.  Training uses a CVAE: an encoder looks at the whole action chunk
and squeezes it into a small latent `z`, so the decoder can explain the
variation between demonstrations instead of averaging it away; at inference
`z` is zero, the mean of the prior.

Dimensions are the recorder's.  State is both arms' seven joints and the
gripper blade angle, 16 numbers; action is the same 16 as servo commands.  A
demonstration only moves one arm, so half of every action is constant - the
policy learns to leave the other arm alone, which is what it should do.

Frames are cached on disk at a reduced size and memory-mapped.  At 224 px
a 100-episode set is 13 GB of pixels; at 128 px it is 4.4 GB, and a 50 mm
ball at 128 px is still 15 px across in the wrist camera.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from os import PathLike
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .orchestration import ToolResult
from .teleop import load_demo

CAMERAS = ("head_cam", "wrist_right_cam", "wrist_left_cam")
STATE_DIM = ACTION_DIM = 16
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---- data --------------------------------------------------------------
def state_and_action(a: dict) -> tuple[np.ndarray, np.ndarray]:
    """(rows, 16) measured state and (rows, 16) commanded action."""
    state = np.concatenate([a["arm_right"], a["grip_right_qpos"][:, None],
                            a["arm_left"], a["grip_left_qpos"][:, None]], axis=1)
    action = np.concatenate([a["arm_right_cmd"], a["grip_right"][:, None],
                             a["arm_left_cmd"], a["grip_left"][:, None]], axis=1)
    return state.astype(np.float32), action.astype(np.float32)


class Demos:
    """Every kept episode of one or more datasets, frames on disk at `img` px.

    The frames are decoded and downsized once into a `.npy` per episode under
    `cache/`, then memory-mapped: 280 episodes at 128 px is 14 GB, which does
    not fit in RAM beside the model but reads fine off an SSD one row at a
    time.  States and actions are small and stay in memory.
    """

    def __init__(self, items: list[tuple[Path, dict]], img: int = 128,
                 cache: Path | None = None, log=print):
        self.img = img
        self.frames, self.states, self.actions, self.ends = [], [], [], []
        for i, (root, e) in enumerate(items):
            meta, a = load_demo(root / e["episode"])
            cache_dir = cache or (root / "cache")
            cache_dir.mkdir(exist_ok=True)
            npy = cache_dir / f"{Path(e['episode']).stem}_{img}.npy"
            if not npy.exists():
                with np.load(root / e["frames"]) as z:
                    cams = torch.stack([torch.from_numpy(z[c]) for c in CAMERAS])
                cams = cams.permute(1, 0, 4, 2, 3).contiguous()      # rows, cam, 3, H, W
                rows = cams.shape[0]
                small = F.interpolate(cams.view(rows * 3, 3, *cams.shape[-2:]).float(),
                                      size=(img, img), mode="area")
                np.save(npy, small.round().to(torch.uint8).view(rows, 3, 3, img, img).numpy())
            self.frames.append(np.load(npy, mmap_mode="r"))
            s, act = state_and_action(a)
            self.states.append(torch.from_numpy(s))
            self.actions.append(torch.from_numpy(act))
            self.ends.append(len(s))
            if (i + 1) % 20 == 0:
                log(f"  loaded {i + 1}/{len(items)} episodes")
        self.rows = int(sum(self.ends))
        allstate = torch.cat(self.states)
        allact = torch.cat(self.actions)
        self.norm = {"state_mean": allstate.mean(0), "state_std": allstate.std(0).clamp(min=1e-2),
                     "action_mean": allact.mean(0), "action_std": allact.std(0).clamp(min=1e-2)}

    def batch(self, rng: np.random.Generator, size: int, chunk: int):
        """Random (episode, row) starts; the chunk is padded past the end."""
        eps = rng.integers(0, len(self.ends), size)
        imgs, states, acts, pads = [], [], [], []
        for e in eps:
            n = self.ends[e]
            t = int(rng.integers(0, n))
            imgs.append(torch.from_numpy(np.ascontiguousarray(self.frames[e][t])))
            states.append(self.states[e][t])
            a = self.actions[e][t:t + chunk]
            pad = chunk - len(a)
            if pad:
                a = torch.cat([a, a[-1:].expand(pad, -1)])
            acts.append(a)
            pads.append(torch.arange(chunk) >= chunk - pad)
        return (torch.stack(imgs), torch.stack(states), torch.stack(acts),
                torch.stack(pads))


def random_shift(imgs: torch.Tensor, pad: int, rng: np.random.Generator) -> torch.Tensor:
    """Shift every image by up to `pad` px, edge-padded: the standard cheap
    augmentation for image policies on small datasets.  Without it a policy
    trained on 70 demonstrations keys on exact pixel positions and misses by
    the width of its own confidence."""
    b, c = imgs.shape[:2]
    x = F.pad(imgs.flatten(0, 1).float(), (pad,) * 4, mode="replicate")
    out = torch.empty_like(imgs, dtype=torch.float32)
    h = imgs.shape[-1]
    for i in range(b * c):
        dy, dx = rng.integers(0, 2 * pad + 1, 2)
        out.view(b * c, *imgs.shape[2:])[i] = x[i, :, dy:dy + h, dx:dx + h]
    return out.round().to(torch.uint8)


def prep_images(imgs: torch.Tensor, device) -> torch.Tensor:
    """(B, cam, 3, H, W) uint8 -> float on the device, ImageNet-normalised."""
    x = imgs.to(device, non_blocking=True).float() / 255.0
    b, c = x.shape[:2]
    x = x.view(b * c, *x.shape[2:])
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return x.view(b, c, *x.shape[1:])


# ---- model -------------------------------------------------------------
def sinusoid_2d(h: int, w: int, dim: int) -> torch.Tensor:
    """(h*w, dim) fixed 2-D position embedding, half for y and half for x."""
    d = dim // 2
    freq = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    ys = torch.arange(h).float()[:, None] * freq
    xs = torch.arange(w).float()[:, None] * freq
    y = torch.cat([ys.sin(), ys.cos()], 1)          # (h, d)
    x = torch.cat([xs.sin(), xs.cos()], 1)          # (w, d)
    return torch.cat([y[:, None].expand(h, w, d), x[None].expand(h, w, d)], 2).reshape(h * w, dim)


def sinusoid_1d(n: int, dim: int) -> torch.Tensor:
    pos = torch.arange(n).float()[:, None]
    freq = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    out = torch.zeros(n, dim)
    out[:, 0::2] = (pos * freq).sin()
    out[:, 1::2] = (pos * freq).cos()
    return out


class ACT(nn.Module):
    def __init__(self, chunk: int = 32, hidden: int = 256, heads: int = 8,
                 enc_layers: int = 4, dec_layers: int = 4, ff: int = 1024,
                 latent: int = 32, cameras: int = 3, dropout: float = 0.1):
        super().__init__()
        self.chunk, self.hidden, self.latent = chunk, hidden, latent

        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])   # -> (512, H/32, W/32)
        self.proj = nn.Conv2d(512, hidden, 1)
        self.cam_embed = nn.Parameter(torch.zeros(cameras, 1, hidden))
        self.state_in = nn.Linear(STATE_DIM, hidden)
        self.latent_in = nn.Linear(latent, hidden)
        self.extra_pos = nn.Parameter(torch.zeros(2, hidden))          # latent, state
        self.register_buffer("img_pos", torch.zeros(0), persistent=False)

        layer = nn.TransformerEncoderLayer(hidden, heads, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, enc_layers)
        dlayer = nn.TransformerDecoderLayer(hidden, heads, ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dlayer, dec_layers)
        self.queries = nn.Parameter(torch.zeros(chunk, hidden))
        self.action_out = nn.Linear(hidden, ACTION_DIM)

        # CVAE encoder: [cls, state, action_1..K] -> z
        self.cvae_cls = nn.Parameter(torch.zeros(1, hidden))
        self.cvae_state = nn.Linear(STATE_DIM, hidden)
        self.cvae_action = nn.Linear(ACTION_DIM, hidden)
        self.register_buffer("cvae_pos", sinusoid_1d(chunk + 2, hidden), persistent=False)
        clayer = nn.TransformerEncoderLayer(hidden, heads, ff, dropout, batch_first=True)
        self.cvae = nn.TransformerEncoder(clayer, enc_layers)
        self.cvae_out = nn.Linear(hidden, 2 * latent)

        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.cam_embed, std=0.02)
        nn.init.normal_(self.extra_pos, std=0.02)
        nn.init.normal_(self.cvae_cls, std=0.02)

    def encode_latent(self, state, actions):
        b = state.shape[0]
        tokens = torch.cat([self.cvae_cls.expand(b, -1, -1),
                            self.cvae_state(state)[:, None],
                            self.cvae_action(actions)], 1) + self.cvae_pos
        out = self.cvae(tokens)[:, 0]
        mu, logvar = self.cvae_out(out).chunk(2, -1)
        return mu, logvar

    def features(self, imgs):
        """(B, cam, 3, H, W) -> (B, cam*h*w, hidden) with position embeddings."""
        b, c = imgs.shape[:2]
        f = self.proj(self.backbone(imgs.flatten(0, 1)))          # (B*cam, hidden, h, w)
        h, w = f.shape[-2:]
        if self.img_pos.numel() != h * w * self.hidden:
            self.img_pos = sinusoid_2d(h, w, self.hidden).to(f.device)
        f = f.flatten(2).transpose(1, 2).view(b, c, h * w, self.hidden)
        f = f + self.img_pos + self.cam_embed
        return f.flatten(1, 2)

    def forward(self, imgs, state, actions=None):
        """Predict a chunk.  With `actions`, train through the CVAE and return
        (pred, mu, logvar); without, decode from z = 0."""
        b = state.shape[0]
        if actions is not None:
            mu, logvar = self.encode_latent(state, actions)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            mu = logvar = None
            z = torch.zeros(b, self.latent, device=state.device)
        tokens = torch.cat([self.latent_in(z)[:, None] + self.extra_pos[0],
                            self.state_in(state)[:, None] + self.extra_pos[1],
                            self.features(imgs)], 1)
        memory = self.encoder(tokens)
        out = self.decoder(self.queries.expand(b, -1, -1), memory)
        return self.action_out(out), mu, logvar


def act_loss(pred, actions, pads, mu, logvar, kl_weight: float):
    l1 = (pred - actions).abs().mean(-1)
    l1 = (l1 * ~pads).sum() / (~pads).sum().clamp(min=1)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
    return l1 + kl_weight * kl, l1.detach(), kl.detach()


def save_checkpoint(path: Path, model, optim, step: int, norm: dict, config: dict,
                    best_val: float) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                "step": step, "norm": norm, "config": config, "best_val": best_val}, tmp)
    tmp.replace(path)           # never a half-written checkpoint on disk


def load_checkpoint(path: Path, device):
    return torch.load(path, map_location=device, weights_only=False)


def export_checkpoint(src: Path, dst: Path) -> None:
    """A copy fit for git: weights in half precision, no optimiser state.

    A training checkpoint is 263 MB, most of it AdamW moments.  The 22M
    weights are 44 MB in fp16, which `load_state_dict` widens back to fp32
    on load, and GitHub refuses files over 100 MB.
    """
    ck = torch.load(src, map_location="cpu", weights_only=False)
    slim = {"model": {k: v.half() if v.is_floating_point() else v
                      for k, v in ck["model"].items()},
            "step": ck["step"], "norm": ck["norm"], "config": ck["config"],
            "best_val": ck["best_val"]}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1))


# ---- the demo's integration point --------------------------------------
ACT_DESCRIPTION = (
    "ACT (Action Chunking with Transformers) for the red ball/box, station='ball'; not VLA."
)


class ACTUnavailableError(RuntimeError):
    pass


def _checkpoint_status(checkpoint: str | PathLike[str] | None) -> str:
    """How the supplied checkpoint stands, worded the same wherever we report it."""
    if checkpoint is None:
        return 'No ACT checkpoint was supplied.'
    if not Path(checkpoint).is_file():
        return f'ACT checkpoint unavailable: {checkpoint}.'
    return f'Checkpoint {checkpoint} cannot be used without the real ACT backend.'


def prepare_act(checkpoint: str | PathLike[str] | None = None,
                view: bool = False) -> Callable[[], ToolResult]:
    checkpoint_status = _checkpoint_status(checkpoint)
    raise ACTUnavailableError(
        f'{ACT_DESCRIPTION} Scaffold only: ACT backend is not implemented. '
        f'{checkpoint_status} Supply the real ACT implementation and a compatible checkpoint, '
        'then implement rlbot.act.prepare_act to return a callable producing ToolResult. '
        'Preflight stops before navigation or manipulation; '
        'no VLA, scripted, Fable, or Flybrain fallback is used.'
    )


def prepare_act_live(sim, checkpoint: str | PathLike[str] | None = None) -> Callable[[], ToolResult]:
    """ACT at station 'ball', on the one sim the whole demo drives. Scaffold only.

    This is the integration point for the real backend, so the contract is
    stated rather than implied. Once supplied, it must return an `execute()`
    that:

      * reads its observations out of `sim.data` - head and wrist camera
        renders, joint qpos, and the pose of body 'ball' - and nothing else;
      * writes the arm and gripper actuators through
        `rlbot.arm.Arm(sim.model, sim.data, side)`, whose `hold(qpos)`,
        `grip(amount)` and `grip_pos` set `sim.data.ctrl`;
      * advances physics ONLY through `sim.rig.step(n)`. Calling
        `mujoco.mj_step` directly skips the rig's per-step hooks, which are
        what pace the viewer and hold the parking brake on the base;
      * returns a `ToolResult(ok, details)`.

    The robot is already parked at the ball table when this is called, so a
    failure here has to happen before anything moves.
    """
    if not sim.parked or sim.station != 'ball':
        raise ValueError(
            "ACT runs at station 'ball' on a parked robot; the live sim is "
            f'parked={bool(sim.parked)} at station={sim.station!r}. Nothing was moved.'
        )
    raise ACTUnavailableError(
        f'{ACT_DESCRIPTION} Scaffold only: ACT backend is not implemented. '
        f'{_checkpoint_status(checkpoint)} Supply the real ACT implementation and a '
        'compatible checkpoint. Live sim: implement rlbot.act.prepare_act_live to return '
        'execute() driving sim.rig.step; until then the robot stays parked where it is, and '
        'no VLA, scripted, Fable, or Flybrain fallback is used.'
    )
