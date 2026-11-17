"""One occupancy architecture, sized by a profile (no forked code).

A profile picks the existing ``model_size.PRESETS`` entry (backbone width and
GRU width), input scale, camera set, temporal frames, stale-camera age and
precision. ``build(profile)`` returns ``SceneNet(..., occ_sparse=True)``.

Weight compatibility, by knob:

- keeps weights loadable: ``scale`` (input res), ``frames`` (temporal
  window), ``stale_s``, ``precision``, ``keep``. The stem is convolutional, the
  lift is per pixel and the ray->voxel table is a rebuilt, non-persistent buffer.
- forces a retrain: ``preset`` (width/depth) and ``cams`` (the fused token width
  feeds the GRU). ``load_into`` refuses those with ``ProfileMismatch``. It never
  loads a partial state silently.

Memory figures from ``estimate`` are CPU forward-pass estimates (params plus
activations at batch 1, fp32 bytes scaled by precision). They are not a VRAM
measurement. The budget only holds once nvidia-smi on Desktop-IQU45, with
BeamNG running, shows it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from python.data.strip_writer import STRIP_ORDER

PRECISION_BYTES = {"fp32": 4, "fp16": 2, "int8": 1}


@dataclass(frozen=True)
class OccProfile:
    name: str
    preset: str
    scale: float
    cams: tuple[str, ...]
    frames: int
    stale_s: float
    precision: str
    keep: int = 4096  # sparse voxels kept per tick (current + carried memory)


PROFILES: dict[str, OccProfile] = {
    # 1080 Ti shares the card with BeamNG. Pascal (6.1) keeps fp32 compute
    # (``model_size.resolve_precision``), so fp16 here would save memory only, not
    # time. int8 needs TensorRT plus a calibration set and is not claimed.
    "1080ti": OccProfile("1080ti", "small", 1.0, tuple(STRIP_ORDER), 4, 0.5, "fp32", 4096),
    # Live fallback when 8 cams cannot refresh every tick: main+wide+narrow.
    "1080ti-lite": OccProfile("1080ti-lite", "tiny", 0.75, ("wide", "main", "narrow"), 4, 0.5, "fp32", 2048),
    # Michael's knob: drop narrow. Res cannot go above the recorded 192 px strip (scaled_layout caps scale at 1),
    # so the saved compute goes to keep, not pixels, until the recorder writes a taller main sector.
    "1080ti-mw": OccProfile("1080ti-mw", "small", 1.0, ("wide", "main"), 4, 0.5, "fp32", 4096),
    "high": OccProfile("high", "large", 1.0, tuple(STRIP_ORDER), 8, 0.5, "fp16", 16384),
}


class ProfileMismatch(ValueError):
    """A checkpoint from one profile cannot load into another without a retrain."""


def get(name: str) -> OccProfile:
    if name not in PROFILES:
        raise KeyError(f"no occ profile {name!r}; have {sorted(PROFILES)}")
    return PROFILES[name]


def build(profile: OccProfile | str) -> Any:
    from python.train.model_size import scene_net_kwargs
    from python.train.scene_net import SceneNet

    p = get(profile) if isinstance(profile, str) else profile
    net = SceneNet(cams=p.cams, scale=p.scale, cam_age=True, occ_sparse=True, **scene_net_kwargs(p.preset))
    net.occ_core.stale_s = float(p.stale_s)
    net.occ_core.keep = int(p.keep)
    net.occ_profile = p.name
    return net


def retrain_knobs(a: OccProfile, b: OccProfile) -> list[str]:
    return [k for k in ("preset", "cams") if getattr(a, k) != getattr(b, k)]


def load_into(state: dict, src: OccProfile, dst: OccProfile) -> Any:
    """Build ``dst`` and load ``state`` (saved under ``src``) strictly, or raise ``ProfileMismatch``."""
    bad = retrain_knobs(src, dst)
    if bad:
        raise ProfileMismatch(f"{src.name} -> {dst.name} changes {', '.join(bad)}: retrain, weights do not carry over")
    net = build(dst)
    net.load_state_dict(state, strict=True)
    return net


def estimate(profile: OccProfile | str) -> dict[str, Any]:
    """Params and a CPU-estimated inference memory for one tick at batch 1. Estimate only."""
    import torch

    from python.train.occ_sparse3d import FULL_GRID

    p = get(profile) if isinstance(profile, str) else profile
    net = build(p).eval()
    acts = 0

    def hook(_m: Any, _i: Any, out: Any) -> None:
        nonlocal acts
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if torch.is_tensor(t):
                acts = max(acts, t.numel())

    handles = [m.register_forward_hook(hook) for m in net.modules() if not list(m.children())]
    strip = torch.zeros(1, 3, int(net.canvas_h), int(net.canvas_w))
    ages = torch.zeros(1, 8)
    with torch.no_grad():
        pred, _ = net.forward_batch(strip, torch.tensor([0.1]), net.initial_state(), ages=ages)
        cur, st = net.occ_core.step(pred.occ_maps, 0.1)
        vox, st = net.occ_core.step(pred.occ_maps, 0.1, st)  # second tick: memory merged
        q = torch.arange(4096).reshape(1, -1)
        logits = net.occ_head(pred.occ_feat, q, vox)
    for h in handles:
        h.remove()
    core = net.occ_core
    # The per-tick candidate tensor (cams x H' x W' x top-k bins x C) is not a module output; count it explicitly.
    acts = max(acts, int(core.samples) * int(vox.feat.shape[-1]))
    params = sum(t.numel() for t in net.parameters())
    nbytes = PRECISION_BYTES[p.precision]
    # Weights + the largest live activation x4 (in/out/workspace margin). Not measured.
    est_mb = (params + 4 * acts) * nbytes / 2**20
    return {
        "profile": p.name, "params": int(params), "occ_params": int(sum(t.numel() for t in net.occ_head.parameters())
        + sum(t.numel() for t in net.occ_core.parameters())), "canvas": (int(net.canvas_h), int(net.canvas_w)),
        "frustum_voxels": int(core.frustum.numel()), "candidates_cap": int(core.samples),
        "active_voxels": int(torch.unique(cur.ids[cur.ids >= 0]).numel()), "stored": int((vox.ids >= 0).sum()),
        "full_grid": FULL_GRID,
        "logits": tuple(logits.shape), "est_infer_mb_cpu_estimate": round(est_mb, 1),
        # Temporal memory is ONE carried sparse set (+ GRU h), not K cached frames: cost does not grow with frames.
        "temporal_state_kb": round((int(p.keep) * (vox.feat.shape[-1] + 2) + int(net.gru_hidden)) * nbytes / 1024, 1),
    }
