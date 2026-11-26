"""Self-supervised stem pretraining (W4; Trainer 2.0 PR 5).

Pretext runs on unlabeled **train** frames only (never holdout). Saves stem
weights that ``train_scene --init-stem`` can load before supervised training.
Deployed perception stays camera-only.

Trainer 2.0 (docs/specs/trainer_2.md): SSL is **default ON for every
``train_scene`` run** (decided by Michael 2026-11-25). ``ssl_plan`` decides
per run and gives the reason when it is off (``--no-ssl-pretrain``, an
explicit ``--init-stem``, dry run, sanity, a pretrained backbone, or a
resume). ``default_ssl_stem`` writes one stem per content key under the cache
and reuses it. The pretext target is an EMA copy of the stem (momentum
``EMA_MOMENTUM`` = 0.996) and a collapse guard aborts when the mean per-dim
token std on a fixed train probe falls below ``COLLAPSE_STD`` = 0.01. Both
figures, the opt-out and the default row cap are spec proposals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

FORMAT = "gvd-ssl-stem-1"
EMA_MOMENTUM = 0.996
COLLAPSE_STD = 0.01
PROBE_N = 8
GUARD_EVERY = 200
DEFAULT_MAX_ROWS = 4000
# Bump when the pretext changes (mask ratio, target, guard, optimiser, code): it is part of
# the stem cache key, so a stem from the old recipe is never reused silently.
SSL_RECIPE = 1
MASK_RATIO = 0.25


class SslError(ValueError):
    """SSL row list or teacher/frame source is invalid."""


def require_row_kind(row: Mapping[str, Any]) -> str:
    if "kind" not in row or row["kind"] is None or str(row["kind"]).strip() == "":
        raise SslError(f"row i={row.get('i')!r} has no kind; refuse untagged rows")
    return str(row["kind"])


def select_ssl_rows(rows: Iterable[Mapping[str, Any]], holdout_kinds: set[str] | None = None) -> list[Mapping[str, Any]]:
    """Keep train rows after the #187 4 s purge. Untagged rows are refused."""
    from python.train.continual import HOLDOUT_GAP_S, _row_time, apply_holdout_time_gap

    held = set(holdout_kinds or {"holdout"})
    collected = list(rows)
    labels: list[str] = []
    for row in collected:
        kind = require_row_kind(row)
        if kind in held:
            labels.append("holdout")
        elif kind == "gap":
            labels.append("gap")
        else:
            labels.append("train")
    times = [_row_time(row) for row in collected]
    labels = apply_holdout_time_gap(
        labels,
        times,
        gap_s=HOLDOUT_GAP_S,
        hold_label="holdout",
    )
    return [row for row, lab in zip(collected, labels) if lab == "train"]


def ssl_rows_from_split(
    rows: list[Mapping[str, Any]],
    source_meta: Mapping[str, Any] | None = None,
    *,
    holdout_blocks: int = 1,
    length: int | None = None,
    stride: int | None = None,
) -> list[Mapping[str, Any]]:
    """Train frames from the #187 split, then ``select_ssl_rows``.

    Raw rows usually have no ``kind``. Kinds come from the plan windows.
    Holdout-window rows are included in the list passed to ``select_ssl_rows``
    so a mis-tagged train frame 1 s after a holdout timestamp is purged.
    """
    from python.train.continual import CHUNK_MAX, pass_plan
    from python.train.strip_sources import combined_pass_plan

    meta = dict(source_meta or {})
    chunk = CHUNK_MAX if length is None else int(length)
    step = (CHUNK_MAX // 2) if stride is None else int(stride)
    if meta.get("multi"):
        plan = combined_pass_plan(
            list(rows),
            length=chunk,
            stride=step,
            blocks=int(holdout_blocks),
            holdout_key=meta.get("holdout_key"),
        )
    else:
        plan = pass_plan(list(rows), chunk, step, blocks=int(holdout_blocks))
    tagged: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for item in plan:
        kind = str(item.get("kind") or "")
        if kind not in {"train", "holdout"}:
            continue
        for raw in list(item.get("rows") or []):
            row = dict(raw)
            row["kind"] = kind
            key = (row.get("source"), row.get("i"), kind)
            if key in seen:
                continue
            seen.add(key)
            tagged.append(row)
    return select_ssl_rows(tagged)


def frame_to_nchw(frame: Any, net: Any) -> Any:
    """Turn a training canvas into the NCHW batch ``_encode`` expects."""
    import torch

    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] == 3:
        ten = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        ten = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().unsqueeze(0)
    elif arr.ndim == 4 and arr.shape[1] == 3:
        ten = torch.from_numpy(np.ascontiguousarray(arr)).float()
    else:
        raise SslError(f"unsupported frame shape {getattr(arr, 'shape', None)}")
    if float(ten.max()) > 1.5:
        ten = ten / 255.0
    height = int(getattr(net, "canvas_h", ten.shape[-2]))
    width = int(getattr(net, "canvas_w", ten.shape[-1]))
    if int(ten.shape[-2]) != height or int(ten.shape[-1]) != width:
        ten = torch.nn.functional.interpolate(ten, size=(height, width), mode="bilinear", align_corners=False)
    return ten


def ema_update(target: Any, online: Any, momentum: float = EMA_MOMENTUM) -> None:
    """``target.stem <- m * target.stem + (1 - m) * online.stem`` (in place, no grad)."""
    import torch

    with torch.no_grad():
        for t_param, o_param in zip(target.stem.parameters(), online.stem.parameters()):
            t_param.mul_(float(momentum)).add_(o_param.detach(), alpha=1.0 - float(momentum))


def collapse_std(net: Any, images: list[Any]) -> float:
    """Mean over token dims of the std across the probe images (train mode left as it was)."""
    import torch

    with torch.no_grad():
        toks = [net._encode(img)[0].reshape(int(img.shape[0]), -1) for img in images]
    stack = torch.cat(toks, dim=0)
    if int(stack.shape[0]) < 2:
        return float("nan")
    return float(stack.std(dim=0, unbiased=False).mean())


def make_ema_target(net: Any) -> Any:
    """A frozen copy of ``net`` for the EMA pretext target."""
    import copy

    target = copy.deepcopy(net)
    for param in target.parameters():
        param.requires_grad_(False)
    return target


def run_ssl_pretrain(
    net: Any,
    rows: Iterable[Mapping[str, Any]],
    *,
    image_fn,
    out_path: Path | str,
    lr: float,
    seed: int,
    epochs: int = 1,
    ema: float = EMA_MOMENTUM,
    _after_step=None,
) -> dict[str, float]:
    """Masked-token pretext on the provided train rows only (no zero canvas).

    The target token comes from an EMA copy of the stem. The first
    ``PROBE_N`` (8) frames of the first epoch are the fixed probe; its
    reference std is taken with the initial weights. Every ``GUARD_EVERY``
    steps and at the end of each epoch the mean per-dim token std on it must
    stay >= ``COLLAPSE_STD``, or the run aborts and nothing is written. A run
    with fewer than ``PROBE_N`` frames, or whose probe is already below the
    floor at init (e.g. constant test canvases), cannot be guarded: that is
    reported in the stats (``probe_std_start`` NaN or < floor), not hidden.
    ``_after_step`` is a test hook called with ``net`` after each step.
    """
    import torch

    collected = list(rows)
    if not collected:
        raise SslError("no train frames after the #187 split")
    device = next(net.stem.parameters()).device
    opt = torch.optim.AdamW(net.stem.parameters(), lr=float(lr))
    rng = np.random.RandomState(int(seed))
    target = make_ema_target(net)
    init_net = make_ema_target(net)  # initial weights: the probe's reference std
    probe: list[Any] = []
    probe_start: float | None = None
    probe_now = float("nan")
    last = 0.0
    used = 0

    def guard() -> None:
        nonlocal probe_now
        if probe_start is None or not (probe_start >= COLLAPSE_STD):
            return
        probe_now = collapse_std(net, probe)
        if not probe_now >= COLLAPSE_STD:
            raise SslError(
                f"SSL collapse guard: probe token std {probe_now:.5f} < {COLLAPSE_STD} "
                f"(was {probe_start:.5f}) after {used} steps; no stem written"
            )

    for epoch in range(max(1, int(epochs))):
        for row in collected:
            image = image_fn(row)
            if image is None:
                continue
            if not hasattr(image, "shape") or not torch.is_tensor(image):
                image = frame_to_nchw(image, net)
            image = image.to(device)
            if epoch == 0 and init_net is not None and len(probe) < PROBE_N:
                probe.append(image.detach().clone())
                if len(probe) >= PROBE_N:
                    probe_start = collapse_std(init_net, probe)
                    init_net = None
            opt.zero_grad(set_to_none=True)
            loss = masked_token_pretext_loss(net, image, rng=rng, target=target)
            loss.backward()
            opt.step()
            ema_update(target, net, ema)
            if _after_step is not None:
                _after_step(net)
            last = float(loss.detach().item())
            used += 1
            if used % GUARD_EVERY == 0:
                guard()
        init_net = None
        guard()
    if used <= 0:
        raise SslError("no train-frame images loaded; refusing to pretrain on a zero canvas")
    if len(probe) >= PROBE_N:
        probe_now = collapse_std(net, probe)
    save_stem_weights(out_path, net)
    return {
        "last_loss": last,
        "n": float(used),
        "rows": float(len(collected)),
        "ema": float(ema),
        "probe_n": float(len(probe)),
        "probe_std_start": float("nan") if probe_start is None else float(probe_start),
        "probe_std_end": float(probe_now),
    }


def ssl_plan(args: Any, *, resuming: bool) -> tuple[bool, str]:
    """(on, reason) for the default SSL stage of one ``train_scene`` run.

    Raises ``ValueError`` for ``--holdout-cells``: the SSL row split
    (``ssl_rows_from_split``) is time-based, not cell-aware, so it could pull
    spatial-holdout frames into the pretext.
    """
    if bool(getattr(args, "no_ssl_pretrain", False)):
        return False, "off (--no-ssl-pretrain)"
    if str(getattr(args, "init_stem", "") or ""):
        return False, "off (--init-stem given; that stem is used)"
    if bool(getattr(args, "dry_run", False)):
        return False, "off (dry run)"
    if bool(getattr(args, "sanity", False)):
        return False, "off (sanity check run)"
    if str(getattr(args, "backbone", "scratch") or "scratch") != "scratch":
        return False, f"off (backbone {getattr(args, 'backbone', '')!r} has no SharedStem)"
    if resuming:
        return False, "off (resume: the stem is already trained in the pause file)"
    if bool(getattr(args, "holdout_cells", False)):
        raise ValueError(
            "default SSL is not cell-aware; --holdout-cells needs --no-ssl-pretrain "
            "(or an --init-stem built without the held-out cells)"
        )
    return True, "on (default)"


def _frame_digest(image: Any) -> str:
    """sha256 over one canvas as the pretext reads it (dtype, shape, bytes); a missing frame has its own digest."""
    import hashlib

    if image is None:
        return "none"
    if hasattr(image, "detach"):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    arr = np.ascontiguousarray(arr)
    digest = hashlib.sha256()
    digest.update(f"{arr.dtype.str}{arr.shape}".encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def content_hashes(rows: list[Mapping[str, Any]], image_fn) -> dict[str, str]:
    """Per source: one sha256 over its rows' frame digests, in row order.

    Same ids with different pixels (a re-export in place) give a different hash.
    """
    import hashlib

    per_source: dict[str, Any] = {}
    for row in rows:
        src = str(row.get("source") or "")
        digest = per_source.setdefault(src, hashlib.sha256())
        digest.update(f"{int(row['i'])}:{_frame_digest(image_fn(row))};".encode("ascii"))
    return {src: digest.hexdigest() for src, digest in sorted(per_source.items())}


def _even_subset(rows: list[Mapping[str, Any]], max_rows: int) -> list[Mapping[str, Any]]:
    if int(max_rows) <= 0 or len(rows) <= int(max_rows):
        return list(rows)
    idx = np.linspace(0, len(rows) - 1, int(max_rows)).round().astype(int)
    return [rows[int(k)] for k in sorted(set(idx.tolist()))]


def default_ssl_stem(
    net: Any,
    rows: list[Mapping[str, Any]],
    *,
    image_fn,
    out_dir: Path | str,
    lr: float,
    seed: int,
    epochs: int = 1,
    max_rows: int = DEFAULT_MAX_ROWS,
    key_extra: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Default-on SSL stem for a supervised run: ``(stem path, info)``.

    ``rows`` must be train rows (``ssl_rows_from_split``); any other kind is
    refused. ``max_rows`` evenly spaced rows are used (0 = all). The file is
    ``ssl_stem_<key>.npz``, keyed by the row list, the settings, ``SSL_RECIPE``
    and a hash of the frame content (read once per picked row, per source); an
    existing file with that key is reused, never overwritten, and no stem is deleted.
    """
    import hashlib
    import json
    import os

    bad = [r for r in rows if str(r.get("kind", "train")) != "train"]
    if bad:
        raise SslError(f"default SSL got {len(bad)} non-train row(s), first i={bad[0].get('i')!r}")
    picked = _even_subset(list(rows), int(max_rows))
    key_doc = {
        "format": FORMAT,
        "rows": [[str(r.get("source") or ""), int(r["i"])] for r in picked],
        "lr": float(lr),
        "seed": int(seed),
        "epochs": int(epochs),
        "max_rows": int(max_rows),
        "ema": EMA_MOMENTUM,
        "recipe": SSL_RECIPE,
        "mask_ratio": MASK_RATIO,
        "collapse_std": COLLAPSE_STD,
        "probe_n": PROBE_N,
        "guard_every": GUARD_EVERY,
        "content": content_hashes(picked, image_fn),
        "extra": dict(key_extra or {}),
    }
    key = hashlib.sha256(json.dumps(key_doc, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / f"ssl_stem_{key}.npz"
    info: dict[str, Any] = {"key": key, "rows": len(picked), "reused": False}
    if final.is_file():
        load_stem_into_net(net, final)
        info["reused"] = True
        return final, info
    tmp = folder / f"ssl_stem_{key}.partial.npz"
    try:
        stats = run_ssl_pretrain(net, picked, image_fn=image_fn, out_path=tmp, lr=lr, seed=seed, epochs=epochs)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()
    info.update(stats)
    return final, info


def save_stem_weights(path: Path | str, net: Any) -> None:
    import json

    import numpy as np

    path = Path(path)
    weights = {name: tensor.detach().cpu().numpy() for name, tensor in net.stem.state_dict().items()}
    meta = {"format": FORMAT, "keys": list(weights.keys())}
    np.savez(path, meta=np.array(json.dumps(meta, sort_keys=True)), **{f"w{i}": weights[k] for i, k in enumerate(meta["keys"])})


def load_stem_into_net(net: Any, path: Path | str) -> None:
    import json

    import numpy as np
    import torch

    path = Path(path)
    with np.load(path, allow_pickle=False) as blob:
        meta = json.loads(str(blob["meta"].item() if blob["meta"].shape == () else blob["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path.name}: not a {FORMAT} file")
        keys = list(meta.get("keys") or [])
        weights = {key: np.array(blob[f"w{i}"], copy=True) for i, key in enumerate(keys)}
    current = net.stem.state_dict()
    missing = [k for k in current if k not in weights]
    extra = [k for k in weights if k not in current]
    if missing or extra:
        raise KeyError(f"stem mismatch extra={extra[:3]} missing={missing[:3]}")
    mapped = {k: torch.tensor(weights[k], dtype=current[k].dtype, device=current[k].device) for k in current}
    net.stem.load_state_dict(mapped)


def masked_token_pretext_loss(
    net: Any,
    image: Any,
    *,
    mask_ratio: float = MASK_RATIO,
    rng: np.random.RandomState | None = None,
    target: Any = None,
) -> Any:
    """Predict the stem token of a masked canvas from the unmasked stem token.

    Simple contrastive/masked stand-in: zero a random spatial block on a clone,
    encode both, and pull the tokens together with L1. Train rows only at the
    caller. Gradients update the stem (and only the stem if other params are frozen).
    ``target`` (an EMA copy of the net) encodes the full canvas when given;
    without it the same net, stop-grad, is the target (no EMA).
    """
    import torch

    rng = rng or np.random.RandomState(0)
    if target is None:
        token_full, _feat = net._encode(image)
    else:
        with torch.no_grad():
            token_full, _feat = target._encode(image)
    masked = image.clone()
    _b, _c, height, width = masked.shape
    mh = max(1, int(height * float(mask_ratio)))
    mw = max(1, int(width * float(mask_ratio)))
    y0 = int(rng.randint(0, max(1, height - mh + 1)))
    x0 = int(rng.randint(0, max(1, width - mw + 1)))
    masked[:, :, y0 : y0 + mh, x0 : x0 + mw] = 0
    token_masked, _ = net._encode(masked)
    return torch.mean(torch.abs(token_masked - token_full.detach()))
