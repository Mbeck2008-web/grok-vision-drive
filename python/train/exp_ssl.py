"""Opt-in self-supervised stem pretraining (W4).

Off by default. Pretext runs on unlabeled **train** frames only (never holdout).
Saves stem weights that ``train_scene --init-stem`` can load before supervised
training. Deployed perception stays camera-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

FORMAT = "gvd-ssl-stem-1"


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


def run_ssl_pretrain(
    net: Any,
    rows: Iterable[Mapping[str, Any]],
    *,
    image_fn,
    out_path: Path | str,
    lr: float,
    seed: int,
    epochs: int = 1,
) -> dict[str, float]:
    """Masked-token pretext on the provided train rows only (no zero canvas)."""
    import torch

    collected = list(rows)
    if not collected:
        raise SslError("no train frames after the #187 split")
    opt = torch.optim.AdamW(net.stem.parameters(), lr=float(lr))
    rng = np.random.RandomState(int(seed))
    last = 0.0
    used = 0
    for _epoch in range(max(1, int(epochs))):
        for row in collected:
            image = image_fn(row)
            if image is None:
                continue
            if not hasattr(image, "shape"):
                image = frame_to_nchw(image, net)
            opt.zero_grad(set_to_none=True)
            loss = masked_token_pretext_loss(net, image, rng=rng)
            loss.backward()
            opt.step()
            last = float(loss.detach().item())
            used += 1
    if used <= 0:
        raise SslError("no train-frame images loaded; refusing to pretrain on a zero canvas")
    save_stem_weights(out_path, net)
    return {"last_loss": last, "n": float(used), "rows": float(len(collected))}


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


def masked_token_pretext_loss(net: Any, image: Any, *, mask_ratio: float = 0.25, rng: np.random.RandomState | None = None) -> Any:
    """Predict the stem token of a masked canvas from the unmasked stem token.

    Simple contrastive/masked stand-in: zero a random spatial block on a clone,
    encode both, and pull the tokens together with L1. Train rows only at the
    caller. Gradients update the stem (and only the stem if other params are frozen).
    """
    import torch

    rng = rng or np.random.RandomState(0)
    token_full, _feat = net._encode(image)
    masked = image.clone()
    _b, _c, height, width = masked.shape
    mh = max(1, int(height * float(mask_ratio)))
    mw = max(1, int(width * float(mask_ratio)))
    y0 = int(rng.randint(0, max(1, height - mh + 1)))
    x0 = int(rng.randint(0, max(1, width - mw + 1)))
    masked[:, :, y0 : y0 + mh, x0 : x0 + mw] = 0
    token_masked, _ = net._encode(masked)
    return torch.mean(torch.abs(token_masked - token_full.detach()))
