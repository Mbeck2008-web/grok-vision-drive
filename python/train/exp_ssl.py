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


def select_ssl_rows(rows: Iterable[Mapping[str, Any]], holdout_kinds: set[str] | None = None) -> list[Mapping[str, Any]]:
    held = set(holdout_kinds or {"holdout"})
    return [row for row in rows if str(row.get("kind") or "train") not in held]


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
