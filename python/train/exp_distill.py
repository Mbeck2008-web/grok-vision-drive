"""Opt-in teacher-feature distillation for fast GRU/head sweeps (W2).

Off by default. Caches frozen teacher tokens for **train rows only** (never
holdout). A tiny student trains from those tokens; deployed perception stays
camera-only (this path is training-only and is not exported).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

FORMAT = "gvd-distill-feats-1"
DEFAULT_TOKEN_DIM = 64


class FeatCache:
    def __init__(self, root: Path, indices: list[int], tokens: np.ndarray, meta: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.indices = list(indices)
        self._index = {int(i): n for n, i in enumerate(self.indices)}
        self.tokens = tokens
        self.meta = dict(meta)

    def token(self, row_i: int) -> np.ndarray:
        return np.asarray(self.tokens[self._index[int(row_i)]], dtype=np.float32)


def build_feat_cache(
    dest: Path | str,
    rows: Iterable[Mapping[str, Any]],
    *,
    encode: Callable[[Mapping[str, Any]], np.ndarray],
    holdout_kinds: set[str] | None = None,
    token_dim: int = DEFAULT_TOKEN_DIM,
) -> list[int]:
    """Write train-only tokens. Returns the row ids that were written."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    held = set(holdout_kinds or {"holdout"})
    kept: list[np.ndarray] = []
    ids: list[int] = []
    for row in rows:
        kind = str(row.get("kind") or "train")
        if kind in held:
            continue
        vec = np.asarray(encode(row), dtype=np.float32).reshape(-1)
        if int(vec.size) != int(token_dim):
            raise ValueError(f"token dim {vec.size} != {token_dim}")
        kept.append(vec)
        ids.append(int(row["i"]))
    stack = np.stack(kept, axis=0) if kept else np.zeros((0, int(token_dim)), dtype=np.float32)
    np.save(dest / "tokens.npy", stack)
    meta = {"format": FORMAT, "token_dim": int(token_dim), "indices": ids, "n": len(ids)}
    (dest / "meta.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    return ids


def load_feat_cache(root: Path | str) -> FeatCache:
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{root}: not a {FORMAT} cache")
    tokens = np.load(root / "tokens.npy")
    indices = [int(x) for x in meta.get("indices") or []]
    if int(tokens.shape[0]) != len(indices):
        raise ValueError(f"{root}: token rows {tokens.shape[0]} != indices {len(indices)}")
    return FeatCache(root, indices, tokens, meta)


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class StudentFromFeats(nn.Module):
        """Tiny GRU + lane_gt head on cached teacher tokens (training only)."""

        def __init__(self, token_dim: int = DEFAULT_TOKEN_DIM, hidden: int = 128) -> None:
            super().__init__()
            from python.debug.lane_targets import LANE_GT_DIM

            self.token_dim = int(token_dim)
            self.hidden = int(hidden)
            self.gru = nn.GRU(self.token_dim + 1, self.hidden, num_layers=1, batch_first=True)
            self.lane_gt = nn.Linear(self.hidden, int(LANE_GT_DIM))

        def forward(self, token: Any, dt: Any | None = None) -> Any:
            if token.dim() == 1:
                token = token.unsqueeze(0)
            if token.dim() == 2:
                token = token.unsqueeze(1)
            if dt is None:
                dt = token.new_zeros(token.shape[0], token.shape[1], 1)
            elif not hasattr(dt, "dim"):
                dt = token.new_full((token.shape[0], token.shape[1], 1), float(dt))
            elif dt.dim() == 1:
                dt = dt.view(token.shape[0], token.shape[1], 1)
            gru_in = torch.cat([token, dt], dim=-1)
            out, _h = self.gru(gru_in)
            return self.lane_gt(out[:, -1, :])

    def student_lane_loss(student: StudentFromFeats, token: Any, target: Any) -> Any:
        pred = student(token)
        return torch.mean(torch.abs(pred - target.reshape(pred.shape)))

else:  # pragma: no cover

    class StudentFromFeats:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("torch required")

    def student_lane_loss(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("torch required")


def run_student_epochs(
    cache: FeatCache,
    *,
    epochs: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: int = 128,
) -> dict[str, float]:
    """Train StudentFromFeats on cached tokens only (no PNG / holdout access)."""
    if torch is None:
        raise RuntimeError("torch required")
    torch.manual_seed(int(seed))
    student = StudentFromFeats(token_dim=int(cache.meta.get("token_dim") or DEFAULT_TOKEN_DIM), hidden=int(hidden))
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr))
    # Zero targets keep the mechanism honest without needing lane labels here;
    # real sweeps pass labels through a caller-supplied target_fn later.
    target = torch.zeros(1, student.lane_gt.out_features)
    last = 0.0
    for _epoch in range(max(1, int(epochs))):
        for row_i in cache.indices:
            token = torch.from_numpy(cache.token(row_i)).unsqueeze(0)
            opt.zero_grad(set_to_none=True)
            loss = student_lane_loss(student, token, target)
            loss.backward()
            opt.step()
            last = float(loss.detach().item())
    return {"last_loss": last, "n": float(len(cache.indices)), "epochs": float(epochs)}
