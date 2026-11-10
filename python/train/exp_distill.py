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
TEACHER_FORMAT = "gvd-distill-teacher-1"
DEFAULT_TOKEN_DIM = 64


class DistillError(ValueError):
    """Teacher, cache, or row kind is missing; never fall back to zeros."""


def require_row_kind(row: Mapping[str, Any]) -> str:
    """Return the row kind, or refuse an untagged row (no train default)."""
    if "kind" not in row or row["kind"] is None or str(row["kind"]).strip() == "":
        raise DistillError(f"row i={row.get('i')!r} has no kind; refuse untagged rows")
    return str(row["kind"])


def row_frame_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return (str(row.get("source") or ""), int(row["i"]))


class FeatCache:
    def __init__(
        self,
        root: Path,
        indices: list[int],
        tokens: np.ndarray,
        meta: Mapping[str, Any],
        targets: np.ndarray | None = None,
        keys: list[tuple[str, int]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.indices = list(indices)
        self.keys = list(keys) if keys is not None else [("", int(i)) for i in self.indices]
        # Bare-i map keeps the last duplicate (legacy). Lookups go through
        # _resolve, which keys on (source, i) the same way as #196.
        self._index = {int(i): n for n, i in enumerate(self.indices)}
        self._key_index = {tuple(k): n for n, k in enumerate(self.keys)}
        self.tokens = tokens
        self.targets = targets
        self.meta = dict(meta)

    def _resolve(self, key: int | Mapping[str, Any] | tuple[str, int]) -> int:
        """Index of the row for ``(source, i)``, a row mapping, or a bare ``i``.

        Combined ``--strips`` share bare ``i`` values, so callers must pass a
        row or ``(source, i)``. A bare int is only accepted when it matches
        exactly one cached key (or the empty-source key ``("", i)``).
        """
        if isinstance(key, Mapping):
            k = row_frame_key(key)
        elif isinstance(key, tuple) and len(key) == 2:
            k = (str(key[0]), int(key[1]))
        else:
            i = int(key)  # type: ignore[arg-type]
            empty = ("", i)
            if empty in self._key_index:
                return self._key_index[empty]
            matches = [n for n, kk in enumerate(self.keys) if int(kk[1]) == i]
            if len(matches) == 1:
                return matches[0]
            if len(matches) == 0 and i in self._index:
                return self._index[i]
            raise DistillError(
                f"{self.root}: bare i={i} is ambiguous across sources; pass (source, i) or a row"
            )
        if k in self._key_index:
            return self._key_index[k]
        fallback = ("", int(k[1]))
        if k[0] and fallback in self._key_index:
            return self._key_index[fallback]
        raise DistillError(f"{self.root}: cache has no entry for {k}")

    def token(self, key: int | Mapping[str, Any] | tuple[str, int]) -> np.ndarray:
        return np.asarray(self.tokens[self._resolve(key)], dtype=np.float32)

    def target(self, key: int | Mapping[str, Any] | tuple[str, int]) -> np.ndarray:
        if self.targets is None:
            raise DistillError(f"{self.root}: cache has no teacher targets")
        return np.asarray(self.targets[self._resolve(key)], dtype=np.float32)


def split_tagged_rows(
    rows: list[Mapping[str, Any]],
    source_meta: Mapping[str, Any] | None = None,
    *,
    holdout_blocks: int = 1,
    length: int | None = None,
    stride: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Tag train/holdout rows from the #187 ``pass_plan`` / ``combined_pass_plan`` split.

    Gap frames never appear in the plan. Raw strip rows typically have no
    ``kind``; kinds are taken from the window, not defaulted on the raw row.
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
    return _tagged_from_plan(plan, "train"), _tagged_from_plan(plan, "holdout")


def _tagged_from_plan(plan: Iterable[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in plan:
        if str(item.get("kind") or "") != kind:
            continue
        for raw in list(item.get("rows") or []):
            tagged = dict(raw)
            tagged["kind"] = kind
            key = row_frame_key(tagged)
            if key in seen:
                continue
            seen.add(key)
            out.append(tagged)
    return out


def build_feat_cache(
    dest: Path | str,
    rows: Iterable[Mapping[str, Any]],
    *,
    encode: Callable[[Mapping[str, Any]], np.ndarray],
    target_fn: Callable[[Mapping[str, Any]], np.ndarray] | None = None,
    holdout_kinds: set[str] | None = None,
    token_dim: int = DEFAULT_TOKEN_DIM,
) -> list[int]:
    """Write train-only tokens (and teacher targets when given). Returns written ids."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    held = set(holdout_kinds or {"holdout", "gap"})
    kept: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ids: list[int] = []
    keys: list[tuple[str, int]] = []
    for row in rows:
        kind = require_row_kind(row)
        if kind in held:
            continue
        vec = np.asarray(encode(row), dtype=np.float32).reshape(-1)
        if int(vec.size) != int(token_dim):
            raise ValueError(f"token dim {vec.size} != {token_dim}")
        kept.append(vec)
        ids.append(int(row["i"]))
        keys.append(row_frame_key(row))
        if target_fn is not None:
            tgt = np.asarray(target_fn(row), dtype=np.float32).reshape(-1)
            targets.append(tgt)
    stack = np.stack(kept, axis=0) if kept else np.zeros((0, int(token_dim)), dtype=np.float32)
    np.save(dest / "tokens.npy", stack)
    meta: dict[str, Any] = {
        "format": FORMAT,
        "token_dim": int(token_dim),
        "indices": ids,
        "keys": [list(k) for k in keys],
        "n": len(ids),
        "has_targets": bool(targets),
    }
    if target_fn is not None:
        if len(targets) != len(ids):
            raise DistillError("target_fn must return a target for every written row")
        tgt_stack = np.stack(targets, axis=0) if targets else np.zeros((0, 0), dtype=np.float32)
        np.save(dest / "targets.npy", tgt_stack)
    (dest / "meta.json").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    return ids


def encode_rows_to_cache(
    rows: Iterable[Mapping[str, Any]],
    *,
    encode: Callable[[Mapping[str, Any]], np.ndarray],
    target_fn: Callable[[Mapping[str, Any]], np.ndarray],
    token_dim: int = DEFAULT_TOKEN_DIM,
) -> FeatCache:
    """In-memory cache for already-filtered rows (holdout eval). Does not skip kinds."""
    kept: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ids: list[int] = []
    keys: list[tuple[str, int]] = []
    for row in rows:
        require_row_kind(row)
        vec = np.asarray(encode(row), dtype=np.float32).reshape(-1)
        if int(vec.size) != int(token_dim):
            raise ValueError(f"token dim {vec.size} != {token_dim}")
        tgt = np.asarray(target_fn(row), dtype=np.float32).reshape(-1)
        kept.append(vec)
        targets.append(tgt)
        ids.append(int(row["i"]))
        keys.append(row_frame_key(row))
    tokens = np.stack(kept, axis=0) if kept else np.zeros((0, int(token_dim)), dtype=np.float32)
    tgt_arr = np.stack(targets, axis=0) if targets else None
    meta = {"format": FORMAT, "token_dim": int(token_dim), "indices": ids, "n": len(ids), "has_targets": tgt_arr is not None}
    return FeatCache(Path("memory"), ids, tokens, meta, targets=tgt_arr, keys=keys)


def load_feat_cache(root: Path | str) -> FeatCache:
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{root}: not a {FORMAT} cache")
    tokens = np.load(root / "tokens.npy")
    indices = [int(x) for x in meta.get("indices") or []]
    raw_keys = meta.get("keys") or []
    keys = [(str(k[0]), int(k[1])) for k in raw_keys] if raw_keys else [("", i) for i in indices]
    if int(tokens.shape[0]) != len(indices):
        raise ValueError(f"{root}: token rows {tokens.shape[0]} != indices {len(indices)}")
    targets = None
    tgt_path = root / "targets.npy"
    if meta.get("has_targets") or tgt_path.is_file():
        targets = np.load(tgt_path)
        if int(targets.shape[0]) != len(indices):
            raise ValueError(f"{root}: target rows {targets.shape[0]} != indices {len(indices)}")
    return FeatCache(root, indices, tokens, meta, targets=targets, keys=keys)


class TeacherSource:
    """Frozen teacher: tokens and lane targets. Missing lookups raise; never zeros."""

    def encode(self, row: Mapping[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def target(self, row: Mapping[str, Any]) -> np.ndarray:
        raise NotImplementedError


class FeatureFileTeacher(TeacherSource):
    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        self.path = path
        self._tokens: dict[tuple[str, int], np.ndarray] = {}
        self._targets: dict[tuple[str, int], np.ndarray] = {}
        self.token_dim = DEFAULT_TOKEN_DIM
        if path.suffix.lower() == ".json":
            self._load_json(path)
        elif path.suffix.lower() == ".npz":
            self._load_npz(path)
        elif path.suffix.lower() == ".npy":
            raise DistillError(
                f"{path}: a bare .npy is ambiguous; use a {TEACHER_FORMAT} .json or .npz "
                "with tokens and targets. Refusing zero fallback."
            )
        else:
            raise DistillError(f"{path}: unrecognized teacher features file")

    def _load_json(self, path: Path) -> None:
        blob = json.loads(path.read_text(encoding="utf-8"))
        if blob.get("format") != TEACHER_FORMAT:
            raise DistillError(f"{path}: not a {TEACHER_FORMAT} features file")
        self.token_dim = int(blob.get("token_dim") or DEFAULT_TOKEN_DIM)
        for entry in blob.get("entries") or []:
            key = (str(entry.get("source") or ""), int(entry["i"]))
            token = np.asarray(entry["token"], dtype=np.float32).reshape(-1)
            target = np.asarray(entry["target"], dtype=np.float32).reshape(-1)
            self._tokens[key] = token
            self._targets[key] = target

    def _load_npz(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as blob:
            meta = json.loads(str(blob["meta"].item() if getattr(blob["meta"], "shape", None) == () else blob["meta"]))
            if meta.get("format") != TEACHER_FORMAT:
                raise DistillError(f"{path}: not a {TEACHER_FORMAT} features file")
            self.token_dim = int(meta.get("token_dim") or DEFAULT_TOKEN_DIM)
            indices = [int(x) for x in blob["indices"]]
            sources = [str(x) for x in blob["sources"]] if "sources" in blob.files else [""] * len(indices)
            tokens = np.asarray(blob["tokens"], dtype=np.float32)
            targets = np.asarray(blob["targets"], dtype=np.float32)
            if tokens.shape[0] != len(indices) or targets.shape[0] != len(indices):
                raise DistillError(f"{path}: tokens/targets length mismatch")
            for n, idx in enumerate(indices):
                key = (sources[n], idx)
                self._tokens[key] = np.asarray(tokens[n], dtype=np.float32).reshape(-1)
                self._targets[key] = np.asarray(targets[n], dtype=np.float32).reshape(-1)

    def _lookup(self, store: dict[tuple[str, int], np.ndarray], row: Mapping[str, Any], what: str) -> np.ndarray:
        key = row_frame_key(row)
        if key in store:
            return store[key]
        fallback = ("", int(row["i"]))
        if key[0] and fallback in store:
            return store[fallback]
        raise DistillError(f"{self.path}: teacher has no {what} for {key}; refusing zero fallback")

    def encode(self, row: Mapping[str, Any]) -> np.ndarray:
        return self._lookup(self._tokens, row, "token")

    def target(self, row: Mapping[str, Any]) -> np.ndarray:
        return self._lookup(self._targets, row, "target")


def write_teacher_features(path: Path | str, entries: Iterable[Mapping[str, Any]], *, token_dim: int = DEFAULT_TOKEN_DIM) -> Path:
    """Write a {TEACHER_FORMAT} JSON features file (tests / offline export)."""
    path = Path(path)
    payload = {
        "format": TEACHER_FORMAT,
        "token_dim": int(token_dim),
        "entries": [
            {
                "i": int(item["i"]),
                "source": str(item.get("source") or ""),
                "token": np.asarray(item["token"], dtype=np.float32).reshape(-1).tolist(),
                "target": np.asarray(item["target"], dtype=np.float32).reshape(-1).tolist(),
            }
            for item in entries
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def load_teacher(path: Path | str) -> TeacherSource:
    """Load a teacher features file or refuse a missing/unsupported teacher.

    Offline CPU boxes typically have no live SceneNet teacher weights that
    can encode PNGs. A ``gvd-distill-teacher-1`` features file (JSON/NPZ)
    carries the teacher's tokens and targets. A missing path never falls
    back to zeros. An epoch checkpoint path is recognized and refused with
    a clear error unless it is actually a features file.
    """
    raw = str(path or "").strip()
    if not raw:
        raise DistillError(
            "--exp-distill-teacher is required (checkpoint or features file); refusing zero fallback"
        )
    dest = Path(raw)
    if not dest.exists():
        raise DistillError(
            f"--exp-distill-teacher not found: {dest}. Provide a {TEACHER_FORMAT} "
            "features file or a teacher checkpoint; refusing zero fallback"
        )
    suffix = dest.suffix.lower()
    if suffix in {".json", ".npz", ".npy"}:
        return FeatureFileTeacher(dest)
    # Epoch ckpt path: try to detect the format so we do not silently zero-fill.
    try:
        from python.train.epoch_ckpt import FORMAT as CKPT_FORMAT
        from python.train.epoch_ckpt import load_epoch_ckpt

        blob = load_epoch_ckpt(dest)
        fmt = (blob.get("meta") or {}).get("format") if isinstance(blob, Mapping) else None
        raise DistillError(
            f"{dest}: found {fmt or CKPT_FORMAT} checkpoint, but this offline path "
            f"needs exported teacher features ({TEACHER_FORMAT} .json/.npz) so tokens "
            "come from the teacher without loading PNGs. Refusing zero fallback."
        )
    except DistillError:
        raise
    except Exception as exc:
        raise DistillError(
            f"{dest}: not a {TEACHER_FORMAT} features file and not a readable "
            f"teacher checkpoint ({exc}). Refusing zero fallback."
        ) from exc


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
    target_fn: Callable[[Any, FeatCache], Any] | None = None,
    epochs: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: int = 128,
    holdout: FeatCache | None = None,
    holdout_target_fn: Callable[[Any, FeatCache], Any] | None = None,
    holdout_note: str = "",
) -> dict[str, float | int | str]:
    """Train StudentFromFeats on cached teacher tokens and teacher targets.

    ``target_fn`` is required. There is no zero-target fallback.
    """
    if torch is None:
        raise RuntimeError("torch required")
    if target_fn is None:
        raise DistillError("target_fn is required; student will not regress to zeros")
    torch.manual_seed(int(seed))
    student = StudentFromFeats(token_dim=int(cache.meta.get("token_dim") or DEFAULT_TOKEN_DIM), hidden=int(hidden))
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr))
    last = 0.0
    train_keys = list(cache.keys)
    for _epoch in range(max(1, int(epochs))):
        for key in train_keys:
            token = torch.from_numpy(np.asarray(cache.token(key), dtype=np.float32)).unsqueeze(0)
            raw_tgt = target_fn(key, cache)
            # Lock: student trains on the cache's teacher target for this key,
            # not a zeroed/constant stub. A mutation that zeroes raw_tgt after
            # target_fn fails this check.
            expected = np.asarray(cache.target(key), dtype=np.float32).reshape(-1)
            got = np.asarray(raw_tgt, dtype=np.float32).reshape(-1)
            if got.shape != expected.shape or not np.allclose(got, expected, equal_nan=False):
                raise DistillError(
                    f"student target for {key} must equal cache teacher target; "
                    "refusing zero/constant fallback"
                )
            target = torch.from_numpy(got).reshape(1, -1)
            if int(target.numel()) != int(student.lane_gt.out_features):
                raise DistillError(
                    f"teacher target dim {int(target.numel())} != student {int(student.lane_gt.out_features)}"
                )
            opt.zero_grad(set_to_none=True)
            loss = student_lane_loss(student, token, target)
            loss.backward()
            opt.step()
            last = float(loss.detach().item())
    out: dict[str, float | int | str] = {
        "last_loss": last,
        "n": float(len(train_keys)),
        "epochs": float(epochs),
    }
    if holdout is not None:
        fn = holdout_target_fn or target_fn
        maes: list[float] = []
        student.eval()
        with torch.no_grad():
            for key in list(holdout.keys):
                token = torch.from_numpy(np.asarray(holdout.token(key), dtype=np.float32)).unsqueeze(0)
                raw_tgt = fn(key, holdout)
                expected = np.asarray(holdout.target(key), dtype=np.float32).reshape(-1)
                got = np.asarray(raw_tgt, dtype=np.float32).reshape(-1)
                if got.shape != expected.shape or not np.allclose(got, expected, equal_nan=False):
                    raise DistillError(
                        f"holdout target for {key} must equal cache teacher target"
                    )
                target = torch.from_numpy(got).reshape(1, -1)
                pred = student(token)
                maes.append(float(torch.mean(torch.abs(pred - target.reshape(pred.shape))).item()))
        out["holdout_mae"] = float(sum(maes) / len(maes)) if maes else float("nan")
        out["holdout_n"] = int(len(maes))
        if holdout_note:
            out["holdout_note"] = str(holdout_note)
    return out
