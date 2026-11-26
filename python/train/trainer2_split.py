"""Trainer 2.0 split manifest and pre-registration (docs/specs/trainer_2.md, PR 1).

``split_manifest.json`` lists every (source, i) row per split, sorted, with a
sha256, and names the held-out map for the 2.0 bar. It is written **before**
step 0. SSL, every curriculum stage, the teacher and the distill cache read
this one manifest. Rows from the held-out map are in no split at all: not
train, not SSL / teacher / distill cache, and not the matched holdout used for
stage exit, model selection or the 25% RMS gate (GVD call 2026-11-25, a
proposal pending Michael).

``run_manifest.json`` pre-registers one run per arm: git SHA, config sha256,
seed, split sha256 (held-out map included), SSL on/off and the decision tier.
A second seed for the same arm is refused (no re-rolls).

Training-only module. Nothing on a live path imports it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from python.train.continual import (
    HOLDOUT_BUFFER_M,
    HOLDOUT_CELL_M,
    HOLDOUT_GAP_S,
    _row_xy,
    cell_holdout_labels,
    cell_of,
)

FORMAT = "gvd-trainer2-split-1"
RUN_FORMAT = "gvd-trainer2-run-1"
SPLIT_MANIFEST = "split_manifest.json"
RUN_MANIFEST = "run_manifest.json"
SPLITS = ("train", "holdout")
DECISION_TIERS = ("iteration", "2.0")
KEY_FIELDS = ("source", "i", "t", "map", "session")
# PROPOSAL pending Michael (GVD call 2026-11-25, interpreting "held out of all
# training"): the held-out map is also barred from the matched holdout used for
# stage exit, model selection and the 25% RMS gate. False keeps held-out-map
# rows out of train only. The choice is recorded in the manifest and hashed.
BAR_HELDOUT_MAP_FROM_HOLDOUT = True
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

__all__ = [
    "BAR_HELDOUT_MAP_FROM_HOLDOUT",
    "DECISION_TIERS",
    "FORMAT",
    "HOLDOUT_GAP_S",
    "RUN_MANIFEST",
    "SPLIT_MANIFEST",
    "SplitError",
    "build_split_manifest",
    "check_split_manifest",
    "check_subset_of_train",
    "filter_rows",
    "manifest_sha256",
    "preregister",
    "require_heldout_map",
    "require_preregistered",
    "same_split",
    "write_split_manifest",
]


class SplitError(ValueError):
    """The split or run manifest is missing, inconsistent, or refused."""


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of ``manifest`` without its own ``sha256`` key."""
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    return hashlib.sha256(_canon(body)).hexdigest()


def _source(row: Mapping[str, Any]) -> str:
    return str(row.get("source") or "")


def _index(row: Mapping[str, Any]) -> int:
    try:
        return int(row["i"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SplitError(f"row from source {_source(row)!r} has no integer i") from exc


def _time(row: Mapping[str, Any]) -> float | None:
    try:
        stamp = float(row["t"])
    except (KeyError, TypeError, ValueError):
        return None
    return round(stamp, 6) if math.isfinite(stamp) else None


def _group(row: Mapping[str, Any]) -> tuple[str, str]:
    """One recording: (source, session). Timestamps compare only inside it."""
    return (_source(row), str(row.get("session") or ""))


def _row_map(row: Mapping[str, Any], source_maps: Mapping[str, Any]) -> str | None:
    value = row.get("map")
    if value is None or str(value).strip() == "":
        value = source_maps.get(_source(row))
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


def _cell_key(label: Any) -> tuple[int, int]:
    return (int(label[0]), int(label[1]))


def _cells_block(cells: Mapping[str, Any]) -> dict[str, Any]:
    """The listed holdout cells (a ``continual.choose_holdout_cells`` spec) as plain data."""
    try:
        listed = sorted({(int(c["cx"]), int(c["cy"])) for c in cells.get("cells") or []})
        cell_m = float(cells.get("cell_m", HOLDOUT_CELL_M))
        buffer_m = float(cells.get("buffer_m", HOLDOUT_BUFFER_M))
    except (KeyError, TypeError, ValueError) as exc:
        raise SplitError(f"cells spec is malformed: {exc}") from exc
    if not listed:
        raise SplitError("cells spec lists no holdout cell")
    if not (math.isfinite(cell_m) and cell_m > 0 and math.isfinite(buffer_m) and buffer_m >= 0):
        raise SplitError("cells spec needs a positive finite cell_m and a finite buffer_m >= 0")
    return {
        "cell_m": cell_m,
        "buffer_m": buffer_m,
        "listed": [list(c) for c in listed],
        "train": [],
        "holdout": [],
    }


def build_split_manifest(
    plan: Iterable[Mapping[str, Any]],
    *,
    source_maps: Mapping[str, Any] | None = None,
    heldout_map: str | None,
    gap_s: float = HOLDOUT_GAP_S,
    cells: Mapping[str, Any] | None = None,
    bar_heldout_from_holdout: bool = BAR_HELDOUT_MAP_FROM_HOLDOUT,
) -> dict[str, Any]:
    """Split manifest from a #187 pass plan (``continual.pass_plan`` / ``combined_pass_plan``).

    Each key is ``[source, i, t, map, session]``, sorted by (source, i), unique per split.
    A (source, i) in both splits is refused. A train row within ``gap_s`` of a
    holdout row of the same recording is purged (counted in ``purged_gap_rows``).
    A recording that has holdout rows needs a finite ``t`` on every one of its
    rows, train and holdout: without it the purge cannot be proven, so the build
    is refused (never silently kept).
    Rows on ``heldout_map`` go to no split (counted in ``heldout_map_rows``).
    With ``bar_heldout_from_holdout=False`` (PROPOSAL pending Michael) they are
    barred from train only and stay in the holdout.
    When ``heldout_map`` is set, a row whose map is unknown is refused: the
    manifest cannot prove it is not on the held-out map.
    ``cells`` (a ``continual.choose_holdout_cells`` spec, ``--holdout-cells``)
    turns the geographic split on: every holdout row must sit in a listed cell,
    every train row outside the listed cells and their ``buffer_m``. A train row
    in the buffer or without a position is purged (counted in
    ``purged_buffer_rows``); a train row inside a listed cell, or a holdout row
    outside every listed cell, is refused.
    """
    maps = dict(source_maps or {})
    held_map = None if heldout_map is None or str(heldout_map).strip() == "" else str(heldout_map).strip()
    cells_block = None if cells is None else _cells_block(cells)
    seen: dict[str, dict[tuple[str, int], tuple[float | None, str | None, tuple[str, str]]]] = {s: {} for s in SPLITS}
    cell_of_key: dict[str, dict[tuple[str, int], tuple[int, int]]] = {s: {} for s in SPLITS}
    excluded: set[tuple[str, int]] = set()
    purged_buffer = 0
    for item in plan:
        kind = str(item.get("kind") or "")
        if kind not in SPLITS:
            continue
        for row in list(item.get("rows") or []):
            key = (_source(row), _index(row))
            row_map = _row_map(row, maps)
            if held_map is not None and row_map is None:
                raise SplitError(
                    f"row {key} has no map (no row 'map' and no source_maps entry); "
                    f"cannot prove it is not on held-out map {held_map!r}"
                )
            if held_map is not None and row_map == held_map and (kind == "train" or bar_heldout_from_holdout):
                excluded.add(key)
                continue
            if cells_block is not None:
                label = cell_holdout_labels([row], cells, gap_s=0.0)[0]
                if kind == "holdout" and label != "hold":
                    raise SplitError(f"holdout row {key} is not inside a listed holdout cell (label {label!r})")
                if kind == "train":
                    if label == "hold":
                        raise SplitError(f"train row {key} sits inside a listed holdout cell")
                    if label == "gap":
                        if key not in seen["train"]:
                            purged_buffer += 1
                        continue
                xy = _row_xy(row)
                if xy is None:  # unreachable: a hold/train label needs a position
                    raise SplitError(f"row {key} has no position under a cells split")
                cell_of_key[kind].setdefault(key, cell_of(xy, cells_block["cell_m"]))
            seen[kind].setdefault(key, (_time(row), row_map, _group(row)))
    both = sorted(set(seen["train"]) & set(seen["holdout"]))
    if both:
        raise SplitError(f"{len(both)} row(s) in both train and holdout, first {both[0]}")
    hold_times: dict[tuple[str, str], list[float]] = {}
    hold_groups = {val[2] for val in seen["holdout"].values()}
    for split in SPLITS:
        for key, (stamp, _m, group) in seen[split].items():
            if stamp is None and group in hold_groups:
                raise SplitError(
                    f"{split} row {key} has no finite t but its recording {group} has holdout rows; "
                    f"the {gap_s:g} s gap cannot be enforced"
                )
    for _key, (stamp, _m, group) in seen["holdout"].items():
        if stamp is not None:
            hold_times.setdefault(group, []).append(stamp)
    purged = 0
    gap = float(gap_s)
    for key in list(seen["train"]):
        stamp, _m, group = seen["train"][key]
        near = hold_times.get(group)
        if stamp is None or not near:
            continue
        if min(abs(stamp - h) for h in near) < gap:
            del seen["train"][key]
            purged += 1
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "heldout_map": held_map,
        "heldout_map_bars_holdout": bool(bar_heldout_from_holdout),
        "gap_s": gap,
        "purged_gap_rows": purged,
        "heldout_map_rows": len(excluded),
    }
    for split in SPLITS:
        manifest[split] = [
            [key[0], key[1], val[0], val[1], val[2][1]]
            for key, val in sorted(seen[split].items(), key=lambda kv: kv[0])
        ]
    if cells_block is not None:
        for split in SPLITS:
            cells_block[split] = [list(cell_of_key[split][key]) for key in sorted(seen[split])]
        manifest["cells"] = cells_block
        manifest["purged_buffer_rows"] = purged_buffer
    manifest["sha256"] = manifest_sha256(manifest)
    check_split_manifest(manifest)
    return manifest


def _keys(manifest: Mapping[str, Any], split: str) -> list[tuple[str, int]]:
    return [(str(k[0]), int(k[1])) for k in manifest.get(split) or []]


def _check_cells(manifest: Mapping[str, Any]) -> None:
    """Cells disjoint when the geographic split is on."""
    block = manifest.get("cells")
    if block is None:
        return
    try:
        listed = {_cell_key(c) for c in block["listed"]}
        per_split = {s: [_cell_key(c) for c in block[s]] for s in SPLITS}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise SplitError(f"cells block is malformed: {exc}") from exc
    if not listed:
        raise SplitError("cells block lists no holdout cell")
    for split in SPLITS:
        if len(per_split[split]) != len(manifest.get(split) or []):
            raise SplitError(f"cells block has {len(per_split[split])} {split} cells for {len(manifest.get(split) or [])} rows")
    for key, cell in zip(manifest.get("train") or [], per_split["train"]):
        if cell in listed:
            raise SplitError(f"train row {key[:2]} is inside holdout cell {cell}")
    for key, cell in zip(manifest.get("holdout") or [], per_split["holdout"]):
        if cell not in listed:
            raise SplitError(f"holdout row {key[:2]} is in cell {cell}, which is not a listed holdout cell")


def check_split_manifest(manifest: Mapping[str, Any]) -> None:
    """``check_trainer2_split_no_overlap`` + ``check_heldout_map_excluded`` on a manifest."""
    if manifest.get("format") != FORMAT:
        raise SplitError(f"split manifest format {manifest.get('format')!r} != {FORMAT!r}")
    if manifest.get("sha256") != manifest_sha256(manifest):
        raise SplitError("split manifest sha256 does not match its content")
    for split in SPLITS:
        for key in manifest.get(split) or []:
            if not isinstance(key, (list, tuple)) or len(key) != len(KEY_FIELDS):
                raise SplitError(f"{split} key {key!r} is not [source, i, t, map, session]")
        keys = _keys(manifest, split)
        if keys != sorted(set(keys)):
            raise SplitError(f"{split} keys are not sorted and unique by (source, i)")
    train = set(_keys(manifest, "train"))
    hold = set(_keys(manifest, "holdout"))
    both = sorted(train & hold)
    if both:
        raise SplitError(f"{len(both)} row(s) in both train and holdout, first {both[0]}")
    held_map = manifest.get("heldout_map")
    if held_map is not None:
        bars_holdout = manifest.get("heldout_map_bars_holdout", True)
        for split in SPLITS:
            for key in manifest.get(split) or []:
                if key[3] is None:
                    raise SplitError(f"{split} row {key[:2]} has no map; cannot prove it is not on held-out map {held_map!r}")
                if str(key[3]) == str(held_map) and (split == "train" or bars_holdout):
                    raise SplitError(f"{split} row {key[:2]} is on held-out map {held_map!r}")
    hold_times: dict[tuple[str, str], list[float]] = {}
    hold_groups = {(str(key[0]), str(key[4])) for key in manifest.get("holdout") or []}
    for split in SPLITS:
        for key in manifest.get(split) or []:
            if key[2] is None and (str(key[0]), str(key[4])) in hold_groups:
                raise SplitError(f"{split} row {key[:2]} has no finite t in a recording with holdout rows")
    for key in manifest.get("holdout") or []:
        if key[2] is not None:
            hold_times.setdefault((str(key[0]), str(key[4])), []).append(float(key[2]))
    gap = float(manifest.get("gap_s", HOLDOUT_GAP_S))
    for key in manifest.get("train") or []:
        near = hold_times.get((str(key[0]), str(key[4])))
        if key[2] is not None and near and min(abs(float(key[2]) - h) for h in near) < gap:
            raise SplitError(f"train row {key[:2]} is within the {gap:g} s gap of a holdout row")
    _check_cells(manifest)


def require_heldout_map(manifest: Mapping[str, Any]) -> str:
    """The held-out map name, or refuse: the 2.0 bar cannot be scored without one."""
    name = manifest.get("heldout_map")
    if name is None or str(name).strip() == "":
        raise SplitError("split manifest names no held-out map; the 2.0 bar needs one (GVD names it)")
    return str(name)


def filter_rows(rows: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any], split: str) -> list[Mapping[str, Any]]:
    """Rows whose (source, i) the manifest puts in ``split``, in input order."""
    if split not in SPLITS:
        raise SplitError(f"unknown split {split!r}")
    keep = set(_keys(manifest, split))
    return [row for row in rows if (_source(row), _index(row)) in keep]


def check_subset_of_train(rows: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any], *, what: str) -> None:
    """Refuse ``what`` (ssl, teacher, distill_cache, ...) rows that are not train rows."""
    train = set(_keys(manifest, "train"))
    for row in rows:
        key = (_source(row), _index(row))
        if key not in train:
            raise SplitError(f"{what} row {key} is not a train row in the split manifest")


def same_split(manifest: Mapping[str, Any], a: tuple[str, int], b: tuple[str, int]) -> bool:
    """True when both (source, i) keys are in one split (temporal targets must not cross)."""
    for split in SPLITS:
        keys = set(_keys(manifest, split))
        if (str(a[0]), int(a[1])) in keys and (str(b[0]), int(b[1])) in keys:
            return True
    return False


def _write_atomic(path: Path, data: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load(path: Path, what: str) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SplitError(f"{what} {path} is missing") from exc
    except ValueError as exc:
        raise SplitError(f"{what} {path} is not valid JSON") from exc


def write_split_manifest(path: Path | str, manifest: Mapping[str, Any]) -> Path:
    """Write once. Identical content is a no-op; different content is refused."""
    target = Path(path)
    check_split_manifest(manifest)
    if target.exists():
        old = _load(target, "split manifest")
        if old.get("sha256") == manifest.get("sha256"):
            return target
        raise SplitError(f"{target} already holds a different split (sha256 {old.get('sha256')}); not overwritten")
    _write_atomic(target, manifest)
    return target


def _arm(run: Mapping[str, Any]) -> str:
    body = {k: run[k] for k in ("split_sha256", "config_sha256", "ssl_pretrain", "decision_tier")}
    return hashlib.sha256(_canon(body)).hexdigest()


def preregister(
    out_dir: Path | str,
    *,
    split_path: Path | str,
    git_sha: str,
    config: Mapping[str, Any],
    seed: int,
    ssl_pretrain: bool,
    decision_tier: str,
    registry: Path | str,
) -> dict[str, Any]:
    """Write ``run_manifest.json`` before training. Returns it.

    The same run again (a resume) returns the existing manifest. Another seed
    for the same arm (split, config, SSL on/off, tier) is refused, in this
    folder and in ``registry``. ``registry`` is required (keyword, no default):
    a re-roll in a new run folder is only visible through a folder shared by
    every run of the experiment, so a caller cannot forget it. Point it at one
    directory per experiment, e.g. next to ``split_manifest.json``.
    """
    if registry is None or str(registry).strip() == "":
        raise SplitError("registry is required: pass the experiment's shared run registry folder")
    out = Path(out_dir)
    if decision_tier not in DECISION_TIERS:
        raise SplitError(f"decision tier {decision_tier!r} not in {DECISION_TIERS}")
    sha = str(git_sha).strip().lower()
    if not _SHA_RE.match(sha):
        raise SplitError(f"git sha {git_sha!r} is not a hex commit id")
    split = _load(Path(split_path), "split manifest")
    check_split_manifest(split)
    if decision_tier == "2.0":
        require_heldout_map(split)
    run: dict[str, Any] = {
        "format": RUN_FORMAT,
        "git_sha": sha,
        "config": json.loads(_canon(dict(config))),
        "config_sha256": hashlib.sha256(_canon(dict(config))).hexdigest(),
        "seed": int(seed),
        "split_path": str(Path(split_path).resolve()),
        "split_sha256": split["sha256"],
        "heldout_map": split.get("heldout_map"),
        "ssl_pretrain": bool(ssl_pretrain),
        "decision_tier": decision_tier,
    }
    run["sha256"] = manifest_sha256(run)
    target = out / RUN_MANIFEST
    if target.exists():
        old = _load(target, "run manifest")
        if old.get("sha256") == run["sha256"]:
            return old
        if old.get("seed") != run["seed"]:
            raise SplitError(f"{target} pre-registered seed {old.get('seed')}; seed {seed} would be a re-roll")
        raise SplitError(f"{target} pre-registered a different run; start a new folder for a new arm")
    reg_dir = Path(registry)
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg_file = reg_dir / f"{_arm(run)}.json"
    if reg_file.exists():
        prior = _load(reg_file, "run registry entry")
        if int(prior.get("seed", -1)) != run["seed"]:
            raise SplitError(
                f"arm already pre-registered with seed {prior.get('seed')} ({prior.get('out_dir')}); "
                f"seed {seed} would be a re-roll"
            )
    out.mkdir(parents=True, exist_ok=True)
    _write_atomic(target, run)
    if not reg_file.exists():
        _write_atomic(reg_file, {"seed": run["seed"], "out_dir": str(out.resolve()), "run_sha256": run["sha256"]})
    return run


def require_preregistered(out_dir: Path | str) -> dict[str, Any]:
    """Call before step 0: the run and split manifests exist and match their hashes."""
    run = _load(Path(out_dir) / RUN_MANIFEST, "run manifest")
    if run.get("format") != RUN_FORMAT or run.get("sha256") != manifest_sha256(run):
        raise SplitError("run manifest sha256 does not match its content")
    split = _load(Path(run["split_path"]), "split manifest")
    if split.get("sha256") != run.get("split_sha256"):
        raise SplitError("split manifest changed after pre-registration (sha256 differs)")
    check_split_manifest(split)
    return run
