#!/usr/bin/env python3
"""``--holdout-blocks K`` and ``--holdout-cells`` holdouts for the scene trainer.

``--holdout-blocks K`` spreads the time holdout over K equal blocks.

K=1 must give exactly the plan the trainer built before the option. K>1
holds out the last 20% of each block with a 4 s gap on both sides of every
slice, never lists a row as both train and holdout, and never lets a train
chunk span a holdout slice.

Synthetic checks always run. ``--real DIR`` (or GVD_TEST_STRIPS) also runs
the checks on a recorded index, the same rows the trainer selects.
``--base REV`` compares K=1 with ``python/train/continual.py`` at REV.
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.train import continual  # noqa: E402
from python.train.continual import (  # noqa: E402
    HOLDOUT_BUFFER_M,
    HOLDOUT_CELL_M,
    HOLDOUT_GAP_S,
    _cell_distance_m,
    cell_holdout_labels,
    cell_of,
    choose_holdout_cells,
    load_holdout_cells,
    save_holdout_cells,
    chunk_stride,
    holdout_blocks_from_env,
    pass_plan,
    split_segment_time_blocks,
    split_segment_time_holdout,
)

CHUNK = 16


def ids_of(plan):
    train, hold = set(), set()
    for item in plan:
        (hold if item["kind"] == "holdout" else train).update(int(r["i"]) for r in item["rows"])
    return train, hold


def plan_key(plan):
    return [
        (item["kind"], item["incoming"], item["stride"], item["grad_frames"], tuple(int(r["i"]) for r in item["rows"]))
        for item in plan
    ]


def check_split(rows, plan, blocks, label, need_gap=True):
    t_of = {int(r["i"]): float(r["t"]) for r in rows}
    train, hold = ids_of(plan)
    both = train & hold
    assert not both, f"{label}: {len(both)} ids in train and holdout"
    gap_ids = set(t_of) - train - hold
    tt = sorted(t_of[i] for i in train)
    nearest = float("inf")
    for i in hold:
        t = t_of[i]
        j = bisect.bisect_left(tt, t)
        for k in (j - 1, j):
            if 0 <= k < len(tt):
                nearest = min(nearest, abs(tt[k] - t))
    if need_gap:  # the untagged K=1 tail cut never had a gap; that path is unchanged
        assert nearest >= HOLDOUT_GAP_S - 1e-6, f"{label}: train row {nearest:.3f}s from a holdout row"
    hs = sorted(t_of[i] for i in hold)
    for item in plan:
        if item["kind"] == "holdout":
            continue
        times = [float(r["t"]) for r in item["rows"]]
        lo, hi = min(times), max(times)
        j = bisect.bisect_right(hs, lo)
        assert j >= len(hs) or hs[j] > hi, f"{label}: a train chunk spans a holdout slice"
    slices = sum(1 for item in plan if item["kind"] == "holdout")
    print(
        f"{label}: blocks={blocks} train={len(train)} holdout={len(hold)} gap={len(gap_ids)} "
        f"slices={slices} overlap=0 nearest_train_to_holdout={nearest:.2f}s OK"
    )
    return len(train), len(hold), len(gap_ids)


def synthetic_rows(n=6000, hz=5.0, segment="west_coast_usa#0"):
    rows = []
    for i in range(n):
        if 2000 <= i < 2010:
            continue  # a short hole inside the recording
        rows.append({"i": i, "t": 1000.0 + i / hz, "dt_s": 1.0 / hz, "segment": segment})
    return rows


def old_module(rev):
    text = subprocess.run(
        ["git", "show", f"{rev}:python/train/continual.py"],
        cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    path = Path(tempfile.mkdtemp()) / "continual_base.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("continual_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_checks(rows, label, base):
    stride = chunk_stride(CHUNK)
    k1 = pass_plan(rows, CHUNK, stride)
    assert plan_key(k1) == plan_key(pass_plan(rows, CHUNK, stride, blocks=1)), f"{label}: blocks=1 differs from default"
    if base is not None:
        before = base.pass_plan(rows, CHUNK, stride)
        assert plan_key(before) == plan_key(k1), f"{label}: K=1 plan differs from base"
        print(f"{label}: K=1 plan identical to base ({len(k1)} items, {len(ids_of(k1)[0])} train / {len(ids_of(k1)[1])} holdout ids)")
    else:
        print(f"{label}: K=1 equals the default plan ({len(k1)} items)")
    tagged = all(r.get("segment") for r in rows)
    out = {1: check_split(rows, k1, 1, label, need_gap=tagged)}
    for blocks in (5, 8):
        out[blocks] = check_split(rows, pass_plan(rows, CHUNK, stride, blocks=blocks), blocks, label)
    return out


def check_cells(rows, label, seed=0):
    spec = choose_holdout_cells(rows, seed=seed)
    again = choose_holdout_cells(rows, seed=seed)
    assert spec == again, f"{label}: cell list not deterministic"
    other = choose_holdout_cells(rows, seed=seed + 1)
    path = Path(tempfile.mkdtemp()) / "holdout_cells.json"
    save_holdout_cells(spec, path)
    assert load_holdout_cells(path) == spec, f"{label}: cell list changed in save/load"
    held = {(c["cx"], c["cy"]) for c in spec["cells"]}
    plan = pass_plan(rows, CHUNK, chunk_stride(CHUNK), cells=spec)
    assert plan_key(plan) == plan_key(pass_plan(rows, CHUNK, chunk_stride(CHUNK), cells=load_holdout_cells(path)))
    train, hold = ids_of(plan)
    assert not (train & hold), f"{label}: ids in train and holdout"
    by_id = {int(r["i"]): r for r in rows}
    near = float("inf")
    for i in train:
        xy = by_id[i]["ego"]["pos"][:2]
        key = cell_of(xy)
        assert key not in held, f"{label}: train row {i} in holdout cell {key}"
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                k2 = (key[0] + dx, key[1] + dy)
                if k2 in held:
                    near = min(near, _cell_distance_m(xy, k2, HOLDOUT_CELL_M))
    assert near >= HOLDOUT_BUFFER_M, f"{label}: train row {near:.1f} m from a holdout cell"
    for i in hold:
        assert cell_of(by_id[i]["ego"]["pos"][:2]) in held
    t_of = {int(r["i"]): float(r["t"]) for r in rows}
    tt = sorted(t_of[i] for i in train)
    nearest_t = float("inf")
    for i in hold:
        j = bisect.bisect_left(tt, t_of[i])
        for k in (j - 1, j):
            if 0 <= k < len(tt):
                nearest_t = min(nearest_t, abs(tt[k] - t_of[i]))
    assert nearest_t >= HOLDOUT_GAP_S - 1e-6, f"{label}: train row {nearest_t:.2f}s from a holdout row"
    marks = cell_holdout_labels(rows, spec)
    classes = {}
    for c in spec["cells"]:
        classes[c["class"]] = classes.get(c["class"], 0) + 1
    print(
        f"{label}: cells seed={seed} held={len(spec['cells'])} {classes} train={len(train)} holdout={len(hold)} "
        f"dropped={marks.count('gap')} holdout_share={len(hold) / len(rows):.1%} nearest_train_m={near:.1f} "
        f"nearest_train_s={nearest_t:.2f} deterministic=yes seed{seed + 1}_differs={other['cells'] != spec['cells']} OK"
    )
    print(f"{label}: held cells {[(c['cx'], c['cy'], c['class'], c['rows']) for c in spec['cells']]}")
    return spec, len(train), len(hold), marks.count("gap")


def grid_rows():
    """A drive over a 2 km x 2 km grid, x roads then y roads. West half fast, east half slow."""
    rows, i, t = [], 0, 1000.0
    for lap in range(2):
        for line in range(0, 2000, 250):
            for step in range(0, 2000, 5):
                x, y = (step, line + 10.0) if lap == 0 else (line + 10.0, step)
                v = 25.0 if x < 1000 else 8.0
                rows.append({"i": i, "t": t, "dt_s": 0.2, "segment": "grid#0", "ego": {"pos": [float(x), float(y), 0.0], "speed_mps": v}})
                i += 1
                t += 0.2
            t += 30.0
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default=os.environ.get("GVD_TEST_STRIPS", ""))
    parser.add_argument("--base", default="", help="git rev of the pre-option continual.py")
    args = parser.parse_args()
    base = old_module(args.base) if args.base else None

    # Flat split helper: K=1 matches the original tail cut.
    rows = synthetic_rows()
    pre, hold = split_segment_time_holdout(rows)
    if base is not None:
        b_pre, b_hold = base.split_segment_time_holdout(rows)
        assert [r["i"] for r in pre] == [r["i"] for r in b_pre] and [r["i"] for r in hold] == [r["i"] for r in b_hold]
    pieces, slices = split_segment_time_blocks(rows, 5)
    assert len(slices) == 5 and len(pieces) == 5, (len(pieces), len(slices))
    span = rows[-1]["t"] - rows[0]["t"]
    for k, part in enumerate(slices):
        start = rows[0]["t"] + k * span / 5
        assert abs(part[0]["t"] - (start + 0.8 * span / 5)) < 0.21, (k, part[0]["t"])
    os.environ["GVD_HOLDOUT_BLOCKS"] = "5"
    assert holdout_blocks_from_env() == 5
    os.environ["GVD_HOLDOUT_BLOCKS"] = "0"
    assert holdout_blocks_from_env() == 1
    del os.environ["GVD_HOLDOUT_BLOCKS"]
    assert holdout_blocks_from_env() == 1
    run_checks(rows, "synthetic one segment", base)
    untagged = [{k: v for k, v in r.items() if k != "segment"} for r in rows]
    run_checks(untagged, "synthetic untagged", base)
    multi = synthetic_rows(1500, segment="a") + [dict(r, i=r["i"] + 10000, t=r["t"] + 5000, segment="b") for r in synthetic_rows(1500, segment="b")]
    k1m = pass_plan(multi, CHUNK, chunk_stride(CHUNK))
    k5m = pass_plan(multi, CHUNK, chunk_stride(CHUNK), blocks=5)
    assert plan_key(k1m) == plan_key(k5m), "short segments must keep whole-segment holdout"
    print("synthetic short segments: K=5 keeps the whole-segment holdout OK")
    grid = grid_rows()
    spec, *_ = check_cells(grid, "synthetic grid")
    assert {c["class"] for c in spec["cells"]} == {"fast", "slow"}, "both speed classes must be held out"

    if args.real:
        os.environ.setdefault("GVD_GT_FILE", "gt_painted_v2.jsonl")
        from python.data.raw_frames import load_scene_rows
        from python.train.scene_loader import select_training_rows

        t0 = time.time()
        real, _missing, _excluded = select_training_rows(Path(args.real), load_scene_rows(Path(args.real)))
        print(f"real index {args.real}: {len(real)} rows ({time.time() - t0:.0f}s)")
        out = run_checks(real, "real", base)
        if len(real) == 17018:
            tr, ho, gp = out[5]
            assert abs(tr - 13353) <= 150 and abs(ho - 3297) <= 100, out[5]
            print(f"real K=5 vs bias_split.py estimate 13353/3297/368: {tr}/{ho}/{gp} OK")
        for seed in (0, 1):
            check_cells(real, "real", seed=seed)
    print("test_holdout_blocks: all OK")


if __name__ == "__main__":
    main()
