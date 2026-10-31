#!/usr/bin/env python3
"""Offline junction labels for recorded runs: ``jct.jsonl`` and ``jct_exit.jsonl``.

    py -3 scripts/label_junctions.py <run folder> [<run folder> ...]

Each run folder holds a ``state.jsonl``. The labels come from the same
``JctSidecar`` / ``JunctionLabeler`` code the recorder runs live, so a relabel
of a recorded run gives the same ``jct.jsonl`` bytes. ``jct_exit.jsonl`` (the
turn taken at each junction driven through) needs the row after the junction,
so it is written here only.

Both files are create-only: an existing file is kept and reported.

A collector may still be writing these folders. Before any ``state.jsonl`` is
opened, every folder is checked, and the run stops with exit 3 when the
collector heartbeat (``<status>/collect_heartbeat.json``) says ``running`` and
is under 60 s old, or a ``state.jsonl`` was modified under 120 s ago.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.collect import junction_map as jm  # noqa: E402

HEARTBEAT_FILE = "collect_heartbeat.json"
HEARTBEAT_FRESH_S = 60.0
STATE_FRESH_S = 120.0
BUSY = 3


def now_s() -> float:
    """Wall clock, in one place so the tests can fix it."""
    return time.time()


def _default_status() -> Path | None:
    try:
        from python.debug.remote_status import status_dir

        return status_dir()
    except Exception:  # noqa: BLE001 - no data disk: no heartbeat to read
        return None


def collector_busy(status: Path | None, folders: list[Path], now: float) -> str | None:
    """Why labelling must wait, or None. Opens no ``state.jsonl``."""
    if status is not None:
        hb = Path(status) / HEARTBEAT_FILE
        try:
            with open(hb, "r", encoding="utf-8") as fh:
                beat = json.load(fh)
        except (OSError, ValueError):
            beat = None
        if isinstance(beat, dict) and beat.get("state") == "running":
            try:
                age = now - float(beat.get("t"))
            except (TypeError, ValueError):
                age = 0.0
            if age < HEARTBEAT_FRESH_S:
                return f"collector heartbeat is running ({age:.0f} s old)"
    for folder in folders:
        try:
            mtime = os.stat(folder / "state.jsonl").st_mtime
        except OSError:
            continue
        if now - mtime < STATE_FRESH_S:
            return f"{folder.name}/state.jsonl changed {now - mtime:.0f} s ago"
    return None


def _read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                row = {}
            rows.append(row if isinstance(row, dict) else {})
    return rows


def label_folder(folder: Path, maps_dir: Path) -> int:
    state = folder / "state.jsonl"
    if not state.is_file():
        print(f"label jct {folder.name}: no state.jsonl")
        return 1
    out = folder / "jct.jsonl"
    if out.exists():
        print(f"label jct {folder.name}: jct.jsonl exists; kept")
        return 0
    rows = _read_rows(state)
    side = jm.JctSidecar(out, maps_dir, log=print, prefix=f"label jct {folder.name}")
    lines = []
    try:
        for row in rows:
            line = side.step(row)
            side.commit(line)
            lines.append(line)
    finally:
        side.close()
    labels = []
    for line in lines:
        try:
            got = json.loads(line) if line else {}
        except ValueError:
            got = {}
        labels.append(got if "err" not in got else {})
    exits = jm.junction_exits(rows, labels)
    exit_path = folder / "jct_exit.jsonl"
    if out.exists():
        try:
            with open(exit_path, "x", encoding="utf-8", newline="\n") as fh:
                for rec in exits:
                    fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        except FileExistsError:
            print(f"label jct {folder.name}: jct_exit.jsonl exists; kept")
    print(f"label jct {folder.name}: {len(rows)} rows, {sum(1 for x in labels if x.get('in_jct'))} in a junction, "
          f"{len(exits)} exits, {side.errors} errors")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="run folders holding state.jsonl")
    ap.add_argument("--junctions", type=Path, default=ROOT / "junctions", help="folder of <map>.json.gz junction maps")
    ap.add_argument("--status-dir", type=Path, default=None, help="collector status folder (default: the data disk's)")
    args = ap.parse_args(argv)
    status = args.status_dir if args.status_dir is not None else _default_status()
    folders = [Path(f) for f in args.runs]
    why = collector_busy(status, folders, now_s())
    if why is not None:
        print(f"label jct: a collector is recording ({why}); try again when it stops")
        return BUSY
    code = 0
    for folder in folders:
        code = max(code, label_folder(folder, args.junctions))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
