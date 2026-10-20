#!/usr/bin/env python3
"""Write ``excursion_validity.jsonl`` for a recording whose lines never reached the AI.

Collectors before the ``set_line`` fix sent ``ai.setLine(...)`` for every
weave / lane change / recover. BeamNG.tech 0.39.4.0's vehicle ``ai`` has no
``setLine``: the call failed, the ``setMode("manual")`` before it had dropped
the span route, and the car ran the move on held inputs. The recorded
``excursion.target_offset_m`` / ``target_heading_rad`` / ``direction`` on
those rows describe a line the car never drove.

This tool does not touch ``state.jsonl`` or any existing tag. It reads the
recording and writes a NEW, create-only sidecar beside it (it refuses to
overwrite one):

* ``excursion_validity.jsonl``: one line per weave / lane_change / recover
  row, ``{"i", "t", "phase", "target_offset_m_valid": false,
  "target_heading_rad_valid": false, "direction_valid": false, "reason"}``.
* ``excursion_validity.meta.json``: how the recording was judged.

A recording counts as a setLine recording when none of its active rows has
``excursion.line_ack`` (the readback ack the fixed collector writes on every
active row). The collector commit in ``collect.json`` (``code_sha``) is
reported alongside. A recording with acks, or with no active rows, gets no
sidecar.

    py -3 scripts/excursion_validity.py <recording dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ACTIVE = ("weave", "lane_change", "recover")
SIDECAR = "excursion_validity.jsonl"
META = "excursion_validity.meta.json"
REASON = "ai_setLine_not_in_beamng_0.39.4.0"
VERSION = 1


def _rows(state: Path):
    with state.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def judge(rec: Path) -> dict[str, Any]:
    """Scan ``state.jsonl`` (read-only). Returns the verdict and the rows to mark."""
    state = rec / "state.jsonl"
    if not state.is_file():
        raise FileNotFoundError(f"no state.jsonl in {rec}")
    code_sha = None
    collect = rec / "collect.json"
    if collect.is_file():
        try:
            code_sha = json.loads(collect.read_text(encoding="utf-8")).get("code_sha")
        except (OSError, json.JSONDecodeError):
            code_sha = None
    marks: list[dict[str, Any]] = []
    acked = 0
    rows = 0
    by_phase: dict[str, int] = {}
    for row in _rows(state):
        rows += 1
        exc = row.get("excursion") or {}
        phase = exc.get("phase")
        if phase not in ACTIVE:
            continue
        by_phase[phase] = by_phase.get(phase, 0) + 1
        if "line_ack" in exc:
            acked += 1
            continue
        marks.append({
            "i": row.get("i"),
            "t": row.get("t"),
            "phase": phase,
            "target_offset_m_valid": False,
            "target_heading_rad_valid": False,
            "direction_valid": False,
            "reason": REASON,
        })
    active = sum(by_phase.values())
    if active == 0:
        verdict = "no_active_rows"
    elif acked == 0:
        verdict = "set_line_raw_lua"
    elif acked == active:
        verdict = "line_acked"
    else:
        verdict = "mixed"
    return {
        "verdict": verdict,
        "code_sha": code_sha,
        "rows": rows,
        "active_rows": active,
        "active_by_phase": by_phase,
        "acked_rows": acked,
        "marks": marks if verdict in ("set_line_raw_lua", "mixed") else [],
    }


def write_sidecar(rec: Path, result: dict[str, Any], out_dir: Path | None = None) -> tuple[Path, Path]:
    """Create-only: both files are opened with mode ``x``; an existing sidecar is an error."""
    out_dir = Path(out_dir or rec)
    side, meta = out_dir / SIDECAR, out_dir / META
    for path in (side, meta):
        if path.exists():
            raise FileExistsError(f"{path} exists; this tool never overwrites a sidecar")
    with side.open("x", encoding="utf-8") as fh:
        for mark in result["marks"]:
            fh.write(json.dumps(mark, separators=(",", ":")) + "\n")
    info = {k: v for k, v in result.items() if k != "marks"}
    info.update(
        version=VERSION,
        marked_rows=len(result["marks"]),
        reason=REASON,
        rule="active rows (weave, lane_change, recover) without excursion.line_ack were driven on held "
             "inputs: the collector's raw ai.setLine does not exist in 0.39.4.0, so their target_offset_m, "
             "target_heading_rad and direction are not what the car did",
        source="state.jsonl (read only)",
    )
    with meta.open("x", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
        fh.write("\n")
    return side, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recording", type=Path, help="recording dir with state.jsonl (and collect.json)")
    ap.add_argument("--out-dir", type=Path, default=None, help="write the sidecar here instead of beside state.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args(argv)
    result = judge(args.recording)
    print(
        f"excursion_validity {args.recording.name}: verdict={result['verdict']} code_sha={result['code_sha']} "
        f"rows={result['rows']} active={result['active_rows']} {result['active_by_phase']} "
        f"acked={result['acked_rows']} to_mark={len(result['marks'])}"
    )
    if not result["marks"]:
        print("nothing to mark; no sidecar written")
        return 0
    if args.dry_run:
        print("dry run; no sidecar written")
        return 0
    try:
        side, meta = write_sidecar(args.recording, result, args.out_dir)
    except FileExistsError as exc:
        print(f"excursion_validity: {exc}", file=sys.stderr)
        return 3
    print(f"wrote {side} and {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
