#!/usr/bin/env python3
"""Write ``abort_aggression.jsonl`` for a recording whose aborts left the AI at aggression 0.2.

The collector's abort sends ``ai.setAggression(0.2)``. In BeamNG.tech
0.39.4.0 that sets ai.lua's ``M.extAggression``, and nothing the collector
sent afterwards undid it (``setMode`` resets aggression to ``M.extAggression``
itself), so after the first abort of a segment the AI drove the rest of that
segment at 0.2 instead of the default 0.3. Collectors with the reset write
``ai_aggression.reset_to`` into ``collect.json``; their recordings get no
sidecar.

Rule, per segment (each segment is a level load with a new vehicle, so it
starts at the default): every non-abort row after the end of the segment's
first abort is marked ``ai_aggression_est = 0.2``. Abort rows themselves ran
at 0.2 by design and are not marked.

Read-only on the recording. This tool writes a NEW, create-only sidecar
beside it (or in ``--out-dir``) and refuses to overwrite one:

* ``abort_aggression.jsonl``: one line per marked row, ``{"i", "t",
  "segment", "phase", "ai_aggression_est", "since_abort_end_s", "reason"}``.
* ``abort_aggression.meta.json``: the verdict, the affected segments, counts.

    py -3 scripts/abort_aggression_tag.py <recording dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SIDECAR = "abort_aggression.jsonl"
META = "abort_aggression.meta.json"
REASON = "abort_setAggression_0.2_not_reset"
ABORT_AGGRESSION = 0.2
DEFAULT_AGGRESSION = 0.3
VERSION = 1


def _rows(state: Path):
    with state.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _phase(row: dict) -> Any:
    exc = row.get("excursion")
    return exc.get("phase") if isinstance(exc, dict) else exc


def judge(rec: Path) -> dict[str, Any]:
    """Scan ``state.jsonl`` (read-only). Returns the verdict, the affected segments and the rows to mark."""
    state = rec / "state.jsonl"
    if not state.is_file():
        raise FileNotFoundError(f"no state.jsonl in {rec}")
    code_sha = None
    reset_in_collector = False
    collect = rec / "collect.json"
    if collect.is_file():
        try:
            info = json.loads(collect.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            info = {}
        code_sha = info.get("code_sha")
        agg = info.get("ai_aggression")
        reset_in_collector = isinstance(agg, dict) and agg.get("reset_to") is not None
    marks: list[dict[str, Any]] = []
    segments: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    rows = 0
    state_by_seg: dict[Any, dict[str, Any]] = {}
    for row in _rows(state):
        rows += 1
        seg = row.get("segment")
        st = state_by_seg.setdefault(seg, {"in_abort": False, "end_t": None, "end_i": None, "aborts": 0})
        if seg not in segments:
            segments[seg] = {"segment": seg, "rows": 0, "aborts": 0, "first_abort_end_i": None,
                             "first_abort_end_t": None, "rows_after_abort": 0}
            order.append(seg)
        info_seg = segments[seg]
        info_seg["rows"] += 1
        ph = _phase(row)
        if ph == "abort":
            if not st["in_abort"]:
                st["aborts"] += 1
                info_seg["aborts"] += 1
            st["in_abort"] = True
            continue
        if st["in_abort"]:
            st["in_abort"] = False
            if st["end_t"] is None:
                st["end_t"], st["end_i"] = row.get("t"), row.get("i")
                info_seg["first_abort_end_t"], info_seg["first_abort_end_i"] = st["end_t"], st["end_i"]
        if st["end_t"] is None:
            continue
        info_seg["rows_after_abort"] += 1
        since = None
        try:
            since = round(float(row.get("t")) - float(st["end_t"]), 3)
        except (TypeError, ValueError):
            since = None
        marks.append({
            "i": row.get("i"),
            "t": row.get("t"),
            "segment": seg,
            "phase": ph,
            "ai_aggression_est": ABORT_AGGRESSION,
            "since_abort_end_s": since,
            "reason": REASON,
        })
    affected = [segments[s] for s in order if segments[s]["rows_after_abort"] > 0]
    if reset_in_collector:
        verdict = "reset_in_collector"
    elif not any(segments[s]["aborts"] for s in order):
        verdict = "no_aborts"
    elif not affected:
        verdict = "no_rows_after_abort"
    else:
        verdict = "aggression_not_reset"
    return {
        "verdict": verdict,
        "code_sha": code_sha,
        "rows": rows,
        "segments": len(order),
        "affected_segments": affected if verdict == "aggression_not_reset" else [],
        "marks": marks if verdict == "aggression_not_reset" else [],
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
        rule="per segment (a level load, new vehicle at the default 0.3): every non-abort row after the end of "
             "the segment's first abort ran at the abort's ai.setAggression(0.2), never reset",
        default_aggression=DEFAULT_AGGRESSION,
        source="state.jsonl and collect.json (read only)",
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
        f"abort_aggression {args.recording.name}: verdict={result['verdict']} code_sha={result['code_sha']} "
        f"rows={result['rows']} segments={result['segments']} affected={len(result['affected_segments'])} "
        f"to_mark={len(result['marks'])}"
    )
    for seg in result["affected_segments"]:
        print(
            f"  segment {seg['segment']}: aborts={seg['aborts']} first_abort_end_i={seg['first_abort_end_i']} "
            f"rows_after_abort={seg['rows_after_abort']} of {seg['rows']}"
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
        print(f"abort_aggression: {exc}", file=sys.stderr)
        return 3
    print(f"wrote {side} and {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
