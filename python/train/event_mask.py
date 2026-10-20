"""Mask rows with the collector sidecars (PR 89): ``--mask-events``.

* ``impacts.jsonl``: each ``{"kind": "impact", "t", "pre_t0", ...}`` masks rows
  with ``pre_t0 <= t <= end`` where ``end`` is the ``t`` of the next
  ``impact_end`` of that segment. Rows are matched by time ``t`` only, never by
  ``i``: an impact whose own tick was not written has ``i_row="previous"`` (``i``
  is an earlier row) or ``i=null``. An impact with no end line yet (the
  recording is still running, or the collector stopped) masks to the end of
  its segment.
* Row flags written by the collector: ``settle`` (first 3 s after a spawn,
  teleport, recover or level load) and ``stall`` (under 1 m/s while the AI
  should move).

Read-only: nothing is written next to the recording.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

IMPACTS_FILE = "impacts.jsonl"
ROW_FLAGS = ("settle", "stall")


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_impacts(root: Path | str) -> list[dict[str, Any]]:
    path = Path(root) / IMPACTS_FILE
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with open(path, "rb") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue  # a half-written last line of a live recording
            if isinstance(item, dict):
                out.append(item)
    return out


def impact_spans(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """[pre_t0, end] per impact. ``end`` is None when no impact_end follows."""
    spans: list[dict[str, Any]] = []
    open_by_seg: dict[Any, dict[str, Any]] = {}
    for item in sorted(records, key=lambda r: (_num(r.get("t")) or 0.0, 0 if r.get("kind") == "impact" else 1)):
        kind = item.get("kind")
        seg = item.get("segment")
        t = _num(item.get("t"))
        if t is None:
            continue
        if kind == "impact":
            t0 = _num(item.get("pre_t0"))
            span = {"t0": min(t, t0) if t0 is not None else t, "t_impact": t, "end": None, "segment": seg,
                    "i_row": item.get("i_row")}
            spans.append(span)
            prior = open_by_seg.get(seg)
            if prior is not None and prior["end"] is None:
                prior["end"] = t  # a second impact before the first ended
            open_by_seg[seg] = span
        elif kind == "impact_end":
            span = open_by_seg.pop(seg, None)
            if span is not None and span["end"] is None:
                span["end"] = max(t, span["t_impact"])
    return spans


def event_mask(
    rows: list[Mapping[str, Any]],
    impacts: Iterable[Mapping[str, Any]] = (),
    *,
    flags: tuple[str, ...] = ROW_FLAGS,
) -> tuple[list[bool], dict[str, int]]:
    """Per-row True = masked, plus counts by reason (a row counts once, first reason wins)."""
    spans = impact_spans(impacts)
    seg_end: dict[Any, float] = {}
    for row in rows:
        t = _num(row.get("t"))
        if t is not None:
            seg = row.get("segment")
            seg_end[seg] = max(seg_end.get(seg, t), t)
    windows = []
    for span in spans:
        end = span["end"]
        if end is None:
            end = seg_end.get(span["segment"], math.inf) if span["segment"] in seg_end else math.inf
        windows.append((span["t0"], end))
    windows.sort()
    counts = {"impact": 0, **{name: 0 for name in flags}, "masked": 0, "impact_spans": len(spans),
              "impact_open": sum(1 for s in spans if s["end"] is None)}
    mask: list[bool] = []
    for row in rows:
        t = _num(row.get("t"))
        why = None
        if t is not None and any(a <= t <= b for a, b in windows):
            why = "impact"
        else:
            for name in flags:
                if row.get(name):
                    why = name
                    break
        mask.append(why is not None)
        if why is not None:
            counts[why] += 1
            counts["masked"] += 1
    return mask, counts


def mask_event_rows(root: Path | str, rows: list[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    """Rows with the event rows removed, and the counts."""
    mask, counts = event_mask(rows, read_impacts(root))
    return [row for row, drop in zip(rows, mask) if not drop], counts


def describe(counts: Mapping[str, int], total: int) -> str:
    share = 100.0 * counts.get("masked", 0) / max(1, int(total))
    return (
        f"masked {counts.get('masked', 0)} of {int(total)} rows ({share:.1f}%): "
        f"impact window {counts.get('impact', 0)} ({counts.get('impact_spans', 0)} impacts, "
        f"{counts.get('impact_open', 0)} without an end line), settle {counts.get('settle', 0)}, "
        f"stall {counts.get('stall', 0)}"
    )