"""Mask rows with the collector sidecars (PR 89): ``--mask-events``.

* ``impacts.jsonl``: each ``{"kind": "impact", "t", "pre_t0", ...}`` masks rows
  with ``t0 <= t <= end``. Rows are matched by time ``t`` only, never by
  ``i``: an impact whose own tick was not written has ``i_row="previous"`` (``i``
  is an earlier row) or ``i=null``.

  ``t0`` is the earliest of ``t - PRE_IMPACT_S`` (3 s), ``pre_t0``, the
  collector's ``abort_t0``, and the first row of the run of
  ``excursion.phase == "abort"`` rows that ends at the impact: an abort phase
  that ends in an impact is masked whole.

  ``end`` runs past the ``impact_end`` line (``drove_on`` closes it while the
  car may still be damaged) to the last row before the next respawn of that
  segment: the next row tagged ``recover``, or the first row of a new
  ``settle`` run (every spawn, teleport and recover starts one). With no
  respawn in the rows the window runs to the end of the segment, which also
  covers an impact with no end line yet.
* ``excursion.phase == "abort"``: every abort row is masked whatever its
  ``excursion.reason`` (lane_edge, no_path, line_end, ...), unless the reason
  is listed in ``keep_abort_reasons`` (``--keep-abort-reasons``, default
  empty). An abort run with no impact is masked too. Counts are kept per
  reason.
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
PRE_IMPACT_S = 3.0


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
                    "i_row": item.get("i_row"), "abort_t0": _num(item.get("abort_t0"))}
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


def _phase(row: Mapping[str, Any]) -> Any:
    exc = row.get("excursion")
    return exc.get("phase") if isinstance(exc, Mapping) else exc


def _abort_reason(row: Mapping[str, Any]) -> str:
    exc = row.get("excursion")
    reason = exc.get("reason") if isinstance(exc, Mapping) else None
    return str(reason or "")


def parse_reasons(value: str | Iterable[str] | None) -> frozenset[str]:
    """``"a,b"`` or an iterable -> a set of non-empty, stripped reasons."""
    if value is None:
        return frozenset()
    items = value.split(",") if isinstance(value, str) else list(value)
    return frozenset(str(item).strip() for item in items if str(item).strip())


def _seg_rows(rows: list[Mapping[str, Any]]) -> dict[Any, list[tuple[float, Mapping[str, Any]]]]:
    by_seg: dict[Any, list[tuple[float, Mapping[str, Any]]]] = {}
    for row in rows:
        t = _num(row.get("t"))
        if t is not None:
            by_seg.setdefault(row.get("segment"), []).append((t, row))
    for items in by_seg.values():
        items.sort(key=lambda item: item[0])
    return by_seg


def _abort_start(items: list[tuple[float, Mapping[str, Any]]], t_impact: float) -> float | None:
    """t of the first row of the abort run that reaches the impact, or None."""
    k = max((n for n, (t, _row) in enumerate(items) if t <= t_impact), default=None)
    if k is None or _phase(items[k][1]) != "abort":
        return None
    while k > 0 and _phase(items[k - 1][1]) == "abort":
        k -= 1
    return items[k][0]


def _before_respawn(items: list[tuple[float, Mapping[str, Any]]], t_impact: float) -> float:
    """t of the last row before the next respawn after ``t_impact``; segment end without one."""
    last = t_impact
    prev_settle = False
    for t, row in items:
        settle = bool(row.get("settle"))
        if t > t_impact and (bool(row.get("recover")) or (settle and not prev_settle)):
            return last
        prev_settle = settle
        if t > t_impact:
            last = t
    return last if items else math.inf


def impact_windows(rows: list[Mapping[str, Any]], spans: list[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    """``(t0, end, old_t0, old_end)`` per impact; old is the pre_t0..impact_end span."""
    by_seg = _seg_rows(rows)
    out = []
    for span in spans:
        items = by_seg.get(span["segment"], [])
        t_imp = float(span["t_impact"])
        old_end = span["end"]
        if old_end is None:
            old_end = items[-1][0] if items else math.inf
        t0 = min(span["t0"], t_imp - PRE_IMPACT_S)
        if span.get("abort_t0") is not None:
            t0 = min(t0, float(span["abort_t0"]))
        start = _abort_start(items, t_imp)
        if start is not None:
            t0 = min(t0, start)
        end = max(old_end, _before_respawn(items, t_imp)) if items else old_end
        out.append((t0, end, span["t0"], old_end))
    return out


def event_mask(
    rows: list[Mapping[str, Any]],
    impacts: Iterable[Mapping[str, Any]] = (),
    *,
    flags: tuple[str, ...] = ROW_FLAGS,
    keep_abort_reasons: str | Iterable[str] | None = (),
) -> tuple[list[bool], dict[str, Any]]:
    """Per-row True = masked, plus counts by reason (a row counts once, first reason wins).

    Order: impact window, then an abort row (``abort``, with the per-reason
    split in ``abort_reasons``), then the row flags. ``abort_kept`` counts
    abort rows left in because their reason is in ``keep_abort_reasons``.

    ``impact_extended`` counts the impact rows outside the old pre_t0..impact_end span
    (the 3 s floor, the abort run, and after the impact until the next respawn).
    """
    spans = impact_spans(impacts)
    full = impact_windows(rows, spans)
    windows = sorted((a, b) for a, b, _oa, _ob in full)
    old = [(oa, ob) for _a, _b, oa, ob in full]
    keep = parse_reasons(keep_abort_reasons)
    abort_reasons: dict[str, int] = {}
    counts: dict[str, Any] = {"impact": 0, "abort": 0, **{name: 0 for name in flags}, "masked": 0,
                              "impact_spans": len(spans), "impact_open": sum(1 for s in spans if s["end"] is None),
                              "impact_extended": 0, "abort_kept": 0, "abort_reasons": abort_reasons}
    mask: list[bool] = []
    for row in rows:
        t = _num(row.get("t"))
        why = None
        if t is not None and any(a <= t <= b for a, b in windows):
            why = "impact"
            if not any(a <= t <= b for a, b in old):
                counts["impact_extended"] += 1
        elif _phase(row) == "abort" and _abort_reason(row) not in keep:
            why = "abort"
            reason = _abort_reason(row) or "(none)"
            abort_reasons[reason] = abort_reasons.get(reason, 0) + 1
        else:
            if _phase(row) == "abort":
                counts["abort_kept"] += 1
            for name in flags:
                if row.get(name):
                    why = name
                    break
        mask.append(why is not None)
        if why is not None:
            counts[why] += 1
            counts["masked"] += 1
    return mask, counts


def mask_event_rows(
    root: Path | str,
    rows: list[Mapping[str, Any]],
    *,
    keep_abort_reasons: str | Iterable[str] | None = (),
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Rows with the event rows left out of training, and the counts. Nothing on disk is deleted."""
    mask, counts = event_mask(rows, read_impacts(root), keep_abort_reasons=keep_abort_reasons)
    return [row for row, drop in zip(rows, mask) if not drop], counts


def describe(counts: Mapping[str, Any], total: int) -> str:
    share = 100.0 * counts.get("masked", 0) / max(1, int(total))
    by_reason = counts.get("abort_reasons") or {}
    reasons = ",".join(f"{key}={by_reason[key]}" for key in sorted(by_reason)) or "none"
    return (
        f"masked {counts.get('masked', 0)} of {int(total)} rows ({share:.1f}%): "
        f"impact window {counts.get('impact', 0)} ({counts.get('impact_spans', 0)} impacts, "
        f"{counts.get('impact_open', 0)} without an end line, {counts.get('impact_extended', 0)} from the "
        f"3 s / abort run / until-respawn extension), abort {counts.get('abort', 0)} "
        f"(by reason: {reasons}; {counts.get('abort_kept', 0)} kept by --keep-abort-reasons), "
        f"settle {counts.get('settle', 0)}, stall {counts.get('stall', 0)}"
    )