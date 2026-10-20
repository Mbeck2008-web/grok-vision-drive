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

``--mask-phases`` (on with ``--mask-events``) also masks every row whose
``excursion.phase`` is ``weave``, ``lane_change``, ``abort`` or ``recover``.
``--mask-files a.jsonl,b.jsonl`` masks the rows listed in offline mask
sidecars next to the recording (one JSON object per line, row id in ``i``).
A listed file that is missing, a line that is not JSON with an integer
``i``, or an ``i`` the recording does not have raises ``MaskFileError``.
``row_masks`` computes every source on the full recording rows and returns
the masked row ids per source and their union.

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
MASK_PHASES = ("weave", "lane_change", "abort", "recover")


class MaskFileError(ValueError):
    """A ``--mask-files`` sidecar is missing, malformed, or names a row the recording does not have."""


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

def phase_mask(
    rows: list[Mapping[str, Any]],
    phases: Iterable[str] = MASK_PHASES,
    *,
    keep_abort_reasons: str | Iterable[str] | None = (),
) -> list[bool]:
    """Per-row True when ``excursion.phase`` is one of ``phases``.

    An abort row whose reason is in ``keep_abort_reasons`` is left in, as in
    ``event_mask``.
    """
    wanted = frozenset(phases)
    keep = parse_reasons(keep_abort_reasons)
    out: list[bool] = []
    for row in rows:
        phase = _phase(row)
        out.append(phase in wanted and not (phase == "abort" and _abort_reason(row) in keep))
    return out


def parse_mask_files(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """``"a.jsonl,b.jsonl"`` or an iterable -> names in order, duplicates refused."""
    if value is None:
        return ()
    items = value.split(",") if isinstance(value, str) else list(value)
    names = tuple(str(item).strip() for item in items if str(item).strip())
    if len(set(names)) != len(names):
        raise MaskFileError(f"--mask-files lists a file twice: {','.join(names)}")
    return names


def read_mask_file(root: Path | str, name: str, known: set[int] | frozenset[int]) -> set[int]:
    """Row ids listed in one sidecar (relative to ``root``). Every id must be in ``known``."""
    path = Path(root) / name
    if not path.is_file():
        raise MaskFileError(f"mask file {name} not found in the recording folder")
    ids: set[int] = set()
    unknown: list[int] = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                index = item["i"]
                if isinstance(index, bool) or not isinstance(index, int):
                    raise TypeError(index)
            except (ValueError, KeyError, TypeError) as exc:
                raise MaskFileError(f"mask file {name} line {number}: not a JSON object with an integer i ({exc!r})") from None
            if index not in known:
                unknown.append(index)
            ids.add(index)
    if unknown:
        shown = ",".join(str(i) for i in unknown[:5]) + (" ..." if len(unknown) > 5 else "")
        raise MaskFileError(f"mask file {name}: {len(unknown)} row id(s) not in the recording ({shown})")
    return ids


def _row_id(row: Mapping[str, Any]) -> int | None:
    try:
        return int(row.get("i"))
    except (TypeError, ValueError):
        return None


def row_masks(
    root: Path | str,
    rows: list[Mapping[str, Any]],
    *,
    events: bool = False,
    phases: bool = False,
    files: Iterable[str] = (),
    keep_abort_reasons: str | Iterable[str] | None = (),
) -> tuple[list[tuple[str, set[int]]], set[int], dict[str, Any]]:
    """(per-source masked row ids in order, their union, event counts).

    ``rows`` are the full recording rows (before frame or camera filtering),
    so impact windows and settle runs see every tick. Sources: ``events``
    (``event_mask``), ``phases`` (``phase_mask``), then each mask file.
    """
    sources: list[tuple[str, set[int]]] = []
    counts: dict[str, Any] = {}
    ids = [_row_id(row) for row in rows]
    if events:
        mask, counts = event_mask(rows, read_impacts(root), keep_abort_reasons=keep_abort_reasons)
        sources.append(("events", {i for i, drop in zip(ids, mask) if drop and i is not None}))
    if phases:
        mask = phase_mask(rows, keep_abort_reasons=keep_abort_reasons)
        sources.append(("phases(" + ",".join(MASK_PHASES) + ")", {i for i, drop in zip(ids, mask) if drop and i is not None}))
    names = parse_mask_files(files)
    if names:
        known = frozenset(i for i in ids if i is not None)
        for name in names:
            sources.append((f"file {name}", read_mask_file(root, name, known)))
    union: set[int] = set()
    for _name, masked in sources:
        union |= masked
    return sources, union, counts


def _has_main(row: Mapping[str, Any]) -> bool:
    cams = row.get("cams")
    return isinstance(cams, Mapping) and bool(cams.get("main"))


def _center_valid(row: Mapping[str, Any]) -> bool:
    from python.debug.lane_targets import lane_center_is_valid

    tech = row.get("tech")
    return isinstance(tech, Mapping) and lane_center_is_valid(tech.get("lane_gt_valid"))


def mask_report(
    rows: list[Mapping[str, Any]], sources: list[tuple[str, set[int]]], union: set[int]
) -> tuple[list[str], dict[str, int]]:
    """One log line per source (rows and main frames it masks) and a union line.

    A main frame is a row whose ``cams.main`` is set. Trainable valid counts
    unmasked main frames whose 5 m and 10 m lane centres are valid.
    """
    main_ids = {i for i in (_row_id(r) for r in rows if _has_main(r)) if i is not None}
    total = {"rows": len(rows), "main": len(main_ids)}
    lines = []
    for name, masked in sources:
        lines.append(f"[GVD] mask source {name}: {len(masked)} rows, {len(masked & main_ids)} main frames")
    masked_main = len(union & main_ids)
    valid = sum(1 for r in rows if _has_main(r) and _row_id(r) not in union and _center_valid(r))
    total.update(masked_rows=len(union), masked_main=masked_main, trainable_main=len(main_ids) - masked_main,
                 trainable_valid=valid)
    share = 100.0 * masked_main / max(1, len(main_ids))
    lines.append(
        f"[GVD] mask union: {len(union)} of {len(rows)} rows; {masked_main} of {len(main_ids)} main frames "
        f"({share:.2f}%); trainable main {len(main_ids) - masked_main}, trainable valid "
        f"(5 m + 10 m centre) {valid}"
    )
    return lines, total


def drop_masked(rows: list[Mapping[str, Any]], masked: set[int] | frozenset[int]) -> list[dict[str, Any]]:
    """``rows`` without the masked row ids, ``dt_s`` recomputed.

    A masked run becomes one gap: past MEMORY_S it resets the GRU in
    ``recording_spans`` and in the holdout forward (``eval_resets``), the same
    as any other removal.
    """
    from python.train.scene_loader import recompute_dt

    return recompute_dt([row for row in rows if _row_id(row) not in masked])
