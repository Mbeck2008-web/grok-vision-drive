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
* Row flags written by the collector: ``settle`` (first 3 s after a spawn,
  teleport, recover or level load) and ``stall`` (under 1 m/s while the AI
  should move).

``--mask-files a.jsonl,b.jsonl`` masks the rows listed in offline mask
sidecars next to the recording (one JSON object per line, row id in ``i``).
A listed file that is missing, a line that is not JSON with an integer
``i``, or an ``i`` the recording does not have raises ``MaskFileError``.
``name@sha256prefix`` also refuses a file whose sha256 does not start with
that prefix. ``mask_file_sources`` reads every file against the full
recording rows; ``drop_masked`` removes the union before the holdout split,
so a masked row is in neither the train chunks nor the holdout.

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
) -> tuple[list[bool], dict[str, int]]:
    """Per-row True = masked, plus counts by reason (a row counts once, first reason wins).

    ``impact_extended`` counts the impact rows outside the old pre_t0..impact_end span
    (the 3 s floor, the abort run, and after the impact until the next respawn).
    """
    spans = impact_spans(impacts)
    full = impact_windows(rows, spans)
    windows = sorted((a, b) for a, b, _oa, _ob in full)
    old = [(oa, ob) for _a, _b, oa, ob in full]
    counts = {"impact": 0, **{name: 0 for name in flags}, "masked": 0, "impact_spans": len(spans),
              "impact_open": sum(1 for s in spans if s["end"] is None), "impact_extended": 0}
    mask: list[bool] = []
    for row in rows:
        t = _num(row.get("t"))
        why = None
        if t is not None and any(a <= t <= b for a, b in windows):
            why = "impact"
            if not any(a <= t <= b for a, b in old):
                counts["impact_extended"] += 1
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
    """Rows with the event rows left out of training, and the counts. Nothing on disk is deleted."""
    mask, counts = event_mask(rows, read_impacts(root))
    return [row for row, drop in zip(rows, mask) if not drop], counts


def describe(counts: Mapping[str, int], total: int) -> str:
    share = 100.0 * counts.get("masked", 0) / max(1, int(total))
    return (
        f"masked {counts.get('masked', 0)} of {int(total)} rows ({share:.1f}%): "
        f"impact window {counts.get('impact', 0)} ({counts.get('impact_spans', 0)} impacts, "
        f"{counts.get('impact_open', 0)} without an end line, {counts.get('impact_extended', 0)} from the "
        f"3 s / abort run / until-respawn extension), settle {counts.get('settle', 0)}, "
        f"stall {counts.get('stall', 0)}"
    )


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


SHA_PREFIX_MIN = 8


def split_mask_spec(item: str) -> tuple[str, str | None]:
    """``name`` -> (name, None); ``name@sha256prefix`` -> (name, prefix).

    The prefix is 8 to 64 hex digits (case is ignored). Anything else after
    ``@`` raises ``MaskFileError``.
    """
    text = str(item).strip()
    if "@" not in text:
        return text, None
    name, _sep, want = text.rpartition("@")
    want = want.strip().lower()
    if not name.strip() or not (SHA_PREFIX_MIN <= len(want) <= 64) or any(c not in "0123456789abcdef" for c in want):
        raise MaskFileError(
            f"--mask-files entry {text}: expected name@sha256prefix with {SHA_PREFIX_MIN}-64 hex digits"
        )
    return name.strip(), want


def file_sha256(path: Path | str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_file_sources(
    root: Path | str, rows: list[Mapping[str, Any]], files: str | Iterable[str] | None
) -> list[dict[str, Any]]:
    """One entry per ``--mask-files`` item, in order: name, sha256, row ids.

    ``rows`` are the full recording rows (before frame or camera filtering),
    so every listed id is checked against the whole recording. A file named
    twice (with or without ``@sha``), a sha256 that does not start with the
    given prefix, and every ``read_mask_file`` failure raise ``MaskFileError``.
    """
    items = parse_mask_files(files)
    if not items:
        return []
    specs = [split_mask_spec(item) for item in items]
    names = [name for name, _want in specs]
    if len(set(names)) != len(names):
        raise MaskFileError(f"--mask-files lists a file twice: {','.join(names)}")
    known = frozenset(i for i in (_row_id(row) for row in rows) if i is not None)
    out: list[dict[str, Any]] = []
    for name, want in specs:
        path = Path(root) / name
        if not path.is_file():
            raise MaskFileError(f"mask file {name} not found in the recording folder")
        sha = file_sha256(path)
        if want is not None and not sha.startswith(want):
            raise MaskFileError(f"mask file {name}: sha256 {sha} does not start with the expected {want}")
        out.append({"name": name, "sha256": sha, "expected": want, "ids": read_mask_file(root, name, known)})
    return out


def _empty_tech(row: Mapping[str, Any]) -> bool:
    tech = row.get("tech")
    return isinstance(tech, Mapping) and not tech


def selection_line(
    state_rows: list[Mapping[str, Any]],
    before: list[Mapping[str, Any]],
    after: list[Mapping[str, Any]],
    sources: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """The one ``[GVD] selection:`` log line and its counts.

    Empty tech (``tech == {}``, a hop-masked label) has no loss and no
    holdout score; those rows stay in the windows as GRU context, so they are
    counted, not removed. Each mask file is counted on the full recording rows
    and on ``before`` (rows with frames); ``after`` is what remains.
    """
    with_frames = {i for i in (_row_id(row) for row in before) if i is not None}
    empty_state = sum(1 for row in state_rows if _empty_tech(row))
    empty_frames = sum(1 for row in before if _empty_tech(row))
    parts = [
        f"empty tech (no loss, no score; kept as GRU context) {empty_state} of {len(state_rows)} state rows, "
        f"{empty_frames} of {len(before)} rows with frames"
    ]
    files: list[dict[str, Any]] = []
    union: set[int] = set()
    for source in sources:
        ids = set(source["ids"])
        union |= ids
        hit = len(ids & with_frames)
        parts.append(f"file {source['name']} sha256 {source['sha256']} {len(ids)} rows ({hit} with frames)")
        files.append({"name": source["name"], "sha256": source["sha256"], "rows": len(ids), "with_frames": hit})
    if sources:
        parts.append(f"files union {len(union)} rows")
    else:
        parts.append("no mask files")
    parts.append(f"left out {len(before) - len(after)} of {len(before)} rows with frames; {len(after)} remain")
    meta = {"empty_tech_state": empty_state, "empty_tech_frames": empty_frames, "files": files,
            "files_union": len(union), "before": len(before), "after": len(after)}
    return "[GVD] selection: " + "; ".join(parts), meta


def drop_masked(rows: list[Mapping[str, Any]], masked: set[int] | frozenset[int]) -> list[dict[str, Any]]:
    """``rows`` without the masked row ids, ``dt_s`` recomputed.

    A masked run becomes one gap: past MEMORY_S it resets the GRU in
    ``recording_spans``, the same as any other removal.
    """
    from python.train.scene_loader import recompute_dt

    return recompute_dt([row for row in rows if _row_id(row) not in masked])
