"""Continual scene-net schedule.

The GRU runs through a recording. A gap or a new recording clears it.
Backprop covers one short chunk, and the next chunk starts before that
chunk ends (Williams and Peng 1990). The chunk is not an independent clip.

The 360 strip is not linear in azimuth, so this module does not roll it.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from python.train.scene_net import MEMORY_S, step_dt

# AdamW candidates. The run keeps the one with the better holdout loss.
# Kept so the search helper can be unit-tested. train_directory does not
# call it: the run stays at the requested rate, which defaults to 1e-3.
CANDIDATE_LRS = (1e-3, 1e-4)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
HOLDOUT_FRACTION = 0.2
# Rows within this many seconds of the cut are neither train nor holdout.
HOLDOUT_GAP_S = 4.0
# A segment longer than this is cut in time even when other segments exist.
LONG_SEGMENT_S = 15.0 * 60.0
# Time blocks per long segment for the holdout. 1 is the single tail cut.
# ``--holdout-blocks K`` or ``GVD_HOLDOUT_BLOCKS=K`` spreads the holdout over
# K equal blocks so one stretch of road does not decide the score.
HOLDOUT_BLOCKS = 1
# Spatial holdout (``--holdout-cells``): square map cells of this size.
HOLDOUT_CELL_M = 250.0
# Train rows this close to a holdout cell are dropped (neither train nor scored).
HOLDOUT_BUFFER_M = 50.0
# A cell whose median speed is at or above this is "fast" (highway-like),
# otherwise "slow" (city). The AI caps at about 8 and 12.5 m/s here.
HOLDOUT_FAST_MPS = 10.5
HOLDOUT_CELLS_VERSION = 1
CHUNK_MIN = 8
CHUNK_MAX = 16
# A backward chunk has to fit on an 11 GB card.
VRAM_BUDGET_BYTES = 11 * 1024 * 1024 * 1024
# Rare prefix chunks get a second optimizer step. The holdout is not repeated.
RARE_REPEATS = 2


def strip_roll_is_yaw() -> bool:
    """Whether a horizontal roll of the 360 strip is a real yaw.

    It is not a yaw roll. wide, main, and narrow share yaw 0. The rear
    plate covers both repeater aims. Seams are empty gaps, and rear does
    not join back to repeatL. Rolling the pixels would move the labels
    differently from the image.
    """
    return False


def offcenter_view(pixels: Any, labels: Any) -> tuple[Any, Any]:
    """Leave the strip and the ego-frame labels unchanged.

    The stitch is not a yaw roll, so there is no geometry-true shift.
    Pixels are not warped and lanes, curbs, cars, signs, and lights are
    not rotated.
    """
    if strip_roll_is_yaw():
        raise RuntimeError("strip roll is not a yaw; refusing to warp pixels")
    return pixels, labels


# Not a session id. A missing session is None, and that value can continue.
_NO_PRIOR = object()


def should_reset_state(
    prev_session: Any,
    session: Any,
    dt_s: Any,
    *,
    memory_s: float = MEMORY_S,
) -> bool:
    """Clear the GRU when the recording changes or the gap is too long.

    A null ``dt_s`` is a session boundary. A gap longer than ``memory_s``
    (15 seconds) is a new stretch. A shorter real step keeps the state.
    ``prev_session`` is the previous row's id, not a sentinel: the first row
    of a file is handled by ``recording_spans``.
    """
    if session != prev_session:
        return True
    if dt_s is None:
        return True
    try:
        gap = float(dt_s)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(gap) or gap > float(memory_s):
        return True
    return False


def recording_spans(rows: list[Mapping[str, Any]], *, memory_s: float = MEMORY_S) -> list[tuple[int, int]]:
    """Half-open spans. Each span is one recording with no internal reset."""
    if not rows:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    prev: Any = _NO_PRIOR
    for index, row in enumerate(rows):
        session = row.get("session")
        if prev is _NO_PRIOR:
            prev = session
            continue
        if should_reset_state(prev, session, row.get("dt_s"), memory_s=memory_s):
            spans.append((start, index))
            start = index
        prev = session
    spans.append((start, len(rows)))
    return spans


def overlapping_chunks(n: int, length: int, stride: int) -> list[tuple[int, int]]:
    """Backward windows. ``stride < length`` so the next window starts early.

    Williams and Peng 1990: the update begins before the previous truncation
    window has ended. The window length is the gradient span.
    """
    n = int(n)
    if n <= 0:
        return []
    length = max(1, int(length))
    if length == 1:
        return [(i, i + 1) for i in range(n)]
    stride = max(1, min(int(stride), length - 1))
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(n, start + length)
        chunks.append((start, end))
        if end >= n:
            break
        nxt = start + stride
        if nxt <= start:
            break
        start = nxt
    return chunks


def chunk_stride(length: int) -> int:
    """Half the chunk, and always shorter than the chunk when the chunk is."""
    length = int(length)
    if length <= 1:
        return 1
    return max(1, length // 2)


def iter_bptt(
    rows: list[Mapping[str, Any]],
    length: int,
    stride: int,
    *,
    memory_s: float = MEMORY_S,
) -> list[dict[str, Any]]:
    """Chunks in trainer order. The first chunk of a recording sets ``reset``."""
    stride = int(stride)
    chunks: list[dict[str, Any]] = []
    for start, end in recording_spans(rows, memory_s=memory_s):
        for rel_s, rel_e in overlapping_chunks(end - start, length, stride):
            chunks.append(
                {
                    "start": start + rel_s,
                    "end": start + rel_e,
                    "reset": rel_s == 0,
                    "stride": stride if int(length) > 1 else 1,
                }
            )
    return chunks


def simulate_carry(
    rows: list[Mapping[str, Any]],
    length: int,
    stride: int,
    *,
    memory_s: float = MEMORY_S,
) -> list[dict[str, Any]]:
    """Scalar stand-in for the carried GRU.

    Each frame adds ``step_dt``. A chunk that does not reset takes the value
    already produced at its first frame. That value is the state ``stride``
    frames into the previous chunk, so the boundary keeps the hidden state.
    ``grad_frames`` is the backward length, not the whole recording.
    """
    plan = iter_bptt(rows, length, stride, memory_s=memory_s)
    before = [0.0] * len(rows)
    for start, end in recording_spans(rows, memory_s=memory_s):
        hidden = 0.0
        for index in range(start, end):
            before[index] = hidden
            hidden += step_dt(rows[index].get("dt_s"))
    traced: list[dict[str, Any]] = []
    for chunk in plan:
        hidden_in = 0.0 if chunk["reset"] else before[int(chunk["start"])]
        hidden = hidden_in
        at_stride = None
        span = int(chunk["stride"])
        for step_index, index in enumerate(range(int(chunk["start"]), int(chunk["end"]))):
            hidden += step_dt(rows[index].get("dt_s"))
            if step_index + 1 == span:
                at_stride = hidden
        if at_stride is None:
            at_stride = hidden
        traced.append(
            {
                "start": int(chunk["start"]),
                "end": int(chunk["end"]),
                "reset": bool(chunk["reset"]),
                "grad_frames": int(chunk["end"]) - int(chunk["start"]),
                "hidden_in": hidden_in,
                "hidden_at_stride": at_stride,
                "hidden_end": hidden,
            }
        )
    return traced


def holdout_mask(
    rows: list[Mapping[str, Any]],
    fraction: float = HOLDOUT_FRACTION,
    *,
    memory_s: float = MEMORY_S,
) -> list[bool]:
    """True on rows in the last ``fraction`` of each recording's time span.

    The cut is by timestamp, not by row count. A recording with no span
    stays in the prefix.
    """
    mask = [False] * len(rows)
    portion = min(1.0, max(0.0, float(fraction)))
    for start, end in recording_spans(rows, memory_s=memory_s):
        times: list[float] = []
        for index in range(start, end):
            try:
                times.append(float(rows[index]["t"]))
            except (KeyError, TypeError, ValueError):
                times.append(0.0)
        if not times:
            continue
        t0 = min(times)
        span = max(times) - t0
        if span <= 0.0 or portion <= 0.0:
            continue
        cut = t0 + (1.0 - portion) * span
        for index, stamp in zip(range(start, end), times):
            if stamp >= cut:
                mask[index] = True
        if all(mask[index] for index in range(start, end)):
            mask[start] = False
    return mask


def split_prefix_holdout(
    rows: list[Mapping[str, Any]],
    fraction: float = HOLDOUT_FRACTION,
    *,
    memory_s: float = MEMORY_S,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Prefix first, then the untouched holdout, in file order."""
    mask = holdout_mask(rows, fraction, memory_s=memory_s)
    prefix = [row for row, held in zip(rows, mask) if not held]
    holdout = [row for row, held in zip(rows, mask) if held]
    return prefix, holdout


def _sign_name(sign: Mapping[str, Any]) -> str:
    raw = str(sign.get("cls") or sign.get("class") or "").strip().lower()
    return {
        "stop": "stop_sign",
        "light": "traffic_light",
        "trafficlight": "traffic_light",
    }.get(raw, raw)


def _lamp_name(sign: Mapping[str, Any]) -> str:
    raw = str(sign.get("state") or sign.get("lamp") or "").strip().lower()
    return {"r": "red", "amber": "yellow", "y": "yellow", "g": "green"}.get(raw, raw)


def is_rare_row(row: Mapping[str, Any]) -> bool:
    """Unprotected turn, stop sign, or red light. Other rows are ordinary."""
    tech = row.get("tech")
    if not isinstance(tech, dict):
        tech = {}
    if tech.get("unprotected_turn") or row.get("unprotected_turn"):
        return True
    turn = tech.get("turn", row.get("turn"))
    if isinstance(turn, str) and turn.strip().lower() == "unprotected":
        return True
    signs = tech.get("signs")
    if signs is None:
        signs = row.get("signs") or []
    for sign in signs or []:
        if not isinstance(sign, dict):
            continue
        name = _sign_name(sign)
        if name == "stop_sign":
            return True
        if name == "traffic_light" and _lamp_name(sign) == "red":
            return True
    return False


def oversample_repeats(rows: list[Mapping[str, Any]]) -> int:
    """Extra optimizer steps for a rare training chunk. One otherwise."""
    if any(is_rare_row(row) for row in rows):
        return RARE_REPEATS
    return 1


def _segment_groups(rows: list[Mapping[str, Any]]) -> list[tuple[str, list[Mapping[str, Any]]]] | None:
    """Consecutive ``segment`` blocks. None when any row has no segment."""
    if not rows:
        return []
    groups: list[tuple[str, list[Mapping[str, Any]]]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("segment"):
            return None
        seg = str(row.get("segment"))
        if not groups or groups[-1][0] != seg:
            groups.append((seg, [row]))
        else:
            groups[-1][1].append(row)
    return groups


def holdout_segment_names(rows: list[Mapping[str, Any]]) -> set[str]:
    """Whole segments held out of training.

    Two or more short segments: the last one. Five or more: every fifth.
    One segment, or any segment longer than about 15 minutes, is not in
    this set. Those are cut by time inside the segment.
    """
    groups = _segment_groups(rows)
    if not groups or len(groups) < 2:
        return set()
    names = [seg for seg, _rows in groups]
    if len(names) < 5:
        return {names[-1]}
    return {name for index, name in enumerate(names) if (index + 1) % 5 == 0}


def _row_time(row: Mapping[str, Any]) -> float | None:
    try:
        stamp = float(row["t"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(stamp):
        return None
    return stamp


def segment_span_s(rows: list[Mapping[str, Any]]) -> float:
    """Seconds from the first timestamp to the last. Zero when time is missing."""
    times = [stamp for stamp in (_row_time(row) for row in rows) if stamp is not None]
    if len(times) < 2:
        return 0.0
    return max(times) - min(times)


def holdout_blocks_from_env(default: int = HOLDOUT_BLOCKS) -> int:
    """``GVD_HOLDOUT_BLOCKS`` when it is a whole number of at least 1."""
    raw = os.environ.get("GVD_HOLDOUT_BLOCKS", "").strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return value if value >= 1 else int(default)


def split_segment_time_blocks(
    rows: list[Mapping[str, Any]],
    blocks: int,
    fraction: float = HOLDOUT_FRACTION,
    *,
    gap_s: float = HOLDOUT_GAP_S,
) -> tuple[list[list[Mapping[str, Any]]], list[list[Mapping[str, Any]]]]:
    """Train pieces and holdout slices for ``blocks`` equal time blocks.

    Block ``k`` covers ``[t0 + k*span/K, t0 + (k+1)*span/K)``. Its last
    ``fraction`` is held out. ``gap_s`` before each slice and ``gap_s``
    after it (except after the last block, which ends the segment) are
    neither trained nor scored. Train rows between two slices form one
    piece, so a train chunk never spans a holdout slice. A row without a
    time joins the current train piece. Rows stay in file order.
    """
    k_blocks = max(1, int(blocks))
    times = [_row_time(row) for row in rows]
    known = [stamp for stamp in times if stamp is not None]
    if len(known) < 2:
        return ([list(rows)] if rows else []), []
    t0 = min(known)
    span = max(known) - t0
    portion = min(1.0, max(0.0, float(fraction)))
    if span <= 0.0 or portion <= 0.0:
        return [list(rows)], []
    gap = max(0.0, float(gap_s))
    width = span / float(k_blocks)
    pieces: list[list[Mapping[str, Any]]] = []
    slices: list[list[Mapping[str, Any]]] = []
    piece: list[Mapping[str, Any]] = []
    held: list[Mapping[str, Any]] = []
    held_block: int | None = None
    for row, stamp in zip(rows, times):
        if stamp is None:
            piece.append(row)
            continue
        block = min(k_blocks - 1, int((stamp - t0) / width))
        start = t0 + block * width
        end = t0 + (block + 1) * width
        cut = start + (1.0 - portion) * width
        last = block == k_blocks - 1
        if stamp >= cut and (last or stamp < end - gap):
            kind = "hold"
        elif stamp >= cut - gap or (not last and stamp >= end - gap):
            kind = "gap"
        else:
            kind = "train"
        if kind == "hold":
            if held_block != block:
                if held:
                    slices.append(held)
                held = []
                held_block = block
            held.append(row)
            if piece:
                pieces.append(piece)
                piece = []
            continue
        if held:
            slices.append(held)
            held = []
            held_block = None
        if kind == "train":
            piece.append(row)
        elif piece:
            pieces.append(piece)
            piece = []
    if held:
        slices.append(held)
    if piece:
        pieces.append(piece)
    if not slices:
        return [list(rows)], []
    return pieces, slices


def split_segment_time_holdout(
    rows: list[Mapping[str, Any]],
    fraction: float = HOLDOUT_FRACTION,
    *,
    gap_s: float = HOLDOUT_GAP_S,
    blocks: int = 1,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Last ``fraction`` of this segment's time, with a gap at the cut.

    The gap is neither trained nor scored, so the holdout does not start
    on the same second as the last training frame. A span with no holdout
    sample stays entirely in the prefix. ``blocks > 1`` holds out the last
    ``fraction`` of each of that many equal time blocks instead (see
    ``split_segment_time_blocks``); the prefix and holdout come back flat.
    """
    if int(blocks) > 1:
        pieces, slices = split_segment_time_blocks(rows, int(blocks), fraction, gap_s=gap_s)
        return [row for part in pieces for row in part], [row for part in slices for row in part]
    times = [_row_time(row) for row in rows]
    known = [stamp for stamp in times if stamp is not None]
    if len(known) < 2:
        return list(rows), []
    t0 = min(known)
    span = max(known) - t0
    portion = min(1.0, max(0.0, float(fraction)))
    if span <= 0.0 or portion <= 0.0:
        return list(rows), []
    cut = t0 + (1.0 - portion) * span
    gap_at = cut - max(0.0, float(gap_s))
    prefix: list[Mapping[str, Any]] = []
    hold: list[Mapping[str, Any]] = []
    for row, stamp in zip(rows, times):
        if stamp is None or stamp < gap_at:
            prefix.append(row)
        elif stamp >= cut:
            hold.append(row)
    if not hold:
        return list(rows), []
    return prefix, hold


def valid_fraction(rows: list[Mapping[str, Any]]) -> float:
    """Share of rows whose 5 m and 10 m lane centers are marked valid."""
    from python.debug.lane_targets import lane_center_is_valid, lane_gt_from_record

    if not rows:
        return 0.0
    good = 0
    for row in rows:
        tech = row.get("tech") if isinstance(row.get("tech"), Mapping) else row
        _values, mask = lane_gt_from_record(tech if isinstance(tech, Mapping) else None)
        if lane_center_is_valid(mask):
            good += 1
    return good / float(len(rows))


def window_difficulty(rows: list[Mapping[str, Any]]) -> float:
    """How much this window bends, changes lane, or leaves a straight.

    Straights sit near 0. A weave, a ramp, or a large steer or heading
    error pushes the score up so ``--balance`` can repeat the window.
    """
    from python.debug.lane_targets import lane_gt_from_record

    if not rows:
        return 0.0
    total = 0.0
    prev: float | None = None
    for row in rows:
        tech = row.get("tech") if isinstance(row.get("tech"), Mapping) else row
        if not isinstance(tech, Mapping):
            tech = {}
        try:
            heading = abs(float(tech.get("heading_err_rad") or 0.0))
        except (TypeError, ValueError):
            heading = 0.0
        wheel = row.get("wheel") if isinstance(row.get("wheel"), Mapping) else {}
        try:
            steer = abs(float(wheel.get("steer") or 0.0))
        except (TypeError, ValueError):
            steer = 0.0
        values, mask = lane_gt_from_record(tech)
        center = abs(float(values[0])) if float(mask[0]) > 0.5 else 0.0
        delta = 0.0 if prev is None else abs(center - prev)
        prev = center
        excursion = row.get("excursion") if isinstance(row.get("excursion"), Mapping) else {}
        phase = str(excursion.get("phase") or "")
        bonus = 1.0 if phase in ("weave", "lane_change") else 0.0
        if str(row.get("spawn_type") or "") == "ramp":
            bonus += 0.75
        total += heading * 5.0 + steer + delta + bonus
    return total / float(len(rows))


def _recording_key(rows: list[Mapping[str, Any]], fallback: str = "recording") -> str:
    """Which recording a window comes from: its ``segment``, else its ``session``, else ``fallback``."""
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key in ("segment", "session"):
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
    return str(fallback)


def epoch_windows(
    plan: list[dict[str, Any]],
    rng: Any,
    *,
    min_valid: float = 0.3,
    balance: bool = False,
    stats: dict[str, Any] | None = None,
    recording: str = "recording",
) -> list[dict[str, Any]]:
    """Shuffle train windows. Holdout stays in time order, after them.

    Every training window starts from a fresh GRU state. Carrying a
    straight-road hidden state into the next curve was the loss spike.

    A window whose lane-center valid share is under ``min_valid`` is dropped.
    ``stats``, when given, is filled with what that filter removed (windows,
    rows, and recordings left with no train window at all) so the trainer can
    log it and refuse a recording that silently trains nothing; see
    ``report_valid_filter``. ``recording`` names windows whose rows carry
    neither ``segment`` nor ``session`` (the trainer passes the strips folder).
    """
    train: list[dict[str, Any]] = []
    hold: list[dict[str, Any]] = []
    floor = float(min_valid)
    kept_by_rec: dict[str, int] = {}
    dropped_by_rec: dict[str, int] = {}
    kept_rows: set[int] = set()
    dropped_rows: dict[int, Mapping[str, Any]] = {}
    windows_in = 0
    for item in plan:
        if item.get("kind") == "holdout":
            hold.append(item)
            continue
        rows = list(item.get("rows") or [])
        windows_in += 1
        rec = _recording_key(rows, recording)
        kept_by_rec.setdefault(rec, 0)
        if floor > 0.0 and valid_fraction(rows) < floor:
            dropped_by_rec[rec] = dropped_by_rec.get(rec, 0) + 1
            for row in rows:
                dropped_rows[id(row)] = row
            continue
        kept_by_rec[rec] += 1
        kept_rows.update(id(row) for row in rows)
        fresh = dict(item)
        fresh["incoming"] = "reset"
        fresh["rows"] = rows
        copies = 1
        if balance:
            score = window_difficulty(rows)
            if score >= 0.45:
                copies = 4
            elif score >= 0.15:
                copies = 2
        for _ in range(copies):
            train.append(dict(fresh))
    rng.shuffle(train)
    if stats is not None:
        removed_rows = [row for key, row in dropped_rows.items() if key not in kept_rows]
        stats.clear()
        stats.update(
            {
                "min_valid": floor,
                "windows_in": windows_in,
                "windows_dropped": sum(dropped_by_rec.values()),
                "rows_dropped": len(removed_rows),
                "rows_dropped_no_center": sum(1 for row in removed_rows if valid_fraction([row]) == 0.0),
                "recordings": len(kept_by_rec),
                "recordings_removed": sorted(rec for rec, kept in kept_by_rec.items() if kept == 0 and dropped_by_rec.get(rec)),
                "dropped_by_recording": dict(dropped_by_rec),
            }
        )
    return train + hold


def report_valid_filter(stats: Mapping[str, Any], *, epoch: int | None = None, quiet: bool = False) -> None:
    """Log what the ``--min-valid`` filter removed. Raise when a recording lost every window.

    A recording without lane_gt (no 5 m / 10 m lane centers) has a valid
    share of 0, so the default ``min_valid`` 0.3 removes all its windows
    before batching and it would train zero steps without a word. That
    raises ``LaneGtMissingError`` naming the recording.
    """
    from python.train.scene_net import LaneGtMissingError

    if not stats:
        return
    removed = list(stats.get("recordings_removed") or [])
    dropped = int(stats.get("windows_dropped") or 0)
    floor = float(stats.get("min_valid") or 0.0)
    if not quiet or removed:
        tag = "" if epoch is None else f" epoch={int(epoch)}"
        why = (
            f"lane-center valid share < {floor:g} "
            f"({int(stats.get('rows_dropped_no_center') or 0)} of the removed rows have no lane center at all)"
            if dropped
            else "none below the floor"
        )
        print(
            f"[GVD] min-valid filter{tag}: removed windows={dropped}/{int(stats.get('windows_in') or 0)} "
            f"rows={int(stats.get('rows_dropped') or 0)} "
            f"recordings={len(removed)}/{int(stats.get('recordings') or 0)}; why: {why}",
            flush=True,
        )
    if removed:
        shown = ", ".join(removed[:5]) + (" ..." if len(removed) > 5 else "")
        raise LaneGtMissingError(
            f"the --min-valid {floor:g} filter removed every training window of {len(removed)} "
            f"recording(s) ({shown}): their rows have no lane_gt lane centers, so they would train "
            "zero steps. Re-collect them with lane_gt, drop them from --strips, or pass --min-valid 0."
        )


def pass_plan(
    rows: list[Mapping[str, Any]],
    length: int,
    stride: int,
    *,
    memory_s: float = MEMORY_S,
    fraction: float = HOLDOUT_FRACTION,
    blocks: int = 1,
    cells: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One pass: overlapping train chunks, then the holdout.

    Rows tagged with ``segment`` hold out whole short segments when there
    are two or more. One segment, and any segment longer than about 15
    minutes, holds out the last fraction of its own time, with a few
    seconds dropped at the cut. A recording without a segment tag still
    holds out the last fraction of each span.
    A rare chunk is listed twice so the trainer steps twice from the same
    incoming state. Holdout rows are never listed as train rows. The second
    copy of a rare chunk uses ``incoming == "chunk"`` so it does not advance
    the carry on its own.
    ``blocks > 1`` (``--holdout-blocks``) cuts each time-split segment into
    that many equal blocks and holds out the last ``fraction`` of each, with
    ``HOLDOUT_GAP_S`` on both sides of every slice. Untagged rows with
    ``blocks > 1`` are treated as one segment per recording span. The
    default of 1 is the single tail cut above.
    ``cells`` (a spec from ``choose_holdout_cells``) replaces the time split
    with a spatial one: every row inside a listed map cell is holdout, and
    train rows within the buffer of one are dropped. ``blocks`` is ignored.
    """
    if cells is not None:
        return _cell_plan(rows, length, stride, memory_s=memory_s, spec=cells)
    if _segment_groups(rows) is not None:
        return _pass_plan_segments(rows, length, stride, memory_s=memory_s, fraction=fraction, blocks=blocks)
    if int(blocks) > 1:
        plan_k: list[dict[str, Any]] = []
        for start, end in recording_spans(rows, memory_s=memory_s):
            plan_k.extend(_block_plan(list(rows[start:end]), length, stride, memory_s=memory_s, fraction=fraction, blocks=int(blocks)))
        return plan_k
    plan: list[dict[str, Any]] = []
    for start, end in recording_spans(rows, memory_s=memory_s):
        segment = list(rows[start:end])
        prefix, holdout = split_prefix_holdout(segment, fraction, memory_s=memory_s)
        chunks = iter_bptt(prefix, length, stride, memory_s=memory_s)
        for chunk in chunks:
            piece = [prefix[index] for index in range(int(chunk["start"]), int(chunk["end"]))]
            reps = oversample_repeats(piece)
            incoming = "reset" if chunk["reset"] else "carry"
            for rep in range(reps):
                plan.append(
                    {
                        "kind": "train",
                        "rows": piece,
                        "incoming": incoming if rep == 0 else "chunk",
                        "stride": int(chunk["stride"]),
                        "grad_frames": int(chunk["end"]) - int(chunk["start"]),
                    }
                )
        if holdout:
            plan.append(
                {
                    "kind": "holdout",
                    "rows": holdout,
                    "incoming": "end" if chunks else "reset",
                    "stride": 0,
                    "grad_frames": 0,
                }
            )
    return plan


def _train_only_plan(
    rows: list[Mapping[str, Any]],
    length: int,
    stride: int,
    *,
    memory_s: float,
) -> list[dict[str, Any]]:
    """Train chunks for these rows. No time-based holdout inside the block."""
    return pass_plan(rows, length, stride, memory_s=memory_s, fraction=0.0)


def _block_plan(
    rows: list[Mapping[str, Any]],
    length: int,
    stride: int,
    *,
    memory_s: float,
    fraction: float,
    blocks: int,
) -> list[dict[str, Any]]:
    """Train chunks per piece, then one holdout item per slice, in time order."""
    pieces, slices = split_segment_time_blocks(rows, blocks, fraction)
    plan: list[dict[str, Any]] = []
    for part in pieces:
        plain = [dict(row) for row in part]
        for row in plain:
            row.pop("segment", None)
        plan.extend(_train_only_plan(plain, length, stride, memory_s=memory_s))
    for part in slices:
        plan.append(
            {
                "kind": "holdout",
                "rows": list(part),
                "incoming": "reset",
                "stride": 0,
                "grad_frames": 0,
            }
        )
    return plan


def _pass_plan_segments(
    rows: list[Mapping[str, Any]],
    length: int,
    stride: int,
    *,
    memory_s: float,
    fraction: float = HOLDOUT_FRACTION,
    blocks: int = 1,
) -> list[dict[str, Any]]:
    groups = _segment_groups(rows) or []
    held = holdout_segment_names(rows)
    few = len(groups) < 2
    plan: list[dict[str, Any]] = []
    for seg, block in groups:
        if (few or segment_span_s(block) > LONG_SEGMENT_S) and int(blocks) > 1:
            plan.extend(_block_plan(block, length, stride, memory_s=memory_s, fraction=fraction, blocks=int(blocks)))
            continue
        if few or segment_span_s(block) > LONG_SEGMENT_S:
            prefix, hold = split_segment_time_holdout(block, fraction)
            plain = [dict(row) for row in prefix]
            for row in plain:
                row.pop("segment", None)
            plan.extend(_train_only_plan(plain, length, stride, memory_s=memory_s))
            if hold:
                plan.append(
                    {
                        "kind": "holdout",
                        "rows": list(hold),
                        "incoming": "reset",
                        "stride": 0,
                        "grad_frames": 0,
                    }
                )
            continue
        plain = [dict(row) for row in block]
        for row in plain:
            row.pop("segment", None)
        if seg in held:
            plan.append(
                {
                    "kind": "holdout",
                    "rows": list(block),
                    "incoming": "reset",
                    "stride": 0,
                    "grad_frames": 0,
                }
            )
            continue
        plan.extend(_train_only_plan(plain, length, stride, memory_s=memory_s))
    return plan


def remaining_until_holdout_can_stop(
    remaining_this_pass: int,
    steps_per_pass: int,
    *,
    epoch: int,
    has_holdout: bool,
    rates_after: int,
) -> int:
    """Steps from here until the holdout is allowed to stop the run.

    The first pass cannot stop. Each later learning rate gets the same shape:
    one pass, plus a second when a holdout exists. ``--epochs`` does not
    multiply this. If val loss keeps falling, the run continues past it.
    """
    remaining = max(0, int(remaining_this_pass))
    per = max(0, int(steps_per_pass))
    if has_holdout and int(epoch) <= 0:
        remaining += per
    later = per * (2 if has_holdout else 1)
    remaining += max(0, int(rates_after)) * later
    return remaining


def holdout_stopped(
    losses: list[float],
    *,
    tol: float = 0.0,
    window: int = 0,
) -> bool:
    """True once the latest holdout score is not strictly better than the best so far.

    Lower is better. The first measurement cannot stop the run. There is no
    epoch cap here. ``tol`` and ``window`` stop a plateau: the last ``window``
    scores stay inside ``tol`` (metres, when the score is the lane-center rms).
    """
    if len(losses) < 2:
        return False
    if float(tol) > 0.0 and int(window) >= 2 and len(losses) >= int(window):
        recent = [float(value) for value in losses[-int(window) :]]
        return max(recent) - min(recent) <= float(tol)
    best = min(float(value) for value in losses[:-1])
    return float(losses[-1]) >= best


def pick_learning_rate(scores: Mapping[float, float]) -> float:
    """Lower holdout loss wins. A tie keeps the earlier candidate."""
    if not scores:
        raise ValueError("learning-rate search has no scores")

    def sort_key(rate: float) -> tuple[float, int]:
        try:
            order = CANDIDATE_LRS.index(rate)
        except ValueError:
            order = len(CANDIDATE_LRS)
        return (float(scores[rate]), order)

    return min(scores, key=sort_key)


def choose_chunk_length(frame_bytes: int, budget_bytes: int) -> int:
    """Largest chunk in 8..16 frames that fits in ``budget_bytes``, else 8."""
    frame = max(1, int(frame_bytes))
    budget = int(budget_bytes)
    best = CHUNK_MIN
    if budget <= 0:
        return best
    for length in range(CHUNK_MIN, CHUNK_MAX + 1):
        if length * frame <= budget:
            best = length
    return best


def chunk_vram_budget(free_vram_bytes: int) -> int:
    """Free VRAM, capped at 11 GB. Zero when the card reports nothing free."""
    free = int(free_vram_bytes)
    if free <= 0:
        return 0
    return min(free, VRAM_BUDGET_BYTES)


def resume_learning_rate(requested: float, paused: float | None = None) -> float:
    """The command-line rate. A pause file cannot continue a lower-rate pass.

    ``paused`` is the rate stored in the pause file. It is not returned.
    Holdout stop does not schedule the next candidate. Passing ``--lr`` is
    the only way to leave the default ``1e-3``.
    """
    del paused
    try:
        rate = float(requested)
    except (TypeError, ValueError):
        return float(CANDIDATE_LRS[0])
    if not math.isfinite(rate) or rate <= 0.0:
        return float(CANDIDATE_LRS[0])
    return rate


def ordered_learning_rates(requested: float) -> tuple[float, ...]:
    """The two candidates, with ``requested`` first when it is one of them.

    The trainer does not call this. ``resume_learning_rate`` is the schedule.
    """
    try:
        rate = float(requested)
    except (TypeError, ValueError):
        rate = CANDIDATE_LRS[0]
    ordered: list[float] = []
    if any(math.isclose(rate, candidate, rel_tol=0.0, abs_tol=1e-12) for candidate in CANDIDATE_LRS):
        ordered.append(rate)
    for candidate in CANDIDATE_LRS:
        if not any(math.isclose(candidate, have, rel_tol=0.0, abs_tol=1e-12) for have in ordered):
            ordered.append(candidate)
    return tuple(ordered)


def _row_xy(row: Mapping[str, Any]) -> tuple[float, float] | None:
    ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else {}
    pos = ego.get("pos") if isinstance(ego, Mapping) else None
    if not pos or len(pos) < 2:
        return None
    try:
        x, y = float(pos[0]), float(pos[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _row_speed(row: Mapping[str, Any]) -> float | None:
    ego = row.get("ego") if isinstance(row.get("ego"), Mapping) else {}
    try:
        value = float(ego.get("speed_mps"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def cell_of(xy: tuple[float, float], cell_m: float = HOLDOUT_CELL_M) -> tuple[int, int]:
    return int(math.floor(xy[0] / cell_m)), int(math.floor(xy[1] / cell_m))


def _map_name(rows: list[Mapping[str, Any]]) -> str:
    for row in rows:
        seg = row.get("segment")
        if seg:
            return str(seg).split("#", 1)[0]
    return ""


def choose_holdout_cells(
    rows: list[Mapping[str, Any]],
    *,
    seed: int = 0,
    fraction: float = HOLDOUT_FRACTION,
    cell_m: float = HOLDOUT_CELL_M,
    buffer_m: float = HOLDOUT_BUFFER_M,
    fast_mps: float = HOLDOUT_FAST_MPS,
) -> dict[str, Any]:
    """A fixed, seeded set of map cells holding about ``fraction`` of rows.

    Cells are classed fast or slow by median speed and drawn per class in a
    seeded order, so both highway-like and city cells are held out in
    proportion to their share of rows. The same rows and seed always give
    the same list. Save it with ``save_holdout_cells`` and reload it so a
    later run (or a larger recording) scores on the same cells.
    """
    counts: dict[tuple[int, int], int] = {}
    speeds: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        xy = _row_xy(row)
        if xy is None:
            continue
        key = cell_of(xy, cell_m)
        counts[key] = counts.get(key, 0) + 1
        spd = _row_speed(row)
        if spd is not None:
            speeds.setdefault(key, []).append(spd)
    total = sum(counts.values())

    def _median(vals: list[float]) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals)
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])

    klass = {key: ("fast" if _median(speeds.get(key, [])) >= float(fast_mps) else "slow") for key in counts}
    rng = random.Random(int(seed))
    chosen: list[tuple[int, int]] = []
    portion = min(1.0, max(0.0, float(fraction)))
    for name in ("fast", "slow"):
        keys = sorted(key for key in counts if klass[key] == name)
        class_rows = sum(counts[key] for key in keys)
        target = portion * class_rows
        rng.shuffle(keys)
        held = 0
        for key in keys:
            if held >= target:
                break
            # Skip a cell that would overshoot the class target by more than half of it.
            if held > 0 and held + counts[key] > target + 0.5 * counts[key] and held + counts[key] > 1.25 * target:
                continue
            chosen.append(key)
            held += counts[key]
    chosen.sort()
    return {
        "version": HOLDOUT_CELLS_VERSION,
        "map": _map_name(rows),
        "cell_m": float(cell_m),
        "buffer_m": float(buffer_m),
        "seed": int(seed),
        "fraction": portion,
        "fast_mps": float(fast_mps),
        "rows_at_choice": int(total),
        "cells": [
            {"cx": int(key[0]), "cy": int(key[1]), "class": klass[key], "rows": int(counts[key])}
            for key in chosen
        ],
    }


def save_holdout_cells(spec: Mapping[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(spec), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out


def row_maps(rows: list[Mapping[str, Any]]) -> list[str]:
    """Map names the rows cover (the ``segment`` prefix), in first-seen order."""
    seen: list[str] = []
    for row in rows:
        seg = row.get("segment")
        if not seg:
            continue
        name = str(seg).split("#", 1)[0]
        if name and name not in seen:
            seen.append(name)
    return seen


def check_holdout_cells(spec: Mapping[str, Any], rows: list[Mapping[str, Any]], *, source: Any = "cells") -> None:
    """Raise when a cell list cannot be applied to these rows.

    Cell keys are plain map coordinates, so cells chosen on one map would
    silently land on unrelated road on another. A list for map A refuses
    rows from map B, and rows that span several maps need every cell to
    carry its own ``map``.
    """
    maps = row_maps(rows)
    if not maps:
        return
    cells = list(spec.get("cells") or [])
    per_map = bool(cells) and all(isinstance(c, Mapping) and c.get("map") for c in cells)
    if len(maps) > 1 and not per_map:
        raise ValueError(
            f"{source}: rows span {len(maps)} maps ({', '.join(maps)}) but the holdout cells "
            "have no per-map 'map' field; choose cells per map"
        )
    saved = str(spec.get("map") or "")
    if not per_map and saved and saved != maps[0]:
        raise ValueError(
            f"{source}: holdout cells were chosen on map '{saved}' but the rows are on '{maps[0]}'; "
            "pass a cell list for this map with --holdout-cells-file"
        )
    if per_map:
        cell_maps = sorted({str(c.get("map")) for c in cells})
        if not set(maps) & set(cell_maps):
            raise ValueError(f"{source}: holdout cells are for {cell_maps} but the rows are on {maps}")


def load_holdout_cells(path: Path | str, rows: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Read a saved cell list. With ``rows``, refuse a list for another map."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("cells"), list):
        raise ValueError(f"{path}: not a holdout cell list")
    if rows is not None:
        check_holdout_cells(spec, rows, source=path)
    return spec


def holdout_mix(plan: list[Mapping[str, Any]], fast_mps: float = HOLDOUT_FAST_MPS) -> dict[str, int]:
    """Holdout row count split by row speed (fast at or above ``fast_mps``)."""
    out = {"rows": 0, "fast": 0, "slow": 0, "no_speed": 0, "slices": 0}
    for item in plan:
        if item.get("kind") != "holdout":
            continue
        out["slices"] += 1
        for row in item.get("rows") or []:
            out["rows"] += 1
            spd = _row_speed(row)
            if spd is None:
                out["no_speed"] += 1
            elif spd >= float(fast_mps):
                out["fast"] += 1
            else:
                out["slow"] += 1
    return out


def _pairs_score(preds: list[float], truth: list[float]) -> tuple[float | None, float | None]:
    p = np.asarray(preds, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if p.size == 0:
        return None, None
    rms = float(np.sqrt(np.mean((p - t) ** 2)))
    if p.size < 3 or float(p.std()) < 1e-12 or float(t.std()) < 1e-12:
        return None, rms
    return float(np.corrcoef(p, t)[0, 1]), rms


def cell_scores(
    pairs: list[tuple[float, float]],
    rows: list[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Per held-out cell: n, r, rms of the 8 m center. ``rows[k]`` made ``pairs[k]``."""
    cell_m = float(spec.get("cell_m", HOLDOUT_CELL_M))
    groups: dict[str, tuple[list[float], list[float]]] = {}
    for (pred, truth), row in zip(pairs, rows):
        xy = _row_xy(row)
        key = "none" if xy is None else "{}_{}".format(*cell_of(xy, cell_m))
        bucket = groups.setdefault(key, ([], []))
        bucket[0].append(float(pred))
        bucket[1].append(float(truth))
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(groups):
        p, t = groups[key]
        r, rms = _pairs_score(p, t)
        out[key] = {"n": len(p), "r": r, "rms": rms, "pred": p, "truth": t}
    return out


def bootstrap_cells(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.90,
) -> dict[str, Any]:
    """Resample whole cells with replacement and pool their pairs.

    Rows inside one cell are strongly correlated, so the cell is the unit.
    Returns the ``level`` percentile interval for r and rms. Seeded.
    """
    keys = sorted(cells)
    if not keys:
        return {"n_cells": 0, "r_ci": None, "rms_ci": None, "n_boot": int(n_boot), "seed": int(seed)}
    rng = np.random.RandomState(int(seed))
    rs: list[float] = []
    rmss: list[float] = []
    for _ in range(int(n_boot)):
        pick = rng.randint(0, len(keys), size=len(keys))
        p: list[float] = []
        t: list[float] = []
        for j in pick:
            p.extend(cells[keys[j]]["pred"])
            t.extend(cells[keys[j]]["truth"])
        r, rms = _pairs_score(p, t)
        if r is not None:
            rs.append(r)
        if rms is not None:
            rmss.append(rms)
    lo = (1.0 - float(level)) / 2.0 * 100.0
    hi = 100.0 - lo

    def _ci(vals: list[float]) -> list[float] | None:
        if not vals:
            return None
        return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]

    return {"n_cells": len(keys), "r_ci": _ci(rs), "rms_ci": _ci(rmss), "n_boot": int(n_boot), "seed": int(seed)}


def _cell_distance_m(xy: tuple[float, float], key: tuple[int, int], cell_m: float) -> float:
    x0, y0 = key[0] * cell_m, key[1] * cell_m
    dx = max(x0 - xy[0], 0.0, xy[0] - (x0 + cell_m))
    dy = max(y0 - xy[1], 0.0, xy[1] - (y0 + cell_m))
    return math.hypot(dx, dy)


def cell_holdout_labels(
    rows: list[Mapping[str, Any]],
    spec: Mapping[str, Any],
    *,
    gap_s: float = HOLDOUT_GAP_S,
) -> list[str]:
    """``train``, ``hold``, or ``gap`` per row for a cell spec.

    ``hold``: the row sits in a listed cell. ``gap``: no position, within
    ``buffer_m`` of a listed cell, or within ``gap_s`` seconds of a holdout
    row in the same recording order (so the GRU never trains on the frames
    right before or after a scored stretch). Everything else is ``train``.
    """
    cell_m = float(spec.get("cell_m", HOLDOUT_CELL_M))
    buffer_m = float(spec.get("buffer_m", HOLDOUT_BUFFER_M))
    held = {(int(c["cx"]), int(c["cy"])) for c in spec.get("cells", [])}
    reach = int(math.ceil(buffer_m / cell_m)) if buffer_m > 0 else 0
    labels: list[str] = []
    for row in rows:
        xy = _row_xy(row)
        if xy is None:
            labels.append("gap")
            continue
        key = cell_of(xy, cell_m)
        if key in held:
            labels.append("hold")
            continue
        near = False
        if buffer_m > 0:
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    other = (key[0] + dx, key[1] + dy)
                    if other in held and _cell_distance_m(xy, other, cell_m) < buffer_m:
                        near = True
                        break
                if near:
                    break
        labels.append("gap" if near else "train")
    gap = max(0.0, float(gap_s))
    if gap > 0:
        times = [_row_time(row) for row in rows]
        hold_times = sorted(t for t, lab in zip(times, labels) if lab == "hold" and t is not None)
        if hold_times:
            import bisect

            for index, (stamp, lab) in enumerate(zip(times, labels)):
                if lab != "train" or stamp is None:
                    continue
                j = bisect.bisect_left(hold_times, stamp)
                for k in (j - 1, j):
                    if 0 <= k < len(hold_times) and abs(hold_times[k] - stamp) < gap:
                        labels[index] = "gap"
                        break
    return labels


def _cell_plan(
    rows: list[Mapping[str, Any]],
    length: int,
    stride: int,
    *,
    memory_s: float,
    spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Train chunks per contiguous train run, then each holdout run, in time order."""
    labels = cell_holdout_labels(rows, spec)
    pieces: list[list[Mapping[str, Any]]] = []
    slices: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    current_kind: str | None = None
    for row, lab in zip(rows, labels):
        if lab != current_kind:
            if current_kind == "train" and current:
                pieces.append(current)
            elif current_kind == "hold" and current:
                slices.append(current)
            current = []
            current_kind = lab
        if lab != "gap":
            current.append(row)
    if current_kind == "train" and current:
        pieces.append(current)
    elif current_kind == "hold" and current:
        slices.append(current)
    plan: list[dict[str, Any]] = []
    for part in pieces:
        plain = [dict(row) for row in part]
        for row in plain:
            row.pop("segment", None)
        plan.extend(_train_only_plan(plain, length, stride, memory_s=memory_s))
    for part in slices:
        plan.append({"kind": "holdout", "rows": list(part), "incoming": "reset", "stride": 0, "grad_frames": 0})
    return plan
