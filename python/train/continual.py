"""Continual scene-net schedule.

The GRU runs through a recording. A gap or a new recording clears it.
Backprop covers one short chunk, and the next chunk starts before that
chunk ends (Williams and Peng 1990). The chunk is not an independent clip.

The 360 strip is not linear in azimuth, so this module does not roll it.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from python.train.scene_net import MEMORY_S, step_dt

# AdamW candidates. The run keeps the one with the better holdout loss.
CANDIDATE_LRS = (1e-3, 1e-4)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
HOLDOUT_FRACTION = 0.2
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


def pass_plan(
    rows: list[Mapping[str, Any]],
    length: int,
    stride: int,
    *,
    memory_s: float = MEMORY_S,
    fraction: float = HOLDOUT_FRACTION,
) -> list[dict[str, Any]]:
    """One pass: overlapping train chunks on the prefix, then the holdout.

    A rare chunk is listed twice so the trainer steps twice from the same
    incoming state. Holdout rows are never listed as train rows. The second
    copy of a rare chunk uses ``incoming == "chunk"`` so it does not advance
    the carry on its own.
    """
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


def holdout_stopped(losses: list[float]) -> bool:
    """True once the latest holdout loss is not strictly better than the best so far.

    The first measurement cannot stop the run. There is no epoch cap here.
    """
    if len(losses) < 2:
        return False
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


def ordered_learning_rates(requested: float) -> tuple[float, ...]:
    """The two candidates, with ``requested`` first when it is one of them."""
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
