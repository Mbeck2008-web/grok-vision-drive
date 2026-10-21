"""Hold-last side cameras: one rule for the trainer and the live supervisor.

The slow cameras (pillars, repeaters, rear) do not render every tick. Instead
of a black sector on the ticks they skip, a camera's most recent EARLIER frame
is shown, and the net is told how old it is. ``held_sector`` is the one
function both sides call; ``CamHold`` is the per-camera "last frame" memory
both sides feed in time order.

Rules (``held_sector``):

* A frame newer than ``now`` (by more than half a tick) is look-ahead and is
  never used.
* ``age = now - frame_t`` in seconds, quantised to the supervisor tick
  (``HOLD_TICK_HZ`` = 15, run_vision ``--hz`` default): ``round(age * 15) / 15``.
* An age over ``HOLD_CAP_S`` (1.5 s), or no frame at all: the sector is black
  and the age is ``NO_FRAME_AGE`` (-1).
* ``main`` is never held. Its frame is used only when it is this tick's
  (quantised age 0); otherwise it is black with age -1.

``CamHold.reset`` is the boundary: a respawn, a new segment or session, or a
change between train and holdout runs. A frame from before a reset is never
picked after it.

The ages go to the net in ``CAM_AGE_ORDER`` (the strip order), 8 floats.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

from python.data.strip_writer import STRIP_ORDER

HOLD_TICK_HZ = 15
HOLD_TICK_S = 1.0 / HOLD_TICK_HZ
HOLD_CAP_S = 1.5
NO_FRAME_AGE = -1.0
CAM_AGE_ORDER: tuple[str, ...] = tuple(STRIP_ORDER)
CAM_AGE_N = len(CAM_AGE_ORDER)
NEVER_HELD = frozenset({"main"})
# ONNX metadata keys.
CAM_AGE_META = "gvd_cam_age"
CAM_AGE_ORDER_META = "gvd_cam_age_order"
CAM_AGE_CAP_META = "gvd_cam_age_cap_s"
CAM_AGE_HZ_META = "gvd_cam_age_tick_hz"
# Key a loaded training row carries its 8 ages under.
ROW_AGES_KEY = "_cam_ages"
# Key of the per-row hold plan (cam -> [source row index or None, age]).
ROW_HOLD_KEY = "cam_hold"


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def quantise_age(age_s: float) -> float:
    """Seconds, rounded to the nearest 1/15 s supervisor tick (half up)."""
    ticks = math.floor(float(age_s) * HOLD_TICK_HZ + 0.5)
    return ticks / float(HOLD_TICK_HZ)


# A black sector in age statistics: the first quantised age over the cap
# (1.5 s is 22.5 ticks, rounds to 23 = 1.533 s). Live and train both count black
# here, so a median / p95 never drops the black rows.
BLACK_STAT_AGE_S = math.floor(HOLD_CAP_S * HOLD_TICK_HZ + 0.5) / float(HOLD_TICK_HZ)


def stat_age(age: Any) -> float:
    """An age for statistics: black (< 0, or None) counts as ``BLACK_STAT_AGE_S``."""
    got = _finite(age)
    return BLACK_STAT_AGE_S if got is None or got < 0.0 else got


def held_sector(cam: str, frame_t: Any, frame: Any, now_t: Any) -> tuple[Any, float]:
    """``(sector frame or None, age)`` for one camera at ``now_t``.

    ``frame`` is whatever the caller holds for that camera's last render: the
    pixels live, a source row index in the trainer. It is passed through
    untouched when it is used. ``None`` means a black sector.
    """
    t = _finite(frame_t)
    now = _finite(now_t)
    if frame is None or t is None or now is None:
        return None, NO_FRAME_AGE
    raw = now - t
    if raw < -0.5 * HOLD_TICK_S:
        # A later frame. Never look ahead.
        return None, NO_FRAME_AGE
    age = quantise_age(max(0.0, raw))
    if str(cam) in NEVER_HELD:
        return (frame, 0.0) if age == 0.0 else (None, NO_FRAME_AGE)
    if age > HOLD_CAP_S:
        return None, NO_FRAME_AGE
    return frame, age


class CamHold:
    """Last frame per camera, fed in time order. ``pick`` applies ``held_sector``."""

    def __init__(self, cams: Iterable[str] = CAM_AGE_ORDER) -> None:
        self.cams = tuple(cams)
        self._last: dict[str, tuple[float, Any]] = {}

    def reset(self) -> None:
        self._last.clear()

    def note(self, cam: str, t: Any, frame: Any) -> None:
        """A new render of ``cam`` at ``t``. An older stamp never replaces a newer one."""
        stamp = _finite(t)
        if frame is None or stamp is None or str(cam) not in self.cams:
            return
        prev = self._last.get(str(cam))
        if prev is not None and stamp < prev[0]:
            return
        self._last[str(cam)] = (stamp, frame)

    def forget(self, cam: str) -> None:
        """Drop ``cam``'s last frame (live: a poll timeout). It is black, age -1, until its next render."""
        self._last.pop(str(cam), None)

    def last(self, cam: str) -> tuple[float, Any] | None:
        return self._last.get(str(cam))

    def pick(self, now_t: Any) -> dict[str, tuple[Any, float]]:
        out: dict[str, tuple[Any, float]] = {}
        for cam in self.cams:
            last = self._last.get(cam)
            if last is None:
                out[cam] = (None, NO_FRAME_AGE)
            else:
                out[cam] = held_sector(cam, last[0], last[1], now_t)
        return out


def ages_vector(picks: Mapping[str, Any]) -> np.ndarray:
    """8 float32 ages in ``CAM_AGE_ORDER`` from ``{cam: (frame, age)}`` or ``{cam: age}``."""
    out = np.full((CAM_AGE_N,), NO_FRAME_AGE, dtype=np.float32)
    for k, cam in enumerate(CAM_AGE_ORDER):
        value = picks.get(cam)
        if isinstance(value, (tuple, list)):
            value = value[1] if len(value) > 1 else None
        got = _finite(value)
        if got is not None:
            out[k] = got
    return out


def no_ages() -> np.ndarray:
    return np.full((CAM_AGE_N,), NO_FRAME_AGE, dtype=np.float32)


# ---- trainer side: the hold plan over recorded rows -------------------------


def _row_cam_t(row: Mapping[str, Any], cam: str) -> float | None:
    cams = row.get("cams") if isinstance(row.get("cams"), Mapping) else {}
    meta = cams.get(cam) if isinstance(cams.get(cam), Mapping) else None
    if meta is None:
        return None
    t = _finite(meta.get("t"))
    return t if t is not None else _finite(row.get("t"))


def row_now(row: Mapping[str, Any]) -> float | None:
    """This tick's time: the main frame's stamp, else the row time."""
    t = _row_cam_t(row, "main")
    return t if t is not None else _finite(row.get("t"))


def respawn_starts(row: Mapping[str, Any], prev: Mapping[str, Any] | None) -> bool:
    """A boundary before ``row``: new segment/session, null dt, recover, or a new settle run.

    The settle rule is ``event_mask._before_respawn``'s: every spawn, teleport
    and recover starts a ``settle`` run.
    """
    if prev is None:
        return True
    if row.get("segment") != prev.get("segment") or row.get("session") != prev.get("session"):
        return True
    if "dt_s" in row and row.get("dt_s") is None:
        return True
    if bool(row.get("recover")):
        return True
    if bool(row.get("settle")) and not bool(prev.get("settle")):
        return True
    return False


def plan_labels(plan: Iterable[Mapping[str, Any]]) -> dict[int, str]:
    """Row index -> ``train`` or ``hold:<n>`` (one label per holdout item) from a pass plan."""
    labels: dict[int, str] = {}
    n_hold = 0
    for item in plan:
        if item.get("kind") == "holdout":
            label = f"hold:{n_hold}"
            n_hold += 1
        else:
            label = "train"
        for row in item.get("rows") or []:
            try:
                labels[int(row["i"])] = label
            except (KeyError, TypeError, ValueError):
                continue
    return labels


def hold_plan(
    state_rows: Iterable[Mapping[str, Any]],
    labels: Mapping[int, str] | None = None,
) -> dict[int, dict[str, list[Any]]]:
    """Per row index: ``{cam: [source row index or None, age]}``.

    ``state_rows`` is the whole recording in file order (respawn markers live
    on rows a mask may drop). ``labels`` (``plan_labels``) restricts sources and
    targets to rows of the plan, and a change of label is a boundary, so a
    holdout row never shows a train frame and the reverse. Rows without a label
    are neither sources nor targets, but their respawn markers still count.
    ``labels=None`` treats every row as one label.
    """
    hold = CamHold()
    out: dict[int, dict[str, list[Any]]] = {}
    prev: Mapping[str, Any] | None = None
    run_label: str | None = None
    for row in state_rows:
        if not isinstance(row, Mapping):
            continue
        try:
            index = int(row.get("i"))
        except (TypeError, ValueError):
            continue
        if respawn_starts(row, prev):
            hold.reset()
            run_label = None
        prev = row
        label = "all" if labels is None else labels.get(index)
        if label is None:
            continue
        if run_label is not None and label != run_label:
            hold.reset()
        run_label = label
        for cam in CAM_AGE_ORDER:
            t = _row_cam_t(row, cam)
            if t is not None:
                hold.note(cam, t, index)
        picks = hold.pick(row_now(row))
        out[index] = {cam: [picks[cam][0], float(picks[cam][1])] for cam in CAM_AGE_ORDER}
    return out


def annotate_hold(
    state_rows: list[Mapping[str, Any]],
    rows: Iterable[dict[str, Any]],
    plan: Iterable[Mapping[str, Any]],
) -> int:
    """Write ``ROW_HOLD_KEY`` on every row of ``rows``; returns how many got one."""
    table = hold_plan(state_rows, plan_labels(plan))
    n = 0
    for row in rows:
        try:
            got = table.get(int(row["i"]))
        except (KeyError, TypeError, ValueError):
            got = None
        if got is None:
            got = {cam: [None, NO_FRAME_AGE] for cam in CAM_AGE_ORDER}
        else:
            n += 1
        row[ROW_HOLD_KEY] = got
    return n


def hold_sources(row: Mapping[str, Any]) -> dict[str, tuple[int | None, float]]:
    """``{cam: (source index or None, age)}`` from a row's hold plan."""
    plan = row.get(ROW_HOLD_KEY)
    if not isinstance(plan, Mapping):
        raise ValueError(f"row i={row.get('i')} has no {ROW_HOLD_KEY} (annotate_hold first)")
    out: dict[str, tuple[int | None, float]] = {}
    for cam in CAM_AGE_ORDER:
        entry = plan.get(cam) or (None, NO_FRAME_AGE)
        src = entry[0]
        age = _finite(entry[1]) if len(entry) > 1 else None
        if src is None or age is None or age < 0.0:
            out[cam] = (None, NO_FRAME_AGE)
        else:
            out[cam] = (int(src), float(age))
    return out
