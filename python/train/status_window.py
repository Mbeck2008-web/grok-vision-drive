"""Supervisor-style panel for a scene-training run.

Colors, type, and row spacing come from the nerd panel in ``python/viz/nerd.py``.
The window shows recorded hours, minutes and frames from the strip folder,
time left, loss, learning rate, steps per second, the memory in use, and the
parameter count of the net it was handed. Start begins a run. Pause writes the
pause file at the path the trainer passed in and the panel says paused.

Six graphs share one painter: time left, steps/s, lr, memory (VRAM, system
RAM used and the trainer's own RSS, sampled in this process), driving (hours
and frames) and loss (train, plus val rows since this launch, tailed read-only
from the run's ``metrics.jsonl``). There is no server.

``python -m python.train.status_window --replay <status dir> --out <png>``
renders the panel from a finished run's files without opening a window. The
PNG is the only file this module writes: create-only, never over an existing
file, never under the data root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

# Same ground, type, and rhythm as the supervisor nerd panel: imported, not copied.
from python.viz.nerd import BG, DIM, FG, FONT, FS_BODY, FS_DIM, FS_TITLE, ICE, PAD_X, ROW_H, TH
from python.viz.nerd import BRK_COLOR as WARN
from python.viz.nerd import BRK_PRED_COLOR as VIOLET
from python.viz.nerd import NERD_WIDTH as PANEL_W
from python.viz.nerd import THR_PRED_COLOR as AMBER

RAM_LABEL = "system RAM used"
RSS_LABEL = "trainer RSS"
# Shown for a series the live trainer cannot measure (RSS on Windows without psutil).
NA = "n/a"
VAL_LABEL = "val since this launch"
GRAPH_FILL = (28, 24, 22)
GRAPH_EDGE = (70, 64, 58)
PANEL_H = 520
GRAPH_H = 168
SMALL_H = 84
GAP = 8
HISTORY_N = 2000
LINE_W = 2
WIN_NAME = "GVD TRAIN"


@dataclass
class TrainStatus:
    eta_s: float | None
    loss: float | None
    lr: float
    steps_per_sec: float
    memory_bytes: int
    memory_kind: str
    device: str
    batch: int
    recommended: bool
    step: int
    steps: int
    recorded_s: float = 0.0
    recorded_frames: int = 0


def steps_per_second(done_steps: int, elapsed_s: float) -> float:
    if done_steps <= 0 or elapsed_s <= 0:
        return 0.0
    return float(done_steps) / float(elapsed_s)


def predict_eta_s(done_steps: int, elapsed_s: float, remaining_steps: int) -> float | None:
    rate = steps_per_second(done_steps, elapsed_s)
    if rate <= 0:
        return None
    return float(remaining_steps) / rate


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def sum_recorded_seconds(rows: Iterable[Mapping[str, Any]]) -> float:
    """Seconds of recorded driving.

    Each finite ``dt_s`` is added. A null ``dt_s`` is a session boundary and
    adds nothing, so a parked gap between sessions is not driving time. A row
    with no ``dt_s`` uses the timestamp gap from the previous row in the same
    session. Nothing here assumes a fixed frame rate.
    """
    total = 0.0
    prev_t: float | None = None
    prev_session: Any = _NO_SESSION
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        session = row.get("session", _NO_SESSION)
        total += _row_seconds(row, prev_t, prev_session, session)
        stamp = _finite_number(row.get("t"))
        if stamp is not None:
            prev_t = stamp
        prev_session = session
    return total


def _row_seconds(row: Mapping[str, Any], prev_t: float | None, prev_session: Any, session: Any) -> float:
    if "dt_s" in row:
        raw = row.get("dt_s")
        if raw is None:
            return 0.0
        value = _finite_number(raw)
        if value is None or value < 0:
            return 0.0
        return value
    if prev_t is None:
        return 0.0
    if session != prev_session:
        return 0.0
    stamp = _finite_number(row.get("t"))
    if stamp is None:
        return 0.0
    gap = stamp - prev_t
    if gap < 0:
        return 0.0
    return gap


def read_recorded_seconds(root: Path | str) -> float:
    """Sum ``dt_s`` in ``state.jsonl``. Reads the folder and does not write it."""
    return read_recorded(root)[0]


def read_recorded(root: Path | str) -> tuple[float, int]:
    """``(seconds, frames)`` from ``state.jsonl``: one frame per row. Read-only."""
    path = Path(root) / "state.jsonl"
    try:
        if not path.is_file():
            return 0.0, 0
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0.0, 0
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return sum_recorded_seconds(rows), len(rows)


def _count_unit(count: int, singular: str, plural: str) -> str:
    name = singular if int(count) == 1 else plural
    return f"{int(count)} {name}"


def format_recorded_hm(seconds: float) -> str:
    """Whole hours and minutes. Seconds within the last minute are not rounded up."""
    whole = int(math.floor(max(0.0, float(seconds)) + 1e-9))
    hours, rem = divmod(whole, 3600)
    minutes = rem // 60
    return f"{_count_unit(hours, 'hour', 'hours')} {_count_unit(minutes, 'minute', 'minutes')}"


_NO_SESSION = object()


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total = int(round(max(0.0, float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_bytes(num: int) -> str:
    num = max(0, int(num))
    gib = 1024 ** 3
    if num >= gib:
        return f"{num / gib:.2f} GB"
    return f"{num / (1024 ** 2):.0f} MB"


def format_parameter_count(count: int) -> str:
    """Plain unit. One million is ``1.0 million``."""
    number = max(0, int(count))
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f} million"
    if number >= 1_000:
        return f"{number / 1_000:.1f} thousand"
    return str(number)


def parameter_fact(net: Any = None) -> str:
    """Net size for the panel.

    With ``net`` the count is its trainable ``sum(p.numel())``. Without one the
    words say approximate when the count is the architecture sum and the
    weights have not been built.
    """
    from python.train.scene_net import scene_parameter_count

    count, approximate = scene_parameter_count(net)
    unit = format_parameter_count(count)
    # The rounded unit stays. The integer is the architecture sum, including GroupNorm.
    if approximate:
        return f"params approximate {unit} ({count})"
    return f"params {unit} ({count})"


def format_status(status: TrainStatus, *, params: str | None = None) -> str:
    loss = "--" if status.loss is None else f"{status.loss:.4f}"
    memory = format_bytes(status.memory_bytes)
    # Same words as the memory graph's caption, with the amount inside.
    memory_row = f"VRAM {memory}" if status.memory_kind == "vram" else f"system RAM {memory} used"
    note = "recommended" if status.recommended else "not recommended"
    return "\n".join(
        (
            f"recorded {format_recorded_hm(status.recorded_s)}",
            f"frames {int(status.recorded_frames)}",
            f"step {int(status.step)}/{int(status.steps)}",
            f"time left {format_duration(status.eta_s)}",
            f"loss {loss}",
            f"lr {status.lr:.6g}",
            f"steps/s {status.steps_per_sec:.2f}",
            memory_row,
            f"{status.device}  batch {int(status.batch)}  {note}",
            parameter_fact() if params is None else params,
        )
    )


def loss_graph_box(height: int = PANEL_H, width: int = PANEL_W) -> tuple[int, int, int, int]:
    bottom = height - 16
    top = bottom - GRAPH_H
    return (PAD_X, top, width - PAD_X, bottom)


def loss_plot_box(graph: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Inset under the graph caption, same padding as the supervisor pedal trace."""
    x0, y0, x1, y1 = graph
    return (x0 + 8, y0 + 32, x1 - 8, y1 - 8)


def clean_series(values: Iterable[Any]) -> list[float]:
    """Finite, non-negative values. Anything else is not plotted."""
    series: list[float] = []
    for raw in values:
        number = _finite_number(raw)
        if number is None or number < 0:
            continue
        series.append(number)
    return series


def loss_polyline(
    values: Iterable[float], box: tuple[int, int, int, int], ymax: float | None = None
) -> list[tuple[int, int]]:
    """Pixel points for a series scaled to ``ymax`` (its own max by default)."""
    series = clean_series(values)
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0 or not series:
        return []
    if ymax is None:
        ymax = max(series)
    if ymax <= 0.0:
        ymax = 1.0
    last = max(1, len(series) - 1)
    points: list[tuple[int, int]] = []
    for index, value in enumerate(series):
        frac_x = index / last
        frac_y = max(0.0, min(1.0, value / ymax))
        points.append((int(x0 + frac_x * (x1 - x0)), int(y1 - frac_y * (y1 - y0))))
    return points


def _text_size(text: str, scale: float, thick: int = TH) -> tuple[int, int]:
    (width, height), _ = cv2.getTextSize(text or " ", FONT, scale, thick)
    return int(width), int(height)


def _fit(text: str, max_w: int, scale: float, thick: int = TH) -> str:
    shown = str(text or "")
    while shown and _text_size(shown, scale, thick)[0] > max_w:
        shown = shown[:-1]
    return shown


def _put(img: np.ndarray, text: str, xy: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
    cv2.putText(img, text, xy, FONT, scale, color, TH, cv2.LINE_AA)


def _put_part(img: np.ndarray, text: str, x: int, baseline: int, right: int, color: Color) -> int:
    """One caption part at ``x``, cut to fit before ``right``. Returns the next ``x``."""
    if x >= right:
        return x
    text = _fit(text, right - x, FS_DIM)
    _put(img, text, (x, baseline), FS_DIM, color)
    return x + _text_size(text + "  ", FS_DIM)[0]


def _fmt_rate(value: float) -> str:
    return f"{value:.2f}"


def _fmt_lr(value: float) -> str:
    return f"{value:.3g}"


def _fmt_hours(value: float) -> str:
    return f"{value:.1f} h"


def _fmt_frames(value: float) -> str:
    """Compact, to fit a half-width caption. The text rows carry the exact count."""
    count = int(value)
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M fr"
    if count >= 10_000:
        return f"{count / 1_000:.1f}k fr"
    return f"{count} fr"


def _fmt_loss(value: float) -> str:
    return f"{value:.4f}"


def _fmt_bytes(value: float) -> str:
    return format_bytes(int(value))


def format_rss(value: int | None) -> str:
    """Caption value for the trainer RSS: ``n/a`` when it cannot be measured, never 0."""
    return NA if value is None else format_bytes(int(value))


Color = tuple[int, int, int]
Fmt = Callable[[float], str]
# (graph, title, ((series key, caption label, color, formatter), ...), shared y scale)
GRAPHS: tuple[tuple[str, str, tuple[tuple[str, str, Color, Fmt], ...], bool], ...] = (
    ("eta", "time left", (("eta", "", ICE, format_duration),), False),
    ("sps", "steps/s", (("sps", "", ICE, _fmt_rate),), False),
    ("lr", "lr", (("lr", "", ICE, _fmt_lr),), False),
    ("driving", "driving", (("hours", "", ICE, _fmt_hours), ("frames", "", AMBER, _fmt_frames)), False),
    (
        "memory",
        "memory",
        (("vram", "VRAM", ICE, _fmt_bytes), ("ram", RAM_LABEL, AMBER, _fmt_bytes), ("rss", RSS_LABEL, VIOLET, _fmt_bytes)),
        True,
    ),
    ("loss", "loss", (("train", "", ICE, _fmt_loss), ("val", VAL_LABEL, AMBER, _fmt_loss)), True),
)
SERIES_KEYS: tuple[str, ...] = tuple(series[0] for graph in GRAPHS for series in graph[2])


def graph_boxes(height: int = PANEL_H, width: int = PANEL_W) -> dict[str, tuple[int, int, int, int]]:
    """Every graph's box. Loss is the full-width bottom box; the rest stack above it.

    Rows, top down: time left | steps/s, lr | driving, memory, loss.
    """
    loss = loss_graph_box(height, width)
    half = (width - PAD_X * 2 - GAP) // 2
    left, right = (PAD_X, PAD_X + half), (width - PAD_X - half, width - PAD_X)
    mem_bottom = loss[1] - GAP
    mem_top = mem_bottom - SMALL_H
    row2_bottom = mem_top - GAP
    row2_top = row2_bottom - SMALL_H
    row1_bottom = row2_top - GAP
    row1_top = row1_bottom - SMALL_H
    return {
        "eta": (left[0], row1_top, left[1], row1_bottom),
        "sps": (right[0], row1_top, right[1], row1_bottom),
        "lr": (left[0], row2_top, left[1], row2_bottom),
        "driving": (right[0], row2_top, right[1], row2_bottom),
        "memory": (PAD_X, mem_top, width - PAD_X, mem_bottom),
        "loss": loss,
    }


def _graphs_height() -> int:
    return 3 * (SMALL_H + GAP) + GRAPH_H + 16


def _paint_series(
    img: np.ndarray,
    box: tuple[int, int, int, int],
    title: str,
    series: Sequence[tuple[str, Sequence[float], Color, Fmt]],
    *,
    shared: bool,
    unavailable: Sequence[str] = (),
) -> None:
    """One painter for every graph. No data keeps the plot area ``GRAPH_FILL``.

    ``series`` is ``(caption label, values, color, formatter)``. With ``shared``
    every line uses one y scale (train vs val, VRAM vs RAM). ``unavailable``
    holds caption labels that cannot be measured; each is written as ``n/a``.
    """
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0, y0), (x1, y1), GRAPH_FILL, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), GRAPH_EDGE, 1)
    drawn = [(label, clean_series(values), color, fmt) for label, values, color, fmt in series]
    drawn = [row for row in drawn if row[1]]
    if not drawn and not unavailable:
        _put(img, _fit(f"{title}  --", x1 - x0 - 16, FS_DIM), (x0 + 8, y0 + 22), FS_DIM, DIM)
        return
    # The caption is the legend: each value is written in its line's color.
    x = x0 + 8
    right = x1 - 8
    if len(drawn) + len(unavailable) < 3:
        # Three labelled values fill a full-width caption; the labels name the graph.
        x = _put_part(img, title, x, y0 + 22, right, FG)
    for label, values, color, fmt in drawn:
        x = _put_part(img, f"{label} {fmt(values[-1])}".strip(), x, y0 + 22, right, color if len(drawn) > 1 else FG)
    for label in unavailable:
        x = _put_part(img, f"{label} {NA}".strip(), x, y0 + 22, right, DIM)
    if not drawn:
        return
    ymax = max(max(values) for _label, values, _color, _fmt in drawn) if shared else None
    plot = loss_plot_box(box)
    for index, (_label, values, color, _fmt) in enumerate(drawn):
        points = loss_polyline(values, plot, ymax)
        if not shared and index:
            # Separate scales can land on the same pixels (two flat lines). Nudge each
            # later line by the drawn line height so the one under it stays visible.
            nudge = index * (LINE_W + 1)  # an anti-aliased line covers LINE_W + 1 rows
            points = [(px, py - nudge if py - nudge >= plot[1] else py + nudge) for px, py in points]
        if len(points) == 1:
            cv2.circle(img, points[0], 3, color, -1, cv2.LINE_AA)
        elif points:
            cv2.polylines(img, [np.array(points, dtype=np.int32)], False, color, LINE_W, cv2.LINE_AA)


def _button(
    img: np.ndarray,
    label: str,
    x: int,
    y: int,
    hits: list[dict[str, Any]] | None,
    ident: str,
    *,
    on: bool,
) -> int:
    tw, _th = _text_size(label, FS_BODY)
    rect = (x, y - 22, x + tw + 20, y + 8)
    bg = ICE if on else (42, 38, 36)
    fg = (16, 13, 12) if on else FG
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), bg, -1)
    _put(img, label, (x + 10, y), FS_BODY, fg)
    if hits is not None:
        hits.append({"kind": "train", "id": ident, "rect": rect})
    return rect[2] + 12


def render_train_panel(
    lines: Iterable[str],
    losses: list[float] | None = None,
    *,
    phase: str = "idle",
    hits: list[dict[str, Any]] | None = None,
    series: Mapping[str, Sequence[float]] | None = None,
    unavailable: Iterable[str] = (),
) -> np.ndarray:
    """Dark panel. ``losses`` is the train loss; ``series`` holds the other graphs' values.

    Every series is plotted as given; an empty one stays idle. A key in
    ``unavailable`` with no values is captioned ``n/a`` (not 0, not left out).
    """
    text_lines = [str(line) for line in lines]
    if hits is not None:
        hits.clear()
    first = 34 + 14 + ROW_H
    button_top = first + max(1, len(text_lines)) * ROW_H + 8
    graph_top = button_top + ROW_H
    height = graph_top + _graphs_height()
    img = np.full((height, PANEL_W, 3), BG, dtype=np.uint8)
    y = 34
    _put(img, "GVD  ·  TRAIN", (PAD_X, y), FS_TITLE, ICE)
    y = first
    for text in text_lines:
        color = FG
        if text.strip() == "idle":
            color = DIM
        elif text.strip() == "paused":
            color = ICE
        elif "not recommended" in text:
            color = WARN
        _put(img, _fit(text, PANEL_W - PAD_X * 2, FS_BODY), (PAD_X, y), FS_BODY, color)
        y += ROW_H
    running = phase == "running"
    baseline = button_top + 22
    x = _button(img, "Start", PAD_X, baseline, hits, "start", on=not running)
    _button(img, "Pause", x, baseline, hits, "pause", on=running)
    values: dict[str, Sequence[float]] = dict(series or {})
    values["train"] = list(losses or [])
    missing = set(unavailable)
    boxes = graph_boxes(height, PANEL_W)
    for name, title, keys, shared in GRAPHS:
        rows = [(label, values.get(key) or [], color, fmt) for key, label, color, fmt in keys]
        na = [label for key, label, _color, _fmt in keys if key in missing and not values.get(key)]
        _paint_series(img, boxes[name], title, rows, shared=shared, unavailable=na)
    return img


def read_pause_file(path: Path | str) -> dict[str, Any] | None:
    """Load exactly the pause file at ``path`` (the one the trainer resolved)."""
    from python.train.checkpoint import PAUSE_SUFFIX, load_checkpoint

    target = Path(path)
    if not target.name.endswith(PAUSE_SUFFIX) or not target.is_file():
        return None
    return load_checkpoint(target.name[: -len(PAUSE_SUFFIX)], pause_dir=target.parent)


class TrainWindow:
    """OpenCV panel in the supervisor's type. ``open`` does nothing without a display."""

    def __init__(self, *, live_host: bool = True) -> None:
        # Host RAM and RSS are sampled only for a live trainer. A replay has no
        # host samples to draw, and this machine's numbers would be wrong there.
        self.live_host = bool(live_host)
        # True once a live sample found no RSS source (no psutil, no /proc/self/statm).
        self.rss_unavailable = False
        self.lines = ""
        self.recorded_s = 0.0
        self.recorded_frames = 0
        self.losses: list[float] = []
        self.series: dict[str, list[float]] = {key: [] for key in SERIES_KEYS if key != "train"}
        self._params: str | None = None
        self._metrics: Path | None = None
        self._metrics_at = 0
        self._since: float | None = None
        self.panel: np.ndarray | None = None
        self.phase = "idle"
        self.hits: list[dict[str, Any]] = []
        self._folder: Path | None = None
        self._status: TrainStatus | None = None
        self._checkpoint: dict[str, Any] | None = None
        self._shown = False
        self._start_requested = False
        self._pause_requested = False

    @property
    def available(self) -> bool:
        return self._shown

    def open(self) -> None:
        if self._shown:
            return
        try:
            cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_NAME, PANEL_W, PANEL_H)
            cv2.moveWindow(WIN_NAME, 80, 80)
            cv2.setMouseCallback(WIN_NAME, self._on_mouse)
        except Exception:
            return
        self._shown = True
        self._paint()

    def set_net(self, net: Any) -> None:
        """Show the trainable ``sum(p.numel())`` of the net this run built."""
        self._params = parameter_fact(net)
        self._apply_lines()
        self._paint()

    def _params_line(self) -> str:
        return parameter_fact() if self._params is None else self._params

    def load_folder(self, root: Path | str, pause_path: Path | str | None = None) -> None:
        """Read recorded time and frames from ``root`` and show them. Does not write it.

        ``pause_path`` is the pause file the trainer resolved. When it exists the
        panel shows paused, with that run's loss curve. With no path the panel
        never looks for one.
        """
        self._folder = Path(root)
        loaded = read_pause_file(pause_path) if pause_path is not None else None
        if loaded is not None:
            self._remember_pause(loaded)
        else:
            self.phase = "idle"
            self._checkpoint = None
            if self._status is None:
                self.losses = []
        self._show_recorded(*read_recorded(self._folder))

    def refresh_recorded(self) -> None:
        """Re-read the folder after more strips have been appended."""
        if self._folder is None:
            return
        self._show_recorded(*read_recorded(self._folder))

    def load_status_dir(self, path: Path | str | None, *, since: float | None = None) -> None:
        """Tail ``metrics.jsonl`` in the run's status folder for val rows. Read-only.

        The file is appended across runs, so ``since`` (this run's start time)
        drops val rows whose ``t`` is older. ``None`` keeps every row.
        """
        from python.debug.remote_status import METRICS_NAME

        self._metrics = None if path is None else Path(path) / METRICS_NAME
        self._metrics_at = 0
        self._since = since
        self.series["val"] = []
        self._tail_val()
        self._paint()

    def _tail_val(self) -> None:
        """Read whole new lines since the last call. A half-written line waits."""
        if self._metrics is None:
            return
        try:
            with self._metrics.open("rb") as handle:
                size = handle.seek(0, 2)
                if size < self._metrics_at:
                    self._metrics_at = 0
                    self.series["val"] = []
                handle.seek(self._metrics_at)
                chunk = handle.read()
        except OSError:
            return
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        self._metrics_at += end + 1
        for raw in chunk[: end + 1].splitlines():
            try:
                row = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(row, dict) or row.get("kind") != "val":
                continue
            stamp = _finite_number(row.get("t"))
            if self._since is not None and (stamp is None or stamp < self._since):
                continue
            value = _finite_number(row.get("loss"))
            if value is not None and value >= 0:
                self.series["val"].append(value)
        del self.series["val"][:-HISTORY_N]

    def _sample(self, key: str, value: Any) -> None:
        number = _finite_number(value)
        if number is None:
            return
        values = self.series[key]
        values.append(number)
        del values[:-HISTORY_N]

    def request_start(self) -> None:
        self._start_requested = True

    def request_pause(self) -> None:
        self._pause_requested = True

    def consume_start(self) -> bool:
        armed = self._start_requested
        self._start_requested = False
        return armed

    def consume_pause(self) -> bool:
        armed = self._pause_requested
        self._pause_requested = False
        return armed

    def handle_click(self, x: int, y: int) -> str | None:
        hit = None
        for item in reversed(self.hits):
            rect = item.get("rect") or (0, 0, 0, 0)
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                hit = item
                break
        if hit is None:
            return None
        ident = str(hit.get("id") or "")
        if ident == "start" and self.phase != "running":
            self._start_requested = True
            return "start"
        if ident == "pause" and self.phase == "running":
            self._pause_requested = True
            return "pause"
        return None

    def mark_running(self) -> None:
        self.phase = "running"
        self._apply_lines()
        self._paint()

    def mark_paused(self, checkpoint: Mapping[str, Any]) -> None:
        """Show paused. The loss series becomes the saved one."""
        self._remember_pause(checkpoint)
        self._pause_requested = False
        self._apply_lines()
        self._paint()

    def restore_losses(self, losses: Iterable[float]) -> None:
        self.losses = []
        for raw in losses:
            value = _finite_number(raw)
            if value is not None and value >= 0:
                self.losses.append(value)

    def _remember_pause(self, checkpoint: Mapping[str, Any]) -> None:
        self.phase = "paused"
        self._checkpoint = dict(checkpoint)
        self.restore_losses(checkpoint.get("losses") or [])

    def _standby_lines(self) -> str:
        head = (
            f"recorded {format_recorded_hm(self.recorded_s)}\n"
            f"frames {int(self.recorded_frames)}\n{self._params_line()}"
        )
        if self.phase == "paused" and self._checkpoint is not None:
            raw_loss = self._checkpoint.get("loss")
            loss = _finite_number(raw_loss)
            loss_s = "--" if loss is None else f"{loss:.4f}"
            step = int(self._checkpoint.get("step") or 0)
            return f"{head}\npaused\nstep {step}\nloss {loss_s}"
        return f"{head}\nidle"

    def _apply_lines(self) -> None:
        if self.phase == "paused":
            if self._status is not None:
                self.lines = format_status(self._status, params=self._params_line()) + "\npaused"
                return
            self.lines = self._standby_lines()
            return
        if self._status is None:
            self.lines = self._standby_lines()
            return
        self.lines = format_status(self._status, params=self._params_line())

    def _show_recorded(self, seconds: float, frames: int = 0) -> None:
        self.recorded_s = float(seconds)
        self.recorded_frames = int(frames)
        self._sample("hours", self.recorded_s / 3600.0)
        self._sample("frames", self.recorded_frames)
        if self._status is not None:
            self._status.recorded_s = self.recorded_s
            self._status.recorded_frames = self.recorded_frames
        self._apply_lines()
        self._paint()

    def update(self, status: TrainStatus, *, record_loss: bool = True) -> None:
        self._status = status
        self.phase = "running"
        if self._folder is not None:
            status.recorded_s, status.recorded_frames = read_recorded(self._folder)
        self.recorded_s = float(status.recorded_s)
        self.recorded_frames = int(status.recorded_frames)
        if record_loss:
            value = _finite_number(status.loss)
            if value is not None and value >= 0:
                self.losses.append(value)
        self._sample("eta", status.eta_s)
        self._sample("sps", status.steps_per_sec)
        self._sample("lr", status.lr)
        if status.memory_kind == "vram":
            self._sample("vram", status.memory_bytes)
            if self.live_host:
                self._sample("ram", self.sample_ram())
        else:
            self._sample("ram", status.memory_bytes)
        if self.live_host:
            rss = self.sample_rss()
            self.rss_unavailable = rss is None
            self._sample("rss", rss)
        self._sample("hours", self.recorded_s / 3600.0)
        self._sample("frames", self.recorded_frames)
        self._tail_val()
        self._apply_lines()
        self._paint()

    @staticmethod
    def sample_ram() -> int | None:
        """Used RAM, sampled in this process (no subprocess, no server)."""
        from python.train.fit import read_used_ram_bytes

        try:
            return int(read_used_ram_bytes())
        except Exception:
            return None

    @staticmethod
    def sample_rss() -> int | None:
        """This process's resident set (the trainer's own RSS). ``None`` when unknown."""
        return trainer_rss_bytes()

    def _paint(self) -> None:
        self.panel = render_train_panel(
            self.lines.splitlines(),
            self.losses,
            phase=self.phase,
            hits=self.hits,
            series=self.series,
            unavailable=("rss",) if self.rss_unavailable else (),
        )
        if not self._shown:
            return
        try:
            height, width = self.panel.shape[:2]
            cv2.resizeWindow(WIN_NAME, width, height)
            cv2.imshow(WIN_NAME, self.panel)
            cv2.waitKey(1)
        except Exception:
            self._shown = False

    def window_visible(self) -> bool:
        if not self._shown:
            return False
        try:
            return float(cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE)) >= 1
        except Exception:
            return False

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.panel is None:
            return
        ix, iy = int(x), int(y)
        try:
            rect = cv2.getWindowImageRect(WIN_NAME)
            window_rect = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        except Exception:
            window_rect = None
        from python.viz.nerd import image_point_from_window_mouse

        height, width = self.panel.shape[:2]
        ix, iy = image_point_from_window_mouse(
            ix, iy, image_wh=(width, height), window_rect=window_rect,
        )
        self.handle_click(ix, iy)

    def pump(self) -> None:
        if not self._shown or self.panel is None:
            return
        try:
            cv2.imshow(WIN_NAME, self.panel)
            cv2.waitKey(1)
        except Exception:
            self._shown = False

    def close(self) -> None:
        shown = self._shown
        self._shown = False
        if not shown:
            return
        try:
            cv2.destroyWindow(WIN_NAME)
        except Exception:
            pass


def trainer_rss_bytes() -> int | None:
    """Resident set size of this process: psutil, else Linux ``/proc/self/statm``."""
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def replay_statuses(status_dir: Path | str) -> list[TrainStatus]:
    """One ``TrainStatus`` per train row in a run's ``metrics.jsonl``. Read-only."""
    from python.debug.remote_status import HEARTBEAT_NAME, METRICS_NAME

    folder = Path(status_dir)
    try:
        heart = json.loads((folder / HEARTBEAT_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        heart = {}
    if not isinstance(heart, dict):
        heart = {}
    rows = [row for row in _read_json_lines(folder / METRICS_NAME) if row.get("kind") == "train"]
    per_epoch = int(_finite_number(heart.get("steps_per_epoch")) or 0)
    epochs = int(_finite_number(heart.get("epochs")) or 0)
    last = max((int(_finite_number(row.get("step")) or 0) for row in rows), default=0)
    steps = max(per_epoch * epochs, last)
    device = str(heart.get("device") or "cpu")
    out: list[TrainStatus] = []
    for row in rows:
        step = int(_finite_number(row.get("step")) or 0)
        step_s = _finite_number(row.get("step_s"))
        gpu = _finite_number(row.get("gpu_mem_bytes"))
        out.append(
            TrainStatus(
                eta_s=None if not step_s else (steps - step) * step_s,
                loss=_finite_number(row.get("loss")),
                lr=_finite_number(row.get("lr")) or 0.0,
                steps_per_sec=1.0 / step_s if step_s else 0.0,
                memory_bytes=int(gpu or 0),
                memory_kind="vram" if gpu is not None else "ram",
                device=device,
                batch=int(_finite_number(heart.get("batch")) or 0),
                recommended=device == "cuda",
                step=step,
                steps=steps,
            )
        )
    return out


def render_replay(
    status_dir: Path | str, *, strips: Path | str | None = None, pause: Path | str | None = None
) -> np.ndarray:
    """Panel image from a run's files. No window is opened and nothing is written."""
    window = TrainWindow(live_host=False)
    window._apply_lines()
    if strips is not None:
        window.load_folder(strips)
    window.load_status_dir(status_dir)
    for status in replay_statuses(status_dir):
        window.update(status)
    loaded = read_pause_file(pause) if pause is not None else None
    if loaded is not None:
        window._status = None
        window.mark_paused(loaded)
    assert window.panel is not None
    return window.panel


def _data_root_path() -> Path:
    """``data_root()``; when it refuses the drive, the configured path itself."""
    from python.data.data_root import DEFAULT_DATA_DIR, data_root

    try:
        return data_root()
    except Exception:
        raw = os.environ.get("GVD_DATA_DIR", "").strip()
        return Path(raw) if raw else DEFAULT_DATA_DIR


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def replay_out_refusal(out: Path | str) -> str | None:
    """Why ``--out`` may not be written, or ``None``. Checked before rendering."""
    path = Path(out)
    if path.exists() or path.is_symlink():
        return "it already exists"
    root = _data_root_path()
    for candidate, base in ((path.absolute(), root.absolute()), (path.resolve(), root.resolve())):
        if _under(candidate, base):
            return f"it is under the data root {root}"
    if not path.parent.is_dir():
        return "its folder does not exist"
    return None


def _write_png_create_only(out: Path, panel: np.ndarray) -> None:
    """The module's one file write. Exclusive create: an existing file is never replaced."""
    ok, png = cv2.imencode(".png", panel)
    if not ok:
        raise ValueError("PNG encode failed")
    with open(out, "xb") as handle:
        handle.write(png.tobytes())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the GVD TRAIN panel from a run's files (no window).")
    parser.add_argument("--replay", required=True, help="the run's status folder (metrics.jsonl, heartbeat)")
    parser.add_argument("--strips", default="", help="strip folder for recorded hours and frames")
    parser.add_argument("--pause", default="", help="pause file to show as paused")
    parser.add_argument("--out", required=True, help="PNG to write")
    args = parser.parse_args(argv)
    out = Path(args.out)
    refusal = replay_out_refusal(out)
    if refusal:
        print(f"[GVD] replay: refused --out {out}: {refusal}", flush=True)
        return 2
    panel = render_replay(args.replay, strips=args.strips or None, pause=args.pause or None)
    try:
        _write_png_create_only(out, panel)
    except FileExistsError:
        print(f"[GVD] replay: refused --out {out}: it already exists", flush=True)
        return 2
    except (OSError, ValueError) as exc:
        print(f"[GVD] replay: could not write {out}: {exc}", flush=True)
        return 1
    print(f"[GVD] replay panel {panel.shape[1]}x{panel.shape[0]} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
