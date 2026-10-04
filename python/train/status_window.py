"""Supervisor-style panel for a scene-training run.

Colors, type, and row spacing match the nerd panel in ``python/viz/nerd.py``.
The window shows recorded hours and minutes from the strip folder, time left,
loss, learning rate, steps per second, and the memory in use. The loss graph
is the steps this run has reported. There is no server.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

# Same ground, type, and rhythm as the supervisor nerd panel.
BG = (16, 13, 12)
FG = (212, 204, 200)
ICE = (212, 196, 158)
DIM = (120, 120, 120)
GRAPH_FILL = (28, 24, 22)
GRAPH_EDGE = (70, 64, 58)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FS_TITLE = 0.92
FS_BODY = 0.72
FS_DIM = 0.60
TH = 2
ROW_H = 32
PAD_X = 16
PANEL_W = 560
PANEL_H = 520
GRAPH_H = 168
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
    path = Path(root) / "state.jsonl"
    try:
        if not path.is_file():
            return 0.0
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0.0
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
    return sum_recorded_seconds(rows)


def format_recorded_hm(seconds: float) -> str:
    """Whole hours and minutes. Seconds within the last minute are not rounded up."""
    whole = int(math.floor(max(0.0, float(seconds)) + 1e-9))
    hours, rem = divmod(whole, 3600)
    minutes = rem // 60
    return f"{hours} hours {minutes} minutes"


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


def format_status(status: TrainStatus) -> str:
    loss = "--" if status.loss is None else f"{status.loss:.4f}"
    kind = "VRAM" if status.memory_kind == "vram" else "RAM"
    note = "recommended" if status.recommended else "not recommended"
    return "\n".join(
        (
            f"recorded {format_recorded_hm(status.recorded_s)}",
            f"step {int(status.step)}/{int(status.steps)}",
            f"time left {format_duration(status.eta_s)}",
            f"loss {loss}",
            f"lr {status.lr:.6g}",
            f"steps/s {status.steps_per_sec:.2f}",
            f"{kind} {format_bytes(status.memory_bytes)}",
            f"{status.device}  batch {int(status.batch)}  {note}",
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


def loss_polyline(values: Iterable[float], box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """Pixel points for the loss values. An empty series draws nothing."""
    series: list[float] = []
    for raw in values:
        number = _finite_number(raw)
        if number is None or number < 0:
            continue
        series.append(number)
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0 or not series:
        return []
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


def _paint_loss_graph(img: np.ndarray, values: list[float], graph: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = graph
    cv2.rectangle(img, (x0, y0), (x1, y1), GRAPH_FILL, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), GRAPH_EDGE, 1)
    if not values:
        _put(img, "loss   idle", (x0 + 8, y0 + 22), FS_DIM, DIM)
        return
    latest = values[-1]
    _put(img, _fit(f"loss {latest:.4f}", x1 - x0 - 16, FS_DIM), (x0 + 8, y0 + 22), FS_DIM, FG)
    points = loss_polyline(values, loss_plot_box(graph))
    if not points:
        return
    if len(points) == 1:
        cv2.circle(img, points[0], 3, ICE, -1, cv2.LINE_AA)
        return
    cv2.polylines(img, [np.array(points, dtype=np.int32)], False, ICE, 2, cv2.LINE_AA)


def render_train_panel(lines: Iterable[str], losses: list[float] | None = None) -> np.ndarray:
    """Dark panel. ``losses`` is plotted as given. An empty list stays idle."""
    text_lines = [str(line) for line in lines]
    first = 34 + 14 + ROW_H
    graph_top = first + max(1, len(text_lines)) * ROW_H + 8
    height = graph_top + GRAPH_H + 16
    img = np.full((height, PANEL_W, 3), BG, dtype=np.uint8)
    y = 34
    _put(img, "GVD  ·  TRAIN", (PAD_X, y), FS_TITLE, ICE)
    y = first
    for text in text_lines:
        color = FG
        if text.strip() == "idle":
            color = DIM
        elif "not recommended" in text:
            color = (70, 70, 220)
        _put(img, _fit(text, PANEL_W - PAD_X * 2, FS_BODY), (PAD_X, y), FS_BODY, color)
        y += ROW_H
    graph = (PAD_X, graph_top, PANEL_W - PAD_X, graph_top + GRAPH_H)
    _paint_loss_graph(img, list(losses or []), graph)
    return img


class TrainWindow:
    """OpenCV panel in the supervisor's type. ``open`` does nothing without a display."""

    def __init__(self) -> None:
        self.lines = ""
        self.recorded_s = 0.0
        self.losses: list[float] = []
        self.panel: np.ndarray | None = None
        self._folder: Path | None = None
        self._status: TrainStatus | None = None
        self._shown = False

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
        except Exception:
            return
        self._shown = True
        self._paint()

    def load_folder(self, root: Path | str) -> None:
        """Read recorded time from ``root`` and show it. Does not write the folder."""
        self._folder = Path(root)
        self._show_recorded(read_recorded_seconds(self._folder))

    def refresh_recorded(self) -> None:
        """Re-read the folder after more strips have been appended."""
        if self._folder is None:
            return
        self._show_recorded(read_recorded_seconds(self._folder))

    def _show_recorded(self, seconds: float) -> None:
        self.recorded_s = float(seconds)
        if self._status is None:
            self.lines = f"recorded {format_recorded_hm(self.recorded_s)}\nidle"
            self._paint()
            return
        self._status.recorded_s = self.recorded_s
        self.lines = format_status(self._status)
        self._paint()

    def update(self, status: TrainStatus) -> None:
        self._status = status
        if self._folder is not None:
            status.recorded_s = read_recorded_seconds(self._folder)
            self.recorded_s = float(status.recorded_s)
        else:
            self.recorded_s = float(status.recorded_s)
        value = _finite_number(status.loss)
        if value is not None and value >= 0:
            self.losses.append(value)
        self.lines = format_status(status)
        self._paint()

    def _paint(self) -> None:
        self.panel = render_train_panel(self.lines.splitlines(), self.losses)
        if not self._shown:
            return
        try:
            height, width = self.panel.shape[:2]
            cv2.resizeWindow(WIN_NAME, width, height)
            cv2.imshow(WIN_NAME, self.panel)
            cv2.waitKey(1)
        except Exception:
            self._shown = False

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
