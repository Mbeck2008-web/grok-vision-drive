"""Plain window for a scene-training run.

The window shows recorded hours and minutes from the strip folder, plus time
left, loss, learning rate, steps per second, and the memory in use. It is a
local Tk window. There is no server.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


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


class TrainWindow:
    """Local status window. ``open`` does nothing when Tk has no display."""

    def __init__(self) -> None:
        self.lines = ""
        self.recorded_s = 0.0
        self._folder: Path | None = None
        self._status: TrainStatus | None = None
        self._root = None
        self._label = None

    @property
    def available(self) -> bool:
        return self._root is not None

    def open(self) -> None:
        if self._root is not None:
            return
        try:
            import tkinter as tk
        except ImportError:
            return
        try:
            root = tk.Tk()
        except Exception:
            return
        root.title("Grok Vision Drive")
        root.geometry("520x320+80+80")
        root.resizable(False, False)
        root.configure(bg="white")
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        label = tk.Label(
            root,
            text="measuring",
            justify="left",
            anchor="nw",
            font=("TkFixedFont", 13),
            padx=18,
            pady=16,
            bg="white",
            fg="black",
        )
        label.pack(fill="both", expand=True)
        self._root = root
        self._label = label
        self.pump()

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
            self.lines = f"recorded {format_recorded_hm(self.recorded_s)}"
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
        self.lines = format_status(status)
        self._paint()

    def _paint(self) -> None:
        if self._label is None:
            return
        try:
            self._label.configure(text=self.lines)
            self.pump()
        except Exception:
            self._root = None
            self._label = None

    def pump(self) -> None:
        if self._root is None:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except Exception:
            self._root = None
            self._label = None

    def close(self) -> None:
        root = self._root
        self._root = None
        self._label = None
        if root is None:
            return
        try:
            root.destroy()
        except Exception:
            pass
