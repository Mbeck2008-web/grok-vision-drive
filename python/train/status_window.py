"""Plain window for a scene-training run.

The window shows time left, loss, learning rate, steps per second, and the
memory in use. It is a local Tk window. There is no server.
"""

from __future__ import annotations

from dataclasses import dataclass


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


def steps_per_second(done_steps: int, elapsed_s: float) -> float:
    if done_steps <= 0 or elapsed_s <= 0:
        return 0.0
    return float(done_steps) / float(elapsed_s)


def predict_eta_s(done_steps: int, elapsed_s: float, remaining_steps: int) -> float | None:
    rate = steps_per_second(done_steps, elapsed_s)
    if rate <= 0:
        return None
    return float(remaining_steps) / rate


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
        root.geometry("520x280+80+80")
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

    def update(self, status: TrainStatus) -> None:
        self.lines = format_status(status)
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
