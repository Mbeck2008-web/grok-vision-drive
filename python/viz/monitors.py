"""Win32 multi-monitor helpers for GVD VISION second screen (ctypes; no pywin32 required)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass
class Monitor:
    left: int
    top: int
    right: int
    bottom: int
    primary: bool
    index: int  # 1-based display index in enum order

    @property
    def width(self) -> int:
        return max(1, self.right - self.left)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)


def _set_dpi_aware() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except Exception:
            pass


def list_monitors() -> list[Monitor]:
    """Enumerate displays. Origins may be negative on non-primary screens."""
    if sys.platform != "win32":
        return [Monitor(0, 0, 1280, 800, True, 1)]
    _set_dpi_aware()
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        monitors: list[Monitor] = []

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", wintypes.LONG),
                ("top", wintypes.LONG),
                ("right", wintypes.LONG),
                ("bottom", wintypes.LONG),
            ]

        class MONITORINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", RECT),
                ("rcWork", RECT),
                ("dwFlags", wintypes.DWORD),
            ]

        MonitorEnumProc = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HMONITOR,
            wintypes.HDC,
            ctypes.POINTER(RECT),
            wintypes.LPARAM,
        )

        MONITORINFOF_PRIMARY = 1

        def _cb(hmon, _hdc, _lprc, _lp):
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                r = mi.rcMonitor
                primary = bool(mi.dwFlags & MONITORINFOF_PRIMARY)
                monitors.append(
                    Monitor(
                        int(r.left),
                        int(r.top),
                        int(r.right),
                        int(r.bottom),
                        primary,
                        len(monitors) + 1,
                    )
                )
            return True

        user32.EnumDisplayMonitors(0, 0, MonitorEnumProc(_cb), 0)
        if not monitors:
            return [Monitor(0, 0, 1280, 800, True, 1)]
        return monitors
    except Exception:
        return [Monitor(0, 0, 1280, 800, True, 1)]


def pick_monitor(screen: str = "auto", env_key: str = "GVD_VIZ_MONITOR") -> tuple[Monitor, str]:
    """Return (monitor, note). screen: auto|1|2|..."""
    mons = list_monitors()
    env = os.environ.get(env_key, "").strip()
    if env.isdigit():
        screen = env
    screen = (screen or "auto").lower()
    primary = next((m for m in mons if m.primary), mons[0])
    others = [m for m in mons if not m.primary]
    if screen == "auto":
        if others:
            return others[0], f"auto->non-primary #{others[0].index}"
        return primary, "auto->primary (second screen not found - drag GVD VISION)"
    try:
        idx = int(screen)
    except ValueError:
        return primary, f"bad --viz-screen={screen}; using primary"
    # 1 = primary preference; 2 = first non-primary if present else index
    if idx == 1:
        return primary, "screen 1 (primary)"
    if idx == 2 and others:
        return others[0], "screen 2 (non-primary)"
    for m in mons:
        if m.index == idx:
            return m, f"screen {idx}"
    return primary, f"screen {idx} missing - primary"


def place_opencv_window(
    win_name: str,
    screen: str = "auto",
    fullscreen: bool = False,
    margin: int = 40,
) -> str:
    """Move named OpenCV window onto chosen monitor. Returns status note."""
    import cv2

    mon, note = pick_monitor(screen)
    x = mon.left + margin
    y = mon.top + margin
    try:
        cv2.moveWindow(win_name, int(x), int(y))
    except Exception as e:
        note = f"{note}; moveWindow failed: {e}"
    if fullscreen:
        try:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
    if "not found" in note or "missing" in note:
        print(f"[GVD] {note}")
    else:
        print(f"[GVD] GVD VISION -> monitor {mon.index} @ ({x},{y}) [{note}]")
    return note
