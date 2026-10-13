"""Stdout that survives a pipe.

Windows pipes often use cp1252. A log line with U+2192 (``→``) then raises
UnicodeEncodeError and the supervisor dies. ``errors=replace`` keeps the line.
"""

from __future__ import annotations

import sys


def harden_stdio() -> None:
    """Ask stdout and stderr to replace characters they cannot encode."""
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if not callable(reconf):
            continue
        try:
            reconf(errors="replace")
        except Exception:
            continue


def safe_print(*args: object, **kwargs: object) -> None:
    """Print one log line. Unencodable characters become the stream's replacement."""
    harden_stdio()
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        text = " ".join(str(arg) for arg in args)
        safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
        end = kwargs.get("end", "\n")
        print(safe, end=end, flush=bool(kwargs.get("flush", True)))
