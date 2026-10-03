#!/usr/bin/env python3
"""Engage or disengage GVD from this machine. No supervisor window focus.

Writes ``gvd_bot_engage.json`` in the live GVD bus folder (the same folder
as ``gvd_engage.json``). The running supervisor reads it every tick once
it is this code. BeamNG.tech stays up.

  python scripts/bot_engage.py on
  python scripts/bot_engage.py off

Tech bus, when ``GVD_BEAMNG=1`` or the live ``lua_bus`` says so:
  %LOCALAPPDATA%\\BeamNG\\BeamNG.tech\\current\\Documents\\GVD\\gvd_bot_engage.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.control.actuate import bot_engage_path, write_bot_engage


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0].lower() not in ("on", "off", "engage", "disengage"):
        print("usage: python scripts/bot_engage.py on|off", file=sys.stderr)
        return 2
    engaged = args[0].lower() in ("on", "engage")
    write_bot_engage(engaged)
    print(f"{'engage' if engaged else 'disengage'} -> {bot_engage_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
