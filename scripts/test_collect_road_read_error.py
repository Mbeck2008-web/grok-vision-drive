#!/usr/bin/env python3
"""A road read error during a recovery snap stops the collector with reason
``road_read_error`` and still propagates. Offline; reuses the collector mocks
in test_collect_spawn_fix.

``snap_to_road`` re-raises non-session errors from ``get_road_network``
(road_loader). The recovery loop records why the run stopped before the
exception leaves ``_main``; the outer handler's ``exception`` reason must
not replace it (the first reason sticks).

The data-disk floor (20 GB free) is stubbed so this runs on small CI disks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import collect_root  # noqa: E402
import scripts.collect_scene_lanes as collect  # noqa: E402
import scripts.test_collect_spawn_fix as sf  # noqa: E402


def check_value_error_in_recovery_snap() -> None:
    import python.collect.level_session as level_session
    import python.data.data_root as data_root

    def bad_snap(bng, near):
        raise ValueError("bad road network reply")

    real = (level_session.snap_to_road, data_root.free_bytes)
    level_session.snap_to_road = bad_snap
    data_root.free_bytes = lambda path: 100 * 1024 ** 3
    with collect_root("gvd-road-err-") as root:
        try:
            sf._run_collector({"stop_at": 99999}, ["--no-excursions"], root)  # crash at poll 41 -> recovery
        except ValueError as exc:
            assert "bad road network reply" in str(exc), exc
        else:
            raise AssertionError("ValueError from the recovery snap must propagate")
        finally:
            level_session.snap_to_road, data_root.free_bytes = real
        out = root / "scene"
        status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
        assert status["stop_reason"] == "road_read_error", status
        assert "ValueError: bad road network reply" in (status["stop_detail"] or ""), status
        beat = json.loads((root / "status" / collect.HEARTBEAT_FILE).read_text(encoding="utf-8"))
        assert beat["state"] == "stopped" and beat["reason"] == "road_read_error", beat
        print(f"road_read_error ok: re-raised, stop_reason={status['stop_reason']} detail={status['stop_detail']!r}")


def main() -> None:
    check_value_error_in_recovery_snap()
    print("test_collect_road_read_error: OK")


if __name__ == "__main__":
    main()
