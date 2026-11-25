#!/usr/bin/env python3
"""Automatic Mode options, the stop file, and the supervisor AUTO tab."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.auto_mode import (  # noqa: E402
    AutoOptions,
    AutoSession,
    collect_argv,
    environment_commands,
    format_status,
    rates_table,
    speeds_for,
    stop_file,
)
from python.viz.nerd import render_panel  # noqa: E402
from python.viz.stage import VizUI  # noqa: E402


class _Proc:
    def __init__(self) -> None:
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def wait(self, timeout: float = 0.0):
        del timeout
        self.returncode = 0
        return 0


def check_options_and_status() -> None:
    opts = AutoOptions(map_name="west_coast_usa", route="random", waypoints="a,b", minutes=30, traffic=True)
    argv = collect_argv(opts, ROOT / "scripts" / "collect_scene_lanes.py", python="py")
    assert "--map" in argv and "west_coast_usa" in argv
    assert "--route" in argv and "random" in argv
    assert "--traffic" in argv
    assert "--minutes" in argv and "30.0" in argv
    assert "beamng_ai" not in argv  # the collector writes the label; the flag list does not override it
    opts.until_disk = True
    argv = collect_argv(opts, "collect_scene_lanes.py")
    assert "--until-disk" in argv and "--minutes" not in argv
    assert speeds_for(17, 8)[0] == 8.0 and speeds_for(8, 8) == (8.0,)
    rates = rates_table("rear=1", "main,narrow")
    assert rates["main"] == 10.0 and rates["narrow"] == 10.0 and rates["pillarL"] == 0.0
    assert rates["rear"] == 1.0
    rates = rates_table("main=4", "")
    assert rates["main"] == 4.0 and rates["rear"] == 1.0
    assert any("setTimeOfDay" in line for line in environment_commands("12", ""))
    # BeamNG TimeOfDay.time: 0 is local noon, 0.25 18:00, 0.5 midnight, 0.75 06:00
    # (documentation.beamng.com/modding/levels/level_classes/tod/). An hour maps
    # to (h - 12) / 24 wrapped; a 0-1 value is already BeamNG's own time.
    for hour, want in (("12", 0.0), ("18", 0.25), ("24", 0.5), ("6", 0.75), ("21", 0.375), ("3", 0.625)):
        assert environment_commands(hour, "") == [f"core_environment.setTimeOfDay({{time={want:.4f}}})"], hour
    assert environment_commands("0.5", "") == ["core_environment.setTimeOfDay({time=0.5000})"]
    assert any("rain" in line for line in environment_commands("", "rain"))
    assert environment_commands("", "") == []
    lines = format_status(
        {
            "elapsed_s": 120.0,
            "frames": 40,
            "disk_bytes": 2 * 1024 ** 3,
            "disk_free_bytes": 500 * 1024 ** 3,
            "ai": {"mode": "span", "speed_mps": 11.0},
            "error": "",
        },
        running=True,
    )
    text = "\n".join(lines)
    assert "minutes 2.0" in text and "frames 40" in text and "disk 2.00 GB" in text
    assert "500.0 GB" in text and "span" in text and "driver=beamng_ai" in text
    plain = AutoOptions()
    plain.cycle("traffic")
    assert plain.traffic is True
    plain.cycle("disk")
    assert plain.until_disk is True
    plain.cycle("cameras")
    assert plain.cameras == "main,narrow,wide"


def check_stop_and_panel() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        session = AutoSession(script=ROOT / "scripts" / "collect_scene_lanes.py")
        session.options.out = Path(tmp)
        seen: dict[str, list[str]] = {}

        def popen(cmd):
            seen["cmd"] = list(cmd)
            return _Proc()

        assert session.start(popen=popen)
        assert session.running()
        assert str(ROOT / "scripts" / "collect_scene_lanes.py") in seen["cmd"][-1] or any(
            "collect_scene_lanes.py" in part for part in seen["cmd"]
        )
        session.stop()
        assert stop_file(Path(tmp)).is_file()
        assert not session.running()
        ui = VizUI()
        ui.auto = session
        ui.nerd_tab = "auto"
        render_panel({}, h=900, w=700, ui=ui)
        ids = [hit.get("id") for hit in ui.nerd_hits if hit.get("kind") == "auto"]
        assert "start" in ids and "stop" in ids and "map" in ids
        stop = next(hit for hit in ui.nerd_hits if hit.get("kind") == "auto" and hit.get("id") == "traffic")
        rect = stop["rect"]
        assert ui.handle_click(int(rect[0]) + 2, int((rect[1] + rect[3]) / 2), stage_w=0)
        assert ui.auto.options.traffic is True


def check_default_maps_have_a_dump() -> None:
    """AUTO rotation only uses maps with a road dump; utah stays selectable."""
    from python.collect.auto_mode import DEFAULT_MAPS, MAPS
    from python.collect.spawn_mix import spawn_roads_path

    assert DEFAULT_MAPS, DEFAULT_MAPS
    assert "utah" not in DEFAULT_MAPS and "jungle_rock_island" not in DEFAULT_MAPS, DEFAULT_MAPS
    for name in DEFAULT_MAPS:
        path = spawn_roads_path(name)
        assert path is not None and path.is_file(), (name, path)
    assert "utah" in MAPS and "jungle_rock_island" in MAPS, MAPS


def main() -> None:
    check_options_and_status()
    check_stop_and_panel()
    check_default_maps_have_a_dump()
    print("automatic mode ok")


if __name__ == "__main__":
    main()
