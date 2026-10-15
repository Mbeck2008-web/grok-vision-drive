#!/usr/bin/env python3
"""A parked or nose-up car is not a driving segment."""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.liveness import (  # noqa: E402
    Liveness,
    bad_row_reasons,
    car_is_driving,
    motion_issue_lines,
)
from python.collect.level_session import snap_to_road  # noqa: E402
from python.data.raw_frames import RawFrameWriter, controls_from_state, load_scene_rows  # noqa: E402
from python.debug.dataset_check import validate_dataset  # noqa: E402
from python.debug.gt_geom import RoadPoly  # noqa: E402
from python.debug.lane_targets import tech_from_hit  # noqa: E402
from python.train.scene_loader import select_training_rows  # noqa: E402


def check_guard_and_row_rules() -> None:
    assert not car_is_driving(0.1, 0.0)
    assert not car_is_driving(None, 0.0)
    assert not car_is_driving(8.0, 0.97)
    assert car_is_driving(1.0, 0.0)
    assert car_is_driving(8.0, -0.2)

    live = Liveness()
    for _ in range(100):
        assert live.update(0.1, 0.1, 0.0, 0.0, ai_should_move=True) == ""
    assert abs(live.slow_s - 10.0) < 1e-9
    assert live.update(0.1, 0.0, 0.0, 0.0, ai_should_move=True) == "stationary"
    live.reset()
    assert live.update(0.1, 8.0, 0.97, 0.0, ai_should_move=True) == "tipped"
    live.reset()
    assert live.update(0.1, 8.0, 0.0, 0.0, ai_should_move=True) == ""
    assert live.update(0.1, 8.0, 0.0, 400.0, ai_should_move=True) == "damage"
    live.reset()
    for _ in range(200):
        assert live.update(0.1, 0.0, 0.0, 0.0, ai_should_move=False) == ""

    parked = [
        {"i": i, "dt_s": 0.0 if i == 0 else 1.0, "ego": {"speed_mps": 0.1, "dir": [0.0, 1.0, 0.0]}}
        for i in range(12)
    ]
    reasons = bad_row_reasons(parked)
    assert set(reasons) == set(range(12))
    assert set(reasons.values()) == {"stationary"}
    moving = dict(parked[-1])
    moving["i"] = 12
    moving["ego"] = {"speed_mps": 8.0, "dir": [0.0, 1.0, 0.0]}
    assert 12 not in bad_row_reasons(parked + [moving])
    lines = motion_issue_lines(parked)
    assert len(lines) == 1 and lines[0].startswith("stationary i=0-11")

    tipped = [{"i": 0, "dt_s": 0.1, "ego": {"speed_mps": 8.0, "dir": [0.0, 0.1, 0.97]}}]
    assert bad_row_reasons(tipped) == {0: "tipped"}
    hit = [
        {"i": 0, "dt_s": 0.1, "ego": {"speed_mps": 8.0, "dir": [0.0, 1.0, 0.0], "damage": 0.0}},
        {"i": 1, "dt_s": 0.1, "ego": {"speed_mps": 8.0, "dir": [0.0, 1.0, 0.0], "damage": 500.0}},
    ]
    assert bad_row_reasons(hit) == {1: "damage"}
    blank = [{"i": i, "dt_s": 5.0, "ego": {}} for i in range(6)]
    assert bad_row_reasons(blank) == {}

    sample = controls_from_state({"damage": 12.0, "electrics": {"wheelspeed": 1.5}, "dir": [0, 1, 0]})
    assert sample["damage"] == 12.0
    assert controls_from_state({"electrics": {}})["damage"] is None
    print("liveness rules ok")


class _Net:
    def get_road_network(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "roads": {
                "main": [
                    {"middle": [-717.121, 0.0, 118.0]},
                    {"middle": [-717.121, 200.0, 118.0]},
                ]
            }
        }


def check_snap_uses_node_z() -> None:
    placed = snap_to_road(_Net(), (-717.121, 101.0, 250.0))
    assert placed is not None
    pos, rot = placed
    assert abs(pos[2] - 118.5) < 1e-6
    assert abs(pos[2] - 250.0) > 1.0
    assert rot[0] == 0.0 and rot[1] == 0.0
    assert snap_to_road(_Net(), (245.11, -906.94, 247.46)) is None
    print(f"snap z={pos[2]:.2f}")


def check_validator_drops_parked_span() -> None:
    colour = np.zeros((8, 12, 3), dtype=np.uint8)
    colour[:, :, 1] = 40
    tech = tech_from_hit(0.2, 0.0, 3.5, False)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [])
        for index in range(12):
            writer.write(
                {"main": colour},
                t=float(index),
                ego={"speed_mps": 0.1, "dir": [0.0, 1.0, 0.0]},
                tech=tech,
                driver="beamng_ai",
            )
        writer.write(
            {"main": colour},
            t=12.0,
            ego={"speed_mps": 8.0, "dir": [0.0, 1.0, 0.0]},
            tech=tech,
            driver="beamng_ai",
        )
        rows = load_scene_rows(root)
        report = validate_dataset(root, root / "validate_out", montage_n=1)
        assert any(item.startswith("stationary i=0-11") for item in report["issues"]), report["issues"]
        kept, missing, excluded = select_training_rows(root, rows)
        assert missing == 0 and excluded == 12
        assert [int(row["i"]) for row in kept] == [12]
        assert report["raw_frames"] == 13
    print("parked span excluded")


class _Vehicle:
    def __init__(self, state: dict[str, object], stop: Path | None = None, stop_on: int = 0) -> None:
        self.state = state
        self.stop = stop
        self.stop_on = stop_on
        self.polls = 0
        self.commands: list[str] = []

    def poll_sensors(self) -> None:
        self.polls += 1
        if self.stop is not None and self.polls >= self.stop_on:
            self.stop.write_text("stop", encoding="utf-8")

    def queue_lua_command(self, command: str, *args: object) -> None:
        del args
        self.commands.append(str(command))

    def connect(self, bng: object) -> None:
        del bng

    def is_connected(self) -> bool:
        return True


class _Api:
    def __init__(self, vehicle: _Vehicle) -> None:
        self.vehicle = vehicle

    def get_current(self) -> dict[str, _Vehicle]:
        return {"ego": self.vehicle}

    def get_player_vehicle_id(self) -> dict[str, str]:
        return {"vid": "ego"}


class _Bng:
    def __init__(self, vehicle: _Vehicle) -> None:
        self.vehicles = _Api(vehicle)
        self.commands: list[str] = []

    def queue_lua_command(self, command: str, *args: object) -> None:
        del args
        self.commands.append(str(command))

    def get_road_network(self, **kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "roads": {
                "main": [
                    {"middle": [0.0, 0.0, 0.4]},
                    {"middle": [0.0, 80.0, 0.4]},
                ]
            }
        }


class _Cam:
    def __init__(self, ready_after: int) -> None:
        self.n = 0
        self.ready_after = ready_after

    def poll(self) -> dict[str, object]:
        self.n += 1
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        if self.n >= self.ready_after:
            image[0, 0] = (8, 16, 24)
        return {"colour": image}


def _install_session(bng: _Bng, cams: dict[str, _Cam]) -> tuple[object, object, object]:
    import scripts.collect_scene_lanes as collect
    import python.debug.gt_logger as gt_logger
    import python.sensors.tech as tech

    previous = (
        tech.open_tech_beamngpy,
        tech.widen_socket_timeout,
        tech.release_tech_beamngpy,
        gt_logger.load_road_polys,
        collect._attach_cameras,
        collect.time.sleep,
        collect.time.monotonic,
        collect.tick_sleep_s,
    )

    def _open(*args: object, **kwargs: object) -> _Bng:
        del args, kwargs
        return bng

    def _roads(session: object) -> tuple[list[RoadPoly], str]:
        del session
        return [RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)], ""

    def _cams(session: object, veh: object, cfg: object) -> dict[str, _Cam]:
        del session, veh, cfg
        return cams

    tech.open_tech_beamngpy = _open  # type: ignore[assignment]
    tech.widen_socket_timeout = lambda *a, **k: None  # type: ignore[assignment]
    tech.release_tech_beamngpy = lambda *a, **k: None  # type: ignore[assignment]
    gt_logger.load_road_polys = _roads  # type: ignore[assignment]
    collect._attach_cameras = _cams  # type: ignore[assignment]
    return collect, previous, tech


def _restore(collect: object, previous: tuple, tech: object, gt: object) -> None:
    import python.debug.gt_logger as gt_logger

    (
        tech.open_tech_beamngpy,
        tech.widen_socket_timeout,
        tech.release_tech_beamngpy,
        gt_logger.load_road_polys,
        collect._attach_cameras,
        collect.time.sleep,
        collect.time.monotonic,
        collect.tick_sleep_s,
    ) = previous
    del gt


def check_recover_stall_is_a_break() -> None:
    """The 10 s before a logged recover is a segment break, not a validate issue."""
    stalled = [
        {
            "i": index,
            "dt_s": 0.0 if index == 0 else 1.0,
            "segment": "west_coast_usa#0",
            "excursion": {"phase": "cruise", "reason": ""},
            "ego": {"speed_mps": 0.1, "dir": [0.0, 1.0, 0.0]},
        }
        for index in range(12)
    ]
    recover = {
        "i": 12,
        "dt_s": 5.3,
        "segment": "west_coast_usa#0",
        "recover": True,
        "excursion": {"phase": "cruise", "reason": ""},
        "ego": {"speed_mps": 1.16, "dir": [0.0, 1.0, 0.0]},
    }
    rows = stalled + [recover]
    assert set(bad_row_reasons(rows)) == set(range(12))
    assert motion_issue_lines(rows) == []
    fast = {
        "i": 12,
        "dt_s": 1.0,
        "segment": "west_coast_usa#0",
        "ego": {"speed_mps": 8.0, "dir": [0.0, 1.0, 0.0]},
    }
    assert any(line.startswith("stationary i=0-11") for line in motion_issue_lines(stalled + [fast]))
    assert any(line.startswith("stationary") for line in motion_issue_lines(stalled))
    abort = {
        "i": 12,
        "dt_s": 0.2,
        "segment": "west_coast_usa#0",
        "excursion": {"phase": "abort", "reason": "stationary"},
        "ego": {"speed_mps": 8.0, "dir": [0.0, 1.0, 0.0]},
    }
    assert motion_issue_lines(stalled + [abort]) == []

    colour = np.zeros((8, 12, 3), dtype=np.uint8)
    colour[:, :, 1] = 40
    tech = tech_from_hit(0.2, 0.0, 3.5, False)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [])
        for index in range(12):
            writer.write(
                {"main": colour},
                t=float(index),
                ego={"speed_mps": 0.1, "dir": [0.0, 1.0, 0.0]},
                tech=tech,
                driver="beamng_ai",
                tags={"segment": "west_coast_usa#0"},
            )
        writer.write(
            {"main": colour},
            t=17.0,
            ego={"speed_mps": 1.16, "dir": [0.0, 1.0, 0.0]},
            tech=tech,
            driver="beamng_ai",
            tags={"segment": "west_coast_usa#0", "recover": True},
        )
        stored = load_scene_rows(root)
        report = validate_dataset(root, root / "validate_out", montage_n=1)
        assert not any(item.startswith("stationary") for item in report["issues"]), report["issues"]
        kept, missing, excluded = select_training_rows(root, stored)
        assert missing == 0 and excluded == 12
        assert [int(row["i"]) for row in kept] == [12]
    print("recover stall is a segment break")


def check_warmup_skips_partial_cameras() -> None:
    import scripts.collect_scene_lanes as collect
    from python.collect.auto_mode import stop_file

    root = Path(tempfile.mkdtemp())
    out = root / "scene"
    state = {
        "pos": [0.4, 12.0, 0.5],
        "dir": [0.0, 1.0, 0.0],
        "electrics": {"wheelspeed": 8.0, "steering_input": 0.1, "throttle_input": 0.2, "brake_input": 0.0},
    }
    vehicle = _Vehicle(state, stop_file(out), stop_on=2)
    bng = _Bng(vehicle)
    cams = {"main": _Cam(1), "wide": _Cam(2)}
    collect, previous, tech = _install_session(bng, cams)
    collect.time.sleep = lambda _seconds: None  # type: ignore[method-assign]
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = collect.main(
            ["--minutes", "1", "--smoke-s", "0", "--data-dir", str(root), "--out", str(out), "--host", "127.0.0.1", "--port", "1"]
        )
    finally:
        sys.stdout = old
        _restore(collect, previous, tech, None)
    assert code == 0, (code, buf.getvalue())
    assert vehicle.polls == 2
    rows = [json.loads(line) for line in (out / "state.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1, rows
    assert set(rows[0]["cams"]) >= {"main", "wide"}
    assert "collect cameras live" in buf.getvalue()
    print("warmup skipped the partial tick")


def check_stuck_car_aborts_without_weaving() -> None:
    import scripts.collect_scene_lanes as collect
    import python.collect.liveness as liveness

    root = Path(tempfile.mkdtemp())
    out = root / "scene"
    state = {
        "pos": [0.4, 12.0, 0.5],
        "dir": [0.0, 1.0, 0.0],
        "damage": 0.0,
        "electrics": {"wheelspeed": 0.0, "steering_input": 0.0, "throttle_input": 0.4, "brake_input": 0.0},
    }
    vehicle = _Vehicle(state)
    bng = _Bng(vehicle)
    cams = {"main": _Cam(1)}
    collect, previous, tech = _install_session(bng, cams)
    clock = {"t": 0.0}

    def sleep(seconds: float) -> None:
        clock["t"] += float(seconds or 0.0)

    def monotonic() -> float:
        return clock["t"]

    collect.time.sleep = sleep  # type: ignore[method-assign]
    collect.time.monotonic = monotonic  # type: ignore[method-assign]
    collect.tick_sleep_s = lambda _elapsed, _period=0.1: 1.0  # type: ignore[method-assign]
    old_stall = liveness.STALL_S
    liveness.STALL_S = 21.0
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = collect.main(
            ["--minutes", "1", "--smoke-s", "0", "--data-dir", str(root), "--out", str(out), "--host", "127.0.0.1", "--port", "1"]
        )
    finally:
        sys.stdout = old
        liveness.STALL_S = old_stall
        _restore(collect, previous, tech, None)
    text = buf.getvalue()
    assert code == 2, (code, text)
    assert text.count("collect abort stationary") == 2
    assert "collect segment end" in text and "stationary" in text
    assert "recover lost" in text
    assert "collect recover" in text
    assert "weave" not in text
    assert any("0.90" in line for line in bng.commands), bng.commands
    assert not any("250.00" in line for line in bng.commands)
    assert not (out / "state.jsonl").is_file()
    assert any("ai.setSpeed" in line for line in vehicle.commands)
    print("stuck abort wrote nothing")


def check_tipped_car_exits() -> None:
    import scripts.collect_scene_lanes as collect

    root = Path(tempfile.mkdtemp())
    out = root / "scene"
    state = {
        "pos": [0.4, 12.0, 0.5],
        "dir": [0.0, 0.1, 0.97],
        "electrics": {"wheelspeed": 0.1},
    }
    vehicle = _Vehicle(state)
    bng = _Bng(vehicle)
    cams = {"main": _Cam(1)}
    collect, previous, tech = _install_session(bng, cams)
    collect.time.sleep = lambda _seconds: None  # type: ignore[method-assign]
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        code = collect.main(
            ["--minutes", "1", "--smoke-s", "0", "--data-dir", str(root), "--out", str(out), "--host", "127.0.0.1", "--port", "1"]
        )
    finally:
        sys.stdout = old
        _restore(collect, previous, tech, None)
    text = buf.getvalue()
    assert code == 0, (code, text)
    assert text.count("collect abort tipped") == 2
    assert "collect segment end" in text and "tipped" in text
    assert "ended" in text
    assert "recover lost" not in text
    assert not (out / "state.jsonl").is_file()
    print("tipped abort wrote nothing")


def check_second_stall_ends_the_segment() -> None:
    from scripts.collect_scene_lanes import segment_fail_action

    failed: dict[str, set[str]] = {}
    assert segment_fail_action("stationary", "italy", failed, 0.0, 60.0, 360.0) == "continue"
    assert segment_fail_action("stationary", "italy", failed, 0.0, 60.0, 360.0) == "continue"
    assert failed["stationary"] == {"italy"}
    assert segment_fail_action("stationary", "east_coast_usa", failed, 0.0, 60.0, 360.0) == "continue"
    assert segment_fail_action("stationary", "hirochi_raceway", failed, 0.0, 60.0, 360.0) == "exit"
    # The skipped remainder is reassigned. Only real lost seconds fail the plan.
    assert segment_fail_action("tipped", "west_coast_usa", {}, 0.0, 30.0, 120.0) == "continue"
    assert segment_fail_action("tipped", "west_coast_usa", {}, 30.0, 0.0, 120.0) == "exit"
    assert segment_fail_action("damage", "italy", {}, 0.0, 10.0, 7200.0) == "continue"
    print("second stall ends the segment")


def check_masked_labels_and_map_switch() -> None:
    from python.debug.dataset_check import _issues_for_row
    from python.debug.lane_targets import tech_from_hit

    masked_tech = tech_from_hit(None, None, None, True)
    issues, _t, _i, _seg, masked = _issues_for_row(
        {"i": 0, "t": 0.0, "tech": masked_tech, "ego": {"speed_mps": 8.0}},
        None,
        None,
        "",
    )
    assert masked and not any("NaN" in item for item in issues), issues
    gap_issues, *_rest = _issues_for_row(
        {
            "i": 1,
            "t": 35.0,
            "dt_s": 35.0,
            "segment": "italy#1",
            "tech": tech_from_hit(0.2, 0.0, 3.5, False),
            "ego": {"speed_mps": 8.0, "dir": [0.0, 1.0, 0.0]},
        },
        0.0,
        0,
        "west_coast_usa#0",
    )
    assert not any("gap" in item for item in gap_issues), gap_issues
    same, *_rest = _issues_for_row(
        {
            "i": 1,
            "t": 35.0,
            "dt_s": 35.0,
            "segment": "italy#1",
            "tech": tech_from_hit(0.2, 0.0, 3.5, False),
            "ego": {"speed_mps": 8.0},
        },
        0.0,
        0,
        "italy#1",
    )
    assert any("gap" in item for item in same), same
    # collect6 row 2249: 3.9 s after the italy recover, car still on the same spot.
    recover_gap, *_rest = _issues_for_row(
        {
            "i": 2249,
            "t": 3.888,
            "dt_s": 3.888,
            "segment": "italy#1",
            "recover": True,
            "tech": tech_from_hit(0.2, 0.0, 3.5, False),
            "ego": {"speed_mps": 1.37, "pos": [313.8, -886.9, 266.6]},
        },
        0.0,
        2248,
        "italy#1",
        0.03,
        (315.9, -886.4),
    )
    assert not any("gap" in item for item in recover_gap), recover_gap
    driven_hole, *_rest = _issues_for_row(
        {
            "i": 2,
            "t": 4.0,
            "dt_s": 4.0,
            "segment": "italy#1",
            "tech": tech_from_hit(0.2, 0.0, 3.5, False),
            "ego": {"speed_mps": 12.0, "pos": [40.0, 0.0, 0.0]},
        },
        0.0,
        1,
        "italy#1",
        12.0,
        (0.0, 0.0),
    )
    assert any("gap" in item for item in driven_hole), driven_hole
    switch = [
        {"i": 0, "dt_s": 0.1, "segment": "west_coast_usa#0", "ego": {"speed_mps": 0.1, "dir": [0.0, 1.0, 0.0]}},
        {"i": 1, "dt_s": 36.0, "segment": "italy#1", "ego": {"speed_mps": 0.1, "dir": [0.0, 1.0, 0.0]}},
    ]
    assert bad_row_reasons(switch) == {}
    print("masked lane_gt and map switch are not issues")


def main() -> None:
    check_guard_and_row_rules()
    check_masked_labels_and_map_switch()
    check_snap_uses_node_z()
    check_validator_drops_parked_span()
    check_recover_stall_is_a_break()
    check_warmup_skips_partial_cameras()
    check_stuck_car_aborts_without_weaving()
    check_tipped_car_exits()
    check_second_stall_ends_the_segment()
    print("liveness ok")


if __name__ == "__main__":
    main()
