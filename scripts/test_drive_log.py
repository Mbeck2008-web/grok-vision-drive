#!/usr/bin/env python3
"""Drive-log schema, GT flag isolation, and the offline lane_x report.

Plain asserts so ``run_offline_suite.py`` runs it; functions are named
``test_*`` so pytest collects them.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _straight(x: float) -> dict:
    return {
        "material": "line_white",
        "nodes": [[x, y, 0.0, 0.2] for y in (-40.0, -20.0, 0.0, 20.0, 40.0, 80.0)],
    }


def _control(lane_x: float):
    """One modular pursuit command from a straight scene path. No logger."""
    from python.control.actuate import LaneSteerState, follow_lane_steer, plan_command, steer_sample_point
    from python.train.scene_lane_runtime import straight_lane_path

    path = straight_lane_path(float(lane_x))
    cmd = plan_command(
        path_ego=path,
        planner={"aeb": "off", "target_v": 12.0, "ttc_lead": None},
        ego_speed_mps=8.0,
        seq=1,
    )
    sample = steer_sample_point(path)
    assert sample is not None
    cmd.steer = follow_lane_steer(cmd.steer, float(sample["x"]), LaneSteerState(), 0.1)
    return cmd, float(sample["x"]), float(sample["y"])


def _obs(cmd, lane_x: float, lane_y: float) -> dict:
    return {
        "wall": 1_700_000_000.0,
        "frame_id": 4,
        "frame_age_s": 0.08,
        "lane_x": lane_x,
        "lane_y": lane_y,
        "steer_cmd": float(cmd.steer),
        "steer_why": "pursuit",
        "throttle": float(cmd.throttle),
        "brake": float(cmd.brake),
        "gear": None,
        "speed_mps": 8.0,
        "yaw_rad": None,
        "pos": None,
        "applied_steer": None,
        "applied_steer_unit": None,
        "loop_hz": 3.1,
        "camera_hz": 2.8,
        "model_lane_x": lane_x,
        "steer_law": float(cmd.steer),
        "launch_yaw_rad": None,
    }


def test_row_schema_nulls_missing_fields() -> None:
    from python.debug.drive_log import ROW_FIELDS, DriveLog, record_drive_tick

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "drive.jsonl"
        log = DriveLog(path)
        row = record_drive_tick(log, _bare_cmd(), {}, include_gt=False)
        log.close()
        for key in ROW_FIELDS:
            assert key in row, key
            assert row[key] is None, (key, row[key])
        assert "gt_lane_x" not in row
        saved = json.loads(path.read_text(encoding="utf-8").strip())
        assert saved["frame_id"] is None and saved["steer_cmd"] is None
        assert "gt_lane_x" not in saved


def _bare_cmd():
    from python.control.actuate import DriveCommand

    return DriveCommand(steer=0.04, throttle=0.1, brake=0.0, seq=2, reason="ok")


def test_drive_log_gt_does_not_change_control() -> None:
    """A large logged GT offset is not a steer input. Both flags leave the command."""
    from python.debug.drive_log import DriveLog, record_drive_tick

    cmd_off, lane_x, lane_y = _control(0.31)
    cmd_on, lane_x_b, lane_y_b = _control(0.31)
    assert abs(lane_x - lane_x_b) < 1e-9 and abs(lane_y - lane_y_b) < 1e-9
    before_off = (cmd_off.steer, cmd_off.throttle, cmd_off.brake, cmd_off.reason)
    before_on = (cmd_on.steer, cmd_on.throttle, cmd_on.brake, cmd_on.reason)
    assert before_off == before_on
    assert cmd_on.steer > 0.0

    with tempfile.TemporaryDirectory() as tmp:
        log = DriveLog(Path(tmp) / "drive.jsonl")
        row_off = record_drive_tick(
            log, cmd_off, _obs(cmd_off, lane_x, lane_y), include_gt=False, gt_lane_x=-4.0,
        )
        row_on = record_drive_tick(
            log, cmd_on, _obs(cmd_on, lane_x, lane_y), include_gt=True, gt_lane_x=-4.0, gt_origin="state_pos_ref_node",
        )
        log.close()
    assert (cmd_off.steer, cmd_off.throttle, cmd_off.brake, cmd_off.reason) == before_off
    assert (cmd_on.steer, cmd_on.throttle, cmd_on.brake, cmd_on.reason) == before_on
    assert cmd_off.steer == cmd_on.steer
    assert abs(cmd_on.steer - (-4.0)) > 1.0
    assert "gt_lane_x" not in row_off
    assert row_on["gt_lane_x"] == -4.0
    assert row_on["steer_cmd"] == cmd_on.steer
    assert row_on["lane_x"] == lane_x


def test_gt_lane_x_is_the_training_label_at_8m() -> None:
    from python.debug.drive_log import lane_x_8m_from_paint
    from python.debug.lane_targets import lane_center_is_valid, lane_x_from_vector, tech_from_painted
    from python.debug.painted_lanes import PaintedLanes

    paint = PaintedLanes([_straight(-1.75), _straight(1.75)])
    pos = (0.4, 0.0, 0.0)
    direction = (0.0, 1.0, 0.0)
    got = lane_x_8m_from_paint(paint, pos, direction)
    tech = tech_from_painted(paint.lane_at(pos, direction))
    assert lane_center_is_valid(tech["lane_gt_valid"])
    want = lane_x_from_vector(tech["lane_gt"])
    assert got is not None and want is not None
    assert abs(got - want) < 1e-6
    # Car is right of the paint center, so the 8 m center is to the left.
    assert abs(got - (-0.4)) < 0.05, got


def test_missing_paint_is_null_not_a_centered_lane() -> None:
    from python.debug.drive_log import lane_x_8m_from_paint
    from python.debug.painted_lanes import PaintedLanes

    paint = PaintedLanes([_straight(-1.75)])
    assert lane_x_8m_from_paint(paint, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)) is None


def test_report_scores_model_against_gt_and_flags_opposite_steer() -> None:
    from python.debug.drive_log_report import format_report, main as report_main

    rows = [
        {
            "wall": 10.0,
            "frame_id": 1,
            "lane_x": 0.50,
            "model_lane_x": 0.50,
            "gt_lane_x": 0.20,
            "steer_cmd": 0.07,
            "steer_why": "pursuit",
            "steer_law": 0.07,
        },
        {
            "wall": 10.3,
            "frame_id": 2,
            "lane_x": 0.30,
            "model_lane_x": 0.30,
            "gt_lane_x": -0.10,
            "steer_cmd": -0.22,
            "steer_why": "pursuit",
            "steer_law": 0.045,
            "launch_yaw_rad": -0.37,
        },
        {
            "wall": 10.6,
            "frame_id": 3,
            "lane_x": None,
            "model_lane_x": None,
            "gt_lane_x": 0.2,
            "steer_cmd": 0.0,
            "steer_why": "lane_none",
        },
    ]
    text = format_report(rows)
    assert "ticks 3" in text
    assert "scored 2" in text
    # errors +0.30 and +0.40 → mean +0.35, rms sqrt((0.09+0.16)/2)
    assert "mean +0.350 m" in text
    assert "rms 0.354 m" in text
    assert "sign agreement 50.0% (1/2)" in text
    assert "opposite-sign ticks: 1" in text
    assert "frame_id=2" in text
    assert "steer_cmd=-0.220" in text
    assert "post_law" in text

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "drive.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        assert report_main([str(path)]) == 0


def test_applied_steer_unit_names_the_lock() -> None:
    """BeamNG's reported steer carries its unit. A null unit fails this test."""
    from python.control.actuate import DriveCommand
    from python.debug.drive_log import APPLIED_STEER_UNIT, DriveLog, record_drive_tick

    cmd = DriveCommand(steer=0.0, throttle=0.0, brake=0.0, seq=1, reason="not_engaged")
    with tempfile.TemporaryDirectory() as tmp:
        log = DriveLog(Path(tmp) / "drive.jsonl")
        row = record_drive_tick(log, cmd, {"applied_steer": 0.25, "frame_id": 1}, include_gt=False)
        log.close()
    assert row["applied_steer"] == 0.25
    assert row["applied_steer_unit"] == "steering_input (±1 = ±30 deg road wheel)"
    assert row["applied_steer_unit"] == APPLIED_STEER_UNIT


def test_supervisor_wires_drive_log_without_feeding_viz_or_steer() -> None:
    src = (ROOT / "python" / "run_vision.py").read_text(encoding="utf-8")
    assert "--drive-log" in src
    assert "--drive-log-gt" in src
    assert 'st["gt_lane_x"]' not in src
    assert "st['gt_lane_x']" not in src
    for line in src.splitlines():
        if "gt_lane" in line and "cmd.steer" in line:
            raise AssertionError(line)
    from python.run_vision import build_parser

    args = build_parser().parse_args([])
    assert args.drive_log == ""
    assert args.drive_log_gt is False


def test_pursuit_keeps_the_sign_of_lane_x() -> None:
    """The lane law itself does not explain a -0.22 command on a +0.3 m sample."""
    cmd, lane_x, _lane_y = _control(0.30)
    assert lane_x > 0.2
    assert cmd.steer > 0.0
    assert abs(cmd.steer) < 0.1


def test_rate_limit_leaves_the_opposite_band_in_one_tick() -> None:
    """At the Nov 10 loop rate a +0.3 m sample cannot hold steer at -0.22 for two ticks."""
    from python.control.actuate import LaneSteerState, follow_lane_steer, path_steer_from_ego
    from python.train.scene_lane_runtime import straight_lane_path

    path = straight_lane_path(0.30)
    geometric = path_steer_from_ego(path)
    assert geometric > 0.0
    state = LaneSteerState(steer=-0.22, lateral_m=-1.5, have_lateral=True, armed=True, reason="pursuit")
    first = follow_lane_steer(geometric, 0.30, state, 0.33)
    second = follow_lane_steer(geometric, 0.37, state, 0.26)
    assert not (-0.23 <= first <= -0.21), first
    assert not (-0.23 <= first <= -0.21 and -0.23 <= second <= -0.21)


def test_launch_yaw_can_oppose_positive_lane_x() -> None:
    """Within 6 s of engage, yaw past 0.35 rad replaces pursuit with heading steer.

    0.6 * -0.37 rad is -0.222, the a1/a3 band, while lane_x stays +0.3 m.
    """
    from python.control.actuate import DriveCommand, HEADING_STEER_GAIN
    from python.control.launch_guard import LAUNCH_YAW_RAD, LaunchGuard

    cmd, lane_x, _lane_y = _control(0.30)
    pursuit = float(cmd.steer)
    assert lane_x > 0.0 and pursuit > 0.0
    held = DriveCommand(steer=pursuit, throttle=cmd.throttle, brake=cmd.brake, seq=1, reason="ok")
    guard = LaunchGuard()
    guard.update(
        held, engaged=True, dt_s=0.3, heading_rad=0.0, speed_mps=5.0, damage=0.0, pos_xy=(0.0, 0.0),
    )
    yaw = -(LAUNCH_YAW_RAD + 0.02)
    guard.update(
        held, engaged=True, dt_s=0.3, heading_rad=yaw, speed_mps=5.0, damage=0.0, pos_xy=(1.0, 0.0),
    )
    assert held.steer < 0.0
    assert abs(held.steer - HEADING_STEER_GAIN * yaw) < 1e-6
    assert -0.23 <= held.steer <= -0.21
    assert held.throttle == 0.0


class _StepClock:
    """Three supervisor ticks, 0.25 s apart, then stop. ``perf`` is the loop clock."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.steps = 0

    def mono(self) -> float:
        return self.t

    def perf(self) -> float:
        return self.t

    def wall(self) -> float:
        return 1_700_000_000.0 + self.t

    def sleep(self, _seconds: float) -> None:
        self.steps += 1
        if self.steps >= 3:
            raise SystemExit(0)
        self.t += 0.25


def _install_stub_pose() -> None:
    """Stub camera plus a painted-lane pose. The actuator stays the real cmd_json stub."""
    from python.sensors.cameras import StubBackend
    from python.sensors.tech import VehicleData

    class _Level:
        def queue_lua_command(self, _command: str, *args: object, **kwargs: object) -> str:
            return "gvd_paint_map"

    def poll_vehicle(self: object) -> VehicleData:
        return VehicleData(
            connected=True,
            pos=(0.4, 0.0, 0.0),
            dir=(0.0, 1.0, 0.0),
            speed_mps=0.0,
            steering_input=0.25,
            gear=2,
            model="etk800",
        )

    StubBackend.bng = property(lambda self: _Level())  # type: ignore[method-assign]
    StubBackend.poll_vehicle = poll_vehicle  # type: ignore[method-assign]


def drive_log_supervisor_child() -> int:
    """One real ``run_vision.main`` loop. ``GVD_DRIVE_LOG_GT=1`` turns the GT flag on."""
    _install_stub_pose()
    from python.run_vision import main as run_main

    argv = [
        "run_vision.py",
        "--backend",
        "stub",
        "--vision-only",
        "--no-engage",
        "--encode",
        "cpu",
        "--hz",
        "8",
        "--drive-log",
        os.environ["GVD_DRIVE_LOG_PATH"],
    ]
    if os.environ.get("GVD_DRIVE_LOG_GT") == "1":
        argv.append("--drive-log-gt")
    sys.argv = argv
    run_main(clock=_StepClock())
    return 0


def _write_paint(data_dir: Path) -> None:
    roads = data_dir / "roads"
    roads.mkdir(parents=True, exist_ok=True)
    decals = [_straight(-1.75), _straight(1.75)]
    with gzip.open(roads / "gvd_paint_map_lines.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(decals, fh)


def _run_supervisor(log_path: Path, *, include_gt: bool) -> tuple[int, str]:
    env = dict(os.environ)
    env["GVD_DRIVE_LOG_CHILD"] = "1"
    env["GVD_DRIVE_LOG_GT"] = "1" if include_gt else "0"
    env["GVD_DRIVE_LOG_PATH"] = str(log_path)
    proc = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve())],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_live_supervisor_gt_flag_leaves_applied_command() -> None:
    """The real stub tick applies the same command with ``--drive-log-gt`` on and off.

    Ground truth is on the log row. ``loop_hz`` / ``camera_hz`` are the measured
    EMA (null on frame 0). A hardcoded 99, a null steer unit, or
    ``applied.steer = float(gt_x)`` after the paint sample fails this test.
    """
    from gvd_test_env import isolated_home

    from python.debug.drive_log import APPLIED_STEER_UNIT

    with isolated_home(prefix="gvd-drive-log-") as home:
        data = Path(home) / "data"
        _write_paint(data)
        os.environ["GVD_DATA_DIR"] = str(data)
        off_path = Path(home) / "off.jsonl"
        on_path = Path(home) / "on.jsonl"
        code_off, text_off = _run_supervisor(off_path, include_gt=False)
        code_on, text_on = _run_supervisor(on_path, include_gt=True)
        assert code_off == 0, text_off[-2000:]
        assert code_on == 0, text_on[-2000:]
        off = _rows(off_path)
        on = _rows(on_path)
        assert len(off) == 3 and len(on) == 3, (len(off), len(on), text_on[-1500:])
        command = lambda row: (row["steer_cmd"], row["throttle"], row["brake"])
        assert [command(row) for row in on] == [command(row) for row in off]
        for row in on:
            assert row["gt_lane_x"] is not None, row
            assert abs(float(row["gt_lane_x"]) - (-0.4)) < 0.05, row["gt_lane_x"]
            assert row["steer_cmd"] != row["gt_lane_x"]
        for row in off:
            assert "gt_lane_x" not in row
        for row in on + off:
            assert row["applied_steer"] == 0.25
            assert row["applied_steer_unit"] == APPLIED_STEER_UNIT
            if row["frame_id"] == 0:
                assert row["loop_hz"] is None and row["camera_hz"] is None, row
            else:
                assert abs(float(row["loop_hz"]) - 4.0) < 1e-6, row
                assert row["camera_hz"] == 0.0, row
        state_path = (
            Path(os.environ["LOCALAPPDATA"])
            / "BeamNG"
            / "BeamNG.drive"
            / "current"
            / "Documents"
            / "GVD"
            / "gvd_state.json"
        )
        assert state_path.is_file(), state_path
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert "gt_lane_x" not in state


def main() -> None:
    test_row_schema_nulls_missing_fields()
    test_drive_log_gt_does_not_change_control()
    test_gt_lane_x_is_the_training_label_at_8m()
    test_missing_paint_is_null_not_a_centered_lane()
    test_report_scores_model_against_gt_and_flags_opposite_steer()
    test_applied_steer_unit_names_the_lock()
    test_supervisor_wires_drive_log_without_feeding_viz_or_steer()
    test_live_supervisor_gt_flag_leaves_applied_command()
    test_pursuit_keeps_the_sign_of_lane_x()
    test_rate_limit_leaves_the_opposite_band_in_one_tick()
    test_launch_yaw_can_oppose_positive_lane_x()
    print("test_drive_log: OK")


if __name__ == "__main__":
    if os.environ.get("GVD_DRIVE_LOG_CHILD") == "1":
        raise SystemExit(drive_log_supervisor_child())
    main()
