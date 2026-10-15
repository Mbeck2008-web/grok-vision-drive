#!/usr/bin/env python3
"""Lane-target geometry, the parked-run gate, and the scene-net stop rule."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.data.data_root import (  # noqa: E402
    DataDriveError,
    drive_letter,
    refuse_system_drive,
    space_failure,
)
from python.debug.gt_geom import ego_center_x  # noqa: E402
from python.debug.lane_targets import (  # noqa: E402
    LOOKAHEADS_M,
    lane_x_from_vector,
    lookahead_xs,
    pack_lane_gt,
    score_centers,
    tech_from_hit,
    time_holdout,
)
from python.train.scene_lane_runtime import scene_lane_decision  # noqa: E402
from python.train.scene_net import LANE_GT_DIM, LOSS_TERMS, labels_for_row  # noqa: E402
from scripts.collect_scene_lanes import (  # noqa: E402
    ai_commands,
    format_disk_line,
    load_progress,
    save_progress,
    speed_at,
    waypoint_at,
)


def check_lookaheads_match_the_8m_log() -> None:
    assert LOOKAHEADS_M == (5.0, 10.0, 20.0, 30.0)
    packed = lookahead_xs(-0.4, 0.2, 3.5)
    assert packed["y_m"] == [5.0, 10.0, 20.0, 30.0]
    for y, center, left, right in zip(
        packed["y_m"], packed["center_x"], packed["left_x"], packed["right_x"]
    ):
        expect = ego_center_x(-0.4, 0.2, y)
        assert abs(center - expect) < 1e-9
        assert abs(right - left - 3.5) < 1e-9
    values, mask = pack_lane_gt(
        lane_offset_m=-0.4, heading_err_rad=0.2, lane_width_m=3.5, off_road=False,
    )
    assert values.shape == (LANE_GT_DIM,)
    assert float(mask.sum()) == LANE_GT_DIM
    got = lane_x_from_vector(values, 8.0)
    assert got is not None and abs(got - ego_center_x(-0.4, 0.2, 8.0)) < 1e-6
    empty, empty_mask = pack_lane_gt(
        lane_offset_m=None, heading_err_rad=None, lane_width_m=None, off_road=True,
    )
    assert float(empty_mask.sum()) == 0.0
    assert float(empty[-1]) == 0.0
    labels = labels_for_row({"tech": tech_from_hit(-0.4, 0.2, 3.5, False)})
    assert "lane_gt" in labels and labels["lane_gt"].shape == (LANE_GT_DIM,)
    assert "lane_gt" in LOSS_TERMS


def check_holdout_is_the_end_of_the_clock() -> None:
    rows = [{"t": float(i), "lane_x": float(i)} for i in range(10)]
    train, hold = time_holdout(rows, 0.2)
    assert train[-1]["t"] < 8.0
    assert hold[0]["t"] >= 8.0
    assert hold[-1]["t"] == 9.0
    scored = score_centers([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 4.0])
    assert scored["n"] == 4
    assert scored["r"] is not None and scored["r"] > 0.9
    assert scored["rms"] is not None and scored["rms"] > 0.0


def check_low_confidence_stops() -> None:
    assert scene_lane_decision(0.2, 0.9) == "lane"
    assert scene_lane_decision(0.2, 0.1) == "stop"
    assert scene_lane_decision(None, 0.9) == "stop"


def check_collect_schedule_and_resume() -> None:
    assert speed_at(0.0) == 8.0
    assert speed_at(180.0) == 11.0
    assert waypoint_at(0.0, ()) is None
    assert waypoint_at(200.0, ("a", "b")) == "b"
    span = ai_commands(12.0, None)
    assert any('setMode("span")' in line for line in span)
    assert any("setSpeed(12.0)" in line for line in span)
    manual = ai_commands(9.0, 'west"coast')
    assert any('setMode("manual")' in line for line in manual)
    assert all("westcoast" not in line or "west" in line for line in manual)
    assert 'ai.setTarget("westcoast")' in manual[4] or any("west" in line for line in manual)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "collect.json"
        assert load_progress(path) == 0.0
        save_progress(path, 90.0, 2700.0)
        assert load_progress(path) == 90.0
        (Path(tmp) / "blob.bin").write_bytes(b"x" * 2048)
        line = format_disk_line(120.0, 2 * 1024 * 1024)
        assert "t=2.0 min" in line and "MB/min" in line
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["target_s"] == 2700.0


def check_data_disk_rules() -> None:
    assert drive_letter(r"E:\GVD_data") == "E"
    assert drive_letter("C:/temp") == "C"
    assert drive_letter("D:/GVD") == "D"
    assert drive_letter("/tmp/gvd") == ""
    for bad in (r"C:\GVD_data", r"D:\GVD_data", "C:/foo"):
        try:
            refuse_system_drive(bad)
        except DataDriveError:
            pass
        else:
            raise AssertionError(bad)
    refuse_system_drive(r"E:\GVD_data")
    refuse_system_drive("/tmp/gvd")
    assert space_failure(19 * 1024 ** 3) is not None
    assert space_failure(21 * 1024 ** 3) is None


def check_eval_command_on_a_short_recording() -> None:
    from scripts.eval_scene_lanes import main as eval_main

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lines = []
        for i in range(12):
            tech = tech_from_hit(-0.2, 0.05, 3.5, False)
            lines.append(json.dumps({"i": i, "t": float(i), "dt_s": None if i == 0 else 1.0, "tech": tech}))
        (root / "state.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert eval_main(["--strips", str(root), "--check-targets"]) == 0


def main() -> None:
    check_lookaheads_match_the_8m_log()
    check_holdout_is_the_end_of_the_clock()
    check_low_confidence_stops()
    check_collect_schedule_and_resume()
    check_data_disk_rules()
    check_eval_command_on_a_short_recording()
    print("test_lane_targets: OK")


if __name__ == "__main__":
    main()
