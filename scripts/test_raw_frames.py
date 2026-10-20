#!/usr/bin/env python3
"""Raw camera files stay byte-for-byte. Stitching happens at load time."""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from python.data.raw_frames import (  # noqa: E402
    COLLECT_HZ,
    RawFrameWriter,
    camera_sidecar,
    capture_plan,
    describe_raw_disk,
    gb_per_hour,
    load_raw_cameras,
    load_scene_rows,
    load_training_bgr,
    rig_tick_bytes,
    uncompressed_tick_bytes,
)
from python.data.strip_writer import CANVAS_H, CANVAS_W  # noqa: E402
from python.debug.dataset_check import validate_dataset  # noqa: E402
from python.debug.lane_targets import tech_from_hit  # noqa: E402
from python.sensors.cameras import load_camera_config  # noqa: E402
from scripts.collect_scene_lanes import DISK_CHECK_S  # noqa: E402


def check_rates_and_controls() -> None:
    from python.data.raw_frames import CameraSchedule, controls_from_state, plan_totals, yaw_rate_from_dirs

    cfg = load_camera_config()
    rows = capture_plan(cfg, 0.619)
    totals = plan_totals(rows)
    by_id = {row["id"]: row for row in rows}
    assert by_id["main"]["hz"] == 10.0
    assert by_id["narrow"]["hz"] == 10.0
    assert by_id["wide"]["hz"] == 10.0
    assert by_id["pillarL"]["hz"] == 2.0
    assert by_id["rear"]["hz"] == 1.0
    assert by_id["main"]["uncompressed_bytes"] == 640 * 480 * 3
    assert totals["png_gb_per_hour"] < 80.0
    assert totals["uncompressed_gb_per_hour"] < totals["png_gb_per_hour"] / 0.5
    # Phase schedule (scripts/test_camera_phase_schedule.py has the rest).
    sched = CameraSchedule()
    assert sched.tick(0.0, ["main", "rear"]) == ["main"]  # rear is staggered to 0.7 s
    assert sched.tick(0.1, ["main", "rear"]) == ["main"]
    assert sched.tick(0.15, ["main", "rear"]) == []
    assert sched.tick(0.7, ["main", "rear"]) == ["main", "rear"]
    assert sched.tick(1.0, ["main", "rear"]) == ["main"]
    sample = controls_from_state(
        {
            "pos": [1.0, 2.0, 3.0],
            "dir": [0.0, 1.0, 0.0],
            "vel": [0.0, 4.0, 0.0],
            "electrics": {"steering_input": 0.2, "throttle_input": 0.3, "brake_input": 0.0},
        }
    )
    assert sample["steer"] == 0.2 and sample["throttle"] == 0.3 and sample["brake"] == 0.0
    assert abs(sample["speed_mps"] - 4.0) < 1e-6
    yaw = yaw_rate_from_dirs((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.5)
    assert yaw is not None and abs(yaw - 3.14159265 / 2 / 0.5) < 1e-6
    print(
        f"plan png={totals['png_gb_per_hour']:.2f} GB/h "
        f"uncompressed={totals['uncompressed_gb_per_hour']:.2f} GB/h"
    )
    for row in rows:
        print(
            f"  {row['id']} {row['width']}x{row['height']} @{row['hz']:.0f}Hz "
            f"png={row['png_gb_per_hour']:.2f} GB/h"
        )


def check_disk_ceiling() -> None:
    cfg = load_camera_config()
    sizes = [(cam["resolution"][0], cam["resolution"][1], 3) for cam in camera_sidecar(cfg)]
    assert len(sizes) == 8
    assert uncompressed_tick_bytes(sizes) == 6_912_000
    assert rig_tick_bytes(cfg) == 6_912_000
    got = gb_per_hour(6_912_000, COLLECT_HZ)
    assert abs(got - 231.744) < 0.01, got
    text = describe_raw_disk(6_912_000, 10.0)
    assert "231.7 GB/h" in text
    assert DISK_CHECK_S <= 10.0


def check_pixels_roundtrip_and_load_time_stitch() -> None:
    src = inspect.getsource(RawFrameWriter.write) + inspect.getsource(
        __import__("python.data.raw_frames", fromlist=["_png_bytes"])._png_bytes
    )
    assert "resize" not in src
    assert "cvtColor" not in src
    assert "compose_strip" not in src
    colour = np.zeros((18, 22, 3), dtype=np.uint8)
    colour[:, :, 0] = np.arange(22, dtype=np.uint8)
    colour[:, :, 1] = 7
    colour[:, :, 2] = 255
    alpha = np.zeros((5, 6, 4), dtype=np.uint8)
    alpha[..., 3] = 128
    alpha[0, 0] = (1, 2, 3, 4)
    odd = np.arange(12, dtype=np.float32).reshape(3, 4)
    pose = [
        {
            "id": "main",
            "pos_m": [0.0, 1.2, 1.26],
            "yaw_deg": 0.0,
            "pitch_deg": -6.0,
            "fov_v": 34.0,
            "near_m": 0.05,
            "far_m": 300.0,
            "resolution": [22, 18],
        },
        {
            "id": "wide",
            "pos_m": [-0.03, 1.2, 1.24],
            "yaw_deg": 0.0,
            "pitch_deg": -8.0,
            "fov_v": 88.7,
            "near_m": 0.05,
            "far_m": 300.0,
            "resolution": [6, 5],
        },
    ]
    tech = tech_from_hit(0.2, 0.01, 3.6, False)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, pose)
        writer.write(
            {"main": colour, "wide": alpha},
            t=10.0,
            timestamps={"main": {"wall": 10.0, "sensor_t": 1.5}, "wide": {"wall": 10.01, "sensor_t": None}},
            ego={"speed_mps": 8.0},
            tech=tech,
        )
        writer.write(
            {"main": colour, "wide": odd},
            t=10.1,
            timestamps={"main": {"wall": 10.1, "sensor_t": 1.6}, "wide": 10.1},
            ego={"speed_mps": 8.0},
            tech=tech,
        )
        again = RawFrameWriter(root, pose)
        third = again.write({"main": colour}, t=10.2, timestamps={"main": 10.2}, tech=tech)
        assert third["i"] == 2
        assert writer.stored_bytes > 0 and again.stored_bytes > 0
        main_png = (root / "raw" / "main" / "000" / "000000.png").read_bytes()
        loaded = load_raw_cameras(root, 0)
        assert np.array_equal(loaded["main"], colour)
        assert np.array_equal(loaded["wide"], alpha)
        assert loaded["wide"].shape == (5, 6, 4)
        loaded_odd = load_raw_cameras(root, 1)["wide"]
        assert loaded_odd.dtype == np.float32
        assert np.array_equal(loaded_odd, odd)
        before = (root / "raw" / "main" / "000" / "000000.png").read_bytes()
        canvas = load_training_bgr(root, 0)
        after = (root / "raw" / "main" / "000" / "000000.png").read_bytes()
        assert before == after == main_png
        assert canvas.shape == (CANVAS_H, CANVAS_W, 3)
        state = json.loads((root / "state.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert "tech" not in state and "lane_gt" not in state
        assert state["dt_s"] is None
        assert state["cams"]["main"]["fov_v"] == 34.0
        assert state["cams"]["main"]["t"] == 10.0
        assert state["cams"]["main"]["sensor_t"] == 1.5
        label = json.loads((root / "gt.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert "lane_gt" in label["tech"]
        rows = load_scene_rows(root)
        assert rows[0]["tech"]["lane_gt"] == label["tech"]["lane_gt"]
        assert abs(float(rows[1]["dt_s"]) - 0.1) < 1e-9
        report = validate_dataset(root, root / "validate_out", montage_n=1)
        assert (root / "raw" / "main" / "000" / "000000.png").read_bytes() == main_png
        assert report["format"] == "raw"
        assert report["montages"]
        assert Path(report["montages"][0]).is_file()
        assert report["raw_frames"] == 3
        assert report["raw_files"] == 5
        assert report["per_cam"]["main"] == 3 and report["per_cam"]["wide"] == 2
        assert report["issues"] == ["raw count mismatch at 10 Hz (main=3, wide=2)"], report["issues"]


def check_validator_reads_raw_and_names_time_holes() -> None:
    """Collector output is raw/<cam>/NNNNNN.png, not strips/NNNNNN.jpg.

    A large dt between index i and i+1 is a real hole in the recording
    (the writer dropped ticks). It is not an off-by-one in the file number.
    """
    colour = np.zeros((8, 12, 3), dtype=np.uint8)
    colour[:, :, 1] = 30
    tech = tech_from_hit(0.2, 0.0, 3.5, False)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [])
        writer.write({"main": colour}, t=1000.0, tech=tech, driver="beamng_ai")
        writer.write({"main": colour}, t=1000.2, tech=tech, driver="beamng_ai")
        writer.write({"main": colour}, t=1329.228, tech=tech, driver="beamng_ai")
        canvas = load_training_bgr(root, 0)
        assert canvas.shape[2] == 3
        report = validate_dataset(root, root / "validate_out", montage_n=1)
        text = "\n".join(report["issues"])
        assert report["format"] == "raw"
        assert report["raw_frames"] == 3
        assert "strips" not in text
        assert ".jpg" not in text
        assert "unreadable" not in text
        assert "time hole" in text
        assert "1000" not in text or "329" in text
        hole = [item for item in report["issues"] if "gap" in item]
        assert len(hole) == 1 and "index 1->2" in hole[0] and "329.028s" in hole[0]
        print(hole[0])


def check_raw_count_is_one_camera() -> None:
    """Eight cameras must not be added into the printed raw tick count."""
    colour = np.zeros((8, 12, 3), dtype=np.uint8)
    colour[:, :, 1] = 40
    tech = tech_from_hit(0.2, 0.0, 3.5, False)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = RawFrameWriter(root, [])
        for index in range(3):
            writer.write(
                {"main": colour, "wide": colour},
                t=float(index),
                ego={"speed_mps": 8.0, "dir": [0.0, 1.0, 0.0]},
                tech=tech,
                driver="beamng_ai",
            )
        report = validate_dataset(root, root / "validate_out", montage_n=1)
        assert report["raw_frames"] == 3, report
        assert report["raw_files"] == 6
        assert report["per_cam"] == {"main": 3, "wide": 3}
        assert report["issue_count"] == 0, report["issues"]
    print("raw count is per camera")


def main() -> None:
    check_disk_ceiling()
    check_rates_and_controls()
    check_pixels_roundtrip_and_load_time_stitch()
    check_validator_reads_raw_and_names_time_holes()
    check_raw_count_is_one_camera()
    print("raw frames ok")


if __name__ == "__main__":
    main()
