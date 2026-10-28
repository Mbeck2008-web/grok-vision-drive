#!/usr/bin/env python3
"""One collector tick, so a missing name fails here instead of on the first live read.

``_read_vehicle`` calls ``controls_from_state``. That name used to exist only
inside ``main``, and Automatic Mode died at the first vehicle poll with
``NameError``. This runs that function, then one pass of the collect loop
against a fake BeamNG session.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gvd_test_env import collect_root  # noqa: E402
import scripts.collect_scene_lanes as collect  # noqa: E402
from python.collect.auto_mode import stop_file  # noqa: E402
from python.debug.gt_geom import RoadPoly  # noqa: E402


class _Vehicle:
    model = "etk800"  # no get_bbox: the collector logs the etk800 width

    def __init__(self, stop: Path) -> None:
        self.stop = stop
        self.commands: list[str] = []
        self.polls = 0
        self.state = {
            "pos": [0.4, 12.0, 0.5],
            "dir": [0.0, 1.0, 0.0],
            "vel": [0.0, 8.0, 0.0],
            "electrics": {
                "steering_input": 0.15,
                "throttle_input": 0.4,
                "brake_input": 0.0,
                "wheelspeed": 8.0,
            },
        }

    def poll_sensors(self) -> None:
        self.polls += 1
        # The loop checks the stop file at the top of the next pass.
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


class _Cam:
    def poll(self) -> dict[str, object]:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[0, 0] = (8, 16, 24)
        return {"colour": image}


def check_read_vehicle() -> None:
    vehicle = _Vehicle(Path(tempfile.mkdtemp()) / "STOP")
    sample = collect._read_vehicle(vehicle)
    print(f"read steer={sample['steer']} speed={sample['speed_mps']} pos={sample['pos']}")
    assert vehicle.polls == 1
    assert sample["steer"] == 0.15
    assert sample["throttle"] == 0.4
    assert sample["brake"] == 0.0
    assert sample["speed_mps"] == 8.0
    assert sample["pos"] == [0.4, 12.0, 0.5]
    assert sample["dir"] == [0.0, 1.0, 0.0]


def check_one_tick() -> None:
    with collect_root() as root:
        out = root / "scene"
        vehicle = _Vehicle(stop_file(out))
        bng = _Bng(vehicle)

        def _open(*args: object, **kwargs: object) -> _Bng:
            del args, kwargs
            return bng

        def _roads(session: object) -> tuple[list[RoadPoly], str]:
            del session
            return [RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)], ""

        def _cams(session: object, veh: object, cfg: object) -> dict[str, _Cam]:
            del session, veh, cfg
            return {"main": _Cam()}

        collect.time.sleep = lambda _seconds: None
        import python.debug.gt_logger as gt_logger
        import python.sensors.tech as tech

        tech.open_tech_beamngpy = _open  # type: ignore[assignment]
        tech.widen_socket_timeout = lambda *args, **kwargs: None  # type: ignore[assignment]
        tech.release_tech_beamngpy = lambda *args, **kwargs: None  # type: ignore[assignment]
        gt_logger.load_road_polys = _roads  # type: ignore[assignment]
        collect._attach_cameras = _cams  # type: ignore[assignment]
        code = collect.main(
            [
                "--minutes",
                "1",
                "--data-dir",
                str(root),
                "--out",
                str(out),
                "--host",
                "127.0.0.1",
                "--port",
                "1",
            ]
        )
        print(f"tick exit={code} polls={vehicle.polls} commands={len(vehicle.commands)}")
        assert code == 0, code
        assert vehicle.polls >= 1
        assert any("ai.setSpeed" in line for line in vehicle.commands)
        state = out / "state.jsonl"
        assert state.is_file(), sorted(p.name for p in out.iterdir())
        row = json.loads(state.read_text(encoding="utf-8").splitlines()[0])
        assert row.get("driver") == "beamng_ai"
        assert (row.get("excursion") or {}).get("phase") == "cruise"
        assert row.get("segment") == "current#0"
        wheel = row.get("wheel") or {}
        assert wheel.get("steer") == 0.15
        pngs = list((out / "raw" / "main").rglob("*.png"))
        assert pngs, sorted(p.name for p in out.rglob("*") if p.is_file())
        print(f"state steer={wheel.get('steer')} png={pngs[0].name}")


class _StreamCam:
    def __init__(self, raw: object, *, poison_poll: bool = False) -> None:
        self.raw = raw
        self.poison_poll = poison_poll
        self.resolution = (8, 8)
        self.polls = 0

    def stream_raw(self) -> object:
        return self.raw

    def poll(self) -> object:
        self.polls += 1
        if self.poison_poll:
            raise AssertionError("shared-memory camera must be read with stream_raw")
        return self.raw


def check_stream_raw_not_poll() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)
    cam = _StreamCam({"colour": image}, poison_poll=True)
    frames, _stamps, reasons = collect._poll_cameras({"main": cam})
    assert cam.polls == 0
    assert "main" in frames and frames["main"].shape == (8, 8, 3)
    assert reasons == {}
    print(f"stream_raw kept main shape={frames['main'].shape}")


def check_drop_reasons() -> None:
    empty = _StreamCam(None)
    zeros = np.zeros((4, 4, 3), dtype=np.uint8)
    blank = _StreamCam({"colour": zeros})
    frames, _stamps, reasons = collect._poll_cameras({"main": empty, "wide": blank})
    assert frames == {}
    assert "returned None" in reasons["main"]
    assert "unrendered" in reasons["wide"]
    print(f"drop reasons={reasons}")


def check_all_cameras_dead() -> None:
    reasons = {"main": "stream_raw returned None", "wide": "stream_raw unrendered (all zeros)"}
    since, message = collect.camera_outage(None, 0.0, False, reasons)
    assert since == 0.0 and message is None
    since, message = collect.camera_outage(since, 9.0, False, reasons)
    assert message is None
    _since, message = collect.camera_outage(since, 10.1, False, reasons)
    assert message is not None and "main=" in message and "wide=" in message
    cleared, gone = collect.camera_outage(since, 11.0, True, {})
    assert cleared is None and gone is None
    print(f"outage={message}")


def main() -> None:
    check_read_vehicle()
    check_stream_raw_not_poll()
    check_drop_reasons()
    check_all_cameras_dead()
    check_one_tick()
    print("collect tick ok")


if __name__ == "__main__":
    main()
