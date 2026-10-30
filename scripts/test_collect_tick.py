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


PROBE_REPLY = {
    "src": ["ai"],
    "ai": {"steering": 0.12, "throttle": 0.3, "brake": 0.0, "parkingbrake": 0.0, "clutch": 0.0},
    "g": {"x": 1.1, "y": 0.0, "z": 9.81},
    "g2": {"x": 0.9, "y": 0.0, "z": 9.8},
    "w": {"yaw": 0.02, "pitch": 0.0, "roll": 0.0},
}
TRAFFIC_CAR = {"name": "gvd_traffic_0", "pos": [0.4, 40.0, 0.5], "vel": [0.0, 9.0, 0.0], "dir": [0.0, 1.0, 0.0]}
# Old row keys, in their order before the control fields: these stay byte-identical.
OLD_EGO = ["speed_mps", "yaw_rate", "damage", "pos", "dir", "up", "x", "y"]
OLD_CAPTURE = ["t", "dt_s", "steer", "throttle", "brake", "speed_mps", "yaw_rate", "pos", "dir", "driver", "gt"]
OLD_TOP = ["i", "t", "dt_s", "driver", "ego", "wheel", "pedals", "capture", "cams", "segment", "map", "excursion"]
NEW_TOP = ["veh", "imu", "ai_in_src", "ai_cmd", "road", "traffic"]


class _Vehicle:
    model = "etk800"  # no get_bbox: the collector logs the etk800 width

    def __init__(self, stop: Path, stop_after: int = 1) -> None:
        self.stop = stop
        self.stop_after = int(stop_after)
        self.commands: list[str] = []
        self.probes = 0
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
                "steering": 4.5,
                "throttle": 0.31,
                "brake": 0.0,
                "parkingbrake_input": 0.0,
                "clutch_input": 0.0,
                "gearIndex": 2,
                "rpm": 1500.0,
                "fuel_volume": 40.0,
            },
        }

    def poll_sensors(self) -> None:
        self.polls += 1
        # The loop checks the stop file at the top of the next pass.
        if self.polls >= self.stop_after:
            self.stop.write_text("stop", encoding="utf-8")

    def queue_lua_command(self, command: str, *args: object) -> object:
        self.commands.append(str(command))
        if not (args and args[0]):
            return None
        if "lastInputs" in command:  # the control probe (response=True): a jsonEncode string
            self.probes += 1
            return json.dumps(PROBE_REPLY)
        if "steeringWheelLock" in command:
            return json.dumps({"lock": 450.0, "wb": 2.91})
        return None

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
        self.traffic_calls = 0

    def queue_lua_command(self, command: str, *args: object) -> object:
        if "gvd_traffic_" in command and args and args[0]:
            self.traffic_calls += 1
            return json.dumps([TRAFFIC_CAR])
        return None


class _Game:
    """BeamNG as BeamNGpy 1.36 sees it, for counting round-trips.

    ``Connection.send`` only writes a request; ``Response.recv`` blocks until the
    reply arrives, and replies are matched by request id (out-of-order replies are
    buffered). The game answers on its next frame, and one frame answers every
    request already sent on any socket (vehicle and engine). A ``recv`` whose reply
    is not there yet waits for that frame: one wait is one round-trip.
    """

    def __init__(self) -> None:
        self.pending: list[tuple["_Conn", int, dict]] = []
        self.frames: list[list[dict]] = []  # the requests each wait answered

    def wait(self) -> None:
        batch, self.pending = self.pending, []
        for conn, rid, data in batch:
            conn.replies[rid] = conn.answer(data)
        self.frames.append([data for _conn, _rid, data in batch])


class _Conn:
    def __init__(self, game: _Game, answer) -> None:
        self.game, self.answer = game, answer
        self.req_id = 0
        self.replies: dict[int, dict] = {}

    def send(self, data: dict) -> "_Resp":
        self.req_id += 1
        self.game.pending.append((self, self.req_id, dict(data)))
        return _Resp(self, self.req_id)

    def recv(self, rid: int) -> dict:
        if rid not in self.replies:
            self.game.wait()
        return self.replies.pop(rid)


class _Resp:
    def __init__(self, conn: _Conn, rid: int) -> None:
        self.conn, self.rid = conn, rid

    def recv(self, type: str | None = None) -> dict:  # noqa: A002 - BeamNGpy's name
        msg = self.conn.recv(self.rid)
        if isinstance(msg, Exception):  # BeamNGpy turns a bngError reply into a raise
            raise msg
        if type and msg["type"] != type:
            raise ValueError(f"got {msg['type']} expected {type}")
        return msg


class _WireVehicle(_Vehicle):
    """``_Vehicle`` behind a fake BeamNGpy vehicle socket: every call is a request."""

    def __init__(self, stop: Path, game: _Game, stop_after: int = 1, fail_lua: bool = False) -> None:
        super().__init__(stop, stop_after)
        self.connection = _Conn(game, self._answer)
        self.fail_lua = fail_lua

    def _answer(self, data: dict) -> object:
        if self.fail_lua and "lastInputs" in str(data.get("chunk")):
            self.probes += 1
            return RuntimeError("bngError: lua")
        if data["type"] == "SensorRequest":
            _Vehicle.poll_sensors(self)
            return {"type": "SensorData", "data": {}}
        if data["type"] == "QueueLuaCommandVE":
            resp = _Vehicle.queue_lua_command(self, data["chunk"], data["resp"])
            return {"type": "ExecutedLuaChunkVE", "resp": resp}
        raise AssertionError(data)

    def _send(self, data: dict) -> _Resp:
        return self.connection.send(data)

    def poll_sensors(self) -> None:  # Sensors.poll: State, Electrics, Damage in one SensorRequest
        self._send({"type": "SensorRequest", "sensors": {"state": {"type": "State"}}}).recv("SensorData")

    def queue_lua_command(self, command: str, response: bool = False) -> object:
        data = {"type": "QueueLuaCommandVE", "chunk": command, "resp": response}
        return self._send(data).recv("ExecutedLuaChunkVE").get("resp", None)


class _WireBng(_Bng):
    """``_Bng`` behind a fake BeamNGpy engine socket on the same game."""

    def __init__(self, vehicle: _Vehicle, game: _Game) -> None:
        super().__init__(vehicle)
        self.connection = _Conn(game, self._answer)

    fail_lua = False

    def _answer(self, data: dict) -> object:
        assert data["type"] == "QueueLuaCommandGE", data
        if self.fail_lua and "gvd_traffic_" in str(data.get("chunk")):
            self.traffic_calls += 1
            return RuntimeError("bngError: lua")
        return {"type": "ExecutedLuaChunkGE", "resp": _Bng.queue_lua_command(self, data["chunk"], data["resp"])}

    def _send(self, data: dict) -> _Resp:
        return self.connection.send(data)

    def queue_lua_command(self, command: str, response: bool = False) -> object:
        data = {"type": "QueueLuaCommandGE", "chunk": command, "resp": response}
        return self._send(data).recv("ExecutedLuaChunkGE").get("resp", None)


def _kind(data: dict) -> str:
    chunk = str(data.get("chunk") or "")
    if data["type"] == "SensorRequest":
        return "poll"
    if "lastInputs" in chunk:
        return "probe"
    if "gvd_traffic_" in chunk:
        return "traffic"
    return "other"


def check_probe_round_trips() -> None:
    """The control probe and the traffic read ride the sensor poll's round-trip.

    Per tick the collector polls the sensors once (one round-trip). The probe
    and the 2 Hz traffic read must not add a wait of their own: each is
    answered in the same frame as that tick's poll. Every recorded field stays
    what the separate calls gave.
    """
    for every, fail in ((1, False), (2, False), (1, True)):
        with collect_root() as root:
            out = root / "scene"
            game = _Game()
            vehicle = _WireVehicle(stop_file(out), game, stop_after=4, fail_lua=fail)
            bng = _WireBng(vehicle, game)
            bng.fail_lua = fail
            code = _collect(root, out, bng, ["--ctl-probe-every", str(every)])
            assert code == 0, code
            lines = (out / "state.jsonl").read_text(encoding="utf-8").splitlines()
            rows = [json.loads(x) for x in lines]
            assert len(rows) >= 3, len(rows)
            for k, (row, line) in enumerate(zip(rows, lines)):
                _check_control_row(row, line, probed=not fail and k % every == 0)
            if fail:  # a failed probe or traffic read is null, never 0, and the row is still written
                assert all(r["traffic"] is None for r in rows), [r["traffic"] for r in rows]
            else:
                assert rows[0]["traffic"] == [TRAFFIC_CAR]
            assert all(r["traffic"] is None for r in rows[1:])
            assert rows[0]["veh_meta"] == {"model": "etk800", "steer_lock_deg": 450.0, "wheelbase_m": 2.91}
            kinds = [[_kind(d) for d in frame] for frame in game.frames]
            ticks = sum(k.count("poll") for k in kinds)
            probes = sum(k.count("probe") for k in kinds)
            traffic = sum(k.count("traffic") for k in kinds)
            counts = (ticks, vehicle.polls, probes, vehicle.probes, traffic, bng.traffic_calls)
            assert ticks == vehicle.polls and probes == vehicle.probes and traffic == bng.traffic_calls == 1, counts
            assert probes == (ticks + every - 1) // every, (probes, ticks)
            tick_frames = [k for k in kinds if {"poll", "probe", "traffic"} & set(k)]
            own_waits = sum(1 for k in tick_frames if "poll" not in k)
            print(f"round-trips every={every} fail={fail} ticks={ticks} probes={probes} traffic={traffic} "
                  f"tick_waits={len(tick_frames)} probe_or_traffic_own_waits={own_waits}")
            # One wait per tick: the sensor poll. The probe adds 0 round-trips.
            assert own_waits == 0, kinds
            assert len(tick_frames) == ticks, kinds


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
        code = _collect(root, out, bng, [])
        print(f"tick exit={code} polls={vehicle.polls} commands={len(vehicle.commands)}")
        assert code == 0, code
        assert vehicle.polls >= 1
        assert any("ai.setSpeed" in line for line in vehicle.commands)
        state = out / "state.jsonl"
        assert state.is_file(), sorted(p.name for p in out.iterdir())
        line = state.read_text(encoding="utf-8").splitlines()[0]
        row = json.loads(line)
        assert row.get("driver") == "beamng_ai"
        assert (row.get("excursion") or {}).get("phase") == "cruise"
        assert row.get("segment") == "current#0"
        wheel = row.get("wheel") or {}
        assert wheel.get("steer") == 0.15
        pngs = list((out / "raw" / "main").rglob("*.png"))
        assert pngs, sorted(p.name for p in out.rglob("*") if p.is_file())
        print(f"state steer={wheel.get('steer')} png={pngs[0].name}")
        _check_control_row(row, line, probed=True)
        assert row["traffic"] == [TRAFFIC_CAR], row["traffic"]
        assert row["veh_meta"] == {"model": "etk800", "steer_lock_deg": 450.0, "wheelbase_m": 2.91}, row["veh_meta"]
        print(f"control row ok steer_ai={wheel['steer_ai']} deg={wheel['steer_angle_deg']} traffic=1")


def _check_control_row(row: dict, line: str, *, probed: bool) -> None:
    """New keys hold the fake's values; old keys keep their values and order."""
    wheel, pedals, ego = row["wheel"], row["pedals"], row["ego"]
    # Old keys first, unchanged: the row text up to the first new key is what it was.
    assert list(wheel)[:1] == ["steer"] and wheel["steer"] == 0.15, wheel
    assert list(pedals)[:2] == ["throttle", "brake"] and (pedals["throttle"], pedals["brake"]) == (0.4, 0.0)
    assert list(ego)[: len(OLD_EGO)] == OLD_EGO, list(ego)
    assert list(row["capture"]) == OLD_CAPTURE, list(row["capture"])
    assert [k for k in row if k in OLD_TOP] == OLD_TOP[: len([k for k in row if k in OLD_TOP])], list(row)
    assert list(row).index("excursion") < min(list(row).index(k) for k in NEW_TOP), list(row)
    assert line.startswith('{"i":') and '"wheel":{"steer":0.15,"steer_angle_deg":4.5' in line, line[:400]
    assert '"pedals":{"throttle":0.4,"brake":0.0,' in line, line[:600]
    # Tick-rate fields (no probe needed).
    assert wheel["steer_angle_deg"] == 4.5
    assert pedals["throttle_applied"] == 0.31 and pedals["brake_applied"] == 0.0
    assert pedals["parkingbrake"] == 0.0 and pedals["clutch"] == 0.0
    assert ego["vel"] == [0.0, 8.0, 0.0] and "sim_t" in ego
    assert row["veh"] == {"gear_index": 2, "rpm": 1500.0, "wheelspeed": 8.0}, row["veh"]
    assert row["ai_cmd"]["mode"] == "span" and row["ai_cmd"]["speed_mode"] == "limit", row["ai_cmd"]
    assert row["ai_cmd"]["line_active"] is False and row["ai_cmd"]["set_speed_mps"] > 0
    assert row["road"]["road_id"] == "r" and row["road"]["lanes_here"] >= 1, row["road"]
    assert "junction" not in row["road"]  # not computed for gt today: omitted, not invented
    imu = row["imu"]
    if probed:
        assert wheel["steer_ai"] == 0.12 and pedals["throttle_ai"] == 0.3 and pedals["brake_ai"] == 0.0
        assert pedals["parkingbrake_ai"] == 0.0 and pedals["clutch_ai"] == 0.0
        assert imu["gx"] == 1.1 and imu["g_smoothed"]["x"] == 0.9 and imu["gx"] != imu["g_smoothed"]["x"]
        assert imu["yaw_rate"] == 0.02 and row["ai_in_src"] == ["ai"]
    else:
        assert wheel["steer_ai"] is None and pedals["throttle_ai"] is None and pedals["brake_ai"] is None
        assert imu["gx"] is None and imu["g_smoothed"] == {"x": None, "y": None, "z": None}
        assert imu["yaw_rate"] is None and row["ai_in_src"] is None


def check_probe_every_2_and_traffic_rate() -> None:
    with collect_root() as root:
        out = root / "scene"
        vehicle = _Vehicle(stop_file(out), stop_after=4)
        bng = _Bng(vehicle)
        code = _collect(root, out, bng, ["--ctl-probe-every", "2"])
        assert code == 0, code
        lines = (out / "state.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(x) for x in lines]
        assert len(rows) >= 3, len(rows)
        for k, (row, line) in enumerate(zip(rows, lines)):
            _check_control_row(row, line, probed=k % 2 == 0)
        # The fake ticks are ms apart: traffic on the first row only (2 Hz).
        assert rows[0]["traffic"] == [TRAFFIC_CAR] and all(r["traffic"] is None for r in rows[1:]), [
            r["traffic"] for r in rows
        ]
        assert bng.traffic_calls == 1 and vehicle.probes == (len(rows) + 1) // 2, (bng.traffic_calls, vehicle.probes)
        assert "veh_meta" in rows[0] and all("veh_meta" not in r for r in rows[1:])
        print(f"probe every 2 ok rows={len(rows)} probes={vehicle.probes} traffic_calls={bng.traffic_calls}")


def _collect(root: Path, out: Path, bng: "_Bng", extra: list[str]) -> int:
    """One run of ``collect.main`` against the fake session (``--minutes 1`` plus ``extra``)."""
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
    return collect.main(
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
        + list(extra)
    )


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
    check_probe_every_2_and_traffic_rate()
    check_probe_round_trips()
    print("collect tick ok")


if __name__ == "__main__":
    main()
