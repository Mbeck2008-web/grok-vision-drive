#!/usr/bin/env python3
"""Recorder-only labels v4: main+wide train_gt, per-tick signals, traffic electrics.

No trainer and no head. A fake beamngpy stands in for Tech. LiDAR and radar stay off.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from python.debug.gt_geom import RoadPoly  # noqa: E402
from python.sensors.cameras import CAM_IDS, load_camera_config  # noqa: E402

PAYLOAD = {
    "node_a": {
        "node_b": [
            {
                "instance": "trafficLight 1",
                "pos": {"x": 2.0, "y": 10.0, "z": 0.0},
                "state": "red",
            }
        ]
    }
}
TRAFFIC = [
    {"name": "gvd_traffic_0", "pos": [1.0, 40.0, 0.5], "vel": [0.0, 8.0, 0.0], "dir": [0.0, 1.0, 0.0]},
    {"name": "parked_bus", "pos": [-3.0, 15.0, 0.5], "vel": [0.0, 0.0, 0.0], "dir": [1.0, 0.0, 0.0]},
]
# Ego lamp is 1. Traffic lamps differ so a copied ego lamp fails the check.
BRAKES = {"ego": 1, "gvd_traffic_0": 0, "parked_bus": 1}
PAINT_KEYS = {
    "lanes",
    "lane_gt",
    "lane_gt_valid",
    "gt_lookahead_m",
    "heading_err_rad",
    "off_road",
    "gt_source",
    "gt_origin",
    "gt_ref_right_of_origin_m",
}
IQU45_SMOKE = (
    "py -3 scripts\\collect_scene_lanes.py --minutes 1 --no-excursions "
    "--traffic 2 --smoke-s 15 --ctl-probe-every 1"
)
RGB_ONLY = ("narrow", "pillarL", "pillarR", "repeatL", "repeatR", "rear")


def check_train_gt_main_and_wide_only() -> None:
    cfg = load_camera_config()
    by_id = {str(cam["id"]): cam for cam in cfg["cameras"]}
    assert set(by_id) == set(CAM_IDS)
    assert by_id["main"].get("train_gt") is True
    assert by_id["wide"].get("train_gt") is True
    for cid in RGB_ONLY:
        assert not by_id[cid].get("train_gt"), cid
    print("train_gt true on main and wide only")


def check_lidar_and_radar_stay_off() -> None:
    import os

    from python.sensors.extras import load_sensors_config

    for key in ("GVD_LIDAR", "GVD_RADAR", "GVD_LIDAR_LUA"):
        os.environ.pop(key, None)
    cfg = load_sensors_config()
    assert cfg.get("lidar") is False and cfg.get("radar") is False and cfg.get("lidar_lua") is False
    src = (ROOT / "scripts" / "collect_scene_lanes.py").read_text(encoding="utf-8")
    assert "Lidar(" not in src and "Radar(" not in src
    print("lidar and radar stay false")


def check_traffic_chunk_walks_non_player_objects() -> None:
    from python.collect import control_log as cl

    chunk = cl.TRAFFIC_CHUNK
    assert "getObjectCount" in chunk and "getPlayerVehicle" in chunk
    assert "getPosition" in chunk and "getVelocity" in chunk and "getDirectionVector" in chunk
    assert "name" in chunk and "jsonEncode" in chunk
    assert "gvd_traffic_0" not in chunk and "gvd_traffic_1" not in chunk
    for banned in ("truck", "bus", "pedestrian", "bike"):
        assert banned not in chunk, banned
    assert cl.TRAFFIC_EVERY_S == 0.5
    print("traffic chunk reads every non-player object at 2 Hz")


def check_missing_signal_api_is_err_row() -> None:
    from python.collect.recorder_v4 import append_signal_tick, signal_tick_row

    row = signal_tick_row(1.5, None, "missing traffic_signals")
    assert row == {"t": 1.5, "err": "missing traffic_signals"}
    assert "signals" not in row and "red" not in json.dumps(row)

    class _API:
        @staticmethod
        def get_map_node_signals():
            return PAYLOAD

    class _Bng:
        traffic_signals = _API()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "signals.jsonl"
        append_signal_tick(path, object(), 1.25)
        append_signal_tick(path, _Bng(), 2.5)
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["t"] == 1.25 and "err" in lines[0] and "signals" not in lines[0]
    assert "red" not in json.dumps(lines[0])
    assert lines[1]["t"] == 2.5 and lines[1]["signals"] == PAYLOAD and "err" not in lines[1]
    assert "signs" not in lines[1]
    print("missing signal API writes {t, err} and keeps going")


class _Sensors(dict):
    def attach(self, name, sensor):
        self[name] = sensor
        return sensor


class _Electrics:
    def __init__(self):
        self.data: dict = {}


def _install_sensors(mod: types.ModuleType) -> None:
    mod.Electrics = _Electrics
    mod.Damage = type("Damage", (), {})
    mod.Camera = None  # filled by the collect fake


def check_brake_lights_come_from_each_vehicle() -> None:
    from python.collect.recorder_v4 import with_brake_lights

    class _Veh:
        def __init__(self, name: str, lamp: int) -> None:
            self.name = name
            self.sensors = _Sensors()
            self.lamp = lamp
            self.polls = 0

        def poll_sensors(self) -> None:
            self.polls += 1
            el = self.sensors.get("electrics")
            if el is not None:
                # One car speaks the Lua alias. The row key stays brake_lights.
                if self.name == "parked_bus":
                    el.data["brakelights"] = self.lamp
                else:
                    el.data["brake_lights"] = self.lamp

    ego = _Veh("ego", BRAKES["ego"])
    lead = _Veh("gvd_traffic_0", BRAKES["gvd_traffic_0"])
    parked = _Veh("parked_bus", BRAKES["parked_bus"])

    class _Api:
        def __init__(self) -> None:
            self._all = {"ego": ego, "gvd_traffic_0": lead, "parked_bus": parked}
            self.kwargs: dict = {}

        def get_current(self, **kwargs):
            self.kwargs = dict(kwargs)
            return dict(self._all)

        def get_player_vehicle_id(self):
            return {"vid": "ego"}

    api = _Api()

    class _Bng:
        vehicles = api

    sensors = types.ModuleType("beamngpy.sensors")
    _install_sensors(sensors)
    beamngpy = types.ModuleType("beamngpy")
    beamngpy.sensors = sensors
    old = {key: sys.modules.get(key) for key in ("beamngpy", "beamngpy.sensors")}
    sys.modules["beamngpy"] = beamngpy
    sys.modules["beamngpy.sensors"] = sensors
    try:
        got = with_brake_lights(TRAFFIC + [
            {"name": "loose_cone", "pos": [0.0, 9.0, 0.2], "vel": [0.0, 0.0, 0.0], "dir": [0.0, 1.0, 0.0]},
        ], _Bng())
    finally:
        for key, value in old.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    by = {row["name"]: row for row in got}
    assert set(by) == {"gvd_traffic_0", "parked_bus", "loose_cone"}
    assert by["gvd_traffic_0"]["brake_lights"] == 0
    assert by["parked_bus"]["brake_lights"] == 1
    assert by["loose_cone"]["brake_lights"] is None
    assert by["gvd_traffic_0"]["pos"] == [1.0, 40.0, 0.5]
    assert by["parked_bus"]["vel"] == [0.0, 0.0, 0.0]
    assert by["gvd_traffic_0"]["dir"] == [0.0, 1.0, 0.0]
    assert "electrics" in lead.sensors and "electrics" in parked.sensors
    assert lead.polls == 1 and parked.polls == 1
    assert "electrics" not in ego.sensors  # the player attach is the collector's, not this merge
    assert api.kwargs == {"include_config": False}
    print("brake_lights stay on the vehicle that reported them")


def check_electrics_attach_failure_is_masked() -> None:
    """A scenetree car with no beamngpy handle does not crash the collect."""
    import contextlib

    import python.collect.recorder_v4 as rec

    rec._ELECTRICS_LOGGED.clear()

    class _BoomSensors:
        def __contains__(self, key: object) -> bool:
            return False

        def attach(self, name: str, sensor: object) -> None:
            raise RuntimeError("not a Vehicle")

    class _Boom:
        def __init__(self) -> None:
            self.name = "gvd_traffic_0"
            self.sensors = _BoomSensors()

    class _Api:
        def get_current(self, **_kwargs):
            return {"ego": object(), "gvd_traffic_0": _Boom()}

        def get_player_vehicle_id(self):
            return {"vid": "ego"}

    class _Bng:
        vehicles = _Api()

    sensors = types.ModuleType("beamngpy.sensors")
    _install_sensors(sensors)
    beamngpy = types.ModuleType("beamngpy")
    beamngpy.sensors = sensors
    old = {key: sys.modules.get(key) for key in ("beamngpy", "beamngpy.sensors")}
    sys.modules["beamngpy"] = beamngpy
    sys.modules["beamngpy.sensors"] = sensors
    cars = [dict(row) for row in TRAFFIC]
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            first = rec.with_brake_lights(cars, _Bng())
            second = rec.with_brake_lights(cars, _Bng())
        class _BadApi:
            def get_current(self, **_kwargs):
                raise RuntimeError("vehicles down")

        class _Bad:
            vehicles = _BadApi()

        with contextlib.redirect_stdout(io.StringIO()):
            crashed = rec.with_brake_lights(cars, _Bad())
    finally:
        for key, value in old.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    assert [row["brake_lights"] for row in first] == [None, None]
    assert [row["brake_lights"] for row in second] == [None, None]
    assert [row["brake_lights"] for row in crashed] == [None, None]
    assert first[0]["pos"] == TRAFFIC[0]["pos"] and first[1]["name"] == "parked_bus"
    text = buf.getvalue()
    assert text.count("Electrics attach failed") == 1, text
    assert text.count("no connected beamngpy vehicle") == 1, text
    assert "brake_lights null" in text
    assert "connect each traffic vehicle via beamngpy" in text
    print("electrics attach failure masks brake_lights and logs once")


def check_electrics_poll_failure_is_masked() -> None:
    """poll_sensors raising stays null, logs once, and does not escape."""
    import contextlib

    import python.collect.recorder_v4 as rec

    rec._ELECTRICS_LOGGED.clear()

    class _Veh:
        def __init__(self) -> None:
            self.name = "gvd_traffic_0"
            self.sensors = _Sensors()

        def poll_sensors(self) -> None:
            raise RuntimeError("electrics socket dropped")

    class _Api:
        def get_current(self, **_kwargs):
            return {"ego": object(), "gvd_traffic_0": _Veh()}

        def get_player_vehicle_id(self):
            return {"vid": "ego"}

    class _Bng:
        vehicles = _Api()

    sensors = types.ModuleType("beamngpy.sensors")
    _install_sensors(sensors)
    beamngpy = types.ModuleType("beamngpy")
    beamngpy.sensors = sensors
    old = {key: sys.modules.get(key) for key in ("beamngpy", "beamngpy.sensors")}
    sys.modules["beamngpy"] = beamngpy
    sys.modules["beamngpy.sensors"] = sensors
    cars = [{
        "name": "gvd_traffic_0",
        "pos": [1.0, 2.0, 0.0],
        "vel": [0.0, 1.0, 0.0],
        "dir": [0.0, 1.0, 0.0],
    }]
    try:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                first = rec.with_brake_lights(cars, _Bng())
                second = rec.with_brake_lights(cars, _Bng())
        except Exception as exc:  # noqa: BLE001 — the assertion is the contract
            raise AssertionError(f"poll_sensors failure must not escape: {exc}") from exc
    finally:
        for key, value in old.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    assert [row["brake_lights"] for row in first] == [None]
    assert [row["brake_lights"] for row in second] == [None]
    assert first[0]["pos"] == [1.0, 2.0, 0.0] and first[0]["name"] == "gvd_traffic_0"
    hits = [line for line in buf.getvalue().splitlines() if "collect traffic electrics:" in line]
    assert hits == [
        "collect traffic electrics: electrics poll failed: RuntimeError: "
        "electrics socket dropped for gvd_traffic_0; brake_lights null."
    ], hits
    print("electrics poll failure masks brake_lights and logs once")


def check_iqu45_smoke_is_documented() -> None:
    import scripts.collect_scene_lanes as collect

    text = collect.__doc__ or ""
    assert "IQU45" in text
    assert IQU45_SMOKE in text
    assert "signals.jsonl" in text and "brake_lights" in text
    assert "scenetree" in text
    assert "connect each traffic vehicle via beamngpy" in text
    print("IQU45 smoke command is in the collector docstring")


def _bbox(width: float = 1.9) -> dict:
    x, y, z, h = 0.4, 12.0, 0.5, 0.5 * width
    box = {}
    for end, dy in (("front", 2.3), ("rear", -2.3)):
        for side, dx in (("left", -h), ("right", h)):
            box[f"{end}_bottom_{side}"] = (x + dx, y + dy, z - 0.5)
            box[f"{end}_top_{side}"] = (x + dx, y + dy, z + 1.0)
    return box


class _Vehicle:
    def __init__(self, name: str, stop: Path | None, stop_at: int) -> None:
        self.name = name
        self.stop = stop
        self.stop_at = stop_at
        self.polls = 0
        self.sensors = _Sensors()
        self.commands: list[str] = []
        self.state = {
            "pos": [0.4, 12.0, 0.5],
            "dir": [0.0, 1.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "vel": [0.0, 10.0, 0.0],
            "electrics": {"wheelspeed": 10.0, "steering_input": 0.0, "brake_lights": BRAKES[name]},
        }

    def get_bbox(self) -> dict:
        return _bbox()

    def poll_sensors(self) -> None:
        self.polls += 1
        el = self.sensors.get("electrics")
        if el is not None:
            if self.name == "parked_bus":
                el.data["brakelights"] = BRAKES[self.name]
            else:
                el.data["brake_lights"] = BRAKES[self.name]
            el.data["wheelspeed"] = 10.0
        if self.stop is not None and self.polls >= self.stop_at:
            self.stop.write_text("stop", encoding="utf-8")

    def queue_lua_command(self, command: str, *args: object):
        del args
        self.commands.append(str(command))
        return None

    def connect(self, bng: object) -> None:
        del bng

    def is_connected(self) -> bool:
        return True


class _Signals:
    def __init__(self) -> None:
        self.calls = 0

    def get_map_node_signals(self):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("signals down")
        return PAYLOAD


class _Anno:
    @staticmethod
    def get_annotations():
        return {"ROAD": (128, 64, 128), "SKY": (70, 130, 180)}


class _Bng:
    def __init__(self, ego: _Vehicle, lead: _Vehicle, parked: _Vehicle) -> None:
        self.ego = ego
        self.lead = lead
        self.parked = parked
        self.commands: list[str] = []
        self.traffic_signals = _Signals()
        self.camera = _Anno()

        class _Api:
            def get_current(_self, **_kwargs):
                return {"ego": ego, "gvd_traffic_0": lead, "parked_bus": parked}

            def get_player_vehicle_id(_self):
                return {"vid": "ego"}

        self.vehicles = _Api()

    def queue_lua_command(self, command: str, response: bool = False):
        self.commands.append(str(command))
        if response and "getObjectCount" in str(command):
            return json.dumps(TRAFFIC)
        return None


def _fake_modules():
    made: dict[str, dict] = {}

    class _Cam:
        def __init__(self, name, bng, vehicle, **kwargs):
            del bng, vehicle
            self.name = name
            self.kwargs = kwargs
            made[name] = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)
            if "near_far_planes" not in kwargs:
                self.near_far_planes = (0.05, 300.0)

        def stream_raw(self):
            res = self.resolution
            w, h = int(res[0]), int(res[1])
            colour = np.zeros((h, w, 3), dtype=np.uint8)
            colour[0, 0] = (8, 16, 24)
            raw = {"colour": colour}
            if self.is_render_depth:
                raw["depth"] = np.full((h, w), 0.1, dtype=np.float32)
            if self.is_render_annotations:
                ann = np.zeros((h, w, 4), dtype=np.uint8)
                ann[..., :3] = (128, 64, 128)
                raw["annotation"] = ann
            return raw

        def remove(self):
            return None

    sensors = types.ModuleType("beamngpy.sensors")
    sensors.Camera = _Cam
    sensors.Electrics = _Electrics
    sensors.Damage = type("Damage", (), {})
    beamngpy = types.ModuleType("beamngpy")
    beamngpy.sensors = sensors
    return beamngpy, sensors, made


def check_short_collect_writes_recorder_files() -> None:
    """Fake beamngpy: depth/sem on main and wide, signals per tick, paint-only gt."""
    import python.data.raw_queue as raw_queue
    import python.debug.gt_logger as gt_logger
    import python.sensors.tech as tech
    import scripts.collect_scene_lanes as collect
    from python.collect.auto_mode import stop_file
    from scripts.gvd_test_env import collect_root

    beamngpy, sensors, made = _fake_modules()
    old_mods = {key: sys.modules.get(key) for key in ("beamngpy", "beamngpy.sensors")}
    sys.modules["beamngpy"] = beamngpy
    sys.modules["beamngpy.sensors"] = sensors

    base_queue = raw_queue.RawWriteQueue

    class _DeepQueue(base_queue):
        def __init__(self, writer, max_depth: int = 64, **kwargs) -> None:
            del max_depth
            super().__init__(writer, 100000, **kwargs)

    previous = (
        tech.open_tech_beamngpy,
        tech.widen_socket_timeout,
        tech.release_tech_beamngpy,
        gt_logger.load_road_polys,
        collect.time.sleep,
        collect.time.monotonic,
        collect.tick_sleep_s,
    )
    clock = {"t": 0.0}
    real_sleep = collect.time.sleep
    main_thread = threading.main_thread()

    def sleep(seconds: float) -> None:
        if threading.current_thread() is not main_thread or sys._getframe(1).f_globals.get("__name__") == "subprocess":
            real_sleep(min(float(seconds or 0.0), 0.05))
            return
        clock["t"] += float(seconds or 0.0)

    root_cm = collect_root()
    root = root_cm.__enter__()
    out = root / "scene"
    stop = stop_file(out)
    ego = _Vehicle("ego", stop, 12)
    lead = _Vehicle("gvd_traffic_0", None, 10**9)
    parked = _Vehicle("parked_bus", None, 10**9)
    bng = _Bng(ego, lead, parked)
    tech.open_tech_beamngpy = lambda *a, **k: bng  # type: ignore[assignment]
    tech.widen_socket_timeout = lambda *a, **k: None  # type: ignore[assignment]
    tech.release_tech_beamngpy = lambda *a, **k: None  # type: ignore[assignment]
    gt_logger.load_road_polys = lambda session: (  # type: ignore[assignment]
        [RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)],
        "",
    )
    collect.time.sleep = sleep  # type: ignore[method-assign]
    collect.time.monotonic = lambda: clock["t"]  # type: ignore[method-assign]
    collect.tick_sleep_s = lambda _elapsed, _period=0.1: 0.1  # type: ignore[method-assign]
    raw_queue.RawWriteQueue = _DeepQueue  # type: ignore[misc]
    buf = io.StringIO()
    old_out = sys.stdout
    sys.stdout = buf
    code = 1
    try:
        code = collect.main(
            ["--minutes", "5", "--smoke-s", "0", "--no-excursions", "--data-dir", str(root),
             "--out", str(out), "--host", "127.0.0.1", "--port", "1"]
        )
        text = buf.getvalue()
        assert code == 0, text[-2000:]
        _assert_short_collect(out, made, bng, lead, parked)
    finally:
        sys.stdout = old_out
        (
            tech.open_tech_beamngpy,
            tech.widen_socket_timeout,
            tech.release_tech_beamngpy,
            gt_logger.load_road_polys,
            collect.time.sleep,
            collect.time.monotonic,
            collect.tick_sleep_s,
        ) = previous
        raw_queue.RawWriteQueue = base_queue  # type: ignore[misc]
        for key, value in old_mods.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        root_cm.__exit__(None, None, None)


def _assert_short_collect(out: Path, made: dict, bng: _Bng, lead: _Vehicle, parked: _Vehicle) -> None:
    for cid, kw in made.items():
        gt = cid in ("gvd_collect_main", "gvd_collect_wide")
        assert kw["is_render_depth"] is gt, cid
        assert kw["is_render_annotations"] is gt, cid
        assert kw["is_render_colours"] is True
    assert set(made) == {f"gvd_collect_{cid}" for cid in CAM_IDS}

    raw = out / "raw"
    names = {p.name for p in raw.iterdir() if p.is_dir()}
    for cid in ("main", "wide"):
        assert cid in names
        assert f"{cid}_depth" in names
        assert f"{cid}_sem" in names
        assert any((raw / cid).rglob("*.png"))
        assert any((raw / f"{cid}_depth").rglob("*.png"))
        assert any((raw / f"{cid}_sem").rglob("*.png"))
    for cid in RGB_ONLY:
        assert cid in names, cid
        assert f"{cid}_depth" not in names, cid
        assert f"{cid}_sem" not in names, cid
        assert any((raw / cid).rglob("*.png")), cid

    classes = json.loads((out / "sem_classes.json").read_text(encoding="utf-8"))
    assert classes["kind"] == "gvd_sem_classes" and classes["train_only"] is True
    assert "ROAD" in classes["classes"].values()

    state = [json.loads(line) for line in (out / "state.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    signals = [json.loads(line) for line in (out / "signals.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert state and len(signals) == len(state)
    assert [row["t"] for row in signals] == [row["t"] for row in state]
    assert "err" in signals[0] and "signals" not in signals[0]
    assert "red" not in json.dumps(signals[0])
    assert all(row.get("signals") == PAYLOAD and "err" not in row and "signs" not in row for row in signals[1:])
    assert bng.traffic_signals.calls == len(state)

    filled = [row for row in state if row.get("traffic")]
    blank = [row for row in state if row.get("traffic") is None]
    assert filled and blank and len(filled) < len(state)
    for row in filled:
        by = {car["name"]: car for car in row["traffic"]}
        assert set(by) == {"gvd_traffic_0", "parked_bus"}
        assert by["gvd_traffic_0"]["brake_lights"] == 0
        assert by["parked_bus"]["brake_lights"] == 1
        for car in by.values():
            assert len(car["pos"]) == 3 and len(car["vel"]) == 3 and len(car["dir"]) == 3
    assert any("getObjectCount" in cmd for cmd in bng.commands)
    assert "electrics" in lead.sensors and "electrics" in parked.sensors

    gt_rows = [json.loads(line) for line in (out / "gt.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(gt_rows) == len(state)
    for row in gt_rows:
        tech_row = row["tech"]
        assert set(tech_row) <= PAINT_KEYS, sorted(set(tech_row) - PAINT_KEYS)
        assert "lane_gt" in tech_row and "lane_gt_valid" in tech_row
        for banned in ("signals", "signs", "curbs", "objects", "depth", "brake_lights", "road_edge"):
            assert banned not in tech_row

    import cv2

    colour = cv2.imread(str(next((raw / "main").rglob("*.png"))), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(next((raw / "main_depth").rglob("*.png"))), cv2.IMREAD_UNCHANGED)
    sem = cv2.imread(str(next((raw / "main_sem").rglob("*.png"))), cv2.IMREAD_UNCHANGED)
    assert colour.ndim == 3 and colour.shape[2] == 3 and colour.dtype == np.uint8
    assert depth.dtype == np.uint16 and int(depth[0, 0]) == 3000  # 0.1 * 300 m * 100 cm
    assert sem.dtype == np.uint8 and sem.ndim == 2 and int(sem.max()) > 0
    print(
        f"short collect rows={len(state)} traffic={len(filled)} "
        f"signals={len(signals)} depth_cm={int(depth[0, 0])}"
    )


def main() -> None:
    check_train_gt_main_and_wide_only()
    check_lidar_and_radar_stay_off()
    check_traffic_chunk_walks_non_player_objects()
    check_missing_signal_api_is_err_row()
    check_brake_lights_come_from_each_vehicle()
    check_electrics_attach_failure_is_masked()
    check_electrics_poll_failure_is_masked()
    check_iqu45_smoke_is_documented()
    check_short_collect_writes_recorder_files()
    print("test_labels_v4_recorder: OK")


if __name__ == "__main__":
    main()
