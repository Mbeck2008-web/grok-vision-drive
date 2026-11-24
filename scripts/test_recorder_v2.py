#!/usr/bin/env python3
"""Recorder v2: 8-cam depth/sem, intrinsics, per-cam stamps, actor boxes, chase cam, no pedestrians.

Offline. A fake beamngpy stands in for Tech (same harness as
``test_labels_v4_recorder``). Nothing here talks to BeamNG or a GPU.

    py -3 scripts/test_recorder_v2.py
"""
from __future__ import annotations

import io
import json
import math
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from python.sensors.cameras import CAM_IDS, load_camera_config  # noqa: E402

FIXTURE_037126 = ROOT / "tests" / "fixtures" / "depth_037126"
BOX = {"half": [0.95, 2.35, 0.72], "ax": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]}
TRAFFIC_V2 = [
    {"name": "gvd_traffic_0", "id": 1042, "pos": [1.0, 40.0, 0.5], "vel": [0.0, 8.0, 0.0],
     "dir": [0.0, 1.0, 0.0], "box": BOX},
    {"name": "parked_bus", "id": 1043, "pos": [-3.0, 15.0, 0.5], "vel": [0.0, 0.0, 0.0],
     "dir": [1.0, 0.0, 0.0]},
]


# --------------------------------------------------------------- item 1

def check_train_gt_on_all_eight() -> None:
    cfg = load_camera_config()
    by_id = {str(cam["id"]): cam for cam in cfg["cameras"]}
    assert set(by_id) == set(CAM_IDS)
    for cid in CAM_IDS:
        assert by_id[cid].get("train_gt") is True, cid
    print("train_gt true on all eight cameras")


def check_live_attach_forces_gt_off() -> None:
    """Depth/sem stay training-only: the live kwargs never render them."""
    from python.sensors.cameras import beamng_camera_sensor_kwargs

    src = (ROOT / "python" / "sensors" / "cameras.py").read_text(encoding="utf-8")
    assert "train_gt=False" in src
    kw = beamng_camera_sensor_kwargs(
        pos=(0.0, 0.0, 1.0), direction=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0), fov_v=34.0,
        resolution=(640, 480), update_s=0.1, near_m=0.05, far_m=300.0, shmem=True,
        streaming=True, rgb_only=True, update_priority=0.0, cid="main",
    )
    assert kw["is_render_depth"] is False and kw["is_render_annotations"] is False
    print("live camera kwargs keep depth and semantics off")


def check_gt_gate_halves_when_behind() -> None:
    from python.data.raw_frames import TrainGtGate

    gate = TrainGtGate(period_s=0.1)
    kept = [gate.select(0.1, ["main", "rear"]) for _ in range(20)]
    assert all(k == {"main", "rear"} for k in kept), kept
    assert gate.stride == 1
    # Sustained slow ticks: every other GT frame per camera, colour untouched.
    kept = [gate.select(0.17, ["main"]) for _ in range(40)]
    assert gate.stride == 2
    tail = kept[-20:]
    assert sum(1 for k in tail if "main" in k) == 10, tail
    assert all(("main" in a) != ("main" in b) for a, b in zip(tail, tail[1:])), tail
    # A camera's alternation is its own: a slow camera due once is kept.
    gate2 = TrainGtGate(period_s=0.1)
    for _ in range(40):
        gate2.select(0.17, ["main"])
    assert "rear" in gate2.select(0.17, ["main", "rear"])
    # Back on time: full rate again.
    for _ in range(40):
        gate.select(0.1, ["main"])
    assert gate.stride == 1 and gate.select(0.1, ["main"]) == {"main"}
    print("GT gate: full rate on time, every other frame when behind, recovers")


# --------------------------------------------------------------- item 2

PED = re.compile(r"pedestrian|\bperson\b", re.I)
LABEL_TREES = ("python/perception", "python/train", "python/collect", "python/data", "config")


def check_no_pedestrian_labels_or_classes() -> None:
    from python.perception.detect import coco_class_name
    from python.train.scene_net import N_OBJECTS, OBJECT_CLASSES, scene_head_sizes

    for cls_id in range(80):
        name = coco_class_name(cls_id)
        assert name is None or not PED.search(name), (cls_id, name)
    assert coco_class_name(0) is None  # COCO person is dropped, not renamed
    assert not any(PED.search(c) for c in OBJECT_CLASSES), OBJECT_CLASSES
    assert OBJECT_CLASSES == ("vehicle", "car", "truck", "bus", "bike"), OBJECT_CLASSES
    heads = dict(scene_head_sizes())
    assert heads["object_class_head"] == N_OBJECTS * 5 == 80, heads
    hits = []
    for tree in LABEL_TREES:
        for path in sorted((ROOT / tree).rglob("*")):
            if path.suffix not in (".py", ".yaml", ".yml", ".json"):
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if PED.search(line) and "gvd-allow-pedestrian-word" not in line:
                    hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not hits, "pedestrian label/class still present:\n" + "\n".join(hits)
    print("no pedestrian label, class, or COCO person mapping in perception/train/collect/data/config")


def check_old_checkpoint_class_count_is_refused() -> None:
    from python.train.scene_net import (
        N_OBJECTS,
        OBJECT_CLASSES,
        ObjectClassCountError,
        check_object_class_head,
        scene_to_predict_kwargs,
    )

    good = {"object_class_head.weight": np.zeros((N_OBJECTS * len(OBJECT_CLASSES), 4))}
    check_object_class_head(good)
    old = {"object_class_head.weight": np.zeros((N_OBJECTS * 6, 4)), "object_class_head.bias": np.zeros(96)}
    try:
        check_object_class_head(old)
    except ObjectClassCountError as exc:
        text = str(exc)
        assert "96" in text and "80" in text and "pedestrian" in text, text
    else:
        raise AssertionError("a 6-class (pedestrian) checkpoint loaded")
    # Decoding a pre-v2 6-wide class output never maps slot 5 onto slot 4 silently.
    scene = {
        "lanes": np.zeros((6, 16, 2)), "lane_valid": np.zeros(6), "curbs": np.zeros((2, 16, 2)),
        "curb_valid": np.zeros(2), "objects": np.zeros((N_OBJECTS, 8)),
        "object_class": np.zeros((N_OBJECTS, 6)), "object_valid": np.ones(N_OBJECTS),
        "signs": np.zeros((8, 4)), "sign_class": np.zeros((8, 4)), "sign_state": np.zeros((8, 4)),
    }
    try:
        scene_to_predict_kwargs(scene)
    except ObjectClassCountError:
        pass
    else:
        raise AssertionError("a 6-wide object_class output decoded")
    print("old 96-wide object_class head refused with a clear error")


def check_old_checkpoint_refused_by_loaders() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        from gvd_test_env import skip_check

        skip_check("check_old_checkpoint_refused_by_loaders", "torch not importable")
        return
    from python.train.epoch_ckpt import load_into_net
    from python.train.scene_net import ObjectClassCountError, SceneNet

    net = SceneNet(cams=("main",)).cpu()
    weights = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
    w = weights["object_class_head.weight"]
    weights["object_class_head.weight"] = np.zeros((96, w.shape[1]), dtype=w.dtype)
    weights["object_class_head.bias"] = np.zeros((96,), dtype=w.dtype)
    try:
        load_into_net(net, weights)
    except ObjectClassCountError as exc:
        assert "pedestrian" in str(exc)
    else:
        raise AssertionError("epoch checkpoint with 96 class outputs loaded")
    print("epoch checkpoint loader refuses a pre-v2 class head")


# --------------------------------------------------------------- item 3 / 4

def check_intrinsics_from_fov() -> None:
    from python.data.raw_frames import camera_intrinsics

    k = camera_intrinsics(34.0, (640, 480))
    assert abs(k["fy"] - 785.005) < 1e-3 and k["fx"] == k["fy"], k
    assert k["cx"] == 320.0 and k["cy"] == 240.0
    assert k["K"] == [[k["fx"], 0.0, 320.0], [0.0, k["fy"], 240.0], [0.0, 0.0, 1.0]]
    assert k["model"] == "pinhole" and k["distortion"] is None and "fov_v" in k["source"]
    wide = camera_intrinsics(88.7, (640, 360))
    assert abs(wide["fy"] - 184.131) < 1e-3 and wide["cy"] == 180.0, wide
    print("pinhole K from fov_v and resolution matches tracer's table")


def check_sidecar_has_k_and_sbr_mounts() -> None:
    from python.data.raw_frames import camera_sidecar
    from python.sensors.rig_overrides import rig_for_model, rig_model, rig_overrides

    base = load_camera_config()
    cfg = rig_for_model(base, "sbr")
    side = camera_sidecar(cfg, rig_model=rig_model(base, "sbr"), overrides=rig_overrides(base, "sbr"))
    by = {c["id"]: c for c in side}
    assert set(by) == set(CAM_IDS)
    for cid, cam in by.items():
        k = cam["intrinsics"]
        assert k["fx"] > 0 and k["cx"] == cam["resolution"][0] / 2.0, (cid, k)
        assert cam["mount"]["rig_model"] == "sbr", cam["mount"]
        assert cam["train_gt"] is True and cam["depth_kind"] == "planar_z_cm", cam
    assert by["repeatL"]["mount"]["source"] == "vehicle_override" and by["repeatL"]["pos_m"] == [-1.1, 0.0, 0.9]
    assert by["main"]["mount"]["source"] == "yaml_draft"
    print("cameras.json carries K per camera and the eSBR mount (override vs draft)")


def check_writer_stamps_and_train_only_pose() -> None:
    from python.data.raw_frames import RawFrameWriter, camera_sidecar
    from scripts.gvd_test_env import collect_root

    side = camera_sidecar(load_camera_config())
    with collect_root() as root:
        w = RawFrameWriter(root / "rec", side)
        frames = {
            "main": np.zeros((480, 640, 3), np.uint8),
            "rear": np.zeros((360, 640, 3), np.uint8),
            "main_depth": np.full((480, 640), 1000, np.uint16),
            "main_sem": np.zeros((480, 640), np.uint8),
        }
        stamps = {
            "main": {"wall": 100.000, "mono_before": 5.0, "mono_after": 5.001, "call_s": 1e-5},
            "rear": {"wall": 100.042, "mono_before": 5.04, "mono_after": 5.041, "call_s": 1e-5},
        }
        stamps["main_depth"] = stamps["main"]
        stamps["main_sem"] = stamps["main"]
        line = w.write(frames, t=100.05, timestamps=stamps)
        cams = line["cams"]
        assert abs(line["cam_span_ms"] - 42.0) < 1e-6, line["cam_span_ms"]
        for cid in ("main_depth", "main_sem"):
            entry = cams[cid]
            assert entry["train_only"] is True
            assert entry["t"] == 100.0 and entry["mono_before"] == 5.0, entry
            for key in ("pos_m", "yaw_deg", "pitch_deg", "fov_v", "far_m", "resolution", "K"):
                assert entry[key] == cams["main"][key], (cid, key, entry[key])
        assert cams["main"]["K"] == [785.005, 785.005, 320.0, 240.0] or abs(cams["main"]["K"][0] - 785.005) < 1e-3
        payload = json.loads((root / "rec" / "cameras.json").read_text(encoding="utf-8"))
        assert payload["depth_kind"] == "planar_z_cm"
        assert all("intrinsics" in c for c in payload["cameras"])
    print("writer: per-cam stamps, train-only pose/K, row cam_span_ms")


# --------------------------------------------------------------- item 5

def check_parse_traffic_actor_boxes() -> None:
    from python.collect import control_log as cl

    cars = cl.parse_traffic(json.dumps(TRAFFIC_V2))
    by = {c["name"]: c for c in cars}
    car = by["gvd_traffic_0"]
    assert car["id"] == 1042
    assert abs(car["length_m"] - 4.70) < 1e-9 and abs(car["width_m"] - 1.90) < 1e-9 and abs(car["height_m"] - 1.44) < 1e-9, car
    assert car["dims_source"] == "oobb_half_extents"
    assert abs(car["heading_rad"] - math.pi / 2) < 1e-9
    bus = by["parked_bus"]
    assert bus["length_m"] is None and bus["width_m"] is None and bus["height_m"] is None
    assert bus["dims_source"] is None and abs(bus["heading_rad"]) < 1e-9
    bad = dict(TRAFFIC_V2[0], box={"half": [0.0, 2.0, 0.7], "ax": BOX["ax"]})
    assert cl.parse_traffic(json.dumps([bad]))[0]["length_m"] is None  # never a guessed size
    print("traffic rows: id, heading, length/width/height from the OOBB (null when missing)")


def check_traffic_chunk_runs_in_lua() -> None:
    """Run TRAFFIC_CHUNK under lua5.1 against stub GE objects (one with an OOBB, one without)."""
    from python.collect import control_log as cl

    lua = shutil.which("lua5.1") or shutil.which("lua")
    assert lua, "lua5.1 is required by the offline suite"
    for word in ("getSpawnWorldOOBB", "getHalfExtents", "getAxis", "getID"):
        assert word in cl.TRAFFIC_CHUNK, word
    prelude = r"""
local function vec(x,y,z) return {x=x,y=y,z=z} end
local function enc(v)
  local t = type(v)
  if t == 'number' then return string.format('%.6f', v) end
  if t == 'string' then return '"'..v..'"' end
  if t == 'table' then
    if #v > 0 then local o={} for i=1,#v do o[#o+1]=enc(v[i]) end return '['..table.concat(o,',')..']' end
    local o={} for k,x in pairs(v) do o[#o+1]='"'..k..'":'..enc(x) end table.sort(o) return '{'..table.concat(o,',')..'}'
  end
  return 'null'
end
jsonEncode = enc
local function car(name, id, box)
  local o = {}
  function o:getField(_, _) return name end
  function o:getName() return name end
  function o:getID() return id end
  function o:getPosition() return vec(1,2,3) end
  function o:getVelocity() return vec(0,9,0) end
  function o:getDirectionVector() return vec(0,1,0) end
  if box then
    function o:getSpawnWorldOOBB()
      local b = {}
      function b:getHalfExtents() return vec(0.95,2.35,0.72) end
      function b:getAxis(k) local a={vec(0,-1,0),vec(1,0,0),vec(0,0,1)} return a[k+1] end
      return b
    end
  end
  return o
end
local player = {}
local objs = {player, car('gvd_traffic_0', 7, true), car('parked', 8, false)}
be = {}
function be:getPlayerVehicle(_) return player end
function be:getObjectCount() return #objs end
function be:getObject(i) return objs[i+1] end
"""
    script = prelude + "\nlocal function chunk() " + cl.TRAFFIC_CHUNK + " end\nio.write(chunk())\n"
    out = subprocess.run([lua, "-"], input=script, capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    cars = cl.parse_traffic(out.stdout)
    by = {c["name"]: c for c in cars}
    assert set(by) == {"gvd_traffic_0", "parked"}, out.stdout
    assert by["gvd_traffic_0"]["id"] == 7 and abs(by["gvd_traffic_0"]["length_m"] - 4.7) < 1e-6
    assert by["parked"]["length_m"] is None and by["parked"]["pos"] == [1.0, 2.0, 3.0]
    print("TRAFFIC_CHUNK runs in lua5.1; a car without an OOBB still reports pos/dir")


def check_brake_merge_keeps_actor_fields() -> None:
    from python.collect import control_log as cl
    from python.collect.recorder_v4 import with_brake_lights

    cars = cl.parse_traffic(json.dumps(TRAFFIC_V2))
    merged = with_brake_lights(cars, object())
    for before, after in zip(cars, merged):
        for key in ("id", "heading_rad", "length_m", "width_m", "height_m", "dims_source"):
            assert after[key] == before[key], key
        assert "brake_lights" in after
    print("brake-light merge keeps id/heading/dims")


def check_traffic_slots_beyond_two() -> None:
    import random

    from python.collect.spawn_mix import TRAFFIC_MIN_M, SpawnDeck, traffic_lua

    deck = SpawnDeck.load(ROOT / "tests" / "fixtures" / "roads" / "west_coast_usa.json.gz", random.Random(3))
    best = None
    for _ in range(12):
        pick = deck.take()
        assert pick is not None
        slots, _notes = deck.traffic_slots(pick, 6)
        if best is None or len(slots) > len(best[1]):
            best = (pick, slots)
    pick, slots = best
    assert len(slots) > 2, [s[0] for s in slots]
    for i, a in enumerate(slots):
        for b in slots[i + 1:]:
            assert math.dist(a[1], b[1]) >= TRAFFIC_MIN_M, (a, b)
    two, _ = deck.traffic_slots(pick, 2)
    assert [s[0] for s in two] == [s[0] for s in slots[:2]] or len(two) <= 2
    lua = traffic_lua(pick, len(slots), slots)[0]
    for i in range(len(slots)):
        assert f"gvd_traffic_{i}" in lua
    print(f"traffic_slots places {len(slots)} cars for count 6, none stacked")


def check_spawn_deck_seed_and_kind() -> None:
    import random

    from python.collect.spawn_mix import RAMP, SpawnDeck

    path = ROOT / "tests" / "fixtures" / "roads" / "west_coast_usa.json.gz"
    a = SpawnDeck.load(path, random.Random(1)).take()
    b = SpawnDeck.load(path, random.Random(2)).take()
    assert a is not None and b is not None and a.road_id != b.road_id
    ramp = SpawnDeck.load(path, random.Random(1), only_kind=RAMP)
    for _ in range(5):
        pick = ramp.take()
        assert pick is not None and pick.kind == RAMP, pick
    print("spawn deck: a seed changes the first road; only_kind=ramp gives ramps")


# --------------------------------------------------------------- item 7

def check_chase_cam_is_recording_only() -> None:
    from python.data.raw_frames import (
        STRIP_ORDER,
        camera_sidecar,
        cameras_for_row,
        is_recording_only_cam,
        is_train_only_cam,
    )

    cfg = load_camera_config()
    chase = cfg.get("chase_camera")
    assert isinstance(chase, dict) and chase.get("id") == "chase", chase
    assert "chase" not in CAM_IDS and "chase" not in STRIP_ORDER
    assert all(str(c.get("id")) != "chase" for c in cfg["cameras"])
    assert chase["pos_m"][1] < -3.0 and chase["pos_m"][2] > 2.5, chase  # behind and above
    for cid in ("chase", "chase_depth", "chase_sem"):
        assert is_recording_only_cam(cid) and is_train_only_cam(cid), cid
    assert not is_recording_only_cam("main") and not is_recording_only_cam("main_depth")
    assert cameras_for_row({"cams": {"main": {}, "chase": {}}}) == ["main"]
    assert all(c["id"] != "chase" for c in camera_sidecar(cfg))
    with_chase = camera_sidecar(cfg, chase=True)
    rec = [c for c in with_chase if c["id"] == "chase"]
    assert len(rec) == 1 and rec[0]["recording_only"] is True and "intrinsics" in rec[0]
    print("chase camera: own yaml block, recording-only, never in the strip or CAM_IDS")


# --------------------------------------------------------------- item 8

def check_depth_is_planar_z_on_real_frame() -> None:
    import cv2

    from python.data.depth_check import depth_kind_verdict
    from python.data.raw_frames import camera_intrinsics

    depth = cv2.imread(str(FIXTURE_037126 / "raw" / "main_depth" / "037126.png"), cv2.IMREAD_UNCHANGED)
    sem = cv2.imread(str(FIXTURE_037126 / "raw" / "main_sem" / "037126.png"), cv2.IMREAD_UNCHANGED)
    classes = json.loads((FIXTURE_037126 / "sem_classes.json").read_text(encoding="utf-8"))["classes"]
    got = depth_kind_verdict(depth, sem, classes, camera_intrinsics(34.0, (640, 480)))
    z, ray = got["planar_z"], got["ray"]
    assert got["verdict"] == "planar_z", got
    assert z["rms_m"] < 0.01 and ray["rms_m"] > 3 * z["rms_m"], got
    assert z["inlier_frac"] > 0.99 and ray["inlier_frac"] < 0.95, got
    # The audit's 1.14 m came from reading planar z as a ray length.
    assert 1.10 < ray["height_m"] < 1.15, ray
    assert 1.20 < z["height_m"] < 1.235, z
    print(
        f"037126 depth is planar z: rms {z['rms_m'] * 100:.2f} cm vs ray {ray['rms_m'] * 100:.2f} cm; "
        f"height {z['height_m']:.3f} m (ray reading {ray['height_m']:.3f} m)"
    )


def check_z_to_ray_conversion() -> None:
    from python.data.depth_check import backproject, depth_z_to_ray
    from python.data.raw_frames import camera_intrinsics

    k = camera_intrinsics(34.0, (640, 480))
    z = np.full((480, 640), 10.0)
    ray = depth_z_to_ray(z, k)
    u, v = 0, 0
    xn = (u + 0.5 - k["cx"]) / k["fx"]
    yn = (v + 0.5 - k["cy"]) / k["fy"]
    assert abs(ray[v, u] - 10.0 * math.sqrt(1 + xn * xn + yn * yn)) < 1e-9
    assert abs(ray[240, 320] - 10.0 * math.sqrt(1 + (0.5 / k["fx"]) ** 2 + (0.5 / k["fy"]) ** 2)) < 1e-9
    pts = backproject(z, k, "planar_z")
    assert np.allclose(pts[..., 2], 10.0)
    assert np.allclose(np.linalg.norm(backproject(ray, k, "ray"), axis=-1), ray)
    # Synthetic flat road, camera 1.26 m up, pitched 6 deg down: planar z recovers 1.26 m.
    from python.data.depth_check import fit_ground_plane, synthetic_ground_depth

    synth = synthetic_ground_depth(k, (640, 480), height_m=1.26, pitch_deg=6.0)
    mask = np.isfinite(synth) & (synth < 60.0)
    mask[: 300, :] = False
    fit = fit_ground_plane(np.where(mask, synth, 0.0) * 100.0, mask, k, "planar_z")
    assert abs(fit["height_m"] - 1.26) < 1e-3 and abs(fit["pitch_deg"] - 6.0) < 0.05, fit
    print("z-to-ray conversion and synthetic ground fit OK")


def check_depth_check_script_on_fixture() -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_depth_scale.py"), str(FIXTURE_037126), "--frames", "037126"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "verdict planar_z" in out.stdout, out.stdout
    print("check_depth_scale.py on the 037126 fixture: planar_z")


# --------------------------------------------------------------- short collect (fake Tech)

def _run_fake_collect(extra: list[str], *, tick_s: float = 0.1, stop_at: int = 12):
    import python.data.raw_queue as raw_queue
    import python.debug.gt_logger as gt_logger
    import python.sensors.tech as tech
    import scripts.collect_scene_lanes as collect
    import test_labels_v4_recorder as v4
    from python.collect.auto_mode import stop_file
    from python.debug.gt_geom import RoadPoly
    from scripts.gvd_test_env import collect_root

    beamngpy, sensors, made = v4._fake_modules()
    old_mods = {key: sys.modules.get(key) for key in ("beamngpy", "beamngpy.sensors")}
    sys.modules["beamngpy"] = beamngpy
    sys.modules["beamngpy.sensors"] = sensors
    base_queue = raw_queue.RawWriteQueue

    class _DeepQueue(base_queue):
        def __init__(self, writer, max_depth: int = 64, **kwargs) -> None:
            del max_depth
            super().__init__(writer, 100000, **kwargs)

    previous = (
        tech.open_tech_beamngpy, tech.widen_socket_timeout, tech.release_tech_beamngpy,
        gt_logger.load_road_polys, collect.time.sleep, collect.time.monotonic, collect.tick_sleep_s,
    )
    clock = {"t": 0.0}
    real_sleep = collect.time.sleep
    main_thread = threading.main_thread()

    def sleep(seconds: float) -> None:
        if threading.current_thread() is not main_thread or sys._getframe(1).f_globals.get("__name__") == "subprocess":
            real_sleep(min(float(seconds or 0.0), 0.05))
            return
        clock["t"] += float(seconds or 0.0)

    class _Bng2(v4._Bng):
        def queue_lua_command(self, command: str, response: bool = False):
            self.commands.append(str(command))
            if response and "getObjectCount" in str(command):
                return json.dumps(TRAFFIC_V2)
            return None

    root_cm = collect_root()
    root = root_cm.__enter__()
    out = root / "scene"
    ego = v4._Vehicle("ego", stop_file(out), stop_at)
    lead = v4._Vehicle("gvd_traffic_0", None, 10**9)
    parked = v4._Vehicle("parked_bus", None, 10**9)
    bng = _Bng2(ego, lead, parked)
    tech.open_tech_beamngpy = lambda *a, **k: bng  # type: ignore[assignment]
    tech.widen_socket_timeout = lambda *a, **k: None  # type: ignore[assignment]
    tech.release_tech_beamngpy = lambda *a, **k: None  # type: ignore[assignment]
    gt_logger.load_road_polys = lambda session: ([RoadPoly("r", [(0.0, 0.0), (0.0, 80.0)], 7.0)], "")  # type: ignore[assignment]
    collect.time.sleep = sleep  # type: ignore[method-assign]
    collect.time.monotonic = lambda: clock["t"]  # type: ignore[method-assign]
    collect.tick_sleep_s = lambda _elapsed, _period=0.1: tick_s  # type: ignore[method-assign]
    raw_queue.RawWriteQueue = _DeepQueue  # type: ignore[misc]
    buf = io.StringIO()
    old_out = sys.stdout
    sys.stdout = buf
    try:
        code = collect.main(
            ["--minutes", "5", "--smoke-s", "0", "--no-excursions", "--data-dir", str(root),
             "--out", str(out), "--host", "127.0.0.1", "--port", "1", *extra]
        )
        text = buf.getvalue()
    finally:
        sys.stdout = old_out
        (
            tech.open_tech_beamngpy, tech.widen_socket_timeout, tech.release_tech_beamngpy,
            gt_logger.load_road_polys, collect.time.sleep, collect.time.monotonic, collect.tick_sleep_s,
        ) = previous
        raw_queue.RawWriteQueue = base_queue  # type: ignore[misc]
        for key, value in old_mods.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return code, text, out, made, root_cm


def _rows(out: Path, name: str = "state.jsonl") -> list[dict]:
    return [json.loads(line) for line in (out / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def check_short_collect_v2() -> None:
    code, text, out, made, root_cm = _run_fake_collect(["--chase-cam"])
    try:
        assert code == 0, text[-2000:]
        for cid in CAM_IDS:
            kw = made[f"gvd_collect_{cid}"]
            assert kw["is_render_depth"] is True and kw["is_render_annotations"] is True, cid
        assert "gvd_collect_chase" in made and made["gvd_collect_chase"]["is_render_depth"] is True
        raw = out / "raw"
        for cid in (*CAM_IDS, "chase"):
            for sub in (cid, f"{cid}_depth", f"{cid}_sem"):
                assert any((raw / sub).rglob("*.png")), sub
        state = _rows(out)
        assert state
        for row in state:
            assert "chase" not in [c for c in row["cams"] if c in CAM_IDS]
            for cid, entry in row["cams"].items():
                assert entry["t"] is not None and entry["mono_before"] is not None, (cid, entry)
                assert entry.get("K") and entry.get("pos_m") is not None, (cid, entry)
            rgb = [c for c in row["cams"] if c in CAM_IDS]
            if len(rgb) >= 2:
                assert row["cam_span_ms"] is not None and row["cam_span_ms"] >= 0.0
            # item 6: wheel speed and yaw rate on every row
            assert row["ego"].get("yaw_rate") is not None, row["ego"]
            assert row["veh"].get("wheelspeed") == 10.0, row.get("veh")
            assert row.get("gt_stride") == 1
        filled = [row for row in state if row.get("traffic")]
        assert filled
        for row in filled:
            by = {car["name"]: car for car in row["traffic"]}
            assert abs(by["gvd_traffic_0"]["length_m"] - 4.7) < 1e-9 and by["gvd_traffic_0"]["id"] == 1042
            assert by["parked_bus"]["length_m"] is None
        cams_json = json.loads((out / "cameras.json").read_text(encoding="utf-8"))
        ids = [c["id"] for c in cams_json["cameras"]]
        assert ids.count("chase") == 1 and set(ids) == {*CAM_IDS, "chase"}
        assert all(c["mount"]["rig_model"] == "sbr" for c in cams_json["cameras"] if c["id"] != "chase")
        from python.data.raw_frames import list_raw_images

        listed = {p.parent.parent.name for p in list_raw_images(out)}
        assert "chase" not in listed and "main" in listed, listed
    finally:
        root_cm.__exit__(None, None, None)
    print(f"short collect v2: 8 cams + chase with depth/sem, K, stamps, dims, {len(state)} rows")


def check_short_collect_behind_halves_gt() -> None:
    code, text, out, _made, root_cm = _run_fake_collect([], tick_s=0.2, stop_at=40)
    try:
        assert code == 0, text[-2000:]
        state = _rows(out)
        mains = [r for r in state if "main" in r["cams"]]
        gts = [r for r in mains if "main_depth" in r["cams"]]
        assert len(mains) >= 20, len(mains)
        assert any(r.get("gt_stride") == 2 for r in state), [r.get("gt_stride") for r in state]
        late = [r for r in mains[-12:]]
        kept = sum(1 for r in late if "main_depth" in r["cams"])
        assert kept == 6, [("main_depth" in r["cams"]) for r in late]
        assert len(gts) < len(mains)
        # colour is never thinned by the GT gate
        assert all("main" in r["cams"] for r in state[2:]), [sorted(r["cams"]) for r in state[:4]]
    finally:
        root_cm.__exit__(None, None, None)
    print("behind the 10 Hz plan: colour every tick, depth/sem every other frame")


def main() -> None:
    check_train_gt_on_all_eight()
    check_live_attach_forces_gt_off()
    check_gt_gate_halves_when_behind()
    check_no_pedestrian_labels_or_classes()
    check_old_checkpoint_class_count_is_refused()
    check_old_checkpoint_refused_by_loaders()
    check_intrinsics_from_fov()
    check_sidecar_has_k_and_sbr_mounts()
    check_writer_stamps_and_train_only_pose()
    check_parse_traffic_actor_boxes()
    check_traffic_chunk_runs_in_lua()
    check_brake_merge_keeps_actor_fields()
    check_traffic_slots_beyond_two()
    check_spawn_deck_seed_and_kind()
    check_chase_cam_is_recording_only()
    check_depth_is_planar_z_on_real_frame()
    check_z_to_ray_conversion()
    check_depth_check_script_on_fixture()
    check_short_collect_v2()
    check_short_collect_behind_halves_gt()
    print("test_recorder_v2: OK")


if __name__ == "__main__":
    main()
