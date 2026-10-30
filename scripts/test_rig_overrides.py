#!/usr/bin/env python3
"""Per-ego-model camera overrides (cameras.yaml vehicle_overrides).

The etk800 rig stays byte-identical; the eSBR moves only repeatL/repeatR, with
the camera count and resolutions unchanged. The collector applies the sbr
mounts (no rig warning), and the supervisor's BeamNGPy backend applies them
when the attached car is an sbr. Offline, no BeamNG.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from python.data.raw_frames import RawFrameWriter, camera_sidecar  # noqa: E402
from python.sensors.cameras import BeamNGPyBackend, load_camera_config  # noqa: E402
from python.sensors.rig_overrides import OVERRIDES_KEY, rig_for_model, rig_model, rig_overrides  # noqa: E402

REPEATERS = {"repeatL", "repeatR"}


def _base(cfg: dict) -> dict:
    """The yaml as it was before the overrides block: everything else, untouched."""
    return {k: v for k, v in cfg.items() if k != OVERRIDES_KEY}


def check_etk800_identical() -> None:
    cfg = load_camera_config()
    assert set(cfg[OVERRIDES_KEY]) == {"sbr"}, cfg[OVERRIDES_KEY]
    frozen = copy.deepcopy(cfg)
    for model in ("etk800", "ETK800", None, "", "pickup"):
        got = rig_for_model(cfg, model)
        assert got == _base(cfg), model
        assert json.dumps(camera_sidecar(got)) == json.dumps(camera_sidecar(_base(cfg))), model
        assert rig_model(cfg, model) == "etk800", model
    assert cfg == frozen  # the loaded config is not mutated
    print("etk800: effective rig == cameras.yaml without the overrides block OK")


def check_sbr_moves_only_repeaters() -> None:
    cfg = load_camera_config()
    base, sbr = _base(cfg)["cameras"], rig_for_model(cfg, "sbr")["cameras"]
    assert [c["id"] for c in sbr] == [c["id"] for c in base] and len(sbr) == 8
    for b, s in zip(base, sbr):
        if b["id"] in REPEATERS:
            assert set(k for k in s if s[k] != b.get(k)) == {"pos_m"}, (b, s)
        else:
            assert s == b, b["id"]
        assert s["live_res"] == b["live_res"] and s["yaw_deg"] == b["yaw_deg"] and s["pitch_deg"] == b["pitch_deg"]
    rep = {c["id"]: c["pos_m"] for c in sbr if c["id"] in REPEATERS}
    old = {c["id"]: c["pos_m"] for c in base if c["id"] in REPEATERS}
    for cid in REPEATERS:
        assert rep[cid][1] < old[cid][1], (cid, rep[cid], old[cid])  # rearward
        assert abs(rep[cid][0]) > abs(old[cid][0]) and abs(rep[cid][0]) > 1.0, (cid, rep[cid])  # past the 2.0 m body
    assert rep["repeatL"][0] == -rep["repeatR"][0] and rep["repeatL"][1:] == rep["repeatR"][1:]
    assert rig_model(cfg, "sbr") == "sbr" and rig_model(cfg, "SBR") == "sbr"
    print(f"sbr: only repeatL/R pos_m change {old} -> {rep} OK")


def check_collector() -> None:
    import scripts.test_collect_spawn_fix as sf
    from scripts.test_recorder_ev import _west

    base = sf._CrashVehicle

    class _Sbr(base):
        def __init__(self, *a, **k) -> None:
            k.setdefault("model", "sbr")
            super().__init__(*a, **k)

    sf._CrashVehicle = _Sbr  # type: ignore[misc]
    try:
        code, text, out = _west(lambda n: "", [], 60)
    finally:
        sf._CrashVehicle = base  # type: ignore[misc]
    assert code == 0, text[-2000:]
    assert "camera rig calibrated" not in text, text[:3000]
    ego = json.loads((out / "collect.json").read_text(encoding="utf-8"))["ego"]
    assert ego["camera_rig_model"] == "sbr", ego
    # The tuned mounts are recorded, as applied: only repeatL/R pos_m.
    want_over = load_camera_config()[OVERRIDES_KEY]["sbr"]
    assert ego["camera_overrides"] == json.loads(json.dumps(want_over)), ego
    assert set(ego["camera_overrides"]) == REPEATERS and rig_overrides(load_camera_config(), "sbr") == want_over
    cams = {c["id"]: c for c in json.loads((out / "cameras.json").read_text(encoding="utf-8"))["cameras"]}
    want = {c["id"]: c for c in camera_sidecar(rig_for_model(load_camera_config(), "sbr"))}
    assert cams == json.loads(json.dumps(want)), "sbr cameras.json is the sbr rig"

    code, text, out = _west(lambda n: "", ["--ego-model", "etk800", "--ego-config", ""], 60)
    assert code == 0 and "camera rig calibrated" not in text, text[:2000]
    with tempfile.TemporaryDirectory() as tmp:
        RawFrameWriter(tmp, camera_sidecar(_base(load_camera_config())))
        ref = (Path(tmp) / "cameras.json").read_bytes()
    assert (out / "cameras.json").read_bytes() == ref, "etk800 cameras.json byte-identical to the base rig"
    ego = json.loads((out / "collect.json").read_text(encoding="utf-8"))["ego"]
    assert ego["camera_rig_model"] == "etk800" and ego["camera_overrides"] == {}, ego
    print("collector: sbr gets the sbr rig without the rig warning; etk800 cameras.json byte-identical OK")


def _supervisor_positions(model: str) -> dict[str, tuple]:
    made: dict[str, tuple] = {}

    class Cam:
        def __init__(self, name, _bng, _vehicle, **kwargs):
            made[name] = tuple(kwargs.get("pos") or ())
            self.name, self.kwargs, self.is_streaming = name, kwargs, True
            self.resolution = kwargs.get("resolution", (8, 8))
            self.update_priority = 0.0

        def get_update_priority(self):
            return self.update_priority

        def set_update_priority(self, p):
            self.update_priority = float(p)

        def set_max_pending_requests(self, n):
            return None

        def remove(self):
            return None

    sensors = types.ModuleType("beamngpy.sensors")
    sensors.Camera = Cam  # type: ignore[attr-defined]
    beamngpy = types.ModuleType("beamngpy")
    beamngpy.sensors = sensors  # type: ignore[attr-defined]
    old = {k: sys.modules.get(k) for k in ("beamngpy", "beamngpy.sensors")}
    sys.modules["beamngpy"], sys.modules["beamngpy.sensors"] = beamngpy, sensors
    try:
        be = BeamNGPyBackend(
            config=load_camera_config(),
            tech_config={"wait_vehicle_s": 0, "cameras": {"attach": True, "rgb_only": True, "update_s": 0.067,
                                                          "shared_memory": True, "streaming": True}},
        )

        def _connect(explicit=True):
            be.session.vehicle = types.SimpleNamespace(model=model)
            be.session.bng = object()
            return True

        be.session.connect = _connect  # type: ignore[method-assign]
        be.session.attach_vehicle_sensors = lambda: {}  # type: ignore[method-assign]
        be.open()
        assert OVERRIDES_KEY not in be.config
    finally:
        for key, mod in old.items():
            if mod is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = mod
    return made


def check_supervisor() -> None:
    etk, sbr = _supervisor_positions("etk800"), _supervisor_positions("sbr")
    assert len(etk) == len(sbr) == 8 and etk.keys() == sbr.keys(), (etk, sbr)
    moved = {name for name in etk if etk[name] != sbr[name]}
    assert moved == {"gvd_repeatL", "gvd_repeatR"}, (moved, etk, sbr)
    print(f"supervisor: sbr moves {sorted(moved)} only; etk800 mounts unchanged OK")


def _pose_rows() -> dict:
    from python.perception import lanes

    return {cid: (p.x, p.y, p.z, p.yaw_deg, p.pitch_deg, p.fov_v_deg, p.width, p.height)
            for cid, p in lanes.camera_poses().items()}


def check_lanes_poses() -> None:
    """lanes.py side/rear poses (posed_lanes_from_image, lanes_bev) follow the attached car's rig."""
    from python.perception import lanes
    from python.viz import debug_draw

    from python.sensors.rig_overrides import set_active_model

    set_active_model(None)  # check_supervisor left the sbr attached
    cfg = load_camera_config()
    base_rows = {c["id"]: tuple(c["pos_m"]) for c in _base(cfg)["cameras"]}
    sbr_rows = {c["id"]: tuple(c["pos_m"]) for c in rig_for_model(cfg, "sbr")["cameras"]}
    before = _pose_rows()  # nothing attached yet: the base (etk800) rig
    assert {k: v[:3] for k, v in before.items()} == {k: base_rows[k] for k in before}, before
    etk_frustums = json.dumps(debug_draw.load_frustums())

    _supervisor_positions("sbr")
    got = _pose_rows()
    for cid in got:
        assert got[cid][:3] == sbr_rows[cid], (cid, got[cid], sbr_rows[cid])
    assert {k for k in got if got[k] != before[k]} == REPEATERS, (got, before)
    assert lanes.camera_pose("repeatL").x == -1.10 and lanes.camera_pose("repeatL").y == 0.0
    sbr_fr = {f["id"]: (f["x"], f["y"], f["z"]) for f in debug_draw.load_frustums()}
    assert sbr_fr["repeatR"] == sbr_rows["repeatR"] and sbr_fr["main"] == base_rows["main"], sbr_fr
    # The same repeater pixel lands somewhere else on the ground once the sbr mount is in.
    rep = lanes.camera_pose("repeatR")
    hit_sbr = lanes.project_pose_pixel(rep, rep.width * 0.5, rep.height * 0.9)

    _supervisor_positions("etk800")
    assert _pose_rows() == before, "etk800 poses byte-identical to the base rig"
    assert json.dumps(debug_draw.load_frustums()) == etk_frustums
    rep = lanes.camera_pose("repeatR")
    hit_etk = lanes.project_pose_pixel(rep, rep.width * 0.5, rep.height * 0.9)
    assert hit_sbr and hit_etk and abs(hit_sbr["y"] - hit_etk["y"]) > 0.5, (hit_sbr, hit_etk)
    print(f"lanes: sbr repeater poses reach camera_pose/project_pose_pixel (y {hit_etk['y']:.2f} -> "
          f"{hit_sbr['y']:.2f} m); etk800 poses and frustums unchanged OK")


# repeatL | repeatR | wide from the 8-camera grid of the after smoke (BeamNG.tech 0.39.4,
# west_coast_usa, eSBR electric_800 in Lime, t=+30 s), each 160x120.
ESBR_FIXTURE = ROOT / "tests" / "fixtures" / "esbr_repeaters_wide_t30.jpg"
ESBR_FIXTURE_REPEATERS = {"repeatL": [-1.10, 0.00, 0.90], "repeatR": [1.10, 0.00, 0.90]}


def _lime_frac(bgr) -> float:
    b, g, r = (bgr[..., i].astype(int) for i in range(3))
    return float(((g > 150) & (g > r + 30) & (g > b + 80)).mean())


# repeatL | repeatR of the before smoke (same spawn and t=+30 s, etk800 mounts on the eSBR, white
# car), each 160x120: the front tyre and wheel arch fill the inboard lower corner.
ESBR_BEFORE_FIXTURE = ROOT / "tests" / "fixtures" / "esbr_before_repeaters_t30.jpg"


def _dark_low_frac(bgr) -> float:
    """Near-black share of the lower half: tyres and wheel arches (max channel < 45)."""
    low = bgr[bgr.shape[0] // 2 :]
    return float((low.max(axis=2) < 45).mean())


# Measured: after-smoke repeaters 0.010 / 0.000, before-smoke (tyre in frame) 0.43 / 0.42.
DARK_LOW_MAX = 0.05


def _body_free(bgr) -> bool:
    """No ego body in a repeater frame: no Lime paint, and no dark tyre or arch in the lower half."""
    return _lime_frac(bgr) == 0.0 and _dark_low_frac(bgr) < DARK_LOW_MAX


def check_body_check_sees_tyres() -> None:
    """The body check must refuse a tyre or wheel arch in frame, not just Lime paint."""
    import cv2

    before = cv2.imread(str(ESBR_BEFORE_FIXTURE))
    assert before is not None and before.shape == (120, 320, 3), ESBR_BEFORE_FIXTURE
    for name, frame in (("repeatL", before[:, :160]), ("repeatR", before[:, 160:])):
        assert not _body_free(frame), f"before-smoke {name} shows the tyre: must not pass as body-free"
    after = cv2.imread(str(ESBR_FIXTURE))
    blob = after[:, :160].copy()
    assert _body_free(blob)
    cv2.ellipse(blob, (20, 110), (45, 30), 0, 0, 360, (18, 18, 20), -1)  # a dark tyre, inboard low
    assert not _body_free(blob), "a synthetic dark tyre must not pass as body-free"
    print("body check: before-smoke tyres and a synthetic dark tyre are refused OK")


def check_ego_edge_per_model() -> None:
    """sbr repeaters see no ego body, so the live stitch keeps their inboard columns."""
    import cv2

    from python.data import strip_writer
    from python.perception import stitch360
    from python.sensors.rig_overrides import set_active_model

    cfg = load_camera_config()
    sbr = {c["id"]: c["pos_m"] for c in rig_for_model(cfg, "sbr")["cameras"] if c["id"] in REPEATERS}
    assert sbr == ESBR_FIXTURE_REPEATERS, ("fixture was shot at other mounts; re-shoot it", sbr)
    strip = cv2.imread(str(ESBR_FIXTURE))
    assert strip is not None and strip.shape == (120, 480, 3), ESBR_FIXTURE
    frames = {"repeatL": strip[:, :160], "repeatR": strip[:, 160:320], "wide": strip[:, 320:]}
    assert _lime_frac(frames["wide"][60:]) > 0.1, "Lime hood visible on wide: the detector sees the body"
    assert _body_free(frames["repeatL"]) and _body_free(frames["repeatR"]), "after-smoke repeaters show body"

    try:
        for model in (None, "etk800"):
            set_active_model(model)
            assert stitch360.ego_columns("repeatL", stitch360.SECTOR_W) == (0, 40), model
            assert stitch360.ego_columns("repeatR", stitch360.SECTOR_W) == (120, 160), model
            assert int(stitch360.stitch_frames(frames).ego_body.sum()) == 2 * 40 * stitch360.SECTOR_H
        etk_place = {cid: strip_writer.place_sector(cid, img) for cid, img in frames.items()}

        set_active_model("sbr")
        assert stitch360.ego_columns("repeatL", stitch360.SECTOR_W) is None
        assert stitch360.ego_columns("repeatR", stitch360.SECTOR_W) is None
        st = stitch360.stitch_frames(frames)
        assert int(st.ego_body.sum()) == 0
        for sec in st.sectors:
            if sec.cam_id in REPEATERS:
                assert sec.ego_x0 is None and sec.ego_x1 is None
                edge = st.bgr[:, sec.x0 : sec.x0 + 40] if sec.cam_id == "repeatL" else st.bgr[:, sec.x1 - 40 : sec.x1]
                assert int(edge.max()) > 0, sec.cam_id  # inboard road kept
                assert st.allows_vehicle(sec.x0 + 1 if sec.cam_id == "repeatL" else sec.x1 - 2, 48)
        # The training strip keeps one model-independent mask (live strip == frame cache).
        for cid, img in frames.items():
            assert strip_writer.place_sector(cid, img).tobytes() == etk_place[cid].tobytes(), cid
    finally:
        set_active_model(None)
    print("ego edge: sbr fixture repeaters 0 body px (wide hood lime); live stitch keeps sbr inboard columns, "
          "etk800 25% unchanged; training strip mask model-independent OK")


def main() -> None:
    check_etk800_identical()
    check_sbr_moves_only_repeaters()
    check_collector()
    check_supervisor()
    check_lanes_poses()
    check_body_check_sees_tyres()
    check_ego_edge_per_model()
    print("rig overrides ok")


if __name__ == "__main__":
    main()
