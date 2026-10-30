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


def main() -> None:
    check_etk800_identical()
    check_sbr_moves_only_repeaters()
    check_collector()
    check_supervisor()
    print("rig overrides ok")


if __name__ == "__main__":
    main()
