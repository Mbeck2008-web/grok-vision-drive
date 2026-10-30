#!/usr/bin/env python3
"""View-only scene-net lane viewer (python/viz/scene_view.py): a view-only same-map experiment.

The viewer shows what a trained scene net outputs; it is not a lane source and never drives.
Checks (spec "live scene viewer" rev 3, tests 1-7):

1. Forward, both contracts: tiny exported SceneNet(cams=("main",)) and SceneNet(cam_age=True),
   20 steps through the viewer; lane_gt[14] finite, conf in [0, 1]; the GRU state equals a fresh
   runtime's after step 16 and after reset_state(). Needs torch for the export (skipped without).
2. Strip parity: the viewer's strip vs the trainer's (frame cache + extract_cams + trainer NCHW),
   bit-for-bit float32, 1-cam and 8-cam.
3. Cannot actuate: SceneView has no control fields and is frozen; scene_view.py imports nothing
   from control / engage / actuator; run_vision hands the viewer only frames and timing and stores
   its result only in st["scene_view"]; a one-tick harness gives the same cmd with the viewer
   off, on and raising.
4. Budget: run_vision headless on the wall clock (--hz 15, 10 s, engaged on the fake mod) through
   scripts/gvd_slow_view_child.py with the viewer off, a 400 ms sleeping forward and a 400 ms
   GIL-holding spin. view_ms <= 0.10 P every tick; heartbeat p99 <= P and <= off p99 + 5 ms;
   gvd_cmd.json write gap < 200 ms; no heartbeat_stale, no disengage; dropped > 0, badge "slow".
5. The live hold is invisible to perception: bundle_tick_frames and the stitch are identical with
   enable_live_hold() on vs off, across reset_live_hold() + scrub_after_respawn().
6. Failure + catalog: missing file, wrong strip width, lane71 without cam ages -> badge, no
   exception; e2e_choices() lists e2e_current.onnx and not e2e_scene.onnx.
7. Model file untouched: sha256 and mtime unchanged after load, switch, reset.
8. The child process never blocks the tick: a slow fake forward never blocks submit; drops while
   busy; badge "slow"; close() <= 0.6 s and the shared memory is gone.
9. The main process never imports torch or onnxruntime for the viewer: a fresh interpreter
   selects, steps and switches between two real exported models; sys.modules checked each time.
10. No orphans: SIGKILL the parent of a live viewer child; the child is gone within 3 s.
11. A switch shows badge "loading" and no lane lines until the new model's first result.

Rev 4: the forward runs in a child process (spawn); see the module docstring.

    py -3 scripts/test_scene_view.py
    py -3 scripts/test_scene_view.py --ratio   # report-only: on/off heartbeat p99 ratio, 5 pairs
"""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from gvd_test_env import _fs_type, isolated_home, skip_check  # noqa: E402
from python.data.strip_writer import CANVAS_W, STRIP_ORDER, compose_strip  # noqa: E402
from python.train.cam_strip import canvas_width, extract_cams  # noqa: E402
from python.viz import scene_view as SV  # noqa: E402

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover - CI has torch
    torch = None

HZ = 15.0
P_MS = 1000.0 / HZ
CONTROL_WORDS = ("steer", "throttle", "brake", "accel", "engage", "cmd", "actuat", "target_v", "veto")


class FakeClock:
    """The viewer's monotonic clock in the unit checks: 0.1 s per tick (>= the 90 ms step gate)."""

    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def tick(self, s: float = 0.1) -> None:
        self.t += s


def _bundle(frames: dict, t: float, ages: np.ndarray | None = None) -> SimpleNamespace:
    """A grab: frames + per-camera stamps (all this tick); optional hold frames/ages (cam_age)."""
    b = SimpleNamespace(
        frames=dict(frames),
        timestamps={cid: t for cid in frames},
        main_bgr=lambda: frames.get("main"),
    )
    if ages is not None:
        b.held_frames = dict(frames)
        b.cam_ages = tuple(float(a) for a in ages)
    return b


def _frames(rng: np.random.Generator, cams=STRIP_ORDER) -> dict:
    out = {}
    for cam in cams:
        h, w = (180, 320) if cam in ("wide", "rear") else (150, 200)
        out[cam] = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    return out


def _export(path: Path, **net_kw) -> Path:
    from python.train.scene_net import SceneNet, export_scene_onnx

    cams = net_kw.get("cams")
    net = SceneNet(**net_kw).cpu().eval()
    return export_scene_onnx(path, net, cams=cams, heads=("lane_gt",), cam_age=bool(net_kw.get("cam_age")))


def _fresh_state(path: Path, cam_age: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from python.train.scene_lane_runtime import SceneLaneRuntime

    rt = SceneLaneRuntime(path, feeds_cam_age=cam_age)
    assert rt.ready, rt.reason
    return rt.h.copy(), rt.tokens.copy(), rt.mask.copy()


def _same_state(rt, fresh, what: str) -> None:
    for name, have, want in zip(("h", "tokens", "mask"), (rt.h, rt.tokens, rt.mask), fresh):
        assert have.shape == want.shape and have.dtype == want.dtype, (what, name, have.shape, want.shape)
        assert np.array_equal(have, want), (what, name, "differs from a fresh runtime")


# ---- 1. forward, both contracts ---------------------------------------------------


def check_1_forward_both_contracts() -> None:
    if torch is None:
        skip_check("forward_both_contracts", "torch is not installed (tiny SceneNet export)")
        return
    from python.debug.lane_targets import LANE_GT_DIM

    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        for label, kw, cams in (
            ("v103-like main-only", {"cams": ("main",)}, ("main",)),
            ("lane71-like 8-cam cam_age", {"cam_age": True}, tuple(STRIP_ORDER)),
        ):
            cam_age = bool(kw.get("cam_age"))
            path = _export(Path(tmp) / f"{label.split()[0]}.onnx", **kw)
            fresh = _fresh_state(path, cam_age)
            clock = FakeClock()
            viewer = SV.SceneNetViewer({path.stem: path}, hz=HZ, can_feed_cam_age=cam_age,
                                       inline=True, clock=clock)
            viewer.select(path.stem)
            backend = SimpleNamespace(enable_live_hold=lambda: None, reset_live_hold=lambda: None)
            rt = viewer.runtime
            assert rt is not None and rt.ready, (label, viewer.latest())
            assert rt.session.get_session_options().intra_op_num_threads == SV.VIEW_INTRA_OP_THREADS, label
            ks = []
            for step in range(1, 21):
                clock.tick()
                frames = _frames(rng, cams)
                ages = np.zeros(8, dtype=np.float32) if cam_age else None
                ms = viewer.submit(_bundle(frames, clock(), ages), backend=backend,
                                   prev_heartbeat_ms=10.0, loop_gap_s=1.0 / HZ)
                assert isinstance(ms, float) and ms >= 0.0, (label, ms)
                view = viewer.latest()
                assert view.status == "ok", (label, step, view)
                vec = np.asarray(view.lane_gt, dtype=np.float64)
                assert vec.shape == (LANE_GT_DIM,) and np.isfinite(vec).all(), (label, step, vec)
                assert 0.0 <= view.conf <= 1.0, (label, view.conf)
                assert view.frame == "body_centreline" and view.stem == path.stem
                ks.append(view.k)
                if step == 16:
                    _same_state(rt, fresh, f"{label}: after step 16")
            assert ks == list(range(1, 17)) + [1, 2, 3, 4], (label, ks)
            assert viewer.latest().dt_ms == 100.0 or abs(viewer.latest().dt_ms - 100.0) < 1e-6, viewer.latest()
            rt.reset_state()
            _same_state(rt, fresh, f"{label}: after reset_state()")
            state = viewer.latest().to_state()
            json.dumps(state)  # the stage paints from gvd_state.json
            assert state["status"] == "ok" and len(state["lane_gt"]) == LANE_GT_DIM
            viewer.close()
            print(f"forward {label}: 20 steps, k={ks[15]}/16 then {ks[16]}, lane_gt[14] finite, "
                  f"conf {view.conf:.3f}, state fresh after step 16 and reset_state() OK")


# ---- 2. strip parity ----------------------------------------------------------------


def check_2_strip_parity() -> None:
    import cv2

    from python.train.frame_cache import FrameCache, build_frame_cache
    from python.train.scene_lane_runtime import _strip_nchw as live_nchw

    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rec"
        frames = _frames(rng)
        for cam, image in frames.items():
            (root / "raw" / cam).mkdir(parents=True)
            cv2.imwrite(str(root / "raw" / cam / "000000.png"), image)
        (root / "state.jsonl").write_text(json.dumps({"i": 0, "t": 0.0}) + "\n", encoding="utf-8")
        dest = Path(tmp) / "cache"
        build_frame_cache(root, dest, ",".join(STRIP_ORDER))
        with FrameCache(dest) as cache:
            full = cache.canvas(0)
        for cams in (("main",), tuple(STRIP_ORDER)):
            width = canvas_width(cams)
            # Trainer (train_scene.py): cache canvas, extract_cams when cams != STRIP_ORDER.
            train_hwc = full if tuple(cams) == tuple(STRIP_ORDER) else extract_cams(full, cams)
            live_hwc = SV.view_strip(frames, cams, width)
            assert live_hwc is not None and live_hwc.dtype == np.uint8, cams
            assert live_hwc.shape == train_hwc.shape and live_hwc.tobytes() == train_hwc.tobytes(), cams
            live = live_nchw(live_hwc, width=width)
            assert live is not None and live.dtype == np.float32
            if torch is None:
                skip_check(f"strip_parity_nchw_{len(cams)}cam", "torch is not installed (trainer _strip_nchw)")
                continue
            from python.train.scene_net import _strip_nchw as train_nchw

            want = train_nchw(np.ascontiguousarray(train_hwc), width=width).numpy()
            assert want.dtype == np.float32 and want.shape == live.shape, (want.shape, live.shape)
            assert want.tobytes() == live.tobytes(), (cams, int((want != live).sum()))
            print(f"strip parity {len(cams)}-cam: viewer == trainer bit-for-bit {tuple(live.shape)} float32 OK")


# ---- 3. cannot actuate --------------------------------------------------------------


def _imports(tree: ast.AST) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append(node.module or "")
            out += [f"{node.module}.{a.name}" for a in node.names]
    return out


def check_3_cannot_actuate() -> None:
    from python.control.actuate import DriveCommand, stop_command

    names = [f.name for f in dataclasses.fields(SV.SceneView)]
    bad = [n for n in names if any(w in n.lower() for w in CONTROL_WORDS)]
    assert not bad, ("SceneView carries a control field", bad)
    try:
        SV.SceneView().status = "ok"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("SceneView is not frozen")
    src = (ROOT / "python" / "viz" / "scene_view.py").read_text(encoding="utf-8")
    mods = _imports(ast.parse(src))
    bad = [m for m in mods if m.startswith("python.control") or "engage" in m or "actuat" in m]
    assert not bad, ("scene_view.py imports control / engage / actuator", bad)

    # run_vision: the viewer object only goes into the scene_view helpers, which get frames and
    # timing, never cmd / pout / engage / the actuator; its result only lands in st["scene_view"].
    rv = ast.parse((ROOT / "python" / "run_vision.py").read_text(encoding="utf-8"))
    helpers = {"view_tick", "view_respawn", "view_state"}
    forbidden = {"cmd", "applied", "pout", "engaged", "actuator", "override", "bot_engage", "e2e_policy"}
    calls = []
    for node in ast.walk(rv):
        if isinstance(node, ast.Call):
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if fname in helpers:
                calls.append(fname)
                used = {n.id for a in [*node.args, *[k.value for k in node.keywords]] for n in ast.walk(a)
                        if isinstance(n, ast.Name)}
                assert not used & forbidden, (fname, sorted(used & forbidden))
    assert sorted(set(calls)) == sorted(helpers), ("run_vision must call", sorted(helpers), "got", calls)
    uses = []
    for node in ast.walk(rv):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Name) and child.id == "scene_viewer":
                uses.append(type(node).__name__)
    for node in ast.walk(rv):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "scene_viewer":
            assert node.attr in ("select", "close"), ("run_vision touches scene_viewer." + node.attr)
    assert uses, "run_vision never builds a scene_viewer"
    # view_state(...) is assigned to st["scene_view"] and nothing else.
    for node in ast.walk(rv):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if getattr(fn, "attr", getattr(fn, "id", "")) == "view_state":
                (target,) = node.targets
                assert isinstance(target, ast.Subscript) and getattr(target.value, "id", "") == "st", ast.dump(target)
                assert getattr(target.slice, "value", None) == "scene_view", ast.dump(target)
    # Nothing in run_vision reads st["scene_view"] back.
    for node in ast.walk(rv):
        if isinstance(node, ast.Subscript) and getattr(node.slice, "value", None) == "scene_view":
            assert isinstance(node.ctx, ast.Store), "run_vision reads st['scene_view']"

    # One tick: cmd built, sent, then the viewer; off / on / raising give the same cmd and sends.
    class Rec:
        def __init__(self) -> None:
            self.sent = []

        def apply(self, cmd):
            self.sent.append(("apply", cmd))
            return cmd

        def stop(self, seq=0, reason="stop"):
            c = stop_command(seq=seq, reason=reason)
            self.sent.append(("stop", c))
            return c

    class Raising:
        def __getattr__(self, name):
            raise RuntimeError(f"viewer exploded in {name}")

    rng = np.random.default_rng(3)
    clock = FakeClock()
    on = SV.SceneNetViewer({}, hz=HZ, can_feed_cam_age=False, inline=True, clock=clock)
    results = {}
    for label, viewer in (("off", None), ("on", on), ("raising", Raising())):
        act = Rec()
        cmd = DriveCommand(steer=0.125, throttle=0.25, brake=0.0, seq=7, reason="ok")
        before = dataclasses.asdict(cmd)
        applied = act.apply(cmd)
        clock.tick()
        bundle = _bundle(_frames(rng, ("main",)), clock())
        ms = SV.view_tick(viewer, bundle, backend=None, prev_heartbeat_ms=12.0, loop_gap_s=1.0 / HZ)
        SV.view_respawn(viewer, None, 1.0 / HZ, bundle, None)
        st = {"scene_view": SV.view_state(viewer)}
        json.dumps(st)
        assert isinstance(ms, float)
        assert dataclasses.asdict(cmd) == before and applied is cmd, label
        results[label] = (dataclasses.asdict(applied), [(k, dataclasses.asdict(c)) for k, c in act.sent])
    assert results["off"] == results["on"] == results["raising"], results
    print("cannot actuate: no control fields, frozen, no control imports, run_vision stores st['scene_view'] only, "
          "same cmd with viewer off / on / raising OK")


# ---- 4. budget ------------------------------------------------------------------------


# A RAM folder for the budget runs' files (None: not on this OS, use the default temp folder).
_RAM_DIR = "/dev/shm" if sys.platform.startswith("linux") and os.access("/dev/shm", os.W_OK) else None


@contextlib.contextmanager
def _ram_home(prefix: str):
    """isolated_home() made under _RAM_DIR: the same files and writes, minus disk journal stalls."""
    saved = tempfile.tempdir
    tempfile.tempdir = _RAM_DIR or saved
    try:
        with isolated_home(prefix) as home:
            yield home
    finally:
        tempfile.tempdir = saved


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q)) if xs else float("nan")


def _budget_run(mode: str, seconds: float = 10.0) -> dict:
    import test_unknown_ego as T

    from python.control.actuate import cmd_path, engage_path, write_engage_flag
    from python.runtime.state_io import state_path

    m6 = T._m6()
    write_engage_flag(False)
    for stale in (cmd_path(), state_path()):
        with contextlib.suppress(FileNotFoundError):
            stale.unlink()
    m6._touch_lua_bus()
    mod = T._Mod()
    mod.echoing = True
    tmp = Path(tempfile.mkdtemp(prefix="gvd-view-budget-"))
    log = tmp / "cmd_writes.txt"
    out = tmp / "stdout.txt"
    args = [sys.executable, "-u", str(ROOT / "scripts" / "gvd_slow_view_child.py"), "--backend", "stub",
            "--hz", f"{HZ:g}", "--encode", "cpu", "--allow-preview-drive"]
    if mode != "off":
        args += ["--scene-view", "slow_view.onnx"]
    env = dict(os.environ, PYTHONPATH=str(ROOT), GVD_SLOW_VIEW_MODE=mode, GVD_SLOW_VIEW_LOG=str(log))
    t_spawn = time.time()
    m_spawn = time.monotonic()
    seen = []
    with open(out, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(args, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            while not (cmd_path().exists() and state_path().exists()):
                if proc.poll() is not None:
                    raise AssertionError(f"{mode}: run_vision exited during startup ({proc.returncode})")
                if time.monotonic() > m_spawn + m6.SUPERVISOR_STARTUP_S:
                    raise AssertionError(f"{mode}: no gvd_cmd.json / gvd_state.json in startup")
                m6._touch_lua_bus()
                mod.heartbeat()
                mod.echo(t_spawn)
                time.sleep(0.01)
            m0 = time.monotonic()
            engaged_at = None
            while (t := time.monotonic() - m0) < 0.5 + seconds:
                if proc.poll() is not None:
                    raise AssertionError(f"{mode}: run_vision exited early ({proc.returncode})")
                m6._touch_lua_bus()
                if engaged_at is None and t >= 0.5:
                    engaged_at = time.monotonic()
                    write_engage_flag(True)
                mod.heartbeat()
                mod.echo(t_spawn)
                seen.append({"t": t, "engage": T._read(engage_path()), "cmd": T._read(cmd_path()),
                             "state": T._read(state_path())})
                time.sleep(0.01)
        finally:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
    segs = []
    for line in out.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[GVD] seg "):
            kv = dict(p.split("=", 1) for p in line[len("[GVD] seg "):].split() if "=" in p)
            segs.append(kv)
    writes = [float(x) for x in log.read_text(encoding="utf-8").split()] if log.exists() else []
    return {"seen": seen, "segs": segs, "writes": writes, "engaged_at": engaged_at, "stdout": out}


def _budget_stats(run: dict, mode: str) -> dict:
    segs = run["segs"]
    # Ticks after the engage edge settled (skip the first 1 s: startup, model load).
    warm = segs[int(HZ):] if len(segs) > int(HZ) else segs
    hb = [float(s["heartbeat_ms"]) for s in warm if "heartbeat_ms" in s]
    vms = [float(s.get("view_ms", "0")) for s in warm]
    eng = [s for s in run["seen"] if s["state"].get("engaged") is True or s["cmd"].get("engaged") is True]
    t_eng = min((s["t"] for s in eng), default=None)
    reasons = sorted({str(s["state"].get("disengage_reason")) for s in run["seen"] if t_eng is not None and s["t"] > t_eng})
    after = [s for s in run["seen"] if t_eng is not None and s["t"] > t_eng]
    dis = [s for s in after if s["state"].get("engaged") is False and s["state"].get("disengage_reason")]
    stale = [s for s in run["seen"] if "heartbeat_stale" in json.dumps([s["state"].get("disengage_reason"),
                                                                       s["cmd"].get("reason")])]
    w = run["writes"]
    w_eng = [x for x in w if run["engaged_at"] is not None and x >= run["engaged_at"] + 1.0]
    gaps = [b - a for a, b in zip(w_eng, w_eng[1:])]
    last = segs[-1] if segs else {}
    return {
        "mode": mode, "ticks": len(segs), "hb_p50": _pct(hb, 50), "hb_p99": _pct(hb, 99), "hb_max": max(hb or [0]),
        "view_ms_max": max(vms or [0.0]), "view_ms_p99": _pct(vms, 99), "cmd_gap_max_ms": max(gaps or [0.0]) * 1000.0,
        "cmd_writes": len(w_eng), "engaged": t_eng is not None, "disengages": len(dis), "heartbeat_stale": len(stale),
        "reasons": reasons, "view": last.get("view"), "dropped": int(last.get("view_dropped", "0") or 0),
        "views": sorted({s.get("view", "?") for s in warm}),
    }


def check_4_budget() -> None:
    stats = {}
    with _ram_home("gvd-view-budget-") as home:
        # Every tick writes gvd_sensors.json / gvd_cmd.json / gvd_state.json into this home. On a
        # shared disk (ext4 /tmp) about one open() a second stalled 13-81 ms, in every mode, off
        # included: disk noise, not the loop or the viewer. Same files and writes, in RAM.
        fs = _fs_type(home)
        assert fs == "tmpfs" or not _RAM_DIR, f"budget home on {fs} (disk), not tmpfs {_RAM_DIR}"
        for mode in ("off", "sleep", "spin"):
            stats[mode] = _budget_stats(_budget_run(mode), mode)
            s = stats[mode]
            print(f"budget {mode}: ticks={s['ticks']} heartbeat p50={s['hb_p50']:.2f} p99={s['hb_p99']:.2f} "
                  f"max={s['hb_max']:.2f} ms view_ms max={s['view_ms_max']:.3f} p99={s['view_ms_p99']:.3f} "
                  f"cmd gap max={s['cmd_gap_max_ms']:.1f} ms ({s['cmd_writes']} writes) engaged={s['engaged']} "
                  f"disengages={s['disengages']} heartbeat_stale={s['heartbeat_stale']} view={s['view']} "
                  f"views={s['views']} dropped={s['dropped']}", flush=True)
    problems = []
    off = stats["off"]
    for mode in ("off", "sleep", "spin"):
        s = stats[mode]
        if not s["engaged"]:
            problems.append(f"{mode}: never engaged")
        if s["disengages"] or s["heartbeat_stale"]:
            problems.append(f"{mode}: disengages={s['disengages']} heartbeat_stale={s['heartbeat_stale']} {s['reasons']}")
        if not s["cmd_gap_max_ms"] < 200.0 or s["cmd_writes"] < 10:
            problems.append(f"{mode}: cmd write gap {s['cmd_gap_max_ms']:.1f} ms over {s['cmd_writes']} writes")
        if not s["hb_p99"] <= P_MS:
            problems.append(f"{mode}: heartbeat p99 {s['hb_p99']:.2f} > P {P_MS:.1f} ms")
        if mode == "off":
            continue
        if not s["view_ms_max"] <= 0.10 * P_MS:
            problems.append(f"{mode}: view_ms max {s['view_ms_max']:.3f} > {0.10 * P_MS:.2f} ms")
        if not s["hb_p99"] <= off["hb_p99"] + 5.0:
            problems.append(f"{mode}: heartbeat p99 {s['hb_p99']:.2f} > off {off['hb_p99']:.2f} + 5 ms")
        if not s["dropped"] > 0:
            problems.append(f"{mode}: dropped={s['dropped']}")
        if "slow" not in s["views"]:
            problems.append(f"{mode}: badge 'slow' never shown ({s['views']})")
    assert not problems, " | ".join(problems)
    print("budget: viewer off / 400 ms sleep / 400 ms GIL spin within the tick budget OK")


# ---- 5. live hold invisible to perception -----------------------------------------------


class _VaryCam:
    """beamngpy Camera stand-in: stream_raw shared memory, a different picture every read."""

    def __init__(self, name: str, seed: int) -> None:
        self.name = name
        self.n = 0
        self.seed = seed

    def stream_raw(self) -> dict:
        self.n += 1
        rng = np.random.default_rng(self.seed * 1000 + self.n)
        pix = (rng.random((48, 64, 4)) * 255).astype(np.uint8)
        pix[..., 3] = 255
        return {"colour": pix}


def _hold_backend(seed_base: int):
    from python.sensors.cameras import BeamNGPyBackend

    be = BeamNGPyBackend()
    be._ok = True
    be._sensors = {cam: _VaryCam(cam, k + seed_base) for k, cam in enumerate(STRIP_ORDER)}
    return be


def check_5_live_hold_invisible() -> None:
    from python.perception.stitch360 import stitch_frames
    from python.sensors.cameras import bundle_tick_frames
    from python.sensors.live_hold import scrub_after_respawn

    plain = _hold_backend(0)
    held = _hold_backend(0)
    held.enable_live_hold()
    n_held = 0
    for i in range(60):
        a = plain.grab()
        b = held.grab()
        if i == 30:
            held.reset_live_hold()
            scrub_after_respawn(b)
        fa, fb = bundle_tick_frames(a), bundle_tick_frames(b)
        assert sorted(fa) == sorted(fb), (i, sorted(fa), sorted(fb))
        assert fa, i
        for k in fa:
            assert np.array_equal(fa[k], fb[k]), (i, k)
        sa, sb = stitch_frames(fa), stitch_frames(fb)
        assert sa.bgr.tobytes() == sb.bgr.tobytes(), i
        n_held += int(getattr(b, "cam_ages", None) is not None)
    assert n_held == 60 and getattr(a, "cam_ages", None) is None, n_held
    print("live hold: 60 grabs, reset + scrub at grab 30: bundle_tick_frames and stitch identical OK")


# ---- 6. failure + catalog ---------------------------------------------------------------


def check_6_failure_and_catalog() -> None:
    from python.train.continual import CHUNK_MAX
    from python.train.scene_net import MEMORY_S

    assert SV.CHUNK == CHUNK_MAX and SV.GAP_RESET_S == float(MEMORY_S), (SV.CHUNK, CHUNK_MAX, SV.GAP_RESET_S, MEMORY_S)
    clock = FakeClock()
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "nope.onnx"
        viewer = SV.SceneNetViewer({"nope": missing}, hz=HZ, can_feed_cam_age=False, inline=True, clock=clock)
        viewer.select("nope")
        clock.tick()
        viewer.submit(_bundle(_frames(rng, ("main",)), clock()), backend=None, prev_heartbeat_ms=5.0, loop_gap_s=0.066)
        v = viewer.latest()
        assert v.status == "refused" and "missing" in v.badge, v
        print(f"missing file: status={v.status} badge={v.badge!r} OK")
        if torch is None:
            skip_check("wrong_width_and_lane71", "torch is not installed (tiny SceneNet export)")
        else:
            lane71 = _export(Path(tmp) / "lane71.onnx", cam_age=True)
            v1 = _export(Path(tmp) / "v1.onnx", cams=("main",))
            viewer = SV.SceneNetViewer({"lane71": lane71, "v1": v1}, hz=HZ, can_feed_cam_age=False,
                                       inline=True, clock=clock)
            viewer.select("lane71")
            v = viewer.latest()
            assert v.status == "refused" and v.badge == SV.LANE71_BADGE, v
            print(f"lane71 without cam ages: badge={v.badge!r} OK")
            # Wrong strip width at load: a main-only graph whose metadata claims 8 cams.
            import onnx

            model = onnx.load(str(v1))
            for p in model.metadata_props:
                if p.key == "gvd_cams":
                    p.value = ",".join(STRIP_ORDER)
                if p.key == "gvd_canvas_w":
                    p.value = str(CANVAS_W)
            wrong = Path(tmp) / "wrong.onnx"
            onnx.save(model, str(wrong))
            viewer = SV.SceneNetViewer({"wrong": wrong, "v1": v1}, hz=HZ, can_feed_cam_age=False,
                                       inline=True, clock=clock)
            viewer.select("wrong")
            v = viewer.latest()
            assert v.status == "refused" and "width" in v.badge, v
            print(f"wrong strip width: badge={v.badge!r} OK")
    # A forward that raises: an error badge each time, no exception; three latch the viewer off.
    class Boom:
        cams = ("main",)
        canvas_w = canvas_width(("main",))
        cam_age = False
        ready = True

        def __init__(self, *_a, **_k) -> None:
            self.last_timers = {}

        def observe(self, strip, dt_s, ages=None):
            raise RuntimeError("boom")

        def reset_state(self) -> None:
            pass

    viewer = SV.SceneNetViewer({"boom": Path("boom.onnx")}, hz=HZ, can_feed_cam_age=False,
                               runtime_factory=Boom, inline=True, clock=clock)
    viewer.select("boom")
    for n in range(1, 5):
        clock.tick()
        viewer.submit(_bundle(_frames(rng, ("main",)), clock()), backend=None, prev_heartbeat_ms=5.0, loop_gap_s=0.066)
        v = viewer.latest()
        assert v.status == "error" and v.badge.startswith("error: RuntimeError"), (n, v)
        assert ("latched" in v.badge) == (n >= 3), (n, v)
    assert v.steps == 3, v  # latched off: the 4th tick submits nothing
    print(f"forward raising x3: status={v.status} badge={v.badge!r} OK")
    from python.runtime import models

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "e2e_current.onnx").write_bytes(b"x")
        (d / "e2e_scene.onnx").write_bytes(b"x")
        saved = models.MODELS_DIR
        models.MODELS_DIR = d
        try:
            ids = [c.id for c in models.e2e_choices()]
        finally:
            models.MODELS_DIR = saved
        assert "e2e_current" in ids and "e2e_scene" not in ids, ids
        models.set_scene_view_ids(["v103_pause_s12348", "lane71"])
        try:
            assert models.ids_for("scene_view") == ["off", "v103_pause_s12348", "lane71"]
            assert models.cycle_id("scene_view", "off", 1) == "v103_pause_s12348"
        finally:
            models.set_scene_view_ids([])
        assert models.ids_for("scene_view") == ["off"]
    from python.runtime.debug_opts import MODEL_CONTROL_ROWS, DebugOpts

    row = next(r for r in MODEL_CONTROL_ROWS if r["id"] == "scene_view_id")
    assert row["role"] == "scene_view" and "view-only" in row["hint"], row
    assert DebugOpts().scene_view_id == "off" and DebugOpts().as_dict()["scene_view_id"] == "off"
    print(f"catalog: e2e_choices {ids} (no e2e_scene); MODEL row scene_view_id off + CLI stems OK")


# ---- 7. model file untouched ------------------------------------------------------------


def _fingerprint(p: Path) -> tuple[str, int, int]:
    st = p.stat()
    return hashlib.sha256(p.read_bytes()).hexdigest(), st.st_mtime_ns, st.st_size


def check_7_model_untouched() -> None:
    if torch is None:
        skip_check("model_untouched", "torch is not installed (tiny SceneNet export)")
        return
    rng = np.random.default_rng(9)
    with tempfile.TemporaryDirectory() as tmp:
        a = _export(Path(tmp) / "a.onnx", cams=("main",))
        b = _export(Path(tmp) / "b.onnx", cams=("main",))
        before = {p: _fingerprint(p) for p in (a, b)}
        listing = sorted(os.listdir(tmp))
        clock = FakeClock()
        viewer = SV.SceneNetViewer({"a": a, "b": b}, hz=HZ, can_feed_cam_age=False, inline=True, clock=clock)
        for stem in ("a", "b", "a"):
            viewer.select(stem)
            for _ in range(3):
                clock.tick()
                viewer.submit(_bundle(_frames(rng, ("main",)), clock()), backend=None,
                              prev_heartbeat_ms=5.0, loop_gap_s=0.066)
            viewer.runtime.reset_state()
        viewer.select("off")
        viewer.close()
        after = {p: _fingerprint(p) for p in (a, b)}
        assert before == after, (before, after)
        assert sorted(os.listdir(tmp)) == listing, (listing, sorted(os.listdir(tmp)))
    print("model files: sha256, mtime and folder listing unchanged after load, switch, reset OK")


# ---- 8. the child process never blocks the tick ------------------------------------------


SLOW_FACTORY = "gvd_slow_view_child:SlowRuntime"
SETUP_COST_S = 0.030  # > 0.10 P: a one-time first-step cost must not seed the throttle


def _wait_status(viewer, ok, timeout_s: float = 30.0, *, rng=None, seen=None):
    """Submit frames at HZ until ``ok(view)``; returns the view. Records every view in ``seen``."""
    rng = rng if rng is not None else np.random.default_rng(0)
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        viewer.submit(_bundle(_frames(rng, ("main",)), time.time()), backend=None,
                      prev_heartbeat_ms=5.0, loop_gap_s=1.0 / HZ)
        v = viewer.latest()
        if seen is not None:
            seen.append(v)
        if ok(v):
            return v
        time.sleep(1.0 / HZ)
    raise AssertionError(f"timed out after {timeout_s} s: {viewer.latest()}")


def check_8_child_never_blocks() -> None:
    """The real child process: a slow fake forward never blocks submit; close() <= 0.6 s."""
    rng = np.random.default_rng(11)
    viewer = SV.SceneNetViewer({"slow": Path("slow.onnx")}, hz=HZ, can_feed_cam_age=False,
                               runtime_factory=(SLOW_FACTORY, {"forward_s": 0.2, "mode": "sleep"}))
    real_shm_for = viewer._shm_for

    def costly_first_shm(nbytes):  # one-time setup (first segment, lazy imports) as on a busy CI runner
        if viewer._shm is None:
            time.sleep(SETUP_COST_S)
        return real_shm_for(nbytes)

    viewer._shm_for = costly_first_shm
    viewer.select("slow")
    seen: list = []
    _wait_status(viewer, lambda v: v.status not in ("loading",), rng=rng, seen=seen)
    worst = 0.0
    t_end = time.monotonic() + 1.5
    while time.monotonic() < t_end:
        ms = viewer.submit(_bundle(_frames(rng, ("main",)), time.time()), backend=None,
                           prev_heartbeat_ms=5.0, loop_gap_s=0.066)
        worst = max(worst, ms)
        seen.append(viewer.latest())
        time.sleep(1.0 / HZ)
    v = viewer.latest()
    assert "throttled" not in {s.status for s in seen}, [s.status for s in seen]
    real_submit = viewer._submit

    def costly_submit(*a, **kw):  # a sustained main-thread cost still throttles
        time.sleep(0.010)
        return real_submit(*a, **kw)

    viewer._submit = costly_submit
    for _ in range(8):
        viewer.submit(_bundle(_frames(rng, ("main",)), time.time()), backend=None,
                      prev_heartbeat_ms=5.0, loop_gap_s=0.066)
    assert viewer.latest().status == "throttled", viewer.latest()
    proc, shm_name = viewer.child, viewer.shm_name
    assert proc is not None and proc.is_alive() and shm_name, (proc, shm_name)
    t0 = time.monotonic()
    viewer.close()
    joined = time.monotonic() - t0
    assert worst <= 0.10 * P_MS, worst
    assert v.dropped > 0 and v.status == "slow" and v.badge == "slow", v
    assert v.child_rss_mb > 0.0, v
    assert joined <= 0.6, joined
    assert not proc.is_alive() and proc.exitcode is not None, proc
    if sys.platform.startswith("linux"):
        assert not Path("/dev/shm", shm_name.lstrip("/")).exists(), shm_name
    print(f"child: worst submit {worst:.3f} ms, dropped={v.dropped}, badge {v.badge!r}, "
          f"child rss {v.child_rss_mb:.0f} MB, close {joined:.2f} s (exit {proc.exitcode}, shm gone) OK")


# ---- 9. the main process never imports torch / onnxruntime --------------------------------

_NO_TORCH_CHILD = r"""
import json, sys, time
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/scripts"]
import numpy as np
from types import SimpleNamespace
from python.viz import scene_view as SV

HEAVY = ("torch", "onnxruntime")
def heavy():
    return sorted(m for m in sys.modules if m.split(".")[0] in HEAVY)
a, b = Path(sys.argv[2]), Path(sys.argv[3])
rng = np.random.default_rng(3)
viewer = SV.SceneNetViewer({"a": a, "b": b}, hz=15.0, can_feed_cam_age=False)
out = {"after_import": heavy()}
def bundle():
    return SimpleNamespace(frames={"main": rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)},
                           timestamps={"main": time.time()}, main_bgr=None)
def run_until_ok(stem):
    seen = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 60.0:
        viewer.submit(bundle(), backend=None, prev_heartbeat_ms=5.0, loop_gap_s=1 / 15)
        v = viewer.latest()
        seen.append([v.stem, v.status, v.badge, len(v.lane_gt)])
        if v.status == "ok" and v.stem == stem:
            return time.monotonic() - t0, seen
        time.sleep(1 / 15)
    raise SystemExit(f"timeout {stem}: {viewer.latest()}")
viewer.select("a"); out["after_select"] = heavy()
out["load_a_s"], _ = run_until_ok("a"); out["after_step"] = heavy()
for _ in range(10):
    viewer.submit(bundle(), backend=None, prev_heartbeat_ms=5.0, loop_gap_s=1 / 15); time.sleep(1 / 15)
viewer.select("b"); out["after_switch"] = heavy()
out["load_b_s"], out["seen_b"] = run_until_ok("b"); out["after_switch_step"] = heavy()
v = viewer.latest()
out["rss_mb"] = v.child_rss_mb
out["child_heavy"] = viewer.child_modules(HEAVY)
viewer.close()
print("RESULT " + json.dumps(out))
"""


def check_9_main_never_imports_torch() -> None:
    if torch is None:
        skip_check("main_never_imports_torch", "torch is not installed (tiny SceneNet export)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        a = _export(Path(tmp) / "a.onnx", cams=("main",))
        b = _export(Path(tmp) / "b.onnx", cams=("main",))
        script = Path(tmp) / "no_torch_child.py"
        script.write_text(_NO_TORCH_CHILD, encoding="utf-8")
        proc = subprocess.run([sys.executable, str(script), str(ROOT), str(a), str(b)], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=180)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
    out = json.loads(lines[-1][len("RESULT "):])
    for key in ("after_import", "after_select", "after_step", "after_switch", "after_switch_step"):
        assert out[key] == [], (key, out[key])
    assert "torch" in out["child_heavy"] and "onnxruntime" in out["child_heavy"], out["child_heavy"]
    # The switch: "loading", no lane lines, until b's first result.
    for stem, status, badge, n in out["seen_b"]:
        assert stem == "b", (stem, status)
        if status != "ok":
            assert status == "loading" and badge == "loading" and n == 0, (stem, status, badge, n)
    assert out["rss_mb"] > 0.0, out["rss_mb"]
    print(f"main process: no torch / onnxruntime after import, select, step, switch, step "
          f"(child has {out['child_heavy']}); load a {out['load_a_s']:.2f} s, switch to b "
          f"{out['load_b_s']:.2f} s shown as 'loading'; child rss {out['rss_mb']:.0f} MB OK")


# ---- 10. no orphans ---------------------------------------------------------------------

_ORPHAN_PARENT = r"""
import sys, time
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/scripts"]
import numpy as np
from types import SimpleNamespace
from python.viz import scene_view as SV
viewer = SV.SceneNetViewer({"slow": Path("slow.onnx")}, hz=15.0, can_feed_cam_age=False,
                           runtime_factory=("gvd_slow_view_child:SlowRuntime", {"forward_s": 0.4, "mode": "sleep"}))
viewer.select("slow")
rng = np.random.default_rng(1)
t0 = time.monotonic()
while time.monotonic() - t0 < 60.0:
    viewer.submit(SimpleNamespace(frames={"main": rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)},
                                  timestamps={"main": time.time()}, main_bgr=None),
                  backend=None, prev_heartbeat_ms=5.0, loop_gap_s=1 / 15)
    if viewer.latest().status != "loading":
        print(f"CHILD {viewer.child.pid}", flush=True)
        break
    time.sleep(1 / 15)
time.sleep(60.0)
"""

ORPHAN_BOUND_S = 3.0


def _pid_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    with contextlib.suppress(OSError):
        return stat.read_text().split(") ", 1)[1].startswith("Z")  # a zombie has exited
    return False


def check_10_no_orphans() -> None:
    if os.name != "posix":
        skip_check("no_orphans_sigkill", "SIGKILL test is POSIX; Windows relies on the parent-pid watchdog + daemon")
        return
    import signal

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "orphan_parent.py"
        script.write_text(_ORPHAN_PARENT, encoding="utf-8")
        parent = subprocess.Popen([sys.executable, "-u", str(script), str(ROOT)], cwd=str(ROOT),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        child_pid = None
        try:
            t_end = time.monotonic() + 60.0
            while time.monotonic() < t_end and child_pid is None:
                line = parent.stdout.readline()
                if not line:
                    break
                if line.startswith("CHILD "):
                    child_pid = int(line.split()[1])
            assert child_pid is not None, "the viewer child never came up"
            assert not _pid_gone(child_pid), child_pid
            parent.send_signal(signal.SIGKILL)
            parent.wait(timeout=5)
            t0 = time.monotonic()
            while not _pid_gone(child_pid) and time.monotonic() - t0 < ORPHAN_BOUND_S:
                time.sleep(0.02)
            gone_s = time.monotonic() - t0
        finally:
            with contextlib.suppress(Exception):
                parent.kill()
            if child_pid is not None and not _pid_gone(child_pid):
                with contextlib.suppress(Exception):
                    os.kill(child_pid, signal.SIGKILL)
    assert _pid_gone(child_pid), f"child {child_pid} outlived a SIGKILLed parent by {ORPHAN_BOUND_S} s"
    print(f"no orphans: parent SIGKILLed, viewer child gone in {gone_s:.2f} s (bound {ORPHAN_BOUND_S} s) OK")


# ---- 11. a switch shows "loading", never a stale overlay ---------------------------------


def check_11_switch_shows_loading() -> None:
    rng = np.random.default_rng(12)
    viewer = SV.SceneNetViewer(
        {"a": Path("a.onnx"), "b": Path("b.onnx")}, hz=HZ, can_feed_cam_age=False,
        runtime_factory=(SLOW_FACTORY, {"forward_s": 0.01, "load_s": 1.0, "mode": "sleep", "fill": "stem"}))
    try:
        viewer.select("a")
        first = viewer.latest()
        assert first.status == "loading" and first.badge == "loading" and first.lane_gt == (), first
        va = _wait_status(viewer, lambda v: v.status == "ok", rng=rng)
        assert va.stem == "a" and va.lane_gt[0] == 1.0, va  # the fake fills lane_gt with 1.0 for "a"
        viewer.select("b")
        seen: list = []
        t0 = time.monotonic()
        vb = _wait_status(viewer, lambda v: v.status == "ok", rng=rng, seen=seen)
        waited = time.monotonic() - t0
        state = SV.view_state(viewer)
    finally:
        viewer.close()
    assert waited >= 0.9, waited  # the fake's 1 s load really was on screen
    loading = [v for v in seen if v.status != "ok"]
    assert loading, seen
    for v in loading:
        assert v.stem == "b" and v.status == "loading" and v.badge == "loading" and v.lane_gt == (), v
    assert vb.stem == "b" and vb.lane_gt[0] == 2.0, vb
    assert state["status"] == "ok" and state["stem"] == "b"
    # The stage draws no lines while loading.
    import cv2  # noqa: F401

    img = np.zeros((240, 320, 3), np.uint8)
    cam = SimpleNamespace(project=lambda x, y, z: (160 + int(x * 10), 200 - int(y * 5)))
    SV.draw_scene_view(img, cam, dataclasses.asdict(loading[0]) | {"lane_gt": list(va.lane_gt)}, origin=(0, 300))
    assert int(img.sum()) == 0, "lane lines drawn while loading"
    print(f"switch: {len(loading)} ticks of 'loading' with no lane lines over {waited:.2f} s, "
          f"then b's own result OK")


# ---- report-only ratio metric --------------------------------------------------------------


def ratio_report(pairs: int = 5) -> None:
    """Median over ``pairs`` alternating on/off runs of on p99 / off p99 (report only, <= 1.5)."""
    import statistics

    with _ram_home("gvd-view-ratio-"):
        for mode in ("sleep", "spin"):
            ratios = []
            for i in range(pairs):
                off = _budget_stats(_budget_run("off"), "off")
                on = _budget_stats(_budget_run(mode), mode)
                r = on["hb_p99"] / max(1e-6, off["hb_p99"])
                ratios.append(r)
                print(f"ratio {mode} pair {i + 1}: on p99 {on['hb_p99']:.2f} / off p99 {off['hb_p99']:.2f} "
                      f"= {r:.2f}", flush=True)
            med = statistics.median(ratios)
            print(f"ratio {mode}: median {med:.2f} over {pairs} pairs "
                  f"({'<=' if med <= 1.5 else '>'} 1.5, report only)", flush=True)


CHECKS = [
    check_1_forward_both_contracts,
    check_2_strip_parity,
    check_3_cannot_actuate,
    check_5_live_hold_invisible,
    check_6_failure_and_catalog,
    check_7_model_untouched,
    check_8_child_never_blocks,
    check_9_main_never_imports_torch,
    check_10_no_orphans,
    check_11_switch_shows_loading,
    check_4_budget,
]


def main() -> int:
    failed = 0
    for check in CHECKS:
        try:
            check()
        except Exception:
            failed += 1
            print(f"FAIL {check.__name__}")
            traceback.print_exc()
    print(f"test_scene_view: {len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if "--ratio" in sys.argv[1:]:
        ratio_report()
        raise SystemExit(0)
    raise SystemExit(main())
