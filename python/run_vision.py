"""GVD supervisor — cameras + perception + actuation + M4 clips + M5 shadow/E2E + gvd_state.json."""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.control.actuate import (
    attach_electrics,
    curve_speed_mps,
    curve_steer,
    follow_lane_steer,
    lane_sample_hold_reason,
    LaneSteerState,
    last_electrics_ms,
    make_actuator,
    read_ego_feedback,
    read_electrics_inputs,
    BotEngage,
    read_engage_flag,
    soft_esc_state_write_due,
    soft_esc_state_write_mark,
    soft_esc_state_write_skips,
    steer_sample_point,
    release_command,
    stop_command,
    write_engage_flag,
)
from python.control.ego_speed import UNKNOWN_EGO_REASONS, engage_gate, resolve_ego
from python.sensors.tech import (
    VehicleData,
    apply_nav_missing,
    nav_heading_from_sources,
    nav_snapshot,
    path_ego_to_world,
    run_probe,
)
from python.sensors.extras import ExtraSensors, load_sensors_config, nav_hint_from_bundle, write_sensors_snapshot
from python.viz.foxglove_bridge import FoxgloveBridge
from python.control.e2e import make_e2e
from python.control.override import (
    OverrideDetector,
    config_mirror,
    load_override_config,
)
from python.control.echo_link import MIN_MOD_VERSION, STATUS_OUTDATED, EchoGate, SessionEcho, outdated_message
from python.data.record import ClipRecorder, choose_encoder
from python.perception.pipeline import ModularPerception
from python.perception.stitch360 import stitch_frames
from python.perception.road_model import lanes_ext_for_live, road_edges
from python.viz.debug_draw import cabin_drive_word
from python.control.launch_guard import (
    LaunchGuard,
    apply_lane_exit_brake,
    longitudinal_mps,
    moving_backward,
    apply_stale_speed_cap,
)
from python.debug.gt_geom import heading_from_dir
from python.debug.gt_logger import GroundTruthLogger, gt_log_enabled
from python.viz.review_log import ReviewCapture
from python.viz.nerd import image_point_from_window_mouse
# scene_live, not scene_net / scene_lane_runtime: those import torch at load (~0.8 s), and
# the supervisor only needs torch-free helpers here. The scene runtime is imported lazily below.
from python.train.scene_live import SCENE_SHAPE_HOLD, live_strip_fields
from python.viz.stage import drawn_lane_records
from python.sensors.cameras import CAM_IDS, CamHealth, bundle_tick_frames
from python.runtime.debug_opts import apply_to_command, apply_to_perception
from python.runtime.hw_probe import ema_hz, gpu_vram_used_gb, probe, refuse_live_start, unique_frame_hz_inst
from python.runtime.models import ModelRuntime
from python.runtime.shadow import ShadowConfig, load_shadow_config, shadow_tick
from python.runtime.state_io import (
    default_state,
    load_paint_state,
    print_py_bus,
    read_state,
    state_path,
    steer_preview_path_ego,
    write_state,
)
from python.runtime.paths import bus_identity
from python.sensors.cameras import camera_ge_socket_busy, make_backend, resolve_backend_name
from python.sensors.live_hold import (
    RespawnWatch,
    StaleEgo,
    ages_text,
    bundle_ages,
    main_exposure,
    reads_text,
    scrub_after_respawn,
    uniq_text,
)
from python.data.cam_hold import CAM_AGE_ORDER
from python.viz.monitors import place_opencv_window
from python.viz.stage import STAGE_W, STAGE_H, VizUI, render_stage, smoke



def _load_control_yaml() -> dict:
    try:
        import yaml  # type: ignore

        p = ROOT / "config" / "control.yaml"
        if p.is_file():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return {}


def _rss_mb() -> float:
    try:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        try:
            import psutil  # type: ignore

            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0



def _path_world_from_vehicle(path_ego: list, vehicle, vdata: VehicleData | None = None) -> list[dict[str, float]] | None:
    """Kinematics-only world path from Tech pose (honesty: not a map)."""
    if vdata is not None and vdata.pose_ok:
        return path_ego_to_world(path_ego, vdata.pos, vdata.dir, vdata.up)
    if not path_ego or vehicle is None:
        return None
    try:
        pos = None
        fwd = None
        up = None
        if hasattr(vehicle, "state") and isinstance(vehicle.state, dict):
            pos = vehicle.state.get("pos")
            fwd = vehicle.state.get("dir") or vehicle.state.get("forward")
            up = vehicle.state.get("up")
        if pos is None and hasattr(vehicle, "get_position"):
            pos = vehicle.get_position()
        if pos is None or fwd is None:
            return None
        p = (float(pos[0]), float(pos[1]), float(pos[2]))
        d = (float(fwd[0]), float(fwd[1]), float(fwd[2]))
        u = (float(up[0]), float(up[1]), float(up[2])) if up is not None else (0.0, 0.0, 1.0)
        return path_ego_to_world(path_ego, p, d, u)
    except Exception:
        return None



def _ask_strip_folder() -> str | None:
    """Pick a strip folder. Cancel leaves the current folder in place."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    try:
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        chosen = filedialog.askdirectory(title="GVD training strips")
    except Exception:
        return None
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    text = str(chosen or "").strip()
    return text or None


def _read_ui_prefs() -> dict:
    """Documents/GVD/gvd_ui_prefs.json — UI toggles; honor over forced defaults."""
    try:
        from python.runtime.state_io import gvd_docs_dir
        import json
        p = gvd_docs_dir() / "gvd_ui_prefs.json"
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _forecast_agents(tracks: list, max_agents: int = 6, stride: int = 3) -> list[dict]:
    """Mode-0 constant-yaw-rate fans for the in-game scene (same toy math as the OpenCV view)."""
    if not tracks:
        return []
    try:
        from python.viz.forecast import predict_modes
    except Exception:
        return []
    out: list[dict] = []
    for tr in tracks[:max_agents]:
        try:
            if float(tr.get("speed_mps", 0.0)) < 0.5:
                continue  # a parked car's fan is just noise on screen
            modes = predict_modes(tr, max_modes=1)
            if not modes:
                continue
            pts = modes[0]["points"][::stride]
            if len(pts) < 2:
                continue
            out.append({
                "id": tr.get("id"),
                "path_ego": [{"x": round(float(px), 2), "y": round(float(py), 2)} for px, py in pts],
            })
        except Exception:
            continue
    return out


def _ui_request_is_live(prefs: dict, start_unix: float) -> bool:
    """In-game policy / screen picks apply to the running session only, so a pref left
    over from a past session never silently overrides --policy at launch."""
    try:
        return float(prefs.get("mtime") or 0) >= start_unix - 1.5
    except Exception:
        return False


def _ms_text(value: Any) -> str:
    return "na" if value is None else f"{float(value):.3f}"


def detector_runtime(detector: Any) -> str:
    """``name:providers`` for an onnxruntime detector, ``name:device`` for Ultralytics, else the name."""
    name = str(getattr(detector, "name", type(detector).__name__))
    session = getattr(detector, "session", None)
    if session is not None and hasattr(session, "get_providers"):
        try:
            return f"{name}:{'+'.join(session.get_providers())}"
        except Exception:
            return f"{name}:na"
    model = getattr(detector, "model", None)
    if model is not None:
        device = getattr(model, "device", None)
        if device is None:
            inner = getattr(model, "model", None)
            try:
                device = next(inner.parameters()).device if inner is not None else None
            except Exception:
                device = None
        return f"{name}:{device if device is not None else 'na'}"
    return name


def main(clock=None) -> None:
    """`clock` is test-only (scripts/gvd_clock_child.py). None keeps every stdlib clock call."""
    ap = argparse.ArgumentParser(description="GVD supervisor + M2/M3/M4/M5")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--backend", default="auto", choices=["auto", "beamngpy", "window", "stub"])
    ap.add_argument("--vision-only", action="store_true")
    ap.add_argument(
        "--allow-synthetic-detect",
        action="store_true",
        help="Allow synthetic detections without YOLO weights (default: smoke only)",
    )
    ap.add_argument(
        "--detector",
        default="auto",
        help="Detector catalog id: auto (default), synthetic, empty, a yolox*.onnx id if present, or a yolov8*-onnx/ultra file in models/",
    )
    ap.add_argument(
        "--e2e-model",
        default="auto",
        dest="e2e_model",
        help="E2E catalog id: auto (default), stub, or an e2e*.onnx stem in models/",
    )
    ap.add_argument(
        "--allow-preview-drive",
        action="store_true",
        help="Allow actuation when path_debug_preview=true (default: blocked)",
    )
    ap.add_argument(
        "--force-engage",
        action="store_true",
        help="Dev only: treat as engaged without Alt+G gvd_engage.json (never default)",
    )
    _ctrl_yaml = _load_control_yaml()
    _policy_default = str(_ctrl_yaml.get("policy_default") or "modular").lower()
    if _policy_default not in ("modular", "e2e", "shadow"):
        _policy_default = "modular"
    ap.add_argument(
        "--policy",
        default=_policy_default,
        choices=["modular", "e2e", "shadow"],
        help="M5: modular (default supervisor) | e2e | shadow (compute both; actuate modular unless e2e). "
        "Default from config/control.yaml policy_default.",
    )
    ap.add_argument(
        "--encode",
        default="auto",
        choices=["auto", "qsv", "cpu", "nvenc"],
        help="Clip encode: auto prefers h264_qsv then libx264; nvenc only with --encode nvenc",
    )
    ap.add_argument(
        "--viz-screen",
        default="auto",
        help="GVD VISION monitor: auto|1|2 (or GVD_VIZ_MONITOR). auto→non-primary if present",
    )
    ap.add_argument("--viz-fullscreen", action="store_true", help="Fullscreen GVD VISION on chosen monitor")
    ap.add_argument(
        "--tech-probe",
        action="store_true",
        help="Connect to BeamNG.tech, poll vehicle electrics/pose/damage, exit. No fake cameras.",
    )
    ap.add_argument(
        "--tech-hold",
        action="store_true",
        help="Tech hold preflight only: port LISTENING, mod, vehicle, fresh lua_bus, buses_same. No cameras, no Hz.",
    )
    ap.add_argument(
        "--foxglove",
        action="store_true",
        help="Publish extras to a local Foxglove WebSocket (ws://127.0.0.1:8765). Planner stays vision-only.",
    )
    ap.add_argument(
        "--gt-log",
        action="store_true",
        help="Sim ground truth JSONL under logs/ (or set GVD_GT_LOG=1). Off in normal play.",
    )
    ap.add_argument(
        "--lane-source",
        default=os.environ.get("GVD_LANE_SOURCE", "hough"),
        choices=["hough", "scene", "gt"],
        help="hough keeps the pixel lane finder. scene reads lane_x from models/e2e_scene.onnx "
        "and stops when that confidence is low. gt feeds the road-network center at 8 m "
        "as lane_x (GVD_LANE_SOURCE=gt). A missing road hit stops.",
    )
    ap.add_argument(
        "--scene-onnx",
        default="",
        help="Scene-net ONNX path. Empty uses models/e2e_scene.onnx.",
    )
    ap.add_argument(
        "--no-engage",
        action="store_true",
        default=os.environ.get("GVD_NO_ENGAGE", "").strip().lower() in ("1", "true", "yes"),
        help="Hard engage disable (or GVD_NO_ENGAGE=1): engage is impossible whatever the keys, UI, "
        "bot file or --force-engage say. For captures.",
    )
    ap.add_argument(
        "--cam-age-baseline",
        action="store_true",
        help="Capture baseline, cam-age model OFF: log the per-tick cam_age telemetry with the hold off. "
        "With a non-age scene model (e.g. the default main-only one) it is that model; a gvd_cam_age=1 "
        "model gets main-only ages.",
    )
    ap.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="Stop the supervisor loop after this many seconds (0 = run until stopped). For captures.",
    )
    args = ap.parse_args()

    from python.data.data_root import DataDriveError, ensure_cache_dirs

    try:
        ensure_cache_dirs()
    except DataDriveError as exc:
        print(f"[GVD] {exc}", flush=True)
        raise SystemExit(2)

    if args.vision_only:
        os.environ["GVD_VISION_ONLY"] = "1"

    # Product bus matches the running backend (Python default is Drive/retail).
    if args.tech_hold or args.tech_probe or args.backend == "beamngpy":
        os.environ["GVD_BEAMNG"] = "1"
        os.environ["GVD_BACKEND"] = "beamngpy"
    elif args.backend == "window":
        os.environ["GVD_BACKEND"] = "window"

    if args.tech_hold:
        from python.sensors.tech import run_tech_hold

        raise SystemExit(run_tech_hold())

    if args.tech_probe:
        raise SystemExit(run_probe())

    if args.smoke:
        out = smoke(use_perception=True)
        # M4 dry-run: no BeamNG/QSV required
        rec = ClipRecorder(dry_run=True, encode_prefer="cpu", hz=args.hz)
        import numpy as np

        fake = np.zeros((240, 320, 3), dtype=np.uint8)
        for i in range(8):
            rec.push(fake, {"engaged": True, "planner": {"aeb": "off"}, "ego": {}}, now=time.time() + i * 0.05)
        path = rec.flush("smoke")
        # M5: stub E2E + shadow fields into state
        e2e = make_e2e()
        cfg = load_shadow_config()
        tick = shadow_tick(
            policy=args.policy,
            engaged=False,
            heartbeat_ok=True,
            path_debug_preview=True,
            allow_preview_drive=False,
            path_ego=None,
            planner={"aeb": "off", "target_v": None},
            ego_speed_mps=None,
            lane_conf=0.0,
            path_conf=0.0,
            seq=0,
            e2e_policy=e2e,
            main_bgr=fake,
            wide_bgr=fake,
            cfg=cfg,
        )
        # Keep the perception smoke state (tracks, lanes, signs, stub road model) and add the
        # M5 fields to it, instead of replacing it with a blank default_state.
        st = read_state() or default_state()
        st["policy"] = args.policy
        st["engaged"] = False
        st["shadow"] = {
            "steer": tick.shadow.get("steer", 0.0),
            "throttle": tick.shadow.get("throttle", 0.0),
            "brake": tick.shadow.get("brake", 0.0),
        }
        st["e2e_ok"] = bool(tick.e2e_ok)
        st["veto_reason"] = tick.veto_reason
        st["e2e_backend"] = e2e.backend
        write_state(st)
        print(f"[GVD] smoke frame -> {out}")
        print(f"[GVD] state -> {state_path()}")
        print(f"[GVD] clip dry-run -> {path} trigger={rec.last_clip_trigger} enc={rec.encoder}")
        print(f"[GVD] M5 policy={args.policy} e2e={e2e.name} shadow={st['shadow']} veto={st['veto_reason']}")
        return

    # Ego speed is known only from an echo that advanced in THIS session. Take the baseline
    # seq now, before backend and model init: whatever the mod echoes after this is new.
    session_echo = SessionEcho()
    session_echo.baseline(read_ego_feedback())

    backend_name = resolve_backend_name(None if args.backend == "auto" else args.backend)
    hw = probe(backend=backend_name)
    print(hw.boot_line())
    if hw.is_retail:
        print(
            "[GVD] RETAIL PATH: 1-cam window capture only (main/cam_main); other cam_health stay missing. "
            "Drive = gvd_cmd.json -> mod Lua on the player vehicle. 8-cam rig + direct control = BeamNG.tech + BeamNGpy."
        )
    reason = refuse_live_start(hw, vision_only=args.vision_only)
    if reason:
        print(f"[GVD] REFUSE: {reason}")
        raise SystemExit(1)

    if backend_name == "beamngpy":
        from python.sensors.tech import tech_hold_gate

        # Wait gate before any camera open or unique-frame Hz.
        # tech_hold_gate logs phase=attach / phase=Hello / phase=wait-gate.
        hold = tech_hold_gate(retain=True)
        if not hold.proceed:
            print(f"[GVD] REFUSE: {hold.note}")
            print(
                "[GVD] Tech hold failed before vision. Unique-frame Hz is not measured. "
                "One starter: install-root BeamNG.tech.exe -tcom -console -gfx dx11 already up, "
                "mod loaded, vehicle spawned. Attach only (GVD_TECH_LAUNCH=0). "
                "Hello timeout is REFUSE and is not a missing vehicle. "
                "Do not kill BeamNG.tech or CrashSender."
            )
            raise SystemExit(1)

    backend = make_backend(backend_name if args.backend != "auto" else backend_name)
    backend.open()
    from python.sensors.tech import exit_if_tech_connect_failed

    exit_if_tech_connect_failed(backend, backend_name)
    perc = ModularPerception(allow_synthetic=args.allow_synthetic_detect, clock=clock)
    e2e_policy = make_e2e()
    ui = VizUI()
    ui.record_browser = _ask_strip_folder
    ui.debug.detector_id = str(args.detector or "auto")
    ui.debug.e2e_id = str(args.e2e_model or "auto")
    models_rt = ModelRuntime(detector_id="", e2e_id="")
    perc, e2e_policy, model_notes = models_rt.sync(
        ui.debug, perc, e2e_policy, allow_synthetic=bool(args.allow_synthetic_detect)
    )
    for note in model_notes:
        print(f"[GVD] {note}")

    vehicle = getattr(backend, "vehicle", None)
    bng = getattr(backend, "bng", None)
    if vehicle is not None and not getattr(backend, "session", None):
        attach_electrics(vehicle, bng)
    actuator = make_actuator(vehicle, prefer_beamngpy=True)
    if backend_name == "beamngpy" and actuator.name != "beamngpy":
        print(
            "[GVD] Tech did not yield a vehicle.control handle; falling back to cmd_json "
            "(mod Lua). Cameras may still be missing until a vehicle is spawned."
        )
    shadow_cfg = load_shadow_config(_ctrl_yaml)
    override_cfg = load_override_config(_ctrl_yaml)
    override = OverrideDetector(override_cfg)
    # Retail echo freshness: the filter only eats new echoes; a frozen or pre-contract echo
    # disengages / refuses instead of reading as "no override".
    echo_gate = EchoGate()
    echo_fault_prev: str | None = None

    allow_nvenc = args.encode == "nvenc"
    if args.encode == "nvenc":
        prefer = "nvenc"
    elif args.encode == "cpu":
        prefer = "cpu"
    else:
        prefer = "qsv"
    recorder = ClipRecorder(
        encode_prefer=prefer,
        allow_nvenc=allow_nvenc,
        hz=args.hz,
        dry_run=False,
    )

    print(f"[GVD] camera backend={backend.name} detector={perc.detector.name} actuator={actuator.name}")
    ident0 = bus_identity() if clock is None else bus_identity(now=clock.wall())
    print_py_bus(state_path(), force=True)
    if not ident0.matched:
        print(f"[GVD] link=MISMATCH {ident0.note} — refuse actuation. Steam GELua does not inherit GVD_DOCS_DIR.")
    if actuator.name == "cmd_json":
        print(
            "[GVD] actuator=cmd_json: Documents/GVD/gvd_cmd.json -> gvd_main.applyCmdJson -> player vehicle "
            "(input.event secondary Direct Drive wheel+pedals, source=gvd). Ego speed/inputs echo back via gvd_ego.json; "
            "cmd_applied is claimed only on a fresh Lua ack."
        )
    print(f"[GVD] M5 policy={args.policy} e2e={e2e_policy.name} (modular vetoes E2E; shadow writes both)")
    print(f"[GVD] clip encoder={recorder.encoder} (qsv prefer; never default nvenc)")
    print(f"[GVD] state path: {state_path()}")
    print("[GVD] Vision-only inference: RGB + ego kinematics + GPS nav hint. "
          "LiDAR/radar/IMU extras are Foxglove / future fusion only; the corridor planner ignores them. "
          "Pin is not a route.")
    print("[GVD] M3: no drive on preview unless --allow-preview-drive; engage via Alt+G (gvd_engage.json).")
    print("[GVD] M4: clips on disengage / AEB / near-miss / key C. Live QSV UNPROVEN until Windows smoke.")
    print(
        f"[GVD] player override on the steer residual: enter {override_cfg.steer_enter:.3f} / "
        f"exit {override_cfg.steer_exit:.3f} held {override_cfg.steer_hold_ms:.0f}ms, spike "
        f"{override_cfg.steer_spike:.2f}, lpf {override_cfg.lpf_tau_ms:.0f}ms (force-feedback "
        f"noise must not disengage; live FFB UNPROVEN). Pedals tight: brake "
        f"{override_cfg.brake_enter:.2f} / throttle {override_cfg.throttle_enter:.2f}."
    )
    if args.allow_preview_drive:
        print("[GVD] WARNING: --allow-preview-drive is ON")

    sensors_cfg = load_sensors_config()
    if args.foxglove:
        fox_cfg = dict(sensors_cfg.get("foxglove") or {}) if isinstance(sensors_cfg.get("foxglove"), dict) else {}
        fox_cfg["enabled"] = True
        sensors_cfg["foxglove"] = fox_cfg
    extras = ExtraSensors(sensors_cfg)
    fox = FoxgloveBridge(sensors_cfg)

    win = "GVD VISION" if args.viz else None
    viz_screen_active = str(args.viz_screen)
    viz_note = ""
    if win:
        import cv2

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, STAGE_W + ui.nerd_width, STAGE_H)
        viz_note = place_opencv_window(win, screen=args.viz_screen, fullscreen=bool(args.viz_fullscreen))

        def _on_mouse(event, x, y, flags, param):
            if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
                return
            # Highgui already mapped this point through the resized client
            # into the image last shown. getWindowImageRect is that client
            # (screen origin + client size). Do not scale the point again.
            img_w = STAGE_W + (ui.nerd_width if ui.show_nerd and 0 not in ui.layers else 0)
            img_h = STAGE_H
            window_rect = None
            try:
                rect = cv2.getWindowImageRect(win)
                window_rect = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
            except Exception:
                window_rect = None
            ix, iy = image_point_from_window_mouse(
                int(x), int(y), image_wh=(img_w, img_h), window_rect=window_rect,
            )
            if event == cv2.EVENT_MOUSEMOVE:
                ui.handle_hover(ix, iy, stage_w=STAGE_W)
                return
            ui.handle_click(ix, iy, stage_w=STAGE_W)

        cv2.setMouseCallback(win, _on_mouse)

    session_start = time.time()
    policy_active = args.policy
    frame_i = 0
    cmd_seq = 0
    cam_hz_ema = 0.0
    loop_hz_ema = 0.0
    last_cam_t = time.perf_counter()
    last_loop_t = time.perf_counter()
    if clock is not None:
        last_cam_t = last_loop_t = clock.perf()
    # Engage ON counts only on an off->on edge seen this session: a latch or bot file that is
    # already on at start belongs to an earlier session and is refused (latched_before_start).
    latch_seen_off = False
    bot_seen_off = False
    prev_requested = False
    ego_refusal_prev = None
    last_refusal_reason = "none"
    prev_force = bool(args.force_engage)
    prev_preview = bool(args.allow_preview_drive)
    prev_engaged = False
    bot_engage = BotEngage()
    prev_bot = False
    engage_arm_mono: float | None = None
    review_cap = ReviewCapture()
    lane_steer = LaneSteerState()
    launch_guard = LaunchGuard()
    gt_log = None
    if gt_log_enabled(bool(args.gt_log)) or args.lane_source == "gt":
        gt_log = GroundTruthLogger.open()
        print(f"[GVD] ground truth -> {gt_log.path}", flush=True)
    try:
        scene_lane = None
        scene_guard = None
        # Hold-last side cameras (a gvd_cam_age=1 scene model on BeamNG).
        cam_age_live = False
        # cam_age telemetry: the hold, or its --cam-age-baseline (cam-age model off).
        cam_age_log = False
        respawn_watch = RespawnWatch()
        # poll_vehicle handing back last-good (coalesced) ego: counted every tick;
        # engaged past one tick it takes the existing heartbeat_stale stop.
        stale_ego = StaleEgo()
        prev_main_frame = None
        would_skip_control = False
        compose_ms = 0.0
        detector_seen = None
        if args.no_engage:
            print("[GVD] engage HARD-disabled (--no-engage / GVD_NO_ENGAGE): no key, UI or file can engage", flush=True)
        run_deadline = time.monotonic() + float(args.run_seconds) if float(args.run_seconds) > 0 else None
        if args.lane_source == "scene":
            from python.train.scene_lane_runtime import SceneLaneGuard, SceneLaneRuntime

            # SceneInputShapeError at load refuses Engage; mid-drive it is a HOLD.
            scene_guard = SceneLaneGuard()
            # A gvd_cam_age=1 model needs the 8 per-camera hold ages. Only the
            # BeamNG backend can feed them (all 8 cameras streamed, filtered by
            # the recorder's CameraSchedule); anything else refuses such a model.
            feeds_cam_age = bool(getattr(backend, "can_feed_cam_age", False))
            scene_lane = scene_guard.load(
                functools.partial(SceneLaneRuntime, feeds_cam_age=feeds_cam_age), args.scene_onnx or None
            )
            age_model = bool(getattr(scene_lane, "cam_age", False))
            if (age_model or args.cam_age_baseline) and scene_lane is not None and scene_guard.refused is None:
                cam_age_log = True
                # Per-camera read status / read ms and main_age_ms on the telemetry line.
                enable_read_log = getattr(backend, "enable_read_log", None)
                if callable(enable_read_log):
                    enable_read_log()
                if args.cam_age_baseline:
                    print("[GVD] cam-age: baseline, cam-age model OFF (hold off, telemetry only)", flush=True)
                else:
                    backend.enable_live_hold()
                    cam_age_live = True
                    print(
                        "[GVD] cam-age: all 8 cameras streamed every tick; side cameras held through the "
                        "recorder's CameraSchedule (ages logged per tick as '[GVD] cam_age')",
                        flush=True,
                    )
                try:
                    model_meta = dict(scene_lane.session.get_modelmeta().custom_metadata_map)
                except Exception:
                    model_meta = {}
                print(
                    f"[GVD] cam-age model gvd_cam_age={model_meta.get('gvd_cam_age', 'na')} "
                    f"cams={len(getattr(scene_lane, 'cams', ()) or ())} "
                    f"capture_untrained={model_meta.get('gvd_capture_untrained', '0')} "
                    f"hold={'on' if cam_age_live else 'off'} engage_disabled={int(bool(args.no_engage))}",
                    flush=True,
                )
                try:
                    scene_providers = ",".join(scene_lane.session.get_providers())
                except Exception:
                    scene_providers = "na"
                print(f"[GVD] cam-age providers scene={scene_providers}", flush=True)
            if scene_guard.refused is not None:
                print(f"[GVD] REFUSE ENGAGE: lane source=scene {scene_guard.refused}", flush=True)
            else:
                print(
                    f"[GVD] lane source=scene onnx={scene_lane.path} ready={scene_lane.ready} ({scene_lane.reason})",
                    flush=True,
                )
        elif args.lane_source == "gt":
            print("[GVD] lane source=gt road-network center at 8 m", flush=True)
        while True:
            if run_deadline is not None and time.monotonic() >= run_deadline:
                print(f"[GVD] --run-seconds {float(args.run_seconds):g} reached; stopping", flush=True)
                break
            loop_t0 = time.perf_counter()
            if clock is not None:
                loop_t0 = clock.perf()
            tick_dt: float | None = None
            if frame_i > 0:
                dt_loop = max(1e-6, loop_t0 - last_loop_t)
                tick_dt = dt_loop
                loop_hz_ema = ema_hz(loop_hz_ema, 1.0 / dt_loop)
            last_loop_t = loop_t0

            # In-game GVD app requests ride on the existing prefs file (no second bus).
            prefs = _read_ui_prefs()
            if _ui_request_is_live(prefs, session_start):
                want_policy = str(prefs.get("policy") or "").lower()
                if want_policy in ("modular", "e2e", "shadow") and want_policy != policy_active:
                    policy_active = want_policy
                    print(f"[GVD] policy -> {policy_active} (in-game GVD app; modular veto unchanged)")
                want_screen = str(prefs.get("viz_screen") or "").strip().lower()
                if win is not None and want_screen and want_screen != viz_screen_active:
                    viz_screen_active = want_screen
                    viz_note = place_opencv_window(
                        win, screen=want_screen, fullscreen=bool(args.viz_fullscreen)
                    )

            t_grab = time.perf_counter()
            bundle = backend.grab()
            grab_ms = float(getattr(bundle, "grab_ms", 0.0) or 0.0)
            if grab_ms <= 0:
                grab_ms = (time.perf_counter() - t_grab) * 1000.0
            grab_phase = int(getattr(bundle, "grab_phase", -1))
            grab_poll_free = bool(getattr(bundle, "grab_poll_free", True))
            main = bundle.main_bgr()
            unique_gpu_n = int(getattr(bundle, "unique_gpu_n", 0) or 0)
            now = time.perf_counter()
            if frame_i > 0:
                dt_cam = max(1e-6, now - last_cam_t)
                cam_hz_ema = ema_hz(cam_hz_ema, unique_frame_hz_inst(unique_gpu_n, dt_cam))
            last_cam_t = now

            steer = 0.0
            ego_source = "none"
            ego_fb = None
            vdata = None
            if hasattr(backend, "poll_vehicle"):
                try:
                    vdata = backend.poll_vehicle()
                except Exception:
                    vdata = None
            ego_stale = stale_ego.observe(vdata)
            if cam_age_live:
                # A respawn or reset drops every held side frame; this tick's grab
                # (taken before the edge was seen) keeps main only.
                respawn = respawn_watch.observe(vdata, tick_dt, mono=time.monotonic())
                if respawn is not None:
                    backend.reset_live_hold()
                    scrub_after_respawn(bundle)
                    print(f"[GVD] cam-age: respawn ({respawn}); hold reset, side cameras black until a new frame is stored", flush=True)
            sensors_poll_ms = 0.0
            poll_gps_ms = 0.0
            poll_gps_sent = False
            if vdata is not None:
                sensors_poll_ms = float(getattr(vdata, "sensors_poll_ms", 0.0) or 0.0)
                poll_gps_ms = float(getattr(vdata, "poll_gps_ms", 0.0) or 0.0)
                poll_gps_sent = bool(getattr(vdata, "poll_gps_sent", False))
            el = read_electrics_inputs(vehicle)
            electrics_ms = last_electrics_ms()
            spd, steer_in = el.speed_mps, el.steering_input
            throttle_in, brake_in = el.throttle_input, el.brake_input
            yaw_rate = 0.0
            accel = 0.0
            if vdata is not None and vdata.connected:
                if vdata.speed_mps is not None:
                    spd = vdata.speed_mps
                if vdata.steering_input is not None:
                    steer_in = vdata.steering_input
                if vdata.throttle_input is not None:
                    throttle_in = vdata.throttle_input
                if vdata.brake_input is not None:
                    brake_in = vdata.brake_input
                if vdata.yaw_rate is not None:
                    yaw_rate = float(vdata.yaw_rate)
                if vdata.accel is not None:
                    accel = float(vdata.accel)
            if spd is not None or steer_in is not None:
                ego_source = "beamngpy"
            if vehicle is None:
                # Retail: the mod echoes electrics (wheelspeed, inputs) through gvd_ego.json.
                ego_fb = read_ego_feedback() if clock is None else read_ego_feedback(now=clock.wall())
                if ego_fb is not None and ego_fb.fresh:
                    spd, steer_in = ego_fb.speed_mps, ego_fb.steering_input
                    throttle_in, brake_in = ego_fb.throttle_input, ego_fb.brake_input
                    ego_source = "lua"
                    if ego_fb.yaw_rate is not None:
                        yaw_rate = float(ego_fb.yaw_rate)

            session = getattr(backend, "session", None)
            extras_bundle = extras.poll(vdata=vdata, ego_fb=ego_fb, session=session)
            extras_bundle.foxglove = fox.note if fox.enabled else "off"
            fox.publish(extras_bundle, frames=getattr(bundle, "frames", None))
            write_sensors_snapshot(extras_bundle)

            # The one rule: ego speed is this session's fresh measurement, or None.
            if vehicle is None:
                session_echo.observe(ego_fb=ego_fb, now=time.monotonic())
                ego = resolve_ego(
                    echo_status=session_echo.status,
                    echo_speed_mps=session_echo.speed_mps,
                    tech_speed_mps=None,
                    tech_stale=None,
                )
            else:
                ego = resolve_ego(
                    echo_status=None,
                    echo_speed_mps=None,
                    tech_speed_mps=spd,
                    tech_stale=stale_ego.blocks_drive(True),
                )
            ego_v = ego.mps
            if steer_in is not None:
                steer = float(steer_in) * 30.0

            perc, e2e_policy, model_notes = models_rt.sync(
                ui.debug, perc, e2e_policy, allow_synthetic=bool(args.allow_synthetic_detect)
            )
            for note in model_notes:
                print(f"[GVD] {note}", flush=True)
            if cam_age_log and getattr(perc.detector, "name", None) != detector_seen:
                detector_seen = getattr(perc.detector, "name", None)
                print(f"[GVD] cam-age providers detector={detector_runtime(perc.detector)}", flush=True)

            # Loaded detector / lanes / planner / E2E shadow run every tick, engaged or not.
            # The stitch is this grab only: eight cameras, one timestamp.
            rig_view = stitch_frames(bundle_tick_frames(bundle))
            pout = perc.tick(
                main,
                stitch_bgr=rig_view.bgr,
                ego_speed_mps=ego_v,
                steer_deg=steer,
                yaw_rate=yaw_rate,
                dt_s=tick_dt,
            )
            pout = apply_to_perception(ui.debug, pout)

            if ui.debug.force_engage and not prev_force:
                print("[GVD] DEBUG force engage ON (sim toy; not Alt+G)", flush=True)
            if (not ui.debug.force_engage) and prev_force and not args.force_engage:
                print("[GVD] DEBUG force engage off", flush=True)
            if ui.debug.allow_preview and not prev_preview:
                print("[GVD] DEBUG drive-on-preview ON", flush=True)
            prev_force = bool(ui.debug.force_engage)
            prev_preview = bool(ui.debug.allow_preview)

            bot_on = bot_engage.poll() if clock is None else bot_engage.poll(now=clock.wall())
            if bot_on and not prev_bot:
                print("[GVD] bot engage ON (gvd_bot_engage.json)", flush=True)
            elif prev_bot and not bot_on:
                print("[GVD] bot engage off", flush=True)
            prev_bot = bot_on
            latch_on = (read_engage_flag(default=False) if clock is None
                        else read_engage_flag(default=False, now=clock.wall()))
            if not latch_on:
                latch_seen_off = True
            if not bot_on:
                bot_seen_off = True
            # --force-engage and the debug panel bit are this session's own requests.
            fresh_request = (
                bool(args.force_engage)
                or bool(ui.debug.force_engage)
                or (latch_on and latch_seen_off)
                or (bot_on and bot_seen_off)
            )
            requested = bool(args.force_engage) or bool(ui.debug.force_engage) or latch_on or bot_on
            if requested and not prev_requested:
                last_refusal_reason = "none"  # a new engage attempt clears the sticky reason
            prev_requested = requested
            engaged = requested
            if args.no_engage:
                engaged = False  # hard disable: nothing above can engage
            ident = bus_identity() if clock is None else bus_identity(now=clock.wall())
            if not ident.matched:
                engaged = False
                # Refuse actuation. Do not guess Drive vs Tech or honor a Python-only override.
            # One unknown-ego handler: no fresh speed this session (or an old latch) refuses
            # or disengages by name and releases the car (throttle 0, brake 0).
            ego_refusal = None
            if engaged:
                allow, why = engage_gate(True, fresh_request, ego)
                if not allow:
                    ego_refusal = why
                    if latch_on or bot_on or why != ego_refusal_prev:
                        bot_engage.clear()
                        write_engage_flag(False, disengage_reason=why)
                    if why != ego_refusal_prev:
                        if why == "lua_echo_outdated":
                            print(f"[GVD] REFUSE ENGAGE: {outdated_message(ego_fb)}", flush=True)
                        elif prev_engaged:
                            print(f"[GVD] DISENGAGED: {why} (ego speed unknown: {ego.why})", flush=True)
                        else:
                            print(f"[GVD] REFUSE ENGAGE: {why} (ego speed unknown: {ego.why})", flush=True)
                    engaged = False
            # A new refusal is flushed to gvd_state on its own tick, like a falling edge.
            refusal_new = ego_refusal is not None and ego_refusal != ego_refusal_prev
            ego_refusal_prev = ego_refusal
            scene_refuse = scene_guard is not None and scene_guard.refused is not None and engaged
            if scene_refuse:
                # The scene graph is the wrong size. Never engage on it, and do not fall back
                # to the pixel lane finder.
                engaged = False
                if (read_engage_flag(default=False) if clock is None
                        else read_engage_flag(default=False, now=clock.wall())) or bot_on:
                    bot_engage.clear()
                    write_engage_flag(False, disengage_reason="scene_input_shape")
            if engaged and not prev_engaged:
                # The tick above may have held a pre-engage one-side center.
                # Refit with the seed gate so that value cannot steer.
                perc.arm_lane_seed()
                pout = perc.tick(
                    main,
                    stitch_bgr=rig_view.bgr,
                    ego_speed_mps=ego_v,
                    steer_deg=steer,
                    yaw_rate=yaw_rate,
                    dt_s=tick_dt,
                )
                pout = apply_to_perception(ui.debug, pout)
            disengage_reason = "none" if ident.matched else "bus_mismatch"
            heartbeat_ok = True
            if stale_ego.blocks_drive(engaged):
                # Stale ego (coalesced poll_vehicle) for more than one tick while
                # engaged: the existing heartbeat_stale stop and disengage.
                heartbeat_ok = False
                print(f"[GVD] stale ego {stale_ego.run} ticks while engaged: heartbeat_stale stop", flush=True)
            allow_preview = bool(args.allow_preview_drive) or bool(ui.debug.allow_preview)
            policy_tick = ui.debug.effective_policy(policy_active)
            if engaged:
                if engage_arm_mono is None:
                    engage_arm_mono = time.monotonic()
                engage_age_s: float | None = time.monotonic() - engage_arm_mono
            else:
                engage_arm_mono = None
                engage_age_s = None
            tick_cfg = ShadowConfig(
                lane_conf_min=float(ui.debug.lane_conf_min),
                steer_disagree_max=shadow_cfg.steer_disagree_max,
                path_conf_min=shadow_cfg.path_conf_min,
                min_accept_hz=shadow_cfg.min_accept_hz,
                engage_hz_grace_s=shadow_cfg.engage_hz_grace_s,
                engage_hz_grace_floor=shadow_cfg.engage_hz_grace_floor,
            )

            scene_stop = False
            scene_hold = False
            gt_stop = False
            gt_preview = None
            scene_lane_x = None
            scene_conf = None
            if args.lane_source == "gt":
                from python.train.scene_lane_runtime import straight_lane_path

                if gt_log is not None:
                    gt_preview = gt_log.curve_preview(
                        vdata,
                        bng,
                        socket_busy=bool(getattr(bundle, "grab_read_blocked", False)),
                        dt_s=tick_dt,
                    )
                if gt_preview is None:
                    gt_stop = True
                    pout.path_ego = []
                    pout.lane_conf = 0.0
                    pout.path_conf = 0.0
                    pout.lane_source = "gt/stop"
                else:
                    # LATERAL FRAME: GT preview lane_x is about the state pos (ref node).
                    pout.path_ego = straight_lane_path(float(gt_preview.lane_x))
                    pout.lane_conf = 1.0
                    pout.path_conf = 1.0
                    # Not "hold": the stale-speed cap treats that word as a frozen
                    # camera lane and would parking-brake above 6 m/s.
                    pout.lane_source = "gt"
                    pout.path_debug_preview = False
                    pout.frame_age_s = 0.0
            elif scene_lane is not None:
                from python.data.strip_writer import compose_strip
                from python.train.scene_lane_runtime import (
                    scene_lane_decision,
                    straight_lane_path,
                )

                frames = getattr(bundle, "frames", None)
                tick_ages = None
                if bool(getattr(scene_lane, "cam_age", False)):
                    # Hold-last strip and the 8 ages (python.sensors.live_hold).
                    frames, tick_ages = bundle_ages(bundle)
                t_compose = time.perf_counter()
                strip = compose_strip(frames if isinstance(frames, dict) else None)
                compose_ms = (time.perf_counter() - t_compose) * 1000.0
                was_hold = scene_guard.hold is not None
                lane_hat, conf = scene_guard.observe(scene_lane, strip, tick_dt, engaged=engaged, ages=tick_ages)
                scene_hold = scene_guard.hold is not None
                if scene_hold and not was_hold and engaged:
                    print(f"[GVD] HOLD: scene_input_shape {scene_guard.hold}", flush=True)
                scene_lane_x = lane_hat
                scene_conf = conf
                if scene_lane_decision(lane_hat, conf) != "lane":
                    scene_stop = True
                    pout.path_ego = []
                    pout.lane_conf = float(conf)
                    pout.lane_source = "scene/stop"
                else:
                    # LATERAL FRAME: lane_hat is in the frame the model was trained in (tech.gt_origin):
                    # ref node for v102 models, body centreline for v103 models.
                    pout.path_ego = straight_lane_path(float(lane_hat))
                    pout.lane_conf = float(conf)
                    pout.path_conf = float(conf)
                    pout.lane_source = "scene"
                    pout.path_debug_preview = False
            cmd_seq += 1
            wide = None
            try:
                wide = bundle.frames.get("wide")
            except Exception:
                wide = None
            tick = shadow_tick(
                policy=policy_tick,
                engaged=engaged,
                heartbeat_ok=heartbeat_ok,
                path_debug_preview=bool(pout.path_debug_preview),
                allow_preview_drive=allow_preview,
                path_ego=pout.path_ego,
                planner=pout.planner,
                ego_speed_mps=ego_v,
                lane_conf=float(pout.lane_conf),
                path_conf=float(pout.path_conf),
                seq=cmd_seq,
                e2e_policy=e2e_policy,
                main_bgr=main,
                wide_bgr=wide,
                stitch_bgr=rig_view.bgr,
                steer_deg=steer,
                cfg=tick_cfg,
                loop_hz=loop_hz_ema,
                camera_hz=cam_hz_ema,
                engage_age_s=engage_age_s,
            )
            cmd = tick.applied
            if scene_hold:
                cmd = stop_command(seq=cmd_seq, reason=SCENE_SHAPE_HOLD)
            elif scene_stop:
                cmd = stop_command(seq=cmd_seq, reason="scene_lane_low")
            elif gt_stop:
                cmd = stop_command(seq=cmd_seq, reason="gt_lane_missing")

            if not ident.matched:
                disengage_reason = "bus_mismatch"
                cmd = stop_command(seq=cmd_seq, reason="bus_mismatch")
            elif ego_refusal:
                disengage_reason = ego_refusal
                cmd = release_command(seq=cmd_seq, reason=ego_refusal)
            elif not engaged:
                disengage_reason = "scene_input_shape" if scene_refuse else "not_engaged"
            elif scene_hold:
                # Engaged HOLD: brake on, wheel and throttle at 0. A shadow/e2e
                # low-confidence veto must not turn this into a release.
                disengage_reason = "scene_input_shape"
            elif tick.should_disengage:
                veto_name = str(tick.veto_reason or "veto")
                if ui.debug.ignore_veto_disengage and veto_name != "heartbeat_stale":
                    disengage_reason = veto_name if veto_name != "none" else "veto"
                else:
                    engaged = False
                    disengage_reason = veto_name if veto_name != "none" else "veto"
                    bot_engage.clear()
                    write_engage_flag(False, disengage_reason=disengage_reason)
            elif cmd.reason == "preview_blocked":
                disengage_reason = "preview_blocked"
            elif cmd.reason == "heartbeat_stale":
                disengage_reason = "heartbeat_stale"
                engaged = False

            # Player override. Retail Direct Drive lock: when Lua saw a player device, electrics
            # are GVD's own command — use the physical lastInputs axes (absolute). Otherwise the
            # residual is still steering_input - aligned cmd.steer so FFB noise cannot disengage.
            # Soft Esc beamngpy is not source=gvd, so lastInputs / electrics can echo the command
            # we just sent. own_axes keeps the pedal residual, and the steer baseline follows
            # the resting wheel when the command changes, instead of the absolute player_* path.
            owns_axes = getattr(actuator, "name", "") == "beamngpy"
            # Retail cmd_json bus: the wheel only reaches us through the mod's 10 Hz echo, so
            # only a fresh echo is a new sample. BeamNGpy polls electrics itself every tick.
            echo_bus = vehicle is None and not owns_axes
            echo = echo_gate.observe(engaged=bool(engaged) and echo_bus, ego_fb=ego_fb, now=time.monotonic())
            echo_fault = echo.fault if engaged else None
            if echo_fault:
                # Link fault (echo frozen while engaged) or a mod too old to stamp its echo.
                # Same disengage path a player override takes. Speed is unknown: release, no brake.
                engaged = False
                disengage_reason = echo_fault
                bot_engage.clear()
                write_engage_flag(False, disengage_reason=echo_fault)
                cmd = release_command(seq=cmd_seq, reason=echo_fault)
                if echo_fault != echo_fault_prev:
                    if echo.status == STATUS_OUTDATED:
                        print(f"[GVD] REFUSE ENGAGE: {outdated_message(ego_fb)}", flush=True)
                    else:
                        why = (
                            f"no new electrics echo for {echo.since_ms:.0f} ms"
                            if echo.since_ms > 0.0
                            else "every recent echo arrived after a gap"
                        )
                        print(f"[GVD] DISENGAGED: {echo_fault} ({why})", flush=True)
            echo_fault_prev = echo_fault
            if owns_axes:
                ovr_steer, ovr_thr, ovr_brk, ovr_dev = steer_in, throttle_in, brake_in, False
            elif ego_fb is not None and ego_fb.fresh and ego_fb.player_device:
                ovr_steer, ovr_thr, ovr_brk, ovr_dev = (
                    ego_fb.player_steering, ego_fb.player_throttle, ego_fb.player_brake, True,
                )
            else:
                ovr_steer, ovr_thr, ovr_brk, ovr_dev = steer_in, throttle_in, brake_in, False
            ovr = override.update(
                engaged=engaged,
                steering_input=ovr_steer,
                throttle_input=ovr_thr,
                brake_input=ovr_brk,
                player_device=ovr_dev,
                own_axes=owns_axes,
                applied_seq=None if owns_axes else (
                    ego_fb.applied_seq if ego_fb is not None and ego_fb.fresh else None
                ),
                fresh=echo.fresh if echo_bus else True,
                gap=echo.gap if echo_bus else False,
                now=None if clock is None else clock.mono(),
            )
            if ovr.active and not ui.debug.ignore_override:
                engaged = False
                disengage_reason = ovr.reason
                bot_engage.clear()
                write_engage_flag(False, disengage_reason=ovr.reason)
                print(f"[GVD] DISENGAGED: {ovr.reason}", flush=True)
                # Gate reason rides along on the bus so the mod / nerd panel name it, not just
                # the generic not_engaged the following ticks write.
                cmd = stop_command(seq=cmd_seq, reason=ovr.reason)

            # Lookahead the steer law reads. Logged every tick. The PR 75
            # capture stored curvature and the clipped command, not this point.
            sample = steer_sample_point(pout.path_ego) if pout.path_ego else None
            lane_x = None if sample is None else float(sample.get("x", 0.0))
            lane_y = None if sample is None else float(sample.get("y", 0.0))
            frame_ts = None
            for _stamp_key in ("main", "cam_main"):
                _stamps = getattr(bundle, "timestamps", None) or {}
                if _stamp_key in _stamps:
                    try:
                        frame_ts = float(_stamps[_stamp_key])
                    except (TypeError, ValueError):
                        frame_ts = None
                    break
            frame_cmd_ms = (time.time() - frame_ts) * 1000.0 if frame_ts is not None else -1.0
            # A missing sample, a point closer than 4 m, or a heading past a
            # lane, is not a command to center the wheel. Hold the last steer
            # (creeping toward 0) and say why. preview_blocked still brakes;
            # it no longer snaps steer to 0 while that hold is armed.
            miss = None
            if lane_x is None:
                miss = "lane_hold"
            elif lane_y is not None:
                miss = lane_sample_hold_reason(lane_x, lane_y)
            holding = False
            # GT curvature is not in the straight 8 m center. Steer Stanley
            # plus feedforward, and do not treat a large heading as lane_far:
            # that decay is what parked gt_ctrl1 sideways after the projection
            # jumped. A held sample keeps the last bend instead of centering.
            if (
                engaged
                and args.lane_source == "gt"
                and gt_preview is not None
                and cmd.reason == "ok"
                and ego_v is not None
            ):
                geometric = curve_steer(
                    gt_preview.offset_m,
                    gt_preview.heading_err_rad,
                    gt_preview.curvature,
                    ego_v,
                )
                cmd.steer = follow_lane_steer(geometric, gt_preview.lane_x, lane_steer, tick_dt)
                lane_steer.reason = "curve_hold" if gt_preview.held else "curve"
            elif engaged and cmd.reason == "ok" and miss is None:
                cmd.steer = follow_lane_steer(cmd.steer, lane_x, lane_steer, tick_dt)
            elif (
                engaged
                and cmd.reason in ("ok", "preview_blocked")
                and miss is not None
                and lane_steer.armed
            ):
                cmd.steer = follow_lane_steer(
                    0.0, None, lane_steer, tick_dt, miss_reason=miss,
                )
                holding = True
            else:
                # A gate hold or a disengage sends steer 0. Do not resume the
                # next path command from a value the car was not holding.
                lane_steer.reset()
            # Debug gain, invert, and the steer cap still have the last word
            # before the launch cap. A crash after that brakes and disengages.
            cmd = apply_to_command(ui.debug, cmd)
            hdg = None
            dmg = None
            pos_xy = None
            if vdata is not None:
                dmg = getattr(vdata, "damage", None)
                direction = getattr(vdata, "dir", None)
                if direction is not None and len(direction) >= 2:
                    hdg = heading_from_dir(float(direction[0]), float(direction[1]))
                pose = getattr(vdata, "pos", None)
                if pose is not None and len(pose) >= 2:
                    pos_xy = (float(pose[0]), float(pose[1]))
            crash_reason = launch_guard.update(
                cmd,
                engaged=bool(engaged) and (cmd.reason == "ok" or holding),
                dt_s=tick_dt,
                heading_rad=hdg,
                speed_mps=ego_v,
                damage=dmg,
                pos_xy=pos_xy,
            )
            lane_age_s = float(getattr(pout, "frame_age_s", 0.0) or 0.0)
            if engaged and gt_preview is not None and (cmd.reason == "ok" or holding):
                cap = curve_speed_mps(gt_preview.curvature)
                spd_cap = ego_v
                if spd_cap is not None and float(spd_cap) > cap:
                    cmd.throttle = 0.0
                if gt_preview.lost:
                    # Polyline end or a rejected jump. Brake now. The wheel
                    # stays on the held curve; stop_command would zero it,
                    # which is how gt_ctrl1 left the bend at 14 m/s.
                    cmd.throttle = 0.0
                    cmd.brake = max(float(cmd.brake), 1.0)
            if engaged and (cmd.reason == "ok" or holding):
                apply_stale_speed_cap(
                    cmd,
                    speed_mps=ego_v,
                    lane_age_s=lane_age_s,
                    lane_src=str(getattr(pout, "lane_source", "") or ""),
                )
                yaw_from_engage = None
                if launch_guard.heading is not None and launch_guard.heading0 is not None:
                    yaw_from_engage = float(launch_guard.heading) - float(launch_guard.heading0)
                apply_lane_exit_brake(
                    cmd, lane_x=lane_x, heading_err_rad=yaw_from_engage,
                )
            if crash_reason:
                engaged = False
                disengage_reason = crash_reason
                holding = False
                lane_steer.reset()
                launch_guard.reset()
                bot_engage.clear()
                write_engage_flag(False, disengage_reason=crash_reason)
                print(
                    f"[GVD] DISENGAGED: {crash_reason} damage={dmg}",
                    flush=True,
                )
                cmd = stop_command(seq=cmd_seq, reason=crash_reason)
            long_v = None
            if vdata is not None:
                long_v = longitudinal_mps(getattr(vdata, "vel", None), direction)
            if engaged and moving_backward(long_v):
                engaged = False
                disengage_reason = "reverse"
                holding = False
                lane_steer.reset()
                launch_guard.reset()
                bot_engage.clear()
                write_engage_flag(False, disengage_reason="reverse")
                print(
                    f"[GVD] DISENGAGED: reverse long_mps={long_v:.2f}",
                    flush=True,
                )
                cmd = stop_command(seq=cmd_seq, reason="reverse")
                if hasattr(actuator, "note_park_hold"):
                    actuator.note_park_hold()

            # Actuators only when engaged + command ok (path / tracks / shadow already on this tick).
            # cmd_json: Lua applies whatever we write only while `engaged` rides along in the payload.
            # beamngpy: note_engaged(False) makes stop() release once, then not call vehicle.control.
            if hasattr(actuator, "note_engaged"):
                actuator.note_engaged(engaged)
            if hasattr(actuator, "note_ack"):
                actuator.note_ack(ego_fb)
            # Capture telemetry (read only): would actuate skip vehicle.control this
            # tick because a camera GE call holds the socket? Counted with engage off too.
            would_skip_control = bool(cam_age_log) and camera_ge_socket_busy()
            if engaged and (cmd.reason == "ok" or holding):
                applied = actuator.apply(cmd)
            else:
                applied = actuator.stop(seq=cmd_seq, reason=cmd.reason)
            # Reference for the override residual, keyed by seq so the mod's ack can find the
            # command its echo belongs to. Gate holds ride along as brake=1 and the car echoes
            # those back just like a real command, so they have to be in here too.
            override.note_command(
                seq=applied.seq, steer=applied.steer, throttle=applied.throttle, brake=applied.brake,
                now=None if clock is None else clock.mono(),
            )

            st = default_state(
                engaged=engaged,
                disengage_reason=disengage_reason,
                policy=policy_tick,
                loop_hz=loop_hz_ema,
                camera_hz=cam_hz_ema,
                infer_ms=pout.infer_ms,
                grab_ms=grab_ms,
                path_conf=pout.path_conf,
                path_width=pout.path_width,
                path_debug_preview=pout.path_debug_preview,
                objects_n=pout.objects_n,
                tracks_n=pout.tracks_n,
                lane_conf=pout.lane_conf,
                planner=pout.planner,
                gpu_name=hw.dgpu,
                gpu_vram_total_gb=float(hw.dgpu_vram_gb or 11.0),
                gpu_vram_used_gb=gpu_vram_used_gb(),
            )
            st["ego"]["speed_mps"] = ego_v
            if not engaged and disengage_reason not in ("none", "not_engaged"):
                last_refusal_reason = disengage_reason  # sticky; not_engaged never overwrites it
            elif engaged and not prev_engaged:
                last_refusal_reason = "none"
            st["last_refusal_reason"] = last_refusal_reason
            st["ego_echo"] = session_echo.telemetry()
            st["ego"]["steer_deg"] = steer
            st["ego"]["throttle"] = float(applied.throttle)
            st["ego"]["brake"] = float(applied.brake)
            st["ego"]["yaw_rate"] = float(yaw_rate)
            st["ego"]["accel"] = float(accel)
            if vdata is not None:
                st["vehicle"] = {
                    "vid": vdata.vid,
                    "model": vdata.model,
                    "connected": bool(vdata.connected),
                    "damage": vdata.damage,
                    "gear": vdata.gear,
                    "rpm": vdata.rpm,
                    "pose_ok": bool(vdata.pose_ok),
                    "sensors": dict(vdata.sensors),
                    "note": vdata.note,
                }
            st["nav"] = nav_snapshot(vdata)
            if str((st.get("nav") or {}).get("mode") or "missing") == "missing" and extras_bundle.gps.ok:
                # Tech attach has no gvd_ego.json; ego_fb stays None until Lua retail echo.
                heading = nav_heading_from_sources(vdata, ego_fb)
                st["nav"] = nav_hint_from_bundle(extras_bundle, heading_deg=heading)
            st["sensors"] = extras_bundle.health()
            # Real planner path wins. The 36 m steer preview is only the empty-path stand-in.
            st["path_ego"] = list(pout.path_ego) if pout.path_ego else steer_preview_path_ego(steer, length_m=36.0)
            if not pout.path_ego:
                st["path_debug_preview"] = True
            e2e_path = getattr(tick.e2e, "path", None)
            st["path_e2e"] = list(e2e_path) if e2e_path else []
            st["tracks"] = pout.tracks
            st["lanes_bev"] = pout.lanes_bev
            # Boundaries the fit returned. lanes_ext_for_live does not add a fan
            # from which side cameras delivered a frame.
            live_cams = {
                cid
                for cid in CAM_IDS
                if bundle.health.get(cid) in (CamHealth.OK, CamHealth.STALE)
            }
            st["lanes_ext"] = lanes_ext_for_live(pout.lanes_bev, pout.lane_conf, live_cams)
            st["road_edges"] = road_edges(st["lanes_ext"])
            st["signs"] = pout.signs
            st["agents"] = _forecast_agents(pout.tracks)
            if "show_agent_ghosts" in prefs:
                st["show_agent_ghosts"] = bool(prefs["show_agent_ghosts"])
            elif pout.tracks_n > 0:
                st["show_agent_ghosts"] = True
            if "show_path" in prefs:
                st["gvd_show_path"] = bool(prefs["show_path"])
            pw = _path_world_from_vehicle(st.get("path_ego") or [], vehicle, vdata)
            if pw:
                st["path_world"] = pw

            st["cam_health"] = bundle.health_str()
            st["capture_backend"] = bundle.backend
            st["capture_note"] = bundle.note
            st["rss_mb"] = _rss_mb()
            st["viz_window"] = win is not None
            st["viz_screen"] = viz_screen_active
            st["viz_note"] = viz_note
            st["detector"] = pout.detector_name
            st["actuator"] = actuator.name
            st["cmd_seq"] = int(applied.seq)
            st["cmd_reason"] = applied.reason
            st["cmd_applied"] = bool(applied.applied)
            if scene_guard is not None:
                st["scene_lane"] = scene_guard.status()
            st["lane_hold"] = {
                "lateral_m": None if lane_x is None else float(lane_x),
                "lookahead_y_m": None if lane_y is None else float(lane_y),
                "steer": float(applied.steer),
                "frame_cmd_ms": float(frame_cmd_ms),
                "why": lane_steer.reason,
                "source": getattr(pout, "lane_source", "none"),
            }
            st["ego_source"] = ego_source
            st["cmd_ack_seq"] = int(ego_fb.applied_seq) if ego_fb is not None else -1
            st["lua_applying"] = bool(ego_fb.applying and ego_fb.fresh) if ego_fb is not None else False
            # Mirrored so gvd_main runs the same override thresholds as config/control.yaml.
            st["override_cfg"] = config_mirror(override_cfg)
            st["override"] = {
                "channel": ovr.channel,
                "reason": ovr.reason,
                "steer_raw": ovr.steer_raw,
                "steer_filt": ovr.steer_filt,
                "steer_eff": ovr.steer_eff,
                "pedal_residual": ovr.pedal_residual,
                "steer_held_ms": ovr.steer_held_ms,
                "spike": bool(ovr.spike),
                "opposition": ovr.opposition,
                "ref_seq": ovr.ref_seq,
                "armed": bool(ovr.armed),
                "fresh": bool(ovr.fresh),
                "gap_hold": bool(ovr.gap_hold),
            }
            st["echo"] = {**echo.telemetry(), "min_mod_version": MIN_MOD_VERSION}
            if echo.status == STATUS_OUTDATED:
                st["echo"]["message"] = outdated_message(ego_fb)
            st["shadow"] = {
                "steer": float(tick.shadow.get("steer", 0.0)),
                "throttle": float(tick.shadow.get("throttle", 0.0)),
                "brake": float(tick.shadow.get("brake", 0.0)),
            }
            st["e2e_ok"] = bool(tick.e2e_ok)
            st["veto_reason"] = str(tick.veto_reason or "none")
            st["e2e_backend"] = e2e_policy.backend
            st["debug"] = ui.debug.as_dict()
            st["encode_backend"] = recorder.encoder
            miss = list(pout.missing)
            if pout.tracks_n > 0 and "tracks" in miss:
                miss = [m for m in miss if m != "tracks"]
            base_miss = [
                m
                for m in (st.get("missing_state_keys") or [])
                if m not in ("tracks", "lanes_bev", "real path_ego from planner")
            ]
            st["missing_state_keys"] = sorted(set(base_miss + miss))
            if pout.path_debug_preview is False:
                st["missing_state_keys"] = [m for m in st["missing_state_keys"] if "path_ego" not in m]
            st["missing_state_keys"] = apply_nav_missing(st["missing_state_keys"], st.get("nav"))
            h = extras_bundle.health()
            extra_miss: list[str] = []
            if h.get("lidar") == "missing" and (sensors_cfg.get("lidar") or sensors_cfg.get("lidar_lua")):
                extra_miss.append("lidar")
            if h.get("radar") == "missing" and sensors_cfg.get("radar"):
                extra_miss.append("radar")
            if extra_miss:
                st["missing_state_keys"] = sorted(set(st["missing_state_keys"] + extra_miss))
            st["grab_phase"] = grab_phase
            st["grab_poll_free"] = grab_poll_free
            companion_inflight = int(getattr(bundle, "companion_inflight", 0) or 0)
            grab_read_blocked = bool(getattr(bundle, "grab_read_blocked", False))
            st["companion_inflight"] = companion_inflight
            st["grab_read_blocked"] = grab_read_blocked
            st["sensors_poll_ms"] = sensors_poll_ms
            st["poll_gps_ms"] = poll_gps_ms
            st["poll_gps_sent"] = poll_gps_sent
            st["electrics_ms"] = electrics_ms
            st["heartbeat_ms"] = (time.perf_counter() - loop_t0) * 1000.0
            if cam_age_log:
                # Per-camera hold ages (cam_hold.CAM_AGE_ORDER; -1 = black), this
                # tick's duration and timers, would-skip-control, stale ego and
                # main exposure, for scripts/live_age_capture.py. Only with a
                # cam-age scene model or --cam-age-baseline: the default path logs
                # and writes nothing new. uniq=: per ok read, 1 = new frame,
                # 0 = identical to that camera's previous frame. tuniq=: the same
                # per frame the schedule took, against the previous taken frame
                # (the recorder's stored-frame view). Report only.
                _held, live_ages = bundle_ages(bundle)
                st["cam_ages"] = {cam: float(live_ages[k]) for k, cam in enumerate(CAM_AGE_ORDER)}
                st["stale_ego_n"] = int(stale_ego.total)
                main_now = bundle.main_bgr()
                main_luma, main_flash = main_exposure(main_now, prev_main_frame)
                if main_now is not None:
                    prev_main_frame = main_now
                timers = dict(getattr(scene_lane, "last_timers", None) or {})
                planner_d = pout.planner if isinstance(pout.planner, dict) else {}
                read_log = getattr(bundle, "read_log", None)
                read_log = read_log if isinstance(read_log, dict) else {}

                print(
                    f"[GVD] cam_age t={time.time():.3f} tick_ms={st['heartbeat_ms']:.2f} "
                    f"loop_dt_ms={'na' if tick_dt is None else f'{tick_dt * 1000.0:.2f}'} "
                    f"hold={int(getattr(bundle, 'cam_ages', None) is not None)} "
                    f"baseline={int(bool(args.cam_age_baseline))} engaged={int(bool(engaged))} "
                    f"engage_disabled={int(bool(args.no_engage))} would_skip={int(bool(would_skip_control))} "
                    f"stale_ego={int(bool(ego_stale))} stale_ego_n={stale_ego.total} reset_n={respawn_watch.count} "
                    f"main_luma={'na' if main_luma is None else f'{main_luma:.4f}'} main_flash={int(bool(main_flash))} "
                    f"compose_ms={compose_ms:.3f} prep_ms={_ms_text(timers.get('prep_ms'))} "
                    f"scene_ms={_ms_text(timers.get('scene_ms'))} post_ms={_ms_text(timers.get('post_ms'))} "
                    f"detect_ms={_ms_text(getattr(pout, 'detect_ms', None))} plan_ms={_ms_text(getattr(pout, 'plan_ms', None))} "
                    f"infer_ms={float(pout.infer_ms):.3f} aeb={planner_d.get('aeb', 'na')} "
                    f"grab_ms={_ms_text(getattr(bundle, 'grab_ms', None))} read_log_ms={_ms_text(read_log.get('log_ms'))} "
                    f"main_age_ms={_ms_text(read_log.get('main_age_ms'))} "
                    f"ages={ages_text(live_ages)} reads={reads_text(read_log or None)} "
                    f"uniq={uniq_text(read_log or None)} tuniq={uniq_text(read_log or None, 'tuniq')}",
                    flush=True,
                )
            # Soft Esc: gvd_state.json at most once per Lua pollEvery (0.10).
            # Max rewrite rate. A loop already slower than 100 ms writes at
            # the loop rate, so heartbeat age tracks the loop.
            # Engage writes every tick. The rising edge flushes this tick, and so
            # does the falling edge: the disengage reason is on that tick only.
            rising_engage = bool(engaged) and not prev_engaged
            falling_engage = prev_engaged and not bool(engaged)
            state_due = soft_esc_state_write_due(bool(engaged), rising=rising_engage, falling=falling_engage or refusal_new)
            prev_engaged = bool(engaged)
            state_skips = soft_esc_state_write_skips()
            rel_skips = int(getattr(actuator, "release_cmd_skips", 0) or 0)
            # Segment line is outside the stamp. grab_ms on a poll-free phase is
            # the stream_raw cost; a hitch-phase gap is the companion PollCamera.
            # heartbeat_ms - grab_ms - infer_ms is the ego-poll tail plus the
            # rest of the tick before this stamp. camera_hz is unchanged.
            # state_write_skip=1 is a Soft Esc rewrite that did not hit disk.
            print(
                f"[GVD] seg grab_ms={grab_ms:.2f} grab_phase={grab_phase} "
                f"grab_poll_free={int(grab_poll_free)} "
                f"companion_inflight={companion_inflight} "
                f"grab_read_blocked={int(grab_read_blocked)} "
                f"heartbeat_ms={st['heartbeat_ms']:.2f} "
                f"infer_ms={float(pout.infer_ms):.2f} "
                f"sensors_poll_ms={sensors_poll_ms:.2f} "
                f"poll_gps_ms={poll_gps_ms:.2f} poll_gps_sent={int(poll_gps_sent)} "
                f"electrics_ms={electrics_ms:.2f} "
                f"state_write_skip={0 if state_due else 1} "
                f"state_write_skips={state_skips} "
                f"release_cmd_skips={rel_skips} "
                f"lane_x={'na' if lane_x is None else f'{lane_x:.4f}'} "
                f"lane_y={'na' if lane_y is None else f'{lane_y:.2f}'} "
                f"model_lane_x={'na' if scene_lane_x is None else f'{float(scene_lane_x):.4f}'} "
                f"model_conf={'na' if scene_conf is None else f'{float(scene_conf):.3f}'} "
                f"steer={float(applied.steer):+.4f} "
                f"steer_why={lane_steer.reason} "
                f"lane_src={getattr(pout, 'lane_source', 'none')} "
                f"frame_age_s={float(getattr(pout, 'frame_age_s', 0.0) or 0.0):.3f} "
                f"frame_cmd_ms={frame_cmd_ms:.1f}",
                flush=True,
            )

            # Strip training is its own switch. Engage does not start or stop it.
            fields = live_strip_fields(
                lanes=pout.lanes_bev,
                tracks=pout.tracks,
                signs=pout.signs,
                steer=float(steer_in or 0.0),
                throttle=float(throttle_in or 0.0),
                brake=float(brake_in or 0.0),
                override_reason=ovr.reason,
            )
            ui.feed_strip(
                getattr(bundle, "frames", None),
                timestamps=getattr(bundle, "timestamps", None),
                t=time.time(),
                engaged=bool(engaged),
                ego={"speed_mps": ego_v},
                wheel=fields["wheel"],
                pedals=fields["pedals"],
                tech=fields["tech"],
            )

            # M4 ring + triggers
            wall = time.time()
            recorder.push(main, st, now=wall)
            trig = recorder.note_engage(engaged)
            if trig:
                recorder.request(trig, now=wall)
            if tick.clip_trigger:
                recorder.request(tick.clip_trigger, now=wall)
            aeb_trig = recorder.check_aeb(pout.planner if isinstance(pout.planner, dict) else None)
            if aeb_trig:
                recorder.request(aeb_trig, now=wall)
            flushed = recorder.maybe_flush(now=wall)
            if flushed:
                print(f"[GVD] clip flushed -> {flushed} ({recorder.last_clip_trigger})")
            st["last_clip_trigger"] = recorder.last_clip_trigger
            if recorder.last_clip_path:
                st["last_clip_path"] = recorder.last_clip_path

            # Call-site coalesce only. write_state itself still stamps the beat.
            # Mark after the write so the 100 ms cap is on the file, not the check.
            if state_due:
                write_state(st)
                soft_esc_state_write_mark()

            if gt_log is not None:
                gt_log.write(
                    bng=bng,
                    vdata=vdata,
                    engaged=bool(engaged),
                    lane_x=lane_x,
                    lane_src=str(getattr(pout, "lane_source", "none") or "none"),
                    steer=float(applied.steer),
                    steer_why=lane_steer.reason,
                    frame_cmd_ms=float(frame_cmd_ms),
                    frame_age_s=float(getattr(pout, "frame_age_s", 0.0) or 0.0),
                    stitch=getattr(rig_view, "bgr", None),
                    lanes_bev=getattr(pout, "lanes_bev", None),
                    socket_busy=bool(getattr(bundle, "grab_read_blocked", False)),
                    model_lane_x=None if scene_lane_x is None else float(scene_lane_x),
                    model_conf=None if scene_conf is None else float(scene_conf),
                )

            review_cap.write(
                bool(ui.debug.review_record),
                lanes=drawn_lane_records(st),
                path=list(st.get("path_ego") or []),
                steer=float(applied.steer),
                throttle=float(applied.throttle),
                glance=cabin_drive_word(st)[0],
                disengage_reason=str(st.get("disengage_reason") or "none"),
                frames=getattr(bundle, "frames", None),
                brake=float(applied.brake),
                engaged=bool(st.get("engaged")),
                cmd_reason=str(st.get("cmd_reason") or ""),
            )

            if win is not None:
                import cv2

                paint = load_paint_state()
                frame = render_stage(
                    paint,
                    ui=ui,
                    main_frame=main,
                    cam_frames=getattr(bundle, "frames", None),
                    dets=getattr(pout, "dets", None),
                )
                cv2.imshow(win, frame)
                # Full key code. waitKey() & 0xFF folds Qt Up (65362) into 82,
                # which is also ASCII R, so Up toggled review capture.
                raw_key = cv2.waitKeyEx(1)
                key = -1 if raw_key is None else int(raw_key)
                # q and Esc leave the supervisor. close() disconnects only
                # (quit_on_close=false) and does not kill BeamNG.tech.
                if key in (ord("q"), 27):
                    break
                if key < 0:
                    pass
                elif ui.handle_key(key):
                    pass
                elif key == ord("v"):
                    ui.toggle_nerd()
                elif key == ord("?"):
                    ui.toggle_help()
                elif key == ord("t"):
                    ui.top_down = not ui.top_down
                elif key == ord("c"):
                    recorder.request("manual", now=time.time())
                    print("[GVD] manual clip requested (key C)")
                elif key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                    ui.set_layer(int(chr(key)))

            frame_i += 1
            dt = 1.0 / max(args.hz, 1.0)
            if clock is not None:
                # Test step barrier: report this tick done, block until the test releases the next.
                clock.sleep(max(0.0, dt - (clock.perf() - loop_t0)))
                continue
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t0)))
    except KeyboardInterrupt:
        print("[GVD] stopped.")
    finally:
        try:
            if hasattr(actuator, "note_engaged"):
                actuator.note_engaged(False)  # cmd_json: Lua releases the car at once
            actuator.stop(seq=cmd_seq + 1, reason="shutdown")
        except Exception:
            pass
        try:
            bot_engage.clear()
            write_engage_flag(False, disengage_reason="shutdown")
        except Exception:
            pass
        backend.close()
        try:
            fox.close()
        except Exception:
            pass
        if win is not None:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
