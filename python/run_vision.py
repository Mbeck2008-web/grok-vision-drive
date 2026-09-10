"""GVD supervisor — cameras + perception + actuation + M4 clips + M5 shadow/E2E + gvd_state.json."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.control.actuate import (
    attach_electrics,
    make_actuator,
    read_ego_feedback,
    read_electrics_inputs,
    read_engage_flag,
    stop_command,
    write_engage_flag,
)
from python.control.e2e import make_e2e
from python.control.override import (
    OverrideDetector,
    config_mirror,
    load_override_config,
)
from python.data.record import ClipRecorder, choose_encoder
from python.perception.pipeline import ModularPerception
from python.perception.road_model import lanes_ext, road_edges
from python.runtime.hw_probe import probe, refuse_live_start
from python.runtime.shadow import load_shadow_config, shadow_tick
from python.runtime.state_io import (
    default_state,
    read_state,
    state_path,
    steer_preview_path_ego,
    write_state,
)
from python.sensors.cameras import make_backend, resolve_backend_name
from python.viz.monitors import place_opencv_window
from python.viz.stage import VizUI, render_stage, smoke



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


def _gpu_vram_used_gb() -> float:
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        ).strip().splitlines()[0]
        return float(out) / 1024.0
    except Exception:
        return 0.0



def _path_world_from_vehicle(path_ego: list, vehicle) -> list[dict[str, float]] | None:
    """Kinematics-only world path from BeamNGpy vehicle pose (honesty: not a map)."""
    if not path_ego or vehicle is None:
        return None
    try:
        # BeamNGpy Vehicle: state poll
        pos = None
        fwd = None
        up = None
        if hasattr(vehicle, "state") and isinstance(vehicle.state, dict):
            pos = vehicle.state.get("pos")
            # dir may be missing
        if pos is None and hasattr(vehicle, "get_position"):
            pos = vehicle.get_position()
        if pos is None:
            return None
        # Prefer sensor/state vectors when present — never invent forward (honesty)
        if hasattr(vehicle, "state") and isinstance(vehicle.state, dict):
            fwd = vehicle.state.get("dir") or vehicle.state.get("forward")
            up = vehicle.state.get("up")
        if fwd is None:
            return None
        if up is None:
            up = (0.0, 0.0, 1.0)
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        fx, fy, fz = float(fwd[0]), float(fwd[1]), float(fwd[2])
        ux, uy, uz = float(up[0]), float(up[1]), float(up[2])
        # right = fwd × up
        rx = fy * uz - fz * uy
        ry = fz * ux - fx * uz
        rz = fx * uy - fy * ux
        import math
        def _n(x, y, z):
            L = math.sqrt(x * x + y * y + z * z) + 1e-9
            return x / L, y / L, z / L
        fx, fy, fz = _n(fx, fy, fz)
        ux, uy, uz = _n(ux, uy, uz)
        rx, ry, rz = _n(rx, ry, rz)
        out = []
        for p in path_ego:
            ex, ey, ez = float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0))
            out.append({
                "x": px + rx * ex + fx * ey + ux * ez,
                "y": py + ry * ex + fy * ey + uy * ez,
                "z": pz + rz * ex + fz * ey + uz * ez,
            })
        return out
    except Exception:
        return None



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


def main() -> None:
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
        help="Clip encode: auto/qsv→libx264; nvenc only if explicitly requested",
    )
    ap.add_argument(
        "--viz-screen",
        default="auto",
        help="GVD VISION monitor: auto|1|2 (or GVD_VIZ_MONITOR). auto→non-primary if present",
    )
    ap.add_argument("--viz-fullscreen", action="store_true", help="Fullscreen GVD VISION on chosen monitor")
    args = ap.parse_args()

    if args.vision_only:
        os.environ["GVD_VISION_ONLY"] = "1"

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
            planner={"aeb": "off", "target_v": 0.0},
            ego_speed_mps=0.0,
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

    backend = make_backend(backend_name if args.backend != "auto" else backend_name)
    backend.open()
    perc = ModularPerception(allow_synthetic=args.allow_synthetic_detect)

    vehicle = getattr(backend, "vehicle", None)
    bng = getattr(backend, "bng", None)
    if vehicle is not None:
        attach_electrics(vehicle, bng)
    actuator = make_actuator(vehicle, prefer_beamngpy=True)
    e2e_policy = make_e2e()
    shadow_cfg = load_shadow_config(_ctrl_yaml)
    override_cfg = load_override_config(_ctrl_yaml)
    override = OverrideDetector(override_cfg)

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
    if actuator.name == "cmd_json":
        print(
            "[GVD] actuator=cmd_json: Documents/GVD/gvd_cmd.json -> gvd_main.applyCmdJson -> player vehicle "
            "(input.event steering/throttle/brake). Ego speed/inputs echo back via gvd_ego.json; "
            "cmd_applied is claimed only on a fresh Lua ack."
        )
    print(f"[GVD] M5 policy={args.policy} e2e={e2e_policy.name} (modular vetoes E2E; shadow writes both)")
    print(f"[GVD] clip encoder={recorder.encoder} (qsv prefer; never default nvenc)")
    print(f"[GVD] state path: {state_path()}")
    print("[GVD] Vision-only: no LiDAR/radar/GPS-loc/HD-map in the live loop.")
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

    ui = VizUI()
    win = "GVD VISION" if args.viz else None
    viz_screen_active = str(args.viz_screen)
    viz_note = ""
    if win:
        import cv2

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 800)
        viz_note = place_opencv_window(win, screen=args.viz_screen, fullscreen=bool(args.viz_fullscreen))

    session_start = time.time()
    policy_active = args.policy
    frame_i = 0
    cmd_seq = 0
    cam_hz_ema = 0.0
    last_cam_t = time.perf_counter()
    last_ego_v = 0.0
    try:
        while True:
            loop_t0 = time.perf_counter()

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

            bundle = backend.grab()
            main = bundle.main_bgr()
            now = time.perf_counter()
            if main is not None:
                dt = max(1e-6, now - last_cam_t)
                inst = 1.0 / dt
                cam_hz_ema = inst if cam_hz_ema <= 0 else (0.8 * cam_hz_ema + 0.2 * inst)
                last_cam_t = now

            steer = 0.0
            ego_v = last_ego_v
            ego_source = "none"
            ego_fb = None
            el = read_electrics_inputs(vehicle)
            spd, steer_in = el.speed_mps, el.steering_input
            throttle_in, brake_in = el.throttle_input, el.brake_input
            if spd is not None or steer_in is not None:
                ego_source = "beamngpy"
            if vehicle is None:
                # Retail: the mod echoes electrics (wheelspeed, inputs) through gvd_ego.json.
                ego_fb = read_ego_feedback()
                if ego_fb is not None and ego_fb.fresh:
                    spd, steer_in = ego_fb.speed_mps, ego_fb.steering_input
                    throttle_in, brake_in = ego_fb.throttle_input, ego_fb.brake_input
                    ego_source = "lua"
            if spd is not None:
                ego_v = max(0.0, float(spd))
                last_ego_v = ego_v
            if steer_in is not None:
                steer = float(steer_in) * 30.0

            pout = perc.tick(main, ego_speed_mps=ego_v, steer_deg=steer)

            engaged = bool(args.force_engage) or read_engage_flag(default=False)
            disengage_reason = "none"
            heartbeat_ok = True

            cmd_seq += 1
            wide = None
            try:
                wide = bundle.frames.get("wide")
            except Exception:
                wide = None

            tick = shadow_tick(
                policy=policy_active,
                engaged=engaged,
                heartbeat_ok=heartbeat_ok,
                path_debug_preview=bool(pout.path_debug_preview),
                allow_preview_drive=bool(args.allow_preview_drive),
                path_ego=pout.path_ego,
                planner=pout.planner,
                ego_speed_mps=ego_v,
                lane_conf=float(pout.lane_conf),
                path_conf=float(pout.path_conf),
                seq=cmd_seq,
                e2e_policy=e2e_policy,
                main_bgr=main,
                wide_bgr=wide,
                steer_deg=steer,
                cfg=shadow_cfg,
            )
            cmd = tick.applied

            if not engaged:
                disengage_reason = "not_engaged"
            elif tick.should_disengage:
                engaged = False
                disengage_reason = tick.veto_reason if tick.veto_reason != "none" else "veto"
                write_engage_flag(False, disengage_reason=disengage_reason)
            elif cmd.reason == "preview_blocked":
                disengage_reason = "preview_blocked"
            elif cmd.reason == "heartbeat_stale":
                disengage_reason = "heartbeat_stale"
                engaged = False

            # Player override. Steer is the filtered residual against the command the mod says
            # it applied, so force-feedback noise cannot disengage; the pedals stay tight.
            # Sticky either way — otherwise Lua and the player fight at loop rate — so the
            # engage flag stays false until Alt+G. Commands are noted below, after the actuator.
            ovr = override.update(
                engaged=engaged,
                steering_input=steer_in,
                throttle_input=throttle_in,
                brake_input=brake_in,
                applied_seq=ego_fb.applied_seq if ego_fb is not None and ego_fb.fresh else None,
            )
            if ovr.active:
                engaged = False
                disengage_reason = ovr.reason
                write_engage_flag(False, disengage_reason=ovr.reason)
                print(f"[GVD] DISENGAGED: {ovr.reason}", flush=True)
                # Gate reason rides along on the bus so the mod / nerd panel name it, not just
                # the generic not_engaged the following ticks write.
                cmd = stop_command(seq=cmd_seq, reason=ovr.reason)

            # Actuators only when engaged + command ok (shadow fields already on tick).
            # cmd_json: Lua applies whatever we write only while `engaged` rides along in the payload.
            if hasattr(actuator, "note_engaged"):
                actuator.note_engaged(engaged)
            if hasattr(actuator, "note_ack"):
                actuator.note_ack(ego_fb)
            if engaged and cmd.reason == "ok":
                applied = actuator.apply(cmd)
            else:
                applied = actuator.stop(seq=cmd_seq, reason=cmd.reason)
            # Reference for the override residual, keyed by seq so the mod's ack can find the
            # command its echo belongs to. Gate holds ride along as brake=1 and the car echoes
            # those back just like a real command, so they have to be in here too.
            override.note_command(
                seq=applied.seq, steer=applied.steer, throttle=applied.throttle, brake=applied.brake
            )

            st = default_state(
                engaged=engaged,
                disengage_reason=disengage_reason,
                policy=policy_active,
                loop_hz=args.hz,
                camera_hz=cam_hz_ema,
                infer_ms=pout.infer_ms,
                path_conf=pout.path_conf,
                path_width=pout.path_width,
                path_debug_preview=pout.path_debug_preview,
                objects_n=pout.objects_n,
                tracks_n=pout.tracks_n,
                lane_conf=pout.lane_conf,
                planner=pout.planner,
                gpu_name=hw.dgpu,
                gpu_vram_total_gb=float(hw.dgpu_vram_gb or 11.0),
                gpu_vram_used_gb=_gpu_vram_used_gb(),
            )
            st["ego"]["speed_mps"] = ego_v
            st["ego"]["steer_deg"] = steer
            st["ego"]["throttle"] = float(applied.throttle)
            st["ego"]["brake"] = float(applied.brake)
            st["path_ego"] = pout.path_ego if pout.path_ego else steer_preview_path_ego(steer, length_m=36.0)
            if not pout.path_ego:
                st["path_debug_preview"] = True
            st["tracks"] = pout.tracks
            st["lanes_bev"] = pout.lanes_bev
            # Viz road model: detected boundaries + labelled predictions, never invented lanes.
            st["lanes_ext"] = lanes_ext(pout.lanes_bev, pout.lane_conf)
            st["road_edges"] = road_edges(st["lanes_ext"])
            st["signs"] = pout.signs
            st["agents"] = _forecast_agents(pout.tracks)
            if "show_agent_ghosts" in prefs:
                st["show_agent_ghosts"] = bool(prefs["show_agent_ghosts"])
            elif pout.tracks_n > 0:
                st["show_agent_ghosts"] = True
            if "show_path" in prefs:
                st["gvd_show_path"] = bool(prefs["show_path"])
            pw = _path_world_from_vehicle(st.get("path_ego") or [], vehicle)
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
            }
            st["shadow"] = {
                "steer": float(tick.shadow.get("steer", 0.0)),
                "throttle": float(tick.shadow.get("throttle", 0.0)),
                "brake": float(tick.shadow.get("brake", 0.0)),
            }
            st["e2e_ok"] = bool(tick.e2e_ok)
            st["veto_reason"] = str(tick.veto_reason or "none")
            st["e2e_backend"] = e2e_policy.backend
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
            st["heartbeat_ms"] = (time.perf_counter() - loop_t0) * 1000.0

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

            write_state(st)

            if win is not None:
                import cv2

                frame = render_stage(st, ui=ui, main_frame=main)
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("v"):
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
            write_engage_flag(False, disengage_reason="shutdown")
        except Exception:
            pass
        backend.close()
        if win is not None:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
