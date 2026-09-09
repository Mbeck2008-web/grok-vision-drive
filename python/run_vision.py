"""GVD supervisor — cameras + modular perception + M3 sim actuation + gvd_state.json."""

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
    read_electrics_speed,
    read_engage_flag,
    safe_command,
    write_engage_flag,
)
from python.perception.pipeline import ModularPerception
from python.runtime.hw_probe import probe, refuse_live_start
from python.runtime.state_io import default_state, state_path, steer_preview_path_ego, write_state
from python.sensors.cameras import make_backend, resolve_backend_name
from python.viz.stage import VizUI, render_stage, smoke


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


def main() -> None:
    ap = argparse.ArgumentParser(description="GVD supervisor + cameras + M2 perception + M3 actuation")
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
        help="Dev only: treat as engaged without Alt+A gvd_engage.json (never default)",
    )
    args = ap.parse_args()

    if args.vision_only:
        os.environ["GVD_VISION_ONLY"] = "1"

    if args.smoke:
        out = smoke(use_perception=True)
        print(f"[GVD] smoke frame -> {out}")
        print(f"[GVD] state -> {state_path()}")
        return

    backend_name = resolve_backend_name(None if args.backend == "auto" else args.backend)
    hw = probe(backend=backend_name)
    print(hw.boot_line())
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
    print(f"[GVD] camera backend={backend.name} detector={perc.detector.name} actuator={actuator.name}")
    print(f"[GVD] state path: {state_path()}")
    print("[GVD] Vision-only: no LiDAR/radar/GPS-loc/HD-map in the live loop.")
    print("[GVD] M3: no drive on preview unless --allow-preview-drive; engage via Alt+A (gvd_engage.json).")
    if args.allow_preview_drive:
        print("[GVD] WARNING: --allow-preview-drive is ON")

    ui = VizUI()
    win = "GVD" if args.viz else None
    if win:
        import cv2  # noqa: F401

    frame_i = 0
    cmd_seq = 0
    cam_hz_ema = 0.0
    last_cam_t = time.perf_counter()
    last_ego_v = 0.0
    try:
        while True:
            loop_t0 = time.perf_counter()
            bundle = backend.grab()
            main = bundle.main_bgr()
            now = time.perf_counter()
            if main is not None:
                dt = max(1e-6, now - last_cam_t)
                inst = 1.0 / dt
                cam_hz_ema = inst if cam_hz_ema <= 0 else (0.8 * cam_hz_ema + 0.2 * inst)
                last_cam_t = now

            # Electrics ego speed when available — never invent 10 m/s
            steer = 0.0
            ego_v = last_ego_v
            spd, steer_in = read_electrics_speed(vehicle)
            if spd is not None:
                ego_v = max(0.0, float(spd))
                last_ego_v = ego_v
            if steer_in is not None:
                steer = float(steer_in) * 30.0  # approx deg for preview ribbon

            pout = perc.tick(main, ego_speed_mps=ego_v, steer_deg=steer)

            engaged = bool(args.force_engage) or read_engage_flag(default=False)
            disengage_reason = "none"
            hb_mtime = time.time()
            heartbeat_ok = True  # this process is the heartbeat source while the loop runs

            cmd_seq += 1
            cmd = safe_command(
                engaged=engaged,
                heartbeat_ok=heartbeat_ok,
                path_debug_preview=bool(pout.path_debug_preview),
                allow_preview_drive=bool(args.allow_preview_drive),
                path_ego=pout.path_ego,
                planner=pout.planner,
                ego_speed_mps=ego_v,
                seq=cmd_seq,
                policy="modular",
            )
            if not engaged:
                disengage_reason = "not_engaged"
            elif cmd.reason == "preview_blocked":
                disengage_reason = "preview_blocked"
            elif cmd.reason == "heartbeat_stale":
                disengage_reason = "heartbeat_stale"
                engaged = False

            # Driver override detect (optional): large steering_input while we command ≠ 0
            if engaged and steer_in is not None and abs(float(steer_in)) > 0.55 and abs(cmd.steer) < 0.2:
                engaged = False
                disengage_reason = "driver_override"
                cmd = safe_command(
                    engaged=False,
                    heartbeat_ok=True,
                    path_debug_preview=True,
                    allow_preview_drive=False,
                    seq=cmd_seq,
                )

            applied = actuator.apply(cmd) if cmd.reason == "ok" else actuator.stop(seq=cmd_seq, reason=cmd.reason)

            st = default_state(
                engaged=engaged,
                disengage_reason=disengage_reason,
                policy="modular",
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
            st["cam_health"] = bundle.health_str()
            st["capture_backend"] = bundle.backend
            st["capture_note"] = bundle.note
            st["rss_mb"] = _rss_mb()
            st["detector"] = pout.detector_name
            st["actuator"] = actuator.name
            st["cmd_seq"] = int(applied.seq)
            st["cmd_reason"] = applied.reason
            st["cmd_applied"] = bool(applied.applied)
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
                elif key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                    ui.set_layer(int(chr(key)))

            frame_i += 1
            dt = 1.0 / max(args.hz, 1.0)
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t0)))
    except KeyboardInterrupt:
        print("[GVD] stopped.")
    finally:
        try:
            actuator.stop(seq=cmd_seq + 1, reason="shutdown")
        except Exception:
            pass
        try:
            write_engage_flag(False)
        except Exception:
            pass
        backend.close()
        if win is not None:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
