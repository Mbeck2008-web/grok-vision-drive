"""GVD supervisor — cameras + gvd_state.json + optional OpenCV GVD window."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.runtime.hw_probe import probe
from python.runtime.state_io import default_state, state_path, steer_preview_path_ego, write_state
from python.sensors.cameras import make_backend, resolve_backend_name
from python.viz.stage import VizUI, render_stage, smoke


def _rss_mb() -> float:
    try:
        import resource

        # Linux: ru_maxrss is KB
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        try:
            import psutil  # type: ignore

            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="GVD supervisor + cameras + viz")
    ap.add_argument("--viz", action="store_true", help="Open OpenCV GVD window")
    ap.add_argument("--smoke", action="store_true", help="Write docs/gvd_viz_smoke.png and exit")
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "beamngpy", "window", "stub"],
        help="Camera backend (default: beamngpy if importable else window else stub)",
    )
    ap.add_argument("--vision-only", action="store_true", help="Skip VRAM/BeamNG co-tenant warnings")
    args = ap.parse_args()

    if args.vision_only:
        os.environ["GVD_VISION_ONLY"] = "1"

    if args.smoke:
        # Offline smoke uses stub cameras — never requires BeamNG/QSV
        out = smoke()
        print(f"[GVD] smoke frame -> {out}")
        print(f"[GVD] state -> {state_path()}")
        return

    backend_name = resolve_backend_name(None if args.backend == "auto" else args.backend)
    hw = probe(backend=backend_name)
    print(hw.boot_line())
    for n in hw.notes or []:
        print(f"[GVD] note: {n}")

    backend = make_backend(backend_name if args.backend != "auto" else backend_name)
    backend.open()
    print(f"[GVD] camera backend={backend.name}")
    print(f"[GVD] state path: {state_path()}")
    print("[GVD] Engage in BeamNG with Alt+A. Keys: V nerd, 0-5 layers, ? help, q quit.")

    ui = VizUI()
    win = "GVD" if args.viz else None
    if win:
        import cv2  # noqa: F401

    frame_i = 0
    cam_hz_ema = 0.0
    last_cam_t = time.perf_counter()
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

            steer = 4.0 * ((frame_i // 30) % 5 - 2)
            st = default_state(
                engaged=True,
                loop_hz=args.hz,
                camera_hz=cam_hz_ema,
                path_conf=0.85,
                path_width=2.0,
                path_debug_preview=True,
                objects_n=0,
                tracks_n=0,
                planner={
                    "corridor_width": 2.0,
                    "curvature": 0.0,
                    "target_v": 10.0,
                    "ttc_lead": None,
                    "aeb": "off",
                },
                gpu_name=hw.dgpu,
                gpu_vram_total_gb=float(hw.dgpu_vram_gb or 11.0),
            )
            st["ego"]["steer_deg"] = steer
            st["path_ego"] = steer_preview_path_ego(steer, length_m=36.0)
            st["cam_health"] = bundle.health_str()
            st["capture_backend"] = bundle.backend
            st["capture_note"] = bundle.note
            st["rss_mb"] = _rss_mb()
            st["heartbeat_ms"] = (time.perf_counter() - loop_t0) * 1000.0
            # Keep Lua ribbon alive even with no frames (steer-preview)
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
        backend.close()
        if win is not None:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
