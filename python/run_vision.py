"""GVD supervisor entry — gvd_state.json + optional OpenCV GVD window."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.runtime.state_io import default_state, state_path, steer_preview_path_ego, write_state
from python.viz.stage import VizUI, render_stage, smoke


def main() -> None:
    ap = argparse.ArgumentParser(description="GVD supervisor stub + viz")
    ap.add_argument("--viz", action="store_true", help="Open OpenCV GVD window")
    ap.add_argument("--smoke", action="store_true", help="Write docs/gvd_viz_smoke.png and exit")
    ap.add_argument("--hz", type=float, default=15.0)
    args = ap.parse_args()

    if args.smoke:
        out = smoke()
        print(f"[GVD] smoke frame -> {out}")
        print(f"[GVD] state -> {state_path()}")
        return

    print("[GVD] supervisor stub — writing gvd_state.json for BeamNG path ribbon.")
    print(f"[GVD] state path: {state_path()}")
    print("[GVD] Engage in BeamNG with Alt+A. Keys: V nerd, 0-5 layers, ? help, q quit.")

    ui = VizUI()
    win = "GVD" if args.viz else None
    if win:
        import cv2  # noqa: F401 — needs opencv-python (GUI), not headless-only

    frame_i = 0
    try:
        while True:
            loop_t0 = time.perf_counter()
            steer = 4.0 * ((frame_i // 30) % 5 - 2)
            st = default_state(
                engaged=True,
                loop_hz=args.hz,
                camera_hz=10.0,
                path_conf=0.85,
                path_width=2.0,
                path_debug_preview=True,
                objects_n=0,
                tracks_n=0,
                planner={"corridor_width": 2.0, "curvature": 0.0, "target_v": 10.0, "ttc_lead": None, "aeb": "off"},
            )
            st["ego"]["steer_deg"] = steer
            st["path_ego"] = steer_preview_path_ego(steer, length_m=36.0)
            st["heartbeat_ms"] = (time.perf_counter() - loop_t0) * 1000.0
            write_state(st)

            if win is not None:
                import cv2

                frame = render_stage(st, ui=ui)
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
        if win is not None:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
