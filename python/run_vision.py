"""GVD supervisor entry — writes gvd_state.json and optional OpenCV GVD window."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.runtime.state_io import default_state, state_path, write_state


def main() -> None:
    ap = argparse.ArgumentParser(description="GVD supervisor stub + viz")
    ap.add_argument("--viz", action="store_true", help="Open OpenCV GVD window")
    ap.add_argument("--smoke", action="store_true", help="Write docs/gvd_viz_smoke.png and exit")
    ap.add_argument("--hz", type=float, default=15.0)
    args = ap.parse_args()

    if args.smoke:
        from python.viz.stage import smoke

        out = smoke(window=False)
        print(f"[GVD] smoke frame -> {out}")
        print(f"[GVD] state -> {state_path()}")
        return

    print("[GVD] supervisor stub — writing gvd_state.json for BeamNG path ribbon.")
    print(f"[GVD] state path: {state_path()}")
    print("[GVD] Engage in BeamNG with Alt+A to show ice-blue GVD PATH.")

    win = None
    if args.viz:
        import cv2
        from python.viz.stage import render_stage

        win = "GVD"

    t0 = time.time()
    frame_i = 0
    try:
        while True:
            loop_t0 = time.perf_counter()
            st = default_state(
                engaged=True,
                loop_hz=args.hz,
                camera_hz=10.0,
                path_conf=0.75,
                path_debug_preview=True,
                steer_deg=4.0 * ((frame_i // 30) % 5 - 2),
            )
            # keep preview path in sync with steer
            from python.runtime.state_io import steer_preview_path_ego

            st["path_ego"] = steer_preview_path_ego(float(st["ego"]["steer_deg"]))
            st["heartbeat_ms"] = (time.perf_counter() - loop_t0) * 1000.0
            write_state(st)

            if win is not None:
                import cv2
                from python.viz.stage import render_stage

                frame = render_stage(st)
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_i += 1
            dt = 1.0 / max(args.hz, 1.0)
            time.sleep(max(0.0, dt - (time.perf_counter() - loop_t0)))
            if time.time() - t0 > 3600 * 8:
                break
    except KeyboardInterrupt:
        print("[GVD] stopped.")
    finally:
        if win is not None:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
