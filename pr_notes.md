# Reality View sky backdrop (render-only)

Michael 2026-11-26: "the sky is not visible only the road I would prefer everything to be visible."

The unpainted band under the horizon (ground range in (PRIOR_MAX_M, FAR_M]) was classed `unknown` and drawn black. With a sky colour it is now a 2-D backdrop (ground colour fading to horizon haze) counted as `far`. When no sky has been seen yet (tunnels), the colour falls back to the median of the main camera's top rows. Sky is never a 3-D point at max range, never at road height, and never enters the occupancy grid, CamDepthFilter, planner, or AEB.

## Safety (import isolation + byte-identical hashes)

The safety argument is both of:

1. **Import isolation (S21).** A static scan of every `.py` under `python/` fails if any of them imports `tools.rv_*`, `rv_stable_fill`, or `tools.rv_stable_fill`, in any form (`import`, `from … import`, `importlib.import_module`, or `__import__` with a literal). AEB lives in `python/control/actuate.py`, `python/control/override.py`, `python/planning/speed.py`, `python/perception/cipv.py`, and `python/runtime/frame_gate.py`. Those modules are in the scan. They cannot see the preview or `rv_stable_fill`, so a render-only backdrop cannot reach AEB, the planner, or the occupancy / filter path through an import.

2. **Byte-identical hashes with sky on vs off (S19).** sha256 of the ground-grid occupancy memory, every CamDepthFilter output, the point set handed to `compose`, the observed/filled masks, and the render's depth buffer are identical. The sky colour touches only `compose`'s colour image.

The point-set hash is one of those buffers. It is not a stand-in for planner or AEB inputs: those consumers live under `python/` and are covered by import isolation, not by treating compose's points as their API.

## Known issues until Trainer 2.0 (do not regress in this PR)

- **Wide rings remain.** With wide snap off (Michael 2026-11-26), the wide cam's net depth sits too high on the road (allowed-dev median |dz| 0.247 m at 5-10 m, 0.547 m at 10-15 m). This is a known issue until Trainer 2.0. Breaker adds the KNOWN_FAILING line; this PR does not edit KNOWN_FAILING.
- **pillarR / repeatR gate (b)** is a known failure (Tracer's rv_snapb trace). Breaker adds that line. `ROAD_SNAP_CAMS` and the 0.30 m rejection are unchanged.

Wide snap stays OFF by default and remains switchable.
