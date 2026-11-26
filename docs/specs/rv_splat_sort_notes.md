# Reality View splat depth sort: 16-bit key (`python/viz/rv_splat_sort.py`)

Assignment: GVD swarm-4 `gvd/rv-radix-sort`. Replace the CPU depth sort of the
300k-splat live pass (p95 52 ms on IQU45 against the 33 ms frame cap) with a
16-bit depth-key radix sort, same ordering contract.

## Where the old sort lived

The bench is now vendored at `tools/rv_gl33_splat_bench.py` and wired
(`order = sort_splats(d, 0.05, 200.0)`, same near/far as the bench projection).
The original copy remains in `/workspace/rv-preview/tools` (run on IQU45):

```
d     = (pos - eye) @ fwd                 # float32 view-space depth along forward
order = sort_splats(d, 0.05, 200.0)       # farthest first (painter order)
```

Previously `order = np.argsort(-d, kind="stable")`. No splat sort existed in
the repo at the base commit, and `gvd_viz` is not touched here. So this adds a
new module with the same contract. `sort_splats(depth, near, far, descending=True)`
is the drop-in for the `argsort(-d)` line. `reality_view_c.py` has its own
`np.lexsort((htie, -hz))` for billboards; it is a different sort (per-pixel
overwrite) and is not changed.

## Interface

```
quantize_depth(depth, near, far) -> uint16 array     # NaN rejected (ValueError)
quantum_m(near, far) -> float                         # (far - near) / 65535
sort_splats(depth, near, far, descending=True) -> intp array
```

## Invariants (the tests pin each)

1. **Permutation.** The output is a permutation of `0..N-1` (all indices,
   including NaN ones), dtype `np.intp`, shape `(N,)`. Same length as the input so
   `pos[order]` keeps working.
2. **Quantization.** `depth` is read as float64. `c = clip(depth, near, far)`,
   `t = (c - near) / (far - near)`, `q = floor(t * 65535 + 0.5)` as `uint16`.
   Round half up, monotone non-decreasing in depth, `q(near) = 0`, `q(far) = 65535`.
   `near` and `far` must be finite, `near < far`, not bool, else `ValueError`.
   Depths below `near` (and `-inf`) land in bin 0; depths above `far` (and `+inf`)
   in bin 65535. The renderer culls both, so their relative order is moot.
3. **Order.** With `qkey` the quantized keys of the non-NaN entries:
   - `descending=True` (painter order, farthest first, what the bench does):
     `order == np.argsort(-qkey.astype(int32), kind="stable")`, i.e. larger `q`
     first and equal `q` in ascending input-index order.
   - `descending=False`: `order == np.argsort(qkey, kind="stable")`.
4. **NaN.** Same as today's `argsort(-d)`: NaN entries go **last**, in ascending
   input-index order, in both directions. They are never dropped. `+inf`/`-inf` are
   clipped as in (2). (Today `+inf` sorts first and `-inf` last of the non-NaN, which
   is the same place bin 65535 / bin 0 puts them.)
5. **Empty** input gives an empty `intp` array. A single element gives `[0]`.
   **All-equal** keys give the identity order `0..N-1` in both directions.
6. **No mutation** of the input array. Deterministic: no RNG, no clock, no global state.
7. Input must be 1-D with a real numeric dtype (float or int, not bool/complex).

## Behavior change vs today (named on purpose)

The old sort compares float32 depths. The new one compares 16-bit keys, so
**depths that fall in the same quantum now tie** and are ordered by input index
instead of by their float depth.

- Quantum = `(far - near) / 65535`. For the bench range (near 0.05 m, far 200 m):
  **3.05 mm**. For a 64 m viz range (near 0.05): **0.98 mm**.
- Worst-case misorder: two splats are swapped relative to the float order only when
  they are in the same bin, so their depths differ by **at most one quantum**
  (3.05 mm at the bench range). Each pair across different bins keeps its exact float
  order. This holds for in-range depths; clipped out-of-range depths are culled anyway.
- Tests: `check_float_order_differs_only_within_quantum` shows the new order differs
  from the float argsort only between splats within one quantum, and
  `check_render_instrument_*` renders a fixed scene both ways (see the report for the
  numbers). At the bench range a depth difference of 3 mm is below what the 32-bit
  float depth buffer or the 8-bit colour can show for non-coincident splats; coincident
  splats (same bin, overlapping footprint, different colour) can still blend in a
  different order. That is the accepted cost, and the CPU renderer test measures it.

## Measured cost of the tie rule (CPU render of one fixed scene, test-run numbers)

`check_render_instrument` composites a seeded 5000-splat scene (96x72 px, 5x5 Gaussian
footprints, the bench's premultiplied-alpha pass with the `z*a` depth channel) once in the
old float `argsort(-d)` order and once in the new order. This is a test mimic of the bench
GL pass, not the repo renderer: no CPU splat renderer with an order hook exists on main
(`rv_a_gen.render_sh0` is not merged; `reality_view_c` sorts internally).

- Natural scene (random float depths, 852 of 5000 positions differ): max per-pixel depth
  diff **0.000254 m** (quantum 0.003051 m), max colour diff **21.6/255**, alpha diff 0.
- Adversarial scene (400 coincident splats in one quantum on top): max depth diff
  **0.001571 m**, max colour diff **117.3/255**, alpha diff 0. Proven bound: depth diff
  <= one quantum, because total transmittance per bin is order independent.
- All depths > 1 quantum apart: renders bit-identical.

So the depth instrument moves by at most one quantum (3 mm), but colour can differ at
pixels where splats that are within 3 mm of each other overlap. That is the accepted
behavior change; GVD decides whether it is acceptable on IQU45.

## Why `np.argsort(uint16, kind="stable")`

NumPy (2.5.4 here) implements stable sorts of 16-bit-or-smaller integers as a radix
sort. A hand-written bincount/cumsum counting sort cannot be vectorised stably without
a sort for the within-bucket rank, so it would be a Python-level loop. Measured on the
box (300k, local, loaded box, indicative only): float `argsort(-d, stable)` p50 ~35 ms,
quantize ~2 ms, `argsort(u16, stable)` ~3 ms, 2-pass 8-bit LSD ~5 ms. The single
16-bit stable argsort is the fastest exactly-equivalent option, so that is what is
used. Descending order uses the key `~q` (65535 - q), which keeps the stable
tie-break (input index ascending).

## Perf gate

Target: p95 of the sort <= **15 ms** at 300k on CPU, fixed, not tuned after a run.
The official measurement is GVD's run of the same bench on IQU45, with an A/A <= 1.03
precondition. The box number from `check_perf_proxy` is a **proxy only**: it runs A/A
first (float argsort vs float argsort, 7 interleaved pairs of 50 runs, same p95
statistic) and reports the max A/A ratio. If that exceeds 1.03 the box is too noisy,
and the check says so and skips visibly. It does not pass silently. Otherwise it
asserts new-sort p95 <= 15 ms.

## Out of scope / not touched

No flag, no GPU selection, no GL, no recorder, no `_scene_depth` / `depths_for`, nothing
on the heartbeat / AEB / engage path. The module imports only numpy and math (an AST check pins it).
