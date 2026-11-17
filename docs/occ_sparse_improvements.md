# occ_sparse: ranked improvements

Ranked by expected value per unit of work. All current trained-model numbers are SYNTHETIC
or CPU ESTIMATES; real-data, GPU memory and GPU latency are UNPROVEN.

1. **Train on occluded GT (multi-frame aggregated labels).** The synthetic occluded score is
   0 (model 0.005 vs per-cell prior 0.934). Add occluded targets from neighbouring frames
   (pose-aligned label aggregation, training only) to `make_queries`, plus the falsifier:
   masking occluded GT out of the loss must make the occluded test fail. Not done here.
2. **Real-shard occluded scoring loop** in `offline_eval --shards` (currently validates the
   split and prints UNPROVEN). Needs GVD's held-out map/region split.
3. **Measure GPU memory on IQU45** (nvidia-smi during `train_occ` / offline infer at batch 1
   and 8) to replace every CPU ESTIMATE; then decide fp16 vs fp32 by measurement.
4. **Direct VO in 3 DoF on real data**: add pitch compensation (flat-ground assumption),
   a multi-camera cost (main + wide), and a lateral/yaw accuracy test (synthetic world only
   moves forward today). Fallback currently reuses the last step indefinitely; cap it.
5. **Learned proposals** for unseen cells (replace or augment the fixed 192-cell lattice),
   each source keeping an absolute cap and the grid-growth test.
6. **Separate lane head** (polylines are derived from semantic cells today: a deviation).
7. **Resolution lever study on real clips** (front_main_hi vs main_wide_narrow); synthetic
   IoU differences (0.070 vs 0.059) are too small and too synthetic to decide.
8. **int8 TensorRT** for the backbone only (sparse stages stay fp16/fp32). UNPROVEN.
9. **Mutation testing on model.py** (only the numpy modules and the occ_out validator were
   mutated here; torch mutants are too slow on this shared CPU box).

## Contrarian proposals

| claim | evidence / cost | test that would falsify it |
|---|---|---|
| A learned VO head is unnecessary for v1; direct sparse photometric VO is enough | two learned heads failed on synthetic clips; direct VO drift 0.045 m / 10 m, no params, 384 samples | real clips: integrated VO drift over 100 m > 5% with conf >= 0.3 |
| Nothing dense needs to cross ticks | cache = sparse cells + 384 grey samples (0.072 MB) | an ablation that caches the 1/8 map and beats sparse memory on occluded IoU on GVD's split |
| Lanes as polylines from semantic cells beat a lane head at this size | no lane labels beyond paint strokes; zero extra params | a lane head of <= 50k params with higher polyline F1 on held-out real clips |
| The occluded metric is dominated by layout priors on straight roads | prior 0.934 vs model 0.005 on synthetic | held-out intersections/roundabouts where the prior falls below the memory model |
| Real-time is the wrong gate for this net | Michael relaxed it; latency reported only | an online consumer that needs occ_out within one tick |
