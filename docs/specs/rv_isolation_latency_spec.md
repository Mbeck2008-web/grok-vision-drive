# RV step 3: `check_safety_loop_latency_unchanged` gate (pre-registered)

Fixed before any result was collected with this method (critic requirement).
Parent spec: `docs/specs/reality_view_3dgs.md`, budget row "Safety loop"
(loop Hz and `infer_ms` p95 change < 5% with RV on vs off) and step 3.

## Loop under test
`_StubSafetyLoop` in `scripts/test_rv_isolation.py`: paced 50 Hz (20 ms).
Per tick: heartbeat stamp (`perf_counter`), fixed-work stand-in infer (pure
Python, calibrated to ~3 ms once per process), real
`python.runtime.shadow.shadow_tick` (AEB veto, heartbeat_stale, engage gate),
real `NullActuator.apply/stop`, then `RvHost.after_apply` (RV runs only), then
sleep to the next deadline.

## Conditions
- OFF: no RV host.
- gen: `RvHost` + `rv_isolation_child --mode slow_generator` (80 ms sleep
  stand-in for packing per output, 30 ms stall mid-write of the output slot).
- renderer: `RvHost` + `rv_isolation_child --mode slow_renderer` (holds the
  input slot reader-busy 250 ms per read, then hangs holding it).

## Design (fixed)
- One discarded OFF warm-up run.
- `ROUNDS = 7` reps; each rep runs OFF, gen, OFF, renderer in that order.
  Fresh child per run. `WARMUP_TICKS = 25` discarded, `MEASURE_TICKS = 150`
  measured per run.
- Pairing: each child run is paired with the OFF run immediately before it.
  7 pairs per child.

## Primary gate (wall clock)
For each child and each pair k: `r_k = p95(child run k) / p95(paired OFF run k)`
for (i) stand-in `infer_ms` wall and (ii) loop period wall. p95 = nearest-rank
on the 150 measured ticks. **Gate: `median(r_1..r_7) <= 1.05`** for each metric
and each child. No other wall statistic is substituted.

## Additional gates (on top, not replacements)
- Loop Hz: median of per-pair Hz ratios in [0.95, 1.05].
- Safety-path thread CPU per tick (heartbeat stamp .. actuator.apply; Linux
  `time.thread_time`, win32 `QueryThreadCycleTime`) at p10 per run: median of
  per-pair ratios <= 1.05.
- RV share of supervisor CPU (after_apply CPU / whole-tick CPU, same run) < 5%,
  median over runs.
- `after_apply` wall p95 < 1 ms (5% of the period), median over runs.
- Structural: no `Popen.wait/communicate` on the loop; hung renderer must cause
  backpressure drops; generator stall must cause dropped torn/busy reads; host
  must not degrade on a slow-but-alive child.

## If the primary gate cannot hold on the box
Do not loosen or swap the statistic. Keep per-rep logs
(`notes/rv_isolation_latency_reps*.log`) and report BLOCKED with the numbers.
