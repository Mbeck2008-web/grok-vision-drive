# rv_splat_sort mutation testing (mutmut 3.8.1)

Run on `python/viz/rv_splat_sort.py` with `scripts/test_rv_splat_sort.py` as the only
test file. Two checks are deselected with `-k`: `test_perf_proxy` (timing, noisy and slow
under mutation) and `test_module_is_numpy_only_and_safe` (mutmut injects its own import
into the module, so an import-whitelist check cannot run on a mutated copy).

Result: **126 mutants, 121 killed, 5 survived, all 5 equivalent** (below). The first run
had 18 survivors and a later run still had one real survivor; they were killed by tests
(error-message checks, and a 6000-element tie-heavy NaN case for the NaN branch, where
dropping `kind="stable"` fell back to numpy's non-stable default).

| Mutant | Change | Why it is equivalent |
|---|---|---|
| `_check_depth` #7 | `"fiu"` -> `"XXfiuXX"` in `d.dtype.kind not in ...` | `dtype.kind` is one character from `biufcmMOSUV`; none is `X`, so membership is unchanged. |
| `_quantize` #8 | `.astype(np.float64)` -> `.astype(None)` | `ndarray.astype(None)` means the default dtype, float64 (checked: `np.zeros(2, np.float32).astype(None).dtype == float64`). |
| `sort_splats` #13 | `kind == "f"` -> `(kind == "f") or True` | Then `np.isnan` runs on int/uint arrays too. It returns all-False (checked on int8/int64/uint8/uint64), so `nan.any()` is False and the no-NaN path runs, same as before. |
| `sort_splats` #38 | `kind="stable"` -> `kind="STABLE"` (NaN branch) | NumPy reads the sort kind case-insensitively; `"STABLE"` is the same stable sort (checked: identical argsort on tied uint16 data). |
| `sort_splats` #54 | same, in the no-NaN branch | Same proof. |
