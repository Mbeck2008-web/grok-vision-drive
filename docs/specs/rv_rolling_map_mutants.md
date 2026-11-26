# rv_rolling_map mutmut survivors (mutmut 3.8.1, head 6b8e4a3)

Run 2: 705 mutants, 691 killed, 14 survived. Each survivor is equivalent (no input can tell it apart); one line each.
Evidence: `mutmut results` / `mutmut show` in the swarm-4 mutation workspace (module and test byte-identical to the branch).

| Mutant | Change | Why equivalent |
| --- | --- | --- |
| `_check_boxes__mutmut_53`, `_55` | `np.array([l,w,h], dtype=np.float64)` -> `dtype=None` / no dtype | `vals` come from `_finite_scalar`, which always returns a Python `float`; a list of floats infers float64. |
| `_check_boxes__mutmut_62`, `_64` | `np.array(vals[:3], dtype=np.float64)` -> `dtype=None` / no dtype | Same: `vals[:3]` are Python floats, inferred float64. |
| `merge__mutmut_58`, `_60` | `np.ones((n,), dtype=np.float64)` -> `dtype=None` / no dtype | `np.ones` default dtype is float64. |
| `merge__mutmut_73`, `_83`, `_209`, `_222` | `.astype(np.float64)` -> `.astype(None)` | NumPy maps `dtype=None` to the default float, float64 (checked: int64/uint8 `.astype(None)` -> float64). |
| `merge__mutmut_31`, `_65` | `kind not in "fiu"` -> `kind not in "XXfiuXX"` | `dtype.kind` is one character from `biufcmMOSUV`; `X` is never a kind, so the substring test is unchanged. |
| `merge__mutmut_345` | `keys.shape[0] > n_max` -> `>=` | At equality `excess = 0`, `order[:0]` is empty, `sel` stays all True and `n_cap = 0` (its initial value): identical arrays and stats. |
| `merge__mutmut_375` | `sel[order[:excess]] = False` -> `= None` | `sel` is a bool array; assigning `None` casts to `False`. |
