#!/usr/bin/env python3
"""Dev-only mutant harness for scripts/test_rv_contract.py (Reality View Step 1).

Not part of run_offline_suite.py. Each mutant is an exact text edit applied to the
working tree, checked with py_compile (a mutant that does not compile is INVALID,
never KILLED), then scripts/test_rv_contract.py is run; files are restored byte-for-byte
afterwards. Prints one line per mutant: KILLED / SURVIVED / EQUIVALENT / INVALID.
Exit 0 only if every mutant is KILLED or EQUIVALENT (with a written proof).

Usage:  python scripts/mutants_rv_contract.py [--only NAME ...] [--verbose]
"""
from __future__ import annotations

import argparse
import py_compile
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = "scripts/test_rv_contract.py"
STUB = "python/viz/rv_renderer_stub.py"
CONTRACT = "python/viz/rv_contract.py"


@dataclass
class Mutant:
    name: str
    edits: list[tuple[str, str, str]]
    equivalent_proof: str = ""
    note: str = ""
    platforms: tuple[str, ...] = field(default_factory=tuple)  # empty = all


MUTANTS: list[Mutant] = [
    Mutant("M1_flag_default_ON", [("config/cameras.yaml", "  reality_view_enabled: false", "  reality_view_enabled: true")]),
    Mutant(
        "M2_placeholder_1byte_diff",
        [("python/viz/view_select.py", "PLACEHOLDER_BG = (24, 20, 18)", "PLACEHOLDER_BG = (25, 20, 18)")],
    ),
    Mutant(
        "M3_import_renderer_in_process",
        [
            (
                "python/run_vision.py",
                "from python.viz.view_select import ViewChoice, render_main_view, resolve_view",
                "from python.viz.view_select import ViewChoice, render_main_view, resolve_view\nimport python.viz.rv_renderer_stub  # mutant",
            )
        ],
    ),
    Mutant(
        "M4_drop_shm_version_check",
        [
            (
                CONTRACT,
                "def _require_version(v: Any) -> int:\n    if isinstance(v, bool) or not isinstance(v, int) or v != RV_SCHEMA_VERSION:\n        raise RvSchemaError(f\"unsupported RV schema_version {v!r}; expected {RV_SCHEMA_VERSION}\")\n    return v",
                "def _require_version(v: Any) -> int:\n    return int(v) if isinstance(v, int) else RV_SCHEMA_VERSION",
            )
        ],
    ),
    Mutant("S3_accept_aux_depth", [(CONTRACT, '        "aux_depth",\n', "")]),
    Mutant(
        "S5_accept_nonfinite_K",
        [
            (
                CONTRACT,
                '        _check_matrix(cam.intrinsics_3x3, (3, 3), f"{cam.cam_id}.intrinsics_3x3")',
                "        np.asarray(cam.intrinsics_3x3, dtype=np.float32)  # mutant S5",
            )
        ],
    ),
    Mutant(
        "S7a_blocking_lock_acquire",
        [(CONTRACT, "        if not self._lock.acquire(blocking=False):", "        if not self._lock.acquire():")],
    ),
    Mutant(
        "H4_drop_header_unpack_version_check",
        [(CONTRACT, "        version = _require_version(unpacked[1])", "        version = int(unpacked[1])  # mutant H4")],
    ),
    Mutant(
        "slot_stays_1MiB",
        [
            (
                CONTRACT,
                "    def __init__(self, slot_bytes: int = RV_DEFAULT_SLOT_BYTES, *, module: str = RV_RENDERER_MODULE) -> None:",
                "    def __init__(self, slot_bytes: int = 1024 * 1024, *, module: str = RV_RENDERER_MODULE) -> None:",
            )
        ],
    ),
    Mutant(
        "eager_import_rv_contract",
        [
            (
                "python/run_vision.py",
                "    if not _reality_view_flag_on(args, cameras):\n        return None",
                "    from python.viz.rv_contract import RvClient, reality_view_enabled  # mutant\n    if not _reality_view_flag_on(args, cameras):\n        return None",
            )
        ],
    ),
    Mutant(
        "no_pdeathsig",
        [
            (CONTRACT, "        prctl_fn(_PR_SET_PDEATHSIG, sig, 0, 0, 0)", "        pass  # mutant no_pdeathsig"),
            (STUB, "        if prctl(_PR_SET_PDEATHSIG, sig, 0, 0, 0) == 0:", "        if False:  # mutant no_pdeathsig"),
        ],
    ),
    Mutant(
        # The ppid poll never reports an exit (the r2 version of this mutant was a
        # syntax error; this one compiles). Linux reaches it via injected ENOSYS (case E).
        "win_fallback_no_ppid_poll",
        [
            (
                STUB,
                "    if os.getppid() != int(val):\n        return True\n    if timeout_s > 0:\n        time.sleep(timeout_s)\n        return os.getppid() != int(val)\n    return False",
                "    if timeout_s > 0:\n        time.sleep(timeout_s)\n    return False  # mutant win_fallback_no_ppid_poll",
            )
        ],
    ),
    Mutant(
        # Skeptic 9.1: pidfd_open always fails with ENOSYS inside the stub.
        "force_pidfd_enosys",
        [
            (
                STUB,
                '        return ("pidfd", os.pidfd_open(int(ppid), 0))',
                '        raise OSError(errno.ENOSYS, "mutant force_pidfd_enosys")',
            )
        ],
        note="with the r3 fix this no longer exits under a live parent; killed by the armed mode=pidfd check",
    ),
    Mutant(
        # r2 behaviour: any pidfd_open OSError -> 'parent gone' -> stub exits under a live parent.
        "pidfd_any_oserror_exits",
        [(STUB, "        if exc.errno == errno.ESRCH:\n            return None\n        return (\"ppid\", int(ppid))", "        return None  # mutant")],
    ),
    Mutant(
        "pidfd_esrch_falls_through",
        [(STUB, "        if exc.errno == errno.ESRCH:\n            return None\n", "")],
    ),
    Mutant(
        "win_any_open_error_exits",
        [(STUB, "    if err == _ERROR_INVALID_PARAMETER:\n        return None\n    return (\"ppid\", int(ppid))", "    return None  # mutant")],
    ),
    Mutant(
        "win_err87_falls_through",
        [(STUB, "    if err == _ERROR_INVALID_PARAMETER:\n        return None\n", "")],
    ),
    Mutant(
        "remove_pidfd_path",
        [(STUB, "    if hasattr(os, \"pidfd_open\"):\n        return _open_pidfd_watch(ppid)\n", "")],
    ),
    Mutant(
        "no_armed_line",
        [(STUB, '    _emit(f"RV_WATCH_ARMED mode=', '    (f"RV_WATCH_ARMED mode=')],
    ),
    Mutant(
        "drop_parent_pid",
        [
            (
                CONTRACT,
                '[sys.executable, "-m", self.module, "--serve", "--parent-pid", str(os.getpid())]',
                '[sys.executable, "-m", self.module, "--serve"]',
            )
        ],
    ),
    Mutant(
        "drop_close_fds",
        [(CONTRACT, '            popen_kw: dict[str, Any] = {"close_fds": True}', "            popen_kw: dict[str, Any] = {}")],
        equivalent_proof=(
            "subprocess.Popen defaults close_fds=True on every platform since Python 3.7 "
            "(bpo-19764); CI and IQU45 run 3.12, so removing the explicit kwarg is a no-op. flip_close_fds_false is the killable variant."
        ),
    ),
    Mutant(
        "flip_close_fds_false",
        [(CONTRACT, '            popen_kw: dict[str, Any] = {"close_fds": True}', '            popen_kw: dict[str, Any] = {"close_fds": False}')],
    ),
    Mutant(
        "drop_prctl_ppid_race_check",
        [(CONTRACT, "        if os.getppid() != expected_ppid:\n            os._exit(0)\n", "")],
    ),
]


def _run_test(timeout_s: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-B", TEST], cwd=str(ROOT), capture_output=True, text=True, timeout=timeout_s
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired as exc:
        return 124, f"TIMEOUT after {exc.timeout}s"


def _apply(m: Mutant) -> dict[Path, bytes]:
    originals: dict[Path, bytes] = {}
    for rel, old, new in m.edits:
        path = ROOT / rel
        if path not in originals:
            originals[path] = path.read_bytes()
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise RuntimeError(f"{m.name}: pattern must match exactly once in {rel} (found {text.count(old)}): {old!r}")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return originals


def _restore(originals: dict[Path, bytes]) -> None:
    for path, data in originals.items():
        path.write_bytes(data)


def _compiles(paths: list[Path]) -> str:
    for p in paths:
        if p.suffix != ".py":
            continue
        with tempfile.TemporaryDirectory() as td:
            try:
                py_compile.compile(str(p), cfile=str(Path(td) / "x.pyc"), doraise=True)
            except py_compile.PyCompileError as exc:
                return str(exc).strip().splitlines()[-1]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    rc0, out0 = _run_test(args.timeout)
    print(f"BASELINE rc={rc0} platform={sys.platform} python={sys.version.split()[0]}", flush=True)
    if rc0 != 0:
        print(out0)
        print("BASELINE FAILED: refusing to score mutants", flush=True)
        return 2

    selected = [m for m in MUTANTS if args.only is None or m.name in args.only]
    counts: dict[str, int] = {}
    bad = 0
    for m in selected:
        if m.platforms and not any(sys.platform.startswith(p) for p in m.platforms):
            status, detail = "NOT_RUN", f"platform-bound to {m.platforms}"
        else:
            originals = _apply(m)
            try:
                err = _compiles(list(originals))
                if err:
                    status, detail = "INVALID", f"does not compile: {err}"
                else:
                    rc, out = _run_test(args.timeout)
                    tail = [ln for ln in out.splitlines() if ln.strip()]
                    reason = next((ln for ln in reversed(tail) if "Error" in ln or "TIMEOUT" in ln), tail[-1] if tail else "")
                    if rc != 0:
                        status, detail = "KILLED", f"rc={rc} {reason.strip()[:200]}"
                    elif m.equivalent_proof:
                        status, detail = "EQUIVALENT", m.equivalent_proof
                    else:
                        status, detail = "SURVIVED", "test_rv_contract still passes"
                    if args.verbose:
                        print("\n".join(tail[-8:]))
            finally:
                _restore(originals)
        if m.note:
            detail += f" [note: {m.note}]"
        counts[status] = counts.get(status, 0) + 1
        if status not in ("KILLED", "EQUIVALENT"):
            bad += 1
        print(f"{status:<10} {m.name:<36} {detail}", flush=True)

    summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"SUMMARY total={len(selected)} {summary}", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
