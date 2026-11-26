#!/usr/bin/env python3
"""Dev-only mutant harness for scripts/test_rv_isolation.py (Reality View Step 3).

Not part of run_offline_suite.py (not named test_*). Same rules as
scripts/mutants_rv_contract.py: each mutant is an exact text edit, py_compile'd
(non-compiling = INVALID, never KILLED), the test is run, files are restored
byte-for-byte. One line per mutant: KILLED / SURVIVED / EQUIVALENT / INVALID /
NOT_RUN. Exit 0 only if every run mutant is KILLED or EQUIVALENT.

Each mutant names the checks it runs (``--only`` on the test) so a mutant aimed
at the crash / copy-only checks is not scored through the slow latency check.

Usage:  python scripts/mutants_rv_isolation.py [--only NAME ...] [--verbose]
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
TEST = "scripts/test_rv_isolation.py"
ISO = "python/viz/rv_isolation.py"
LAT = "check_safety_loop_latency_unchanged"
CRASH = "check_renderer_crash_degrades"
COPY = "check_supervisor_after_apply_is_copy_only"


@dataclass
class Mutant:
    name: str
    edits: list[tuple[str, str, str]]
    checks: tuple[str, ...] = (CRASH, COPY)
    equivalent_proof: str = ""
    note: str = ""
    platforms: tuple[str, ...] = field(default_factory=tuple)


MUTANTS: list[Mutant] = [
    Mutant(
        "I1_blocking_wait_on_child",
        [(ISO, "            if not self.client.is_alive():\n                proc", "            if self.client._proc is None or self.client._proc.wait() is not None:\n                proc")],
    ),
    Mutant(
        "I2_blocking_submit_on_backpressure",
        [(ISO, "        if self.reader_busy():\n            self.dropped += 1\n            return False", "        while self.reader_busy():\n            time.sleep(0.001)")],
        checks=(LAT,),
    ),
    Mutant(
        "I2b_bounded_4ms_wait_on_backpressure",
        [(ISO, "        if self.reader_busy():\n            self.dropped += 1\n            return False", "        if self.reader_busy():\n            time.sleep(0.004)\n            self.dropped += 1\n            return False")],
        checks=(LAT,),
    ),
    Mutant(
        "I3_swallow_crash_without_degrade",
        [(ISO, "                self._degrade(f\"child exited (rc={rc})\")\n                return False", "                return True")],
    ),
    Mutant(
        "I4_retain_slot_reference",
        [(ISO, "                    self._scratch, self._view = self._view, self._scratch", "                    self._view = memoryview(self._out._mm)[RV_CTRL_SIZE : RV_CTRL_SIZE + n]")],
    ),
    Mutant(
        "I5_exception_escapes_on_child_death",
        [(ISO, "                self._degrade(f\"child exited (rc={rc})\")\n                return False", "                raise RuntimeError(f\"rv child died rc={rc}\")")],
        note="after_apply's outer except would catch a raise inside the try; mutant also drops the catch-all",
    ),
    Mutant(
        "I6_degrade_to_black",
        [(ISO, "        if self.degraded or self._view_len == 0:\n            return img", "        if self.degraded or self._view_len == 0:\n            return img * 0")],
    ),
    Mutant(
        "I7_full_parse_on_apply",
        [(ISO, "                    hdr = RvShmHeader.unpack(mv[:SLOT_HEADER_SIZE])", "                    from python.viz.rv_contract import unpack_output_slot\n                    hdr, _spec = unpack_output_slot(mv[:n])")],
    ),
    Mutant(
        "I8_skip_header_validate",
        [(ISO, "                    hdr = RvShmHeader.unpack(mv[:SLOT_HEADER_SIZE])", "                    hdr = type(\"H\", (), {\"flags\": RV_FLAG_HAS_DEPTH, \"width\": 64, \"height\": 48, \"payload_bytes\": n - SLOT_HEADER_SIZE})()")],
    ),
    Mutant("I9_accept_torn_read", [(ISO, "        if s1 & 1:\n            return -1, last_seq\n", "")]),
    Mutant(
        "I10_no_seq_recheck_after_copy",
        [(ISO, "        if self.seq() != s1:\n            return -1, last_seq\n", "")],
        checks=(COPY,),
    ),
    Mutant(
        "I11_poll_health_never",
        [(ISO, "            if not self.client.is_alive():\n                proc", "            if False:\n                proc")],
    ),
    Mutant(
        "I12_degrade_note_every_call",
        [(ISO, "        if self.degraded:\n            return\n        self.degraded = True", "        self.degraded = True"),
         (ISO, "            if self.degraded:\n                return False\n            if not self.client.is_alive()", "            if not self.client.is_alive()")],
    ),
    Mutant(
        "I13_generation_on_supervisor",
        [(ISO, "            n, s = self._out.try_copy_into(self._scratch, self._last_out_seq)", "            from python.viz.rv_isolation_child import _packed_frame\n            _packed_frame(seq)\n            n, s = self._out.try_copy_into(self._scratch, self._last_out_seq)")],
    ),
]

# I5 needs the catch-all gone too, otherwise the raise is converted to a degrade.
MUTANTS[[m.name for m in MUTANTS].index("I5_exception_escapes_on_child_death")].edits.append(
    (ISO, "        except Exception as exc:  # noqa: BLE001 - RV never breaks the supervisor", "        except ZeroDivisionError as exc:  # mutant I5")
)


def _run_test(checks: tuple[str, ...], timeout_s: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-B", TEST, "--only", *checks], cwd=str(ROOT), capture_output=True, text=True, timeout=timeout_s
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired as exc:
        return 124, f"TIMEOUT after {exc.timeout}s"


def _apply(m: Mutant, originals: dict[Path, bytes]) -> str:
    for rel, old, new in m.edits:
        path = ROOT / rel
        if path not in originals:
            originals[path] = path.read_bytes()
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            return f"pattern must match exactly once in {rel} (found {text.count(old)}): {old[:80]!r}"
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return ""


def _restore(originals: dict[Path, bytes]) -> None:
    for path, data in originals.items():
        path.write_bytes(data)


def _compiles(paths: list[Path]) -> str:
    for p in paths:
        with tempfile.TemporaryDirectory() as td:
            try:
                py_compile.compile(str(p), cfile=str(Path(td) / "x.pyc"), doraise=True)
            except py_compile.PyCompileError as exc:
                return str(exc).strip().splitlines()[-1]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip-checks", nargs="*", default=(), help="do not run mutants that need these checks")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args()

    selected = [m for m in MUTANTS if args.only is None or m.name in args.only]
    needed = sorted({c for m in selected for c in m.checks if c not in args.skip_checks})
    rc0, out0 = _run_test(tuple(needed), args.timeout)
    print(f"BASELINE rc={rc0} checks={needed} platform={sys.platform} python={sys.version.split()[0]}", flush=True)
    if rc0 != 0:
        print(out0)
        print("BASELINE FAILED: refusing to score mutants", flush=True)
        return 2

    counts: dict[str, int] = {}
    bad = 0
    for m in selected:
        skipped = [c for c in m.checks if c in args.skip_checks]
        if m.platforms and not any(sys.platform.startswith(p) for p in m.platforms):
            status, detail = "NOT_RUN", f"platform-bound to {m.platforms}"
        elif skipped:
            status, detail = "NOT_RUN", f"needs skipped check(s) {skipped}"
        else:
            originals: dict[Path, bytes] = {}
            try:
                perr = _apply(m, originals)
                err = "" if perr else _compiles(list(originals))
                if perr:
                    status, detail = "INVALID", perr
                elif err:
                    status, detail = "INVALID", f"does not compile: {err}"
                else:
                    rc, out = _run_test(m.checks, args.timeout)
                    tail = [ln for ln in out.splitlines() if ln.strip()]
                    reason = next((ln for ln in tail if ln.startswith("FAIL ") or "TIMEOUT" in ln), tail[-1] if tail else "")
                    if rc != 0:
                        status, detail = "KILLED", f"rc={rc} {reason.strip()[:220]}"
                    elif m.equivalent_proof:
                        status, detail = "EQUIVALENT", m.equivalent_proof
                    else:
                        status, detail = "SURVIVED", "test_rv_isolation still passes"
                    if args.verbose:
                        print("\n".join(tail[-8:]))
            finally:
                _restore(originals)
        if m.note:
            detail += f" [note: {m.note}]"
        counts[status] = counts.get(status, 0) + 1
        if status not in ("KILLED", "EQUIVALENT", "NOT_RUN"):
            bad += 1
        print(f"{status:<10} {m.name:<38} {detail}", flush=True)

    summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"SUMMARY total={len(selected)} {summary}", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
