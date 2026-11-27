#!/usr/bin/env python3
"""rec_loop.ps1 is parsed and run under pwsh (no grep of the script text).

Covers: the collector arg list (--long-clips, --long-clip-s, --session-id),
a real child process through Invoke-Batch (ExitCode must not be $null), the
outcome logic for a null / zero / non-zero exit code, the session that
belongs to THIS launch (not the newest folder), and the whole loop against a
stub collector (3 halted batches -> exit 2; STOP file -> exit 0).

    py -3 scripts/test_rec_loop.py

pwsh is looked for in GVD_PWSH, PATH, and /workspace/tools/pwsh/pwsh. Without
it the checks are skipped visibly (gvd_test_env.skip_check), never passed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts" / "rec_loop.ps1"


def find_pwsh() -> str | None:
    for cand in (os.environ.get("GVD_PWSH"), shutil.which("pwsh"), "/workspace/tools/pwsh/pwsh"):
        if cand and Path(cand).is_file():
            return cand
    return None


PWSH = find_pwsh()


def ps(code: str, *, timeout: float = 120.0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("DOTNET_SYSTEM_GLOBALIZATION_INVARIANT", "1")
    env["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1"
    return subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", f". '{SCRIPT}'; {code}"],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


def ps_ok(code: str) -> str:
    r = ps(code)
    assert r.returncode == 0, (code, r.returncode, r.stdout, r.stderr)
    return r.stdout.strip()


def write_session(out: Path, sid: str, *, clips: int, results: int, stop: str | None = None, session_id: str | None = None) -> Path:
    d = out / f"clips_{sid}"
    d.mkdir(parents=True)
    payload = {"clips": clips, "results": [{"dir": f"clip_{i:03d}"} for i in range(results)], "session_id": session_id or sid}
    if stop:
        payload["stop_reason"] = stop
    (d / "session.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


def check_arg_list() -> None:
    line = ps_ok(
        "(Get-BatchArgs -Collect 'c.py' -Out 'O' -Clips 60 -Maps 'a,b' -LongClips 8 "
        "-LongClipRange '600,900' -SessionId 'S1') -join '|'"
    )
    args = line.split("|")
    assert args[0] == "c.py", args
    for flag, val in (("--long-clips", "8"), ("--long-clip-s", "600,900"), ("--clips", "60"),
                      ("--out", "O"), ("--maps", "a,b"), ("--session-id", "S1")):
        assert flag in args and args[args.index(flag) + 1] == val, (flag, args)
    # The long-run default of the loop is 10-15 min clips with --long-clips on.
    dflt = ps_ok("(Get-BatchArgs -Collect c.py -Out O -Clips $Clips -Maps $Maps -LongClips $LongClips "
                 "-LongClipRange $LongClipRange -SessionId S) -join '|'").split("|")
    assert dflt[dflt.index("--long-clips") + 1] == "8", dflt
    assert dflt[dflt.index("--long-clip-s") + 1] == "600,900", dflt
    print("rec_loop args: --long-clips, --long-clip-s 600,900 and --session-id are on the command line")


def check_invoke_batch_exit_codes() -> None:
    py = sys.executable
    with tempfile.TemporaryDirectory() as td:
        slow = Path(td) / "slow.py"
        slow.write_text("import sys, time\ntime.sleep(1.5)\nsys.exit(int(sys.argv[1]))\n", encoding="utf-8")
        for want in (0, 3):
            got = ps_ok(f"$rc = Invoke-Batch -Python '{py}' -ArgList @('{slow}', '{want}'); "
                        "if ($null -eq $rc) { 'NULL' } else { [string]$rc }")
            assert got == str(want), (want, got)
    print("Invoke-Batch: a real child's exit code (0 and 3) comes back, never $null, after a 1.5 s run")


def check_invoke_batch_ast() -> None:
    """The launch is parsed (AST), not grepped: -PassThru, then Handle, WaitForExit, ExitCode in order."""
    code = (
        "$e=$null; $t=$null; "
        "$ast=[System.Management.Automation.Language.Parser]::ParseFile('%s',[ref]$t,[ref]$e); "
        "$fn=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Invoke-Batch'},$true); "
        "$cmd=$fn.Find({param($n) $n -is [System.Management.Automation.Language.CommandAst] -and $n.GetCommandName() -eq 'Start-Process'},$true); "
        "$pt=@($cmd.CommandElements | Where-Object {$_ -is [System.Management.Automation.Language.CommandParameterAst]} | ForEach-Object {$_.ParameterName}) -join ','; "
        "$members=@($fn.FindAll({param($n) $n -is [System.Management.Automation.Language.MemberExpressionAst]},$true) | ForEach-Object {$_.Member.Value}) -join ','; "
        "\"errors=$($e.Count) params=$pt members=$members\"" % SCRIPT
    )
    env = dict(os.environ); env.setdefault("DOTNET_SYSTEM_GLOBALIZATION_INVARIANT", "1")
    r = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-Command", code], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, (r.stdout, r.stderr)
    line = r.stdout.strip()
    assert line.startswith("errors=0 "), line
    params = line.split("params=")[1].split(" members=")[0].split(",")
    members = line.split("members=")[1].split(",")
    assert "PassThru" in params, line
    order = [members.index(m) for m in ("Handle", "WaitForExit", "ExitCode")]
    assert order == sorted(order), line
    print("Invoke-Batch AST: parses clean; Start-Process -PassThru; Handle -> WaitForExit -> ExitCode in order")


def check_outcome_logic() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        write_session(out, "A", clips=4, results=4)
        write_session(out, "B", clips=4, results=2, stop="watchdog")
        write_session(out, "C", clips=4, results=1)
        write_session(out, "D", clips=4, results=4)

        def outcome(sid: str, rc: str) -> dict:
            txt = ps_ok(f"Get-SessionOutcome -OutRoot '{out}' -SessionId '{sid}' -ExitCode {rc} | ConvertTo-Json -Compress")
            return json.loads(txt)

        o = outcome("A", "$null")
        assert o["Halt"] is False and o["Kind"] == "unknown", o  # null rc + complete session: not a halt
        o = outcome("B", "$null")
        assert o["Halt"] is True, o  # null rc + session-stop line
        o = outcome("C", "$null")
        assert o["Halt"] is True, o  # null rc + incomplete session
        o = outcome("A", "0")
        assert o["Halt"] is False and o["Kind"] == "ok", o
        o = outcome("A", "2")
        assert o["Halt"] is True and o["Kind"] == "halt", o
        # Item 3: a launch that died before writing session.json must not borrow the previous session.
        o = outcome("NEWER", "$null")
        assert o["Halt"] is True, ("null rc with no session for this launch is a halt, not 'completed'", o)
        o = outcome("NEWER", "0")
        assert o["Halt"] is True, ("exit 0 but no session.json for this launch", o)
        # A session.json whose session_id is another launch's is not this launch's.
        write_session(out, "X", clips=4, results=4, session_id="OTHER")
        o = outcome("X", "$null")
        assert o["Halt"] is True, o
    print("Get-SessionOutcome: null rc is verified from THIS launch's session.json; no session for it halts")


STUB = r"""
import json, sys
from pathlib import Path
a = sys.argv[1:]
out = Path(a[a.index("--out") + 1]); sid = a[a.index("--session-id") + 1]
modes = (out / "mode.txt").read_text().split()
mode = modes[0]
(out / "mode.txt").write_text(" ".join(modes[1:] or modes[-1:]))
d = out / f"clips_{sid}"; d.mkdir(parents=True)
if mode == "die":
    sys.exit(1)
(d / "session.json").write_text(json.dumps({"clips": 2, "results": [{}, {}], "session_id": sid}))
if mode == "ok_then_stop":
    (out / "STOP").write_text("x")
sys.exit(0)
"""


def run_loop(mode: str, *, halt_limit: int = 3, timeout: float = 180.0) -> tuple[subprocess.CompletedProcess, Path]:
    td = Path(tempfile.mkdtemp())
    out = td / "out"
    out.mkdir()
    (out / "mode.txt").write_text(mode)
    stub = td / "stub_collect.py"
    stub.write_text(STUB, encoding="utf-8")
    env = dict(os.environ)
    env.setdefault("DOTNET_SYSTEM_GLOBALIZATION_INVARIANT", "1")
    r = subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-File", str(SCRIPT), "-Out", str(out), "-Python", sys.executable,
         "-Collect", str(stub), "-Clips", "2", "-HaltLimit", str(halt_limit), "-LongClips", "1"],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    return r, td


def check_whole_loop() -> None:
    r, td = run_loop("die")
    try:
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        assert "3 consecutive halted batches" in (r.stdout + r.stderr), (r.stdout, r.stderr)
    finally:
        shutil.rmtree(td, ignore_errors=True)
    r, td = run_loop("ok_then_stop")
    try:
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert "ok" in r.stdout and "STOP file present" in r.stdout, r.stdout
        assert "halt 1" not in r.stdout
    finally:
        shutil.rmtree(td, ignore_errors=True)
    # Halts are consecutive: a good batch resets the count (die die ok die die ok+stop never reaches 3).
    r, td = run_loop("die die ok die die ok_then_stop")
    try:
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert r.stdout.count("halt 2 / 3") == 2 and "halt 3 / 3" not in r.stdout, r.stdout
    finally:
        shutil.rmtree(td, ignore_errors=True)
    # The limit is -HaltLimit (2 here), not a constant.
    r, td = run_loop("die", halt_limit=2)
    try:
        assert r.returncode == 2 and "2 consecutive halted batches" in r.stdout, (r.returncode, r.stdout)
        assert "halt 3" not in r.stdout
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("rec_loop whole loop: 3 failed batches -> exit 2; a good batch then STOP -> exit 0, no halt counted")


def main() -> None:
    if PWSH is None:
        from gvd_test_env import skip_check

        skip_check("rec_loop under pwsh", "pwsh not found (GVD_PWSH, PATH, /workspace/tools/pwsh/pwsh)")
        print("test_rec_loop: SKIPPED (no pwsh)")
        return
    check_arg_list()
    check_invoke_batch_ast()
    check_invoke_batch_exit_codes()
    check_outcome_logic()
    check_whole_loop()
    print("test_rec_loop: OK")


if __name__ == "__main__":
    main()
