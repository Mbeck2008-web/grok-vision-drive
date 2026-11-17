#!/usr/bin/env python3
"""Offline stub tests for scripts/traffic_stress.py.

Pure Python: no BeamNG, no nvidia-smi, no live attach.

The planner steps a count list, records Hz / drops / tick percentiles /
peak VRAM / RSS, writes a JSON and Markdown table, and stops escalating
when Hz < 9, drops > 1%, VRAM headroom < 1 GB, a VRAM reading is missing,
or supervisor RSS is over 12 GB. Rows use the placed count.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.traffic_stress import (  # noqa: E402
    DEFAULT_COUNTS,
    STOP_DROP_MAX,
    STOP_HZ_MIN,
    STOP_RSS_MAX_MB,
    STOP_VRAM_HEADROOM_GB,
    check_stress_out,
    live_collect_cmd,
    live_measure_factory,
    main as stress_main,
    percentile,
    run_plan,
    should_stop,
    write_report,
)


def check_stop_rules() -> None:
    assert should_stop(hz=12.0, drop_frac=0.0, vram_used_gb=6.0, vram_total_gb=11.0) is None
    assert should_stop(hz=8.9, drop_frac=0.0, vram_used_gb=6.0, vram_total_gb=11.0) == "hz"
    assert should_stop(hz=10.0, drop_frac=0.011, vram_used_gb=6.0, vram_total_gb=11.0) == "drops"
    assert should_stop(hz=10.0, drop_frac=0.01, vram_used_gb=6.0, vram_total_gb=11.0) is None
    assert should_stop(hz=10.0, drop_frac=0.0, vram_used_gb=10.1, vram_total_gb=11.0) == "vram"
    assert should_stop(hz=10.0, drop_frac=0.0, vram_used_gb=10.0, vram_total_gb=11.0) is None
    assert should_stop(hz=10.0, drop_frac=0.0, vram_used_gb=None, vram_total_gb=11.0, vram_missing=True) == "vram"
    assert should_stop(hz=10.0, drop_frac=0.0, vram_used_gb=None, vram_total_gb=None) == "vram"
    assert should_stop(
        hz=10.0, drop_frac=0.0, vram_used_gb=6.0, vram_total_gb=11.0, supervisor_rss_mb=12.0 * 1024.0 + 1
    ) == "rss"
    assert should_stop(
        hz=10.0, drop_frac=0.0, vram_used_gb=6.0, vram_total_gb=11.0, supervisor_rss_mb=12.0 * 1024.0
    ) is None
    assert STOP_HZ_MIN == 9.0 and STOP_DROP_MAX == 0.01 and STOP_VRAM_HEADROOM_GB == 1.0
    assert STOP_RSS_MAX_MB == 12.0 * 1024.0
    print("stop rules: hz < 9, drops > 1%, headroom < 1 GB, missing VRAM, RSS > 12 GB")


def check_percentiles() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    assert abs(percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95) - 4.8) < 1e-9
    assert percentile([2.0], 95) == 2.0
    assert percentile([], 50) is None
    print("percentiles: p50/p95 on a short series")


def check_stub_plan_stops_and_writes(tmp: Path) -> None:
    # 0 and 4 stay healthy; 8 drops under 9 Hz and must stop before 12+
    fixtures = {
        0: dict(hz=12.0, drop_frac=0.0, tick_s=(0.08, 0.10), vram_used_gb=6.2, vram_total_gb=11.0,
                beamng_rss_mb=3500.0, supervisor_rss_mb=800.0),
        4: dict(hz=11.2, drop_frac=0.002, tick_s=(0.09, 0.12), vram_used_gb=6.8, vram_total_gb=11.0,
                beamng_rss_mb=3600.0, supervisor_rss_mb=810.0),
        8: dict(hz=8.5, drop_frac=0.0, tick_s=(0.13, 0.20), vram_used_gb=7.1, vram_total_gb=11.0,
                beamng_rss_mb=3700.0, supervisor_rss_mb=820.0),
        12: dict(hz=8.0, drop_frac=0.05, tick_s=(0.2, 0.4), vram_used_gb=8.0, vram_total_gb=11.0,
                 beamng_rss_mb=4000.0, supervisor_rss_mb=900.0),
    }
    seen: list[int] = []

    def measure(count: int) -> dict:
        seen.append(count)
        return dict(fixtures[count])

    rows = run_plan(DEFAULT_COUNTS, measure, hold_s=60.0)
    assert seen == [0, 4, 8], seen
    assert DEFAULT_COUNTS == (0, 4, 8, 12, 16, 24, 32)
    assert [row["traffic_count"] for row in rows] == [0, 4, 8]
    assert rows[-1]["stop"] == "hz"
    assert all(row.get("stop") is None for row in rows[:-1])
    assert rows[0]["hz"] == 12.0 and rows[0]["tick_p50_s"] == 0.08 and rows[0]["tick_p95_s"] == 0.10
    out_json = tmp / "traffic_stress.json"
    out_md = tmp / "traffic_stress.md"
    write_report(rows, out_json, out_md)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["rows"][2]["stop"] == "hz"
    assert payload["stopped_at"] == 8
    assert payload["max_safe_count"] == 4
    text = out_md.read_text(encoding="utf-8")
    assert "traffic_count" in text and "8.5" in text and "hz" in text
    print("stub plan: stopped at 8 cars (hz); max_safe_count=4; JSON/MD written")


def check_max_safe_when_all_pass_or_first_fails() -> None:
    def _sample(hz: float) -> dict:
        return dict(
            hz=hz,
            drop_frac=0.0,
            tick_s=(0.08, 0.10),
            vram_used_gb=6.0,
            vram_total_gb=11.0,
            beamng_rss_mb=3500.0,
            supervisor_rss_mb=800.0,
        )

    all_ok = run_plan((0, 4, 8), lambda n: _sample(12.0), hold_s=60.0)
    assert all(row.get("stop") is None for row in all_ok)
    # reuse write_report via a temp-less check of the same rule the JSON uses
    healthy = [int(row["traffic_count"]) for row in all_ok if not row.get("stop")]
    assert healthy[-1] == 8

    first_fail = run_plan((0, 4), lambda n: _sample(8.0), hold_s=60.0)
    assert [row["traffic_count"] for row in first_fail] == [0]
    assert first_fail[0]["stop"] == "hz"
    assert [int(row["traffic_count"]) for row in first_fail if not row.get("stop")] == []
    print("max_safe: all-pass last count; first-step fail has no safe count")


def check_stub_cli_prints_max_safe(tmp: Path) -> None:
    import io
    import json
    from contextlib import redirect_stdout

    fixtures = {
        "0": dict(hz=12.0, drop_frac=0.0, tick_s=(0.08, 0.10), vram_used_gb=6.0, vram_total_gb=11.0),
        "4": dict(hz=11.0, drop_frac=0.0, tick_s=(0.09, 0.11), vram_used_gb=6.2, vram_total_gb=11.0),
        "8": dict(hz=8.5, drop_frac=0.0, tick_s=(0.13, 0.20), vram_used_gb=6.5, vram_total_gb=11.0),
    }
    tmp.mkdir(parents=True, exist_ok=True)
    metrics = tmp / "fixtures.json"
    metrics.write_text(json.dumps(fixtures), encoding="utf-8")
    out = tmp / "out"
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = stress_main(
            ["--stub", "--stub-metrics", str(metrics), "--counts", "0,4,8", "--out", str(out)]
        )
    assert code == 0
    text = buf.getvalue()
    payload = json.loads((out / "traffic_stress.json").read_text(encoding="utf-8"))
    assert payload["max_safe_count"] == 4 and payload["stopped_at"] == 8
    printed = json.loads([ln for ln in text.splitlines() if ln.startswith("{")][0])
    assert printed == {"max_safe_count": 4, "stopped_at": 8, "stop": "hz"}
    print("stub CLI: printed max_safe_count JSON")


def check_does_not_kill_existing_beamng() -> None:
    import inspect

    import scripts.traffic_stress as ts

    src = inspect.getsource(ts)
    lowered = src.lower()
    for banned in ("taskkill", "pkill", "kill -", "sigkill", "sigterm", "beamngpy.close", "bng.close", "process.terminate"):
        assert banned not in lowered and banned not in src, banned
    # Live mode only attaches by running the collector, which itself does not launch or close BeamNG.
    factory_src = inspect.getsource(live_measure_factory)
    assert "collect_scene_lanes.py" in factory_src
    assert "will not launch or close" in src or "does not launch or close" in src
    collect_head = (ROOT / "scripts" / "collect_scene_lanes.py").read_text(encoding="utf-8")
    assert "does not launch the game, and does not call BeamNGpy.close()" in collect_head
    print("no-kill: stress script never launches or closes BeamNG")


def check_max_safe_uses_placed_count() -> None:
    def measure(count: int) -> dict:
        # Requested 8 only places 4; 12 is unhealthy. max_safe must be 4, not 8.
        if count <= 8:
            return dict(
                hz=12.0,
                drop_frac=0.0,
                tick_s=(0.08, 0.10),
                vram_used_gb=6.0,
                vram_total_gb=11.0,
                supervisor_rss_mb=800.0,
                placed_count=4 if count == 8 else count,
            )
        return dict(
            hz=8.0,
            drop_frac=0.0,
            tick_s=(0.2, 0.4),
            vram_used_gb=8.0,
            vram_total_gb=11.0,
            supervisor_rss_mb=800.0,
            placed_count=count,
        )

    rows = run_plan((0, 4, 8, 12), measure, hold_s=60.0)
    assert [row["traffic_requested"] for row in rows] == [0, 4, 8, 12]
    assert [row["traffic_count"] for row in rows] == [0, 4, 4, 12]
    assert rows[-1]["stop"] == "hz"
    import tempfile

    with tempfile.TemporaryDirectory(prefix="gvd-stress-placed-") as raw:
        out = Path(raw)
        write_report(rows, out / "traffic_stress.json", out / "traffic_stress.md")
        payload = json.loads((out / "traffic_stress.json").read_text(encoding="utf-8"))
        assert payload["max_safe_count"] == 4
        assert payload["stopped_at"] == 12
    print("placed count: max_safe_count uses placed 4, not requested 8")


def check_live_child_passes_requested_count() -> None:
    cmd = live_collect_cmd(12, maps="west_coast_usa", minutes=1.0, step_dir=Path("/tmp/n12"))
    assert "--traffic" in cmd
    assert cmd[cmd.index("--traffic") + 1] == "12"
    src = __import__("inspect").getsource(live_collect_cmd)
    assert "str(int(count))" in src
    print("live child: --traffic is the requested step count")


def check_out_and_live_flags(tmp: Path) -> None:
    import io
    from contextlib import redirect_stderr, redirect_stdout

    def _run(argv):
        err, out = io.StringIO(), io.StringIO()
        try:
            with redirect_stderr(err), redirect_stdout(out):
                code = stress_main(argv)
        except SystemExit as exc:
            return exc.code, err.getvalue() + out.getvalue()
        return code, err.getvalue() + out.getvalue()

    code, text = _run(["--stub", "--counts", "0"])
    assert code == 2, (code, text)
    assert "--out" in text
    code, text = _run(["--out", str(tmp / "x")])
    assert code == 2, (code, text)
    assert "--live" in text or "--stub" in text
    try:
        check_stress_out(Path.cwd())
    except ValueError as exc:
        assert "current directory" in str(exc)
    else:
        raise AssertionError("cwd --out must be refused")
    try:
        check_stress_out(Path("C:/GVD_data/traffic_stress"))
    except ValueError as exc:
        assert "C:" in str(exc) or "refusing" in str(exc)
    else:
        raise AssertionError("C: --out must be refused")
    ok = check_stress_out(tmp / "stress")
    assert ok.name == "stress"
    print("out/live flags: --out required, cwd and C: refused, --live required for live")


def check_leaves_cars_documented() -> None:
    import inspect
    import scripts.traffic_stress as ts

    src = inspect.getsource(ts)
    assert "left in the running session" in src or "are left" in src.lower()
    assert "does not despawn" in src or "not despawned" in src
    print("cleanup: leftover stress-run cars are documented")


def main() -> None:
    check_stop_rules()
    check_percentiles()
    check_max_safe_when_all_pass_or_first_fails()
    check_max_safe_uses_placed_count()
    check_does_not_kill_existing_beamng()
    check_live_child_passes_requested_count()
    check_leaves_cars_documented()
    import tempfile

    with tempfile.TemporaryDirectory(prefix="gvd-traffic-stress-") as raw:
        root = Path(raw)
        check_stub_plan_stops_and_writes(root / "plan")
        check_stub_cli_prints_max_safe(root / "cli")
        check_out_and_live_flags(root / "flags")
    print("traffic stress ok")


if __name__ == "__main__":
    main()
