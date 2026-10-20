#!/usr/bin/env python3
"""The abort's ai.setAggression(0.2) is put back to the default 0.3 when the 3 s abort ends.

0.39.4.0 ai.lua: ``setAggression`` sets ``M.extAggression`` (275-277) and
``setMode`` -> resetMapAndRoute -> resetAggression (3787, 290-292) resets to
``M.extAggression`` itself, so the cruise ``setMode("span")`` after an abort
kept 0.2. The collector never sets aggression at a segment start, so the
segment-start value is ai.lua's default ``M.extAggression = 0.3`` (ai.lua 12).

* ``excursion_commands``: abort sends 0.2, cruise sends 0.3, lines none;
* the real collect loop (test_collect_spawn_fix harness): an overspeed abort
  sends 0.2, and the cruise commands sent when the abort ends ABORT_S later
  send 0.3; ``collect.json`` records ``ai_aggression``;
* ``scripts/abort_aggression_tag.py``: lists the segments that ran on after an
  abort without the reset, writes a NEW create-only sidecar (refuses a
  rerun), leaves ``state.jsonl`` untouched, ``--dry-run`` writes nothing, and a
  recording from a collector with the reset gets nothing.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect import excursion as exc  # noqa: E402
from python.collect.excursion import ABORT_AGGRESSION, ABORT_S, AI_DEFAULT_AGGRESSION, excursion_commands  # noqa: E402
from scripts import abort_aggression_tag as tagger  # noqa: E402


def _aggr(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.startswith("ai.setAggression(")]


def check_commands() -> None:
    assert (ABORT_AGGRESSION, AI_DEFAULT_AGGRESSION, ABORT_S) == (0.2, 0.3, 3.0)
    assert _aggr(excursion_commands(12.0, "abort", 1)) == ["ai.setAggression(0.2)"]
    cruise = excursion_commands(12.0, "cruise", 1)
    assert _aggr(cruise) == ["ai.setAggression(0.3)"], cruise
    # After setMode: setMode's resetAggression would only re-apply the 0.2.
    assert cruise.index("ai.setAggression(0.3)") > cruise.index('ai.setMode("span")')
    assert _aggr(excursion_commands(12.0, "cruise", 1, traffic=True)) == ["ai.setAggression(0.3)"]
    assert _aggr(excursion_commands(12.0, "weave", 1, path=[(0.0, 0.0), (0.0, 4.0), (0.0, 8.0)])) == []
    print("commands: abort 0.2, cruise back to 0.3 after setMode, lines none")


def check_collector_resets_after_abort() -> None:
    import scripts.collect_scene_lanes as collect
    import scripts.test_collect_spawn_fix as harness

    sent: list[tuple[int, str]] = []
    base = harness._CrashVehicle

    class Vehicle(base):  # type: ignore[misc, valid-type]
        def queue_lua_command(self, command: str, *args: object):  # noqa: ANN201
            sent.append((int(self.polls), str(command)))
            return super().queue_lua_command(command, *args)

    previous = collect.painted_for
    harness._CrashVehicle = Vehicle  # type: ignore[misc]
    collect.painted_for = lambda level: harness._Paint(lambda x, y: True)  # type: ignore[assignment]
    try:
        # Same run as test_collect_speed_plan: planned at 10 m/s, the car speeds to 12 -> overspeed abort.
        code, text, out = harness._run_collector(
            {"speed_at": lambda n: 10.0 if n <= 215 else 12.0, "stop_at": 300, "x": 1.75},
            ["--speed-min", "8", "--speed-max", "17"],
        )
    finally:
        harness._CrashVehicle = base  # type: ignore[misc]
        collect.painted_for = previous  # type: ignore[assignment]
    assert code == 0, text[-2000:]
    assert "abort overspeed" in text, text[-2000:]
    aggr = [(p, c) for p, c in sent if c.startswith("ai.setAggression(")]
    i_abort = next(k for k, (_p, c) in enumerate(aggr) if c == "ai.setAggression(0.2)")
    after = aggr[i_abort + 1:]
    assert after and after[0][1] == "ai.setAggression(0.3)", aggr
    ticks = after[0][0] - aggr[i_abort][0]
    # The reset goes out with the cruise commands when the 3 s abort ends (0.1 s ticks).
    assert abs(ticks - round(ABORT_S / 0.1)) <= 1, (ticks, aggr)
    # Never left at 0.2: the last aggression sent is the default.
    assert aggr[-1][1] == "ai.setAggression(0.3)", aggr
    rows = harness._rows(out)
    phases = [(r.get("excursion") or {}).get("phase") for r in rows]
    k = phases.index("abort")
    n_abort = phases[k:].index("cruise")
    assert abs(n_abort - round(ABORT_S / 0.1)) <= 1, n_abort
    status = json.loads((out / "collect.json").read_text(encoding="utf-8"))
    assert status["ai_aggression"] == {"abort": 0.2, "reset_to": 0.3, "reset_on": "cruise"}, status.get("ai_aggression")
    # A recording from this collector needs no sidecar.
    assert tagger.judge(out)["verdict"] == "reset_in_collector"
    print(f"collector: abort sent 0.2, the abort end {ticks} ticks later sent 0.3")


def _rec(rows: list[dict], collect: dict | None = None) -> Path:
    rec = Path(tempfile.mkdtemp()) / "rec"
    rec.mkdir()
    (rec / "state.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (rec / "collect.json").write_text(json.dumps(collect or {"code_sha": "c1a0a13"}), encoding="utf-8")
    return rec


def _row(i: int, seg: str, phase: str) -> dict:
    return {"i": i, "t": 100.0 + 0.1 * i, "segment": seg, "excursion": {"phase": phase, "reason": ""}}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _main(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = tagger.main(argv)
    return code, buf.getvalue()


def check_tagger() -> None:
    phases_a = ["cruise", "cruise", "weave", "abort", "abort", "cruise", "weave", "abort", "cruise"]
    phases_b = ["cruise", "weave", "cruise"]  # no abort: not affected
    phases_c = ["cruise", "abort", "abort"]  # ends in the abort: nothing ran after it
    rows, i = [], 0
    for seg, phases in (("west_coast_usa#0", phases_a), ("italy#1", phases_b), ("utah#2", phases_c)):
        for ph in phases:
            rows.append(_row(i, seg, ph))
            i += 1
    rec = _rec(rows)
    before = _sha(rec / "state.jsonl")
    res = tagger.judge(rec)
    assert res["verdict"] == "aggression_not_reset", res
    segs = [s["segment"] for s in res["affected_segments"]]
    assert segs == ["west_coast_usa#0"], res["affected_segments"]
    seg = res["affected_segments"][0]
    assert seg["aborts"] == 2 and seg["first_abort_end_i"] == 5 and seg["rows_after_abort"] == 3, seg
    assert [m["i"] for m in res["marks"]] == [5, 6, 8], res["marks"]  # abort rows (7) are 0.2 by design
    assert all(m["ai_aggression_est"] == 0.2 and m["reason"] == tagger.REASON for m in res["marks"])
    assert abs(res["marks"][1]["since_abort_end_s"] - 0.1) < 1e-9
    # --dry-run lists the segments and writes nothing.
    code, text = _main([str(rec), "--dry-run"])
    assert code == 0 and "dry run" in text and "segment west_coast_usa#0: aborts=2" in text, text
    assert not (rec / tagger.SIDECAR).exists() and not (rec / tagger.META).exists()
    # A real run writes the NEW sidecar; state.jsonl is untouched; a rerun is refused.
    code, text = _main([str(rec)])
    assert code == 0 and (rec / tagger.SIDECAR).is_file() and (rec / tagger.META).is_file(), text
    marks = [json.loads(ln) for ln in (rec / tagger.SIDECAR).read_text(encoding="utf-8").splitlines()]
    assert [m["i"] for m in marks] == [5, 6, 8]
    meta = json.loads((rec / tagger.META).read_text(encoding="utf-8"))
    assert meta["marked_rows"] == 3 and meta["verdict"] == "aggression_not_reset" and meta["code_sha"] == "c1a0a13"
    assert _sha(rec / "state.jsonl") == before
    side_before = _sha(rec / tagger.SIDECAR)
    code, _text = _main([str(rec)])
    assert code == 3 and _sha(rec / tagger.SIDECAR) == side_before
    # --out-dir writes there, not beside the recording.
    other = Path(tempfile.mkdtemp())
    rec2 = _rec(rows)
    assert _main([str(rec2), "--out-dir", str(other)])[0] == 0
    assert (other / tagger.SIDECAR).is_file() and not (rec2 / tagger.SIDECAR).exists()
    # Nothing to mark: no aborts; an abort with nothing after it; a collector with the reset.
    for rows_x, collect, verdict in (
        ([_row(0, "a#0", "cruise"), _row(1, "a#0", "weave")], None, "no_aborts"),
        ([_row(0, "a#0", "cruise"), _row(1, "a#0", "abort")], None, "no_rows_after_abort"),
        ([_row(k, "a#0", p) for k, p in enumerate(phases_a)], {"ai_aggression": {"reset_to": 0.3}}, "reset_in_collector"),
    ):
        rx = _rec(rows_x, collect)
        assert tagger.judge(rx)["verdict"] == verdict
        code, text = _main([str(rx)])
        assert code == 0 and "nothing to mark" in text and not (rx / tagger.SIDECAR).exists(), (verdict, text)
    print("tagger: lists the segment that ran on after an abort, create-only sidecar, dry run writes nothing")


def main() -> None:
    check_commands()
    check_collector_resets_after_abort()
    check_tagger()
    print("abort aggression reset ok")


if __name__ == "__main__":
    main()
