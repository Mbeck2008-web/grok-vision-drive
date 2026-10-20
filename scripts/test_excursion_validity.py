#!/usr/bin/env python3
"""``excursion_validity.py``: a create-only sidecar for setLine recordings; state.jsonl untouched."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import excursion_validity as ev  # noqa: E402


def _rec(rows: list[dict], code_sha: str = "1addc938c83929c9b0c16dfd36b1fa95fdb8bd31") -> Path:
    rec = Path(tempfile.mkdtemp()) / "rec"
    rec.mkdir()
    (rec / "state.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (rec / "collect.json").write_text(json.dumps({"code_sha": code_sha}), encoding="utf-8")
    return rec


def _row(i: int, phase: str, ack: bool = False, reason: str = "") -> dict:
    exc = {"phase": phase, "target_offset_m": 0.4 if phase != "cruise" else 0.0, "target_heading_rad": 0.02,
           "direction": 1, "reason": reason}
    if ack:
        exc["line_ack"] = {"ok": True, "mode": "manual", "wait_s": 0.05, "polls": 2}
    return {"i": i, "t": 100.0 + i * 0.1, "excursion": exc}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_legacy_marked() -> None:
    rows = [_row(0, "cruise"), _row(1, "weave"), _row(2, "weave"), _row(3, "lane_change"), _row(4, "abort", reason="lane_edge"), _row(5, "cruise")]
    rec = _rec(rows)
    before = _sha(rec / "state.jsonl")
    assert ev.main([str(rec)]) == 0
    assert _sha(rec / "state.jsonl") == before  # old tags are not rewritten
    marks = [json.loads(x) for x in (rec / ev.SIDECAR).read_text(encoding="utf-8").splitlines()]
    assert [m["i"] for m in marks] == [1, 2, 3]
    assert all(m["target_offset_m_valid"] is False and m["direction_valid"] is False for m in marks)
    meta = json.loads((rec / ev.META).read_text(encoding="utf-8"))
    assert meta["verdict"] == "set_line_raw_lua" and meta["code_sha"].startswith("1addc938") and meta["marked_rows"] == 3
    # Create-only: a second run refuses and leaves the first sidecar as it was.
    side_sha = _sha(rec / ev.SIDECAR)
    assert ev.main([str(rec)]) == 3
    assert _sha(rec / ev.SIDECAR) == side_sha
    print("setLine recording: 3 active rows marked invalid, state.jsonl unchanged, rerun refused")


def check_acked_and_dry_run() -> None:
    rec = _rec([_row(0, "cruise"), _row(1, "weave", ack=True), _row(2, "abort", reason="line_not_active")], code_sha="f" * 40)
    assert ev.main([str(rec)]) == 0
    assert not (rec / ev.SIDECAR).exists() and not (rec / ev.META).exists()
    assert ev.judge(rec)["verdict"] == "line_acked"
    legacy = _rec([_row(0, "weave")])
    assert ev.main([str(legacy), "--dry-run"]) == 0
    assert not (legacy / ev.SIDECAR).exists()
    none = _rec([_row(0, "cruise")])
    assert ev.judge(none)["verdict"] == "no_active_rows" and ev.main([str(none)]) == 0
    assert not (none / ev.SIDECAR).exists()
    print("acked recording, dry run and cruise-only recording: no sidecar")


def main() -> None:
    check_legacy_marked()
    check_acked_and_dry_run()
    print("excursion validity ok")


if __name__ == "__main__":
    main()
