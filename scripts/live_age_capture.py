#!/usr/bin/env python3
"""Read-only: grade a live cam-age capture (hold ON) against a cam-age-model-OFF baseline.

Two ``run_vision`` logs of ~60 s each, back to back from the same spawn, with
engage hard-disabled (``--no-engage``):

* ``--baseline-log``: ``--cam-age-baseline`` (cam-age model off: the hold is
  off; with the default main-only scene model that is the plain supervisor);
* ``--log``: the untrained capture model (``scripts/export_capture_age_model.py``)
  with the hold ON (all 8 cameras streamed, side cameras held through the
  recorder's CameraSchedule).

Every tick prints one ``[GVD] cam_age`` line; the start prints
``[GVD] cam-age model`` and ``[GVD] cam-age providers`` lines. Bars, each
PASS / FAIL:

* model: ON ran a gvd_cam_age=1, 8-camera, capture_untrained=1 model with the
  hold on; the baseline ran with the hold off; both with engage hard-disabled.
* engage_off: every tick in both logs has engaged=0, engage_disabled=1.
* mode: ON ticks are hold=1 baseline=0; baseline ticks hold=0 baseline=1.
* duration: each log spans >= 55 s of ticks.
* compute: on every tick of both logs the detector, the planner (lanes,
  CIPV, AEB) and the scene net ran (``detect_ms``, ``plan_ms``, ``scene_ms``
  present and > 0, ``aeb`` present), and the detector is yolov8n. Engage
  gates actuation only.
* ages: per side camera, live median within 1 tick (1/15 s) and p95 within
  2 ticks of the training ages on the recording the model would train on.
  Black sectors count at the cap, 1.533 s (``cam_hold.BLACK_STAT_AGE_S``), on
  BOTH sides (live ``-1`` ages; training ``train_age_capped_s``): a black
  tick is never dropped before the median / p95.
* black_share: per camera (all 8), the ON log's black share (age -1) within
  ``--black-share-tol`` percentage points of the training black share
  (``black_pct.train`` of the train JSON, or ``--train-black-pct``). The
  tolerance has no default: without it the bar is UNSET and the grade is not
  a PASS.
* loop_dt: ON ``loop_dt_ms`` p50 <= ``--dt-p50`` and p95 <= ``--dt-p95``
  (the training row-gap bars; pre-registered 162 / 215 ms).
* scene_ms: ON ``scene_ms`` (onnxruntime session.run) p95 <= 185 ms.
* would_skip: 0 ON ticks where actuate would skip vehicle.control because a
  camera GE call holds the socket (counted with engage off).
* stale_ego: 0 ticks past one consecutive stale (coalesced) poll_vehicle while
  engaged; the stale count while disengaged is reported.
* main_flash / main_luma_jump: the ON rate of flashed main frames and of
  tick-to-tick mean-luminance jumps (> ``--luma-jump``) is not above the
  baseline rate. Any rise fails.

Training ages come from ``--train-json`` (``cam_hold_report.py --json``) or
from ``--strips`` plus the report's selection arguments, read-only.

Reported, not graded: per-camera read status counts (ok / none / zero /
flash_filtered / reattach / busy / timeout / error / skip) and read ms,
``main_age_ms`` (wall time after the pass's last read minus main's stamp),
``grab_ms`` and the read-log cost ``read_log_ms``.

    py -3 scripts/live_age_capture.py --log <ON_LOG> --baseline-log <BASELINE_LOG> ^
        --strips <RECORDING> --holdout-blocks 1 --mask-files <MASK_JSONL> ^
        --black-share-tol <PP>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from python.data.cam_hold import BLACK_STAT_AGE_S, CAM_AGE_ORDER, HOLD_TICK_S, NEVER_HELD, stat_age  # noqa: E402
from python.sensors.live_hold import parse_ages_text, parse_reads_text  # noqa: E402

MEDIAN_TOL_S = 1.0 * HOLD_TICK_S
P95_TOL_S = 2.0 * HOLD_TICK_S
# Pre-registered training row-gap bars for loop_dt (ms).
DT_P50_MAX_MS = 162.0
DT_P95_MAX_MS = 215.0
SCENE_P95_MAX_MS = 185.0
STALE_EGO_MAX_RUN = 1
MIN_SPAN_S = 55.0
LUMA_JUMP = 0.05  # mean main luminance (0..1) change between consecutive ticks counted as a jump
SIDE_CAMS = tuple(cam for cam in CAM_AGE_ORDER if cam not in NEVER_HELD)
TIMERS = (
    "compose_ms", "prep_ms", "scene_ms", "post_ms", "detect_ms", "plan_ms", "infer_ms",
    "grab_ms", "read_log_ms", "main_age_ms",
)
_LINE = re.compile(r"\[GVD\] cam_age (?P<body>.*)$")
_MODEL = re.compile(r"\[GVD\] cam-age model (?P<body>.*)$")
_PROVIDERS = re.compile(r"\[GVD\] cam-age providers (?P<body>.*)$")
_KV = re.compile(r"(\w+)=(\S+)")


def _num(text: str | None) -> float | None:
    if text is None or text == "na":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_log(lines: Iterable[str]) -> dict[str, Any]:
    ticks: list[dict[str, Any]] = []
    model: dict[str, str] | None = None
    providers: dict[str, str] = {}
    respawns = 0
    for line in lines:
        line = line.rstrip("\n")
        if "[GVD] cam-age: respawn" in line:
            respawns += 1
        got = _MODEL.search(line)
        if got is not None:
            if model is None:
                model = dict(_KV.findall(got.group("body")))
            continue
        got = _PROVIDERS.search(line)
        if got is not None:
            providers.update(dict(_KV.findall(got.group("body"))))
            continue
        got = _LINE.search(line)
        if got is None:
            continue
        kv = dict(_KV.findall(got.group("body")))
        if "tick_ms" not in kv or "ages" not in kv:
            continue
        tick = {
            "t": _num(kv.get("t")),
            "tick_ms": float(kv["tick_ms"]),
            "loop_dt_ms": _num(kv.get("loop_dt_ms")),
            "hold": kv.get("hold") == "1",
            "baseline": kv.get("baseline") == "1",
            "engaged": kv.get("engaged"),
            "engage_disabled": kv.get("engage_disabled"),
            "would_skip": kv.get("would_skip"),
            "stale_ego": kv.get("stale_ego") == "1",
            "main_luma": _num(kv.get("main_luma")),
            "main_flash": kv.get("main_flash"),
            "aeb": kv.get("aeb"),
            "ages": parse_ages_text(kv["ages"]),
            "reads": parse_reads_text(kv.get("reads", "")),
        }
        for key in TIMERS:
            tick[key] = _num(kv.get(key))
        ticks.append(tick)
    ages = {cam: [t["ages"][cam] for t in ticks if cam in t["ages"]] for cam in CAM_AGE_ORDER}
    return {"ticks": ticks, "ages": ages, "model": model, "providers": providers, "respawns": respawns}


def _stat(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), q))


def compare_ages(live: dict[str, Any], train_age_s: dict[str, Any]) -> dict[str, Any]:
    """Per side camera: live age median / p95 vs training, black counted at the cap on both sides.

    ``train_age_s``: the report's ``train_age_capped_s`` (black at 1.533 s).
    """
    out: dict[str, Any] = {}
    all_ok = True
    for cam in SIDE_CAMS:
        values = live["ages"].get(cam) or []
        held = [v for v in values if v >= 0.0]
        capped = [stat_age(v) for v in values]
        train = train_age_s.get(cam) or {}
        live_med, live_p95 = _stat(capped, 50), _stat(capped, 95)
        tr_med, tr_p95 = train.get("median"), train.get("p95")
        ok = (
            live_med is not None
            and live_p95 is not None
            and tr_med is not None
            and tr_p95 is not None
            and abs(live_med - float(tr_med)) <= MEDIAN_TOL_S + 1e-9
            and abs(live_p95 - float(tr_p95)) <= P95_TOL_S + 1e-9
        )
        all_ok = all_ok and ok
        out[cam] = {
            "live_n": len(values),
            "live_held_n": len(held),
            "live_black_pct": None if not values else round(100.0 * (len(values) - len(held)) / len(values), 2),
            "live_median": live_med,
            "live_p95": live_p95,
            "train_median": tr_med,
            "train_p95": tr_p95,
            "pass": bool(ok),
        }
    return {"pass": all_ok, "cams": out}


def black_shares(ticks: list[dict[str, Any]]) -> dict[str, float | None]:
    """Per camera (all 8): percent of ticks whose age is black (-1 or missing)."""
    out: dict[str, float | None] = {}
    for cam in CAM_AGE_ORDER:
        values = [t["ages"].get(cam) for t in ticks]
        out[cam] = None if not values else round(100.0 * sum(1 for v in values if v is None or v < 0.0) / len(values), 2)
    return out


def black_share_bar(
    live: dict[str, float | None], train_black: dict[str, Any], tol_pp: float | None
) -> dict[str, Any]:
    """|live - train| black share <= ``tol_pp`` per camera. ``tol_pp`` None: UNSET, never a PASS."""
    cams: dict[str, Any] = {}
    ok_all = True
    for cam in CAM_AGE_ORDER:
        lv, tv = live.get(cam), train_black.get(cam)
        diff = None if lv is None or tv is None else round(float(lv) - float(tv), 2)
        ok = tol_pp is not None and diff is not None and abs(diff) <= float(tol_pp) + 1e-9
        ok_all = ok_all and ok
        cams[cam] = {"live_pct": lv, "train_pct": tv, "diff_pp": diff, "pass": bool(ok)}
    unset = tol_pp is None
    detail = (
        "UNSET: --black-share-tol not given (no pre-registered tolerance); live vs train black % per camera reported"
        if unset
        else f"|live - train| black share <= {float(tol_pp):g} pp per camera"
    )
    return {"pass": bool(ok_all and not unset), "unset": unset, "detail": detail, "cams": cams}


def read_summary(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """Per camera: read status counts and read-ms p50 / p95 (cam-age telemetry ``reads=``)."""
    out: dict[str, Any] = {}
    for cam in CAM_AGE_ORDER:
        counts: dict[str, int] = {}
        ms = []
        for t in ticks:
            got = (t.get("reads") or {}).get(cam)
            if got is None:
                continue
            counts[got[0]] = counts.get(got[0], 0) + 1
            if got[1] is not None:
                ms.append(got[1])
        out[cam] = {"status": counts, "read_ms_p50": _stat(ms, 50), "read_ms_p95": _stat(ms, 95)}
    return out


def exposure(ticks: list[dict[str, Any]], luma_jump: float = LUMA_JUMP) -> dict[str, Any]:
    lumas = [t["main_luma"] for t in ticks]
    have = [t for t in ticks if t["main_luma"] is not None]
    flashes = sum(1 for t in have if t["main_flash"] == "1")
    pairs = [(a, b) for a, b in zip(lumas, lumas[1:]) if a is not None and b is not None]
    jumps = sum(1 for a, b in pairs if abs(b - a) > luma_jump)
    return {
        "frames": len(have),
        "flashes": flashes,
        "flash_rate": None if not have else flashes / len(have),
        "pairs": len(pairs),
        "jumps": jumps,
        "jump_rate": None if not pairs else jumps / len(pairs),
    }


def stale_runs(ticks: list[dict[str, Any]]) -> dict[str, int]:
    """Stale ticks while disengaged, and engaged ticks past ``STALE_EGO_MAX_RUN`` in a row."""
    run = 0
    engaged_over = 0
    disengaged = 0
    for t in ticks:
        run = run + 1 if t["stale_ego"] else 0
        if t["stale_ego"] and t["engaged"] != "1":
            disengaged += 1
        if t["engaged"] == "1" and run > STALE_EGO_MAX_RUN:
            engaged_over += 1
    return {"engaged_over": engaged_over, "disengaged": disengaged, "total": sum(1 for t in ticks if t["stale_ego"])}


def compute_gaps(ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ticks where the detector, planner (+ AEB) or scene net did not run.

    ``plan_ms``, ``scene_ms`` > 0 and ``aeb`` on every tick; ``detect_ms`` > 0 on
    every tick with a main frame (no frame: the detector returns at once) and
    present on the others.
    """
    out = []
    for t in ticks:
        ran = (t.get("plan_ms") or 0.0) > 0.0 and (t.get("scene_ms") or 0.0) > 0.0 and t.get("aeb") not in (None, "na")
        det = t.get("detect_ms")
        ran = ran and det is not None and (det > 0.0 or t["main_luma"] is None)
        if not ran:
            out.append(t)
    return out


def _span(ticks: list[dict[str, Any]]) -> float:
    ts = [t["t"] for t in ticks if t["t"] is not None]
    return 0.0 if len(ts) < 2 else float(max(ts) - min(ts))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def timer_summary(ticks: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    out = {}
    for key in TIMERS + ("loop_dt_ms", "tick_ms"):
        vals = [t[key] for t in ticks if t.get(key) is not None]
        out[key] = {"p50": _stat(vals, 50), "p95": _stat(vals, 95), "n": len(vals)}
    return out


def grade(
    on: dict[str, Any],
    base: dict[str, Any],
    train_age_s: dict[str, Any],
    *,
    train_black: dict[str, Any] | None = None,
    black_tol_pp: float | None = None,
    dt_p50_ms: float = DT_P50_MAX_MS,
    dt_p95_ms: float = DT_P95_MAX_MS,
    scene_p95_ms: float = SCENE_P95_MAX_MS,
    luma_jump: float = LUMA_JUMP,
    min_span_s: float = MIN_SPAN_S,
) -> dict[str, Any]:
    bars: dict[str, dict[str, Any]] = {}
    m_on, m_base = on.get("model") or {}, base.get("model") or {}
    bars["model"] = {
        "pass": (
            m_on.get("gvd_cam_age") == "1"
            and m_on.get("cams") == str(len(CAM_AGE_ORDER))
            and m_on.get("capture_untrained") == "1"
            and m_on.get("hold") == "on"
            and m_on.get("engage_disabled") == "1"
            and m_base.get("hold") == "off"
            and m_base.get("engage_disabled") == "1"
        ),
        "detail": f"on={m_on or None} baseline={m_base or None}",
    }
    eng_bad = {
        name: sum(1 for t in log["ticks"] if t["engaged"] != "0" or t["engage_disabled"] != "1")
        for name, log in (("on", on), ("baseline", base))
    }
    bars["engage_off"] = {"pass": not any(eng_bad.values()), "detail": f"ticks not engaged=0/engage_disabled=1: {eng_bad}"}
    mode_bad = {
        "on": sum(1 for t in on["ticks"] if not t["hold"] or t["baseline"]),
        "baseline": sum(1 for t in base["ticks"] if t["hold"] or not t["baseline"]),
    }
    bars["mode"] = {
        "pass": bool(on["ticks"]) and bool(base["ticks"]) and not any(mode_bad.values()),
        "detail": f"ticks {len(on['ticks'])} on / {len(base['ticks'])} baseline; wrong mode {mode_bad}",
    }
    spans = {"on": _span(on["ticks"]), "baseline": _span(base["ticks"])}
    bars["duration"] = {
        "pass": all(v >= min_span_s for v in spans.values()),
        "detail": f"span s on={spans['on']:.1f} baseline={spans['baseline']:.1f} (min {min_span_s:g})",
    }
    missing = {name: len(compute_gaps(log["ticks"])) for name, log in (("on", on), ("baseline", base))}
    no_main = {name: sum(1 for t in log["ticks"] if t["main_luma"] is None) for name, log in (("on", on), ("baseline", base))}
    detectors = {name: str((log.get("providers") or {}).get("detector", "")) for name, log in (("on", on), ("baseline", base))}
    yolo = all(d.startswith("yolov8n") for d in detectors.values())
    bars["compute"] = {
        "pass": bool(on["ticks"]) and bool(base["ticks"]) and not any(missing.values()) and yolo,
        "detail": f"ticks without detect/plan/scene timers or aeb: {missing}; ticks without a main frame {no_main}; "
        f"detector {detectors}",
    }
    ages = compare_ages(on, train_age_s)
    bars["ages"] = {
        "pass": ages["pass"],
        "detail": f"median within {MEDIAN_TOL_S * 1000:.0f} ms, p95 within {P95_TOL_S * 1000:.0f} ms; "
        f"black counted at {BLACK_STAT_AGE_S:.3f} s on both sides",
        "cams": ages["cams"],
    }
    bars["black_share"] = black_share_bar(black_shares(on["ticks"]), dict(train_black or {}), black_tol_pp)
    dts = [t["loop_dt_ms"] for t in on["ticks"] if t["loop_dt_ms"] is not None]
    dt50, dt95 = _stat(dts, 50), _stat(dts, 95)
    bars["loop_dt"] = {
        "pass": dt50 is not None and dt95 is not None and dt50 <= dt_p50_ms and dt95 <= dt_p95_ms,
        "detail": f"on p50 {_fmt(dt50)} ms (max {dt_p50_ms:g}), p95 {_fmt(dt95)} ms (max {dt_p95_ms:g})",
    }
    scene = [t["scene_ms"] for t in on["ticks"] if t["scene_ms"] is not None]
    sc95 = _stat(scene, 95)
    bars["scene_ms"] = {
        "pass": sc95 is not None and sc95 <= scene_p95_ms,
        "detail": f"on session.run p95 {_fmt(sc95)} ms (max {scene_p95_ms:g})",
    }
    skips_on = sum(1 for t in on["ticks"] if t["would_skip"] != "0")
    skips_base = sum(1 for t in base["ticks"] if t["would_skip"] != "0")
    bars["would_skip"] = {
        "pass": bool(on["ticks"]) and skips_on == 0,
        "detail": f"would-skip control ticks on={skips_on}/{len(on['ticks'])} baseline={skips_base}/{len(base['ticks'])}",
    }
    st_on, st_base = stale_runs(on["ticks"]), stale_runs(base["ticks"])
    bars["stale_ego"] = {
        "pass": st_on["engaged_over"] == 0 and st_base["engaged_over"] == 0,
        "detail": f"engaged past {STALE_EGO_MAX_RUN} stale tick: on={st_on['engaged_over']} baseline={st_base['engaged_over']}; "
        f"stale while disengaged: on={st_on['disengaged']}/{len(on['ticks'])} baseline={st_base['disengaged']}/{len(base['ticks'])}",
    }
    ex_on, ex_base = exposure(on["ticks"], luma_jump), exposure(base["ticks"], luma_jump)

    def not_above(key: str) -> bool:
        a, b = ex_on[key], ex_base[key]
        return a is not None and b is not None and a <= b + 1e-12

    bars["main_flash"] = {
        "pass": not_above("flash_rate"),
        "detail": f"flashed main frames on={ex_on['flashes']}/{ex_on['frames']} baseline={ex_base['flashes']}/{ex_base['frames']}",
    }
    bars["main_luma_jump"] = {
        "pass": not_above("jump_rate"),
        "detail": f"|d luma| > {luma_jump:g}: on={ex_on['jumps']}/{ex_on['pairs']} baseline={ex_base['jumps']}/{ex_base['pairs']}",
    }
    return {
        "bars": bars,
        "pass": all(bar["pass"] for bar in bars.values()),
        "respawns": {"on": on.get("respawns", 0), "baseline": base.get("respawns", 0)},
        "providers": {"on": on.get("providers") or {}, "baseline": base.get("providers") or {}},
        "timers": {"on": timer_summary(on["ticks"]), "baseline": timer_summary(base["ticks"])},
        "exposure": {"on": ex_on, "baseline": ex_base},
        "reads": {"on": read_summary(on["ticks"]), "baseline": read_summary(base["ticks"])},
        "unset": [name for name, bar in bars.items() if bar.get("unset")],
    }


def table(result: dict[str, Any]) -> str:
    def f(v: Any) -> str:
        return "n/a" if v is None else f"{float(v):.3f}"

    lines = []
    for name, bar in result["bars"].items():
        verdict = "UNSET" if bar.get("unset") else ("PASS" if bar["pass"] else "FAIL")
        lines.append(f"{verdict:<5} {name:<15} {bar['detail']}")
        if name == "black_share":
            lines.append(f"      {'cam':<9}{'live black %':>14}{'train black %':>15}{'diff pp':>10}")
            for cam, row in bar["cams"].items():
                lines.append(
                    f"      {cam:<9}{f(row['live_pct']):>14}{f(row['train_pct']):>15}{f(row['diff_pp']):>10}"
                    + ("" if bar.get("unset") else f"  {'PASS' if row['pass'] else 'FAIL'}")
                )
        if name == "ages":
            lines.append(f"      {'cam':<9}{'live med':>10}{'train med':>11}{'live p95':>10}{'train p95':>11}{'live black %':>14}"
                         "  (black at the cap)")
            for cam, row in bar["cams"].items():
                black = "n/a" if row["live_black_pct"] is None else f"{row['live_black_pct']:.2f}"
                lines.append(
                    f"      {cam:<9}{f(row['live_median']):>10}{f(row['train_median']):>11}{f(row['live_p95']):>10}"
                    f"{f(row['train_p95']):>11}{black:>14}  {'PASS' if row['pass'] else 'FAIL'}"
                )
    for name in ("on", "baseline"):
        tm = result["timers"][name]
        lines.append(
            f"timers {name}: " + " ".join(f"{k}={_fmt(v['p50'])}/{_fmt(v['p95'])}" for k, v in tm.items()) + " (p50/p95 ms)"
        )
    for name in ("on", "baseline"):
        parts = []
        for cam, row in result["reads"][name].items():
            if not row["status"]:
                continue
            st = "/".join(f"{k}:{v}" for k, v in sorted(row["status"].items()))
            parts.append(f"{cam}[{st} ms {_fmt(row['read_ms_p50'])}/{_fmt(row['read_ms_p95'])}]")
        lines.append(f"reads {name}: " + (" ".join(parts) if parts else "n/a"))
    lines.append(f"providers: {result['providers']}")
    lines.append(f"respawns / resets seen: {result['respawns']}")
    unset = result.get("unset") or []
    lines.append(f"OVERALL {'PASS' if result['pass'] else 'FAIL'}" + (f" (UNSET: {', '.join(unset)})" if unset else ""))
    return "\n".join(lines)


def parse_black_pct(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(text or "").split(","):
        if not part.strip():
            continue
        cam, _, raw = part.partition("=")
        if cam.strip() not in CAM_AGE_ORDER:
            raise SystemExit(f"--train-black-pct: unknown camera {cam.strip()!r}")
        out[cam.strip()] = float(raw)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live cam-age capture vs training ages (read-only)")
    parser.add_argument("--log", type=Path, required=True, help="Hold-ON run_vision log ('[GVD] cam_age' lines).")
    parser.add_argument("--baseline-log", type=Path, required=True, help="Cam-age-model-OFF (--cam-age-baseline) log.")
    parser.add_argument("--dt-p50", type=float, default=DT_P50_MAX_MS, help="loop_dt p50 bar, ms (training row gap).")
    parser.add_argument("--dt-p95", type=float, default=DT_P95_MAX_MS, help="loop_dt p95 bar, ms (training row gap).")
    parser.add_argument("--scene-p95", type=float, default=SCENE_P95_MAX_MS, help="scene_ms p95 bar, ms.")
    parser.add_argument("--luma-jump", type=float, default=LUMA_JUMP, help="Mean-luminance jump threshold (0..1).")
    parser.add_argument(
        "--black-share-tol",
        type=float,
        default=None,
        help="black_share bar: max |live - train| black share per camera, percentage points. No default: "
        "without it the bar is UNSET and the grade is not a PASS.",
    )
    parser.add_argument(
        "--train-black-pct",
        default="",
        help="Training black %% per camera, 'cam=pct,...' (e.g. from the rec5c report); overrides the train JSON's "
        "black_pct.train for the cameras it names.",
    )
    parser.add_argument("--train-json", type=Path, default=None, help="cam_hold_report.py --json output.")
    parser.add_argument("--strips", type=Path, default=None, help="Recording to compute the training ages from.")
    parser.add_argument("--cams", default="")
    parser.add_argument("--mask-files", default="")
    parser.add_argument("--mask-events", action="store_true")
    parser.add_argument("--holdout-blocks", type=int, default=None)
    parser.add_argument("--holdout-cells-file", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the table.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.train_json is None) == (args.strips is None):
        raise SystemExit("give exactly one of --train-json or --strips")
    if args.train_json is not None:
        train = json.loads(args.train_json.read_text(encoding="utf-8"))
    else:
        from cam_hold_report import report

        from python.train.cam_strip import parse_cams
        from python.train.train_scene import resolve_holdout_blocks

        train = report(
            args.strips,
            cams=parse_cams(args.cams),
            mask_files=args.mask_files,
            mask_events=bool(args.mask_events),
            holdout_blocks=resolve_holdout_blocks(args.holdout_blocks),
            holdout_cells_file=args.holdout_cells_file,
        )
    logs = {}
    for name, path in (("on", args.log), ("baseline", args.baseline_log)):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            logs[name] = parse_log(handle)
        if not logs[name]["ticks"]:
            raise SystemExit(f"{path}: no '[GVD] cam_age' lines")
    capped = train.get("train_age_capped_s")
    if not capped:
        raise SystemExit(
            "the train JSON has no train_age_capped_s (black at the cap): rerun scripts/cam_hold_report.py --json"
        )
    train_black = dict((train.get("black_pct") or {}).get("train") or {})
    train_black.update(parse_black_pct(args.train_black_pct))
    result = grade(
        logs["on"],
        logs["baseline"],
        capped,
        train_black=train_black,
        black_tol_pp=args.black_share_tol,
        dt_p50_ms=float(args.dt_p50),
        dt_p95_ms=float(args.dt_p95),
        scene_p95_ms=float(args.scene_p95),
        luma_jump=float(args.luma_jump),
    )
    print(json.dumps(result, indent=1) if args.json else table(result), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
