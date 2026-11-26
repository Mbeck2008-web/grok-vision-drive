"""Edge-case clip mode: one session, many 30-60 s clips, each in its own new folder.

Each clip is one run of ``scripts/collect_scene_lanes.py`` (lossless raw PNG,
all eight cameras with training-only depth/sem, the Hirochi eSBR), driven
with arguments drawn from a seeded plan:

* one random edge case from ``EDGE_CASES`` and its recipe (weather, time of
  day, spawn kind, paint filter, traffic floor),
* a random companion-traffic count up to ``max_traffic`` on every map
  (Michael, 2026-11-24 15:14 CST: "Add traffic and random spawns on every
  map"),
* a map from the pool (edge cases that need a highway or ramp spawn stay
  on ``west_coast_usa`` when it is in the pool; other maps still get a
  seeded legal spawn and planned traffic),
* a random spawn: a per-clip ``--spawn-seed`` for the deck shuffle on
  every map. The collector loads that map's road dump, places a legal
  on-road spawn (correct side of the road) and companions, and stops
  with a visible error if the map has no dump or no legal pose.

After each clip, ``quality_check`` reads the recording back (never writes
into ``raw/``) and ``write_tags`` stores ``clip_tags.json`` beside it. A bad
clip is marked ``pretrain_only`` with its reasons; nothing is deleted. A
clip folder or session folder that already exists is never reused.

The recipe asks for the edge case; it cannot force a junction or a cut-in
to happen. ``edge_case_observed`` says whether the recording shows it
(True / False), or None when this offline check cannot see it (rain, fog,
junction). A False marks the clip ``pretrain_only``.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from python.collect.auto_mode import hour_to_beamng_tod
from python.collect.spawn_mix import HIGHWAY, SPEED, classify_roads, load_nav_roads, spawn_roads_path
from python.sensors.tech import TECH_GFX

EDGE_CASES: tuple[str, ...] = (
    "sharp_curve",
    "junction",
    "night",
    "rain",
    "fog",
    "faded_paint",
    "dirt_road",
    "merge",
    "cut_in",
    "heavy_traffic",
)
DECK_MAP = "west_coast_usa"
# Needs a highway or ramp spawn pool; those are not on every map.
DECK_ONLY = ("merge", "cut_in", "heavy_traffic")
# Maps with unpaved roads, from the map descriptions (UNVERIFIED per map: the
# clip's own quality check reads the semantic planes for dirt).
# spawn_mix only classifies city / highway / ramp — there is no unpaved spawn
# kind — so a dirt_road plan picks one of these maps but a legal pose can still
# be pavement (dirt share of the lower third 0.00, as on italy clip_002).
DIRT_MAPS: tuple[str, ...] = ("utah", "jungle_rock_island", "italy", "east_coast_usa", "small_island")
EGO_MODEL = "sbr"
# Requested renderer, recorded in the tags and manifest (--gfx, default Vulkan).
# The collector launches BeamNG.tech with -gfx TECH_GFX; an already-running
# BeamNG keeps the renderer it was started with.
GFX = TECH_GFX
MIN_CLIP_S = 30.0
MAX_CLIP_S = 60.0
# Long highway/arterial clips (opt-in via collect_clips --long-clips).
DEFAULT_LONG_CLIPS = 0
DEFAULT_LONG_CLIP_MIN_S = 180.0
DEFAULT_LONG_CLIP_MAX_S = 300.0
LONG_HIGHWAY_TAG = "long_highway"
HIGHWAY_SPEED_MIN, HIGHWAY_SPEED_MAX = SPEED[HIGHWAY]
SPAWN_KIND = {"merge": "ramp", "cut_in": "highway", "heavy_traffic": "highway", "sharp_curve": "city", "junction": "city"}
CUT_IN_AHEAD_M = 25.0
# Quality thresholds (quick check, not a training filter).
STOP_FILE_REASON = "stop_file"  # session.json stop_reason when the pause/stop file is present
MIN_MAIN_HZ = 5.0            # half the 10 Hz plan
MIN_MOVING_MPS = 2.0         # median ego speed
CLIP_CLASSES = ("train", "pretrain_only", "reject")
SHARP_CURVE_PER_M = 1.0 / 50.0   # |yaw_rate| / speed: a 50 m radius or tighter
SHARP_CURVE_MIN_MPS = 5.0
NIGHT_MAX_LUMA = 60.0        # mean main-camera luma on sampled frames
FADED_MAX_PAINT_SHARE = 0.7  # rows with any painted lane label
DIRT_MIN_SHARE = 0.2         # lower third of main_sem
DIRT_CLASSES = ("MUD", "SAND", "ROCK", "GRAVEL", "DIRT")
CUT_IN_FROM_M = 2.5          # lateral offset before
CUT_IN_TO_M = 1.0            # lateral offset after, ahead within CUT_IN_AHEAD_MAX_M
CUT_IN_AHEAD_MAX_M = 30.0
CUT_IN_WINDOW_S = 6.0        # the "out" sighting must be this recent when the car is in lane
CUT_IN_BEND_MAX_M = 0.75     # road bend at the car's range (k * ahead^2 / 2) above this: row not read
HEAVY_PLACED_SHARE = 0.75


@dataclass
class ClipPlan:
    index: int
    seed: int
    edge_case: str
    map: str
    traffic: int
    weather: str
    time_of_day_h: float
    duration_s: float
    spawn_seed: int
    spawn_kind: str | None
    min_paint_cov: float | None
    traffic_ahead_m: float | None
    ego_model: str = EGO_MODEL
    notes: list[str] = field(default_factory=list)
    long_highway: bool = False
    speed_min: float | None = None
    speed_max: float | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clip_rng(seed: int, index: int) -> random.Random:
    return random.Random(f"gvd-clip:{int(seed)}:{int(index)}")


def long_clip_indices(clips: int, long_clips: int, seed: int) -> tuple[int, ...]:
    """Spread ``long_clips`` indices through ``[0, clips)`` using ``seed``.

    Deterministic. One index is drawn in each equal-sized bucket so the
    long clips are mixed through the batch, not packed at the end.
    """
    clips = int(clips)
    n = int(long_clips)
    if n < 0:
        raise ValueError(f"long_clips must be 0 or more, not {n}")
    if n == 0 or clips <= 0:
        return ()
    if n > clips:
        raise ValueError(f"long_clips {n} exceeds clips {clips}")
    rng = random.Random(f"gvd-long-clips:{int(seed)}:{clips}:{n}")
    picked = []
    for i in range(n):
        lo = i * clips // n
        hi = (i + 1) * clips // n
        picked.append(rng.randrange(lo, hi))
    return tuple(picked)


@lru_cache(maxsize=32)
def map_has_highway_pool(map_name: str) -> bool:
    """True when the map's road dump has at least one highway-classified road.

    Uses ``spawn_roads_path`` (a live dump under roads_dir() wins, otherwise
    the bundled fixture). A missing or unreadable dump is not a highway map.
    """
    try:
        path = spawn_roads_path(str(map_name))
        if path is None or not path.is_file():
            return False
        pools = classify_roads(load_nav_roads(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return bool(pools.get(HIGHWAY))


def highway_maps(maps: Iterable[str]) -> tuple[str, ...]:
    """Session maps that have a highway spawn pool, in pool order."""
    return tuple(m for m in maps if str(m) and map_has_highway_pool(str(m)))


def apply_long_highway(
    plan: ClipPlan, *, min_s: float, max_s: float, maps: Iterable[str] | None = None
) -> ClipPlan:
    """Turn one planned clip into a long highway/arterial run at highway speed.

    The clip is placed on a map from ``maps`` (or the planned map) that has a
    highway road pool. If none of those maps has a highway pool, the plan is
    left unchanged and is not tagged ``long_highway``.
    """
    if float(min_s) <= 0.0 or float(max_s) < float(min_s):
        raise ValueError(f"long clip length range {min_s}-{max_s} s is empty")
    pool = tuple(str(m) for m in (maps if maps is not None else (plan.map,)) if str(m))
    hwy = highway_maps(pool)
    if not hwy:
        return plan
    rng = random.Random(f"gvd-long-clip:{int(plan.seed)}:{int(plan.index)}")
    duration = round(rng.uniform(float(min_s), float(max_s)), 3)
    map_name = plan.map if plan.map in hwy else rng.choice(hwy)
    tags = list(plan.tags)
    if LONG_HIGHWAY_TAG not in tags:
        tags.append(LONG_HIGHWAY_TAG)
    notes = list(plan.notes)
    if LONG_HIGHWAY_TAG not in notes:
        notes.append(LONG_HIGHWAY_TAG)
    return replace(
        plan,
        map=map_name,
        duration_s=duration,
        spawn_kind=HIGHWAY,
        long_highway=True,
        speed_min=HIGHWAY_SPEED_MIN,
        speed_max=HIGHWAY_SPEED_MAX,
        min_paint_cov=None,
        tags=tags,
        notes=notes,
    )


def plan_clip(
    index: int,
    seed: int,
    maps: Iterable[str],
    max_traffic: int,
    *,
    min_s: float = MIN_CLIP_S,
    max_s: float = MAX_CLIP_S,
    edge_cases: Iterable[str] | None = None,
) -> ClipPlan:
    """The plan for clip ``index`` of a session seeded ``seed``. Same inputs, same plan."""
    pool = tuple(str(m) for m in maps if str(m))
    if not pool:
        raise ValueError("clip mode needs at least one map")
    cases = tuple(edge_cases or EDGE_CASES)
    bad = [c for c in cases if c not in EDGE_CASES]
    if bad:
        raise ValueError(f"unknown edge case(s) {bad}; choose from {EDGE_CASES}")
    if float(min_s) <= 0.0 or float(max_s) < float(min_s):
        raise ValueError(f"clip length range {min_s}-{max_s} s is empty")
    max_traffic = max(0, int(max_traffic))
    rng = clip_rng(seed, index)
    case = rng.choice(cases)
    notes: list[str] = []
    if case in DECK_ONLY:
        if DECK_MAP in pool:
            map_name = DECK_MAP
        else:
            map_name = rng.choice(pool)
            notes.append(f"{case} wants {DECK_MAP} (spawn deck/companions); not in the map pool")
    elif case == "dirt_road":
        # Map pool only. Do not set spawn_kind: the deck has no dirt/unpaved kind.
        dirt = [m for m in pool if m in DIRT_MAPS]
        map_name = rng.choice(dirt or list(pool))
        if not dirt:
            notes.append("no dirt map in the pool")
    else:
        map_name = rng.choice(pool)
    if case == "heavy_traffic":
        traffic = max_traffic
    elif case == "cut_in":
        traffic = rng.randint(min(2, max_traffic), max_traffic) if max_traffic else 0
    else:
        traffic = rng.randint(0, max_traffic)
    if case == "night":
        tod = (21.0 + rng.uniform(0.0, 7.5)) % 24.0
    else:
        tod = rng.uniform(8.0, 17.0)
    weather = {"rain": "rain", "fog": "fog"}.get(case, "clear")
    duration = round(rng.uniform(float(min_s), float(max_s)), 3)
    spawn_seed = rng.randrange(1, 2**31 - 1)
    return ClipPlan(
        index=int(index),
        seed=int(seed),
        edge_case=case,
        map=map_name,
        traffic=int(traffic),
        weather=weather,
        time_of_day_h=round(tod, 3),
        duration_s=duration,
        spawn_seed=int(spawn_seed),
        spawn_kind=(
            None
            if case in DECK_ONLY and map_name != DECK_MAP
            else SPAWN_KIND.get(case)
        ),
        min_paint_cov=0.0 if case in ("faded_paint", "dirt_road") else None,
        traffic_ahead_m=CUT_IN_AHEAD_M if case == "cut_in" else None,
        notes=notes,
    )


def clip_argv(plan: ClipPlan, out_dir: Path, *, chase: bool = False, extra: Iterable[str] = ()) -> list[str]:
    """``collect_scene_lanes.py`` arguments for one clip (lossless raw writer, one map, the eSBR)."""
    argv = [
        # --clip-s is the planned length; --minutes is not passed (it has a 60 s
        # floor and would be overridden anyway once --clip-s is set).
        "--clip-s", repr(float(plan.duration_s)),
        # A one-map rotation, not --map: only a segment load runs the seeded deck
        # spawn and queues companion traffic (--map kept the loaded pose, 0 cars).
        "--maps", plan.map,
        # Full-rate depth/sem: at a steady 4-5 Hz the gate would thin the whole clip.
        "--gt-thin", "off",
        "--out", str(out_dir),
        "--ego-model", plan.ego_model,
        "--traffic", str(int(plan.traffic)),
        "--weather", plan.weather,
        # BeamNG's own 0-1 time (0 noon, 0.5 midnight): an hour such as 00:30
        # would otherwise be read as a 0-1 value by environment_commands.
        "--time-of-day", f"{hour_to_beamng_tod(plan.time_of_day_h):.4f}",
        "--no-vary-env",
        "--smoke-s", "0",
        "--spawn-seed", str(int(plan.spawn_seed)),
        "--excursion-seed", str(int(plan.spawn_seed) % 100000),
    ]
    if plan.spawn_kind:
        argv += ["--spawn-kind", plan.spawn_kind]
    if plan.min_paint_cov is not None:
        argv += ["--min-paint-cov", f"{plan.min_paint_cov:g}"]
    if plan.traffic_ahead_m is not None:
        argv += ["--traffic-ahead-m", f"{plan.traffic_ahead_m:g}"]
    if plan.speed_min is not None:
        argv += ["--speed-min", repr(float(plan.speed_min))]
    if plan.speed_max is not None:
        argv += ["--speed-max", repr(float(plan.speed_max))]
    if chase:
        argv.append("--chase-cam")
    argv += [str(x) for x in extra]
    return argv


def _create_new(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=False)
        return True
    except FileExistsError:
        return False


def new_session_dir(root: Path | str, stamp: str | None = None) -> Path:
    """``<root>/clips_<stamp>`` (``_2``, ``_3``... when taken). Always a new folder."""
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    base = Path(root) / f"clips_{stamp}"
    n = 1
    while True:
        cand = base if n == 1 else base.with_name(f"{base.name}_{n}")
        if _create_new(cand):
            return cand
        n += 1


def new_clip_dir(session: Path | str, plan: ClipPlan) -> Path:
    """``clip_<NNN>_<edge>_<map>`` (``_r1``, ``_r2``... when taken). Always a new, empty folder."""
    base = f"clip_{int(plan.index):03d}_{plan.edge_case}_{plan.map}"
    for n in range(0, 10000):
        cand = Path(session) / (base if n == 0 else f"{base}_r{n}")
        if _create_new(cand):
            return cand
    raise RuntimeError(f"no free clip folder for {base}")


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _sample_files(folder: Path, n: int = 5) -> list[Path]:
    files = sorted(folder.rglob("*.png"))
    if len(files) <= n:
        return files
    step = (len(files) - 1) / float(n - 1)
    return [files[int(round(i * step))] for i in range(n)]


def _observe(plan: ClipPlan, clip: Path, rows: list[dict[str, Any]], gt: list[dict[str, Any]], placed: int) -> tuple[bool | None, str]:
    import cv2
    import numpy as np

    case = plan.edge_case
    if case in ("rain", "fog"):
        return None, f"{case}: weather command sent; not observable from the recording offline"
    if case == "junction":
        return None, "junction: requested via a city spawn; not observable from state.jsonl offline"
    if case == "sharp_curve":
        best = 0.0
        for row in rows:
            ego = row.get("ego") or {}
            yr, sp = _num(ego.get("yaw_rate")), _num(ego.get("speed_mps"))
            if yr is None or sp is None or sp < SHARP_CURVE_MIN_MPS:
                continue
            best = max(best, abs(yr) / sp)
        radius = math.inf if best <= 0 else 1.0 / best
        return best >= SHARP_CURVE_PER_M, f"sharp_curve: tightest radius {radius:.0f} m (need <= {1.0 / SHARP_CURVE_PER_M:.0f} m)"
    if case == "merge":
        kinds = {str(r.get("spawn_type")) for r in rows if r.get("spawn_type")}
        return "ramp" in kinds, f"merge: spawn types {sorted(kinds)}"
    if case == "heavy_traffic":
        need = max(1, math.ceil(HEAVY_PLACED_SHARE * max(1, plan.traffic)))
        return placed >= need, f"heavy_traffic: {placed} companions placed, need {need} of {plan.traffic}"
    if case == "night":
        lumas = []
        for f in _sample_files(clip / "raw" / "main"):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                lumas.append(float(img.mean()))
        if not lumas:
            return False, "night: no main frame to measure"
        mean = sum(lumas) / len(lumas)
        return mean <= NIGHT_MAX_LUMA, f"night: mean main luma {mean:.0f} (need <= {NIGHT_MAX_LUMA:.0f})"
    if case == "faded_paint":
        vals = []
        for row in gt:
            valid = (row.get("tech") or {}).get("lane_gt_valid")
            if isinstance(valid, list) and valid:
                vals.append(any((_num(v) or 0.0) > 0.5 for v in valid[:-1]))
        if not vals:
            return False, "faded_paint: no lane labels"
        share = sum(vals) / len(vals)
        return share <= FADED_MAX_PAINT_SHARE, f"faded_paint: painted-label share {share:.2f} (need <= {FADED_MAX_PAINT_SHARE})"
    if case == "dirt_road":
        try:
            classes = json.loads((clip / "sem_classes.json").read_text(encoding="utf-8")).get("classes") or {}
        except Exception:
            return False, "dirt_road: no sem_classes.json"
        ids = [int(k) for k, v in classes.items() if str(v).upper() in DIRT_CLASSES]
        best = 0.0
        for f in _sample_files(clip / "raw" / "main_sem"):
            sem = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if sem is None or not ids:
                continue
            low = sem[int(sem.shape[0] * 2 / 3):, :]
            best = max(best, float(np.isin(low, ids).mean()))
        return best >= DIRT_MIN_SHARE, f"dirt_road: dirt share of the lower third {best:.2f} (need >= {DIRT_MIN_SHARE})"
    if case == "cut_in":
        # A car seen >= CUT_IN_FROM_M out, then within CUT_IN_TO_M of the ego
        # line, both ahead, within CUT_IN_WINDOW_S. A plain lead car is never
        # "out". Rows where the road bends enough to fake that offset at the
        # car's range (a lead car on a curve) are not read, and an old "out"
        # sighting expires, so tailing a car long after it merged is not one.
        seen: dict[str, float | None] = {}
        for row in rows:
            ego = row.get("ego") or {}
            pos, d = ego.get("pos"), ego.get("dir")
            if not (isinstance(pos, list) and isinstance(d, list) and len(pos) >= 2 and len(d) >= 2):
                continue
            norm = math.hypot(d[0], d[1]) or 1.0
            fx, fy = d[0] / norm, d[1] / norm
            t = _num(row.get("t"))
            yr, sp = _num(ego.get("yaw_rate")), _num(ego.get("speed_mps"))
            bend_k = abs(yr) / sp if yr is not None and sp is not None and sp > 0.5 else 0.0
            for name, t_out in list(seen.items()):
                if t is not None and t_out is not None and t - t_out > CUT_IN_WINDOW_S:
                    del seen[name]
            for car in row.get("traffic") or []:
                cp = car.get("pos") if isinstance(car, dict) else None
                if not (isinstance(cp, list) and len(cp) >= 2):
                    continue
                dx, dy = cp[0] - pos[0], cp[1] - pos[1]
                ahead = dx * fx + dy * fy
                lat = abs(-dx * fy + dy * fx)
                name = str(car.get("name"))
                if 0.5 * bend_k * ahead * ahead > CUT_IN_BEND_MAX_M:
                    continue
                if 0.0 < ahead < CUT_IN_AHEAD_MAX_M:
                    if lat >= CUT_IN_FROM_M:
                        seen[name] = t
                    elif lat <= CUT_IN_TO_M and name in seen:
                        return True, f"cut_in: {name} moved into the ego lane {ahead:.0f} m ahead"
        return False, "cut_in: no car crossed into the ego lane ahead"
    return None, f"{case}: no offline observation rule"


def quality_check(clip: Path | str, plan: ClipPlan, *, exit_code: int) -> dict[str, Any]:
    """Quick read-only check of one clip. Bad clips are marked ``pretrain_only``, never deleted."""
    import cv2

    from python.sensors.cameras import CAM_IDS

    clip = Path(clip)
    reasons: list[str] = []
    if int(exit_code) != 0:
        reasons.append(f"collector exit {int(exit_code)}")
    rows = _rows(clip / "state.jsonl")
    gt = _rows(clip / "gt.jsonl")
    raw = clip / "raw"
    lossy = [str(p.relative_to(clip)) for p in raw.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")] if raw.is_dir() else []
    if lossy:
        reasons.append(f"lossy frame files present ({len(lossy)})")
    ts = [t for t in (_num(r.get("t")) for r in rows) if t is not None]
    span = (max(ts) - min(ts)) if len(ts) >= 2 else 0.0
    mains = sum(1 for r in rows if "main" in (r.get("cams") or {}))
    hz = mains / span if span > 0 else 0.0
    if hz < MIN_MAIN_HZ:
        reasons.append(f"main camera {hz:.1f} Hz over {span:.1f} s (need >= {MIN_MAIN_HZ:.0f})")
    if span < 0.5 * float(plan.duration_s):
        reasons.append(f"recorded {span:.1f} s of a {plan.duration_s:.0f} s clip")
    missing = []
    for cid in CAM_IDS:
        for sub in (cid, f"{cid}_depth", f"{cid}_sem"):
            if not any((raw / sub).rglob("*.png")):
                missing.append(sub)
    if missing:
        reasons.append(f"no frames for {','.join(missing)}")
    undecodable = []
    for cid in CAM_IDS:
        for sub in (cid, f"{cid}_depth", f"{cid}_sem"):
            files = sorted((raw / sub).rglob("*.png"))
            for f in files[:1] + files[-1:]:
                if cv2.imread(str(f), cv2.IMREAD_UNCHANGED) is None:
                    undecodable.append(str(f.relative_to(clip)))
    if undecodable:
        reasons.append(f"undecodable frames {undecodable[:3]}")
    speeds = sorted(s for s in (_num((r.get("ego") or {}).get("speed_mps")) for r in rows) if s is not None)
    median = speeds[len(speeds) // 2] if speeds else 0.0
    if median < MIN_MOVING_MPS:
        reasons.append(f"ego median speed {median:.1f} m/s (need >= {MIN_MOVING_MPS:.0f})")
    if any((_num((r.get("ego") or {}).get("damage")) or 0.0) > 0.0 for r in rows) or any(r.get("impact") for r in rows):
        reasons.append("ego damaged or impact logged")
    placed = 0
    for r in rows:
        names = {str(c.get("name")) for c in (r.get("traffic") or []) if isinstance(c, dict)}
        placed = max(placed, sum(1 for n in names if n.startswith("gvd_traffic_")))
    observed, note = _observe(plan, clip, rows, gt, placed)
    if plan.long_highway:
        observed, note = None, f"{LONG_HIGHWAY_TAG}: edge-case observation skipped; long highway/arterial run"
    elif observed is False:
        reasons.append(f"edge case not observed ({note})")
    clip_class = "train" if not reasons else ("reject" if (len(rows) == 0 or lossy or undecodable) else "pretrain_only")
    return {
        "quality": "pretrain_only" if reasons else "train",
        "clip_class": clip_class,
        "reasons": reasons,
        "rows": len(rows),
        "span_s": round(span, 3),
        "main_hz": round(hz, 3),
        "median_speed_mps": round(median, 3),
        "traffic_placed": int(placed),
        "edge_case_observed": observed,
        "edge_case_note": note,
    }


def write_tags(
    clip: Path | str, plan: ClipPlan, qc: Mapping[str, Any], extra: Mapping[str, Any] | None = None, *, gfx: str = GFX
) -> Path:
    """``clip_tags.json`` beside the recording (created once; an existing file is never replaced)."""
    path = Path(clip) / "clip_tags.json"
    payload = {
        "kind": "gvd_clip_tags",
        "edge_case": plan.edge_case,
        "map": plan.map,
        "traffic_requested": plan.traffic,
        "traffic_placed": qc.get("traffic_placed"),
        "weather": plan.weather,
        "time_of_day_h": plan.time_of_day_h,
        "duration_s": plan.duration_s,
        "seed": plan.seed,
        "index": plan.index,
        "spawn_seed": plan.spawn_seed,
        "spawn_kind": plan.spawn_kind,
        "ego_model": plan.ego_model,
        "gfx": gfx,
        "lossless": True,
        "quality": qc.get("quality"),
        "reasons": list(qc.get("reasons") or []),
        "edge_case_observed": qc.get("edge_case_observed"),
        "edge_case_note": qc.get("edge_case_note"),
        "qc": dict(qc),
        "plan_notes": list(plan.notes),
        "tags": list(plan.tags),
        "long_highway": bool(plan.long_highway),
        **dict(extra or {}),
    }
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2) + "\n")
    return path


def run_clips(
    root: Path | str,
    *,
    clips: int,
    seed: int,
    maps: Iterable[str],
    max_traffic: int,
    runner: Callable[[list[str]], int],
    min_s: float = MIN_CLIP_S,
    max_s: float = MAX_CLIP_S,
    edge_cases: Iterable[str] | None = None,
    chase: bool = False,
    extra: Iterable[str] = (),
    log: Callable[[str], None] = print,
    gfx: str = GFX,
    after_clip: Callable[..., str | None] | None = None,
    stop_file: Path | str | None = None,
    long_clips: int = DEFAULT_LONG_CLIPS,
    long_min_s: float = DEFAULT_LONG_CLIP_MIN_S,
    long_max_s: float = DEFAULT_LONG_CLIP_MAX_S,
) -> Path:
    """Record ``clips`` clips into a new session folder. ``runner(argv)`` runs the collector.

    ``after_clip(clip, plan, code, qc)`` may return a ``stop_reason`` to halt
    the batch; that reason is written to ``session.json`` and remaining clips
    are not started.

    ``stop_file``: checked before each clip. When it exists the batch ends with
    ``stop_reason=stop_file``; a clip already recording finishes and is tagged
    first. The file is never deleted here (remove it to resume).
    """
    maps = tuple(maps)
    cases = tuple(edge_cases or EDGE_CASES)
    session = new_session_dir(root)
    results: list[dict[str, Any]] = []
    stop_reason: str | None = None
    manifest = session / "session.json"

    def _save() -> None:
        tmp = manifest.with_suffix(".json.tmp")
        payload: dict[str, Any] = {
            "kind": "gvd_clip_session",
            "clips": int(clips),
            "seed": int(seed),
            "maps": list(maps),
            "max_traffic": int(max_traffic),
            "edge_cases": list(cases),
            "min_s": float(min_s),
            "max_s": float(max_s),
            "chase_cam": bool(chase),
            "ego_model": EGO_MODEL,
            "gfx": gfx,
            "long_clips": int(long_clips),
            "long_min_s": float(long_min_s),
            "long_max_s": float(long_max_s),
            "results": results,
        }
        if stop_reason:
            payload["stop_reason"] = stop_reason
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(manifest)

    _save()
    long_at = set(long_clip_indices(int(clips), int(long_clips), int(seed)))
    for index in range(int(clips)):
        if stop_file is not None and Path(stop_file).exists():
            stop_reason = STOP_FILE_REASON
            log(f"session stop reason={stop_reason} ({stop_file}); {index} clip(s) recorded")
            _save()
            break
        plan = plan_clip(index, seed, maps, max_traffic, min_s=min_s, max_s=max_s, edge_cases=cases)
        if index in long_at:
            plan = apply_long_highway(plan, min_s=long_min_s, max_s=long_max_s, maps=maps)
        clip = new_clip_dir(session, plan)
        argv = clip_argv(plan, clip, chase=chase, extra=extra)
        log(f"clip {index + 1}/{clips} {clip.name}: {plan.edge_case} map={plan.map} traffic={plan.traffic} "
            f"weather={plan.weather} tod={plan.time_of_day_h:.1f}h {plan.duration_s:.0f}s")
        started = time.time()
        try:
            code = int(runner(argv))
        except Exception as exc:  # a crashed clip is kept and marked, the session goes on
            log(f"clip {clip.name}: collector raised {type(exc).__name__}: {exc}")
            code = 99
        qc = quality_check(clip, plan, exit_code=code)
        write_tags(clip, plan, qc, {"collector_exit": code, "argv": argv, "started_wall": started}, gfx=gfx)
        log(f"clip {clip.name}: {qc['quality']} {'; '.join(qc['reasons']) or 'ok'}")
        results.append({"dir": clip.name, "edge_case": plan.edge_case, "map": plan.map,
                        "quality": qc["quality"], "exit": code})
        _save()
        if after_clip is not None:
            halt = after_clip(clip, plan, code, qc)
            if halt:
                stop_reason = str(halt)
                log(f"session stop reason={stop_reason}")
                _save()
                break
    return session
