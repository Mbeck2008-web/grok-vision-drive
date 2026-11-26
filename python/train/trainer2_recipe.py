"""Trainer 2.0 curriculum runner (docs/specs/trainer_2.md, Appendix C PR 6).

``train_scene --recipe trainer2`` trains SceneNet through a fixed curriculum.
The recipe flag itself stays default off (**spec proposal**). Stages 0, 1, 2
and 2b run; stages 3-6 are skipped and logged until Reality View is built and
integrated (decided by Michael 2026-11-25). Stage 7 (joint) runs over the
heads that have already run, at peak LR x 0.1 (**spec proposal**).

This recipe is meant for a live export: ``--scale < 1`` and ``--augment`` are
refused. Those refusals were deferred from Historian PR 2b; this module does
not invent ``strip_to_nchw`` and does not import or edit ``scene_ckpt_runtime``.

Exit ("no gain > 1% over 3 evals, or a max epoch count"), the 5% non-regression
gate, joint-head 2% cap, and ``--accum`` limits are **spec proposals** and are
pinned in ``recipe_run_config`` / the run manifest. This PR does not decide them.

`--kd` is refused: teacher KD is Appendix C PR 8, not this PR.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

# --- proposal / decided constants (do not retune in this PR) -----------------

# Spec proposal: the recipe flag itself stays default off.
RECIPE_DEFAULT = ""
# Spec proposal: holdout primary metric has no gain > 1% over 3 evals (--eval-every).
EXIT_GAIN = 0.01
EXIT_PATIENCE_EVALS = 3
# Spec proposal: a max epoch count, pinned in the run manifest before training.
STAGE_MAX_EPOCHS = 20
# Spec proposal: a stage may not make any earlier-stage holdout metric > 5% (relative) worse.
NON_REGRESSION_FRAC = 0.05
# Spec proposal: joint "no head worse than its stage exit by > 2%".
JOINT_HEAD_REGRESSION_FRAC = 0.02
# Spec proposal: --accum used only if the auto-fit batch < 4 windows (floor 2, cap 8).
ACCUM_BATCH_LT = 4
ACCUM_FLOOR = 2
ACCUM_CAP = 8
# Decided by Michael 2026-11-25: live inference VRAM stays <= 4 GB.
LIVE_INFER_VRAM_GB = 4.0
# Decided by Michael 2026-11-25: stages 3-6 wait until Reality View is built and integrated.
REALITY_VIEW_BUILT_AND_INTEGRATED = False
# Deploy check: only these presets may ship as the live student.
DEPLOY_PRESETS = frozenset({"tiny", "small"})

PROPOSAL_FLAGS = {
    "RECIPE_DEFAULT_OFF": "the --recipe flag itself stays default off",
    "EXIT_GAIN": "no gain > 1% over 3 evals (EXIT_GAIN = 0.01)",
    "EXIT_PATIENCE_EVALS": "patience of 3 evals",
    "STAGE_MAX_EPOCHS": "max epoch count per stage (STAGE_MAX_EPOCHS = 20)",
    "NON_REGRESSION_FRAC": "earlier-stage metric may not get > 5% relative worse",
    "JOINT_HEAD_REGRESSION_FRAC": "joint: no head worse than its stage exit by > 2%",
    "ACCUM": "--accum only if auto-fit batch < 4 (floor 2, cap 8)",
    "WEIGHT_LR": "GradNorm WEIGHT_LR = 0.025 (owned by trainer2_balance, unchanged)",
    "JOINT_PEAK_MULT": "joint peak = 0.1 x CLI LR (owned by lr_schedule, unchanged)",
}

STAGE_ORDER = ("0", "1", "2", "2b", "3", "4", "5", "6", "7")

# Primary holdout metrics used for exit / non-regression. Direction: True = higher is better.
METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "ssl_pretext": False,
    "depth_absrel": False,
    "sem_miou": True,
    "lane_r": True,
    "lane_rms": False,
    "missed_stop": False,
    "missed_light": False,
    "false_brake": False,
}

STAGE_PRIMARY_METRICS: dict[str, tuple[str, ...]] = {
    "0": ("ssl_pretext",),
    "1": ("depth_absrel", "sem_miou"),
    "2": ("lane_rms",),
    "2b": ("missed_stop", "missed_light", "false_brake"),
    "3": ("bev_iou",),
    "4": ("future_bev_iou",),
    "5": ("depth_absrel",),
    "6": ("refine",),
    "7": ("lane_rms",),
}


class RecipeRefused(ValueError):
    """Illegal --recipe trainer2 request (CLI, deploy check, or entry)."""


class CheckpointRejected(ValueError):
    """Non-regression (or joint-head) gate rejected this stage's checkpoint; the run stops."""


@dataclass(frozen=True)
class StageSpec:
    id: str
    name: str
    requires_reality_view: bool
    entering_loss_terms: tuple[str, ...]
    entering_modules: tuple[str, ...]


# Loss-term names follow train_scene / scene_net.LOSS_TERMS plus aux / ctl extras.
# Module names are SceneNet children used by lr_schedule.scene_param_groups.
STAGES: dict[str, StageSpec] = {
    "0": StageSpec("0", "SSL", False, (), ()),
    "1": StageSpec("1", "depth/sem", False, ("aux_depth", "aux_sem"), ("aux_depth_head", "aux_sem_head")),
    "2": StageSpec(
        "2",
        "lanes",
        False,
        ("points", "valid", "lane_gt"),
        ("lane_head", "lane_valid_head", "lane_gt_head", "road_edge_head"),
    ),
    "2b": StageSpec(
        "2b",
        "stop",
        False,
        ("class", "lamps", "ctl", "stop_line"),
        ("sign_head", "sign_class_head", "sign_state_head", "ctl_head", "stop_line_head"),
    ),
    "3": StageSpec("3", "BEV", True, ("bev",), ("bev_head",)),
    "4": StageSpec("4", "temporal", True, ("future_bev",), ("future_bev_head",)),
    "5": StageSpec("5", "3DGS", True, ("gs",), ("gs_head",)),
    "6": StageSpec("6", "refine", True, (), ()),
    "7": StageSpec("7", "joint", False, (), ()),
}


@dataclass
class PlannedStage:
    id: str
    name: str
    action: str  # "run" or "skip"
    reason: str
    log_skip: bool = False
    requires_reality_view: bool = False


@dataclass
class StageResult:
    ckpt: str | None
    metrics: dict[str, float]
    evals: list[float]
    epochs: int


@dataclass
class CurriculumResult:
    trained: list[str]
    skipped: list[str]
    metrics: dict[str, float]
    last_ckpt: str | None
    logs: list[str] = field(default_factory=list)


def validate_live_export_recipe(
    *,
    recipe: str | None,
    scale: float,
    augment: bool,
    kd: bool = False,
) -> None:
    """Refuse unknown recipes, and for trainer2 refuse train-only input transforms and --kd."""
    name = "" if recipe is None else str(recipe).strip()
    if not name or name in ("off", "false", "0"):
        return
    if name != "trainer2":
        raise RecipeRefused(f"unknown --recipe {name!r}; only 'trainer2' is defined")
    if not math.isfinite(float(scale)) or float(scale) < 1.0:
        raise RecipeRefused(
            "--recipe trainer2 refuses --scale < 1 (train-only downscale; this recipe is meant for a live export)"
        )
    if augment:
        raise RecipeRefused(
            "--recipe trainer2 refuses --augment (train-only photometric jitter; this recipe is meant for a live export)"
        )
    if kd:
        raise RecipeRefused("--kd is PR 8 (optional teacher KD) and is not enabled in PR 6")


def validate_recipe_args(args: Any) -> None:
    """Validate argparse / namespace flags. No-op when the recipe is off."""
    recipe = getattr(args, "recipe", "") or ""
    validate_live_export_recipe(
        recipe=recipe,
        scale=float(getattr(args, "scale", 1.0) or 1.0),
        augment=bool(getattr(args, "augment", False)),
        kd=bool(getattr(args, "kd", False)),
    )
    if recipe and recipe not in ("off", "false", "0"):
        parse_stage_filter(getattr(args, "stage", "") or "")


def parse_stage_filter(text: str | None) -> tuple[str, ...] | None:
    """``None`` means the full curriculum. Values are ids from STAGE_ORDER, in that order."""
    raw = str(text or "").strip()
    if not raw:
        return None
    wanted: list[str] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token and token not in STAGE_ORDER:
            start, _, end = token.partition("-")
            start, end = start.strip(), end.strip()
            if start not in STAGE_ORDER or end not in STAGE_ORDER:
                raise RecipeRefused(f"unknown --stage {token!r}; expected ids in {STAGE_ORDER}")
            lo, hi = STAGE_ORDER.index(start), STAGE_ORDER.index(end)
            if lo > hi:
                raise RecipeRefused(f"--stage range {token!r} is backwards")
            wanted.extend(STAGE_ORDER[lo : hi + 1])
        else:
            if token not in STAGE_ORDER:
                raise RecipeRefused(f"unknown --stage {token!r}; expected ids in {STAGE_ORDER}")
            wanted.append(token)
    # Keep STAGE_ORDER, drop duplicates.
    return tuple(s for s in STAGE_ORDER if s in set(wanted))


def default_heads_exist() -> dict[str, bool]:
    """What this checkout actually has. BEV / temporal / 3DGS / refine are not on main."""
    train = Path(__file__).resolve().parent
    repo = train.parent.parent
    bev = (train / "bev_lss.py").is_file()
    return {
        "0": (train / "exp_ssl.py").is_file(),
        "1": (train / "sensor_aux.py").is_file(),
        "2": (train / "scene_net.py").is_file(),
        "2b": (train / "stop_events.py").is_file(),
        "3": bev,
        "4": bev,
        "5": (repo / "docs" / "specs" / "reality_view_3dgs.md").is_file() and False,  # not integrated
        "6": False,
        "7": True,
    }


def default_data_exist() -> dict[str, bool]:
    """Entry also requires the stage's data path to exist on main (modules, not a recording)."""
    heads = default_heads_exist()
    # Data sources for 0-2 / 2b already landed; 3-6 wait on Reality View labels.
    return {k: (bool(heads[k]) if k not in ("3", "4", "5", "6") else False) for k in STAGE_ORDER}


def plan_curriculum(
    *,
    reality_view_ready: bool | None = None,
    heads_exist: Mapping[str, bool] | None = None,
    data_exist: Mapping[str, bool] | None = None,
    stages: Iterable[str] | None = None,
) -> list[PlannedStage]:
    """Fixed stage order. Stages 3-6 are skip-logged until Reality View is ready *and* heads exist."""
    rv = REALITY_VIEW_BUILT_AND_INTEGRATED if reality_view_ready is None else bool(reality_view_ready)
    heads = dict(default_heads_exist() if heads_exist is None else heads_exist)
    data = dict(default_data_exist() if data_exist is None else data_exist)
    only = None if stages is None else tuple(stages)
    planned: list[PlannedStage] = []
    for sid in STAGE_ORDER:
        if only is not None and sid not in only:
            continue
        spec = STAGES[sid]
        if spec.requires_reality_view and not rv:
            planned.append(
                PlannedStage(
                    sid,
                    spec.name,
                    "skip",
                    "skipped until Reality View is built and integrated",
                    log_skip=True,
                    requires_reality_view=True,
                )
            )
            continue
        missing = []
        if not heads.get(sid, False):
            missing.append("head")
        if not data.get(sid, False):
            missing.append("data")
        if missing:
            planned.append(
                PlannedStage(
                    sid,
                    spec.name,
                    "skip",
                    f"{' and '.join(missing)} not on main",
                    log_skip=True,
                    requires_reality_view=spec.requires_reality_view,
                )
            )
            continue
        planned.append(
            PlannedStage(sid, spec.name, "run", "entry: previous stage exited and head/data exist on main", log_skip=False)
        )
    return planned


def entry_allowed(
    stage_id: str,
    *,
    prev_exited: bool,
    heads_exist: bool,
    data_exist: bool,
    reality_view_ready: bool | None = None,
) -> tuple[bool, str]:
    """Entry = the previous stage exited, plus the stage's data and head exist on main.

    For stages 3-6, entry also requires Reality View built and integrated.
    """
    if stage_id not in STAGES:
        return False, f"unknown stage {stage_id}"
    spec = STAGES[stage_id]
    rv = REALITY_VIEW_BUILT_AND_INTEGRATED if reality_view_ready is None else bool(reality_view_ready)
    if spec.requires_reality_view and not rv:
        return False, "skipped until Reality View is built and integrated"
    if stage_id != "0" and not prev_exited:
        return False, "previous stage has not exited"
    if not heads_exist:
        return False, "head does not exist on main"
    if not data_exist:
        return False, "data does not exist on main"
    return True, "ok"


def metric_gain(old: float, new: float, *, metric: str) -> float:
    """Positive gain is an improvement. Relative to ``|old|``; a zero baseline with no change is 0."""
    higher = METRIC_HIGHER_IS_BETTER.get(metric, False)
    old_f, new_f = float(old), float(new)
    if not math.isfinite(old_f) or not math.isfinite(new_f):
        return 0.0
    if old_f == 0.0:
        if new_f == old_f:
            return 0.0
        # A move away from a zero baseline counts as a full gain (or a loss → 0).
        improved = new_f > old_f if higher else new_f < old_f
        return 1.0 if improved else 0.0
    if higher:
        return (new_f - old_f) / abs(old_f)
    return (old_f - new_f) / abs(old_f)


def _worse_frac(old: float, new: float, *, metric: str) -> float:
    """Positive means worse, relative to ``|old|``."""
    return -metric_gain(old, new, metric=metric)


def stage_exited(stage_id: str, evals: list[float], *, epochs: int, metric: str | None = None) -> bool:
    """True when the max-epoch cap is hit, or the last ``EXIT_PATIENCE_EVALS`` steps each gain <= EXIT_GAIN."""
    if int(epochs) >= STAGE_MAX_EPOCHS:
        return True
    if len(evals) < EXIT_PATIENCE_EVALS + 1:
        return False
    name = metric or STAGE_PRIMARY_METRICS.get(stage_id, ("lane_rms",))[0]
    window = [float(x) for x in evals[-(EXIT_PATIENCE_EVALS + 1) :]]
    for prev, cur in zip(window, window[1:]):
        if metric_gain(prev, cur, metric=name) > EXIT_GAIN:
            return False
    return True


def check_non_regression(
    earlier: Mapping[str, float],
    current: Mapping[str, float],
    *,
    frac: float = NON_REGRESSION_FRAC,
) -> None:
    """Reject the checkpoint if any earlier-stage metric is > ``frac`` relative worse.  Exactly ``frac`` is allowed."""
    for name, old in earlier.items():
        if name not in current:
            continue
        worse = _worse_frac(float(old), float(current[name]), metric=name)
        # "> frac": exact frac (and 0.05-vs-1.05/1.0 binary residue) is allowed.
        if worse - float(frac) > 1e-12:
            raise CheckpointRejected(
                f"stage checkpoint rejected: {name} got {worse:.6%} worse "
                f"({old} -> {current[name]}; limit is {float(frac):.2%} relative, a spec proposal)"
            )



def live_vram_fits(vram: float) -> bool:
    """True iff the estimate is <= the 4 GB live-inference cap (Michael 2026-11-25)."""
    return float(vram) <= LIVE_INFER_VRAM_GB


def student_export_fits_deploy(*, preset: str, student_param_count: int, exported_param_count: int) -> None:
    """The exported student's param count matches the architecture, and the preset fits a 4 GB live budget."""
    from python.train.model_size import PRESETS

    name = str(preset or "")
    if name not in DEPLOY_PRESETS:
        raise RecipeRefused(
            f"student export preset {name!r} is not a live deploy size "
            f"(must be one of {sorted(DEPLOY_PRESETS)})"
        )
    if name not in PRESETS:
        raise RecipeRefused(f"unknown model-size preset {name!r}")
    vram = float(PRESETS[name].est_vram_gb)
    if not live_vram_fits(vram):
        raise RecipeRefused(
            f"student export preset {name!r} estimates {vram} GB; live inference VRAM must be "
            f"<= {LIVE_INFER_VRAM_GB} GB"
        )
    if int(student_param_count) != int(exported_param_count):
        raise RecipeRefused(
            f"exported ONNX param count {int(exported_param_count)} != student "
            f"architecture_parameter_count {int(student_param_count)}"
        )


def may_unlink_checkpoint(path: Path | str, *, owned_partials: Iterable[Path | str] | None = None) -> bool:
    """The trainer may only unlink its own ``.partial`` files, never an existing checkpoint."""
    p = Path(path)
    if owned_partials is not None:
        owned = {Path(x).resolve() for x in owned_partials}
        try:
            return p.resolve() in owned
        except OSError:
            return False
    return ".partial" in p.name


def safe_unlink_checkpoint(path: Path | str, *, owned_partials: Iterable[Path | str] | None = None) -> None:
    p = Path(path)
    if not may_unlink_checkpoint(p, owned_partials=owned_partials):
        raise RecipeRefused(f"refusing to unlink existing checkpoint {p}")
    if p.exists():
        p.unlink()


def snapshot_checkpoints(root: Path | str) -> dict[str, str]:
    """sha256 of non-partial checkpoint-like files under ``root`` (files only)."""
    out: dict[str, str] = {}
    base = Path(root)
    if not base.is_dir():
        return out
    for path in sorted(base.rglob("*")):
        if not path.is_file() or ".partial" in path.name:
            continue
        if path.suffix.lower() not in {".ckpt", ".npz", ".onnx", ".pt", ".pth", ".bin"} and "ckpt" not in path.name.lower():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out[str(path.relative_to(base))] = digest
    return out


def assert_old_checkpoints_untouched(before: Mapping[str, str], after: Mapping[str, str]) -> None:
    for name, digest in before.items():
        if name not in after:
            raise RecipeRefused(f"existing checkpoint {name} was removed")
        if after[name] != digest:
            raise RecipeRefused(f"existing checkpoint {name} was modified (sha256 changed)")


def stage_train_kwargs(stage_id: str) -> dict[str, Any]:
    """Flags / heads for one stage. ``depth_hires`` stays opt-in (PR #265); curbs stay omitted."""
    if stage_id not in STAGES:
        raise RecipeRefused(f"unknown stage {stage_id}")
    if stage_id == "0":
        return {"kind": "ssl", "heads": (), "aux_depth": False, "aux_sem": False, "joint": False}
    if stage_id == "1":
        # aux_depth / aux_sem are flags, not --heads names. An empty heads tuple means
        # "do not fall back to the default lane/sign set". depth_hires is not defaulted on.
        return {"kind": "supervised", "heads": (), "aux_depth": True, "aux_sem": True, "joint": False}
    if stage_id == "2":
        # curb_* is in the spec's active-head list, but --heads curbs is refused on main
        # until a labeler sets curb_valid. This PR does not invent that labeler.
        return {
            "kind": "supervised",
            "heads": ("lanes", "lane_gt", "road_edge"),
            "aux_depth": False,
            "aux_sem": False,
            "joint": False,
        }
    if stage_id == "2b":
        # ctl_refusal: ctl needs lane_gt (--heads lane_gt,ctl).
        return {
            "kind": "supervised",
            "heads": ("lane_gt", "signs", "stop_line", "ctl"),
            "aux_depth": False,
            "aux_sem": False,
            "joint": False,
        }
    if stage_id == "7":
        return {
            "kind": "supervised",
            "heads": ("lanes", "objects", "signs", "lane_gt", "road_edge", "stop_line", "ctl"),
            "aux_depth": True,
            "aux_sem": True,
            "joint": True,
        }
    return {"kind": "supervised", "heads": (), "aux_depth": False, "aux_sem": False, "joint": False}


def _prior_modules(stage_id: str) -> list[str]:
    out: list[str] = []
    for sid in STAGE_ORDER:
        if sid == stage_id:
            break
        if STAGES[sid].requires_reality_view:
            continue  # not yet trained when 3-6 are skipped
        out.extend(STAGES[sid].entering_modules)
    return out


def stage_lr_mults_for(stage_id: str) -> dict[str, float]:
    """Differential LR for this stage. Numbers are the existing lr_schedule proposals, unchanged."""
    from python.train.lr_schedule import stage_lr_mults

    if stage_id == "0":
        return stage_lr_mults(ssl_done=False, trained_heads=(), entering_heads=())
    entering = STAGES[stage_id].entering_modules if stage_id != "7" else ()
    trained = _prior_modules(stage_id) if stage_id != "7" else _prior_modules("7") + list(STAGES["7"].entering_modules)
    # A module listed as entering must not also be trained.
    trained = tuple(n for n in trained if n not in set(entering))
    return stage_lr_mults(ssl_done=True, trained_heads=trained, entering_heads=entering)


def resolve_accum(*, requested: int, batch: int) -> int:
    """``1`` unless the recipe asked for accum *and* the auto-fit batch is < 4 windows."""
    want = int(requested or 0)
    if want <= 0 or int(batch) >= ACCUM_BATCH_LT:
        return 1
    return min(ACCUM_CAP, max(ACCUM_FLOOR, want))


def recipe_run_config(**extra: Any) -> dict[str, Any]:
    """Pinned values written into the run manifest before training."""
    cfg = {
        "recipe": "trainer2",
        "exit_gain": EXIT_GAIN,
        "exit_patience_evals": EXIT_PATIENCE_EVALS,
        "stage_max_epochs": STAGE_MAX_EPOCHS,
        "non_regression_frac": NON_REGRESSION_FRAC,
        "joint_head_regression_frac": JOINT_HEAD_REGRESSION_FRAC,
        "proposal_flags": dict(PROPOSAL_FLAGS),
        "reality_view_built_and_integrated": REALITY_VIEW_BUILT_AND_INTEGRATED,
        "stage_order": list(STAGE_ORDER),
    }
    cfg.update(extra)
    return cfg


def run_curriculum(
    train_fn: Callable[[PlannedStage, str | None], StageResult],
    *,
    log: Callable[[str], None] | None = None,
    reality_view_ready: bool | None = None,
    heads_exist: Mapping[str, bool] | None = None,
    data_exist: Mapping[str, bool] | None = None,
    stages: Iterable[str] | None = None,
    earlier_metrics: Mapping[str, float] | None = None,
) -> CurriculumResult:
    """Run live stages in order, skip-log 3-6 until Reality View, reject a regressing checkpoint."""
    notes: list[str] = []

    def _log(msg: str) -> None:
        notes.append(msg)
        if log is not None:
            log(msg)

    plan = plan_curriculum(
        reality_view_ready=reality_view_ready,
        heads_exist=heads_exist,
        data_exist=data_exist,
        stages=stages,
    )
    trained: list[str] = []
    skipped: list[str] = []
    kept = dict(earlier_metrics or {})
    prev_exited = True
    prev_ckpt: str | None = None
    for stage in plan:
        if stage.action == "skip":
            skipped.append(stage.id)
            _log(f"[GVD] recipe trainer2: skip stage {stage.id} ({stage.name}): {stage.reason}")
            continue
        exist_h = True if heads_exist is None else bool(heads_exist.get(stage.id, False))
        exist_d = True if data_exist is None else bool(data_exist.get(stage.id, False))
        if heads_exist is None:
            exist_h = bool(default_heads_exist().get(stage.id, False))
        if data_exist is None:
            exist_d = bool(default_data_exist().get(stage.id, False))
        ok, reason = entry_allowed(
            stage.id,
            prev_exited=prev_exited,
            heads_exist=exist_h,
            data_exist=exist_d,
            reality_view_ready=reality_view_ready,
        )
        if not ok:
            raise RecipeRefused(f"stage {stage.id} entry refused: {reason}")
        result = train_fn(stage, prev_ckpt)
        trained.append(stage.id)
        try:
            check_non_regression(kept, result.metrics)
        except CheckpointRejected as exc:
            _log(f"[GVD] recipe trainer2: stage {stage.id} checkpoint rejected (non-regression): {exc}")
            raise
        kept.update(result.metrics)
        prev_ckpt = result.ckpt
        prev_exited = stage_exited(stage.id, list(result.evals), epochs=int(result.epochs))
        if not prev_exited:
            raise RecipeRefused(
                f"stage {stage.id} did not exit (need no gain > {EXIT_GAIN:.0%} over "
                f"{EXIT_PATIENCE_EVALS} evals, or {STAGE_MAX_EPOCHS} epochs)"
            )
        _log(f"[GVD] recipe trainer2: stage {stage.id} ({stage.name}) exited, ckpt={result.ckpt}")
    return CurriculumResult(trained=trained, skipped=skipped, metrics=kept, last_ckpt=prev_ckpt, logs=notes)
