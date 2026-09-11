"""Discover and load the nets GVD can actually run.

There is no in-repo model zoo (weights are gitignored). This catalog lists:

* Detector: YOLOv8 *detect* ONNX or Ultralytics ``.pt`` found under ``models/``
  (stems ``yolov8n`` / ``s`` / ``m`` / ``l`` / ``x``, plus any other ``yolov8*.onnx|pt``).
  Always-available fallbacks: ``synthetic`` (smoke leads) and ``empty`` (no boxes).
* E2E: PilotNet-scale ONNX (``e2e*.onnx`` / ``config/control.yaml`` ``e2e.model``) or the
  numpy stub. Named feeds: main / wide 1x3x180x320 + kin 1x2.

Not loadable here: lane nets (Hough only), transformers / ViT / BEV, YOLO-seg,
Ultralytics YOLO-world, or anything that is not that detect/E2E contract.

The nerd MODEL tab cycles *available* ids only. Missing files do not appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
YOLO_STEMS = ("yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x")


@dataclass(frozen=True)
class ModelChoice:
    id: str
    role: str  # detector | e2e
    label: str
    kind: str
    path: Path | None = None
    hint: str = ""


def _e2e_yaml_path() -> Path:
    cfg = ROOT / "config" / "control.yaml"
    fallback = MODELS_DIR / "e2e_current.onnx"
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        rel = str(((data.get("e2e") or {}).get("model")) or "").strip()
        if rel:
            p = Path(rel)
            return p if p.is_absolute() else (ROOT / p)
    except Exception:
        pass
    return fallback


def _yolo_files() -> list[tuple[str, Path, str]]:
    """Return (id, path, kind) for YOLOv8 weight files that exist."""
    if not MODELS_DIR.is_dir():
        return []
    found: dict[str, tuple[str, Path, str]] = {}
    stems = list(YOLO_STEMS)
    extra = sorted(
        p.stem
        for p in list(MODELS_DIR.glob("yolov8*.onnx")) + list(MODELS_DIR.glob("yolov8*.pt"))
        if p.stem not in stems
    )
    for stem in [*stems, *extra]:
        onnx = MODELS_DIR / f"{stem}.onnx"
        if onnx.is_file():
            found[f"{stem}-onnx"] = (f"{stem}-onnx", onnx, "yolo-onnx")
        pt = MODELS_DIR / f"{stem}.pt"
        if pt.is_file():
            found[f"{stem}-ultra"] = (f"{stem}-ultra", pt, "yolo-pt")
    return [found[k] for k in sorted(found)]


def _e2e_files() -> list[tuple[str, Path]]:
    if not MODELS_DIR.is_dir():
        files: list[Path] = []
    else:
        files = sorted(p for p in MODELS_DIR.glob("e2e*.onnx") if p.is_file())
    yaml_p = _e2e_yaml_path()
    out: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    if yaml_p.is_file():
        out.append((yaml_p.stem, yaml_p))
        seen.add(yaml_p.resolve())
    for p in files:
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append((p.stem, p))
    return out


def detector_choices() -> list[ModelChoice]:
    """Ids the MODEL tab can cycle. Always includes auto / empty / synthetic."""
    rows = [
        ModelChoice("auto", "detector", "auto", "auto", hint="onnx then pt then synthetic/empty"),
    ]
    for did, path, kind in _yolo_files():
        rows.append(
            ModelChoice(
                did,
                "detector",
                did,
                kind,
                path=path,
                hint="YOLOv8 detect COCO",
            )
        )
    rows.append(
        ModelChoice("synthetic", "detector", "synthetic", "synthetic", hint="smoke leads, no weights")
    )
    rows.append(ModelChoice("empty", "detector", "empty", "empty", hint="no boxes"))
    return rows


def e2e_choices() -> list[ModelChoice]:
    rows = [
        ModelChoice("auto", "e2e", "auto", "auto", hint="ONNX from control.yaml if present, else stub"),
        ModelChoice("stub", "e2e", "stub", "e2e-stub", hint="numpy PilotNet-scale MLP"),
    ]
    for eid, path in _e2e_files():
        rows.append(
            ModelChoice(
                eid,
                "e2e",
                eid,
                "e2e-onnx",
                path=path,
                hint="PilotNet-scale ONNX",
            )
        )
    return rows


def choices_for(role: str) -> list[ModelChoice]:
    if role == "e2e":
        return e2e_choices()
    return detector_choices()


def ids_for(role: str) -> list[str]:
    return [c.id for c in choices_for(role)]


def cycle_id(role: str, current: str, direction: int = 1) -> str:
    ids = ids_for(role)
    if not ids:
        return current or "auto"
    cur = str(current or ids[0])
    i = ids.index(cur) if cur in ids else 0
    step = 1 if direction >= 0 else -1
    return ids[(i + step) % len(ids)]


def load_detector(spec: str, *, allow_synthetic: bool = True) -> tuple[Any, list[str]]:
    """Load a detector by catalog id. Raises ValueError if the id is unknown or the file fails."""
    from python.perception.detect import (
        EmptyDetector,
        OnnxYoloDetector,
        SyntheticDetector,
        UltraYoloDetector,
        make_detector,
    )

    spec = (spec or "auto").strip()
    if spec in ("auto", ""):
        return make_detector(allow_synthetic=allow_synthetic)
    by_id = {c.id: c for c in detector_choices()}
    choice = by_id.get(spec)
    if choice is None:
        raise ValueError(f"unknown detector {spec!r}")
    if choice.kind == "synthetic":
        return SyntheticDetector(), ["yolo_weights"]
    if choice.kind == "empty":
        return EmptyDetector(), ["yolo_weights"]
    if choice.kind == "yolo-onnx":
        assert choice.path is not None
        return OnnxYoloDetector(choice.path), []
    if choice.kind == "yolo-pt":
        assert choice.path is not None
        return UltraYoloDetector(choice.path), []
    raise ValueError(f"unhandled detector kind {choice.kind}")


def load_e2e(spec: str) -> Any:
    from python.control.e2e import E2EPolicy, make_e2e

    spec = (spec or "auto").strip()
    if spec in ("auto", ""):
        return make_e2e(model_path=_e2e_yaml_path())
    if spec == "stub":
        return E2EPolicy(model_path=MODELS_DIR / "__gvd_no_e2e__.onnx")
    by_id = {c.id: c for c in e2e_choices()}
    choice = by_id.get(spec)
    if choice is None or choice.path is None:
        raise ValueError(f"unknown e2e {spec!r}")
    return make_e2e(model_path=choice.path)


@dataclass
class ModelRuntime:
    """Hot-swap detector / E2E when the nerd MODEL tab (or CLI) changes ids."""

    detector_id: str = "auto"
    e2e_id: str = "auto"
    last_note: str = ""

    def sync(self, opts: Any, perc: Any, e2e_policy: Any, *, allow_synthetic: bool = True) -> tuple[Any, Any, list[str]]:
        notes: list[str] = []
        want_d = str(getattr(opts, "detector_id", None) or "auto")
        if want_d != self.detector_id:
            try:
                allow = True if want_d == "synthetic" else allow_synthetic
                det, missing = load_detector(want_d, allow_synthetic=allow)
            except Exception as e:
                opts.detector_id = self.detector_id
                notes.append(f"detector {want_d} failed ({type(e).__name__}: {e}); kept {perc.detector.name}")
            else:
                perc.set_detector(det, missing)
                self.detector_id = want_d
                notes.append(f"detector -> {det.name}")
        want_e = str(getattr(opts, "e2e_id", None) or "auto")
        if want_e != self.e2e_id:
            try:
                pol = load_e2e(want_e)
            except Exception as e:
                opts.e2e_id = self.e2e_id
                notes.append(f"e2e {want_e} failed ({type(e).__name__}: {e}); kept {e2e_policy.name}")
            else:
                e2e_policy = pol
                self.e2e_id = want_e
                notes.append(f"e2e -> {pol.name}")
        if notes:
            self.last_note = notes[-1]
        return perc, e2e_policy, notes
