"""Per-epoch checkpoints: the exact weights behind one holdout val row.

The trainer keeps only the best weights in memory (``best_blob``) and exports
them at the end. A run that is stopped early leaves nothing to score, and the
exported file cannot be tied to one val row. At each epoch's val the trainer
now writes the weights that produced that row, with the step and epoch in the
file name and inside the file:

    <dir>/<strips>.run<UTC start>.ckpt_ep{N}_step{S}.npz

The directory is ``--ckpt-dir`` or, by default, the folder that holds the
strip folder (beside the pause file, never inside the recording).

Writes are create-only: an existing file is never replaced. Before writing,
the file is serialised in memory and its size is checked against the free
space on the target volume; a write that would leave less than the floor
(``--ckpt-min-free-gb``, default 8) is refused. A refusal is logged and the
run keeps training.

``export_epoch_onnx`` builds the ONNX from one of these files, never from
``best_blob``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from python.data.binary_io import link_or_copy_new

FORMAT = "gvd-epoch-ckpt-1"
DEFAULT_MIN_FREE_GB = 8.0
GB = 1_000_000_000
_NAME_RE = re.compile(r"^(?P<strips>.+)\.run(?P<run>\d{8}T\d{6}Z)\.ckpt_ep(?P<epoch>\d+)_step(?P<step>\d+)\.npz$")


class EpochCkptRefused(RuntimeError):
    """The checkpoint was not written (file exists, low space, or a write error)."""


def run_id(started: float | None = None) -> str:
    """UTC start time of the run, second resolution: ``20261021T040702Z``."""
    stamp = time.time() if started is None else float(started)
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(stamp))


def strips_name(strip_root: Path | str) -> str:
    name = Path(strip_root).name
    return name if name not in ("", ".", "..") else "gvd-train"


def default_ckpt_dir(strip_root: Path | str) -> Path:
    """The folder that holds the strip folder (where the pause file lives)."""
    return Path(strip_root).parent


def epoch_ckpt_path(ckpt_dir: Path | str, strip_root: Path | str, run: str, epoch: int, step: int) -> Path:
    return Path(ckpt_dir) / f"{strips_name(strip_root)}.run{run}.ckpt_ep{int(epoch)}_step{int(step)}.npz"


def parse_ckpt_name(path: Path | str) -> dict[str, Any] | None:
    """``{"strips", "run", "epoch", "step"}`` from a checkpoint file name, or None."""
    match = _NAME_RE.match(Path(path).name)
    if match is None:
        return None
    return {
        "strips": match.group("strips"),
        "run": match.group("run"),
        "epoch": int(match.group("epoch")),
        "step": int(match.group("step")),
    }


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def serialise(weights: Mapping[str, Any], meta: Mapping[str, Any]) -> bytes:
    """The exact bytes of the ``.npz``: ``meta`` (JSON) plus one array per weight."""
    arrays: dict[str, np.ndarray] = {}
    names: list[str] = []
    for index, (key, value) in enumerate(weights.items()):
        arrays[f"w{index}"] = np.array(value, copy=True)
        names.append(str(key))
    body = dict(meta)
    body["format"] = FORMAT
    body["weight_names"] = names
    buffer = io.BytesIO()
    np.savez(buffer, meta=np.array(json.dumps(body, sort_keys=True)), **arrays)
    return buffer.getvalue()


def write_epoch_ckpt(
    path: Path | str,
    weights: Mapping[str, Any],
    meta: Mapping[str, Any],
    *,
    min_free_bytes: int = int(DEFAULT_MIN_FREE_GB * GB),
    strip_root: Path | str | None = None,
    disk_free: Callable[[Path], int] | None = None,
) -> dict[str, Any]:
    """Write one checkpoint create-only. Returns ``{"path", "sha256", "bytes", "free_after"}``.

    Raises ``EpochCkptRefused`` when the file exists, when the write would
    leave less than ``min_free_bytes`` free on the target volume, when the
    target is inside the strip folder, or on an OS error. Nothing is replaced.
    """
    path = Path(path)
    if not weights:
        raise EpochCkptRefused("no weights")
    if strip_root is not None and _inside(path, Path(strip_root)):
        raise EpochCkptRefused(f"{path} is inside the strip folder")
    if path.exists():
        raise EpochCkptRefused(f"{path} already exists (create-only)")
    blob = serialise(weights, meta)
    size = len(blob)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = disk_free or (lambda p: int(shutil.disk_usage(p).free))
        free = int(probe(path.parent))
    except OSError as exc:
        raise EpochCkptRefused(f"cannot check free space at {path.parent}: {exc}") from exc
    if free - size < int(min_free_bytes):
        raise EpochCkptRefused(
            f"low space: {path.parent} has {free / GB:.2f} GB free; this {size / 1e6:.1f} MB checkpoint "
            f"would leave {(free - size) / GB:.2f} GB < floor {int(min_free_bytes) / GB:.2f} GB"
        )
    try:
        handle = open(path, "xb")
    except FileExistsError as exc:
        raise EpochCkptRefused(f"{path} already exists (create-only)") from exc
    except OSError as exc:
        raise EpochCkptRefused(f"cannot create {path}: {exc}") from exc
    try:
        with handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        try:
            path.unlink()
        except OSError:
            pass
        raise EpochCkptRefused(f"write failed for {path}: {exc}") from exc
    return {
        "path": str(path),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": size,
        "free_after": free - size,
    }


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_epoch_ckpt(path: Path | str) -> dict[str, Any]:
    """``{"meta": {...}, "weights": {name: ndarray}}``. Checks the embedded step/epoch against the name."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as blob:
        meta = json.loads(str(blob["meta"].item() if blob["meta"].shape == () else blob["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path.name}: not a {FORMAT} file")
        names = list(meta.get("weight_names") or [])
        weights = {name: np.array(blob[f"w{index}"], copy=True) for index, name in enumerate(names)}
    parsed = parse_ckpt_name(path)
    if parsed is not None and (parsed["epoch"] != int(meta.get("epoch", -1)) or parsed["step"] != int(meta.get("step", -1))):
        raise ValueError(
            f"{path.name}: name says epoch {parsed['epoch']} step {parsed['step']}, "
            f"file says epoch {meta.get('epoch')} step {meta.get('step')}"
        )
    return {"meta": meta, "weights": weights}


def find_epoch_ckpt(ckpt_dir: Path | str, epoch: int, run: str | None = None) -> Path:
    """The one checkpoint for ``epoch`` in ``ckpt_dir`` (optionally of one ``run``). Ambiguity raises."""
    hits = []
    for item in sorted(Path(ckpt_dir).glob("*.ckpt_ep*_step*.npz")):
        parsed = parse_ckpt_name(item)
        if parsed is None or parsed["epoch"] != int(epoch):
            continue
        if run is not None and parsed["run"] != run:
            continue
        hits.append(item)
    if not hits:
        raise FileNotFoundError(f"no epoch {int(epoch)} checkpoint in {ckpt_dir}" + (f" for run {run}" if run else ""))
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} epoch {int(epoch)} checkpoints in {ckpt_dir}; pass --run: " + ", ".join(h.name for h in hits))
    return hits[0]


def build_net(meta: Mapping[str, Any]) -> Any:
    """A CPU net of the checkpoint's architecture (cams, scale, backbone, cam_age).

    A checkpoint written before ``cam_age`` existed has no key: no ages.
    The same holds for ``ctl``: no key, no ctl head.
    """
    cams = tuple(meta.get("cams") or ()) or None
    scale = float(meta.get("scale") or 1.0)
    backbone = str(meta.get("backbone") or "scratch")
    if backbone in ("", "scratch"):
        from python.train.scene_net import SceneNet

        return SceneNet(cams=cams, scale=scale, cam_age=bool(meta.get("cam_age")), ctl=bool(meta.get("ctl")))
    if meta.get("cam_age") or meta.get("ctl"):
        raise ValueError("a --cam-age or ctl checkpoint is a scratch-net checkpoint; this one names a backbone")
    from python.train.pretrained_net import build_scene_net

    return build_scene_net(cams, scale, backbone, freeze=bool(meta.get("freeze_backbone")))


def load_into_net(net: Any, weights: Mapping[str, Any]) -> None:
    """Copy every weight; a missing or extra key raises (no silent partial load)."""
    import torch

    current = net.state_dict()
    extra = [key for key in weights if key not in current]
    missing = [key for key in current if key not in weights]
    if extra or missing:
        raise KeyError(f"checkpoint does not match the net: extra={extra[:3]} missing={missing[:3]}")
    mapped = {
        key: torch.tensor(np.array(value, copy=True), dtype=current[key].dtype, device=current[key].device)
        for key, value in weights.items()
    }
    net.load_state_dict(mapped)


def export_epoch_onnx(ckpt: Path | str, out: Path | str, *, trace: Any = None) -> dict[str, Any]:
    """Export the ONNX of one epoch checkpoint. ``out`` is create-only."""
    from python.train.cam_strip import stamp_onnx_metadata
    from python.train.scene_net import export_scene_onnx

    ckpt = Path(ckpt)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists (create-only)")
    loaded = load_epoch_ckpt(ckpt)
    meta = loaded["meta"]
    net = build_net(meta)
    load_into_net(net, loaded["weights"])
    net = net.cpu().eval()
    heads = tuple(meta.get("heads") or ()) or None
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f"{out.name}.partial-{os.getpid()}.onnx")
    try:
        export_scene_onnx(partial, net, heads=heads, trace=trace)
        ckpt_sha = sha256_file(ckpt)
        stamp_onnx_metadata(
            str(partial),
            {
                "gvd_ckpt_file": ckpt.name,
                "gvd_ckpt_sha256": ckpt_sha,
                "gvd_ckpt_epoch": str(int(meta["epoch"])),
                "gvd_ckpt_step": str(int(meta["step"])),
                "gvd_ckpt_run": str(meta.get("run") or ""),
                "gvd_scale": str(float(meta.get("scale") or 1.0)),
            },
        )
        link_or_copy_new(partial, out)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "onnx": str(out),
        "onnx_sha256": sha256_file(out),
        "ckpt": str(ckpt),
        "ckpt_sha256": ckpt_sha,
        "epoch": int(meta["epoch"]),
        "step": int(meta["step"]),
        "val": meta.get("val"),
    }
