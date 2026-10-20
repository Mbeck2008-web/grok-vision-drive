"""Train on several recordings: one combined folder of links, never re-encoded.

``combine_recordings`` builds a folder the trainer reads like any recording.
Every camera file is a hard link to the source file (a byte copy when the
two folders are on different volumes); no frame is decoded or re-encoded.
``state.jsonl`` / ``gt.jsonl`` are rewritten with new contiguous ``i`` and
each row carries ``recording`` (the source folder name), ``src_i`` (its
index there) and ``session`` (the recording, so the GRU resets between
recordings). ``segment`` is prefixed ``<recording>/`` so segment holdouts,
event windows and stall spans never join two recordings. ``impacts.jsonl``
is merged the same way.

A per-recording row cut (``usable_until``) keeps rows with ``i <= max_i``.
It comes from ``--usable-until NAME=ROW[:REASON]`` or a ``usable_until.json``
sidecar (``{"max_i": N, "reason": "..."}``) in the recording, the flag
winning. ``combined.json`` records every recording, its cut and reason, and
whether the collector finished it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

USABLE_UNTIL = "usable_until.json"
COMBINED = "combined.json"
COMBINED_VERSION = 1
SIDECARS = ("state.jsonl", "gt.jsonl", "impacts.jsonl")


class CombineError(ValueError):
    """The recordings cannot be combined as asked (nothing was overwritten)."""


def recording_id(path: Path | str) -> str:
    return Path(path).name


def parse_usable_until(values: Iterable[str] | None) -> dict[str, dict[str, Any]]:
    """``["rec5b=33661:stopped early"]`` -> ``{"rec5b": {"max_i": 33661, "reason": ...}}``."""
    out: dict[str, dict[str, Any]] = {}
    for text in values or ():
        name, sep, rest = str(text).partition("=")
        row, _colon, reason = rest.partition(":")
        if not sep or not name.strip():
            raise CombineError(f"--usable-until {text!r}: expected NAME=ROW[:REASON]")
        try:
            max_i = int(row.strip())
        except ValueError as exc:
            raise CombineError(f"--usable-until {text!r}: ROW must be an integer") from exc
        out[name.strip()] = {"max_i": max_i, "reason": reason.strip() or "cut by --usable-until", "source": "flag"}
    return out


def read_usable_until(root: Path | str) -> dict[str, Any] | None:
    path = Path(root) / USABLE_UNTIL
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("max_i"), int):
        raise CombineError(f"{path}: expected {{\"max_i\": int, \"reason\": str}}")
    return {"max_i": int(payload["max_i"]), "reason": str(payload.get("reason") or "usable_until.json"), "source": USABLE_UNTIL}


def resolve_cut(root: Path | str, flags: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any] | None:
    """The row cut for one recording: the flag for its name, else its sidecar, else None."""
    flagged = (flags or {}).get(recording_id(root))
    return dict(flagged) if flagged else read_usable_until(root)


def apply_cut(rows: list[Mapping[str, Any]], cut: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not cut:
        return list(rows)
    limit = int(cut["max_i"])
    kept = []
    for row in rows:
        try:
            if int(row.get("i")) <= limit:
                kept.append(row)
        except (TypeError, ValueError):
            continue
    return kept


def read_jsonl_tolerant(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Rows, and the count of lines that did not parse (a live file's half-written tail)."""
    if not path.is_file():
        return [], 0
    rows: list[dict[str, Any]] = []
    bad = 0
    with open(path, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                bad += 1
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows, bad


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finished(root: Path) -> dict[str, Any]:
    """What ``collect.json`` says about the run (``finished`` None when it does not say)."""
    path = root / "collect.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"finished": None}
    out: dict[str, Any] = {"collect_state": payload.get("state")}
    done, target = payload.get("done_s"), payload.get("target_s")
    if isinstance(done, (int, float)) and isinstance(target, (int, float)) and target > 0:
        out["done_s"], out["target_s"] = float(done), float(target)
        out["finished"] = bool(done >= target - 1.0)
    else:
        out["finished"] = None
    return out


def _link(src: Path, dst: Path, counts: dict[str, int]) -> None:
    """Hard-link ``src`` at ``dst``, else byte-copy it into a new file.

    ``dst`` must not exist: writing into an existing hard link would write
    into the source recording's frame, so that is refused, never overwritten.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dst):
        raise CombineError(f"{dst} already exists; refusing to write over a possibly linked frame")
    try:
        os.link(src, dst)
        counts["hardlink"] += 1
        return
    except FileExistsError as exc:
        raise CombineError(f"{dst} already exists; refusing to write over a possibly linked frame") from exc
    except OSError:
        pass
    # Bytes as they are, into a new file only ("xb"): never decoded or re-encoded.
    with open(src, "rb") as reader, open(dst, "xb") as writer:
        shutil.copyfileobj(reader, writer)
    counts["copy"] += 1


def _cams_payload(root: Path) -> Any:
    path = root / "cameras.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("cameras")


def describe_recording(item: Mapping[str, Any]) -> str:
    """``rec5b: cut at row 33661 (stopped early ...), not finished, 33662 of 33662 rows``."""
    cut = item.get("cut")
    fin = item.get("finished")
    state = "finished" if fin is True else ("not finished" if fin is False else "finish unknown")
    where = f"cut at row {cut['max_i']} ({cut['reason']})" if cut else "no cut"
    return f"{item['id']}: {where}, {state}, {item['rows_kept']} of {item['rows_source']} rows"


def combine_recordings(
    sources: Iterable[Path | str],
    dest: Path | str,
    *,
    usable_until: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Link ``sources`` into ``dest`` (a new or empty folder) and return the manifest.

    An existing ``dest`` built from the same sources, cuts and state files is
    reused as is; anything else in ``dest`` is refused, never overwritten.
    """
    roots = [Path(src) for src in sources]
    ids = [recording_id(root) for root in roots]
    if len(set(ids)) != len(ids):
        raise CombineError(f"recording folder names must differ (they are the recording ids): {ids}")
    unknown = sorted(set(usable_until or {}) - set(ids))
    if unknown:
        raise CombineError(f"--usable-until names {unknown} match no recording ({','.join(ids)})")
    plan = []
    cams_ref = None
    for root, rid in zip(roots, ids):
        if not (root / "raw").is_dir() or not (root / "state.jsonl").is_file():
            raise CombineError(f"{root}: not a raw recording (needs state.jsonl and raw/)")
        cams = _cams_payload(root)
        if cams_ref is None:
            cams_ref = cams
        elif cams is not None and cams != cams_ref:
            raise CombineError(f"{root}: cameras.json differs from {roots[0]}; recordings must share one rig")
        plan.append({"id": rid, "source": str(root), "state_sha256": _sha256(root / "state.jsonl"),
                     "cut": resolve_cut(root, usable_until)})
    out = Path(dest)
    key = [{k: item[k] for k in ("id", "source", "state_sha256", "cut")} for item in plan]
    if (out / COMBINED).is_file():
        old = json.loads((out / COMBINED).read_text(encoding="utf-8"))
        if [{k: r.get(k) for k in ("id", "source", "state_sha256", "cut")} for r in old.get("recordings") or []] == key:
            return old
        raise CombineError(f"{out} holds a different combination; pick a new --combined-out folder")
    if out.exists() and any(out.iterdir()):
        raise CombineError(f"{out} is not empty; pick a new --combined-out folder")
    out.mkdir(parents=True, exist_ok=True)

    counts = {"hardlink": 0, "copy": 0}
    state_out: list[str] = []
    gt_out: list[str] = []
    impacts_out: list[str] = []
    recordings: list[dict[str, Any]] = []
    n = 0
    for root, item in zip(roots, plan):
        rid = item["id"]
        rows, bad_state = read_jsonl_tolerant(root / "state.jsonl")
        labels, bad_gt = read_jsonl_tolerant(root / "gt.jsonl")
        gt_by_i = {}
        for label in labels:
            try:
                gt_by_i[int(label.get("i"))] = label
            except (TypeError, ValueError):
                continue
        kept = apply_cut(rows, item["cut"])
        first = n
        last_t = None
        for row in kept:
            src_i = int(row["i"])
            new = dict(row)
            new.update({"i": n, "src_i": src_i, "recording": rid, "session": rid})
            if new.get("segment") is not None:
                new["segment"] = f"{rid}/{new['segment']}"
            cams = new.get("cams") if isinstance(new.get("cams"), dict) else {}
            for folder in ("raw", "raw_repaired"):
                base = root / folder
                if not base.is_dir():
                    continue
                for cam_dir in base.iterdir():
                    for suffix in (".png", ".npy"):
                        src = cam_dir / f"{src_i:06d}{suffix}"
                        if src.is_file():
                            _link(src, out / folder / cam_dir.name / f"{n:06d}{suffix}", counts)
                            if folder == "raw" and isinstance(cams.get(cam_dir.name), dict):
                                cams[cam_dir.name] = dict(cams[cam_dir.name], file=f"raw/{cam_dir.name}/{n:06d}{suffix}")
            state_out.append(json.dumps(new))
            label = gt_by_i.get(src_i)
            if label is not None:
                gt_out.append(json.dumps(dict(label, i=n, src_i=src_i, recording=rid)))
            last_t = new.get("t")
            n += 1
        impacts, _bad = read_jsonl_tolerant(root / "impacts.jsonl")
        for imp in impacts:
            if last_t is not None and isinstance(imp.get("t"), (int, float)) and imp["t"] > float(last_t):
                continue  # after the cut
            imp = dict(imp, recording=rid)
            if imp.get("segment") is not None:
                imp["segment"] = f"{rid}/{imp['segment']}"
            impacts_out.append(json.dumps(imp))
        recordings.append({
            **item,
            "rows_source": len(rows),
            "rows_kept": len(kept),
            "i_new": [first, n - 1] if kept else None,
            "unparsed_lines": {"state.jsonl": bad_state, "gt.jsonl": bad_gt},
            **_finished(root),
        })
    for name, lines in (("state.jsonl", state_out), ("gt.jsonl", gt_out), ("impacts.jsonl", impacts_out)):
        with open(out / name, "x", encoding="utf-8", newline="\n") as handle:  # new file, never a link
            handle.write("".join(line + "\n" for line in lines))
    first_cams = roots[0] / "cameras.json"
    if first_cams.is_file():
        with open(first_cams, "rb") as reader, open(out / "cameras.json", "xb") as writer:
            shutil.copyfileobj(reader, writer)
    manifest = {"kind": "combined", "version": COMBINED_VERSION, "rows": n, "files": counts, "recordings": recordings}
    (out / COMBINED).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def recordings_metadata(root: Path | str, rows: list[Mapping[str, Any]] | None = None,
                        flags: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Recording list for the run metadata: ``combined.json``'s, or this one folder with its cut."""
    path = Path(root) / COMBINED
    if path.is_file():
        return list(json.loads(path.read_text(encoding="utf-8")).get("recordings") or [])
    root = Path(root)
    return [{"id": recording_id(root), "source": str(root), "cut": resolve_cut(root, flags),
             "rows_kept": len(rows or []), "rows_source": None, **_finished(root)}]
