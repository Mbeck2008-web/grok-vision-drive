"""Where the BeamNG state ``pos`` sits on the car, read from the vehicle's jbeam.

BeamNG's State sensor reports ``pos = obj:getPosition()`` (techCore.lua,
``getVehicleState``), the position of the vehicle's ref node: the first entry
of the jbeam ``refNodes`` table (``"ref:"``). That node is not the body
centre. On the etk800 the ref node is ``f2r`` at jbeam x = -0.33 and its
``"left:"`` partner ``f2l`` is at x = +0.33. jbeam +x is the car's left
(``f2l``, ``f15ll`` and every other ``*l`` node has x > 0), so the ref node is
right of the body centreline by half the f2r-f2l spacing. That number is
always read from the jbeam here; nothing in GVD types it in. Cameras are
mounted about the vehicle origin on the centreline (``config/cameras.yaml`` main x = 0.00), so labels
measured from the ref node put the painted lines that far left of where the
main camera sees them.

Body lateral centre = midpoint of the ref node and its ``left:`` refNode.
That pair is the one BeamNG itself uses for the car's lateral axis, and it
does not move with asymmetric parts (mirrors, tow hooks). The midpoint of the
extreme left/right nodes is computed too, as a check.

Frames here are GVD vehicle frames: x right, y forward.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_NODE = re.compile(r'^\s*\[\s*"([A-Za-z0-9_]+)"\s*,\s*(-?\d+(?:\.\d*)?)\s*,\s*(-?\d+(?:\.\d*)?)\s*,\s*(-?\d+(?:\.\d*)?)')
_STR = re.compile(r'"([^"]*)"')


@dataclass(frozen=True)
class RefOffset:
    ref_node: str
    left_node: str
    ref_right_of_centre_m: float  # > 0: the ref node is right of the body centreline
    extremes_ref_right_of_centre_m: float | None
    source: str


def parse_ref_offset(text: str, source: str = "") -> RefOffset | None:
    """The ref node's lateral offset from the body centre in one jbeam file, or None."""
    lines = text.splitlines()
    nodes: dict[str, float] = {}
    for line in lines:
        m = _NODE.match(line)
        if m:
            nodes.setdefault(m.group(1), float(m.group(2)))
    for k, line in enumerate(lines):
        if '"refNodes"' not in line:
            continue
        rows = []
        for nxt in lines[k + 1 : k + 6]:
            got = _STR.findall(nxt)
            if got:
                rows.append(got)
            if len(rows) == 2:
                break
        if len(rows) < 2 or "ref:" not in rows[0] or "left:" not in rows[0]:
            return None
        head, vals = rows
        ref, left = vals[head.index("ref:")], vals[head.index("left:")]
        if ref not in nodes or left not in nodes:
            return None
        # jbeam +x is left, GVD +x is right.
        ref_x, left_x = -nodes[ref], -nodes[left]
        centre = 0.5 * (ref_x + left_x)
        xs = [-v for v in nodes.values()]
        ext = ref_x - 0.5 * (min(xs) + max(xs)) if xs else None
        return RefOffset(ref, left, round(ref_x - centre, 4), None if ext is None else round(ext, 4), f"{source}:{k + 1}")
    return None


def ref_offset_from_vehicle_zip(zip_path: Path | str, model: str) -> RefOffset:
    """Ref-node offset for ``model`` from its vehicle zip; every body that defines refNodes must agree."""
    found: list[RefOffset] = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            if not name.endswith(".jbeam") or f"vehicles/{model}/" not in name:
                continue
            text = z.read(name).decode("utf-8", "replace")
            if '"refNodes"' in text:
                got = parse_ref_offset(text, name)
                if got is not None:
                    found.append(got)
    return _agree(found, model)


def _agree(found: Iterable[RefOffset], model: str) -> RefOffset:
    found = list(found)
    if not found:
        raise ValueError(f"no refNodes for {model}")
    vals = {f.ref_right_of_centre_m for f in found}
    if len(vals) != 1:
        raise ValueError(f"{model}: bodies disagree on the ref node offset {sorted(vals)}; pick the config")
    return found[0]


def ref_offset_from_jbeam_file(path: Path | str) -> RefOffset:
    """Ref-node offset from one jbeam file (or an excerpt holding its refNodes and those nodes)."""
    got = parse_ref_offset(Path(path).read_text(encoding="utf-8", errors="replace"), Path(path).name)
    if got is None:
        raise ValueError(f"{Path(path).name}: no refNodes with ref:/left: nodes")
    return got


def ref_right_of_centre_m(
    model: str,
    vehicle_zip: Path | str | None = None,
    *,
    jbeam: Path | str | None = None,
) -> tuple[float, str]:
    """``(metres the ref node sits right of the body centreline, source)`` from the jbeam.

    The only source of this offset in GVD. ``vehicle_zip`` is
    ``content/vehicles/<model>.zip``; ``jbeam`` is one jbeam file. There is
    no table and no default: without a jbeam it raises.
    """
    if jbeam is not None:
        got = ref_offset_from_jbeam_file(jbeam)
    elif vehicle_zip is not None and Path(vehicle_zip).is_file():
        got = ref_offset_from_vehicle_zip(vehicle_zip, model)
    else:
        raise ValueError(f"no jbeam for {model}: pass its vehicle zip (content/vehicles/{model}.zip)")
    return got.ref_right_of_centre_m, f"jbeam {got.source} ref={got.ref_node} left={got.left_node}"


def centreline_origin_right_m(model: str, vehicle_zip: Path | str | None = None, *, jbeam: Path | str | None = None) -> float:
    """``origin_right_m`` for ``PaintedLanes.lane_at`` that puts x = 0 on the body centreline."""
    return -ref_right_of_centre_m(model, vehicle_zip, jbeam=jbeam)[0]
