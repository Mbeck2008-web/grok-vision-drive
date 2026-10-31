#!/usr/bin/env python3
"""Top-down overlay of a junction map, optionally against a saved navgraph.

    py -3 scripts/render_junction_overlay.py junctions/west_coast_usa.json.gz --out logs/jct_wcu.png
    py -3 scripts/render_junction_overlay.py junctions/west_coast_usa.json.gz --out logs/jct_wcu_nav.png \\
        --navgraph logs/navgraph_wcu.json.gz

Roads (the map's ``graph`` edges) are grey. Each junction is its ``r_m``
circle, coloured by ``type`` (T orange, Y magenta, 4way blue, multi red,
merge cyan); a minor junction is drawn thin. ``--navgraph`` adds the
navgraph's candidate nodes (major green, minor yellow, unmapped white crosses)
by ``junction_map.classify_navgraph_nodes`` against ``--roads`` (default
``tests/fixtures/roads/<map>.json.gz``) and prints metric 1(a) as JSON with a
``verdict`` (``PASS``, ``FAIL`` or ``VOID``) from the frozen golden values
(``--golden``, default ``tests/fixtures/junctions/<map>.golden.json``): VOID
when the unmapped share is above ``UNMAPPED_MAX``, else PASS when major recall
and precision meet the golden bars.
``--crop x0,y0,x1,y1`` (world metres) renders one area. The PNG is not
overwritten unless ``--force``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from python.collect import junction_map as jm  # noqa: E402

TYPE_BGR = {"T": (0, 140, 255), "Y": (255, 0, 255), "4way": (255, 120, 0), "multi": (0, 0, 255), "merge": (255, 255, 0)}
NAV_BGR = {"major": (0, 220, 0), "minor": (0, 220, 220), "unmapped": (255, 255, 255)}


def _load_json(path: Path) -> dict:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def render(doc: dict, *, px_per_m: float = 1.0, crop: tuple[float, float, float, float] | None = None,
           navgraph: dict | None = None, max_px: int = 6000) -> np.ndarray:
    nodes = doc["graph"]["nodes"]
    pts = [p for e in doc["graph"]["edges"] for p in e["xy"]] or [[0.0, 0.0]]
    if crop is None:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        crop = (min(xs) - 20, min(ys) - 20, max(xs) + 20, max(ys) + 20)
    x0, y0, x1, y1 = crop
    scale = min(px_per_m, max_px / max(x1 - x0, y1 - y0, 1.0))
    w, h = int(math.ceil((x1 - x0) * scale)), int(math.ceil((y1 - y0) * scale))
    img = np.zeros((h, w, 3), np.uint8)

    def px(p) -> tuple[int, int]:  # world +y is up in the image
        return int(round((p[0] - x0) * scale)), int(round((y1 - p[1]) * scale))

    for e in doc["graph"]["edges"]:
        line = np.array([px(p) for p in e["xy"]], np.int32)
        cv2.polylines(img, [line], False, (110, 110, 110), 1, cv2.LINE_AA)
    for p in nodes:
        cv2.circle(img, px(p), 1, (160, 160, 160), -1)
    for j in doc["junctions"]:
        c = px(j["c"])
        r = max(2, int(round(j["r_m"] * scale)))
        colour = TYPE_BGR.get(j["type"], (200, 200, 200))
        cv2.circle(img, c, r, colour, 1 if j["minor"] else 2, cv2.LINE_AA)
        if not j["minor"]:
            for arm in j["arms"]:
                a = float(arm["heading_rad"])
                tip = (c[0] + int(round(math.cos(a) * (r + 6))), c[1] - int(round(math.sin(a) * (r + 6))))
                cv2.line(img, c, tip, colour, 1, cv2.LINE_AA)
    if navgraph is not None:
        for kind in ("minor", "unmapped", "major"):
            for n in navgraph[kind]:
                cv2.drawMarker(img, px(n["pos"]), NAV_BGR[kind], cv2.MARKER_CROSS, 7, 1)
    y = 18
    for name, colour in TYPE_BGR.items():
        cv2.putText(img, name, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        y += 18
    return img


def navgraph_report(doc: dict, nav: dict, network: dict, golden: dict) -> tuple[dict, dict]:
    """The printed metric-1(a) report (metric, verdict, unmapped positions) and the node classes."""
    classified = jm.classify_navgraph_nodes(nav, network)
    metric = jm.navgraph_metric(doc, classified, match_m=float(golden["params"]["MATCH_M"]))
    report = {"metric": metric, "verdict": jm.navgraph_verdict(metric, golden),
              "unmapped_pos": [n["pos"] for n in classified["unmapped"]]}
    return report, classified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("map", type=Path, help="junctions/<map>.json.gz")
    parser.add_argument("--out", type=Path, required=True, help="PNG path.")
    parser.add_argument("--navgraph", type=Path, default=None, help="Saved navgraph (dump_navgraph.py output).")
    parser.add_argument("--roads", type=Path, default=None, help="Road dump for --navgraph arm mapping.")
    parser.add_argument("--golden", type=Path, default=None, help="Golden file for the --navgraph verdict.")
    parser.add_argument("--px-per-m", type=float, default=1.0)
    parser.add_argument("--crop", default=None, help="x0,y0,x1,y1 in world metres.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing PNG.")
    args = parser.parse_args(argv)
    doc = jm.load_map(args.map)
    crop = None if not args.crop else tuple(float(v) for v in args.crop.split(","))
    classified = None
    if args.navgraph is not None:
        roads = args.roads or ROOT / "tests" / "fixtures" / "roads" / f"{doc['map']}.json.gz"
        golden_path = args.golden or ROOT / "tests" / "fixtures" / "junctions" / f"{doc['map']}.golden.json"
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        report, classified = navgraph_report(doc, _load_json(args.navgraph), _load_json(roads)["road_network"], golden)
        print(json.dumps(report, sort_keys=True), flush=True)
    if args.out.exists() and not args.force:
        print(f"overlay {args.out.name} exists; not written (use --force)", flush=True)
        return 1
    img = render(doc, px_per_m=args.px_per_m, crop=crop, navgraph=classified)  # type: ignore[arg-type]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        print("overlay encode failed", flush=True)
        return 1
    args.out.write_bytes(buf.tobytes())
    s = jm.summary(doc)
    print(f"overlay {args.out.name} {img.shape[1]}x{img.shape[0]} junctions={s['junctions']} "
          f"major={s['major']} minor={s['minor']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
