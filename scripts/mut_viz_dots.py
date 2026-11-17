#!/usr/bin/env python3
"""Apply the Skeptic #235 mutants to this tree, one at a time, and report
KILLED / SURVIVED / NOT-APPLIED. Restores sources after each mutant.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
R = "cpp/viz/src/renderer.cpp"
D = "cpp/viz/src/dots.hpp"
S = "cpp/viz/src/scene.cpp"
P = "python/viz/cpp_publisher.py"
T = "cpp/viz/tests/test_render.cpp"

# Mutants listed in the #235 REWORK (plus the original G-set, now expected
# NOT-APPLIED because flags were removed from production).
M = [
    ("G1 kResamplePolylines=true", R,
     "constexpr bool kResamplePolylines = false;",
     "constexpr bool kResamplePolylines = true;"),
    ("G2 kDotAlphaFollowsProb=false", R,
     "constexpr bool kDotAlphaFollowsProb = true;",
     "constexpr bool kDotAlphaFollowsProb = false;"),
    ("G3 kDotFade=false", R,
     "constexpr bool kDotFade = true; ",
     "constexpr bool kDotFade = false; "),
    ("G4 kFixedBoxSize=true", R,
     "constexpr bool kFixedBoxSize = false;",
     "constexpr bool kFixedBoxSize = true;"),
    ("K2 fade 200->1000 ms", D,
     "constexpr float kDotFadeSec = 0.20f;",
     "constexpr float kDotFadeSec = 1.0f;"),
    ("K3 fade 200->100 ms (out of 150-250)", D,
     "constexpr float kDotFadeSec = 0.20f;",
     "constexpr float kDotFadeSec = 0.10f;"),
    ("K4 fade 200->400 ms (out of 150-250)", D,
     "constexpr float kDotFadeSec = 0.20f;",
     "constexpr float kDotFadeSec = 0.40f;"),
    ("K15 fade-out removed (disappear pops)", D,
     "      L.fade = std::max(0.f, L.fade - fade_step);\n      L.alpha = L.fade * L.target;\n      if (L.fade > kDotAliveEps) next.push_back(L);",
     "      continue;"),
    ("K19 alpha=sqrt(prob)", D,
     "  float a = std::clamp(d.prob, 0.f, 1.f);",
     "  float a = std::sqrt(std::clamp(d.prob, 0.f, 1.f));"),
    ("K10 index-order matching", D,
     "    if (L.id >= 0) {\n      for (size_t i = 0; i < incoming.size(); ++i) {\n        if (used[i]) continue;\n        if (incoming[i].id == L.id) return static_cast<int>(i);\n      }\n    }",
     "    if (true) {\n      for (size_t i = 0; i < incoming.size(); ++i) {\n        if (used[i]) continue;\n        return static_cast<int>(i);\n      }\n    }"),
    ("K16 NN match radius 0.75->0", D,
     "constexpr float kDotMatchRadius = 0.75f;",
     "constexpr float kDotMatchRadius = 0.0f;"),
    ("K17 C++ kMaxDots cap removed", S,
     "      if (s.dots.size() >= static_cast<size_t>(kMaxDots)) break;\n",
     ""),
    ("K14 publisher prob clamp removed", P,
     "    pr = min(1.0, max(0.0, pr))\n",
     ""),
]

def run(cmd, cwd):
    return subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1500
    )


def main() -> int:
    only = sys.argv[1:]
    env = dict(os.environ)
    rows = []
    for mid, rel, old, new in M:
        if only and not any(mid.startswith(o + " ") or mid.startswith(o) for o in only):
            continue
        path = W / rel
        src = path.read_text()
        n = src.count(old)
        if n != 1:
            print((mid, f"NOT-APPLIED count={n}"), flush=True)
            rows.append((mid, "NOT-APPLIED"))
            continue
        path.write_text(src.replace(old, new))
        killed = []
        try:
            if rel.startswith("cpp"):
                b = run(["cmake", "--build", "--preset", "linux-release"], W / "cpp/viz")
                if b.returncode:
                    killed.append("build")
                else:
                    c = run(["ctest", "--preset", "linux-release", "--output-on-failure"], W / "cpp/viz")
                    if c.returncode:
                        fails = [l for l in c.stdout.splitlines() if l.strip().startswith("FAIL") or "FAILED" in l]
                        killed.append("ctest")
                        if fails[:2]:
                            killed[-1] += " (" + "; ".join(fails[:2]) + ")"
            if "publisher" in rel or mid.startswith("K14"):
                c = run([sys.executable, "scripts/test_viz_cpp_publisher.py"], W)
                if c.returncode:
                    killed.append("publisher")
        finally:
            path.write_text(src)
        status = "KILLED by " + ", ".join(killed) if killed else "SURVIVED"
        print((mid, status), flush=True)
        rows.append((mid, status))
    # restore the production build
    run(["cmake", "--build", "--preset", "linux-release"], W / "cpp/viz")
    print("----")
    for mid, st in rows:
        print(f"{st:10}  {mid}")
    survived = [m for m, s in rows if s == "SURVIVED"]
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
