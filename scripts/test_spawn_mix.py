#!/usr/bin/env python3
"""West-coast spawns from the dumped navgraph, mixed by road type."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.collect.spawn_mix import (  # noqa: E402
    SPEED,
    SpawnDeck,
    traffic_count,
    traffic_lua,
)

FIXTURE = ROOT / "tests" / "fixtures" / "roads" / "west_coast_usa.json.gz"


def check_mix_and_traffic() -> None:
    deck = SpawnDeck.load(FIXTURE)
    picks = []
    for _ in range(80):
        picked = deck.take()
        assert picked is not None
        picks.append(picked)
    assert len({item.road_id for item in picks}) == 80
    counts = Counter(item.kind for item in picks)
    assert set(counts) == {"city", "highway", "ramp"}
    assert abs(counts["city"] / 80 - 0.40) <= 0.08
    assert abs(counts["highway"] / 80 - 0.35) <= 0.08
    assert abs(counts["ramp"] / 80 - 0.25) <= 0.08
    for item in picks:
        assert item.speed == SPEED[item.kind]
        assert item.pos[2] > 0.0
        assert abs(sum(item.rot[2:])) > 0.0 or abs(item.rot[3] - 1.0) < 1e-6
    assert traffic_count(None, "west_coast_usa") == 2
    assert traffic_count(None, "italy") == 0
    assert traffic_count(0, "west_coast_usa") == 0
    assert traffic_count(2, "italy") == 2
    script = "\n".join(traffic_lua(picks[0], 2))
    assert "gvd_traffic_0" in script and "gvd_traffic_1" in script
    assert "spawnNewVehicle" in script and "ai.setSpeed" in script
    assert "getPlayerVehicle" in script
    print(
        "spawn mix",
        {kind: counts[kind] for kind in ("city", "highway", "ramp")},
    )


def main() -> None:
    check_mix_and_traffic()
    print("spawn mix ok")


if __name__ == "__main__":
    main()
