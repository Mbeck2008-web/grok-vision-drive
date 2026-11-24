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
    queue_companion_traffic,
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




def check_empty_slots_still_delete_traffic() -> None:
    """A respawn that finds no new slots must still delete leftover gvd_traffic_* cars.

    The 30 min bughunt left companion cars at the previous spawn (km away)
    because the collector called traffic_lua(pick, len(slots), slots) and
    skipped the Lua entirely when slots was empty.
    """
    deck = SpawnDeck.load(FIXTURE)
    picked = deck.take()
    assert picked is not None
    none = traffic_lua(picked, 0, [])
    assert none, "count=0 must still emit the delete script"
    script0 = " ".join(none)
    assert "gvd_traffic" in script0 and "v:delete()" in script0
    assert "spawnNewVehicle" not in script0
    script2 = " ".join(traffic_lua(picked, 2, []))
    assert "v:delete()" in script2 and "spawnNewVehicle" not in script2
    sent: list[str] = []
    n = queue_companion_traffic(sent.append, picked, 2, [])
    assert n == 1 and sent and "v:delete()" in sent[0]
    assert "spawnNewVehicle" not in sent[0]
    placed: list[str] = []
    n = queue_companion_traffic(placed.append, picked, 2, None)
    assert n == 1 and "spawnNewVehicle" in placed[0] and "gvd_traffic_0" in placed[0]
    assert queue_companion_traffic(None, picked, 2, []) == 0
    print("empty traffic slots still delete leftover cars")

def main() -> None:
    check_mix_and_traffic()
    check_empty_slots_still_delete_traffic()
    print("spawn mix ok")


if __name__ == "__main__":
    main()
