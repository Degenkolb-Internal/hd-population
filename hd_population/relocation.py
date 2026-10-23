"""Relocation of people that have no receiving building in their own tract.

A tract can hold workers, students or guests while the inventory has no building of the
matching occupancy there. Dropping them would remove part of the daytime nonresidential
population, so they are moved to the nearest tracts that do have such buildings: the
search walks out ring by ring through tracts whose polygons touch, stops at the first ring
with a receiving building, and splits the people over all receivers of that ring by floor
area. The indoor/outdoor split of the people is kept. Occupancy classes are not changed.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.strtree import STRtree


class Relocator:
    def __init__(self, tracts, max_rings=3):
        self.max_rings = max_rings
        self.neighbors = self._neighbors(tracts)

    @staticmethod
    def _neighbors(tracts):
        geoids, geoms = [], []
        for gid, w in zip(tracts["GEOID"], tracts["the_geom"]):
            g = wkt.loads(w)
            if g is not None and not g.is_empty:
                geoids.append(str(gid))
                geoms.append(g.buffer(0))
        out = {g: set() for g in geoids}
        if not geoms:
            return out
        left, right = STRtree(geoms).query(geoms, predicate="intersects")
        for i, j in zip(left, right):
            if i != j:
                out[geoids[i]].add(geoids[j])
        return out

    def nearest_ring(self, start, tag, has_tag):
        """(ring, tracts) of the first ring around `start` with a building of `tag`."""
        seen, frontier = {start}, [start]
        for ring in range(1, self.max_rings + 1):
            nxt = []
            for g in frontier:
                for n in self.neighbors.get(g, ()):
                    if n not in seen:
                        seen.add(n)
                        nxt.append(n)
            hits = sorted(g for g in nxt if tag in has_tag.get(g, ()))
            if hits:
                return ring, hits
            if not nxt:
                break
            frontier = nxt
        return None, []

    def relocate(self, inventory, people, unplaced):
        """Add the unplaced people to their receivers in `people` (modified in place).
        Returns a log with one row per tract, pool, receiver and hour."""
        b = inventory.buildings
        tagged = b[b["tract"].notna() & b["tag"].notna()]
        has_tag = tagged.groupby("tract")["tag"].agg(set).to_dict()
        receivers = defaultdict(list)
        for idx, gid, tag in zip(tagged.index, tagged["tract"], tagged["tag"]):
            receivers[(gid, tag)].append(idx)

        log = []
        for (gid, tag), rows in unplaced.groupby(["GEOID", "tag"]):
            ring, dest = self.nearest_ring(gid, tag, has_tag)
            if ring is None:
                log += [dict(GEOID=gid, tag=tag, pool=r.pool, slot=r.slot, people=r.people,
                             action="not placed", ring=None, destinations="", receivers=0)
                        for r in rows.itertuples()]
                continue
            idx = [i for g in dest for i in receivers[(g, tag)]]
            fa = b.loc[idx, "floor_area"].to_numpy(dtype=float)
            w = fa / fa.sum() if fa.sum() > 0 else np.full(len(idx), 1.0 / len(idx))
            for r in rows.itertuples():
                people.loc[idx, f"n_people_{r.slot}_in"] += w * r.people_in
                people.loc[idx, f"n_people_{r.slot}_out"] += w * (r.people - r.people_in)
                log.append(dict(GEOID=gid, tag=tag, pool=r.pool, slot=r.slot, people=r.people,
                                action="relocated", ring=ring, destinations=";".join(dest),
                                receivers=len(idx)))
        return pd.DataFrame(log, columns=["GEOID", "tag", "pool", "slot", "people", "action",
                                          "ring", "destinations", "receivers"])
