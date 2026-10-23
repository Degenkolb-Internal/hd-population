"""Distribution of the tract pools to the buildings of each tract.

Within a tract, every pool goes to the buildings of its receiving group in proportion to
gross floor area (equal shares if the group has no floor area). Residents are first split
between single-family (RES1) and other residential buildings by their floor areas.
Grade-school enrollment goes to grade schools (EDU1) and college enrollment to colleges
(EDU2); when a tract lacks the subtype, the students go to all its educational buildings.
"""
import numpy as np
import pandas as pd

from .schedule import SLOTS, TERMS

COLUMNS = [f"n_people_{s}_{loc}" for s in SLOTS for loc in ("in", "out")]


def shares(weights, keys):
    """Each weight divided by the total of its key; equal shares where a key totals zero."""
    total = weights.groupby(keys).transform("sum")
    count = weights.groupby(keys).transform("count")
    return np.where(total > 0, weights / total.where(total > 0, 1.0), 1.0 / count)


class Allocator:
    def __init__(self, terms=TERMS):
        self.terms = terms

    def receiver_weights(self, b):
        """Building weights for each receiver. Weights of a receiver sum to one in every
        tract that has at least one of its buildings."""
        w = {}
        group_share = pd.Series(shares(b["floor_area"], [b["tract"], b["group"]]), index=b.index)

        res = b[b["group"].isin(["RES1", "RES_OTHER"])]
        if not res.empty:
            area = res.groupby(["tract", "group"])["floor_area"].sum().unstack(fill_value=0.0)
            count = res.groupby(["tract", "group"]).size().unstack(fill_value=0)
            area = area.reindex(columns=["RES1", "RES_OTHER"], fill_value=0.0)
            count = count.reindex(columns=["RES1", "RES_OTHER"], fill_value=0)
            # no residential floor area in the tract: split by number of buildings
            no_area = area.sum(axis=1) <= 0
            area[no_area] = count[no_area].astype(float)
            res1 = (area["RES1"] / area.sum(axis=1)).fillna(0.0)
            split = np.where(res["group"] == "RES1", res["tract"].map(res1), 1.0 - res["tract"].map(res1))
            w["RES"] = pd.Series(split * group_share[res.index], index=res.index)
        else:
            w["RES"] = pd.Series(dtype=float)

        for group in ("COM", "IND", "HOTEL", "EDU"):
            w[group] = group_share[b["group"] == group]

        edu = b[b["group"] == "EDU"]
        has_college = edu.groupby("tract")["college"].transform("any")
        has_grade = (~edu["college"]).groupby(edu["tract"]).transform("any")
        grade = edu[~edu["college"] | ~has_grade]
        college = edu[edu["college"] | ~has_college]
        w["EDU_GRADE"] = pd.Series(shares(grade["floor_area"], grade["tract"]), index=grade.index)
        w["EDU_COLLEGE"] = pd.Series(shares(college["floor_area"], college["tract"]), index=college.index)
        return w

    def allocate(self, inventory, tracts):
        """People per building (columns n_people_<slot>_<in|out>) and the table of
        people that have no receiving building in their tract."""
        b = inventory.buildings
        people = pd.DataFrame(0.0, index=b.index, columns=COLUMNS)
        known = set(tracts["GEOID"])
        placed = b[b["tract"].isin(known) & (b["group"] != "")]
        weights = self.receiver_weights(placed)

        t = tracts.set_index("GEOID")
        present = placed.groupby("tract")["tag"].agg(set)
        unplaced = []
        for term in self.terms:
            indoor, outdoor = term.people(t)
            w = weights[term.receiver]
            if len(w):
                tract_of = placed.loc[w.index, "tract"]
                people.loc[w.index, f"n_people_{term.slot}_in"] += w * tract_of.map(indoor).to_numpy()
                people.loc[w.index, f"n_people_{term.slot}_out"] += w * tract_of.map(outdoor).to_numpy()

            # tracts with buildings but none that can receive this term
            covered = indoor.index.isin(present.index)
            lacks = np.array([term.tag not in present.get(g, set()) for g in indoor.index])
            total = indoor + outdoor
            lost = covered & lacks & (total.to_numpy() > 0)
            if lost.any():
                unplaced.append(pd.DataFrame({
                    "GEOID": indoor.index[lost], "pool": term.pool, "tag": term.tag,
                    "slot": term.slot, "people": total[lost].to_numpy(),
                    "people_in": indoor[lost].to_numpy()}))
        unplaced = (pd.concat(unplaced, ignore_index=True) if unplaced else
                    pd.DataFrame(columns=["GEOID", "pool", "tag", "slot", "people", "people_in"]))
        return people, unplaced
