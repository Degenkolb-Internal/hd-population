import numpy as np
import pandas as pd
from shapely import points, wkt
from shapely.strtree import STRtree

GROUPS = ("RES1", "RES_OTHER", "HOTEL", "COM", "IND", "EDU")
GROUP_TAG = {"RES1": "RES", "RES_OTHER": "RES", "HOTEL": "HOTEL",
             "COM": "COM", "IND": "IND", "EDU": "EDU"}


def population_group(occupancy):
    """Hazus occupancy class (RES1, COM4, EDU2, ...) -> population group."""
    if not isinstance(occupancy, str) or not occupancy:
        return ""
    occ = occupancy.upper()
    if occ.startswith("RES1"):
        return "RES1"
    if occ.startswith("RES4"):
        return "HOTEL"
    if occ.startswith("RES"):
        return "RES_OTHER"
    if occ.startswith("EDU"):
        return "EDU"
    if occ.startswith("IND"):
        return "IND"
    # agriculture, religion and government follow the commercial schedule
    if occ.startswith("COM") or occ.startswith("GOV") or occ in ("AGR1", "REL1"):
        return "COM"
    return ""


class Inventory:
    """Building inventory as seen by the population model.

    `data` is any table with one row per building. The column names default to the
    SimCenter R2D inventory format; pass other names if your inventory differs.
    `hotel_guests` is the column with the number of guests in each hotel (RES4)
    building, e.g. the NSI night population. Leave it as None to run without hotel guests.
    """

    def __init__(self, data, latitude="Latitude", longitude="Longitude",
                 occupancy="OccupancyClass", plan_area="PlanArea",
                 stories="NumberOfStories", hotel_guests="NightPopulation"):
        self.data = data.reset_index(drop=True)
        df = self.data

        stories_n = pd.to_numeric(df[stories], errors="coerce").fillna(1).round()
        area = pd.to_numeric(df[plan_area], errors="coerce").fillna(0.0)
        floor_area = np.where(area > 0, area * np.where(stories_n > 0, stories_n, 1), 0.0)

        occ = df[occupancy].where(df[occupancy].notna(), "").astype(str)
        self.buildings = pd.DataFrame({
            "lat": pd.to_numeric(df[latitude], errors="coerce"),
            "lon": pd.to_numeric(df[longitude], errors="coerce"),
            "occupancy": occ,
            "group": occ.map(population_group),
            "floor_area": floor_area.astype(float),
            "college": occ.str.upper().str.startswith("EDU2"),
            "tract": None,
        })
        guests = 0.0
        if hotel_guests is not None and hotel_guests in df.columns:
            guests = pd.to_numeric(df[hotel_guests], errors="coerce").fillna(0.0)
        self.buildings["hotel_guests"] = np.where(self.buildings["group"] == "HOTEL", guests, 0.0)
        self.buildings["tag"] = self.buildings["group"].map(GROUP_TAG)

    def __len__(self):
        return len(self.buildings)

    def bounds(self):
        b = self.buildings.dropna(subset=["lat", "lon"])
        if b.empty:
            raise ValueError("inventory has no buildings with coordinates")
        return b["lon"].min(), b["lat"].min(), b["lon"].max(), b["lat"].max()

    def assign_tracts(self, tracts):
        """Point-in-polygon of every building on the tract polygons (WKT in `the_geom`).
        A building on a shared boundary goes to the first tract in table order."""
        geoms = [wkt.loads(w) for w in tracts["the_geom"]]
        tree = STRtree(geoms)
        b = self.buildings
        ok = b["lat"].notna() & b["lon"].notna()
        pts = points(b.loc[ok, "lon"].to_numpy(), b.loc[ok, "lat"].to_numpy())
        hit_pt, hit_tract = tree.query(pts, predicate="intersects")
        first = pd.Series(hit_tract).groupby(hit_pt).min()
        geoid = np.asarray(tracts["GEOID"], dtype=object)
        tract = np.full(len(b), None, dtype=object)
        rows = np.flatnonzero(ok.to_numpy())
        tract[rows[first.index.to_numpy()]] = geoid[first.to_numpy()]
        b["tract"] = tract
        return self

    def hotel_guests_by_tract(self):
        b = self.buildings
        return b[b["tract"].notna()].groupby("tract")["hotel_guests"].sum()
