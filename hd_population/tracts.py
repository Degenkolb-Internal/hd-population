"""Tract population pools for the Hazus schedule.

Pools (people per tract):
  POP, NRES   residents (decennial census, optionally scaled to a later year)
  COMM        residents who commute to work (ACS B08301: workers minus work from home)
  DRES        residents at home during the day: NRES - COMM - resident K-12 - resident college
  COMW/INDW/EDUW   jobs at workplaces in the tract (LODES, by NAICS sector)
  GRADE, COLLEGE   resident enrollment (ACS B14001), only used for DRES
  GRADE_edu, COLLEGE_edu   enrollment of the schools and colleges located in the tract
  HOTEL       hotel guests (filled in later from the inventory)
  VISIT       external visitors (zero)
  PRFIL       share of commuters who drive
"""
import numpy as np
import pandas as pd
from shapely import points, wkt
from shapely.prepared import prep
from shapely.strtree import STRtree

from .config import PRFIL_BINS, PRFIL_DEFAULT
from .schedule import TRACT_COLUMNS
from .sources import Cache, CensusApi, CountyEstimates, Http, Lodes, Schools, TigerWeb

SQ_M_PER_SQ_MI = 2_589_988.0

COM_SECTORS = ["CNS01", "CNS06", "CNS07", "CNS09", "CNS10", "CNS11", "CNS12", "CNS13", "CNS14",
               "CNS16", "CNS17", "CNS18", "CNS19", "CNS20"]
IND_SECTORS = ["CNS02", "CNS03", "CNS04", "CNS05", "CNS08"]
EDU_SECTORS = ["CNS15"]

ACS_VARS = ["B08301_001E", "B08301_021E",                          # workers, worked from home
            "B14001_004E", "B14001_005E", "B14001_006E", "B14001_007E",   # K-12
            "B14001_008E", "B14001_009E"]                          # college, graduate

# person pools that scale with the share of a tract inside the city
PERSON_POOLS = ["POP", "NRES", "DRES", "COMM", "COMW", "INDW", "EDUW", "GRADE", "COLLEGE",
                "GRADE_edu", "COLLEGE_edu", "VISIT"]


class TractTableBuilder:
    def __init__(self, config):
        self.config = config
        http = Http()
        cache = Cache(config.cache_dir, refresh=config.refresh)
        self.tiger = TigerWeb(http, cache)
        self.census = CensusApi(http, cache, config.census_api_key)
        self.lodes = Lodes(http, cache)
        self.schools = Schools(http, cache)
        self.estimates = CountyEstimates(http, cache)
        self.counties = []

    def build(self, inventory):
        self.counties = self.find_counties(inventory)
        geom = pd.concat([self.tiger.tracts(c) for c in self.counties], ignore_index=True)
        geom = geom.drop_duplicates("GEOID").reset_index(drop=True)

        table = (geom.merge(self.residents(), on="GEOID", how="left")
                     .merge(self.jobs(), on="GEOID", how="left")
                     .merge(self.enrollment(geom), on="GEOID", how="left"))
        table["VISIT"] = 0.0
        table["HOTEL"] = 0.0
        table["PRFIL"] = self.prfil(table)
        if self.config.city:
            table = self.clip_to_city(table, self.config.city)
        for c in TRACT_COLUMNS + ["GRADE", "COLLEGE"]:
            table[c] = pd.to_numeric(table.get(c, 0.0), errors="coerce").fillna(0.0)
        return table

    def find_counties(self, inventory):
        """Counties that contain at least one building."""
        candidates = self.tiger.counties(inventory.bounds())
        if not candidates:
            raise RuntimeError("no county found around the inventory")
        tree = STRtree([g for _, g in candidates])
        b = inventory.buildings.dropna(subset=["lat", "lon"])
        hit = tree.query(points(b["lon"].to_numpy(), b["lat"].to_numpy()), predicate="within")[1]
        found = sorted({candidates[i][0] for i in np.unique(hit)})
        if not found:
            raise RuntimeError("no county contains the inventory buildings")
        return found

    def residents(self):
        cfg = self.config
        dec = self.census.tracts(f"{cfg.decennial_year}/dec/pl", ["P1_001N"], self.counties)
        dec["POP"] = pd.to_numeric(dec["P1_001N"], errors="coerce").fillna(0).astype(int)
        if cfg.population_estimate_year:
            growth = self.estimates.growth(cfg.population_estimate_year, self.counties)
            missing = sorted(set(self.counties) - set(growth))
            if missing:
                print(f"no county estimate for {missing}; those tracts keep the census count")
            factor = dec["GEOID"].str[:5].map(growth).fillna(1.0)
            dec["POP"] = (dec["POP"] * factor).round().astype(int)

        acs = self.census.tracts(f"{cfg.acs_year}/acs/acs5", ACS_VARS, self.counties)
        for v in ACS_VARS:
            acs[v] = pd.to_numeric(acs[v], errors="coerce").fillna(0).astype(int)

        df = dec[["GEOID", "POP"]].merge(acs[["GEOID"] + ACS_VARS], on="GEOID", how="outer")
        df = df.fillna(0)
        commuters = (df["B08301_001E"] - df["B08301_021E"]).clip(lower=0)
        df["NRES"] = df["POP"]
        df["GRADE"] = df[["B14001_004E", "B14001_005E", "B14001_006E", "B14001_007E"]].sum(axis=1)
        df["COLLEGE"] = df[["B14001_008E", "B14001_009E"]].sum(axis=1)
        df["DRES"] = (df["NRES"] - commuters - df["GRADE"] - df["COLLEGE"]).clip(lower=0)
        df["COMM"] = commuters
        return df[["GEOID", "POP", "NRES", "DRES", "COMM", "GRADE", "COLLEGE"]]

    def jobs(self):
        frames = []
        for state in sorted({c[:2] for c in self.counties}):
            year, wac = self.lodes.state(state, self.config.lodes_years)
            if wac is None:
                print(f"no LODES file for state {state}; its workers are zero")
                continue
            frames.append(wac[wac["w_geocode"].str.startswith(tuple(self.counties))])
        if not frames:
            return pd.DataFrame(columns=["GEOID", "COMW", "INDW", "EDUW"])
        wac = pd.concat(frames, ignore_index=True)
        sectors = COM_SECTORS + IND_SECTORS + EDU_SECTORS
        for s in sectors:
            wac[s] = pd.to_numeric(wac[s], errors="coerce").fillna(0).astype(int) if s in wac else 0
        wac["GEOID"] = wac["w_geocode"].str[:11]
        by_tract = wac.groupby("GEOID", as_index=False)[sectors].sum()
        by_tract["COMW"] = by_tract[COM_SECTORS].sum(axis=1)
        by_tract["INDW"] = by_tract[IND_SECTORS].sum(axis=1)
        by_tract["EDUW"] = by_tract[EDU_SECTORS].sum(axis=1)
        return by_tract[["GEOID", "COMW", "INDW", "EDUW"]]

    def enrollment(self, geom):
        """Enrollment of the schools and colleges located in each tract."""
        cfg = self.config
        states = sorted({c[:2] for c in self.counties})
        counties = set(self.counties)

        colleges = pd.concat([self.schools.colleges(s, cfg.ipeds_directory_year,
                                                    cfg.ipeds_enrollment_years) for s in states])
        grade = pd.concat([self.schools.public_schools(s, cfg.ccd_year) for s in states])
        colleges = colleges[colleges["county"].isin(counties)] if len(colleges) else colleges
        grade = grade[grade["county"].isin(counties)] if len(grade) else grade
        if cfg.pss_vintage:
            grade = pd.concat([grade, self.schools.private_schools(states, cfg.pss_vintage)],
                              ignore_index=True)

        tree = STRtree([wkt.loads(w) for w in geom["the_geom"]])
        out = pd.DataFrame({"GEOID": geom["GEOID"]})
        for name, schools in (("COLLEGE_edu", colleges), ("GRADE_edu", grade)):
            out[name] = 0.0
            schools = schools.dropna(subset=["lat", "lon"])
            if schools.empty:
                continue
            pts = points(schools["lon"].astype(float).to_numpy(), schools["lat"].astype(float).to_numpy())
            school_idx, tract_idx = tree.query(pts, predicate="within")
            first = pd.Series(tract_idx).groupby(school_idx).min()
            enroll = schools["enroll"].to_numpy(dtype=float)[first.index.to_numpy()]
            sums = pd.Series(enroll).groupby(first.to_numpy()).sum()
            out.loc[sums.index, name] = sums.to_numpy()
        return out

    def prfil(self, table):
        sq_mi = pd.to_numeric(table["AREALAND"], errors="coerce") / SQ_M_PER_SQ_MI
        density = pd.to_numeric(table["POP"], errors="coerce").fillna(0) / sq_mi.where(sq_mi > 0)
        prfil = pd.Series(PRFIL_DEFAULT, index=table.index)
        for cutoff, share in sorted(PRFIL_BINS):
            prfil[density >= cutoff] = share
        return prfil

    def clip_to_city(self, table, city):
        """Keep the tracts that meet the city and scale their person pools by the share
        of the tract area inside the city."""
        place = self.tiger.place(city, {c[:2] for c in self.counties})
        if place is None:
            print(f"no boundary found for {city!r}; whole counties are kept")
            return table
        inside = prep(place)
        share = np.zeros(len(table))
        for i, w in enumerate(table["the_geom"]):
            tract = wkt.loads(w)
            if not inside.intersects(tract):
                continue
            share[i] = 1.0 if inside.contains(tract) else tract.intersection(place).area / tract.area
        table = table.assign(city_share=share)
        table = table[table["city_share"] > 0].reset_index(drop=True)
        for c in PERSON_POOLS:
            table[c] = pd.to_numeric(table[c], errors="coerce").fillna(0.0) * table["city_share"]
        return table
