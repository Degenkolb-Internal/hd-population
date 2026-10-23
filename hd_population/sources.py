"""Public data sources for the tract pools, each cached on disk as parquet.

TIGERweb      tract, county and place boundaries
Census API    decennial counts (P1) and ACS 5-year tables
LODES         workplace area characteristics (jobs by NAICS sector)
NCES          school and college enrollment (Urban Institute API; PSS for private schools)
Population estimates   county totals for scaling the decennial counts to a later year
"""
import gzip
import hashlib
import io
import os
import time
import zipfile

import pandas as pd
import requests
from shapely import wkt
from shapely.geometry import shape

TIGERWEB = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb"
COUNTIES_URL = TIGERWEB + "/State_County/MapServer/1/query"
TRACTS_URL = TIGERWEB + "/Tracts_Blocks/MapServer/10/query"
PLACES_URL = TIGERWEB + "/Places_CouSub_ConCity_SubMCD/MapServer/4/query"
CENSUS_API = "https://api.census.gov/data"
LODES_URL = "https://lehd.ces.census.gov/data/lodes/{dataset}/{st}/wac/{st}_wac_S000_JT01_{year}.csv.gz"
URBAN_API = "https://educationdata.urban.org/api/v1"
EDGE_PRIVATE_URL = ("https://nces.ed.gov/opengis/rest/services/K12_School_Locations/"
                    "EDGE_GEOCODE_PRIVATESCH_{vintage}/MapServer/0/query")
PSS_URL = "https://nces.ed.gov/surveys/pss/zip/pss{vintage}_pu_csv.zip"
PEP_URL = ("https://www2.census.gov/programs-surveys/popest/datasets/2020-{year}/"
           "counties/totals/co-est{year}-alldata.csv")

STATE_USPS = {
    "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct",
    "10": "de", "11": "dc", "12": "fl", "13": "ga", "15": "hi", "16": "id", "17": "il",
    "18": "in", "19": "ia", "20": "ks", "21": "ky", "22": "la", "23": "me", "24": "md",
    "25": "ma", "26": "mi", "27": "mn", "28": "ms", "29": "mo", "30": "mt", "31": "ne",
    "32": "nv", "33": "nh", "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd",
    "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd",
    "47": "tn", "48": "tx", "49": "ut", "50": "vt", "51": "va", "53": "wa", "54": "wv",
    "55": "wi", "56": "wy", "72": "pr",
}


class Cache:
    """Parquet files under <root>/<category>/<key>.parquet. Empty results are not stored."""

    def __init__(self, root, refresh=False):
        self.root = os.path.expanduser(root)
        self.refresh = refresh

    def get(self, category, key, fetch):
        path = os.path.join(self.root, category, f"{key}.parquet")
        if not self.refresh and os.path.exists(path):
            return pd.read_parquet(path)
        df = fetch()
        if df is None:
            return pd.DataFrame()
        if not df.empty:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_parquet(path, index=False)
        return df


class Http:
    def __init__(self, retries=3, timeout=90):
        self.retries = retries
        self.timeout = timeout
        self.session = requests.Session()

    def get(self, url, params=None, timeout=None):
        error = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, params=params, timeout=timeout or self.timeout)
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                error = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"GET {url} failed after {self.retries} attempts: {error}")

    def arcgis(self, url, where=None, fields="*", extra=None):
        """All features of an ArcGIS REST query as GeoJSON, following the paging."""
        features, offset = [], 0
        while True:
            params = {"outFields": fields, "outSR": "4326", "f": "geojson",
                      "resultOffset": offset}
            if where is not None:
                params["where"] = where
            params.update(extra or {})
            r = self.get(url, params)
            text = r.text.lstrip()
            if not text.startswith(("{", "[")):
                raise RuntimeError(f"unexpected response from {url}: {text[:80]!r}")
            page = r.json()
            batch = page.get("features") or []
            features.extend(batch)
            if not batch or not page.get("exceededTransferLimit"):
                return features
            offset += len(batch)

    def urban(self, url, params):
        """All rows of an Urban Institute Education Data API query."""
        rows, first = [], True
        while url:
            page = self.get(url, params if first else None).json()
            rows.extend(page.get("results") or [])
            url, first = page.get("next"), False
        return rows


class TigerWeb:
    def __init__(self, http, cache):
        self.http, self.cache = http, cache

    def counties(self, bounds):
        """Counties whose extent meets the box (xmin, ymin, xmax, ymax), as (fips, polygon)."""
        xmin, ymin, xmax, ymax = bounds
        features = self.http.arcgis(COUNTIES_URL, fields="GEOID", extra={
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope", "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects"})
        return [(str(f["properties"]["GEOID"]).zfill(5), shape(f["geometry"]))
                for f in features if f.get("geometry")]

    def tracts(self, county):
        def fetch():
            features = self.http.arcgis(
                TRACTS_URL, f"STATE='{county[:2]}' AND COUNTY='{county[2:]}'",
                "GEOID,STATE,COUNTY,TRACT,AREALAND")
            return pd.DataFrame([{
                "GEOID": str(f["properties"]["GEOID"]).zfill(11),
                "the_geom": shape(f["geometry"]).wkt,
                "AREALAND": pd.to_numeric(f["properties"].get("AREALAND"), errors="coerce"),
            } for f in features if f.get("geometry")])
        return self.cache.get("tract_geom", county, fetch)

    def place(self, name, states):
        """Polygon of the incorporated place called `name`. With several matches the
        largest one wins."""
        key = name.lower().replace(" ", "_") + "_" + "_".join(sorted(states))

        def fetch():
            states_sql = " OR ".join(f"STATE='{s}'" for s in sorted(states))
            features = self.http.arcgis(PLACES_URL, f"BASENAME='{name}' AND ({states_sql})",
                                        "GEOID,BASENAME,STATE,AREALAND")
            return pd.DataFrame([{
                "GEOID": f["properties"].get("GEOID"),
                "AREALAND": pd.to_numeric(f["properties"].get("AREALAND"), errors="coerce"),
                "the_geom": shape(f["geometry"]).wkt,
            } for f in features if f.get("geometry")])
        df = self.cache.get("place_geom", key, fetch)
        if df.empty:
            return None
        return wkt.loads(df.sort_values("AREALAND", ascending=False).iloc[0]["the_geom"])


class CensusApi:
    def __init__(self, http, cache, api_key=None):
        self.http, self.cache, self.api_key = http, cache, api_key

    def tracts(self, dataset, variables, counties):
        """One row per tract with the requested variables (as strings) and GEOID."""
        tag = dataset.replace("/", "_")
        vhash = hashlib.sha1(",".join(variables).encode()).hexdigest()[:8]
        parts = []
        for county in counties:
            def fetch(county=county):
                params = {"get": ",".join(variables), "for": "tract:*",
                          "in": f"state:{county[:2]}+county:{county[2:]}"}
                if self.api_key:
                    params["key"] = self.api_key
                data = self.http.get(f"{CENSUS_API}/{dataset}", params).json()
                return pd.DataFrame(data[1:], columns=data[0])
            parts.append(self.cache.get("census", f"{tag}_{vhash}_{county}", fetch))
        df = pd.concat(parts, ignore_index=True)
        df["GEOID"] = df["state"] + df["county"] + df["tract"]
        return df


class Lodes:
    """LODES workplace area characteristics, primary jobs (JT01), all workers (S000)."""

    def __init__(self, http, cache, dataset="LODES8"):
        self.http, self.cache, self.dataset = http, cache, dataset

    def state(self, fips, years):
        """(year, table) for the first year in `years` that the state has, else (None, None)."""
        st = STATE_USPS[fips]
        for year in years:
            def fetch(year=year):
                url = LODES_URL.format(dataset=self.dataset, st=st, year=year)
                try:
                    r = self.http.session.get(url, timeout=300)
                except requests.RequestException:
                    return None
                if r.status_code != 200:
                    return None
                with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
                    df = pd.read_csv(gz, dtype=str)
                return df if "w_geocode" in df.columns else None
            df = self.cache.get("lodes_wac", f"{self.dataset}_{st}_{year}", fetch)
            if not df.empty:
                return year, df
        return None, None


class Schools:
    """School and college locations with enrollment."""

    def __init__(self, http, cache):
        self.http, self.cache = http, cache

    def colleges(self, state, directory_year, enrollment_years):
        """IPEDS institutions of a state: unitid, lat, lon, county, total fall enrollment."""
        def fetch_dir():
            rows = self.http.urban(f"{URBAN_API}/college-university/ipeds/directory/{directory_year}/",
                                   {"fips": int(state)})
            return pd.DataFrame(rows).reindex(columns=["unitid", "latitude", "longitude", "county_fips"])
        directory = self.cache.get("ipeds_dir", f"{directory_year}_{state}", fetch_dir)
        if directory.empty:
            return directory

        enrollment = {}
        for year in enrollment_years:
            def fetch_enr(year=year):
                rows = self.http.urban(f"{URBAN_API}/college-university/ipeds/fall-enrollment/{year}/99/",
                                       {"fips": int(state), "ftpt": 99, "sex": 99, "race": 99})
                return pd.DataFrame(rows).reindex(
                    columns=["unitid", "degree_seeking", "class_level", "enrollment_fall"])
            e = self.cache.get("ipeds_enr", f"{year}_{state}", fetch_enr)
            if e.empty:
                continue
            total = e[(e["degree_seeking"] == 99) & (e["class_level"] == 99)]
            enrollment = dict(zip(total["unitid"],
                                  pd.to_numeric(total["enrollment_fall"], errors="coerce").fillna(0)))
            break
        out = directory.rename(columns={"latitude": "lat", "longitude": "lon"})
        out["county"] = out["county_fips"].astype(str).str.zfill(5)
        out["enroll"] = [max(0, int(enrollment.get(u, 0) or 0)) for u in out["unitid"]]
        return out[["lat", "lon", "county", "enroll"]]

    def public_schools(self, state, year):
        """CCD public K-12 schools of a state: lat, lon, county, enrollment."""
        def fetch():
            rows = self.http.urban(f"{URBAN_API}/schools/ccd/directory/{year}/", {"fips": int(state)})
            return pd.DataFrame(rows).reindex(columns=["latitude", "longitude", "county_code", "enrollment"])
        d = self.cache.get("ccd_dir", f"{year}_{state}", fetch)
        if d.empty:
            return d
        enroll = pd.to_numeric(d["enrollment"], errors="coerce")
        enroll = enroll.where(enroll >= 0, 0).fillna(0)       # NCES codes missing values as negatives
        return pd.DataFrame({"lat": d["latitude"], "lon": d["longitude"],
                             "county": d["county_code"].astype(str).str.zfill(5),
                             "enroll": enroll.astype(int)})

    def private_schools(self, states, vintage):
        """PSS private K-12 schools: EDGE geocodes joined to PSS enrollment on PPIN.
        Returns an empty table if either source cannot be read."""
        cols = ["lat", "lon", "enroll"]
        try:
            locations = []
            for state in sorted(states):
                def fetch_edge(state=state):
                    features = self.http.arcgis(EDGE_PRIVATE_URL.format(vintage=vintage),
                                                f"STFIP='{state}'", "PPIN,LAT,LON")
                    return pd.DataFrame([{"PPIN": str(f["properties"]["PPIN"]),
                                          "LAT": f["properties"].get("LAT"),
                                          "LON": f["properties"].get("LON")}
                                         for f in features
                                         if f.get("properties", {}).get("PPIN") is not None])
                key = state if vintage == "2122" else f"{vintage}_{state}"
                locations.append(self.cache.get("edge_private", key, fetch_edge))
            loc = pd.concat(locations, ignore_index=True)
            if loc.empty:
                return pd.DataFrame(columns=cols)

            def fetch_pss():
                r = self.http.get(PSS_URL.format(vintage=vintage), timeout=300)
                zf = zipfile.ZipFile(io.BytesIO(r.content))
                name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                with zf.open(name) as f:
                    return pd.read_csv(f, dtype=str, usecols=["PPIN", "NUMSTUDS"], encoding="latin-1")
            pss = self.cache.get("pss_csv", f"pss{vintage}", fetch_pss)
        except (RuntimeError, KeyError, ValueError, zipfile.BadZipFile) as e:
            print(f"private schools skipped ({type(e).__name__}: {e})")
            return pd.DataFrame(columns=cols)

        enroll = dict(zip(pss["PPIN"].astype(str),
                          pd.to_numeric(pss["NUMSTUDS"], errors="coerce").fillna(0.0)))
        loc["PPIN"] = loc["PPIN"].astype(str)
        loc = loc.drop_duplicates("PPIN", keep="last")
        loc = loc[loc["PPIN"].isin(enroll)]
        return pd.DataFrame({"lat": loc["LAT"].to_numpy(), "lon": loc["LON"].to_numpy(),
                             "enroll": [max(0.0, float(enroll[p])) for p in loc["PPIN"]]})


class CountyEstimates:
    """Census Bureau county population estimates (Vintage <year>)."""

    def __init__(self, http, cache):
        self.http, self.cache = http, cache

    def growth(self, year, counties):
        """county -> July 1 estimate of `year` / April 1, 2020 estimates base."""
        def fetch():
            r = self.http.get(PEP_URL.format(year=year), timeout=300)
            df = pd.read_csv(io.BytesIO(r.content), dtype=str, encoding="latin-1")
            df = df[df["SUMLEV"] == "050"]
            return pd.DataFrame({"county": df["STATE"] + df["COUNTY"],
                                 "base": df["ESTIMATESBASE2020"],
                                 "estimate": df[f"POPESTIMATE{year}"]})
        pep = self.cache.get("pep_county", f"co-est{year}", fetch)
        pep = pep[pep["county"].isin(set(counties))]
        return dict(zip(pep["county"], pd.to_numeric(pep["estimate"]) / pd.to_numeric(pep["base"])))
