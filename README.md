# hd-population

Building-level population for earthquake casualty analysis, following the Hazus population
schedule (FEMA Hazus Earthquake Model Technical Manual, Table 12-2).

Input: a building inventory (one row per building, e.g. the R2D inventories of the
[Building Inventory Generation](https://github.com/mlochhead/Building_Inventory_Generation)
workflow).
Output: the same inventory with the number of people in each building at 2 a.m., 2 p.m. and
5 p.m., indoors and outdoors:

```
n_people_2am_in   n_people_2am_out
n_people_2pm_in   n_people_2pm_out
n_people_5pm_in   n_people_5pm_out
```

plus `census_tract` and `population_group`.

## Install

```
pip install -e .
```

Python 3.10 or later. A free Census API key is recommended
(https://api.census.gov/data/key_signup.html).

## Use

```python
import pandas as pd
from hd_population import HDPopulationModel, Inventory, ModelConfig

df = pd.read_csv("San Francisco_R2D_Inventory.csv")
model = HDPopulationModel(ModelConfig(city="San Francisco", census_api_key="..."))
result = model.populate(Inventory(df))

result.buildings.to_csv("San Francisco_with_population.csv", index=False)
print(result.totals())
```

or in one line:

```python
from hd_population import populate
out = populate(df, city="San Francisco", census_api_key="...")
```

`Inventory` reads the R2D column names (`Latitude`, `Longitude`, `OccupancyClass`,
`PlanArea`, `NumberOfStories`, `NightPopulation`). Pass other names as keyword arguments if
your inventory differs. `NightPopulation` is only used for the hotel guests of RES4 buildings;
without it, hotels get no guests.

## What it does

1. **Tract pools.** For every census tract of the counties that contain the inventory, the
   model builds the Hazus pools from public data:

   | Pool | Meaning | Source |
   |---|---|---|
   | POP, NRES | residents | 2020 Census (P1), optionally scaled to a later year with the county population estimates |
   | COMM | residents who commute | ACS 5-year B08301 (workers minus work from home) |
   | DRES | residents at home during the day | NRES minus COMM, resident K-12 and resident college enrollment (ACS B14001) |
   | COMW, INDW, EDUW | jobs at workplaces in the tract | LODES workplace area characteristics, by NAICS sector |
   | GRADE_edu, COLLEGE_edu | enrollment of schools and colleges in the tract | NCES CCD, PSS and IPEDS |
   | HOTEL | hotel guests | inventory (RES4 night population) |
   | PRFIL | share of commuters who drive | 0.60 / 0.80 / 0.85 by tract population density |

   With `city=...`, only tracts that meet the city boundary are kept, and a tract cut by the
   boundary contributes the share of its pools that lies inside the city.

2. **Allocation.** Each pool goes to the buildings of its occupancy group in the tract in
   proportion to floor area, with the Table 12-2 indoor/outdoor fractions for each hour.
   Two departures from the manual: grade-school students come from school enrollment
   instead of residents aged 16 and under, and education-sector workers are placed in
   educational buildings with the commercial-worker schedule.

3. **Relocation.** When a tract has workers, students or guests but no building of the
   matching occupancy, those people are moved to the nearest neighboring tracts (up to
   three rings) that have such buildings, by floor area. No occupancy class is changed.
   `result.relocation` lists every move; `ModelConfig(relocate=False)` turns this off.

Downloads are cached as parquet files under `~/.hd_population` (`cache_dir`).

## Data years

The defaults are the 2020 Census, ACS 2019-2023, LODES 2022, CCD and IPEDS 2022, and PSS
2021-22. To align residents with a later reference year (NSI 2026 uses April 2025):

```python
ModelConfig(city="Seattle", population_estimate_year=2025, acs_year=2024,
            lodes_years=(2023, 2022), ipeds_directory_year=2024,
            ipeds_enrollment_years=(2024, 2023), ccd_year=2024, pss_vintage="2324")
```
