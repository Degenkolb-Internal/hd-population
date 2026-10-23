"""Offline checks on a small made-up city of three tracts in a row:

    A | B | C       A has homes only, B has homes and offices, C has a school.
"""
import pandas as pd
import pytest
from shapely.geometry import box

from hd_population import HDPopulationModel, Inventory, ModelConfig

TRACTS = ["06001000100", "06001000200", "06001000300"]


def tract_table(**pools):
    t = pd.DataFrame({"GEOID": TRACTS,
                      "the_geom": [box(i, 0, i + 1, 1).wkt for i in range(3)]})
    for col in ("POP", "NRES", "DRES", "COMM", "COMW", "INDW", "EDUW", "GRADE_edu",
                "COLLEGE_edu", "HOTEL", "VISIT"):
        t[col] = pools.get(col, [0.0, 0.0, 0.0])
    t["PRFIL"] = 0.8
    return t


def inventory():
    df = pd.DataFrame({
        "Latitude": [0.5, 0.5, 0.5, 0.5, 0.5],
        "Longitude": [0.3, 0.7, 1.3, 1.6, 2.5],
        "OccupancyClass": ["RES1", "RES3A", "RES1", "COM4", "EDU1"],
        "PlanArea": [1000.0, 3000.0, 2000.0, 5000.0, 4000.0],
        "NumberOfStories": [1, 1, 1, 2, 1],
    })
    return Inventory(df, hotel_guests=None)


def run(relocate=True, **pools):
    model = HDPopulationModel(ModelConfig(relocate=relocate))
    return model.populate(inventory(), tracts=tract_table(**pools))


def test_residents_split_by_floor_area():
    res = run(NRES=[1000.0, 0.0, 0.0])
    b = res.buildings
    night = b["n_people_2am_in"] + b["n_people_2am_out"]
    assert night.iloc[0] == pytest.approx(0.99 * 1000 * 0.25)
    assert night.iloc[1] == pytest.approx(0.99 * 1000 * 0.75)
    assert b["n_people_2am_out"].iloc[0] == pytest.approx(0.001 * 0.99 * 250)


def test_workers_without_offices_are_relocated():
    # tract A has 500 commercial workers but no commercial building; B has one
    res = run(COMW=[500.0, 0.0, 0.0])
    office = res.buildings.iloc[3]
    assert office["n_people_2pm_in"] == pytest.approx(0.99 * 0.98 * 500)
    assert office["n_people_2pm_out"] == pytest.approx(0.01 * 0.98 * 500)
    moved = res.relocation
    assert set(moved["action"]) == {"relocated"}
    assert set(moved["ring"]) == {1}


def test_without_relocation_people_are_reported():
    res = run(relocate=False, COMW=[500.0, 0.0, 0.0])
    assert res.buildings["n_people_2pm_in"].sum() == 0
    assert res.unplaced["people"].sum() == pytest.approx(500 * (0.02 + 0.98 + 0.50))


def test_students_and_staff():
    res = run(GRADE_edu=[0.0, 0.0, 300.0], EDUW=[0.0, 0.0, 40.0])
    school = res.buildings.iloc[4]
    assert school["n_people_2pm_in"] == pytest.approx(0.9 * 0.8 * 300 + 0.99 * 0.98 * 40)
    assert school["n_people_5pm_in"] == pytest.approx(0.98 * 0.5 * 40)
    assert school["n_people_2am_in"] == pytest.approx(0.999 * 0.02 * 40)


def test_hours_conserve_the_pools():
    pools = dict(POP=[1000.0, 800.0, 0.0], NRES=[1000.0, 800.0, 0.0], DRES=[300.0, 200.0, 0.0],
                 COMM=[400.0, 300.0, 0.0], COMW=[0.0, 900.0, 0.0])
    b = run(**pools).buildings
    night = (b["n_people_2am_in"] + b["n_people_2am_out"]).sum()
    assert night == pytest.approx(0.99 * 1800 + 0.02 * 900)
