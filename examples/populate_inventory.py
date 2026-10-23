"""Add H-D populations to an R2D inventory.

    python examples/populate_inventory.py "San Francisco_R2D_Inventory.csv" "San Francisco"

The Census API key is read from the CENSUS_API_KEY environment variable.
"""
import os
import sys

import pandas as pd

from hd_population import HDPopulationModel, Inventory, ModelConfig


def main(path, city):
    df = pd.read_csv(path, low_memory=False)
    config = ModelConfig(city=city, census_api_key=os.environ.get("CENSUS_API_KEY"))
    result = HDPopulationModel(config).populate(Inventory(df))

    out = os.path.splitext(path)[0] + "_HD_population.csv"
    result.buildings.to_csv(out, index=False)
    print(f"{len(result.buildings):,} buildings -> {out}")
    for hour, people in result.totals().items():
        print(f"  {hour}: {people:,.0f} people")
    moved = result.relocation
    if not moved.empty:
        done = moved[moved["action"] == "relocated"]
        print(f"  relocated at 2 p.m.: {done.loc[done['slot'] == '2pm', 'people'].sum():,.0f} people")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
