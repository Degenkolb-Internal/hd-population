from dataclasses import dataclass, field

import pandas as pd

from .allocation import COLUMNS, Allocator
from .config import ModelConfig
from .inventory import Inventory
from .relocation import Relocator
from .tracts import TractTableBuilder


@dataclass
class PopulationResult:
    buildings: pd.DataFrame          # the input inventory plus tract, group and n_people_* columns
    tracts: pd.DataFrame             # the tract pools used
    unplaced: pd.DataFrame           # people without a receiving building in their tract
    relocation: pd.DataFrame = field(default_factory=pd.DataFrame)

    def totals(self):
        """People per hour, indoors plus outdoors."""
        b = self.buildings
        return {s: float(b[f"n_people_{s}_in"].sum() + b[f"n_people_{s}_out"].sum())
                for s in ("2am", "2pm", "5pm")}


class HDPopulationModel:
    """Hazus-Degenkolb (H-D) building population model.

        model = HDPopulationModel(ModelConfig(city="San Francisco", census_api_key=KEY))
        result = model.populate(Inventory(df))
        result.buildings[["n_people_2pm_in", "n_people_2pm_out"]]

    `populate` builds the tract pools from public data, assigns every building to its
    tract, distributes the pools with the Hazus Table 12-2 schedule and, if configured,
    relocates the people that have no receiving building in their tract.
    """

    def __init__(self, config=None):
        self.config = config or ModelConfig()
        self.allocator = Allocator()

    def tract_pools(self, inventory):
        return TractTableBuilder(self.config).build(inventory)

    def populate(self, inventory, tracts=None):
        if not isinstance(inventory, Inventory):
            inventory = Inventory(inventory)
        if tracts is None:
            tracts = self.tract_pools(inventory)
        tracts = tracts.copy()
        tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)

        inventory.assign_tracts(tracts)
        tracts["HOTEL"] = tracts["GEOID"].map(inventory.hotel_guests_by_tract()).fillna(0.0)

        people, unplaced = self.allocator.allocate(inventory, tracts)
        log = pd.DataFrame()
        if self.config.relocate and not unplaced.empty:
            log = Relocator(tracts, self.config.max_rings).relocate(inventory, people, unplaced)

        out = inventory.data.copy()
        out["census_tract"] = inventory.buildings["tract"].to_numpy()
        out["population_group"] = inventory.buildings["group"].to_numpy()
        for c in COLUMNS:
            out[c] = people[c].to_numpy()
        return PopulationResult(out, tracts, unplaced, log)


def populate(inventory, city=None, census_api_key=None, **settings):
    """Shortcut: the inventory table with H-D populations added."""
    config = ModelConfig(city=city, census_api_key=census_api_key, **settings)
    return HDPopulationModel(config).populate(inventory).buildings
