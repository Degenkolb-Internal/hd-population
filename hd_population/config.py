from dataclasses import dataclass
from typing import Optional, Sequence

# PRFIL (share of commuters who drive) by tract population density, people per square mile.
# The shares are the Hazus values; the density cutoffs are ours: 1,000 people/mi2 is the
# Census urban-core threshold and 9,560 people/mi2 (3,825 housing units/mi2) the lower bound
# of the densest class in Gardner (2025), "Finding Suburbia in the Census".
PRFIL_BINS = ((9560.0, 0.60), (1000.0, 0.80), (0.0, 0.85))
PRFIL_DEFAULT = 0.80


@dataclass
class ModelConfig:
    """Settings of the H-D population model.

    city            name of the incorporated place the inventory covers. Tracts cut by the
                    city boundary then contribute only the share of their pools that lies
                    inside the city. None keeps whole counties.
    population_estimate_year
                    None uses the decennial counts as they are. A year scales each tract
                    by the growth of its county from April 2020 to July 1 of that year.
    relocate        move people that have no receiving building in their tract to the
                    nearest tracts that have one (up to `max_rings` rings of neighbors).
    """
    city: Optional[str] = None
    census_api_key: Optional[str] = None
    cache_dir: str = "~/.hd_population"
    refresh: bool = False

    decennial_year: int = 2020
    population_estimate_year: Optional[int] = None
    acs_year: int = 2023
    lodes_years: Sequence[int] = (2022, 2021, 2020, 2019)
    ipeds_directory_year: int = 2022
    ipeds_enrollment_years: Sequence[int] = (2022, 2021)
    ccd_year: int = 2022
    pss_vintage: Optional[str] = "2122"     # None leaves out private schools

    relocate: bool = True
    max_rings: int = 3
