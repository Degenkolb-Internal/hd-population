"""The Hazus Table 12-2 population schedule.

Coefficients follow the FEMA Hazus Earthquake Model Technical Manual (Table 12-2, p. 12-5
in editions 4.2 to 6.1). Each manual cell reads (indoor or outdoor split)(fraction present)
(pool), so every term below carries the two products for indoor and outdoor.

Two departures from the manual:
  * grade-school students come from enrollment at the schools in the tract (GRADE_edu),
    not from tract residents aged 16 and under;
  * education-sector workers (EDUW) are placed on educational buildings with the
    commercial-worker schedule. The manual has no workers in schools.
The commute rows of the table are only used for bridge casualties in Hazus and are left out.
"""
from dataclasses import dataclass
from typing import Callable

import pandas as pd

SLOTS = ("2am", "2pm", "5pm")

# receiver -> population groups that can take the people
RECEIVER_GROUPS = {
    "RES": ("RES1", "RES_OTHER"),
    "COM": ("COM",),
    "IND": ("IND",),
    "HOTEL": ("HOTEL",),
    "EDU": ("EDU",),
    "EDU_GRADE": ("EDU",),
    "EDU_COLLEGE": ("EDU",),
}
# receiver -> the building type that must exist in the tract for the term to land
RECEIVER_TAG = {"RES": "RES", "COM": "COM", "IND": "IND", "HOTEL": "HOTEL",
                "EDU": "EDU", "EDU_GRADE": "EDU", "EDU_COLLEGE": "EDU"}

TRACT_COLUMNS = ["POP", "NRES", "DRES", "COMM", "COMW", "INDW", "EDUW",
                 "GRADE_edu", "COLLEGE_edu", "HOTEL", "VISIT", "PRFIL"]


def column(name):
    return lambda t: t[name]


def commuters_2pm(t):
    # non-auto commuters outdoors at 2 p.m.; car commuters are not in any building
    return 0.5 * (1.0 - t["PRFIL"]) * 0.05 * t["POP"]


def commuters_5pm(t):
    return 0.5 * (1.0 - t["PRFIL"]) * (0.05 * t["POP"] + t["COMM"])


@dataclass(frozen=True)
class Term:
    pool: str
    slot: str
    receiver: str
    base: Callable[[pd.DataFrame], pd.Series]
    indoor: float
    outdoor: float

    @property
    def tag(self):
        return RECEIVER_TAG[self.receiver]

    def people(self, tracts):
        base = self.base(tracts).astype(float)
        return self.indoor * base, self.outdoor * base


TERMS = [
    # residents
    Term("residents", "2am", "RES", column("NRES"), 0.999 * 0.99, 0.001 * 0.99),
    Term("residents", "2pm", "RES", column("DRES"), 0.70 * 0.75, 0.30 * 0.75),
    Term("residents", "2pm", "COM", column("DRES"), 0.80 * 0.20, 0.20 * 0.20),
    Term("residents", "5pm", "RES", column("NRES"), 0.70 * 0.50, 0.30 * 0.50),
    Term("residents", "5pm", "COM", column("NRES"), 0.98 * 0.10, 0.02 * 0.10),
    # commercial workers
    Term("commercial_workers", "2am", "COM", column("COMW"), 0.999 * 0.02, 0.001 * 0.02),
    Term("commercial_workers", "2pm", "COM", column("COMW"), 0.99 * 0.98, 0.01 * 0.98),
    Term("commercial_workers", "5pm", "COM", column("COMW"), 0.98 * 0.50, 0.02 * 0.50),
    # industrial workers
    Term("industrial_workers", "2am", "IND", column("INDW"), 0.999 * 0.10, 0.001 * 0.10),
    Term("industrial_workers", "2pm", "IND", column("INDW"), 0.90 * 0.80, 0.10 * 0.80),
    Term("industrial_workers", "5pm", "IND", column("INDW"), 0.90 * 0.50, 0.10 * 0.50),
    # students and school staff
    Term("students_staff", "2am", "EDU", column("EDUW"), 0.999 * 0.02, 0.001 * 0.02),
    Term("students_staff", "2pm", "EDU_GRADE", column("GRADE_edu"), 0.90 * 0.80, 0.10 * 0.80),
    Term("students_staff", "2pm", "EDU_COLLEGE", column("COLLEGE_edu"), 0.80, 0.20),
    Term("students_staff", "2pm", "EDU", column("EDUW"), 0.99 * 0.98, 0.01 * 0.98),
    Term("students_staff", "5pm", "EDU_COLLEGE", column("COLLEGE_edu"), 0.80 * 0.50, 0.20 * 0.50),
    Term("students_staff", "5pm", "EDU", column("EDUW"), 0.98 * 0.50, 0.02 * 0.50),
    # hotel guests; at 2 p.m. 80% are out shopping (indoors in commercial buildings)
    Term("hotel_guests", "2am", "HOTEL", column("HOTEL"), 0.999, 0.001),
    Term("hotel_guests", "2pm", "HOTEL", column("HOTEL"), 0.19, 0.01),
    Term("hotel_guests", "2pm", "COM", column("HOTEL"), 0.80, 0.0),
    Term("hotel_guests", "5pm", "HOTEL", column("HOTEL"), 0.299, 0.001),
    Term("hotel_guests", "5pm", "COM", column("HOTEL"), 0.98 * 0.70, 0.02 * 0.70),
    # external visitors
    Term("visitors", "2pm", "COM", column("VISIT"), 0.80, 0.20),
    # commuters outside, booked against commercial buildings
    Term("commuters", "2pm", "COM", commuters_2pm, 0.0, 1.0),
    Term("commuters", "5pm", "COM", commuters_5pm, 0.0, 1.0),
]
