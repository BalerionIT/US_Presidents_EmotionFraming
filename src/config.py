from datetime import date
from dataclasses import dataclass
from typing import List, Dict

# Canonical speaker names
TRUMP = "Donald Trump"
OBAMA = "Barack Obama"
BIDEN = "Joe Biden"

# Phase labels
PHASE_PRESIDENCY = "presidency"
PHASE_CAMPAIGN = "campaign"
PHASE_OTHER = "other"

@dataclass(frozen=True)
class PhaseInterval:
    start: date
    end: date

@dataclass(frozen=True)
class ActorPhaseConfig:
    presidency: List[PhaseInterval]
    campaigns: List[PhaseInterval]

def d(yyyy_mm_dd: str) -> date:
    year, month, day = map(int, yyyy_mm_dd.split("-"))
    return date(year, month, day)

# Date ranges are based on public inauguration / election / campaign announcement dates.
# We keep them simple and inclusive at the date level.
PHASE_CONFIG: Dict[str, ActorPhaseConfig] = {
    OBAMA: ActorPhaseConfig(
        presidency=[
            # Barack Obama was U.S. President from Jan 20, 2009 to Jan 20, 2017.
            PhaseInterval(d("2009-01-20"), d("2017-01-20")),
        ],
        campaigns=[
            # 2008 presidential campaign: Feb 10, 2007 (announcement) → Nov 4, 2008 (election day)
            PhaseInterval(d("2007-02-10"), d("2008-11-04")),
            # 2012 re-election campaign: Apr 4, 2011 (announcement) → Nov 6, 2012 (election day)
            PhaseInterval(d("2011-04-04"), d("2012-11-06")),
        ],
    ),
    TRUMP: ActorPhaseConfig(
        presidency=[
            # Donald Trump first term: Jan 20, 2017 → Jan 20, 2021
            PhaseInterval(d("2017-01-20"), d("2021-01-20")),
        ],
        campaigns=[
            # 2016 campaign: Jun 16, 2015 (announcement) → Nov 8, 2016 (election day)
            PhaseInterval(d("2015-06-16"), d("2016-11-08")),
            # 2020 re-election campaign: Jun 18, 2019 (Orlando rally) → Nov 3, 2020 (election day)
            PhaseInterval(d("2019-06-18"), d("2020-11-03")),
        ],
    ),
    BIDEN: ActorPhaseConfig(
        presidency=[
            # Joe Biden presidency: Jan 20, 2021 → Jan 20, 2025 (we cut at this window)
            PhaseInterval(d("2021-01-20"), d("2025-01-20")),
        ],
        campaigns=[
            # 2020 campaign: Apr 25, 2019 (announcement) → Nov 3, 2020 (election day)
            PhaseInterval(d("2019-04-25"), d("2020-11-03")),
        ],
    ),
}
