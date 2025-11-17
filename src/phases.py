from datetime import date
from typing import Optional

import pandas as pd

from .config import (
    PHASE_CONFIG,
    PHASE_CAMPAIGN,
    PHASE_PRESIDENCY,
    PHASE_OTHER,
)

def _to_date(value) -> Optional[date]:
    """Robust conversion of timestamps or strings to a date object."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert("UTC")
        return value.date()

    if isinstance(value, date):
        return value

    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:
        return None

    if pd.isna(ts):
        return None
    return ts.date()

def assign_phase(actor: str, timestamp) -> str:
    """Assign campaign / presidency / other based on actor and date.

    Campaign intervals have priority over presidency.
    """
    d = _to_date(timestamp)
    if d is None:
        return PHASE_OTHER

    conf = PHASE_CONFIG.get(actor)
    if conf is None:
        return PHASE_OTHER

    # Campaign has priority
    for interval in conf.campaigns:
        if interval.start <= d <= interval.end:
            return PHASE_CAMPAIGN

    # Otherwise, presidency if within any presidency interval
    for interval in conf.presidency:
        if interval.start <= d <= interval.end:
            return PHASE_PRESIDENCY

    return PHASE_OTHER
