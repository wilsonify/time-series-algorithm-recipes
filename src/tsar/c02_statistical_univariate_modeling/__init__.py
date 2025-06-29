from collections import deque
from typing import List

import numpy as np
from dateutil.parser import parse
import pandas as pd
from collections import deque
import pandas as pd
import json

def default_serializer(obj):
    """Handle non-serializable objects like deque or pandas.Timestamp."""
    if isinstance(obj, deque):
        return list(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DateOffset):
        return {"_date_offset": True, "kwds": obj.kwds}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")



def default_deserializer(obj):
    """Reconstruct known types from saved JSON structure."""
    # Handle pandas DateOffset
    if isinstance(obj, dict) and obj.get("_date_offset"):
        return pd.DateOffset(**obj["kwds"])
    return obj

def _looks_like_datetime(s):
    try:
        parse(s)
        return True
    except Exception:
        return False


def timestamp_to_float_year(ts: pd.Timestamp) -> float:
    # Calculate fractional year
    year = ts.year
    day_of_year = ts.dayofyear
    days_in_year = 366 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 365
    fraction_of_day = (ts.hour + ts.minute / 60 + ts.second / 3600 + ts.microsecond / 3.6e9) / 24
    return year + (day_of_year - 1 + fraction_of_day) / days_in_year


def float_year_to_datetime(t: float) -> pd.Timestamp:
    """Convert float year (e.g., 2002.5) to pandas Timestamp."""
    year = int(t)
    remainder = t - year
    return pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(remainder * 365.25, unit="D")


def moving_average_centered_filled(data: List[float], window: int) -> List[float]:
    """
    Compute a centered moving average with forward/backward fill using pure Python + NumPy.

    Parameters:
        data : List[float] - input data
        window : int       - rolling window size

    Returns:
        List[float] - smoothed data
    """
    n = len(data)
    result = [np.nan] * n
    half_window = window // 2

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window_vals = data[start:end]
        result[i] = np.mean(window_vals)

    # Forward-fill
    for i in range(n):
        if np.isnan(result[i]):
            result[i] = next((result[j] for j in range(i + 1, n) if not np.isnan(result[j])), result[i])

    # Backward-fill
    for i in reversed(range(n)):
        if np.isnan(result[i]):
            result[i] = next((result[j] for j in range(i - 1, -1, -1) if not np.isnan(result[j])), result[i])

    return result
