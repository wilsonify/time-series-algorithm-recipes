from pathlib import Path
from typing import Union

import pandas as pd


def save_time_series_to_csv(data: pd.DataFrame, csv_path: Union[str, Path]) -> None:
    """Save time series DataFrame to CSV."""
    data.to_csv(csv_path, index=True, sep=',')
