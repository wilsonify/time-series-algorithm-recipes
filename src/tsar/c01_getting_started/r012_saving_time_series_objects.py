import pandas as pd
from pathlib import Path
from typing import Union


def load_airpassenger_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load AirPassengers data from a CSV file with proper date parsing."""
    return pd.read_csv(
        filepath_or_buffer=csv_path,
        parse_dates=['Month'],
        index_col='Month',
        date_parser=pd.to_datetime
    )


def save_time_series_to_csv(data: pd.DataFrame, csv_path: Union[str, Path]) -> None:
    """Save time series DataFrame to CSV."""
    data.to_csv(csv_path, index=True, sep=',')


def reload_time_series_from_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Reload time series DataFrame from CSV."""
    return pd.read_csv(csv_path)
