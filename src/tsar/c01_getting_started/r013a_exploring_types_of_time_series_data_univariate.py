from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd


def load_temperature_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load daily minimum temperature data from CSV.
    Assumes date is in the first column and should be parsed.
    """
    return pd.read_csv(
        filepath_or_buffer=csv_path,
        header=0,
        index_col=0,
        parse_dates=True,
    )


def extract_series_column(df: pd.DataFrame) -> pd.Series:
    """
    Extract the first column as a Series.
    Used when the DataFrame is single-column.
    """
    return df.iloc[:, 0]


def plot_temperature_series(
        series: pd.Series,
        ylabel: str = "Minimum Temp",
        title: str = "Min temp in Southern Hemisphere From 1981 to 1990",
        show=True
) -> None:
    """Plot a time series with predefined labels."""
    plt.figure(figsize=(12, 5))
    series.plot()
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()
