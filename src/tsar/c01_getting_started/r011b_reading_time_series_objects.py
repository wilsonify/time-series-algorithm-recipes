from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd


def load_gdp_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load GDP data from CSV."""
    return pd.read_csv(csv_path, header=0)


def add_time_index(df: pd.DataFrame, start_year: str = '1960', end_year: str = '2017') -> pd.DataFrame:
    """Add a yearly time index to the GDP dataframe."""
    date_range = pd.date_range(start=f'1/1/{start_year}', end=f'31/12/{end_year}', freq='YE')
    df = df.copy()
    df['TimeIndex'] = date_range
    return df


def plot_gdp(df: pd.DataFrame, x_col: str = 'TimeIndex', y_col: str = 'GDPpercapita',show=True) -> None:
    """Plot GDP per capita over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[y_col], label="GDP per Capita")
    plt.title("India GDP per Capita Over Time")
    plt.xlabel("Year")
    plt.ylabel("GDP per Capita")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()
