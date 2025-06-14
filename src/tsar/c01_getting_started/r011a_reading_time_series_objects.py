# airpassengers_analysis.py
from io import StringIO
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from c01_getting_started import path_to_data


def load_airpassenger_data(csv_path: str|StringIO) -> pd.DataFrame:
    """Load AirPassengers dataset from a CSV file."""
    return pd.read_csv(
        filepath_or_buffer=csv_path,
        parse_dates=['Month'],
        index_col='Month',
        date_parser=pd.to_datetime
    )


def plot_passenger_data(data: pd.DataFrame, ax: Optional[plt.Axes] = None,show=False) -> None:
    """Plot passenger count from the given DataFrame."""
    if ax is None:
        fig, ax = plt.subplots()

    data.plot(ax=ax, label='Passenger Count')
    ax.set_title("Monthly Air Passenger Count")
    ax.set_ylabel("Passengers")
    ax.set_xlabel("Date")
    ax.legend()

    if show:
        plt.show()


def main():
    csv_path = f"{path_to_data}/input/airpassengers.csv"
    data = load_airpassenger_data(csv_path)
    plot_passenger_data(data)


if __name__ == "__main__":
    main()
