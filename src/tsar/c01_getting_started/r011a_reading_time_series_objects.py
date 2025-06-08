# airpassengers_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def load_airpassenger_data(csv_path: str) -> pd.DataFrame:
    """Load AirPassengers dataset from a CSV file."""
    return pd.read_csv(
        filepath_or_buffer=csv_path,
        parse_dates=['Month'],
        index_col='Month',
        date_parser=pd.to_datetime
    )


def plot_passenger_data(data: pd.DataFrame, ax: Optional[plt.Axes] = None) -> None:
    """Plot passenger count from the given DataFrame."""
    if ax is None:
        fig, ax = plt.subplots()
    data.plot(ax=ax, label='Passenger Count')
    ax.set_title("Monthly Air Passenger Count")
    ax.set_ylabel("Passengers")
    ax.set_xlabel("Date")
    ax.legend()
    plt.show()


def main():
    from c01_getting_started import path_to_data

    csv_path = f"{path_to_data}/input/airpassengers.csv"
    data = load_airpassenger_data(csv_path)
    plot_passenger_data(data)


if __name__ == "__main__":
    main()
