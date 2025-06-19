"""
=======================
# Air Passenger Tests
=======================
"""
from io import StringIO

import pandas as pd
from matplotlib import pyplot as plt

from c01_getting_started import path_to_data
from c01_getting_started.r011a_reading_time_series_objects import load_airpassenger_data, plot_passenger_data

sample_air_csv = """Month,Passengers
1949-01,112
1949-02,118
"""

def test_load_airpassenger_data():
    df = load_airpassenger_data(f"{path_to_data}/input/airpassengers.csv")
    assert isinstance(df, pd.DataFrame), "Expected DataFrame"
    assert list(df.columns) == ["#Passengers"], "Expected column '#Passengers'"
    assert isinstance(df.index[0], pd.Timestamp), "Expected datetime index"
    assert df.loc["1949-01-01", "#Passengers"] == 112


def test_plot_passenger_data():
    df = load_airpassenger_data(f"{path_to_data}/input/airpassengers.csv")
    fig, ax = plt.subplots()
    plot_passenger_data(df, ax, show=False)
    plt.close(fig)


def test_load_airpassenger_data_parses_dates():
    df = load_airpassenger_data(StringIO(sample_air_csv))
    assert df.shape == (2, 1), "Expected 2 rows, 1 column"
    assert isinstance(df.index[0], pd.Timestamp), "Expected datetime index"
    assert "Passengers" in df.columns
