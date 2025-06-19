# =======================
# Temperature Series Tests
# =======================

from io import StringIO

import pandas as pd
from matplotlib import pyplot as plt

from c01_getting_started.r013a_exploring_types_of_time_series_data_univariate import (
    plot_temperature_series, extract_series_column, load_temperature_data
)

sample_temp_csv = """Date,Temp
1981-01-01,20.7
1981-01-02,17.9
1981-01-03,18.8
"""


def test_load_temperature_data():
    df = load_temperature_data(StringIO(sample_temp_csv))
    assert isinstance(df.index[0], pd.Timestamp)
    assert "Temp" in df.columns
    assert df.shape == (3, 1)


def test_extract_series_column_returns_series():
    df = load_temperature_data(StringIO(sample_temp_csv))
    series = extract_series_column(df)
    assert isinstance(series, pd.Series)
    assert series.name == "Temp"
    assert series.index.equals(df.index)


def test_plot_temperature_series_executes():
    series = pd.Series([20.7, 17.9, 18.8], index=pd.date_range("1981-01-01", periods=3))
    plot_temperature_series(series, show=False)
    plt.close()
