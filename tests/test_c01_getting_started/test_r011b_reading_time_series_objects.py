from io import StringIO

import pandas as pd
from matplotlib import pyplot as plt

from c01_getting_started import path_to_data, save_pickle, load_pickle
from c01_getting_started.r011b_reading_time_series_objects import load_gdp_data
from c01_getting_started.r011b_reading_time_series_objects import plot_gdp, add_time_index

sample_gdp_csv = """GDPpercapita
100
150
200
"""


def test_load_gdp_data_from_stringio():
    df = pd.read_csv(StringIO(sample_gdp_csv))
    assert not df.empty, "DataFrame should not be empty"
    assert "GDPpercapita" in df.columns
    assert df.shape == (3, 1)


def test_add_time_index():
    df = pd.read_csv(StringIO(sample_gdp_csv))
    df_with_time = add_time_index(df, start_year='2000', end_year='2002')
    assert "TimeIndex" in df_with_time.columns
    assert pd.api.types.is_datetime64_any_dtype(df_with_time["TimeIndex"])
    assert df_with_time["TimeIndex"].iloc[0].year == 2000


def test_plot_gdp_runs_without_error():
    df = pd.read_csv(StringIO(sample_gdp_csv))
    df_with_time = add_time_index(df, start_year='2000', end_year='2002')
    plot_gdp(df_with_time, show=False)
    plt.close()


def test_pickle_roundtrip():
    df = pd.read_csv(StringIO(sample_gdp_csv))
    path = f"{path_to_data}/output/gdp_test.obj"
    save_pickle(df, path)
    loaded = load_pickle(path)
    pd.testing.assert_frame_equal(df, loaded)


def test_test_r011b_reading_time_series_objects():
    csv_path = f"{path_to_data}/input/gdp_india.csv"
    df = load_gdp_data(csv_path)
    df = add_time_index(df, start_year='1960', end_year='2017')
    plot_gdp(df, x_col='TimeIndex', y_col='GDPpercapita', show=False)
