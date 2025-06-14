from io import StringIO

import pandas as pd
from matplotlib import pyplot as plt

from c01_getting_started import path_to_data, save_pickle, load_pickle
from c01_getting_started.r011a_reading_time_series_objects import load_airpassenger_data, plot_passenger_data
from c01_getting_started.r011b_reading_time_series_objects import plot_gdp, add_time_index
from c01_getting_started.r012_saving_time_series_objects import reload_time_series_from_csv, save_time_series_to_csv


def test_load_airpassenger_data():
    df = load_airpassenger_data(f"{path_to_data}/input/airpassengers.csv")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["#Passengers"]
    assert pd.Timestamp("1949-01-01") in df.index
    assert df.loc["1949-01-01", "#Passengers"] == 112


def test_plot_passenger_data():
    df = load_airpassenger_data(f"{path_to_data}/input/airpassengers.csv")
    fig, ax = plt.subplots()
    plot_passenger_data(df, ax, show=False)


def test_load_gdp_data_from_stringio():
    """Test loading GDP data using a mock path with StringIO."""
    df = pd.read_csv(StringIO("""GDPpercapita
        100
        150
        200
        """))
    assert not df.empty
    assert "GDPpercapita" in df.columns
    assert df.shape == (3, 1)


def test_add_time_index():
    """Ensure TimeIndex column is correctly added."""
    sample_df = pd.read_csv(StringIO("""GDPpercapita
        100
        150
        200
        """))
    df_with_time = add_time_index(sample_df, start_year='2000', end_year='2002')
    assert "TimeIndex" in df_with_time.columns
    assert pd.api.types.is_datetime64_any_dtype(df_with_time["TimeIndex"])
    assert len(df_with_time["TimeIndex"]) == 3
    assert df_with_time.iloc[0]["TimeIndex"].year == 2000


def test_pickle_roundtrip():
    """Test saving and loading DataFrame with pickle."""
    sample_df = pd.read_csv(StringIO("""GDPpercapita
        100
        150
        200
        """))
    file_path = f"{path_to_data}/output/gdp_test.obj"
    save_pickle(sample_df, file_path)
    loaded_df = load_pickle(file_path)

    pd.testing.assert_frame_equal(sample_df, loaded_df)


def test_plot_gdp_runs_without_error():
    """Check plot_gdp executes (we don’t validate the plot visually)."""
    sample_df = pd.read_csv(StringIO("""GDPpercapita
        100
        150
        200
        """))
    df_with_time = add_time_index(sample_df, start_year='2000', end_year='2002')
    plot_gdp(df_with_time, show=False)


def test_load_airpassenger_data_parses_dates():
    """Ensure dates are parsed and index is correct."""
    sample_csv = """Month,Passengers
        1949-01,112
        1949-02,118
        """
    df = load_airpassenger_data(StringIO(sample_csv))
    assert isinstance(df.index[0], pd.Timestamp)
    assert df.shape == (2, 1)
    assert "Passengers" in df.columns


def test_save_and_reload_csv_roundtrip():
    """Test saving and reloading preserves structure."""
    sample_csv = """Month,Passengers
        1949-01,112
        1949-02,118
        """
    df_original = pd.read_csv(StringIO(sample_csv), parse_dates=['Month'], index_col='Month')
    file_path = f"{path_to_data}/output/saved.csv"
    save_time_series_to_csv(df_original, file_path)
    df_reloaded = reload_time_series_from_csv(file_path)
    # Check content (not index — reloaded CSV doesn't parse dates)
    assert df_reloaded.shape == (2, 2)
    assert list(df_reloaded.columns) == ["Month", "Passengers"]
