from io import StringIO

import pandas as pd
import pytest
from matplotlib import pyplot as plt

from c01_getting_started import path_to_data, save_pickle, load_pickle
from c01_getting_started.r011a_reading_time_series_objects import load_airpassenger_data, plot_passenger_data
from c01_getting_started.r011b_reading_time_series_objects import plot_gdp, add_time_index, load_gdp_data


@pytest.fixture
def sample_df():
    """Provides a sample DataFrame mimicking GDP data."""
    MOCK_CSV = """GDPpercapita
    100
    150
    200
    """
    return pd.read_csv(StringIO(MOCK_CSV))


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


def test_load_gdp_data_from_stringio(monkeypatch):
    """Test loading GDP data using a mock path with StringIO."""

    def mock_open(*args, **kwargs):
        return StringIO(MOCK_CSV)

    monkeypatch.setattr("builtins.open", mock_open)
    df = load_gdp_data("mock/path/to/csv.csv")
    assert not df.empty
    assert "GDPpercapita" in df.columns
    assert df.shape == (3, 1)


def test_add_time_index(sample_df):
    """Ensure TimeIndex column is correctly added."""
    df_with_time = add_time_index(sample_df, start_year='2000', end_year='2002')
    assert "TimeIndex" in df_with_time.columns
    assert pd.api.types.is_datetime64_any_dtype(df_with_time["TimeIndex"])
    assert len(df_with_time["TimeIndex"]) == 3
    assert df_with_time.iloc[0]["TimeIndex"].year == 2000


def test_pickle_roundtrip(tmp_path, sample_df):
    """Test saving and loading DataFrame with pickle."""
    file_path = tmp_path / "gdp_test.obj"
    save_pickle(sample_df, file_path)
    loaded_df = load_pickle(file_path)

    pd.testing.assert_frame_equal(sample_df, loaded_df)


def test_plot_gdp_runs_without_error(sample_df):
    """Check plot_gdp executes (we don’t validate the plot visually)."""
    df_with_time = add_time_index(sample_df, start_year='2000', end_year='2002')
    plot_gdp(df_with_time, show=False)
