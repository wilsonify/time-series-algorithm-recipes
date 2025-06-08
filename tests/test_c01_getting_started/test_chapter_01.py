import pandas as pd

from c01_getting_started import path_to_data
from c01_getting_started.r011a_reading_time_series_objects import load_airpassenger_data


def test_load_airpassenger_data():
    df = load_airpassenger_data(f"{path_to_data}/input/airpassengers.csv")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["#Passengers"]
    assert pd.Timestamp("1949-01-01") in df.index
    assert df.loc["1949-01-01", "#Passengers"] == 112
