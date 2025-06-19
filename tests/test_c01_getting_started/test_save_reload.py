# =======================
# CSV Save/Reload Tests
# =======================
from io import StringIO

import pandas as pd

from c01_getting_started import path_to_data
from c01_getting_started.r012_saving_time_series_objects import reload_time_series_from_csv, save_time_series_to_csv
from tests.test_c01_getting_started.test_air_passenger import sample_air_csv


def test_save_and_reload_csv_roundtrip():
    df = pd.read_csv(StringIO(sample_air_csv), parse_dates=['Month'], index_col='Month')
    path = f"{path_to_data}/output/saved.csv"
    save_time_series_to_csv(df, path)
    reloaded = reload_time_series_from_csv(path)
    assert reloaded.shape == (2, 2)
    assert "Month" in reloaded.columns
    assert "Passengers" in reloaded.columns
