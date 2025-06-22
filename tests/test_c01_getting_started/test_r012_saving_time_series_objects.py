from c01_getting_started import path_to_data
from c01_getting_started.r011a_reading_time_series_objects import load_airpassenger_data
from c01_getting_started.r012_saving_time_series_objects import save_time_series_to_csv


def test_test_r012_saving_time_series_objects():
    df1 = load_airpassenger_data(f"{path_to_data}/input/airpassengers.csv")
    save_time_series_to_csv(
        data=df1,
        csv_path=f"{path_to_data}/output/airpassengers.csv"
    )
    df2 = load_airpassenger_data(f"{path_to_data}/output/airpassengers.csv")
    assert df1.shape == df2.shape
    assert list(set(df1.columns)) == list(set(df2.columns))
