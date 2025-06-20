from c01_getting_started import path_to_data
from c01_getting_started.r011b_reading_time_series_objects import load_gdp_data, add_time_index, plot_gdp


def test_test_r011b_reading_time_series_objects():
    csv_path = f"{path_to_data}/input/gdp_india.csv"
    df = load_gdp_data(csv_path)
    df = add_time_index(df, start_year='1960', end_year='2017')
    plot_gdp(df, x_col='TimeIndex', y_col='GDPpercapita', show=False)
