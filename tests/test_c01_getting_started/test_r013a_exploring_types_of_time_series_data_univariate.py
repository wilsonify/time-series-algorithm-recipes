from c01_getting_started import path_to_data
from c01_getting_started.r013a_exploring_types_of_time_series_data_univariate import load_temperature_data, \
    extract_series_column, plot_temperature_series


def test_test_r013a_exploring_types_of_time_series_data_univariate():
    df = load_temperature_data(f"{path_to_data}/input/airpassengers.csv")
    series = extract_series_column(df)
    plot_temperature_series(
        series=series,
        ylabel="Minimum Temp",
        title="Min temp in Southern Hemisphere From 1981 to 1990",
        show=False
    )
