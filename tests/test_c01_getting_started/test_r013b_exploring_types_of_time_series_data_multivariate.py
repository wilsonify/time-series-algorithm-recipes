

from c01_getting_started import path_to_data
from c01_getting_started.r013b_exploring_types_of_time_series_data_multivariate import read_weather_data, \
    plot_weather_data


def test_test_r013b_exploring_types_of_time_series_data_multivariate():
    data1 = read_weather_data(f'{path_to_data}/input/raw.csv')
    print(data1.head(5))
    plot_weather_data(data1, show=False)
