from c01_getting_started import path_to_data
from c01_getting_started.r015b_time_series_decomposition_multiplicative_model import (
    plot_decomp_air_passengers,
    read_air_passengers_data
)


def test_r015b_time_series_decomposition_multiplicative_model():
    air_passengers_data = read_air_passengers_data(f'{path_to_data}/input/airpax.csv')
    assert air_passengers_data.shape == (144, 4)
    decomp_air_passengers_data = plot_decomp_air_passengers(air_passengers_data, show=False)
    assert decomp_air_passengers_data.seasonal.shape == (144,)
