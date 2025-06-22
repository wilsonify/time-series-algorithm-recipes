# %%

from c01_getting_started import path_to_data

from c01_getting_started.r014b_time_series_components_seasonality import (
    load_temperature_data,
    plot_minimum_temp,
    plot_monthly_seasonality,
    plot_yearly_distribution
)


def test_load_temperature_data():
    data = load_temperature_data(f'{path_to_data}/input/daily-min-temperatures.csv')
    assert data.shape == (3650,)


def test_r014b_time_series_components_seasonality():
    data = load_temperature_data(f'{path_to_data}/input/daily-min-temperatures.csv')
    plot_minimum_temp(data, show=False)
    plot_monthly_seasonality(data, year='1990', show=False)
    plot_yearly_distribution(data, show=False)
