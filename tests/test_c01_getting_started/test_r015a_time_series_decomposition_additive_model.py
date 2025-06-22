from c01_getting_started import path_to_data
from c01_getting_started.r015a_time_series_decomposition_additive_model import (
    read_retail_turnover_data,
    plot_turnover,
    plot_decomp_turn_over
)


def test_r015a_time_series_decomposition_additive_model():
    turn_over_data = read_retail_turnover_data(f'{path_to_data}/input/retailturnover.csv')
    plot_turnover(turn_over_data, show=False)
    plot_decomp_turn_over(turn_over_data, show=False)
