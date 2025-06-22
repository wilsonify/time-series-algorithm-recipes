from c01_getting_started import path_to_data
from c01_getting_started.r014a_time_series_components_trends import read_shampoo_sales, plot_shampoo_sales


def test_r014a_time_series_components_trends():
    data = read_shampoo_sales(filepath_or_buffer=f'{path_to_data}/input/shampoo-sales.csv')
    plot_shampoo_sales(data, show=False)
