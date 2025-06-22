# %%

from c01_getting_started import path_to_data
from c01_getting_started.r014c_time_series_components_seasonality_contd import (
    read_tractor_sales_data,
    plot_tractor_sales,
    boxplot_tractor_sales_one_year
)


def test_r014c_time_series_components_seasonality_contd():
    tractor_sales_data = read_tractor_sales_data(f"{path_to_data}/input/tractor_salessales.csv")
    assert tractor_sales_data.shape == (144,)
    plot_tractor_sales(tractor_sales_data, show=False)
    boxplot_tractor_sales_one_year(tractor_sales_data, show=False)
