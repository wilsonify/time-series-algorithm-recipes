from c01_getting_started import path_to_data
from c02_statistical_univariate_modeling.r021a_ma_model import plot_gdp_ma, plot_gpd, read_us_gdp_data


def test_r021a_ma_model():
    us_gdp_data = read_us_gdp_data(f'{path_to_data}/input/gdpus.csv')
    plot_gpd(us_gdp_data, show=False)
    plot_gdp_ma(us_gdp_data, show=False)
