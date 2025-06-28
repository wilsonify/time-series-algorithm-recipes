from c01_getting_started import path_to_data
from c02_statistical_univariate_modeling.r021a_ma_model import (
    plot_gdp_ma,
    read_us_gdp_data,
    MAModelFMU,
    plot_gdp,
    plot_predictions_ma_model
)


def test_r021a_ma_model():
    """Quick test for plotting and model structure."""
    data = read_us_gdp_data(f'{path_to_data}/input/gdpus.csv')
    gdp_series = data["GDP"].dropna()

    # Initialize and save model
    model_ma = MAModelFMU()
    model_ma.fit(gdp_series)
    model_ma.read()
    model_ma.save(f"{path_to_data}/output/model_ma.json")

    # Load and test new model instance
    model_ma2 = MAModelFMU()
    model_ma2.load(f"{path_to_data}/output/model_ma.json")
    model_ma2.read()

    # Visual tests
    plot_gdp(data, show=False)
    plot_predictions_ma_model(model_ma, gdp_series, show=False)
    plot_predictions_ma_model(model_ma2, gdp_series, show=False)
    plot_gdp_ma(data, show=False)
