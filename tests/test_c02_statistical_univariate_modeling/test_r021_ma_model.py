from logging.config import dictConfig

from tsar import LOGGING_CONFIG
from tsar.c01_getting_started import path_to_data
from tsar.c02_statistical_univariate_modeling.r021_ma_model import read_us_gdp_data, MAModelTracker, MAModelFMU


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
    tr = MAModelTracker()
    tr.plot_gdp(data, show=False)
    tr.plot_predictions_ma_model(model_ma, gdp_series, show=False)
    tr.plot_predictions_ma_model(model_ma2, gdp_series, show=False)
    tr.plot_gdp_ma(data, show=False)


def test_demo():
    dictConfig(LOGGING_CONFIG)
    df = read_us_gdp_data(f"{path_to_data}/input/gdpus.csv")
    tracker = MAModelTracker()
    # tracker.plot(df, f"{path_to_data}/output/ma_model_tracker.png")
    tracker.animate(df, f"{path_to_data}/output/ma_model_tracker.gif")
