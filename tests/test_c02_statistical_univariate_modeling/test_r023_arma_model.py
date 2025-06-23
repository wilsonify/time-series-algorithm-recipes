from c01_getting_started import path_to_data
from c02_statistical_univariate_modeling.r023_arma_model import (
    score_arma_model,
    plot_arma_model_predictions,
    predict_arma_model,
    fit_arma_model,
    plot_price_btc,
    train_test_split_btc,
    plot_btc_usd,
    read_btc
)



def test_r023_arma_model():
    filepath_or_buffer = f"{path_to_data}/input/btc.csv"
    btc_data = read_btc(filepath_or_buffer)
    plot_btc_usd(btc_data)
    test_data, train_data = train_test_split_btc(btc_data)
    plot_price_btc(test_data, train_data)
    actuals = train_data['BTC-USD']
    arma_model = fit_arma_model(actuals)
    predictions_arma, predictions_df = predict_arma_model(arma_model, test_data)
    plot_arma_model_predictions(predictions_arma, test_data, train_data)
    score_arma_model(predictions_df, test_data)
