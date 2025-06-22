import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from c01_getting_started import path_to_data


def test_r025_grid_search_tuning_for_arima():
    # %%

    # %%
    btc_data = pd.read_csv(f"{path_to_data}/input/btc.csv")
    print(btc_data.head())

    # %%
    btc_data.index = pd.to_datetime(btc_data['Date'],
                                    format='%Y-%m-%d')
    del btc_data['Date']

    # %%
    plt.ylabel('Price-BTC')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.plot(btc_data.index, btc_data['BTC-USD'], )

    # %%
    train_data = btc_data[btc_data.index < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
    test_data = btc_data[btc_data.index > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
    print(train_data.shape)
    print(test_data.shape)

    # %%
    plt.plot(train_data, color="black", label='Train')
    plt.plot(test_data, color="green", label='Test')
    plt.ylabel('Price-BTC')
    plt.xlabel('Date')
    plt.xticks(rotation=35)
    plt.title("Train/Test split")
    plt.show()

    # %%
    actuals = train_data['BTC-USD']

    # %%
    def arima_model_evaluate(train_actuals, test_data, order):
        # Model initalize and fit
        ARIMA_model = ARIMA(train_actuals, order=order)
        ARIMA_model = ARIMA_model.fit()
        # Getting the predictions
        predictions = ARIMA_model.get_forecast(len(test_data.index))
        predictions_df = predictions.conf_int(alpha=0.05)
        predictions_df["Predictions"] = ARIMA_model.predict(start=predictions_df.index[0], end=predictions_df.index[-1])
        predictions_df.index = test_data.index
        predictions_arima = predictions_df["Predictions"]
        # calculate RMSE score
        rmse_score = np.sqrt(mean_squared_error(test_data["BTC-USD"].values, predictions_df["Predictions"]))
        return rmse_score

    # %%
    def evaluate_models(train_actuals, test_data, list_p_values,
                        list_d_values, list_q_values):
        best_rmse, best_config = float("inf"), None
        for p in list_p_values:
            for d in list_d_values:
                for q in list_q_values:
                    arima_order = (p, d, q)
                    rmse = arima_model_evaluate(train_actuals, test_data, arima_order)
                    if rmse < best_rmse:
                        best_rmse, best_config = rmse, arima_order
                    print('ARIMA%s RMSE=%.3f' % (arima_order, rmse))
        print('Best Configuration: ARIMA%s , RMSE=%.3f' % (best_config, best_rmse))
        return best_config

    # %%
    p_values = range(0, 4)
    d_values = range(0, 4)
    q_values = range(0, 4)
    warnings.filterwarnings("ignore")
    best_config = evaluate_models(actuals, test_data, p_values, d_values, q_values)

    # %%
    ARIMA_model = ARIMA(actuals, order=best_config)
    ARIMA_model = ARIMA_model.fit()

    # %%
    predictions = ARIMA_model.get_forecast(len(test_data.index))
    predictions_df = predictions.conf_int(alpha=0.05)
    predictions_df["Predictions"] = ARIMA_model.predict(
        start=predictions_df.index[0],
        end=predictions_df.index[-1]
    )
    predictions_df.index = test_data.index
    predictions_arima = predictions_df["Predictions"]

    # %%
    plt.plot(train_data, color="black", label='Train')
    plt.plot(test_data, color="green", label='Test')
    plt.ylabel('Price-BTC')
    plt.xlabel('Date')
    plt.xticks(rotation=35)
    plt.title("ARIMA model predictions")
    plt.plot(predictions_arima, color="red", label='Predictions')
    plt.legend()
    plt.show()

    # %%
    rmse_arima = np.sqrt(mean_squared_error(test_data["BTC-USD"].
                                            values, predictions_df["Predictions"]))
    print("RMSE: ", rmse_arima)
