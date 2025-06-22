import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from c01_getting_started import path_to_data

def test_r023_arma_model():
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
    ARMA_model = ARIMA(actuals, order=(1, 0, 1))
    ARMA_model = ARMA_model.fit()

    # %%
    predictions = ARMA_model.get_forecast(len(test_data.index))
    predictions_df = predictions.conf_int(alpha=0.05)
    predictions_df["Predictions"] = ARMA_model.predict(start=predictions_df.index[0], end=predictions_df.index[-1])
    predictions_df.index = test_data.index
    predictions_arma = predictions_df["Predictions"]

    # %%
    plt.plot(train_data, color="black", label='Train')
    plt.plot(test_data, color="green", label='Test')
    plt.ylabel('Price-BTC')
    plt.xlabel('Date')
    plt.xticks(rotation=35)
    plt.title("ARMA model predictions")
    plt.plot(predictions_arma, color="red", label='Predictions')
    plt.legend()
    plt.show()

    # %%
    rmse_arma = np.sqrt(mean_squared_error(test_data["BTC-USD"].values, predictions_df["Predictions"]))
    print("RMSE: ", rmse_arma)
