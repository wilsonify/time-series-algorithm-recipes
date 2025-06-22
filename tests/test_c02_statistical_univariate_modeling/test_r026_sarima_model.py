# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from c01_getting_started import path_to_data

def test_r026_sarima_model():
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
    SARIMA_model = SARIMAX(actuals, order=(1, 2, 0), seasonal_order=(2, 2, 2, 12))
    SARIMA_model = SARIMA_model.fit()

    # %%
    predictions = SARIMA_model.get_forecast(len(test_data.index))
    predictions_df = predictions.conf_int(alpha=0.05)
    predictions_df["Predictions"] = SARIMA_model.predict(start=predictions_df.index[0], end=predictions_df.index[-1])
    predictions_df.index = test_data.index
    predictions_sarima = predictions_df["Predictions"]

    # %%
    plt.plot(train_data, color="black", label='Train')
    plt.plot(test_data, color="green", label='Test')
    plt.ylabel('Price-BTC')
    plt.xlabel('Date')
    plt.xticks(rotation=35)
    plt.title("SARIMA model predictions")
    plt.plot(predictions_sarima, color="red", label='Predictions')
    plt.legend()
    plt.show()

    # %%
    rmse_sarima = np.sqrt(mean_squared_error(test_data["BTC-USD"].values, predictions_df["Predictions"]))
    print("RMSE: ", rmse_sarima)
