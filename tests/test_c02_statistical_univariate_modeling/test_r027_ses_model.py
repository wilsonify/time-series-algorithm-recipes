# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
from c01_getting_started import path_to_data

def test_R027_ses_model():
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
    SES_model = SimpleExpSmoothing(actuals)
    SES_model = SES_model.fit(smoothing_level=0.8, optimized=False)

    # %%
    predictions_ses = SES_model.forecast(len(test_data.index))

    # %%
    plt.plot(train_data, color="black", label='Train')
    plt.plot(test_data, color="green", label='Test')
    plt.ylabel('Price-BTC')
    plt.xlabel('Date')
    plt.xticks(rotation=35)
    plt.title("SImple Exponential Smoothing (SES) modelpredictions")
    plt.plot(predictions_ses, color='red', label='Predictions')
    plt.legend()
    plt.show()

    # %%
    rmse_ses = np.sqrt(mean_squared_error(test_data["BTC-USD"].values, predictions_ses))
    print("RMSE: ", rmse_ses)
