import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

from c01_getting_started import path_to_data

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

ts_diff = actuals - actuals.shift(periods=4)
ts_diff.dropna(inplace=True)

# %%
# checking for stationarity
from statsmodels.tsa.stattools import adfuller


def test_r024_arima_model():
    result = adfuller(ts_diff)
    pval = result[1]
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

    # %%

    lag_acf = acf(ts_diff, nlags=20)
    lag_pacf = pacf(ts_diff, nlags=20, method='ols')

    # %%
    # Ploting ACF:
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.stem(lag_acf)
    plt.axhline(y=0, linestyle='--', color='black')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts_diff)), linestyle='--', color='gray')
    plt.xticks(range(0, 22, 1))
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.title('Autocorrelation Function')

    # %%
    # Plotting PACF:
    plt.figure(figsize=(15, 5))
    plt.subplot(122)
    plt.stem(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='black')
    plt.axhline(y=-1.96 / np.sqrt(len(actuals)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(actuals)), linestyle='--', color='gray')
    plt.xlabel('Lag')
    plt.xticks(range(0, 22, 1))
    plt.ylabel('PACF')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()

    # %%
    ARIMA_model = ARIMA(actuals, order=(10, 4, 1))
    ARIMA_model = ARIMA_model.fit()

    # %%
    predictions = ARIMA_model.get_forecast(len(test_data.index))
    predictions_df = predictions.conf_int(alpha=0.05)
    predictions_df["Predictions"] = ARIMA_model.predict(start=
                                                        predictions_df.index[0], end=predictions_df.index[-1])
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
