# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pandas import read_csv, Grouper, DataFrame, concat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf, pacf
import warnings

# %%
us_gdp_data = pd.read_csv('../../data/GDPUS.csv', header=0)

# %%
date_rng = pd.date_range(start='1/1/1929', end='31/12/1991',freq='A')
print(date_rng)
us_gdp_data['TimeIndex'] = pd.DataFrame(date_rng,columns=['Year'])

# %%
plt.plot(us_gdp_data.TimeIndex, us_gdp_data.GDP)
plt.legend(loc='best')
plt.show()

# %%
mvg_avg_us_gdp = us_gdp_data.copy()
#calculating the rolling mean - with window 5
mvg_avg_us_gdp['moving_avg_forecast'] = us_gdp_data['GDP'].rolling(5).mean()

# %%
plt.plot(us_gdp_data['GDP'], label='US GDP')
plt.plot(mvg_avg_us_gdp['moving_avg_forecast'], label='USGDP MA(5)')
plt.legend(loc='best')
plt.show()

# %%
url='opsd_germany_daily.csv'
data = pd.read_csv(url,sep=",")
data['Consumption'].plot()

# %%
data_stationarity_test = adfuller(data['Consumption'],autolag='AIC')
print("P-value: ", data_stationarity_test[1])

# %%
pacf = plot_pacf(data['Consumption'], lags=25)

# %%
train_df = data['Consumption'][:len(data)-100]
test_df = data['Consumption'][len(data)-100:]

# %%
model_ar = AutoReg(train_df, lags=8).fit()

# %%
print(model_ar.summary())

# %%
predictions = model_ar.predict(start=len(train_df),
end=(len(data)-1), dynamic=False)

# %%
from matplotlib import pyplot
pyplot.plot(predictions)
pyplot.plot(test_df, color='red')

# %%

# %%
btc_data = pd.read_csv("btc.csv")
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
plt.plot(train_data, color = "black", label = 'Train')
plt.plot(test_data, color = "green", label = 'Test')
plt.ylabel('Price-BTC')
plt.xlabel('Date')
plt.xticks(rotation=35)
plt.title("Train/Test split")
plt.show()

# %%
actuals = train_data['BTC-USD']

# %%
ARMA_model = ARIMA(actuals, order = (1, 0, 1))
ARMA_model = ARMA_model.fit()

# %%
predictions = ARMA_model.get_forecast(len(test_data.index))
predictions_df = predictions.conf_int(alpha = 0.05)
predictions_df["Predictions"] = ARMA_model.predict(start = predictions_df.index[0], end = predictions_df.index[-1])
predictions_df.index = test_data.index
predictions_arma = predictions_df["Predictions"]

# %%
plt.plot(train_data, color = "black", label = 'Train')
plt.plot(test_data, color = "green", label = 'Test')
plt.ylabel('Price-BTC')
plt.xlabel('Date')
plt.xticks(rotation=35)
plt.title("ARMA model predictions")
plt.plot(predictions_arma, color="red", label = 'Predictions')
plt.legend()
plt.show()

# %%
rmse_arma = np.sqrt(mean_squared_error(test_data["BTC-USD"].values, predictions_df["Predictions"]))
print("RMSE: ",rmse_arma)

# %%
# differencing
ts_diff = actuals - actuals.shift(periods=4)
ts_diff.dropna(inplace=True)

# %%
# checking for stationarity
from statsmodels.tsa.stattools import adfuller
result = adfuller(ts_diff)
pval = result[1]
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# %%
from statsmodels.tsa.stattools import adfuller
lag_acf = acf(ts_diff, nlags=20)
lag_pacf = pacf(ts_diff, nlags=20, method='ols')

# %%
#Ploting ACF:
plt.figure(figsize = (15,5))
plt.subplot(121)
plt.stem(lag_acf)
plt.axhline(y = 0, linestyle='--',color='black')
plt.axhline(y = -1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y = 1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.xticks(range(0,22,1))
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.title('Autocorrelation Function')

# %%
#Plotting PACF:
plt.figure(figsize = (15,5))
plt.subplot(122)
plt.stem(lag_pacf)
plt.axhline(y = 0, linestyle = '--', color = 'black')
plt.axhline(y =-1.96/np.sqrt(len(actuals)), linestyle = '--',color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(actuals)),linestyle = '--',color = 'gray')
plt.xlabel('Lag')
plt.xticks(range(0,22,1))
plt.ylabel('PACF')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

# %%
ARIMA_model = ARIMA(actuals, order = (10, 4, 1))
ARIMA_model = ARIMA_model.fit()

# %%
predictions = ARIMA_model.get_forecast(len(test_data.index))
predictions_df = predictions.conf_int(alpha = 0.05)
predictions_df["Predictions"] = ARIMA_model.predict(start =
predictions_df.index[0], end = predictions_df.index[-1])
predictions_df.index = test_data.index
predictions_arima = predictions_df["Predictions"]


# %%
plt.plot(train_data, color = "black", label = 'Train')
plt.plot(test_data, color = "green", label = 'Test')
plt.ylabel('Price-BTC')
plt.xlabel('Date')
plt.xticks(rotation=35)
plt.title("ARIMA model predictions")
plt.plot(predictions_arima, color="red", label = 'Predictions')
plt.legend()
plt.show()

# %%
rmse_arima = np.sqrt(mean_squared_error(test_data["BTC-USD"].
values, predictions_df["Predictions"]))
print("RMSE: ",rmse_arima)


# %%
def arima_model_evaluate(train_actuals, test_data, order):
    # Model initalize and fit
    ARIMA_model = ARIMA(actuals, order = order)
    ARIMA_model = ARIMA_model.fit()
    # Getting the predictions
    predictions = ARIMA_model.get_forecast(len(test_data.index))
    predictions_df = predictions.conf_int(alpha = 0.05)
    predictions_df["Predictions"] = ARIMA_model.predict(start = predictions_df.index[0], end = predictions_df.index[-1])
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
                arima_order = (p,d,q)
                rmse = arima_model_evaluate(train_actuals,test_data, arima_order)
                if rmse < best_rmse:
                    best_rmse, best_config = rmse, arima_order
                print('ARIMA%s RMSE=%.3f' % (arima_order,rmse))
    print('Best Configuration: ARIMA%s , RMSE=%.3f' % (best_config, best_rmse))
    return best_config


# %%
p_values = range(0, 4)
d_values = range(0, 4)
q_values = range(0, 4)
warnings.filterwarnings("ignore")
best_config = evaluate_models(actuals,test_data, p_values,d_values, q_values)

# %%
ARIMA_model = ARIMA(actuals, order = best_config)
ARIMA_model = ARIMA_model.fit()

# %%
predictions = ARIMA_model.get_forecast(len(test_data.index))
predictions_df = predictions.conf_int(alpha = 0.05)
predictions_df["Predictions"] = ARIMA_model.predict(start =
predictions_df.index[0], end = predictions_df.index[-1])
predictions_df.index = test_data.index
predictions_arima = predictions_df["Predictions"]

# %%
plt.plot(train_data, color = "black", label = 'Train')
plt.plot(test_data, color = "green", label = 'Test')
plt.ylabel('Price-BTC')
plt.xlabel('Date')
plt.xticks(rotation=35)
plt.title("ARIMA model predictions")
plt.plot(predictions_arima, color="red", label = 'Predictions')
plt.legend()
plt.show()

# %%
rmse_arima = np.sqrt(mean_squared_error(test_data["BTC-USD"].
values, predictions_df["Predictions"]))
print("RMSE: ",rmse_arima)

# %%

# %%
SARIMA_model = SARIMAX(actuals, order = (1, 2, 0), seasonal_order=(2,2,2,12))
SARIMA_model = SARIMA_model.fit()

# %%
predictions = SARIMA_model.get_forecast(len(test_data.index))
predictions_df = predictions.conf_int(alpha = 0.05)
predictions_df["Predictions"] = SARIMA_model.predict(start =predictions_df.index[0], end = predictions_df.index[-1])
predictions_df.index = test_data.index
predictions_sarima = predictions_df["Predictions"]

# %%
plt.plot(train_data, color = "black", label = 'Train')
plt.plot(test_data, color = "green", label = 'Test')
plt.ylabel('Price-BTC')
plt.xlabel('Date')
plt.xticks(rotation=35)
plt.title("SARIMA model predictions")
plt.plot(predictions_sarima, color="red", label ='Predictions')
plt.legend()
plt.show()

# %%
rmse_sarima = np.sqrt(mean_squared_error(test_data["BTC-USD"].values, predictions_df["Predictions"]))
print("RMSE: ",rmse_sarima)

# %%

# %%
SES_model = SimpleExpSmoothing(actuals)
SES_model = SES_model.fit(smoothing_level=0.8,optimized=False)

# %%
predictions_ses = SES_model.forecast(len(test_data.index))

# %%
plt.plot(train_data, color = "black", label = 'Train')
plt.plot(test_data, color = "green", label = 'Test')
plt.ylabel('Price-BTC')
plt.xlabel('Date')
plt.xticks(rotation=35)
plt.title("SImple Exponential Smoothing (SES) model
predictions")
plt.plot(predictions_ses, color='red', label = 'Predictions')
plt.legend()
plt.show()

# %%
rmse_ses = np.sqrt(mean_squared_error(test_data["BTC-USD"].
values, predictions_ses))
print("RMSE: ",rmse_ses)

# %%
HW_model = ExponentialSmoothing(actuals, trend='add')
HW_model = HW_model.fit()

# %%
predictions_hw = HW_model.forecast(len(test_data.index))


# %%
plt.plot(train_data, color = "black", label = 'Train')
plt.plot(test_data, color = "green", label = 'Test')
plt.ylabel('Price-BTC')
plt.xlabel('Date')
plt.xticks(rotation=35)
plt.title("HW model predictions")
plt.plot(predictions_hw, color='red', label = 'Predictions')
plt.legend()
plt.show()

# %%
rmse_hw = np.sqrt(mean_squared_error(test_data["BTC-USD"].
values, predictions_hw))
print("RMSE: ",rmse_hw)
