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
# #!pip install fbprophet

# %%
import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
from sklearn.model_selection import train_test_split
import plotly.offline as py
import matplotlib.pyplot as plt
py.init_notebook_mode()
# %matplotlib inline

# %%
df = pd.read_csv("./data/avocado.csv").drop(columns=["Unnamed: 0"])
df.head()

# %%
df.Date.nunique()

# %%
train_df = pd.DataFrame()
train_df['ds'] = pd.to_datetime(df["Date"])
train_df['y'] = df.iloc[:,1]
train_df.head()

# %%
# Initializing basic prophet model:
basic_prophet_model = Prophet()
basic_prophet_model.fit(train_df)

# %%
# Creating future dataframe for forecast:
future_df = basic_prophet_model.make_future_dataframe(include_history=True,periods=300)
future_df.tail()

# %%
future_df.head()

# %%
# Getting the forecast
forecast_df = basic_prophet_model.predict(future_df)

# %%
plot1 = basic_prophet_model.plot(forecast_df)

# %%
# to view  the forecast components
plot2 = basic_prophet_model.plot_components(forecast_df)

# %% [markdown]
# ## Change points

# %%
# Prophet detects changepoints by first specifying a large number of potential changepoints at which the rate is allowed to change
# Plotting the changepoints
plot3 = basic_prophet_model.plot(forecast_df)
adding_changepoints = add_changepoints_to_plot(plot3.gca(), basic_prophet_model, forecast_df)

# %%
# printing the change points:
basic_prophet_model.changepoints

# %%
# checking the magnitude of the change points:
deltas = basic_prophet_model.params['delta'].mean(0)
plot4 = plt.figure(facecolor='w')
ax = plot4.add_subplot(111)
ax.bar(range(len(deltas)), deltas)
ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
ax.set_ylabel('Rate change')
ax.set_xlabel('Potential changepoint')
plot4.tight_layout()

# %%
# setting the n_changepoints as hyperparameter:
prophet_model_changepoint= Prophet(n_changepoints=20, yearly_seasonality=True)
# getting the forecast
forecast_df_changepoint = prophet_model_changepoint.fit(train_df).predict(future_df)
# plotting the forecast with change points
plot5 = prophet_model_changepoint.plot(forecast_df_changepoint)
adding_changepoints = add_changepoints_to_plot(plot5.gca(), prophet_model_changepoint, forecast_df_changepoint)

# %%
# setting the changepoint_range as hyperparameter:
prophet_model_changepoint2 = Prophet(changepoint_range=0.9, yearly_seasonality=True)
# getting the forecast
forecast_df_changepoint2 = prophet_model_changepoint2.fit(train_df).predict(future_df)
# plotting the forecast with change points
plot6 = prophet_model_changepoint2.plot(forecast_df_changepoint2)
adding_changepoints = add_changepoints_to_plot(plot5.gca(), prophet_model_changepoint2, forecast_df_changepoint2)

# %% [markdown]
# ## Adjusting trend

# %%
# setting the changepoint_prior_scale as hyperparameter:
prophet_model_trend = Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.08)
# getting the forecast
forecast_df_trend = prophet_model_trend.fit(train_df).predict(future_df)
# plotting the forecast with change points
plot7 = prophet_model_trend.plot(forecast_df_trend)
adding_changepoints = add_changepoints_to_plot(plot7.gca(), prophet_model_trend, forecast_df_trend)

# %%
# setting the changepoint_prior_scale as hyperparameter:
prophet_model_trend2 = Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.001)
# getting the forecast
forecast_df_trend2 = prophet_model_trend2.fit(train_df).predict(future_df)
# plotting the forecast with change points
plot8 = prophet_model_trend2.plot(forecast_df_trend2)
adding_changepoints = add_changepoints_to_plot(plot8.gca(), prophet_model_trend2, forecast_df_trend2)

# %% [markdown]
# ## Holidays

# %%
# creating a custom holidays dataframe
holidays_df = pd.DataFrame({
  'holiday': 'avocado season',
  'ds': pd.to_datetime(['2014-07-31', '2014-09-16', 
                        '2015-07-31', '2015-09-16',
                        '2016-07-31', '2016-09-16',
                        '2017-07-31', '2017-09-16',
                       '2018-07-31', '2018-09-16',
                        '2019-07-31', '2019-09-16']),
  'lower_window': -1,
  'upper_window': 0,
})

# %%
# Initializing prophet model with holidays dataframe:
prophet_model_holiday = Prophet(holidays=holidays_df)
prophet_model_holiday.fit(train_df)
# Creating future dataframe for forecast:
future_df = prophet_model_holiday.make_future_dataframe(periods=12, freq = 'm')
# Getting the forecast
forecast_df = prophet_model_holiday.predict(future_df)
prophet_model_holiday.plot(forecast_df)

# %% [markdown]
# ## Adding multiple regressors

# %%
# Label encoding type column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.iloc[:,10] = le.fit_transform(df.iloc[:,10])
df.head(2)

# %%
data = df[['Date', 'Total Volume', '4046', '4225', '4770', 'Small Bags', 'type']]
data.rename(columns={'Date':'ds'},inplace=True)
data['y'] = df.iloc[:,1]

# %%
# train-test split
train_df = data[:18000]
test_df = data[18000:]

# %%
#Initializing Prophet model and adding additional Regressor
prophet_model_regressor = Prophet()
prophet_model_regressor.add_regressor('type')
prophet_model_regressor.add_regressor('Total Volume')
prophet_model_regressor.add_regressor('4046')
prophet_model_regressor.add_regressor('4225')
prophet_model_regressor.add_regressor('4770')
prophet_model_regressor.add_regressor('Small Bags')

# %%
# Fitting the data
prophet_model_regressor.fit(train_df)
future_df = prophet_model_regressor.make_future_dataframe(periods=249)

# %%
# forecast the data on test
forecast_df = prophet_model_regressor.predict(test_df)
prophet_model_regressor.plot(forecast_df)

# %%
#VAR

# %%
#import all the required libraries
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR

# %%
#read data
var_data = pd.read_excel('../data/AirQualityUCI.xlsx', parse_dates=[['Date', 'Time']])
var_data.head()

# %%
var_data['Date_Time'] = pd.to_datetime(var_data.Date_Time ,
format = '%d/%m/%Y %H.%M.%S')
var_data1 = var_data.drop(['Date_Time'], axis=1)
var_data1.index = var_data.Date_Time
var_data1.head()

# %%
#missing value treatment
cols = var_data1.columns
for j in cols:
    for i in range(0,len(var_data1)):
        if var_data1[j][i] == -200:
            var_data1[j][i] = var_data1[j][i-1]

# %%
#checking stationarity
from statsmodels.tsa.vector_ar.vecm import coint_johansen
#since the test works for only 12 variables, I have randomly dropped
#in the next iteration, I would drop another and check the eigenvalues
test = var_data1.drop([ 'CO(GT)'], axis=1)
coint_johansen(test,-1,1).eig

# %%
#creating the train and validation set
train_data = var_data1[:int(0.8*(len(var_data1)))]
valid_data = var_data1[int(0.8*(len(var_data1))):]

# %%
##fit the model
from statsmodels.tsa.vector_ar.var_model import VAR
var_model = VAR(endog=train_data)
var_model_fit = var_model.fit()
# make prediction on validation
pred = var_model_fit.forecast(var_model_fit.endog,
steps=len(valid_data))
pred

# %%
##converting predictions to dataframe
pred1 = pd.DataFrame(index=range(0,len(pred)),columns=[cols])
for j in range(0,13):
    for i in range(0, len(pred1)):
        pred1.iloc[i][j] = pred[i][j]
pred1

# %%
##check rmse
for i in cols:
    print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred1[i], valid_data[i])))

# %%
#Auto-Arima

# %%
#import all the required libraries
import pandas as pd
from pmdarima.arima import auto_arima
from pmdarima.arima import ADFTest
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

# %%
#read data
auto_arima_data = pd.read_csv('auto_arima_data.txt')
auto_arima_data.head()

# %%
#check missing values
auto_arima_data.isnull().sum()

# %%
#check datatype
auto_arima_data.info()

# %%
#convert object to datatime and set index
auto_arima_data['Month'] = pd.to_datetime(auto_arima_data['Month'])
auto_arima_data.set_index('Month',inplace=True)
auto_arima_data.head()

# %%
#line plot to understand the pattern
auto_arima_data.plot()

# %%
#Stationarity check
stationary_test = ADFTest(alpha= 0.05)
stationary_test.should_diff(auto_arima_data)

# %%
#train test split and plot
train_data = auto_arima_data[:85]
test_data = auto_arima_data[-20:]
plt.plot(train_data)
plt.plot(test_data)

# %%
#model building with parameters
auto_arima_model = auto_arima(train_data, start_p = 0, d=1,
start_q = 0, max_p = 5, max_d = 5,max_q= 5, start_P = 0, D=1,
start_Q = 0, max_P = 5, max_D = 5,
max_Q= 5, m=12, seasonal = True,
error_action = 'warn', trace = True, supress_warnings= True,
stepwise = True, random_state =20,
n_fits = 50)

# %%
#model summary
auto_arima_model.summary()

# %%
#forecasting on test set
pred = pd.DataFrame(auto_arima_model.predict(n_periods = 20),
index = test_data.index)
pred.columns= ['pred_sales']
#plot
plt.figure(figsize=(8,5))
plt.plot(train_data, label = "Training data")
plt.plot(test_data, label = "Test data")
plt.plot(pred, label = "Predicted data")
plt.legend()
plt.show()

# %%
#Evaluating using r square score
test_data['prediction'] = pred
r2_score(test_data['Champagne sales'],test_data['prediction'])

# %%
