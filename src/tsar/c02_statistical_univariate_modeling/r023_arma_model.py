import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModelFMU:
    def __init__(self):
        self.model = None
        self.model_fit = None
        self.order = None
        self.train_data = None
        self.prediction_index = None
        self.initial_values = None
        self.current_time = 0

    def create(self, order, initial_train_data):
        """
        Initialize the ARIMA model structure with order and data.
        Parameters:
            order (tuple): ARIMA(p, d, q)
            initial_train_data (pd.Series): time-indexed training series
        """
        self.order = order
        self.train_data = initial_train_data.copy()
        self.initial_values = initial_train_data.copy()
        self.model = ARIMA(self.train_data, order=self.order)
        self.model_fit = self.model.fit()
        self.current_time = len(initial_train_data)

    def fit(self, train_data):
        """
        Shortcut for creating and fitting in one step with default ARMA(1,0,1).
        """
        self.create(order=(1, 0, 1), initial_train_data=train_data)

    def do_step(self, steps=1):
        """
        Forecast `steps` steps ahead, update internal time index.
        """
        if self.model_fit is None:
            raise RuntimeError("Model not initialized. Call `create()` or `fit()` first.")

        forecast = self.model_fit.get_forecast(steps=steps)
        pred_mean = forecast.predicted_mean
        self.current_time += steps
        self.prediction_index = pred_mean.index
        return pred_mean

    def read(self):
        """
        Return the most recent prediction or model state summary.
        """
        if self.model_fit is None:
            return None
        return self.model_fit.summary()

    def update(self, new_observation):
        """
        Append new observed value and refit the model incrementally.
        """
        self.train_data = pd.concat([self.train_data, pd.Series([new_observation])])
        self.model = ARIMA(self.train_data, order=self.order)
        self.model_fit = self.model.fit()

    def reset(self):
        """
        Reset to original training data.
        """
        self.train_data = self.initial_values.copy()
        self.model = ARIMA(self.train_data, order=self.order)
        self.model_fit = self.model.fit()
        self.current_time = len(self.initial_values)

    def delete(self):
        """
        Clear internal state.
        """
        self.model = None
        self.model_fit = None
        self.order = None
        self.train_data = None
        self.initial_values = None
        self.current_time = 0


def score_arma_model(predictions_df, test_data):
    rmse_arma = np.sqrt(mean_squared_error(test_data["BTC-USD"].values, predictions_df["Predictions"]))
    print("RMSE: ", rmse_arma)


def plot_arma_model_predictions(predictions_arma, test_data, train_data, show=True):
    plt.plot(train_data, color="black", label='Train')
    plt.plot(test_data, color="green", label='Test')
    plt.ylabel('Price-BTC')
    plt.xlabel('Date')
    plt.xticks(rotation=35)
    plt.title("ARMA model predictions")
    plt.plot(predictions_arma, color="red", label='Predictions')
    plt.legend()
    if show: plt.show()


def predict_arma_model(arma_model, test_data):
    predictions = arma_model.get_forecast(len(test_data.index))
    predictions_df = predictions.conf_int(alpha=0.05)
    predictions_df["Predictions"] = arma_model.predict(start=predictions_df.index[0], end=predictions_df.index[-1])
    predictions_df.index = test_data.index
    predictions_arma = predictions_df["Predictions"]
    return predictions_arma, predictions_df


def fit_arma_model(actuals):
    arma_model = ARIMA(actuals, order=(1, 0, 1))
    arma_model = arma_model.fit()
    return arma_model


def plot_price_btc(test_data, train_data, show=True):
    plt.plot(train_data, color="black", label='Train')
    plt.plot(test_data, color="green", label='Test')
    plt.ylabel('Price-BTC')
    plt.xlabel('Date')
    plt.xticks(rotation=35)
    plt.title("Train/Test split")
    if show: plt.show()


def train_test_split_btc(btc_data):
    train_data = btc_data[btc_data.index < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
    test_data = btc_data[btc_data.index > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
    print(train_data.shape)
    print(test_data.shape)
    return test_data, train_data


def plot_btc_usd(btc_data, show=True):
    plt.ylabel('Price-BTC')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.plot(btc_data.index, btc_data['BTC-USD'], )
    if show: plt.show()


def read_btc(filepath_or_buffer):
    btc_data = pd.read_csv(filepath_or_buffer)
    btc_data.index = pd.to_datetime(btc_data['Date'], format='%Y-%m-%d')
    del btc_data['Date']
    return btc_data
