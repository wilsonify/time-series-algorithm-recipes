import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot, pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller


class ARModelFMU:
    def __init__(self):
        self.params = None
        self.lags = None
        self.initial_obs = None
        self.history = None  # will be deque
        self.current_time = 0

    def fit(self, train_series):
        model_ar = AutoReg(train_series, lags=8).fit()
        print(model_ar.summary())
        endog = model_ar.model.endog.tolist()
        self.params = model_ar.params.tolist()
        self.lags = model_ar.model.ar_lags
        self.last_obs = endog[-max(self.lags):]
        self.history = deque(endog[-max(self.lags):], maxlen=max(self.lags))

    def read(self):
        print(f"params = {self.params}")
        print(f"lags = {self.lags}")
        print(f"initial_obs = {self.initial_obs}")
        print(f"history = {self.history}")
        print(f"current_time = {self.current_time}")

    def create(self, params, lags, last_obs):
        self.params = np.array(params)
        self.lags = lags
        self.initial_obs = list(last_obs)
        self.history = deque(last_obs, maxlen=max(lags))
        self.current_time = 0

    def update(self, value):
        self.history.append(value)

    def delete(self):
        self.params = None
        self.lags = None
        self.initial_obs = None
        self.history = None
        self.current_time = 0

    def do_step(self, step_size=1):
        preds = []
        history = deque(self.history, maxlen=self.history.maxlen)

        for _ in range(step_size):
            if len(history) < len(self.lags):
                raise ValueError("Insufficient history to perform prediction.")

            lagged_vals = list(history)[-len(self.lags):][::-1]
            yhat = self.params[0] + np.dot(self.params[1:], lagged_vals)
            preds.append(yhat)
            history.append(yhat)

        self.history = history
        self.current_time += step_size
        return np.array(preds)

    def reset(self):
        self.history = deque(self.initial_obs, maxlen=max(self.lags))
        self.current_time = 0

    def save(self, filepath):
        """
        Save AR model weights and last p observations for future predictions.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                "params": self.params,
                "lags": self.lags,
                "last_obs": self.last_obs
            }, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            model_data = json.load(f)
            self.params = model_data["params"]
            self.lags = model_data["lags"]
            self.last_obs = model_data["last_obs"]
            self.create(self.params, self.lags, self.last_obs)  # <-- important


def plot_predictions_ar_model(data, model_ar, test_df, show=True):
    predictions = model_ar.do_step(len(test_df))
    pyplot.plot(predictions)
    pyplot.plot(test_df, color='red')
    if show: plt.show()


def train_test_split_consumption(series):
    n = len(series)
    train_end = int(n * 0.8)
    test_end = int(n * 0.9)
    train_df = series.loc[:train_end, 'Consumption']
    test_df = series.loc[train_end:test_end, 'Consumption']
    eval_df = series.loc[test_end:, 'Consumption']
    return test_df, train_df, eval_df


def plot_opsd_germany_daily_pacf(data, show=True):
    data['Consumption'].plot()
    data_stationarity_test = adfuller(data['Consumption'], autolag='AIC')
    print("P-value: ", data_stationarity_test[1])
    plot_pacf(data['Consumption'], lags=25)
    if show: plt.show()


def read_opsd_germany_daily(filepath_or_buffer):
    data = pd.read_csv(filepath_or_buffer, sep=",")
    return data
