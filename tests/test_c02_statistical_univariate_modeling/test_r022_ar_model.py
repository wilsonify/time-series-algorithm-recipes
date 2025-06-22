import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot, pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

from c01_getting_started import path_to_data


class ARModelFMU:
    def __init__(self):
        self.params = None  # model weights
        self.lags = None  # list of lag indices
        self.initial_obs = None  # initial lagged values
        self.history = []  # buffer for current state
        self.current_time = 0  # simulation time step

    def fit(self, train_series):
        model_ar = AutoReg(train_series, lags=8).fit()
        print(model_ar.summary())
        endog = model_ar.model.endog.tolist()
        self.params = model_ar.params.tolist()
        self.lags = model_ar.model.ar_lags
        self.last_obs = endog[-max(model_ar.model.ar_lags):]
        self.history = list(endog)

    def create(self, params, lags, last_obs):
        """Initialize the model from saved state."""
        self.params = np.array(params)
        self.lags = lags
        self.initial_obs = list(last_obs)
        self.history = list(last_obs)
        self.current_time = 0

    def read(self):
        """Return the current observable state (last prediction)."""
        if len(self.history) == 0:
            return None
        return self.history[-1]

    def update(self, value):
        """Externally inject a new value (e.g. measurement override)."""
        self.history.append(value)

    def delete(self):
        """Clean up the model."""
        self.params = None
        self.lags = None
        self.initial_obs = None
        self.history = []
        self.current_time = 0

    def do_step(self, step_size=1):
        """
        Simulate the next `step_size` predictions.

        Returns:
            np.ndarray: Array of predicted values.
        """
        preds = []
        history = self.history.copy()

        for _ in range(step_size):
            if len(history) < len(self.lags):
                raise ValueError("Insufficient history to perform prediction.")

            lagged_vals = history[-len(self.lags):][::-1]
            yhat = self.params[0]  # intercept
            yhat += np.dot(self.params[1:], lagged_vals)
            preds.append(yhat)
            history.append(yhat)

        self.history = history
        self.current_time += step_size
        return np.array(preds)

    def reset(self):
        """Reset the model to its initial state."""
        self.history = list(self.initial_obs)
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


def test_r022_ar_model():
    filepath_or_buffer = f'{path_to_data}/input/opsd_germany_daily.csv'
    data = read_opsd_germany_daily(filepath_or_buffer)
    plot_opsd_germany_daily_pacf(data, show=False)
    test_df, train_df, eval_df = train_test_split_consumption(data)
    model_ar = ARModelFMU()
    model_ar.fit(train_df)
    model_ar.save(f"{path_to_data}/output/model_ar.json")
    model_ar2 = ARModelFMU()
    model_ar2.load(f"{path_to_data}/output/model_ar.json")
    plot_predictions_ar_model(data, model_ar2, test_df, show=False)
    plot_predictions_ar_model(data, model_ar2, eval_df, show=False)
