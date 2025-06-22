import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class MAModelFMU:
    def __init__(self):
        self.window = None
        self.history = None
        self.initial_obs = None
        self.current_time = 0

    def create(self, window, last_obs):
        """
        Initialize the model with window size and past observations.
        """
        self.window = window
        self.initial_obs = list(last_obs)
        self.history = deque(last_obs, maxlen=window)
        self.current_time = 0

    def read(self):
        """Render the model in a human-readable format"""
        print(f"window = {self.window}")
        print(f"history = {self.history}")
        print(f"initial_obs = {self.initial_obs}")
        print(f"current_time = {self.current_time}")

    def update(self, value):
        """Externally provide a new observation."""
        self.history.append(value)

    def do_step(self, step_size=1):
        """
        Predict next `step_size` values using moving average forecast.
        """
        preds = []
        temp_history = deque(self.history, maxlen=self.window)

        for _ in range(step_size):
            if len(temp_history) < self.window:
                raise ValueError("Insufficient history to perform prediction.")
            yhat = np.mean(temp_history)
            preds.append(yhat)
            temp_history.append(yhat)

        self.history = temp_history
        self.current_time += step_size
        return np.array(preds)

    def reset(self):
        """Reset the model to its initial state."""
        self.history = deque(self.initial_obs, maxlen=self.window)
        self.current_time = 0

    def delete(self):
        """Clean up the model state."""
        self.window = None
        self.history = None
        self.initial_obs = None
        self.current_time = 0

    def save(self, filepath):
        """
        Save model weights and last p observations for future predictions.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                "window": self.window,
                "history": self.history,
                "initial_obs": self.initial_obs,
                "current_time": self.current_time
            }, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            model_data = json.load(f)
            self.window = model_data["window"]
            self.initial_obs = model_data["initial_obs"]
            self.history = model_data["history"]
            self.current_time = model_data["current_time"]
            self.create(window=model_data["window"], last_obs=model_data["last_obs"])


def plot_gdp_ma(us_gdp_data, show=True):
    mvg_avg_us_gdp = us_gdp_data.copy()
    # calculating the rolling mean - with window 5
    mvg_avg_us_gdp['moving_avg_forecast'] = us_gdp_data['GDP'].rolling(5).mean()
    plt.plot(us_gdp_data['GDP'], label='US GDP')
    plt.plot(mvg_avg_us_gdp['moving_avg_forecast'], label='USGDP MA(5)')
    plt.legend(loc='best')
    if show: plt.show()


def plot_gpd(us_gdp_data, show=True):
    plt.plot(us_gdp_data.TimeIndex, us_gdp_data.GDP)
    plt.legend(loc='best')
    if show: plt.show()


def read_us_gdp_data(filepath_or_buffer):
    us_gdp_data = pd.read_csv(filepath_or_buffer=filepath_or_buffer, header=0)
    date_rng = pd.date_range(start='1/1/1929', end='31/12/1991', freq='A')
    us_gdp_data['TimeIndex'] = pd.DataFrame(date_rng, columns=['Year'])
    return us_gdp_data
