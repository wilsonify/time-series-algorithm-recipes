import json
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from c01_getting_started import path_to_data


class ModelStateError(Exception):
    """Raised when the model is used before initialization."""


class MAModelFMU:
    """
    A strict Moving Average forecasting model.
    Enforces initialization before use.
    """

    def __init__(self):
        self.window: int = -1
        self.initial_obs: float = 0
        self.last_obs: float = 0
        self.history: deque = deque([0, 0])
        self.current_time: int = 0
        self.initialized: bool = False

    def create(self, window: int, obs: List[float]):
        """Initialize the model with a fixed-size window and past observations."""
        assert window > 0, "Window size must be positive."
        self.window = window
        self.history = deque(list(obs), maxlen=window)
        self.initial_obs = self.history[0]
        self.last_obs = self.history[-1]
        self.current_time = 0
        self.initialized = True

    def fit(self, series: pd.Series, window: int = 5):
        """A convenience method to fit from a pandas Series."""
        self.create(
            window=window,
            obs=series.dropna().iloc[-window:].tolist()
        )

    def read(self):
        """Print current model state."""
        print(f"Window Size           : {self.window}")
        print(f"Current History       : {list(self.history)}")
        print(f"Initial Observations  : {self.initial_obs}")
        print(f"Current Time Step     : {self.current_time}")

    def update(self, value: float):
        """Append a new observation to the history."""
        assert self.initialized, "Model must be initialized using `create` or `fit`."
        self.history.append(value)

    def do_step(self) -> float:
        """Take exactly one step. Return the next predicted value."""
        assert self.initialized, "Model must be initialized using `create` or `fit`."
        yhat = float(np.mean(self.history))
        self.history.append(yhat)
        self.current_time += 1
        return yhat

    def simulate(self, nsteps: int = 100) -> np.ndarray:
        """Simulate forward n steps using repeated do_step."""
        assert self.initialized, "Model must be initialized using `create` or `fit`."
        predictions = np.empty(nsteps)
        for i in range(nsteps):
            predictions[i] = self.do_step()
        return predictions

    def reset(self):
        """Reset the model to its original state."""
        assert self.initialized, "Model must be initialized using `create` or `fit`."
        self.history = deque([self.initial_obs], maxlen=self.window)
        self.current_time = 0

    def save(self, filepath: str):
        """Persist model state to a JSON file."""
        assert self.initialized, "Model must be initialized using `create` or `fit`."
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                "window": self.window,
                "history": list(self.history),
                "current_time": self.current_time,
                "initialized": self.initialized
            }, f)

    def load(self, filepath: str):
        """Load model state from a JSON file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        self.create(window=model_data["window"], obs=model_data["history"])
        self.current_time = model_data.get("current_time", 0)


# === Plotting & Utilities === #

def plot_gdp(us_gdp_data: pd.DataFrame, title: str = "US GDP Over Time", show: bool = True):
    """Plot raw GDP data."""
    plt.figure(figsize=(10, 4))
    plt.plot(us_gdp_data["TimeIndex"], us_gdp_data["GDP"], label='US GDP')
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("GDP")
    plt.legend(loc='best')
    plt.grid(True)
    if show: plt.show()


def plot_gdp_ma(us_gdp_data: pd.DataFrame, window: int = 5, show: bool = True):
    """Plot GDP with moving average overlay."""
    mvg_avg = us_gdp_data["GDP"].rolling(window=window).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(us_gdp_data["TimeIndex"], us_gdp_data["GDP"], label='US GDP')
    plt.plot(us_gdp_data["TimeIndex"], mvg_avg, label=f'MA({window}) Forecast')
    plt.title(f"US GDP with MA({window})")
    plt.xlabel("Year")
    plt.ylabel("GDP")
    plt.legend(loc='best')
    plt.grid(True)
    if show: plt.show()


def read_us_gdp_data(filepath_or_buffer: str) -> pd.DataFrame:
    """Load GDP data and assign a time index."""
    us_gdp_data = pd.read_csv(filepath_or_buffer)
    us_gdp_data['TimeIndex'] = pd.date_range(start='1929', periods=len(us_gdp_data), freq='A')
    return us_gdp_data


def plot_predictions_ma_model(model_ma: MAModelFMU, test_df: pd.Series, show=True):
    """Plot predictions versus ground truth for a given model."""
    predictions = model_ma.simulate(len(test_df))
    plt.figure(figsize=(10, 4))
    plt.plot(test_df.reset_index(drop=True), label="Actual")
    plt.plot(predictions, label="Predicted", linestyle="--")
    plt.legend()
    plt.grid(True)
    if show: plt.show()


def test_r021a_ma_model():
    """Quick test for plotting and model structure."""
    data = read_us_gdp_data(f'{path_to_data}/input/gdpus.csv')
    gdp_series = data["GDP"].dropna()

    # Initialize and save model
    model_ma = MAModelFMU()
    model_ma.fit(gdp_series)
    model_ma.read()
    model_ma.save(f"{path_to_data}/output/model_ma.json")

    # Load and test new model instance
    model_ma2 = MAModelFMU()
    model_ma2.load(f"{path_to_data}/output/model_ma.json")
    model_ma2.read()

    # Visual tests
    plot_gdp(data)
    plot_predictions_ma_model(model_ma, gdp_series)
    plot_predictions_ma_model(model_ma2, gdp_series)
    plot_gdp_ma(data)
