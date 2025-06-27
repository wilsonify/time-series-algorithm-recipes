import json
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

from c01_getting_started import path_to_data


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
        self.params: List[float] = [0]
        self.lags: int = -1

    def create(self, window: int, obs: List[float]):
        """
        Initialize the model with a fixed-size window and fit a linear model of obs ~ mvg_avg.
        equation: $\hat{y} = \beta_0 + \beta_1 \cdot t$ + \beta_2 \cdot \text{mvg\_avg}
        params:  [β₀, β₁, β₂]
        """
        assert window > 0, "Window size must be positive."
        assert len(obs) > 0, f"Need at least 1 observations."

        # Compute moving average
        mvg_avg = pd.Series(obs).rolling(
            window=window,
            min_periods=1,
            center=True
        ).mean().ffill().bfill().to_list()

        model = AutoReg(endog=obs, exog=mvg_avg, lags=0, trend='ct').fit()
        print(model.summary())

        # Save model state
        self.params = model.params.tolist()
        self.window = window
        self.history = deque(list(obs), maxlen=window)
        self.initial_obs = self.history[0]
        self.last_obs = self.history[-1]
        self.lags = 0
        self.current_time = 0
        self.initialized = True

    def fit(self, series: pd.Series, window: int = 5):
        """A convenience method to fit from a pandas Series."""
        self.create(
            window=window,
            obs=series.tolist()
        )

    def read(self):
        """Print current model state."""
        print(f"Window Size           : {self.window}")
        print(f"Current History       : {list(self.history)}")
        print(f"Initial Observations  : {self.initial_obs}")
        print(f"Current Time Step     : {self.current_time}")

    def update(self, values: List[float], alpha: float = 0.1, lambda_reg: float = 1.0):
        """
        Append new observation and take a small L2-regularized gradient step toward MLE estimate.

        Parameters:
            values : List[float]
                The new observations.
            alpha : float
                Step size (learning rate) toward new optimal parameters.
            lambda_reg : float
                L2 regularization strength.
        """
        assert self.initialized, "Model must be initialized using `create` or `fit`."
        self.history.append(values)
        self.last_obs = self.history[-1]

        # construct features (X matrix)
        y = np.array(self.history)
        n = len(y)
        t = np.arange(n)
        mvg_avg = pd.Series(y).rolling(
            window=self.window,
            min_periods=1,
            center=True
        ).mean().ffill().bfill().to_numpy()
        x_mat = np.column_stack([np.ones(n), mvg_avg, t])

        # Ridge Regression: (XᵀX + λI)β = Xᵀy
        identity = np.eye(x_mat.shape[1])
        beta_opt = np.linalg.inv(x_mat.T @ x_mat + lambda_reg * identity) @ x_mat.T @ y

        # Move params slightly toward regularized estimate
        self.params = [
            (1 - alpha) * p_old + alpha * p_new
            for p_old, p_new in zip(self.params, beta_opt)
        ]

        self.current_time = 0

    def do_step(self) -> float:
        """Take exactly one step using the fitted linear model. Return the next predicted value."""
        assert self.initialized, "Model must be initialized using `create` or `fit`."
        print(f"current_time={self.current_time}")
        mvg_avg = np.mean(self.history)
        print(f"mvg_avg={mvg_avg}")
        yhat = self.params[0] + self.params[1] * self.current_time + self.params[2] * float(mvg_avg)
        print(f"yhat={yhat}")
        self.current_time += 1
        return float(yhat)

    def simulate(self, nsteps: int = 10, start_time=None, start_obs=None) -> np.ndarray:
        """Simulate forward n steps using repeated do_step."""
        assert self.initialized, "Model must be initialized using `create` or `fit`."
        # Use provided overrides or default to internal state
        if start_time is not None:
            self.current_time = start_time
        if start_obs is not None:
            self.last_obs = start_obs

        predictions = np.empty(nsteps)
        for i in range(nsteps):
            predictions[i] = self.do_step()
        return predictions

    def reset(self):
        """Reset the model to its original state."""
        assert self.initialized, "Model must be initialized using `create` or `fit`."
        self.current_time = 0
        self.last_obs = self.initial_obs

    def save(self, filepath: str):
        """Persist full model state to a JSON file."""
        assert self.initialized, "Model must be initialized using `create` or `fit`."
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                "window": self.window,
                "initial_obs": self.initial_obs,
                "last_obs": self.last_obs,
                "history": list(self.history),
                "current_time": self.current_time,
                "initialized": self.initialized,
                "params": self.params,
                "lags": self.lags

            }, f)

    def load(self, filepath: str):
        """Load full model state from a JSON file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        self.window = model_data["window"]
        self.initial_obs = model_data["initial_obs"]
        self.last_obs = model_data["last_obs"]
        self.history = deque(model_data["history"], maxlen=self.window)
        self.current_time = model_data["current_time"]
        self.initialized = model_data["initialized"]
        self.params = model_data["params"]
        self.lags = model_data["lags"]


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
    mvg_avg = us_gdp_data["GDP"].rolling(
        window=window,
        min_periods=1,
        center=True
    ).mean().ffill().bfill().interpolate()
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


def plot_predictions_ma_model(model_ma: MAModelFMU, series: pd.Series, nsteps: int = 5, show=True):
    """
    Plot predictions at the end of the actual series.

    Parameters:
        model_ma: A fitted MAModelFMU instance.
        series: The actual data as a Series.
        nsteps: Number of steps to simulate into the future.
        show: Whether to immediately show the plot.
    """
    mvg_avg = series.rolling(
        window=model_ma.window,
        min_periods=1,
        center=True
    ).mean().ffill().bfill().interpolate()

    predictions = model_ma.simulate(
        nsteps=nsteps,
        start_time=series.index[-1],
        start_obs=series.iloc[-1]
    )
    predictions_index = list(range(len(series.index), len(series.index) + nsteps))
    predictions_series = pd.Series(predictions, index=predictions_index)

    # Extend x-axis with future indices
    plt.figure(figsize=(10, 4))

    x1 = list(series.index)
    y1 = series.to_list()
    plt.plot(x1, y1, label="Actual", color='blue')

    x2 = list(mvg_avg.index)
    y2 = list(mvg_avg.values)
    plt.plot(x2, y2, label="mvg_avg", color='black')

    x3 = predictions_series.index
    y3 = predictions_series.values
    plt.plot(x3, y3, label="Predicted", linestyle="--", color='orange')

    plt.axvline(len(series) - 1, color='gray', linestyle=':', label='Forecast Start')
    plt.title("Forecast at End of Series")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
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
