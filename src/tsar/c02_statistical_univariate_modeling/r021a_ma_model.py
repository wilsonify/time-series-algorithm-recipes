import json
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

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
        self.params: List[float] = [0]
        self.lags: int = -1
        self.selected_trend: str = "ct"

    def create(self, window: int, obs: List[float]):
        """Initialize the model with a fixed-size window and fit a linear model of obs ~ mvg_avg.
        | Trend  | Equation                                                              | `params` mapping |
        | ------ | --------------------------------------------------------------------- | ---------------- |
        | `'n'`  | $\hat{y} = \beta_1 \cdot \text{mvg\_avg}$                             | `[β₁]`           |
        | `'c'`  | $\hat{y} = \beta_0 + \beta_1 \cdot \text{mvg\_avg}$                   | `[β₀, β₁]`       |
        | `'t'`  | $\hat{y} = \beta_1 \cdot \text{mvg\_avg} + \beta_2 \cdot t$           | `[β₁, β₂]`       |
        | `'ct'` | $\hat{y} = \beta_0 + \beta_1 \cdot \text{mvg\_avg} + \beta_2 \cdot t$ | `[β₀, β₁, β₂]`   |
        """
        assert window > 0, "Window size must be positive."
        assert len(obs) > 0, f"Need at least 1 observations."

        # Compute moving average
        mvg_avg = pd.Series(obs).rolling(
            window=window,
            min_periods=1,
            center=True
        ).mean().ffill().bfill().to_list()

        # Fit multiple models with different trends
        trends = ['n', 'c', 't', 'ct']
        models = {}
        bics = {}

        for trend in trends:
            try:
                model = AutoReg(endog=obs, exog=mvg_avg, lags=0, trend=trend).fit()
                models[trend] = model
                bics[trend] = model.bic
                print(f"Trend='{trend}' BIC={model.bic:.2f}")
            except Exception as e:
                print(f"Trend='{trend}' failed: {e}")
                continue

        # Choose the trend with the lowest BIC
        best_trend = min(bics, key=bics.get)
        best_model = models[best_trend]

        print(f"\nSelected model with trend='{best_trend}' (BIC={bics[best_trend]:.2f})")
        print(best_model.summary())

        # Save model state
        self.window = window
        self.history = deque(list(obs), maxlen=window)
        self.initial_obs = self.history[0]
        self.last_obs = self.history[-1]
        self.params = best_model.params.tolist()
        self.selected_trend = best_trend
        self.lags = 0
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
        """Take exactly one step using the fitted linear model. Return the next predicted value."""
        assert self.initialized, "Model must be initialized using `create` or `fit`."

        # Compute moving average of current history
        mvg_avg = np.mean(self.history)

        trend = self.selected_trend
        params = self.params
        t = self.current_time  # time trend

        # Apply the fitted model according to selected trend
        if trend == 'n':
            yhat = params[0] * mvg_avg

        elif trend == 'c':
            yhat = params[0] + params[1] * mvg_avg

        elif trend == 't':
            yhat = params[0] * mvg_avg + params[1] * t

        elif trend == 'ct':
            yhat = params[0] + params[1] * mvg_avg + params[2] * t

        else:
            raise ValueError(f"Unknown trend type: {trend}")

        # Update model state
        #self.history.append(yhat)
        self.current_time += 1
        return float(yhat)

    def simulate(self, nsteps: int = 10) -> np.ndarray:
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
                "selected_trend": self.selected_trend,
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
        self.selected_trend = model_data["selected_trend"]


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
    ).mean().ffill().bfill()

    predictions = model_ma.simulate(nsteps)
    predictions_index = list(range(len(series.index), len(series.index) + nsteps))
    predictions_series = pd.Series(predictions, index=predictions_index)

    # Extend x-axis with future indices
    plt.figure(figsize=(10, 4))

    x1 = list(series.index)
    y1 = list(series.values)
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
