import json
from collections import deque
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import ArtistAnimation
from matplotlib.gridspec import GridSpec
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
        assert len(obs) > 2, f"Need at least 1 observations."

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
        print(f"obs = {obs}")
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
        self.history.extend(values)
        self.last_obs = self.history[-1]
        # construct features (X matrix)
        y = np.array(list(self.history))
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
        mvg_avg = np.mean(self.history)
        yhat = self.params[0] + self.params[1] * self.current_time + self.params[2] * float(mvg_avg)
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


class MAModelTracker:
    def __init__(self):
        """
        Initialize the HurricaneTracker with a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing hurricane data.
        """
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = GridSpec(1, 1, height_ratios=[1])
        self.ax0 = self.fig.add_subplot(self.gs[0])
        self.scat = self.ax0.scatter([], [], color='black', label="GDP")
        self.line = self.ax0.plot([], [], linestyle='-', color='blue', label="GDP_MA")
        self.forecast_line = self.ax0.plot([], [], linestyle='-', marker='o', color='grey', label="forecast")
        self.ax0.set_xlabel("Year")
        self.ax0.set_ylabel("GDP")
        self.artists_list = []
        self.MA: MAModelFMU = MAModelFMU()

    def plot_gdp(self, us_gdp_data: pd.DataFrame, title: str = "US GDP Over Time", show: bool = True):
        """Plot raw GDP data."""
        plt.figure(figsize=(10, 4))
        sns.lineplot(
            x="TimeIndex",
            y="GDP",
            data=us_gdp_data,
            label="US GDP"
        )
        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel("GDP")
        plt.legend(loc='best')

        if show:
            plt.show()

    def plot_gdp_ma(self, us_gdp_data: pd.DataFrame, window: int = 5, show: bool = True):
        """Plot GDP with moving average overlay."""
        mvg_avg = us_gdp_data["GDP"].rolling(
            window=window,
            min_periods=1,
            center=True
        ).mean().ffill().bfill().interpolate()
        us_gdp_data["mvg_avg"] = mvg_avg
        plt.figure(figsize=(10, 4))
        sns.lineplot(
            x="TimeIndex",
            y="GDP",
            data=us_gdp_data,
            label="US GDP"
        )
        sns.lineplot(
            x="TimeIndex",
            y="mvg_avg",
            data=us_gdp_data,
            label="US GDP mvg_avg"
        )
        plt.title(f"US GDP with MA({window})")
        plt.xlabel("Year")
        plt.ylabel("GDP")
        plt.legend(loc='best')
        if show: plt.show()

    def plot_predictions_ma_model(self, model_ma: MAModelFMU, series: pd.Series, ax, nsteps: int = 5):
        """Forecast overlay using seaborn."""
        predictions = model_ma.simulate(
            nsteps=nsteps,
            start_time=series.index[-1],
            start_obs=series.iloc[-1]
        )
        pred_index = list(range(len(series), len(series) + nsteps))
        pred_series = pd.Series(predictions, index=pred_index)

        sns.lineplot(x=series.index, y=series.values, ax=ax, label="Actual")
        sns.lineplot(x=pred_series.index, y=pred_series.values, ax=ax, label="Predicted", linestyle="--")

        ax.axvline(x=len(series) - 1, color='gray', linestyle=':', label='Forecast Start')
        ax.set_title("Forecast at End of Series")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.legend()

    def plot_actuals(self, df, ax):
        """Plot GDP over time using seaborn."""
        sns.lineplot(x='Year', y='GDP', data=df, ax=ax, label="GDP")
        sns.scatterplot(x='Year', y='GDP', data=df, ax=ax, color='blue', s=20)
        ax.set_title("GDP over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("GDP")

    def plot_moving_average(self, df, ax):
        """Overlay moving average using seaborn."""
        sns.lineplot(x='Year', y='GDP_MA', data=df, ax=ax, label="Moving Avg", color='green', marker='o')
        ax.set_title("GDP with Moving Average")
        ax.set_xlabel("Year")
        ax.set_ylabel("GDP")
        ax.tick_params(axis='x', rotation=45)

    def plot_forecast(self, df, ax):
        """Plot longitude vs time using seaborn."""
        sns.lineplot(x='timestamp', y='Longitude_float', data=df, ax=ax, marker='o', color='red', label="Forecast")
        ax.set_title("Longitude vs Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Longitude")
        ax.tick_params(axis='x', rotation=45)

    def animate(self, df, output_path):
        """
        Create an animation of the hurricane data.

        Parameters:
        df (pd.DataFrame): The DataFrame containing hurricane data.
        output_path (str): The path to save the animation.
        """
        self.MA.fit(df["GDP"].head(5))
        artists_list = []
        for frame in range(4, len(df)):
            current_df = df.iloc[:frame + 1]
            self.MA.update(current_df["GDP"].tolist())
            current_dp = current_df.tail(1)
            nsteps = 5
            predictions = self.MA.simulate(
                nsteps=nsteps,
                start_time=current_dp["GDP"].index[-1],
                start_obs=current_dp["GDP"].iloc[-1]
            )
            scat = self.ax0.scatter(current_df['TimeIndex'], current_df['GDP'], color='black')
            line0, = self.ax0.plot(current_df['TimeIndex'], current_df['GDP_MA'], color='blue')
            line1, = self.ax0.plot([current_dp["TimeIndex"] + pd.DateOffset(years=i) for i in range(0, nsteps)],
                                   predictions, color='blue')
            artists_list.append([scat, line0, line1])
        self.ax0.legend()

        anim = ArtistAnimation(self.fig, artists_list, interval=200, blit=True)
        anim.save(output_path)

    def plot(self, df, output_path):
        """
        Create static plots of the hurricane data.

        Parameters:
        df (pd.DataFrame): The DataFrame containing hurricane data.
        output_path (str): The path to save the plot.
        """
        self.plot_actuals(df, self.ax0)
        self.plot_moving_average(df, self.ax0)
        model_ma = MAModelFMU()
        model_ma.fit(df["GDP"])
        self.plot_predictions_ma_model(
            model_ma=model_ma,
            series=df["GDP"],
            ax=self.ax0,
            nsteps=5,
        )
        self.fig.tight_layout()
        plt.savefig(output_path)


def read_us_gdp_data(filepath_or_buffer: str) -> pd.DataFrame:
    """Load GDP data and assign a time index."""
    us_gdp_data = pd.read_csv(filepath_or_buffer)
    us_gdp_data['TimeIndex'] = pd.to_datetime(us_gdp_data["Year"], format="%Y")
    us_gdp_data = us_gdp_data.sort_values('TimeIndex')
    us_gdp_data['GDP_MA'] = us_gdp_data["GDP"].rolling(
        window=5,
        min_periods=1,
        center=True
    ).mean().ffill().bfill().interpolate()
    return us_gdp_data


def test_demo():
    df = read_us_gdp_data(f"{path_to_data}/input/gdpus.csv")
    tracker = MAModelTracker()
    # tracker.plot(df, f"{path_to_data}/output/ma_model_tracker.png")
    tracker.animate(df.head(25), f"{path_to_data}/output/ma_model_tracker.gif")
