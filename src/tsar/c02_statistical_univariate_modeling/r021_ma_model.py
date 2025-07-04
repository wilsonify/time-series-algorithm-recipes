import json
import logging
from collections import deque
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import ArtistAnimation
from matplotlib.gridspec import GridSpec
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.ar_model import AutoReg

from tsar.c02_statistical_univariate_modeling import (
    default_serializer,
    _looks_like_datetime,
    default_deserializer,
    timestamp_to_float_year,
    moving_average_centered_filled, float_year_to_datetime
)


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
        self.initial_history: deque = deque([0, 0])
        self.current_time: float = 0
        self.initialized: bool = False
        self.params: List[float] = [0, 0, 0]
        self.lags: int = -1
        self.start_year: float = 1970.0

    def create(self, window: int, obs: List[float]):
        """
        Initialize the model with a fixed-size window and fit a linear model of obs ~ mvg_avg.
        equation: $\hat{y} = \beta_0 + \beta_1 \cdot t$ + \beta_2 \cdot \text{mvg\_avg}
        params:  [β₀, β₁, β₂]
        """
        assert window > 0, "Window size must be positive."
        assert len(obs) > 2, f"Need at least 1 observations."
        mvg_avg = moving_average_centered_filled(obs, window=window)
        model = AutoReg(endog=obs, exog=mvg_avg, lags=0, trend='c').fit()
        self.params[0] = model.params.tolist()[0]
        self.params[1] = 0
        self.params[2] = model.params.tolist()[1]
        print(model.summary())

        # Save model state
        self.window = window
        self.history = deque(list(obs), maxlen=window)
        self.initial_history = deque(list(obs), maxlen=window)
        self.initial_obs = self.history[0]
        self.last_obs = self.history[-1]
        self.lags = 0
        self.current_time = 0
        self.initialized = True

    def fit(self, series: pd.Series, window: int = 5):
        """A convenience method to fit from a pandas Series."""
        assert isinstance(series.index, pd.DatetimeIndex), "Series must have a DatetimeIndex."
        self.start_year = timestamp_to_float_year(series.index[0])
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

    def update(self, values: List[float], step_size: float = 0.1, lambda_reg: float = 1.0):
        """
        Append new observation and take a small L2-regularized gradient step toward MLE estimate.

        Parameters:
            values : List[float]
                The new observations.
            step_size : float
                Step size (learning rate) toward new optimal parameters.
            lambda_reg : float
                L2 regularization strength.
        """

        self.history.extend(values)
        self.last_obs = self.history[-1]
        # construct features (X matrix)
        y = np.array(list(self.history))
        n = len(y)
        t = np.arange(n)
        mvg_avg = np.array(moving_average_centered_filled(list(y), window=self.window))
        x_mat = np.column_stack([np.ones(n), mvg_avg, t])

        # Ridge Regression: (XᵀX + λI)β = Xᵀy
        identity = np.eye(x_mat.shape[1])
        beta_opt = np.linalg.inv(x_mat.T @ x_mat + lambda_reg * identity) @ x_mat.T @ y
        print(f"beta_opt={beta_opt}")

        # Move params slightly toward regularized estimate
        self.params = [
            (1 - step_size) * p_old + step_size * p_new
            for p_old, p_new in zip(self.params, beta_opt)
        ]

        self.current_time = 0

    def do_step(self) -> float:
        """
        Take one step using the fitted linear model.
        Uses self.current_time as a float year (e.g., 2005.0).
        Returns predicted value and advances time.
        """

        t = float(self.current_time) - self.start_year
        mvg_avg = np.mean(self.history)
        yhat = self.params[0] + self.params[1] * t + self.params[2] * float(mvg_avg)
        logging.debug(f"t={t}")
        logging.debug(f"mvg_avg={mvg_avg}")
        logging.debug(f"yhat={yhat}")
        self.current_time += 1.0  # advance by one year
        self.history.append(yhat)
        return float(yhat)

    def simulate(self, nsteps: int = 10, start_time: float = None, start_obs: float = None) -> pd.Series:
        """
        Simulate forward `nsteps` using the model. Returns a time-indexed Series.

        Parameters:
            nsteps: int
                number of time steps to simulate
            start_time : float
                Float year to start from (e.g., 2005.0).
            start_obs : float
                Override for the last observed value (optional).

        Returns:
            pd.Series: Forecast values indexed by datetime.
        """
        self.reset()
        # Override state if needed
        if start_time is not None:
            self.current_time = float(start_time)
        if start_obs is not None:
            self.last_obs = start_obs

        predictions = np.empty(nsteps)
        float_times = np.empty(nsteps)

        for i in range(nsteps):
            float_times[i] = self.current_time
            predictions[i] = self.do_step()

        # Convert float years → datetime index
        datetime_index = [float_year_to_datetime(t) for t in float_times]

        return pd.Series(predictions, index=datetime_index)

    def reset(self):
        """Reset the model to its original state."""
        assert self.initialized, "Model must be initialized using `create` or `fit`."
        self.current_time = 0
        self.history = self.initial_history
        self.last_obs = self.initial_obs

    def save(self, filepath: str):
        """Generic save: persist all instance attributes to JSON."""
        assert self.initialized, "Model must be initialized using `create` or `fit`."
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, default=default_serializer, indent=2)

    def load(self, filepath: str):
        """Generic load: restore all instance attributes from JSON."""
        with open(filepath, 'r') as f:
            raw_data = json.load(f)

        for key, value in raw_data.items():
            # Restore deque
            if key == "history":
                self.history = deque(value, maxlen=raw_data.get("window", None))
            # Restore Timestamp
            elif isinstance(value, str) and _looks_like_datetime(value):
                try:
                    setattr(self, key, pd.to_datetime(value))
                except Exception:
                    setattr(self, key, value)
            # Restore DateOffset
            elif isinstance(value, dict) and value.get("_date_offset"):
                setattr(self, key, pd.DateOffset(**value["kwds"]))
            else:
                setattr(self, key, default_deserializer(value))


class MAModelTracker:
    def __init__(self, ma_model: MAModelFMU = None):
        """
        Initialize the HurricaneTracker with a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing hurricane data.
        """
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = GridSpec(3, 1, height_ratios=[1, 1, 1])
        self.ax0 = self.fig.add_subplot(self.gs[0])
        self.ax1 = self.fig.add_subplot(self.gs[1])
        self.ax2 = self.fig.add_subplot(self.gs[2])
        self.ax0.scatter([], [], color='black', label="GDP")
        self.ax0.plot([], [], linestyle='-', color='blue', label="GDP_MA")
        self.ax0.plot([], [], linestyle='-', marker='o', color='grey', label="forecast")
        self.ax1.scatter([], [], color='red', label="constant")
        self.ax1.scatter([], [], color='green', label="trend")
        self.ax1.scatter([], [], color='blue', label="x1")
        self.ax2.scatter([], [], color='black', label="rmse")
        self.ax0.set_xlabel("Year")
        self.ax0.set_ylabel("GDP")
        self.ax1.set_xlabel("Year")
        self.ax1.set_ylabel("coefficients")
        self.ax2.set_xlabel("Year")
        self.ax2.set_ylabel("score")
        self.artists_list = []
        if ma_model is None:
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

    def plot_predictions_ma_model(self, model_ma: MAModelFMU, series: pd.Series, ax, nsteps: int = 5, show=True):
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
        Create an animation of the data.

        Parameters:
        df (pd.DataFrame): The DataFrame containing data.
        output_path (str): The path to save the animation.
        """
        self.MA.fit(df.head(20)["GDP"])
        coef_dict = {"year": [], "const": [], "trend": [], "x1": [], "score": []}
        artists_list = []

        for frame in range(6, len(df)):
            current_df = df.iloc[:frame + 1]
            current_dp = current_df.tail(1)
            nsteps = 5
            evaluations = self.MA.simulate(
                nsteps=nsteps,
                start_time=current_df.iloc[-nsteps]["Year"],
                start_obs=current_df.iloc[-nsteps]["GDP"]
            )
            score = root_mean_squared_error(evaluations, current_df["GDP"].tail(nsteps))
            if len(current_df > 20):
                self.MA.fit(current_df.tail(100)["GDP"])
            predictions = self.MA.simulate(
                nsteps=nsteps,
                start_time=current_df.iloc[-1]["Year"],
                start_obs=current_df.iloc[-1]["GDP"]
            )
            coef_dict["year"].append(current_dp.index[-1])
            coef_dict["const"].append(self.MA.params[0])
            coef_dict["trend"].append(self.MA.params[1])
            coef_dict["x1"].append(self.MA.params[2])
            coef_dict["score"].append(score)

            scat = self.ax0.scatter(current_df.index, current_df['GDP'], color='black')
            line0, = self.ax0.plot(current_df.index, current_df['GDP_MA'], color='blue')
            line1a, = self.ax0.plot(evaluations.index, evaluations, color='orange')
            line1b, = self.ax0.plot(predictions.index, predictions, color='blue')
            line2, = self.ax1.plot(coef_dict["year"], coef_dict["const"], color='red')
            line3, = self.ax1.plot(coef_dict["year"], coef_dict["trend"], color='green')
            line4, = self.ax1.plot(coef_dict["year"], coef_dict["x1"], color='blue')

            line5, = self.ax2.plot(coef_dict["year"], coef_dict["score"], color='black')

            artists_list.append([scat, line0, line1a, line1b, line2, line3, line4, line5])
        self.ax0.legend()
        self.ax1.legend()

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
    """Load GDP data, regularize time index to annual frequency, and interpolate missing years."""
    # Load and parse time index
    us_gdp_data = pd.read_csv(filepath_or_buffer)
    us_gdp_data['TimeIndex'] = pd.to_datetime(us_gdp_data["Year"], format="%Y")
    us_gdp_data = us_gdp_data.sort_values('TimeIndex')

    # Step 1: Construct regular yearly time index
    start = us_gdp_data['TimeIndex'].iloc[0].to_period("Y").to_timestamp()
    end = us_gdp_data['TimeIndex'].iloc[-1].to_period("Y").to_timestamp()
    yearly_index = pd.date_range(start=start, end=end, freq="YS")

    # Step 2: Set index and reindex to regular timeline
    df_indexed = us_gdp_data.set_index("TimeIndex")
    df_regularized = df_indexed.reindex(yearly_index)

    # Step 3: Interpolate numeric columns only
    df_regularized["GDP"] = df_regularized["GDP"].interpolate(method="linear").ffill().bfill()

    # Step 4: Assign Year back and recompute GDP_MA
    df_regularized["Year"] = df_regularized.index.year
    df_regularized["GDP_MA"] = moving_average_centered_filled(df_regularized["GDP"].tolist(), window=5)

    return df_regularized
