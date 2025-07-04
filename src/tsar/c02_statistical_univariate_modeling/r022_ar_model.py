import json
import logging
from collections import deque
from pathlib import Path
from typing import List, Deque

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot, pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.gridspec import GridSpec
from sklearn.metrics import root_mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

from tsar.c01_getting_started import path_to_data
from tsar.c02_statistical_univariate_modeling import _looks_like_datetime, default_deserializer, default_serializer, \
    float_day_to_datetime, timestamp_to_float_day, timestamp_to_float_day, float_day_to_datetime


class ARModelFMU:
    def __init__(self, lags: int = 8, window: int = 10):
        self.lags: int = lags
        self.window: int = window
        self.params: List[float] = []
        self.history: Deque[float] = deque(maxlen=window)
        self.initial_history: Deque[float] = deque(maxlen=window)
        self.current_time: float = 0.0
        self.start_day: float = 0.0  # jan 1st 1970
        self.initialized: bool = False

    def fit(self, series: pd.Series, window: int = 10):
        """A convenience method to fit from a pandas Series."""
        assert isinstance(series.index, pd.DatetimeIndex), "Series must have a DatetimeIndex."
        self.start_day = timestamp_to_float_day(series.index[0])
        self.create(
            window=window,
            obs=series.tolist()
        )

    def read(self):
        print(f"params = {self.params}")
        print(f"lags = {self.lags}")
        print(f"initial_obs = {self.initial_obs}")
        print(f"history = {self.history}")
        print(f"current_time = {self.current_time}")

    def create(self, window: int, obs: List[float]):
        """
        Initialize the model with a fixed-size window and fit a linear model of obs ~ lagged obs.
        """
        assert window > 0, "Window size must be positive."
        assert len(obs) > self.lags, f"Need at least {self.lags} observations."
        model = AutoReg(endog=obs, lags=self.lags, trend='c').fit()
        self.params = model.params.tolist()
        print(model.summary())
        # Save model state
        self.window = window
        self.history = deque(list(obs), maxlen=window)
        self.initial_history = deque(list(obs), maxlen=window)
        self.initial_obs = self.history[0]
        self.last_obs = self.history[-1]
        self.current_time = 0
        self.initialized = True

    def update(self, value):
        self.history.append(value)

    def delete(self):
        self.params = None
        self.lags = None
        self.initial_obs = None
        self.history = None
        self.current_time = 0
        self.initialized = False

    def do_step(self) -> float:
        """
        Take one step using the fitted AR model.
        Uses self.current_time as a float day (e.g., 2005.0).
        Returns predicted value and advances time.
        """
        lags = 8  # AR(8)

        # Last 8 observations, most recent last
        recent_values = np.array(self.history[-lags:][::-1])  # Reverse to match AutoReg lag order

        # Compute AR prediction: intercept + dot(lagged obs, AR params)
        intercept = self.params[0]
        ar_coeffs = np.array(self.params[1:])  # Assumes [const, phi1, phi2, ..., phi8]

        yhat = float(intercept + np.dot(ar_coeffs, recent_values))

        logging.debug(f"recent_values={recent_values}")
        logging.debug(f"intercept={intercept}")
        logging.debug(f"ar_coeffs={ar_coeffs}")
        logging.debug(f"yhat={yhat}")

        self.current_time += 1.0  # advance by one day
        self.history.append(yhat)
        return yhat

    def simulate(self, nsteps, start_time=0.0, start_obs=0.0):
        """
        Simulate forward `nsteps` using the model. Returns a time-indexed Series.

        Parameters:
            nsteps: int
                number of time steps to simulate
            start_time : float
                day to start from (e.g., 2005.0).
            start_obs : float
                Override for the last observed value (optional).

        Returns:
            pd.Series: Forecast values indexed by datetime.
        """
        self.reset()
        # Override state if needed
        if start_time is not None:
            self.current_time = timestamp_to_float_day(start_time)
        if start_obs is not None:
            self.last_obs = start_obs

        predictions = np.empty(nsteps)
        float_times = np.empty(nsteps)

        for i in range(nsteps):
            float_times[i] = self.current_time
            predictions[i] = self.do_step()

        # Convert float days → datetime index
        datetime_index = [float_day_to_datetime(t) for t in float_times]

        return pd.Series(predictions, index=datetime_index)

    def reset(self):
        self.history = self.initial_history
        self.current_time = 0

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

    def plot_predictions_ar_model(self, model_ar, values, show=True):
        predictions = model_ar.do_step(len(values))
        pyplot.plot(predictions)
        pyplot.plot(values, color='red')
        if show: plt.show()


class ARModelTracker:
    def __init__(self, model: ARModelFMU = None):
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
        self.ax0.set_xlabel("day")
        self.ax0.set_ylabel("GDP")
        self.ax1.set_xlabel("day")
        self.ax1.set_ylabel("coefficients")
        self.ax2.set_xlabel("day")
        self.ax2.set_ylabel("score")
        self.artists_list = []
        if model is None:
            self.model: ARModelFMU = ARModelFMU()

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
        plt.xlabel("day")
        plt.ylabel("GDP")
        plt.legend(loc='best')

        if show:
            plt.show()

    def plot_gdp_ma(self, us_gdp_data: pd.DataFrame, window: int = 10, show: bool = True):
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
        plt.xlabel("day")
        plt.ylabel("GDP")
        plt.legend(loc='best')
        if show: plt.show()

    def plot_predictions(self, model: ARModelFMU, series: pd.Series, nsteps: int = 5, show=True):
        """Forecast overlay using seaborn."""
        predictions = model.simulate(
            nsteps=nsteps,
            start_time=series.index[-1],
            start_obs=series.iloc[-1]
        )
        pred_index = list(range(len(series), len(series) + nsteps))
        pred_series = pd.Series(predictions, index=pred_index)

        sns.lineplot(x=series.index, y=series.values, ax=self.ax0, label="Actual")
        sns.lineplot(x=pred_series.index, y=pred_series.values, ax=self.ax0, label="Predicted", linestyle="--")

        self.ax0.axvline(x=len(series) - 1, color='gray', linestyle=':', label='Forecast Start')
        self.ax0.set_title("Forecast at End of Series")
        self.ax0.set_xlabel("Index")
        self.ax0.set_ylabel("Value")
        self.ax0.legend()
        if show: plt.show()

    def plot_actuals(self, df, ax):
        """Plot GDP over time using seaborn."""
        sns.lineplot(x='day', y='GDP', data=df, ax=ax, label="GDP")
        sns.scatterplot(x='day', y='GDP', data=df, ax=ax, color='blue', s=20)
        ax.set_title("GDP over Time")
        ax.set_xlabel("day")
        ax.set_ylabel("GDP")

    def plot_moving_average(self, df, ax):
        """Overlay moving average using seaborn."""
        sns.lineplot(x='day', y='GDP_MA', data=df, ax=ax, label="Moving Avg", color='green', marker='o')
        ax.set_title("GDP with Moving Average")
        ax.set_xlabel("day")
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
        self.model.fit(df.head(20)["GDP"])
        coef_dict = {"day": [], "const": [], "trend": [], "x1": [], "score": []}
        artists_list = []

        for frame in range(6, len(df)):
            current_df = df.iloc[:frame + 1]
            current_dp = current_df.tail(1)
            nsteps = 5
            evaluations = self.MA.simulate(
                nsteps=nsteps,
                start_time=current_df.iloc[-nsteps]["day"],
                start_obs=current_df.iloc[-nsteps]["GDP"]
            )
            score = root_mean_squared_error(evaluations, current_df["GDP"].tail(nsteps))
            if len(current_df > 20):
                self.MA.fit(current_df.tail(100)["GDP"])
            predictions = self.MA.simulate(
                nsteps=nsteps,
                start_time=current_df.iloc[-1]["day"],
                start_obs=current_df.iloc[-1]["GDP"]
            )
            coef_dict["day"].append(current_dp.index[-1])
            coef_dict["const"].append(self.MA.params[0])
            coef_dict["trend"].append(self.MA.params[1])
            coef_dict["x1"].append(self.MA.params[2])
            coef_dict["score"].append(score)

            scat = self.ax0.scatter(current_df.index, current_df['GDP'], color='black')
            line0, = self.ax0.plot(current_df.index, current_df['GDP_MA'], color='blue')
            line1a, = self.ax0.plot(evaluations.index, evaluations, color='orange')
            line1b, = self.ax0.plot(predictions.index, predictions, color='blue')
            line2, = self.ax1.plot(coef_dict["day"], coef_dict["const"], color='red')
            line3, = self.ax1.plot(coef_dict["day"], coef_dict["trend"], color='green')
            line4, = self.ax1.plot(coef_dict["day"], coef_dict["x1"], color='blue')

            line5, = self.ax2.plot(coef_dict["day"], coef_dict["score"], color='black')

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
        model_ar = ARModelFMU()
        model_ar.fit(df["GDP"])
        self.plot_predictions_ma_model(
            model_ma=model_ar,
            series=df["GDP"],
            ax=self.ax0,
            nsteps=5,
        )
        self.fig.tight_layout()
        plt.savefig(output_path)

    def train_test_split_consumption(self, series):
        n = len(series)
        train_end = int(n * 0.8)
        test_end = int(n * 0.9)
        train_df = series.iloc[:train_end]['Consumption']
        test_df = series.iloc[train_end:test_end]['Consumption']
        eval_df = series.iloc[test_end:]['Consumption']
        return test_df, train_df, eval_df

    def plot_opsd_germany_daily_pacf(self, data, show=True):
        data['Consumption'].plot()
        data_stationarity_test = adfuller(data['Consumption'], autolag='AIC')
        print("P-value: ", data_stationarity_test[1])
        plot_pacf(data['Consumption'], lags=25)
        if show: plt.show()


def read_opsd_germany_daily(filepath_or_buffer):
    data = pd.read_csv(filepath_or_buffer, sep=",")
    data['TimeIndex'] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
    data = data.sort_values('TimeIndex')
    # regular time index
    start = data['TimeIndex'].iloc[0].to_period("D").to_timestamp()
    end = data['TimeIndex'].iloc[-1].to_period("D").to_timestamp()
    regular_index = pd.date_range(start=start, end=end, freq="D")
    data = data.set_index("TimeIndex")
    data = data.reindex(regular_index)
    return data


def test_r022_ar_model():
    filepath_or_buffer = f'{path_to_data}/input/opsd_germany_daily.csv'
    data = read_opsd_germany_daily(filepath_or_buffer)
    tr = ARModelTracker()
    tr.plot_opsd_germany_daily_pacf(data, show=True)
    test_df, train_df, eval_df = tr.train_test_split_consumption(data)
    model = ARModelFMU()
    model.fit(train_df)
    model.read()
    model.save(f"{path_to_data}/output/model_ar.json")
    model2 = ARModelFMU()
    model2.load(f"{path_to_data}/output/model_ar.json")
    model2.read()
    tr.plot_predictions(model2, data, show=True)

