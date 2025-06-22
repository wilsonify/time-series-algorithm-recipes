import matplotlib.pyplot as plt

import pandas as pd


def load_temperature_data(filepath_or_buffer):
    """Load daily minimum temperature data as a pandas Series."""
    df = pd.read_csv(
        filepath_or_buffer=filepath_or_buffer,
        header=0,
        index_col=0,
        parse_dates=True
    )
    series = df.iloc[:, 0]
    return series


def plot_minimum_temp(data, show=True):
    """Plot the full time series."""
    data.plot()
    plt.ylabel('Minimum Temp')
    plt.title('Min temp in Southern Hemisphere from 1981 to 1990')
    plt.tight_layout()
    if show:
        plt.show()


def plot_monthly_seasonality(data, year='1990', show=True):
    """Plot a boxplot of monthly seasonality for a single year."""
    one_year = data[year]
    grouped = one_year.groupby(pd.Grouper(freq='M'))
    month_df = pd.concat([pd.DataFrame(month.values) for _, month in grouped], axis=1)
    month_df.columns = range(1, 13)
    month_df.boxplot()
    plt.title(f'Monthly seasonality (boxplot) for {year}')
    plt.xlabel('Month')
    plt.ylabel('Min Temp')
    plt.tight_layout()
    if show:
        plt.show()


def plot_yearly_distribution(data, show=False):
    """Plot boxplots to compare temperature distributions across years."""
    one_year_ser = data['1990']
    grouped_df = one_year_ser.groupby(pd.Grouper(freq='M'))
    month_df = pd.concat([pd.DataFrame(x[1].values) for x in grouped_df], axis=1)
    month_df = pd.DataFrame(month_df)
    month_df.columns = range(1, 13)
    month_df.boxplot()
    if show:
        plt.show()
