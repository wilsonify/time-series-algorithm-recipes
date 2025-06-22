import pandas as pd
from matplotlib import pyplot as plt


def plot_gdp_ma(us_gdp_data, show=True):
    mvg_avg_us_gdp = us_gdp_data.copy()
    # calculating the rolling mean - with window 5
    mvg_avg_us_gdp['moving_avg_forecast'] = us_gdp_data['GDP'].rolling(5).mean()
    plt.plot(us_gdp_data['GDP'], label='US GDP')
    plt.plot(mvg_avg_us_gdp['moving_avg_forecast'], label='USGDP MA(5)')
    plt.legend(loc='best')
    plt.show()


def plot_gpd(us_gdp_data, show=True):
    plt.plot(us_gdp_data.TimeIndex, us_gdp_data.GDP)
    plt.legend(loc='best')
    plt.show()


def read_us_gdp_data(filepath_or_buffer):
    us_gdp_data = pd.read_csv(filepath_or_buffer=filepath_or_buffer, header=0)
    date_rng = pd.date_range(start='1/1/1929', end='31/12/1991', freq='A')
    us_gdp_data['TimeIndex'] = pd.DataFrame(date_rng, columns=['Year'])
    return us_gdp_data
