import pandas as pd
from matplotlib import pyplot as plt
from statsmodels import api as sm


def plot_decomp_air_passengers(air_passengers_data, show=True):
    decomp_air_passengers_data = sm.tsa.seasonal_decompose(
        x=air_passengers_data.Passenger,
        model="multiplicative",
        filt=None,
        period=12,  # Use this instead of `freq`
        two_sided=True,
        extrapolate_trend=0
    )
    decomp_air_passengers_data.plot()
    if show: plt.show()
    return decomp_air_passengers_data


def read_air_passengers_data(filepath_or_buffer):
    air_passengers_data = pd.read_csv(filepath_or_buffer)
    date_range = pd.date_range(start='1/1/1949', end='31/12/1960', freq='M')
    air_passengers_data['TimeIndex'] = pd.DataFrame(date_range, columns=['Month'])
    return air_passengers_data
