import pandas as pd
from matplotlib import pyplot as plt
from statsmodels import api as sm


def read_retail_turnover_data(filepath_or_buffer):
    turn_over_data = pd.read_csv(filepath_or_buffer)
    date_range = pd.date_range(start='1/7/1982', end='31/3/1992', freq='Q')
    turn_over_data['TimeIndex'] = pd.DataFrame(date_range, columns=['Quarter'])
    return turn_over_data


def plot_turnover(turn_over_data, show=True):
    plt.plot(turn_over_data.TimeIndex, turn_over_data.Turnover)
    plt.legend(loc='best')
    if show: plt.show()


def plot_decomp_turn_over(turn_over_data, show=True):
    decomp_turn_over = sm.tsa.seasonal_decompose(turn_over_data.Turnover, model="additive", period=4)
    decomp_turn_over.plot()
    if show: plt.show()
