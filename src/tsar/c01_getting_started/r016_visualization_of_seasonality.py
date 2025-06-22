import pandas as pd
from matplotlib import pyplot as plt

from c01_getting_started import path_to_data


def boxplot_quarterly_turn_over_data(quarterly_turn_over_data, show=True):
    quarterly_turn_over_data.boxplot()
    if show: plt.show()


def plot_quarterly_turn_over_data(turn_over_data, show=True):
    quarterly_turn_over_data = pd.pivot_table(turn_over_data, values="Turnover", columns="Quarter", index="Year")
    quarterly_turn_over_data.plot()
    if show: plt.show()
    return quarterly_turn_over_data


def read_turn_over_data():
    turn_over_data = pd.read_csv(f'{path_to_data}/input/retailturnover.csv')
    date_range = pd.date_range(start='1/7/1982', end='31/3/1992', freq='Q')
    turn_over_data['TimeIndex'] = pd.DataFrame(date_range, columns=['Quarter'])
    return turn_over_data
