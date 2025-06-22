from datetime import datetime

import matplotlib.pyplot as plt
# %%
import pandas as pd

from c01_getting_started import path_to_data


def parsing_fn(x):
    return datetime.strptime('190' + x, '%Y-%m')


def read_shampoo_sales(filepath_or_buffer):
    data = pd.read_csv(
        filepath_or_buffer=filepath_or_buffer,
        header=0,
        parse_dates=[0],
        index_col=0,
        date_parser=parsing_fn
    )
    data = data.iloc[:, 0]  # Assume single-column data
    return data


def plot_shampoo_sales(data, show=True):
    data.plot()
    if show:
        plt.show()
