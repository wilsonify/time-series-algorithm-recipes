# %%

import matplotlib.pyplot as plt
# %%
import pandas as pd


def read_weather_data(filepath_or_buffer):
    data1 = pd.read_csv(filepath_or_buffer=filepath_or_buffer, index_col=0)
    data1.columns = data1.columns.str.lower()
    data1 = data1.rename(columns={
        'pm2.5': 'pollution',
        'dewp': 'dew',
        'temp': 'temp',
        'pres': 'press',
        'cbwd': 'wnd_dir',
        'iws': 'wnd_spd',
        'is': 'snow',
        'ir': 'rain'
    })
    data1['date'] = pd.to_datetime(
        data1[['year', 'month', 'day', 'hour']].astype(str).agg(' '.join, axis=1),
        format='%Y %m %d %H'
    )
    data1.set_index("date", inplace=True)
    data1['pollution'].fillna(0, inplace=True)
    data1 = data1[24:]
    return data1


def plot_weather_data(data1, show=True):
    vals = data1.values
    group_list = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    plt.figure()
    for group in group_list:
        plt.subplot(len(group_list), 1, i)
        plt.plot(vals[:, group])
        plt.title(data1.columns[group], y=0.5, loc='right')
        i += 1
    if show:
        plt.show()
