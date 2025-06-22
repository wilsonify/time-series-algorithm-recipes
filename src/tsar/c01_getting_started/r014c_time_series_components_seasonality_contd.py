import pandas as pd
from matplotlib import pyplot as plt


def read_tractor_sales_data(filepath_or_buffer):
    tractor_sales_data = pd.read_csv(filepath_or_buffer=filepath_or_buffer)
    tractor_sales_data.head(5)
    date_ser = pd.date_range(start='2003-01-01', freq='MS', periods=len(tractor_sales_data))
    tractor_sales_data.rename(columns={'Number of Tractor Sold': 'Tractor-Sales'}, inplace=True)
    tractor_sales_data.set_index(date_ser, inplace=True)
    tractor_sales_data = tractor_sales_data.loc[:, 'Tractor-Sales']
    return tractor_sales_data


def plot_tractor_sales(tractor_sales_data, show=True):
    tractor_sales_data.plot()
    plt.ylabel('Tractor Sales')
    plt.title("Tractor Sales from 2003 to 2014")
    if show: plt.show()


def boxplot_tractor_sales_one_year(tractor_sales_data, show=True):
    one_year_ser = tractor_sales_data['2011']
    grouped_ser = one_year_ser.groupby(pd.Grouper(freq='M'))
    month_df = pd.concat([pd.DataFrame(x[1].values) for x in grouped_ser], axis=1)
    month_df = pd.DataFrame(month_df)
    month_df.columns = range(1, 13)
    month_df.boxplot()
    if show: plt.show()
