# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
date_parser_fn = lambda dates: pd.datetime.strptime(dates,'%Y-%m')

# %%
data = pd.read_csv("./data/AirPassengers.csv", parse_dates =['Month'], index_col = 'Month', date_parser = date_parser_fn)
plt.plot(data)
plt.show()

# %%
indian_gdp_data = pd.read_csv('./data/GDPIndia.csv', header=0)
date_range = pd.date_range(start='1/1/1960', end='31/12/2017',freq='A')
indian_gdp_data ['TimeIndex'] = pd.DataFrame(date_range,columns=['Year'])
indian_gdp_data.head(5).T

# %%
plt.plot(indian_gdp_data.TimeIndex, indian_gdp_data.GDPpercapita)
plt.legend(loc='best')
plt.show()

# %%
import pickle
with open('gdp_india.obj', 'wb') as fp:
     pickle.dump(IndiaGDP, fp)
### Retrieve the pickle object
with open('./data/gdp_india.obj', 'rb') as fp:
    indian_gdp_data1 = pickle.load(fp)
indian_gdp_data1.head(5).T

# %%
### Saving the TS object as csv
data.to_csv('./data/ts_data.csv', index = True, sep = ',')
### Check the obj stored
data1 = pd.read_csv('./data/ts_data.csv')
### Check
data1.head(2)

# %%
    

# %%
data = pd.read_csv('./data/daily-min-temperatures.csv',
header = 0, index_col = 0, parse_dates = True, squeeze = True)
print(data.head())

# %%
data.plot()
plt.ylabel('Minimum Temp')
plt.title('Min temp in Southern Hemisphere From 1981 to 1990')
plt.show()

# %%
from datetime import datetime
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


# %%
data1 = pd.read_csv('./data/raw.csv', parse_dates = [['year','month', 'day', 'hour']],index_col=0, date_parser=parse)

# %%
data1.drop('No', axis=1, inplace=True)

# %%
data1.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
data1.index.name = 'date'

# %%
data1['pollution'].fillna(0, inplace=True)

# %%
data1 = data1[24:]

# %%
print(data1.head(5))

# %%
vals = data1.values
# specify columns to plot
group_list = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
plt.figure()
for group in group_list:
    plt.subplot(len(group_list), 1, i)
    plt.plot(vals[:, group])
    plt.title(data1.columns[group], y=0.5, loc='right')
    i += 1
plt.show()


# %%

# %%
def parsing_fn(x):
    return datetime.strptime('190'+x, '%Y-%m')


# %%
data = pd.read_csv('./data/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser= parsing_fn)

# %%
data.plot()
plt.show()

# %%
data = pd.read_csv('./data/daily-min-temperatures.csv',header = 0, index_col = 0, parse_dates = True, squeeze = True)

# %%
data.plot()
plt.ylabel('Minimum Temp')
plt.title('Min temp in Southern Hemisphere from 1981 to 1990')
plt.show()

# %%
month_df = pd.DataFrame()
one_year_ser = data['1990']
grouped_df = one_year_ser.groupby(pd.Grouper(freq='M'))
month_df = pd.concat([pd.DataFrame(x[1].values) for x in grouped_df], axis=1)
month_df = pd.DataFrame(month_df)
month_df.columns = range(1,13)
month_df.boxplot()
plt.show()

# %%
grouped_ser = data.groupby(pd.Grouper(freq='A'))
year_df = pd.DataFrame()
for name, group in grouped_ser:
    year_df[name.year] = group.values
year_df.boxplot()
plt.show()

# %%

# %%
tractor_sales_data = pd.read_csv("./data/tractor_salesSales.csv")
tractor_sales_data.head(5)

# %%
date_ser = pd.date_range(start='2003-01-01', freq='MS',periods=len(tractor_sales_data))

# %%
tractor_sales_data.rename(columns={'Number of Tractor Sold':'Tractor-Sales'}, inplace=True)
tractor_sales_data.set_index(date_ser, inplace=True)
tractor_sales_data = tractor_sales_data[['Tractor-Sales']]
tractor_sales_data.head(5)

# %%
tractor_sales_data.plot()
plt.ylabel('Tractor Sales')
plt.title("Tractor Sales from 2003 to 2014")
plt.show()

# %%
month_df = pd.DataFrame()
one_year_ser = tractor_sales_data['2011']
grouped_ser = one_year_ser.groupby(pd.Grouper(freq='M'))
month_df = pd.concat([pd.DataFrame(x[1].values) for x in
grouped_ser], axis=1)
month_df = pd.DataFrame(month_df)
month_df.columns = range(1,13)
month_df.boxplot()
plt.show()

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# %%
turn_over_data = pd.read_csv('./data/RetailTurnover.csv')
date_range = pd.date_range(start='1/7/1982', end='31/3/1992',freq='Q')
turn_over_data['TimeIndex'] = pd.DataFrame(date_range,columns=['Quarter'])

# %%
plt.plot(turn_over_data.TimeIndex, turn_over_data.Turnover)
plt.legend(loc='best')
plt.show()

# %%
decomp_turn_over = sm.tsa.seasonal_decompose(turn_over_data.Turnover, model="additive", freq=4)
decomp_turn_over.plot()
plt.show()

# %%
trend = decomp_turn_over.trend
seasonal = decomp_turn_over.seasonal
residual = decomp_turn_over.resid

# %%
air_passengers_data = pd.read_csv('./data/AirPax.csv')

# %%
date_range = pd.date_range(start='1/1/1949', end='31/12/1960',freq='M')
air_passengers_data ['TimeIndex'] = pd.DataFrame(date_range,columns=['Month'])
print(air_passengers_data.head())

# %%
decomp_air_passengers_data = sm.tsa.seasonal_decompose(air_passengers_data.Passenger, model="multiplicative", freq=12)
decomp_air_passengers_data.plot()
plt.show()

# %%
Seasonal_comp = decomp_air_passengers_data.seasonal
Seasonal_comp.head(4)

# %%

# %%
turn_over_data = pd.read_csv('./data/RetailTurnover.csv')
date_range = pd.date_range(start='1/7/1982', end='31/3/1992',freq='Q')
turn_over_data['TimeIndex'] = pd.DataFrame(date_range,columns=['Quarter'])

# %%
quarterly_turn_over_data = pd.pivot_table(turn_over_data,values = "Turnover", columns = "Quarter", index = "Year")
quarterly_turn_over_data

# %%
quarterly_turn_over_data.plot()
plt.show()

# %%
quarterly_turn_over_data.boxplot()
plt.show()

# %%
