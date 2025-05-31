# %%
data = pd.read_csv(f'{path_to_data}/input/daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True)
data = data.iloc[:, 0]
# %%
data.plot()
plt.ylabel('Minimum Temp')
plt.title('Min temp in Southern Hemisphere from 1981 to 1990')
plt.show()

# %%

one_year_ser = data['1990']
grouped_df = one_year_ser.groupby(pd.Grouper(freq='M'))
month_df = pd.concat([pd.DataFrame(x[1].values) for x in grouped_df], axis=1)
month_df = pd.DataFrame(month_df)
month_df.columns = range(1, 13)
month_df.boxplot()
plt.show()

# %%
grouped_ser = data.groupby(pd.Grouper(freq='YE'))
year_df = pd.DataFrame()
for name, group in grouped_ser:
    year_df[name.year] = group.values
year_df.boxplot()
plt.show()

# %%

