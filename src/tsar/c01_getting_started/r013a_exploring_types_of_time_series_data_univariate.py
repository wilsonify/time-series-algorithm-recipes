# %%
data = pd.read_csv(
    filepath_or_buffer=f'{path_to_data}/input/daily-min-temperatures.csv',
    header=0,
    index_col=0,
    parse_dates=True,
)
data = data.iloc[:, 0]  # Assume single-column data
print(data.head())

# %%
data.plot()
plt.ylabel('Minimum Temp')
plt.title('Min temp in Southern Hemisphere From 1981 to 1990')
plt.show()
