
# %%
from datetime import datetime


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


# %%
data1 = pd.read_csv(
    filepath_or_buffer=f'{path_to_data}/input/raw.csv',
    parse_dates=['year', 'month', 'day', 'hour'],
    index_col=0,
    date_parser=parse
)

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
