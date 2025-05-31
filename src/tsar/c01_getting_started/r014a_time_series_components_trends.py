
# %%
def parsing_fn(x):
    return datetime.strptime('190' + x, '%Y-%m')


# %%
data = pd.read_csv(
    filepath_or_buffer=f'{path_to_data}/input/shampoo-sales.csv',
    header=0,
    parse_dates=[0],
    index_col=0,

    date_parser=parsing_fn
)
data = data.iloc[:, 0]  # Assume single-column data

# %%
data.plot()
plt.show()

