
# %%
indian_gdp_data = pd.read_csv(
    filepath_or_buffer=f'{path_to_data}/input/gdp_india.csv',
    header=0
)
date_range = pd.date_range(start='1/1/1960', end='31/12/2017', freq='YE')
indian_gdp_data['TimeIndex'] = pd.DataFrame(date_range, columns=['Year'])
indian_gdp_data.head(5).T

# %%
plt.plot(indian_gdp_data.TimeIndex, indian_gdp_data.GDPpercapita, label="GDPerCapita")
plt.legend(loc='best')
plt.show()

# %%
import pickle

with open(f'{path_to_data}/output/gdp_india.obj', 'wb') as fp:
    # noinspection PyTypeChecker
    pickle.dump(indian_gdp_data, fp)

### Retrieve the pickle object
with open(f'{path_to_data}/output/gdp_india.obj', 'rb') as fp:
    indian_gdp_data1 = pickle.load(fp)
indian_gdp_data1.head(5).T
