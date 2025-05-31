import matplotlib.pyplot as plt
# %%
import pandas as pd
import statsmodels.api as sm

# %%
air_passengers_data = pd.read_csv('./data/AirPax.csv')

# %%
date_range = pd.date_range(start='1/1/1949', end='31/12/1960', freq='M')
air_passengers_data['TimeIndex'] = pd.DataFrame(date_range, columns=['Month'])
print(air_passengers_data.head())

# %%
decomp_air_passengers_data = sm.tsa.seasonal_decompose(
    x=air_passengers_data.Passenger,
    model="multiplicative",
    filt=None,
    period=12,  # Use this instead of `freq`
    two_sided=True,
    extrapolate_trend=0
)
decomp_air_passengers_data.plot()
plt.show()

# %%
Seasonal_comp = decomp_air_passengers_data.seasonal
Seasonal_comp.head(4)
