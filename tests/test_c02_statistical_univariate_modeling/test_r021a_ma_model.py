import matplotlib.pyplot as plt
import pandas as pd

from c01_getting_started import path_to_data

def test_r021a_ma_model():
    # %%
    us_gdp_data = pd.read_csv(f'{path_to_data}/input/gdpus.csv', header=0)

    # %%
    date_rng = pd.date_range(start='1/1/1929', end='31/12/1991', freq='A')
    print(date_rng)
    us_gdp_data['TimeIndex'] = pd.DataFrame(date_rng, columns=['Year'])

    # %%
    plt.plot(us_gdp_data.TimeIndex, us_gdp_data.GDP)
    plt.legend(loc='best')
    plt.show()

    # %%
    mvg_avg_us_gdp = us_gdp_data.copy()
    # calculating the rolling mean - with window 5
    mvg_avg_us_gdp['moving_avg_forecast'] = us_gdp_data['GDP'].rolling(5).mean()

    # %%
    plt.plot(us_gdp_data['GDP'], label='US GDP')
    plt.plot(mvg_avg_us_gdp['moving_avg_forecast'], label='USGDP MA(5)')
    plt.legend(loc='best')
    plt.show()
