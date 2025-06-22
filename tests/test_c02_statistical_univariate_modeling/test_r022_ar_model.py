import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

from c01_getting_started import path_to_data

# %%
def test_r022_ar_model():
    data = pd.read_csv(f'{path_to_data}/input/opsd_germany_daily.csv', sep=",")
    data['Consumption'].plot()

    # %%
    data_stationarity_test = adfuller(data['Consumption'], autolag='AIC')
    print("P-value: ", data_stationarity_test[1])

    # %%
    pacf = plot_pacf(data['Consumption'], lags=25)

    # %%
    train_df = data['Consumption'][:len(data) - 100]
    test_df = data['Consumption'][len(data) - 100:]

    # %%
    model_ar = AutoReg(train_df, lags=8).fit()

    # %%
    print(model_ar.summary())

    # %%
    predictions = model_ar.predict(start=len(train_df),
                                   end=(len(data) - 1), dynamic=False)

    # %%
    from matplotlib import pyplot

    pyplot.plot(predictions)
    pyplot.plot(test_df, color='red')
