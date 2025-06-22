import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from c01_getting_started import path_to_data

# %%

# %%
btc_data = pd.read_csv(f"{path_to_data}/input/btc.csv")
print(btc_data.head())

# %%
btc_data.index = pd.to_datetime(btc_data['Date'],
                                format='%Y-%m-%d')
del btc_data['Date']

# %%
plt.ylabel('Price-BTC')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.plot(btc_data.index, btc_data['BTC-USD'], )

# %%
train_data = btc_data[btc_data.index < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
test_data = btc_data[btc_data.index > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
print(train_data.shape)
print(test_data.shape)

# %%
plt.plot(train_data, color="black", label='Train')
plt.plot(test_data, color="green", label='Test')
plt.ylabel('Price-BTC')
plt.xlabel('Date')
plt.xticks(rotation=35)
plt.title("Train/Test split")
plt.show()

# %%
actuals = train_data['BTC-USD']
# %%
HW_model = ExponentialSmoothing(actuals, trend='add')
HW_model = HW_model.fit()

# %%
predictions_hw = HW_model.forecast(len(test_data.index))

# %%
plt.plot(train_data, color="black", label='Train')
plt.plot(test_data, color="green", label='Test')
plt.ylabel('Price-BTC')
plt.xlabel('Date')
plt.xticks(rotation=35)
plt.title("HW model predictions")
plt.plot(predictions_hw, color='red', label='Predictions')
plt.legend()
plt.show()

# %%
rmse_hw = np.sqrt(mean_squared_error(test_data["BTC-USD"].
                                     values, predictions_hw))
print("RMSE: ", rmse_hw)
