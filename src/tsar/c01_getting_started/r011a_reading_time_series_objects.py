
import matplotlib.pyplot as plt
# %%
import pandas as pd

from c01_getting_started import path_to_data

# %%


# %%
data = pd.read_csv(
    filepath_or_buffer=f"{path_to_data}/input/airpassengers.csv",
    parse_dates=['Month'],
    index_col='Month',
    date_parser=pd.to_datetime
)

plt.plot(data, label='Passenger Count')
plt.show()
