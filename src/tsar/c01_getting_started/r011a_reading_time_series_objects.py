import os.path

import matplotlib.pyplot as plt
# %%
import pandas as pd

path_to_here = os.path.abspath(os.path.dirname(__file__))
path_to_data = os.path.abspath(f"{path_to_here}/../../../data")
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
