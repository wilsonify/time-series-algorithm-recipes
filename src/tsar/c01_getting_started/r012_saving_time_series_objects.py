import pandas as pd

from c01_getting_started import path_to_data

# %%
data = pd.read_csv(
    filepath_or_buffer=f"{path_to_data}/input/airpassengers.csv",
    parse_dates=['Month'],
    index_col='Month',
    date_parser=pd.to_datetime
)
# %%
### Saving the TS object as csv
data.to_csv(f'{path_to_data}/output/ts_data.csv', index=True, sep=',')
### Check the obj stored
data1 = pd.read_csv(f'{path_to_data}/output/ts_data.csv')
### Check
data1.head(2)
# %%
