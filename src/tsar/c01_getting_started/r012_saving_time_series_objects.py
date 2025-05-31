# %%
### Saving the TS object as csv
data.to_csv(f'{path_to_data}/output/ts_data.csv', index=True, sep=',')
### Check the obj stored
data1 = pd.read_csv(f'{path_to_data}/output/ts_data.csv')
### Check
data1.head(2)

# %%