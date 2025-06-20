import matplotlib.pyplot as plt
# %%
import pandas as pd

from c01_getting_started import path_to_data

# %%

# %%
turn_over_data = pd.read_csv(f'{path_to_data}/input/retailturnover.csv')
date_range = pd.date_range(start='1/7/1982', end='31/3/1992', freq='Q')
turn_over_data['TimeIndex'] = pd.DataFrame(date_range, columns=['Quarter'])

# %%
quarterly_turn_over_data = pd.pivot_table(turn_over_data, values="Turnover", columns="Quarter", index="Year")
quarterly_turn_over_data

# %%
quarterly_turn_over_data.plot()
plt.show()

# %%
quarterly_turn_over_data.boxplot()
plt.show()

# %%
