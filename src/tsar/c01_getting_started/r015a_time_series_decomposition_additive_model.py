# %%
import statsmodels.api as sm

# %%
turn_over_data = pd.read_csv('./data/RetailTurnover.csv')
date_range = pd.date_range(start='1/7/1982', end='31/3/1992', freq='Q')
turn_over_data['TimeIndex'] = pd.DataFrame(date_range, columns=['Quarter'])

# %%
plt.plot(turn_over_data.TimeIndex, turn_over_data.Turnover)
plt.legend(loc='best')
plt.show()

# %%
decomp_turn_over = sm.tsa.seasonal_decompose(turn_over_data.Turnover, model="additive", period=4)
decomp_turn_over.plot()
plt.show()

# %%
trend = decomp_turn_over.trend
seasonal = decomp_turn_over.seasonal
residual = decomp_turn_over.resid
