# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import glob
import time
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)

# %%
df = pd.read_csv('train_6BJx641.csv')

# %%
df.head()

# %%
del df['ID']

# %%
df.isnull().sum()

# %%
df.info()

# %%
#Creating datetime features to use in model to capture seasonality 
df['time'] = pd.to_datetime(df['datetime'])   
df['year'] = df.time.dt.year
df['month'] = df.time.dt.month
df['day'] = df.time.dt.day
df['hour'] = df.time.dt.hour
df.drop('time', axis=1, inplace=True)

# %%
df.head()

# %%
df=df.sort_values(by='datetime')

# %%
df.head()

# %%
del df['datetime']

# %%
df.head()

# %%
#convering all categorical columns to numerical.
df1=pd.get_dummies(df)

# %%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# %%
df1.head()

# %%
#creating target and features objects 
x = df1.drop(columns=['electricity_consumption'])
y = df1.iloc[:,4]

# %%
#implementing selectKbest
st=time.time()
bestfeatures = SelectKBest(score_func=f_regression)
fit = bestfeatures.fit(x,y)
et=time.time()-st
print(et)

# %%
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Featuress','Score']
best_features=featureScores.nlargest(5,'Score')

# %%
best_features

# %%
df1.shape

# %%
test=df1.tail(7940)

# %%
test1=test.head(7440)

# %%
train=df1.head(18556)

# %%
pred=test.tail(500)

# %%
# train-test-validation split
test=df1.tail(7940)
#test set
test1=test.head(7440)
#training set
train=df1.head(18556)
#validation set
pred=test.tail(500)

# %%
test1.tail()

 # %%
 pred.tail()

# %%
y_train=train.iloc[:,4]

# %%
X_train=train.drop(columns=['electricity_consumption'])

# %%
y_test=test1.iloc[:,4]

# %%
X_test=test1.drop(columns=['electricity_consumption'])

# %%
y_pred=pred.iloc[:,4]

# %%
X_pred=pred.drop(columns=['electricity_consumption'])

# %%
import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 100, alpha = 10, n_estimators = 140)
xg_reg.fit(X_train,y_train)

# %%
from sklearn.metrics import mean_squared_error
predictions = xg_reg.predict(X_test)
errors = abs(predictions - y_test)
mape = 100 * np.mean(errors / y_test)
mse=mean_squared_error(y_test,predictions)
RMSE=np.sqrt(mse)
print("XGBOOST model")
print("mape value for test set",mape)
print("mse value for test set",mse)
print("RMSE value for test set",RMSE)

# %%
predictions = xg_reg.predict(X_pred)
errors = abs(predictions - y_pred)
mape = 100 * np.mean(errors / y_pred)
mse=mean_squared_error(y_pred,predictions)
RMSE=np.sqrt(mse)
print("XGBOOST model")
print("mape value for validation set",mape)
print("mse value for validation set",mse)
print("RMSE value for validation set",RMSE)

# %%
from lightgbm import LGBMRegressor
lgb_reg = LGBMRegressor(n_estimators=100, random_state=42)
lgb_reg.fit(X_train, y_train)

# %%
from sklearn.metrics import mean_squared_error
predictions = lgb_reg.predict(X_test)
errors = abs(predictions - y_test)
mape = 100 * np.mean(errors / y_test)
mse=mean_squared_error(y_test,predictions)
RMSE=np.sqrt(mse)
print("LIGHTGBM model")
print("mape value for test set",mape)
print("mse value for test set",mse)
print("RMSE value for test set",RMSE)

# %%
predictions = lgb_reg.predict(X_pred)
errors = abs(predictions - y_pred)
mape = 100 * np.mean(errors / y_pred)
mse=mean_squared_error(y_pred,predictions)
RMSE=np.sqrt(mse)
print("LIGHTGBM model")
print("mape value for validation set",mape)
print("mse value for validation set",mse)
print("RMSE value for validation set",RMSE)

# %%
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=100, random_state=42)
regr.fit(X_train, y_train)

# %%
from sklearn.metrics import mean_squared_error
predictions = regr.predict(X_test)
errors = abs(predictions - y_test)
mape = 100 * np.mean(errors / y_test)
mse=mean_squared_error(y_test,predictions)
RMSE=np.sqrt(mse)
print("RANDOM FOREST model")
print("mape value for test set",mape)
print("mse value for test set",mse)
print("RMSE value for test set",RMSE)

# %%
predictions = regr.predict(X_pred)
errors = abs(predictions - y_pred)
mape = 100 * np.mean(errors / y_pred)
mse=mean_squared_error(y_pred,predictions)
RMSE=np.sqrt(mse)
print("RANDOM FOREST model")
print("mape value for validation set",mape)
print("mse value for validation set",mse)
print("RMSE value for validation set",RMSE)


# %%
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    mse=mean_squared_error(test_labels,predictions)
    RMSE=np.sqrt(mse)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('RMSE = {:0.2f}'.format(RMSE))
    return accuracy,predictions,RMSE


# %%
models=[xg_reg,lgb_reg,regr]
model_name=['XGBoost','LightGBM','RandomForest']
model_RMSE=[]
model_predictions=[]
for item in models:
    base_accuracy,predictions,RMSE=evaluate(item,X_test,y_test)
    model_RMSE.append(RMSE)
    model_predictions.append(predictions)
r=model_RMSE.index(min(model_RMSE))
best_model_predictions=model_predictions[r]
best_model_name=model_name[r]
best_model=models[r]    

# %%
print('Best Model:')
print(best_model_name)
print('Model Object:')
print(best_model)
print('Predictions:')
print(best_model_predictions)

# %%
#Plot timeseries
y_test=pd.DataFrame(y_test)

y_test['predictions']=best_model_predictions

X_test['datetime']=pd.to_datetime(X_test[['year','month','day','hour']])

y_test['datetime']=X_test['datetime']

y_test=y_test.sort_values(by='datetime')

trace0 = go.Scatter(x=y_test['datetime'].astype(str), y=y_test['electricity_consumption'].values, opacity = 0.8, name='actual_value')
trace1 = go.Scatter(x=y_test['datetime'].astype(str), y=y_test['predictions'].values, opacity = 0.8, name='prediction')
layout = dict(
    title= "Prediction vs actual:",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=12, label='12m', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible = True),
        type='date'
    )
)
fig = dict(data= [trace0,trace1], layout=layout)
iplot(fig)

# %%
models=[xg_reg,lgb_reg,regr]
model_name=['XGBoost','LightGBM','RandomForest']
model_RMSE=[]
model_predictions=[]
for item in models:
    base_accuracy,predictions,RMSE=evaluate(item,X_pred,y_pred)
    model_RMSE.append(RMSE)
    model_predictions.append(predictions)
r=model_RMSE.index(min(model_RMSE))
best_model_predictions=model_predictions[r]
best_model_name=model_name[r]
best_model=models[r]

# %%
print('Best Model:')
print(best_model_name)
print('Model Object:')
print(best_model)
print('Predictions:')
print(best_model_predictions)

# %%
#Plot timeseries
y_pred=pd.DataFrame(y_pred)

y_pred['predictions']=best_model_predictions

X_pred['datetime']=pd.to_datetime(X_pred[['year','month','day','hour']])

y_pred['datetime']=X_pred['datetime']

y_pred=y_pred.sort_values(by='datetime')

trace0 = go.Scatter(x=y_pred['datetime'].astype(str), y=y_pred['electricity_consumption'].values, opacity = 0.8, name='actual_value')
trace1 = go.Scatter(x=y_pred['datetime'].astype(str), y=y_pred['predictions'].values, opacity = 0.8, name='prediction')
layout = dict(
    title= "Prediction vs actual:",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=12, label='12m', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible = True),
        type='date'
    )
)
fig = dict(data= [trace0,trace1], layout=layout)
iplot(fig)
