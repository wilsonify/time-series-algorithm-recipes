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

# %% id="yuk5geVFQ6fZ"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.preprocessing
from sklearn.metrics import r2_score
from keras.layers import Dense,Dropout,SimpleRNN,LSTM
from keras.models import Sequential

# %% id="CE0Ti6TdQsLx" outputId="16656a99-1283-4cf7-c7b3-f9173525b289"
#Plotting hourly energy usage:

AEP = pd.read_csv('../input/hourly-energy-consumption/AEP_hourly.csv', index_col=[0], parse_dates=[0])

mau = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
bieudo = AEP.plot(style='.',figsize=(15,5), color=mau[0], title='AEP')
    
#Data transformation
def create_features(df, label=None):
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

X, y = create_features(AEP, label='AEP_MW')
features_and_target = pd.concat([X, y], axis=1)
print(features_and_target)
plt.show()

plt.figure(figsize=(15,6))
data_csv = AEP.dropna()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: (x-min_value) / scalar, dataset))
plt.plot(dataset)
print(max_value, min_value)

# %% id="l6lkLrGXQ0s2" outputId="57808b66-b776-49d0-de89-c941b915c93e"
#choosing DOM_hourly.csv data for analysis
fpath='../input/hourly-energy-consumption/DOM_hourly.csv'

#Let's use datetime(2012-10-01 12:00:00,...) as index instead of numbers(0,1,...)
#This will be helpful for further data analysis as we are dealing with time series data
df = pd.read_csv(fpath, index_col='Datetime', parse_dates=['Datetime'])
df.head()

#checking missing data
df.isna().sum()

#Data visualization

df.plot(figsize=(16,4),legend=True)

plt.title('DOM hourly power consumption data - BEFORE NORMALIZATION')

plt.show()


# %% id="V1DFwbMURQNA" outputId="45618fa4-cf96-4790-e617-ab8807bb6fee"
#Normalize DOM hourly power consumption data

def normalize_data(df):
    scaler = sklearn.preprocessing.MinMaxScaler()
    df['DOM_MW']=scaler.fit_transform(df['DOM_MW'].values.reshape(-1,1))
    return df

df_norm = normalize_data(df)
df_norm.shape

#Visualize data after normalization

df_norm.plot(figsize=(16,4),legend=True)

plt.title('DOM hourly power consumption data - AFTER NORMALIZATION')

plt.show()


# %% id="owlSH0vYRUSc"
# train data for deep learning models

def load_data(stock, seq_len):
    X_train = []
    y_train = []
    for i in range(seq_len, len(stock)):
        X_train.append(stock.iloc[i - seq_len: i, 0])
        y_train.append(stock.iloc[i, 0])

    # 1 last 6189 days are going to be used in test
    X_test = X_train[110000:]
    y_test = y_train[110000:]

    # 2 first 110000 days are going to be used in training
    X_train = X_train[:110000]
    y_train = y_train[:110000]

    # 3 convert to numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 4 reshape data to input into RNN models
    X_train = np.reshape(X_train, (110000, seq_len, 1))

    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))

    return [X_train, y_train, X_test, y_test]


# %% id="nH2AUUjARXTG" outputId="9d9a539c-807d-4a27-b3e1-e878d376522c"
#create train, test data
seq_len = 20 #choose sequence length

X_train, y_train, X_test, y_test = load_data(df, seq_len)

print('X_train.shape = ',X_train.shape)
print('y_train.shape = ', y_train.shape)
print('X_test.shape = ', X_test.shape)
print('y_test.shape = ',y_test.shape)

# %% id="UiE2EELWRZU7" outputId="16dee05a-ba04-4b46-f7ac-73e1b9e3e98f"
#RNN model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
rnn_model.add(Dropout(0.15))
rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True))
rnn_model.add(Dropout(0.15))
rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=False))
rnn_model.add(Dropout(0.15))
rnn_model.add(Dense(1))
rnn_model.summary()
rnn_model.compile(optimizer="adam",loss="MSE")
rnn_model.fit(X_train, y_train, epochs=10, batch_size=1000)

# %% id="HV2ytNwGRfAM" outputId="d49d978f-9678-43f7-dc4c-1514e588fee8"
#r2 score for the values predicted by the above trained SIMPLE RNN model

rnn_predictions = rnn_model.predict(X_test)
rnn_score = r2_score(y_test,rnn_predictions)
print("R2 Score of RNN model = ",rnn_score)


# %% id="RFPnh0YDRiZ8" outputId="9571a02f-5615-4298-d068-2c98c8db9454"
# compare the actual values vs predicted values by plotting a graph

def plot_predictions(test, predicted, title):
    plt.figure(figsize=(16, 4))
    plt.plot(test, color='blue', label='Actual power consumption data')
    plt.plot(predicted, alpha=0.7, color='orange', label='Predicted power consumption data')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized power consumption scale')
    plt.legend()
    plt.show()
plot_predictions(y_test, rnn_predictions, "Predictions made by simple RNN model")

# %% id="PrgqnLPFRliQ" outputId="ea41ccb1-d786-482b-ade9-d0b8ce5c7f80"
#train model for LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
lstm_model.add(Dropout(0.15))
lstm_model.add(LSTM(40,activation="tanh",return_sequences=True))
lstm_model.add(Dropout(0.15))
lstm_model.add(LSTM(40,activation="tanh",return_sequences=False))
lstm_model.add(Dropout(0.15))
lstm_model.add(Dense(1))
lstm_model.summary()
lstm_model.compile(optimizer="adam",loss="MSE")
lstm_model.fit(X_train, y_train, epochs=10, batch_size=1000)

# %% id="VGebEaFuRn0x" outputId="ccee91e6-e08e-409a-c03e-ce2e48cfd085"
#r2 score for the values predicted by the above trained LSTM model
lstm_predictions = lstm_model.predict(X_test)
lstm_score = r2_score(y_test, lstm_predictions)
print("R^2 Score of LSTM model = ",lstm_score)

# %% id="5ebGmPngRuzo" outputId="df264352-860e-4f67-8b1c-5e46f8d97a84"
#actual values vs predicted values by plotting a graph
plot_predictions(y_test, lstm_predictions, "Predictions made by LSTM model")

# %% id="n75opihRRxlQ" outputId="ef01bb78-a1f1-4550-e973-a9ef9de28cf4"
#RNN, LSTM model by plotting data in a single graph
plt.figure(figsize=(15,8))
plt.plot(y_test, c="orange", linewidth=3, label="Original values")
plt.plot(lstm_predictions, c="red", linewidth=3, label="LSTM predictions")
plt.plot(rnn_predictions, alpha=0.5, c="blue", linewidth=3, label="RNN predictions")
plt.legend()
plt.title("Predictions vs actual data", fontsize=20)
plt.show()

# %% id="TEKndsuVRz4U"
#GRU

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
train_data = pd.read_csv("../input/daily-climate-time-series-data/DailyDelhiClimateTrain.csv",index_col=0)
# Display dimensions of dataframe
print(train_data.shape)
print(train_data.info())

# %%
print("-----------------------------------------------------------------------")
print("Original dataset  : \n",train_data.sample(10)) 

# %%
print("-----------------------------------------------------------------------")
# Display statistics for numeric columns
print(train_data.describe())

# %%
print("-----------------------------------------------------------------------") 
train_data.plot(figsize=(12,8),subplots=True)

# %%
# Now lets plot them all
train_data.hist(figsize=(12, 16), bins=50, xlabelsize=8, ylabelsize=8)

# %%
# To check Missing Values 
print("null values : \n",train_data.isnull().sum()) 
sns.heatmap(train_data.isnull(), cbar=False, yticklabels=False, cmap="viridis")

# %%
# We choose a specific feature (features). In this example,
my_dataset = train_data[["meantemp",'humidity','wind_speed','meanpressure']]


print("Our new dataset : \n",my_dataset.sample(5))

# %%
print("-----------------------------------------------------------------------")
# ensure all data is float
my_dataset = my_dataset.astype("float32")
values     = my_dataset.values
print("values : \n",values)

# %%
print("-----------------------------------------------------------------------")
# normalize features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print("scaled : \n",scaled)

# %%
values.shape


# %%
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
              names += [("var%d(t)" % (j+1)) for j in range(n_vars)]
        else:
              names += [("var%d(t+%d)" % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
# ******************************************************************************


# %%
# frame as supervised learning
# reshape into X=t and Y=t+1
i_in  = 100 # past observations
n_out = 1 # future observations
reframed = series_to_supervised(scaled, i_in, n_out)
print("Represent the dataset as a supervised learning problem : \n",reframed.head(10))
print("-----------------------------------------------------------------------")

# %%
# split into train and test sets
# convert an array of values into a dataset matrix
values_spl = reframed.values
train_size = int(len(values_spl) * 0.80)
test_size  = len(values_spl) - train_size
train, test = values_spl[0:train_size,:], values_spl[train_size:len(values_spl),:]
print("len train and test : ",len(train), "  ", len(test))

# %%
print("-----------------------------------------------------------------------")
# split into input and outputs
X_train, y_train = train[:, :-4], train[:, -4:]
X_test, y_test   = test[:, :-4],  test[:, -4:]

print("X_train shape : ",X_train.shape," y_train shape : ",y_train.shape)
print("X_test shape  : ",X_test.shape, " y_test shape  : ",y_test.shape)

# %%
print("-----------------------------------------------------------------------")
# reshape input to be 3D [samples, timesteps, features]
# The LSTM network expects the input data (X) to be provided with 
# a specific array structure in the form of: [samples, time steps, features].
# Currently, our data is in the form: [samples, features] 
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print("X_train shape 3D : ",X_train.shape," y_train shape : ",y_train.shape)
print("X_test shape  3D : ",X_test.shape, " y_test shape  : ",y_test.shape)

# %%
#import and define the layers
model = keras.models.Sequential()
model.add(keras.layers.GRU(64, return_sequences=True, activation="relu", 
            kernel_initializer="he_normal", recurrent_initializer="he_normal", 
            dropout=0.15, recurrent_dropout=0.15,
						input_shape=(X_train.shape[1], X_train.shape[2]) ))
model.add(keras.layers.GRU(32,return_sequences=True, activation="relu", kernel_initializer="he_normal", 
            recurrent_initializer="he_normal", dropout=0.15, recurrent_dropout=0.15 ))
model.add(keras.layers.GRU(8, activation="relu", kernel_initializer="he_normal", 
            recurrent_initializer="he_normal", dropout=0.15, recurrent_dropout=0.15 ))
model.add(keras.layers.Dense(4, activation="relu"))

print(model.summary())

# %%
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True)

# %%
# Compiling the model
optimizer = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mse","mae"])

# Learning rate scheduling
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.00001, patience=3,
                                      monitor="val_loss", min_lr=0.00000001)

# %%
# Training and evaluating the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2,
                    callbacks=[lr_scheduler])

# %%
# plot the learning curves
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()



# %%
print("-----------------------------------------------------------------------")
# Evaluate the model
model_evaluate = model.evaluate(X_test, y_test)
print("Loss                   : ",model_evaluate[0])
print("Mean Squared Error     : ",model_evaluate[1])
print("Mean Absolute Error    : ",model_evaluate[2])  


# %%
# make predictions
trainPredict = model.predict(X_train)
testPredict  = model.predict(X_test)
print("trainPredict : ",trainPredict.shape)
print("testPredict  : ",testPredict.shape)


# %%
print(trainPredict)

# %%
testPredict = scaler.inverse_transform(testPredict)

# %%
print(testPredict.shape) 
print(y_test.shape)

# %%
y_test=scaler.inverse_transform(y_test)

# %%
#plot for meantemp

plt.plot(testPredict[:,0], color="blue", 
         label="Predict meantemp ", linewidth=2)

plt.plot(y_test[:,0], color="red", 
         label="Actual meantemp ", linewidth=2)

plt.legend()
# Show the major grid lines with dark grey lines
plt.grid(visible=True, which="major", color="#666666", linestyle="-")
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(visible=True, which="minor", color="#999999", linestyle="-", alpha=0.2)

plt.show()




# %%
#plot for humidity

plt.plot(testPredict[:,1], color="blue", 
         label="Predict humidity", linewidth=2)
plt.plot(y_test[:,1], color="red", 
         label="Actual humidity", linewidth=2)
plt.legend()

# Show the major grid lines with dark grey lines
plt.grid(visible=True, which="major", color="#666666", linestyle="-")
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(visible=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
plt.show()

# %%
#plot for windspeed

plt.plot(testPredict[:,2], color="blue", 
         label="predict wind_speed", linewidth=2)
plt.plot(y_test[:,2], color="red", 
         label="Actual wind_speed", linewidth=2)
plt.legend()

# Show the major grid lines with dark grey lines
plt.grid(visible=True, which="major", color="#666666", linestyle="-")
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(visible=True, which="minor", color="#999999", linestyle="-", alpha=0.2)

plt.show()

# %%
#plot for meanpressure

plt.plot(testPredict[:,3], color="blue", 
         label="predict meanpressure", linewidth=4)
plt.plot(y_test[:,3], color="red", 
         label="Actual meanpressure", linewidth=4)

plt.legend()

# Show the major grid lines with dark grey lines
plt.grid(visible=True, which="major", color="#666666", linestyle="-")
# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(visible=True, which="minor", color="#999999", linestyle="-", alpha=0.2)

plt.show()

# %%
# Neural Prophet

# %%
import pandas as pd 
from neuralprophet import NeuralProphet

# %%
#read and pre-process the data
df_np = pd.read_csv("DailyDelhiClimateTrain.csv", parse_dates=["date"]) 
df_np = df_np[["date", "meantemp"]] 
df_np.rename(columns={"date": "ds", "meantemp": "y"}, inplace=True)

# %%
# model = NeuralProphet() if you're using default variables below.
model = NeuralProphet(
    growth="linear",  # Determine trend types: 'linear', 'discontinuous', 'off'
    changepoints=None, # list of dates that may include change points (None -> automatic )
    n_changepoints=5,
    changepoints_range=0.8,
    trend_reg=0,
    trend_reg_threshold=False,
    yearly_seasonality="auto",
    weekly_seasonality="auto",
    daily_seasonality="auto",
    seasonality_mode="additive",
    seasonality_reg=0,
    n_forecasts=1,
    n_lags=0,
    num_hidden_layers=0,
    d_hidden=None,     # Dimension of hidden layers of AR-Net
    learning_rate=None,
    epochs=40,
    loss_func="Huber",
    normalize="auto",  # Type of normalization ('minmax', 'standardize', 'soft', 'off')
    impute_missing=True,
)

# %%
#make predictions
metrics = model.fit(df_np, freq="D") 
future = model.make_future_dataframe(df_np, periods=365, n_historic_predictions=len(df_np)) 
forecast = model.predict(future)

# %%
import matplotlib.pyplot as plt 
#forecast plot
fig, ax = plt.subplots(figsize=(14, 10)) 
model.plot(forecast, xlabel="Date", ylabel="Temp", ax=ax)
ax.set_title("Mean Temperature in Delhi", fontsize=28, fontweight="bold")

# %%
#plotting model parameters
model.plot_parameters()

# %%
#ploting Evaluation
fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(metrics["MAE"], 'ob', linewidth=6)  
ax.plot(metrics["RMSE"], '-r', linewidth=2)

# %%

# %%
