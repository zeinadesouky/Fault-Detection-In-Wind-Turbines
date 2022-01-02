import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pandas import read_csv

from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Model
import seaborn as sns

#load the dataset
dataframe = read_csv('scada_data.csv')
#dataframe=read_csv('fixed/wind_farm_signals.csv')
dataframe.head()

#CREATE PLOT
df = dataframe[['time', 'wind_speed','kw']]
#df = dataframe[['Timestamp', 'Amb_WindSpeed_Avg']]
#sns.lineplot(x=df['time'], y=df['wind_speed'])

print("Start date is: ", df['time'].min())
print("End date is: ", df['time'].max())

#TEST AND TRAIN
train, test = df.loc[df['time'] <= '2015-12-11'], df.loc[df['time'] > '2015-12-11']

# RESHAPE DATA
seq_size = 30  # Number of time steps to look back 
#Larger sequences (look further back) may improve forecasting.

def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x)-seq_size):
        #print(i)
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)

trainX, trainY = to_sequences(train[['wind_speed']], train['kw'], seq_size)
testX, testY = to_sequences(test[['wind_speed']], test['kw'], seq_size)

# LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(rate=0.2))

model.add(RepeatVector(trainX.shape[1]))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(trainX.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()

# fit model
history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()