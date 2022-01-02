#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping 
from keras.wrappers.scikit_learn import KerasRegressor


# In[2]:


#read in data
df = pd.read_csv('data/hourly_nm.csv',index_col='Date/Time')
df.head()


# In[3]:


print(df.info())
print()
print(df.describe())


# In[4]:


#split data into training and testing, testing data will be one month
start_test = '2018-11-31'

train, test = df.loc[:start_test], df.loc[start_test:]


# In[5]:


train.tail(1)


# In[6]:


test.head(1)


# In[7]:


print(len(train))
print(len(test))


# In[8]:


# scale the data using MinMax Scaler from -1 to 1 as LSTM has a default tanh activation function
SCALER = MinMaxScaler(feature_range=(-1,1))

scaler = SCALER.fit(train.to_numpy())

train_scaled = scaler.transform(train.to_numpy())
test_scaled = scaler.transform(test.to_numpy())


# In[9]:


# create a function to split the datasets into two week windows
timestep = 24*7*2 # 24hours,7days,2weeks

def create_dataset(dataset, timestep=timestep):
    """
    Function which creates two week chunks of x_train data, and a single
    value for y_train.
    """
    X, y = [], []
    for i in range(len(dataset)):
        target_value = i + timestep
        if target_value == len(dataset):
            break
        feature_chunk, target = dataset[i:target_value, 1:], dataset[target_value, 0]
        X.append(feature_chunk)
        y.append(target)
    
    return np.array(X), np.array(y) 


# In[10]:


#create x_train, y_train, X_test,y_test
X_train, y_train = create_dataset(train_scaled)
X_test, y_test = create_dataset(test_scaled)


# In[11]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[12]:


# use sample of th data to train network to have a rough understanding of hyperparameters
samp_len = int(len(X_train)*0.5)

X_sample_train, y_sample_train = X_train[:samp_len], y_train[:samp_len]


# In[13]:


print(X_sample_train.shape)
print(y_sample_train.shape)


# In[14]:


# create X_train, y_train, X_test, y_test datasets
# create a function to build a stacked LSTM model
# input needs to be [samples, timesteps, features]
    def create_model(X_train, y_train):
        units = 32
        dropout = 0.05
        epochs = 35
        batch_size = 14
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        early_stopping = EarlyStopping(patience=7, monitor='loss')

        model = keras.Sequential()

        model.add(LSTM(units=units, dropout=dropout, return_sequences=True,
                       input_shape=(X_train.shape[1], X_train.shape[2])))
        
        model.add(LSTM(units=units, dropout=dropout))
        
        model.add(Dense(units=1))

        model.compile(optimizer=optimizer, loss='mean_squared_error')
        history = model.fit(X_train, y_train, validation_split=0.3, shuffle=False,
                  epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
       
        return model, history


# In[15]:


# function to predict a single value 
def single_prediction(model, history, timestep=timestep):
        
        history = np.array(history)
        history = history.reshape(history.shape[0]*history.shape[1], history.shape[2])
        
        input_value = history[-timestep:]
        input_value = input_value.reshape(1, input_value.shape[0], input_value.shape[1])
        
        yhat = model.predict(input_value, verbose=0)
        return yhat


# In[16]:


# function which takes first test chunk, makes a prediction, add the test chunk back into training data 
#to make next prediction

def walk_forward_prediction(X_train, y_train, X_test, timestep):
    
    MODEL, history = create_model(X_train=X_train, y_train=y_train)
    hist_train = [i for i in X_train]
    predictions = []
    
    for i in range(len(X_test)):
        test = X_test[i]
        yhat = single_prediction(model=MODEL, history=hist_train, timestep=timestep)
        predictions.append(yhat) 
        hist_train.append(test)
    
    return predictions, history, MODEL


# In[17]:


def prior_inverse(features, targets):
    '''
    Append prediction value to test dataset and return a test shape format.
    '''
    dataset = []
    
    for i in range(features.shape[0]):
        last_row, target = features[i][0], targets[i]
        appended = np.append(last_row, target)
        dataset.append(appended)
    
    return np.array(dataset) 


# In[18]:


#run experiemnt returning the real, predicted values
def experiment(X_train, y_train, X_test, timestep):
    
    pred_seq, history, MODEL = walk_forward_prediction(X_train, y_train, X_test, timestep)
    
    pred_seq = np.array(pred_seq).reshape(-1)

    pred = prior_inverse(X_test, pred_seq)
    real = prior_inverse(X_test, y_test)

    inv_pred = scaler.inverse_transform(pred)
    inv_real = scaler.inverse_transform(real)

    power_pred = inv_pred[:,-1]
    power_real = inv_real[:,-1]
    
    return power_real, power_pred, history, MODEL


# In[19]:


power_real, power_pred, history, MODEL = experiment(X_train, y_train, X_test, timestep)

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[22]:


#plot validation and training convergence graph
plt.figure(figsize=(10,5))
plt.plot(loss, label='train')
plt.plot(val_loss, label='validation')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.title('LSTM Training Validation Loss')
plt.tight_layout()
plt.savefig('figures/train_val_plot.png')
plt.show()


# In[30]:


x_plot = test[timestep:].index
pred_df = pd.DataFrame({'Date':x_plot, 'Prediction': power_pred, 'True': power_real})
pred_df.set_index('Date', inplace=True)


# In[31]:


pred_df2 = pred_df['2018-12-15 01:00:00	':'2018-12-29 02:00:00 ']


# In[32]:


#plot predictions
pred_df2.plot(rot='60',figsize=(10,5))
plt.title('Predicted Power vs Actual Power with LSTM an Model.')
plt.ylabel('Power(KWh)')
plt.tight_layout()
plt.savefig('figures/prediction.png')
plt.show()


# In[33]:


#compute metrics
rmse = np.sqrt(mean_squared_error(pred_df2['True'], pred_df2['Prediction']))
mae = mean_absolute_error(pred_df2['True'], pred_df2['Prediction'])
r2 = r2_score(pred_df2['True'], pred_df2['Prediction'])
print('RMSE: {}\nMAE: {}\nR2: {}'.format(round(rmse,2),round(mae,2), round(r2,2)))


# I decided to manually change the hyperparameters and record the results of the network to see the impact on the training and validation loss as well as making small adjustments to see the overall effect of the parameter. 

# In[34]:


results = {'Test':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
           'sample_size':[0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,1,1,1,1],
           'units':[8,8,8,8,8,12,16,16,16,32,32,32,32,64,64,32,32,32,32,32,32,32],
           'layers':[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
           'drop_out':[0.2,0.2,0.1,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
           'batch_size':[14,14,14,14,14,14,14,14,14,14,14,7,21,21,14,14,14,21,14,14,14,14],
           'learning_rate':[0.0001,0.0005,0.0005,0.0005,0.001,0.001,0.001,0.001,0.0005,0.0005,0.001,0.0005,0.0005,0.0005,0.0005,0.0005,0.001,0.0005,0.0005,0.0005,0.0005,0.0005],
           'epochs':[50,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,40,40,40,40,30,40],
           'RMSE':[689.87,603.55,577.52,525.36,507.46,484.93,478.51,477.91,489.13,476.17,478.19,487.14,479.56,535.83,497.55,487.31,498.46,535.63,496.71,493.76,496.2,496.35],
           'MAE':[544.42,509.95,476.48,381.44,370.11,342.71,343.28,348.48,332.21,334.14,344.09,349.62,336.19,394.55,343.45,320.26,328.72,389.17,337.31,334.41,338.35,342.69],
           'R2':['-','-','-','-','-','-','-','-',0.86,0.87,0.87,0.86,0.87,0.83,0.86,0.86,0.86,0.83,0.86,0.86,0.86,0.86]}


# In[35]:


results_df = pd.DataFrame(data=results)
results_df

