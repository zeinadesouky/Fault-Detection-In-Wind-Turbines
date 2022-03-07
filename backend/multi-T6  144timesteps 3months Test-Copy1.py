#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed


# In[2]:


df = pd.read_csv('turbine6.csv')


# In[3]:


df.head()


# In[4]:


df = df[['Timestamp','Gen_Bear_Temp_Avg','Gen_Phase1_Temp_Avg','Hyd_Oil_Temp_Avg','HVTrafo_Phase2_Temp_Avg','Gear_Bear_Temp_Avg','Gear_Oil_Temp_Avg',]]


# In[5]:


df.dtypes


# In[6]:


df['Timestamp'] = pd.to_datetime(df['Timestamp'])


# In[7]:


df['Timestamp'].min(), df['Timestamp'].max()


# In[8]:


train, test = df.loc[df['Timestamp'] <= '2016-09-01 00:00:00+0000'], df.loc[df['Timestamp'] > '2016-09-01 00:00:00+0000']


# In[9]:


train.shape,test.shape


# In[10]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Gen_Phase1_Temp_Avg'], name='Gen_Phase1_Temp_Avg'))
fig.update_layout(showlegend=True, title='Gen_Phase1_Temp_Avg 2016')
fig.show()


# In[11]:


timestamp=df['Timestamp']
train, test = df.loc[df['Timestamp'] <= '2016-09-01 00:00:00+0000'], df.loc[df['Timestamp'] > '2016-09-01 00:00:00+0000']


# In[12]:


timestamp=df['Timestamp']
train2, test2 = df.loc[df['Timestamp'] <= '2016-09-01 00:00:00+0000'], df.loc[df['Timestamp'] > '2016-09-01 00:00:00+0000']


# In[13]:


train=train.drop(['Timestamp'], axis=1)
test=test.drop(['Timestamp'], axis=1)

#train2=train2.drop(['Timestamp'], axis=1)
#test2=test2.drop(['Timestamp'], axis=1)


# In[14]:


def create_dataset( X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        u = y.iloc[i:(i + time_steps)].values
        ys.append(v)
    return np.array(Xs), np.array(ys)


# In[15]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(train)
#train = pd.DataFrame(scaler.transform(train))
#test = pd.DataFrame(scaler.transform(test))


scaled_features = scaler.fit_transform(train.values)
scaled_features_df = pd.DataFrame(scaled_features, index=train2.index, columns=train2.drop(['Timestamp'], axis=1).columns)


# In[16]:


scaled_features2 = scaler.fit_transform(test.values)
scaled_features_df2 = pd.DataFrame(scaled_features2, index=test2.index, columns=test2.drop(['Timestamp'], axis=1).columns)


# In[17]:


train=scaled_features_df


# In[18]:


test=scaled_features_df2


# In[19]:


train


# In[20]:


test


# In[21]:


TIME_STEPS=144
X_train,Y_train = create_dataset(train, train, TIME_STEPS)
X_test, Y_test = create_dataset(test, test, TIME_STEPS)


# In[ ]:





# In[22]:


#Python
model = Sequential()
model.add(LSTM(units=128, input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(n=X_train.shape[1]))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2])))
model.compile(optimizer='adam', loss='mae')


# In[23]:


#history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.1, shuffle=False)


history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)


# In[24]:


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend();


# In[25]:


model.evaluate(X_test, Y_test)


# In[26]:


X_train_pred = model.predict(X_train, verbose=0)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE loss')
plt.ylabel('Number of Samples');

# threshold = np.max(train_mae_loss)
# print(f'Reconstruction error threshold: {threshold}')


# In[27]:


X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.max(np.abs(X_test_pred-X_test), axis=1)

plt.hist(test_mae_loss, bins=50)
plt.xlabel('Test MAE loss')
plt.ylabel('Number of samples');

threshold = np.mean(test_mae_loss)+3*np.std(test_mae_loss)
print(f'Reconstruction error threshold: {threshold}')

standarddev=np.std(train_mae_loss, axis=0) 
means=np.mean(train_mae_loss, axis=0)   # Maxima along the first axis
threshold0=means[0]+3*standarddev[0]
threshold1=means[1]+3*standarddev[1]
threshold2=means[2]+3*standarddev[2]
threshold3=means[3]+3*standarddev[3]
threshold4=means[4]+3*standarddev[4]

print(threshold0,threshold1,threshold2, threshold3,threshold4)


# In[ ]:





# ### Gen_Bear_Temp_Avg

# In[29]:


threshold0=means[0]+8*standarddev[0]


# In[30]:


test_score_df0 = pd.DataFrame(test[TIME_STEPS:])
test_score_df0['loss'] = test_mae_loss[:, 0]
test_score_df0['threshold'] =threshold0
test_score_df0['anomaly'] = test_score_df0['loss'] > test_score_df0['threshold']
test_score_df0['Gen_Bear_Temp_Avg'] = test[TIME_STEPS:]['Gen_Bear_Temp_Avg']


# In[31]:


test2Timestamp=test2['Timestamp']


# In[32]:


test2Timestamp


# In[33]:


test_score_df0.insert(loc=0, column='Timestamp',value=test2Timestamp)
test_score_df0


# In[34]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df0['Timestamp'], y=test_score_df0['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df0['Timestamp'], y=test_score_df0['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
fig.show()


# In[35]:


anomalies = test_score_df0.loc[test_score_df0['anomaly'] == True]
#anomalies.tail()


# In[36]:


anomalies.shape


# In[37]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df0['Timestamp'], y=test_score_df0['Gen_Bear_Temp_Avg'], name='Gen_Bear_Temp_Avg'))
fig.add_trace(go.Scatter(x=anomalies['Timestamp'], y=anomalies['Gen_Bear_Temp_Avg'], mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
fig.show()


# In[38]:


### Failures


# In[39]:


failures=pd.read_csv('htw-failures-2016.csv',';')


# In[40]:


failures=failures[failures['Turbine_ID'] == 'T06']
failures = failures.sort_values(by='Timestamp',ascending=True)

failures=failures.reset_index()
d = {}
for i in range(len(failures)):
    #print(failures.loc[i])
    d[i] = pd.DataFrame()
    
failures=failures.drop(columns=['index'])

#failures.loc[3]
#d


# In[41]:


failures= pd.to_datetime(failures['Timestamp'])


# In[42]:


for i in range(3,len(failures)):
    d[i]=anomalies.loc[((anomalies['Timestamp'] >= failures[i-1]) & (anomalies['Timestamp'] < failures[i])) ]


# In[43]:


for i in range(3,len(failures)):
    timestampfail=d[i]
    if not timestampfail.empty:
        timestampfail=timestampfail['Timestamp']
    #d[i]
        timestampfail=timestampfail.reset_index(drop=True)
        timestampfail
    #failuredf1 = anomalies.loc[anomalies['Timestamp'] < failures[i]]
    #timef1=failuredf1['Timestamp']
    #timef1  
        timestampfail
        print(failures[i]-timestampfail.loc[0])
    else:
        print("no anomaly detected for this fault")
   


# In[44]:


test_mae_loss


# In[ ]:





# ## Gen_Phase1_Temp_Avg

# In[45]:


threshold1=means[1]+8*standarddev[1]
test_score_df1 = pd.DataFrame(test[TIME_STEPS:])
test_score_df1['loss'] = test_mae_loss[:, 1]
test_score_df1['threshold'] = threshold1
test_score_df1['anomaly'] = test_score_df1['loss'] > test_score_df1['threshold']
test_score_df1['Gen_Phase1_Temp_Avg'] = test[TIME_STEPS:]['Gen_Phase1_Temp_Avg']

test_score_df1.insert(loc=0, column='Timestamp',value=test2Timestamp)
#test_score_df0


# In[46]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df1['Timestamp'], y=test_score_df1['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df1['Timestamp'], y=test_score_df1['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
fig.show()


# In[47]:


#anomalies.tail()
#anomalies.shape


# In[48]:


anomalies1 = test_score_df1.loc[test_score_df1['anomaly'] == True]
#anomalies.tail()
anomalies1.shape

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df1['Timestamp'], y=test_score_df1['Gen_Phase1_Temp_Avg'], name='Gen_Phase1_Temp_Avg'))
fig.add_trace(go.Scatter(x=anomalies1['Timestamp'], y=anomalies1['Gen_Phase1_Temp_Avg'], mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
fig.show()


# In[49]:


#anomalies.tail()
anomalies1.shape


# In[50]:


failures


# ##### Failures

# In[51]:


d = {}
for i in range(len(failures)):
    #print(failures.loc[i])
    d[i] = pd.DataFrame()
    
failures=failures.drop(columns=['index'])

#failures.loc[3]
#d


# In[52]:


#failures= pd.to_datetime(failures['Timestamp'])
for i in range(3,len(failures)):
    d[i]=anomalies1.loc[((anomalies1['Timestamp'] >= failures[i-1]) & (anomalies1['Timestamp'] < failures[i])) ]
    
for i in range(3,len(failures)):
    timestampfail=d[i]
    if not timestampfail.empty:
        timestampfail=timestampfail['Timestamp']
    #d[i]
        timestampfail=timestampfail.reset_index(drop=True)
        timestampfail
    #failuredf1 = anomalies.loc[anomalies['Timestamp'] < failures[i]]
    #timef1=failuredf1['Timestamp']
    #timef1  
        timestampfail
        print(failures[i]-timestampfail.loc[0])
    else:
        print("no anomaly detected for this fault")
   


# ## Hyd_Oil_Temp_Avg

# In[53]:


threshold2=means[2]+8*standarddev[2]
test_score_df2 = pd.DataFrame(test[TIME_STEPS:])
test_score_df2['loss'] = test_mae_loss[:, 2]
test_score_df2['threshold'] = threshold2
test_score_df2['anomaly'] = test_score_df2['loss'] > test_score_df2['threshold']
test_score_df2['Hyd_Oil_Temp_Avg'] = test[TIME_STEPS:]['Hyd_Oil_Temp_Avg']

test_score_df2.insert(loc=0, column='Timestamp',value=test2Timestamp)
#test_score_df0


# In[54]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df2['Timestamp'], y=test_score_df2['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df2['Timestamp'], y=test_score_df2['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
fig.show()


# In[55]:


anomalies2 = test_score_df2.loc[test_score_df2['anomaly'] == True]
#anomalies.tail()
anomalies2.shape

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df2['Timestamp'], y=test_score_df2['Hyd_Oil_Temp_Avg'], name='Hyd_Oil_Temp_Avg'))
fig.add_trace(go.Scatter(x=anomalies2['Timestamp'], y=anomalies2['Hyd_Oil_Temp_Avg'], mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
fig.show()


# ###### Failures

# In[56]:


d = {}
for i in range(len(failures)):
    #print(failures.loc[i])
    d[i] = pd.DataFrame()
    
failures=failures.drop(columns=['index'])

#failures.loc[3]
#d


# In[57]:


#failures= pd.to_datetime(failures['Timestamp'])
for i in range(3,len(failures)):
    d[i]=anomalies2.loc[((anomalies2['Timestamp'] >= failures[i-1]) & (anomalies2['Timestamp'] < failures[i])) ]
    
for i in range(3,len(failures)):
    timestampfail=d[i]
    if not timestampfail.empty:
        timestampfail=timestampfail['Timestamp']
    #d[i]
        timestampfail=timestampfail.reset_index(drop=True)
        timestampfail
    #failuredf1 = anomalies.loc[anomalies['Timestamp'] < failures[i]]
    #timef1=failuredf1['Timestamp']
    #timef1  
        timestampfail
        print(failures[i]-timestampfail.loc[0])
    else:
        print("no anomaly detected for this fault")
   


# In[58]:


threshold1


# ## HVTrafo_Phase2_Temp_Avg	

# In[64]:


threshold3=means[3]+16*standarddev[3]
test_score_df3 = pd.DataFrame(test[TIME_STEPS:])
test_score_df3['loss'] = test_mae_loss[:, 3]
test_score_df3['threshold'] = threshold3
test_score_df3['anomaly'] = test_score_df3['loss'] > test_score_df3['threshold']
test_score_df3['HVTrafo_Phase2_Temp_Avg'] = test[TIME_STEPS:]['HVTrafo_Phase2_Temp_Avg']

test_score_df3.insert(loc=0, column='Timestamp',value=test2Timestamp)
#test_score_df0


# In[65]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df3['Timestamp'], y=test_score_df3['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df3['Timestamp'], y=test_score_df3['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
fig.show()


# In[66]:


anomalies3 = test_score_df3.loc[test_score_df3['anomaly'] == True]
#anomalies.tail()
anomalies3.shape

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df3['Timestamp'], y=test_score_df3['HVTrafo_Phase2_Temp_Avg'], name='HVTrafo_Phase2_Temp_Avg'))
fig.add_trace(go.Scatter(x=anomalies3['Timestamp'], y=anomalies3['HVTrafo_Phase2_Temp_Avg'], mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
fig.show()


# In[67]:


###### Failures


# In[68]:


d = {}
for i in range(len(failures)):
    #print(failures.loc[i])
    d[i] = pd.DataFrame()
    
failures=failures.drop(columns=['index'])

#failures.loc[3]
#d


# In[69]:


#failures= pd.to_datetime(failures['Timestamp'])
for i in range(3,len(failures)):
    d[i]=anomalies3.loc[((anomalies3['Timestamp'] >= failures[i-1]) & (anomalies3['Timestamp'] < failures[i])) ]
    
for i in range(3,len(failures)):
    timestampfail=d[i]
    if not timestampfail.empty:
        timestampfail=timestampfail['Timestamp']
    #d[i]
        timestampfail=timestampfail.reset_index(drop=True)
        timestampfail
    #failuredf1 = anomalies.loc[anomalies['Timestamp'] < failures[i]]
    #timef1=failuredf1['Timestamp']
    #timef1  
        timestampfail
        print(failures[i]-timestampfail.loc[0])
    else:
        print("no anomaly detected for this fault")
   


# ## Gear_Bear_Temp_Avg

# In[77]:


#threshold4 = np.max(train_mae_loss[:, 4])
threshold4=means[4]+8*standarddev[4]


# In[78]:


test_score_df4 = pd.DataFrame(test[TIME_STEPS:])
test_score_df4['loss'] = test_mae_loss[:, 4]
test_score_df4['threshold'] = threshold4
test_score_df4['anomaly'] = test_score_df4['loss'] > test_score_df4['threshold']
test_score_df4['Gear_Bear_Temp_Avg'] = test[TIME_STEPS:]['Gear_Bear_Temp_Avg']

test_score_df4.insert(loc=0, column='Timestamp',value=test2Timestamp)
#test_score_df0


# In[79]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df4['Timestamp'], y=test_score_df4['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df4['Timestamp'], y=test_score_df4['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
fig.show()


# In[80]:


anomalies4 = test_score_df4.loc[test_score_df4['anomaly'] == True]
#anomalies.tail()
anomalies4.shape

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df4['Timestamp'], y=test_score_df4['Gear_Bear_Temp_Avg'], name='Gear_Bear_Temp_Avg'))
fig.add_trace(go.Scatter(x=anomalies4['Timestamp'], y=anomalies4['Gear_Bear_Temp_Avg'], mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
fig.show()


# In[81]:


##Failures


# In[82]:


d = {}
for i in range(len(failures)):
    #print(failures.loc[i])
    d[i] = pd.DataFrame()
    
failures=failures.drop(columns=['index'])

#failures.loc[3]
#d


# In[83]:


#failures= pd.to_datetime(failures['Timestamp'])
for i in range(3,len(failures)):
    d[i]=anomalies4.loc[((anomalies4['Timestamp'] >= failures[i-1]) & (anomalies4['Timestamp'] < failures[i])) ]
    
for i in range(3,len(failures)):
    timestampfail=d[i]
    if not timestampfail.empty:
        timestampfail=timestampfail['Timestamp']
    #d[i]
        timestampfail=timestampfail.reset_index(drop=True)
        timestampfail
    #failuredf1 = anomalies.loc[anomalies['Timestamp'] < failures[i]]
    #timef1=failuredf1['Timestamp']
    #timef1  
        timestampfail
        print(failures[i]-timestampfail.loc[0])
    else:
        print("no anomaly detected for this fault")
   


# ## Gear_Oil_Temp_Avg

# In[84]:


#threshold4 = np.max(train_mae_loss[:, 4])
threshold5=means[5]+8*standarddev[5]


# In[85]:



test_score_df5 = pd.DataFrame(test[TIME_STEPS:])
test_score_df5['loss'] = test_mae_loss[:, 5]
test_score_df5['threshold'] = threshold5
test_score_df5['anomaly'] = test_score_df5['loss'] > test_score_df5['threshold']
test_score_df5['Gear_Oil_Temp_Avg'] = test[TIME_STEPS:]['Gear_Oil_Temp_Avg']

test_score_df5.insert(loc=0, column='Timestamp',value=test2Timestamp)
#test_score_df0


# In[86]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df5['Timestamp'], y=test_score_df5['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df5['Timestamp'], y=test_score_df5['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
fig.show()


# In[87]:


anomalies5 = test_score_df5.loc[test_score_df5['anomaly'] == True]
#anomalies.tail()
anomalies1.shape

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df5['Timestamp'], y=test_score_df5['Gear_Oil_Temp_Avg'], name='Gear_Oil_Temp_Avg'))
fig.add_trace(go.Scatter(x=anomalies5['Timestamp'], y=anomalies5['Gear_Oil_Temp_Avg'], mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
fig.show()


# ##### Failures

# In[88]:


d = {}
for i in range(len(failures)):
    #print(failures.loc[i])
    d[i] = pd.DataFrame()
    
failures=failures.drop(columns=['index'])

#failures.loc[3]
#d


# In[89]:


#failures= pd.to_datetime(failures['Timestamp'])
for i in range(3,len(failures)):
    d[i]=anomalies5.loc[((anomalies1['Timestamp'] >= failures[i-1]) & (anomalies5['Timestamp'] < failures[i])) ]
    
for i in range(3,len(failures)):
    timestampfail=d[i]
    if not timestampfail.empty:
        timestampfail=timestampfail['Timestamp']
    #d[i]
        timestampfail=timestampfail.reset_index(drop=True)
        timestampfail
    #failuredf1 = anomalies.loc[anomalies['Timestamp'] < failures[i]]
    #timef1=failuredf1['Timestamp']
    #timef1  
        timestampfail
        print(failures[i]-timestampfail.loc[0])
    else:
        print("no anomaly detected for this fault")
   


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[103]:


train2,test2= df.loc[df['Timestamp'] <= '2016-09-01 00:00:00+0000'], df.loc[df['Timestamp'] > '2016-09-01 00:00:00+0000']


# In[104]:


test2Timestamp=test2['Timestamp']


# In[105]:


test2Timestamp


# In[106]:


test_score_df.insert(loc=0, column='Timestamp',value=test2Timestamp)


# In[107]:


test_score_df


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['Timestamp'], y=test_score_df['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df['Timestamp'], y=test_score_df['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
fig.show()


# In[ ]:


anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
anomalies.tail()


# In[ ]:


anomalies.shape


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['Timestamp'], y=test_score_df['Gen_RPM_Max'], name='Gen_Phase1_Temp_Avg'))
fig.add_trace(go.Scatter(x=anomalies['Timestamp'], y=anomalies['Gen_RPM_Max'], mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
fig.show()


# # Failures

# In[ ]:


failures=pd.read_csv('htw-failures-2016.csv',';')
failures=failures[failures['Turbine_ID'] == 'T06']
failures = failures.sort_values(by='Timestamp',ascending=True)

failures=failures.reset_index()
d = {}
for i in range(len(failures)):
    #print(failures.loc[i])
    d[i] = pd.DataFrame()
    
failures=failures.drop(columns=['index'])

failures.loc[3]


# In[ ]:


d


# In[ ]:


failures


# In[ ]:


for i in range(3,len(failures)):
    d[i]=anomalies.loc[((anomalies['Timestamp'] >= failures[i-1]) & (anomalies['Timestamp'] < failures[i])) ]


# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


from sklearn.preprocessing import MinMaxScaler
# Python
# ensure all data is float
values = df.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# In[ ]:


df = df.set_index('Turbine_ID')

