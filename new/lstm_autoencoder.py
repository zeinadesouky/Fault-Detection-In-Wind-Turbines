import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


from tensorflow import keras
model=keras.models.load_model('./savedModel/my_model')


def LSTMPrediction(test):
    # THRESHOLDS 
    train_mae_loss = np.load('./savedModel/np_array.npy')
    standarddev=np.std(train_mae_loss, axis=0) 
    means=np.mean(train_mae_loss, axis=0) 
    threshold0=means[0]+3*standarddev[0]
    threshold1=means[1]+3*standarddev[1]
    threshold2=means[2]+3*standarddev[2]
    threshold3=means[3]+3*standarddev[3]
    threshold4=means[4]+3*standarddev[4]

    # PREPROCESSING 
    # test=pd.read_csv('blog\datasets\Turbine6_test.csv',',')
    # test2=pd.read_csv('blog\datasets\Turbine6_test.csv',',')
  
    test2= test
    test2['Timestamp'] = pd.to_datetime(test2['Timestamp']) #test2 has all features
    test2Timestamp=test2['Timestamp']

    test=test.drop(columns='Timestamp') # test.csv has 5 features (components only)
    test=test.astype('float')
     
    scaler = MinMaxScaler().fit(test)
    scaled_features = scaler.fit_transform(test.values)
    scaled_features_df = pd.DataFrame(scaled_features, index=test.index, columns=test.columns)
    test=scaled_features_df

    # LSTM PREPROCESSING
    TIME_STEPS=144
    def create_sequences( X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            u = y.iloc[i:(i + time_steps)].values
            ys.append(v)
        return np.array(Xs), np.array(ys)

    
    X_test, Y_test = create_sequences(test, test, TIME_STEPS)


    # DATA PREDICTION
    X_test_pred = model.predict(X_test, verbose=0)
    test_mae_loss = np.max(np.abs(X_test_pred-X_test), axis=1)


    # GENERATOR BEARING
    threshold0=means[0]+8*standarddev[0]
    def generator_bearing():
        dataframes_list_html=[]
        test_score_df0 = pd.DataFrame(test[TIME_STEPS:])
        test_score_df0['loss'] = test_mae_loss[:, 0]
        test_score_df0['threshold'] = threshold0
        test_score_df0['anomaly'] = test_score_df0['loss'] > test_score_df0['threshold']
        test_score_df0['Gen_Bear_Temp_Avg'] = test[TIME_STEPS:]['Gen_Bear_Temp_Avg']
        test_score_df0.insert(loc=0, column='Timestamp',value=test2Timestamp)
        test_score_df0=test_score_df0.drop(columns=['Gen_Phase1_Temp_Avg','Hyd_Oil_Temp_Avg','HVTrafo_Phase2_Temp_Avg', 'Gear_Bear_Temp_Avg','Gear_Oil_Temp_Avg','loss','threshold'])
        anomalies0 = test_score_df0.loc[test_score_df0['anomaly'] == True]

        if not anomalies0.empty:
            anomalies0 = anomalies0.resample('M', on='Timestamp').mean()
            anomalies0=anomalies0.dropna()
            anomalies0=anomalies0.reset_index()
            csv_data= anomalies0.to_csv(r'C:\Users\user\Desktop\Fault Detection App\LSTM_Anomalies\generatorbearing_anomalies.csv',index=False)
            dataframes_list_html.append(anomalies0.to_html(index=False))
            dataframes_list_html = u''.join(dataframes_list_html)
            return dataframes_list_html

        else:
            return 'WARNING: No Failures Pending'
    
    # GENERATOR TEMPERATURE 
    threshold1=means[1]+8*standarddev[1]
    def generator_temperature():
        dataframes_list_html=[]
        test_score_df1 = pd.DataFrame(test[TIME_STEPS:])
        test_score_df1['loss'] = test_mae_loss[:, 1]
        test_score_df1['threshold'] = threshold1
        test_score_df1['anomaly'] = test_score_df1['loss'] > test_score_df1['threshold']
        test_score_df1['Gen_Phase1_Temp_Avg'] = test[TIME_STEPS:]['Gen_Phase1_Temp_Avg']
        test_score_df1.insert(loc=0, column='Timestamp',value=test2Timestamp)
        test_score_df1=test_score_df1.drop(columns=['Gen_Bear_Temp_Avg','Hyd_Oil_Temp_Avg','HVTrafo_Phase2_Temp_Avg', 'Gear_Bear_Temp_Avg','Gear_Oil_Temp_Avg','loss','threshold'])
        anomalies1 = test_score_df1.loc[test_score_df1['anomaly'] == True]

        if not anomalies1.empty:
            anomalies1 = anomalies1.resample('M', on='Timestamp').mean()
            anomalies1=anomalies1.dropna()
            anomalies1=anomalies1.reset_index()
            csv_data= anomalies1.to_csv(r'C:\Users\user\Desktop\Fault Detection App\LSTM_Anomalies\generatortemp_anomalies.csv',index=False)
            dataframes_list_html.append(anomalies1.to_html(index=False))
            dataframes_list_html = u''.join(dataframes_list_html)
            return dataframes_list_html

        else:
            return 'WARNING: No Failures Pending'
    
    # HYDRAULIC OIL
    threshold2=means[2]+8*standarddev[2]
    def hydraulic_group():
        dataframes_list_html=[]
        test_score_df2 = pd.DataFrame(test[TIME_STEPS:])
        test_score_df2['loss'] = test_mae_loss[:, 1]
        test_score_df2['threshold'] = threshold2
        test_score_df2['anomaly'] = test_score_df2['loss'] > test_score_df2['threshold']
        test_score_df2['Hyd_Oil_Temp_Avg'] = test[TIME_STEPS:]['Hyd_Oil_Temp_Avg']
        test_score_df2.insert(loc=0, column='Timestamp',value=test2Timestamp)
        test_score_df2=test_score_df2.drop(columns=['Gen_Bear_Temp_Avg','Gen_Phase1_Temp_Avg','HVTrafo_Phase2_Temp_Avg', 'Gear_Bear_Temp_Avg','Gear_Oil_Temp_Avg','loss','threshold'])
        anomalies2 = test_score_df2.loc[test_score_df2['anomaly'] == True]

        if not anomalies2.empty:
            anomalies2 = anomalies2.resample('M', on='Timestamp').mean()
            anomalies2=anomalies2.dropna()
            anomalies2=anomalies2.reset_index()
            csv_data= anomalies2.to_csv(r'C:\Users\user\Desktop\Fault Detection App\LSTM_Anomalies\hydraulic_anomalies.csv',index=False)
            dataframes_list_html.append(anomalies2.to_html(index=False))
            dataframes_list_html = u''.join(dataframes_list_html)
            return dataframes_list_html

        else:
            return 'WARNING: No Failures Pending'
    
    # TRANSFORMER
    threshold3=means[3]+8*standarddev[3]
    def transformer():
        dataframes_list_html=[]
        test_score_df3 = pd.DataFrame(test[TIME_STEPS:])
        test_score_df3['loss'] = test_mae_loss[:, 1]
        test_score_df3['threshold'] = threshold3
        test_score_df3['anomaly'] = test_score_df3['loss'] > test_score_df3['threshold']
        test_score_df3['HVTrafo_Phase2_Temp_Avg'] = test[TIME_STEPS:]['HVTrafo_Phase2_Temp_Avg']
        test_score_df3.insert(loc=0, column='Timestamp',value=test2Timestamp)
        test_score_df3=test_score_df3.drop(columns=['Gen_Bear_Temp_Avg','Gen_Phase1_Temp_Avg','Hyd_Oil_Temp_Avg', 'Gear_Bear_Temp_Avg','Gear_Oil_Temp_Avg','loss','threshold'])
        anomalies3 = test_score_df3.loc[test_score_df3['anomaly'] == True]

        if not anomalies3.empty:
            anomalies3 = anomalies3.resample('M', on='Timestamp').mean()
            anomalies3=anomalies3.dropna()
            anomalies3=anomalies3.reset_index()
            csv_data= anomalies3.to_csv(r'C:\Users\user\Desktop\Fault Detection App\LSTM_Anomalies\transformer_anomalies.csv',index=False)
            dataframes_list_html.append(anomalies3.to_html(index=False))
            dataframes_list_html = u''.join(dataframes_list_html)
            return dataframes_list_html

        else:
            return 'WARNING: No Failures Pending'
    
    # GEARBOX BEARING 
    threshold4=means[4]+8*standarddev[4]
    def gearbox_bearing():
        dataframes_list_html=[]
        test_score_df4 = pd.DataFrame(test[TIME_STEPS:])
        test_score_df4['loss'] = test_mae_loss[:, 1]
        test_score_df4['threshold'] = threshold4
        test_score_df4['anomaly'] = test_score_df4['loss'] > test_score_df4['threshold']
        test_score_df4['Gear_Bear_Temp_Avg'] = test[TIME_STEPS:]['Gear_Bear_Temp_Avg']
        test_score_df4.insert(loc=0, column='Timestamp',value=test2Timestamp)
        test_score_df4=test_score_df4.drop(columns=['Gen_Bear_Temp_Avg','Gen_Phase1_Temp_Avg','Hyd_Oil_Temp_Avg', 'HVTrafo_Phase2_Temp_Avg','Gear_Oil_Temp_Avg','loss','threshold'])
        anomalies4 = test_score_df4.loc[test_score_df4['anomaly'] == True]

        if not anomalies4.empty:
            anomalies4 = anomalies4.resample('M', on='Timestamp').mean()
            anomalies4=anomalies4.dropna()
            anomalies4=anomalies4.reset_index()
            csv_data= anomalies4.to_csv(r'C:\Users\user\Desktop\Fault Detection App\LSTM_Anomalies\gearboxbearing_anomalies.csv',index=False)
            dataframes_list_html.append(anomalies4.to_html(index=False))
            dataframes_list_html = u''.join(dataframes_list_html)
            return dataframes_list_html

        else:
            return 'WARNING: No Failures Pending'
    
    # GEARBOX OIL
    threshold5=means[5]+8*standarddev[5]
    def gearbox_oil():
        dataframes_list_html=[]
        test_score_df5 = pd.DataFrame(test[TIME_STEPS:])
        test_score_df5['loss'] = test_mae_loss[:, 1]
        test_score_df5['threshold'] = threshold5
        test_score_df5['anomaly'] = test_score_df5['loss'] > test_score_df5['threshold']
        test_score_df5['Gear_Bear_Temp_Avg'] = test[TIME_STEPS:]['Gear_Bear_Temp_Avg']
        test_score_df5.insert(loc=0, column='Timestamp',value=test2Timestamp)
        test_score_df5=test_score_df5.drop(columns=['Gen_Bear_Temp_Avg','Gen_Phase1_Temp_Avg','Hyd_Oil_Temp_Avg', 'HVTrafo_Phase2_Temp_Avg','Gear_Bear_Temp_Avg','loss','threshold'])
        anomalies5 = test_score_df5.loc[test_score_df5['anomaly'] == True]
        
        if not anomalies5.empty:
            anomalies5 = anomalies5.resample('M', on='Timestamp').mean()
            anomalies5=anomalies5.dropna()
            anomalies5=anomalies5.reset_index()
            csv_data= anomalies5.to_csv(r'C:\Users\user\Desktop\Fault Detection App\LSTM_Anomalies\gearboxoil_anomalies.csv',index=False)

            dataframes_list_html.append(anomalies5.to_html(index=False))
            dataframes_list_html = u''.join(dataframes_list_html)
            return dataframes_list_html

        else:
            return 'WARNING: No Failures Pending'

    generator_bearing_results=generator_bearing()
    generator_temperature_results=generator_temperature()
    hydraulic_group_results=hydraulic_group()
    transformer_results=transformer()
    gearbox_bearing_results=gearbox_bearing()
    gearbox_oil_results=gearbox_oil()
    

    return generator_bearing_results, generator_temperature_results, hydraulic_group_results,transformer_results, gearbox_bearing_results, gearbox_oil_results