import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


from tensorflow import keras
model=keras.models.load_model('./savedModel/my_model')

def lstm(x):
    TIME_STEPS=30
    #test=pd.DataFrame(columns = ['Turbine_ID','Timestamp','Gen_Bear_Temp_Avg','Gen_Phase1_Temp_Avg','Hyd_Oil_Temp_Avg','HVTrafo_Phase2_Temp_Avg', 'Gear_Bear_Temp_Avg'])
    
    test=x
    test2=x
    test2['Timestamp'] = pd.to_datetime(test2['Timestamp'])

    test=test.drop(columns='Timestamp')
    test=test.astype('float')
     
    scaler = MinMaxScaler().fit(test)
    scaled_features = scaler.fit_transform(test.values)
    scaled_features_df = pd.DataFrame(scaled_features, index=test.index, columns=test.columns)
    test=scaled_features_df

    def create_sequences( X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            u = y.iloc[i:(i + time_steps)].values
            ys.append(v)
        return np.array(Xs), np.array(ys)
    
    X_test, Y_test = create_sequences(test, test, TIME_STEPS)

    X_test_pred = model.predict(X_test, verbose=0)
    test_mae_loss = np.max(np.abs(X_test_pred-X_test), axis=1)

    threshold=0.25


    def generator_bearing():
        test_score_df0 = pd.DataFrame(test[TIME_STEPS:])
        test_score_df0['loss'] = test_mae_loss[:, 0]
        test_score_df0['threshold'] = threshold
        test_score_df0['anomaly'] = test_score_df0['loss'] > test_score_df0['threshold']
        test_score_df0['Gen_Bear_Temp_Avg'] = test[TIME_STEPS:]['Gen_Bear_Temp_Avg']
        testTimestamp=test2['Timestamp']
        test_score_df0.insert(loc=0, column='Timestamp',value=testTimestamp)
        test_score_df0=test_score_df0.drop(columns=['Gen_Phase1_Temp_Avg','Hyd_Oil_Temp_Avg','HVTrafo_Phase2_Temp_Avg', 'Gear_Bear_Temp_Avg'])
        anomalies = test_score_df0.loc[test_score_df0['anomaly'] == True]

        failures=pd.read_csv('blog\datasets\wind_farm_failures.csv',',')
        failures=failures[failures['Turbine_ID'] == 'T06']
        failures = failures.sort_values(by='Timestamp',ascending=True)
        failures=failures.reset_index()
        d = {}
        for i in range(len(failures)):
            d[i] = pd.DataFrame()
        
        failures=failures.drop(columns=['index'])
        failures= pd.to_datetime(failures['Timestamp'])
        
        for i in range(3,len(failures)):
            d[i]=anomalies.loc[((anomalies['Timestamp'] >= failures[i-1]) & (anomalies['Timestamp'] < failures[i])) ]
        for i in range(3,len(failures)):
            timestampfail=d[i]
            if not timestampfail.empty:
                timestampfail=timestampfail['Timestamp']
                timestampfail=timestampfail.reset_index(drop=True)
                return 'The failure is predicted' +' '+ str(failures[i]-timestampfail.loc[0]) +' '+'prior to the actual fault.'
            else:
                return "No fault detected in generator bearing!"

    def hydraulic_group():
        test_score_df2 = pd.DataFrame(test[TIME_STEPS:])
        test_score_df2['loss'] = test_mae_loss[:, 0]
        test_score_df2['threshold'] = threshold
        test_score_df2['anomaly'] = test_score_df2['loss'] > test_score_df2['threshold']
        test_score_df2['Hyd_Oil_Temp_Avg'] = test[TIME_STEPS:]['Hyd_Oil_Temp_Avg']
        testTimestamp=test2['Timestamp']
        test_score_df2.insert(loc=0, column='Timestamp',value=testTimestamp)
        test_score_df2=test_score_df2.drop(columns=['Gen_Phase1_Temp_Avg','Gen_Bear_Temp_Avg','HVTrafo_Phase2_Temp_Avg', 'Gear_Bear_Temp_Avg'])
        anomalies2 = test_score_df2.loc[test_score_df2['anomaly'] == True]

        failures=pd.read_csv('blog\datasets\wind_farm_failures.csv',',')
        failures=failures[failures['Turbine_ID'] == 'T06']
        failures = failures.sort_values(by='Timestamp',ascending=True)
        failures=failures.reset_index()
        d = {}
        for i in range(len(failures)):
            d[i] = pd.DataFrame()
        
        failures=failures.drop(columns=['index'])
        failures= pd.to_datetime(failures['Timestamp'])
        
        for i in range(3,len(failures)):
            d[i]=anomalies2.loc[((anomalies2['Timestamp'] >= failures[i-1]) & (anomalies2['Timestamp'] < failures[i])) ]
        for i in range(3,len(failures)):
            timestampfail2=d[i]
            if not timestampfail.empty:
                timestampfail=timestampfail['Timestamp']
                timestampfail=timestampfail.reset_index(drop=True)
                return 'The failure is predicted' +' '+ str(failures[i]-timestampfail.loc[0]) +' '+'prior to the actual fault.'
            else:
                return "No fault detected in hydraulic group!"
    
    # def gearbox():
    #     test_score_df1 = pd.DataFrame(test[TIME_STEPS:])
    #     test_score_df1['loss'] = test_mae_loss[:, 0]
    #     test_score_df1['threshold'] = threshold
    #     test_score_df1['anomaly'] = test_score_df1['loss'] > test_score_df1['threshold']
    #     test_score_df1['Gear_Bear_Temp_Avg'] = test[TIME_STEPS:]['Gear_Bear_Temp_Avg']
    #     testTimestamp=test2['Timestamp']
    #     test_score_df1.insert(loc=0, column='Timestamp',value=testTimestamp)
    #     test_score_df1=test_score_df1.drop(columns=['Gen_Phase1_Temp_Avg','Hyd_Oil_Temp_Avg','HVTrafo_Phase2_Temp_Avg', 'Gen_Phase1_Temp_Avg'])
    #     anomalies1 = test_score_df1.loc[test_score_df1['anomaly'] == True]

    #     failures=pd.read_csv('blog\datasets\wind_farm_failures.csv',',')
    #     failures=failures[failures['Turbine_ID'] == 'T06']
    #     failures = failures.sort_values(by='Timestamp',ascending=True)
    #     failures=failures.reset_index()
    #     d = {}
    #     for i in range(len(failures)):
    #         d[i] = pd.DataFrame()
        
    #     failures=failures.drop(columns=['index'])
    #     failures= pd.to_datetime(failures['Timestamp'])
        
    #     for i in range(3,len(failures)):
    #         d[i]=anomalies1.loc[((anomalies1['Timestamp'] >= failures[i-1]) & (anomalies1['Timestamp'] < failures[i])) ]
    #     for i in range(3,len(failures)):
    #         timestampfail=d[i]
    #         if not timestampfail.empty:
    #             timestampfail=timestampfail['Timestamp']
    #             timestampfail=timestampfail.reset_index(drop=True)
    #             return 'The failure is predicted' +' '+ str(failures[i]-timestampfail.loc[0]) +' '+'prior to the actual fault.'
    #         else:
    #             return "No fault detected in Gearbox!"

    # def transformer():
    #     test_score_df3 = pd.DataFrame(test[TIME_STEPS:])
    #     test_score_df3['loss'] = test_mae_loss[:, 0]
    #     test_score_df3['threshold'] = threshold
    #     test_score_df3['anomaly'] = test_score_df3['loss'] > test_score_df3['threshold']
    #     test_score_df3['HVTrafo_Phase2_Temp_Avg'] = test[TIME_STEPS:]['HVTrafo_Phase2_Temp_Avg']
    #     testTimestamp=test2['Timestamp']
    #     test_score_df3.insert(loc=0, column='Timestamp',value=testTimestamp)
    #     test_score_df3=test_score_df3.drop(columns=['Gen_Phase1_Temp_Avg','Hyd_Oil_Temp_Avg'])
    #     anomalies3 = test_score_df3.loc[test_score_df3['anomaly'] == True]

    #     failures=pd.read_csv('blog\datasets\wind_farm_failures.csv',',')
    #     failures=failures[failures['Turbine_ID'] == 'T06']
    #     failures = failures.sort_values(by='Timestamp',ascending=True)
    #     failures=failures.reset_index()
    #     d = {}
    #     for i in range(len(failures)):
    #         d[i] = pd.DataFrame()
        
    #     failures=failures.drop(columns=['index'])
    #     failures= pd.to_datetime(failures['Timestamp'])
        
    #     for i in range(3,len(failures)):
    #         d[i]=anomalies3.loc[((anomalies3['Timestamp'] >= failures[i-1]) & (anomalies3['Timestamp'] < failures[i])) ]
    #     for i in range(3,len(failures)):
    #         timestampfail=d[i]
    #         if not timestampfail.empty:
    #             timestampfail=timestampfail['Timestamp']
    #             timestampfail=timestampfail.reset_index(drop=True)
    #             return 'The failure is predicted' +' '+ str(failures[i]-timestampfail.loc[0]) +' '+'prior to the actual fault.'
    #         else:
    #             return "No fault detected in Transformer!"

    # def generator():
    #     test_score_df4 = pd.DataFrame(test[TIME_STEPS:])
    #     test_score_df4['loss'] = test_mae_loss[:, 0]
    #     test_score_df4['threshold'] = threshold
    #     test_score_df4['anomaly'] = test_score_df4['loss'] > test_score_df4['threshold']
    #     test_score_df4['Gen_Phase1_Temp_Avg'] = test[TIME_STEPS:]['Gen_Phase1_Temp_Avg']
    #     testTimestamp=test2['Timestamp']
    #     test_score_df4.insert(loc=0, column='Timestamp',value=testTimestamp)
    #     test_score_df4=test_score_df4.drop(columns=['Gen_Phase1_Temp_Avg','Hyd_Oil_Temp_Avg'])
    #     anomalies4 = test_score_df4.loc[test_score_df4['anomaly'] == True]

    #     failures=pd.read_csv('blog\datasets\wind_farm_failures.csv',',')
    #     failures=failures[failures['Turbine_ID'] == 'T06']
    #     failures = failures.sort_values(by='Timestamp',ascending=True)
    #     failures=failures.reset_index()
    #     d = {}
    #     for i in range(len(failures)):
    #         d[i] = pd.DataFrame()
        
    #     failures=failures.drop(columns=['index'])
    #     failures= pd.to_datetime(failures['Timestamp'])
        
    #     for i in range(3,len(failures)):
    #         d[i]=anomalies4.loc[((anomalies4['Timestamp'] >= failures[i-1]) & (anomalies4['Timestamp'] < failures[i])) ]
    #     for i in range(3,len(failures)):
    #         timestampfail=d[i]
    #         if not timestampfail.empty:
    #             timestampfail=timestampfail['Timestamp']
    #             timestampfail=timestampfail.reset_index(drop=True)
    #             return 'The failure is predicted' +' '+ str(failures[i]-timestampfail.loc[0]) +' '+'prior to the actual fault.'
    #         else:
    #             return 'The failure is predicted' +' '+ str(failures[i]-timestampfail.loc[0]) +' '+'prior to the actual fault.'
    
    result='Generator Bearing:'+ generator_bearing()
    result2=' Generator:'+ generator()
    result3=' Hydraulic group:'+hydraulic_group()
    result4=' Gearbox:'+gearbox()
    return result+'\n'+result2+'\n'+result3+'\n'+result4
