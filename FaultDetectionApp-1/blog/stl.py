from tensorflow import keras
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import time
from statsmodels.tsa.seasonal import STL
from django.core.files.storage import FileSystemStorage
import os
np.random.seed(1)
tf.random.set_seed(1)

def WeatherFaultDetection(df):
    
    # df = pd.read_csv(df)
    # df=pd.read_csv('blog\datasets\weather_pca.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp',ascending=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.date
    df.set_index('Timestamp',inplace=True)
    df.index=pd.to_datetime(df.index)    
    
    #PREPROCESSING
    df = df.groupby('Timestamp')[['Min_Windspeed2','Avg_Winddirection2','Max_Humidity','Max_AmbientTemp','Max_Pressure',]].sum()
    df.copy()
    df[['Min_Windspeed2','Avg_Winddirection2','Max_Humidity','Max_AmbientTemp','Max_Pressure',]] = (df[['Min_Windspeed2','Avg_Winddirection2','Max_Humidity','Max_AmbientTemp','Max_Pressure',]] - df[['Min_Windspeed2','Avg_Winddirection2','Max_Humidity','Max_AmbientTemp','Max_Pressure',]].min()) / (df[['Min_Windspeed2','Avg_Winddirection2','Max_Humidity','Max_AmbientTemp','Max_Pressure',]].max() - df[['Min_Windspeed2','Avg_Winddirection2','Max_Humidity','Max_AmbientTemp','Max_Pressure',]].min())
    
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)
    
    start_date = datetime(2016,1,1)
    start_month = start_date.month
    end_date = datetime(2016,12,31)
    end_month = end_date.month

    
    
    # COMPONENETS
    # WIND DIRECTION 
    def wind_direction():
        dataframes_list_html=[]
        stl_winddirection = STL(df['Avg_Winddirection2'], seasonal = 13, period=30)
        result_winddirection = stl_winddirection.fit()
        seasonal_winddirection, trend_winddirection, resid_winddirection = result_winddirection.seasonal, result_winddirection.trend, result_winddirection.resid
        estimated_winddirection = trend_winddirection + seasonal_winddirection
        resid_mu = resid_winddirection.mean()
        resid_dev = resid_winddirection.std()
        lower = resid_mu - 3*resid_dev
        upper = resid_mu + 3*resid_dev
        df0 = pd.DataFrame(df['Avg_Winddirection2'])
        anomalies0 = df0[(resid_winddirection < lower) | (resid_winddirection > upper)]
        anomalies0=anomalies0.reset_index()
        anomalies0['Timestamp']=pd.to_datetime(anomalies0['Timestamp'])
        if not anomalies0.empty:
            anomalies0=anomalies0.resample('M', on='Timestamp').mean()
            anomalies0=anomalies0.dropna()
            anomalies0=anomalies0.reset_index()
            
            csv_data= anomalies0.to_csv(r'/Users/nourelgarhy/Downloads/Fault Detection App-1/STL_Anomalies/wind_direction_anomalies.csv',index=False)
            dataframes_list_html.append(anomalies0.to_html(index=False))
            dataframes_list_html = u''.join(dataframes_list_html)
            return dataframes_list_html

        else:
            return 'WARNING: No Anomalies Found'
        
    
    def wind_speed():
        dataframes_list_html=[]
        stl_windspeed = STL(df['Min_Windspeed2'], seasonal = 13, period=30)
        result_windspeed = stl_windspeed.fit()
        seasonal_windspeed, trend_windspeed, resid_windspeed = result_windspeed.seasonal, result_windspeed.trend, result_windspeed.resid
        estimated_windspeed = trend_windspeed + seasonal_windspeed
        resid_mu1 = resid_windspeed.mean()
        resid_dev1 = resid_windspeed.std()
        lower1 = resid_mu1 - 3*resid_dev1
        upper1 = resid_mu1 + 3*resid_dev1

        df1 = pd.DataFrame(df['Min_Windspeed2'])
        anomalies1 = df1[(resid_windspeed < lower1) | (resid_windspeed > upper1)]
        anomalies1=anomalies1.reset_index()
        anomalies1['Timestamp']=pd.to_datetime(anomalies1['Timestamp'])
        if not anomalies1.empty:
            anomalies1=anomalies1.resample('M', on='Timestamp').mean()
            anomalies1=anomalies1.dropna()
            anomalies1=anomalies1.reset_index()
            
            csv_data= anomalies1.to_csv(r'/Users/nourelgarhy/Downloads/Fault Detection App-1/STL_Anomalies/wind_speed_anomalies.csv',index=False)
            dataframes_list_html.append(anomalies1.to_html(index=False))
            dataframes_list_html = u''.join(dataframes_list_html)
            return dataframes_list_html
        else:
            return 'WARNING: No Anomalies Found'
    
    def humidity():
        dataframes_list_html=[]
        stl_humidity = STL(df['Max_Humidity'], seasonal = 13, period=30)
        result_humidity = stl_humidity.fit()
        seasonal_humidity, trend_humidity, resid_humidity = result_humidity.seasonal, result_humidity.trend, result_humidity.resid
        estimated_humidity = trend_humidity + seasonal_humidity
        resid_mu = resid_humidity.mean()
        resid_dev = resid_humidity.std()
        lower = resid_mu - 3*resid_dev
        upper = resid_mu + 3*resid_dev

        df2 = pd.DataFrame(df['Max_Humidity'])
        anomalies2 = df2[(resid_humidity < lower) | (resid_humidity > upper)]
        anomalies2=anomalies2.reset_index()
        anomalies2['Timestamp']=pd.to_datetime(anomalies2['Timestamp'])

        if not anomalies2.empty:
            anomalies2=anomalies2.resample('M', on='Timestamp').mean()
            anomalies2=anomalies2.dropna()
            anomalies2=anomalies2.reset_index()
            csv_data= anomalies2.to_csv(r'/Users/nourelgarhy/Downloads/Fault Detection App-1/STL_Anomalies/humidity_anomalies.csv',index=False)
            dataframes_list_html.append(anomalies2.to_html(index=False))
            dataframes_list_html = u''.join(dataframes_list_html)
            return dataframes_list_html
        else:
            return 'WARNING: No Anomalies Found'
    
    def ambient_temp():
        dataframes_list_html=[]
        stl_ambient = STL(df['Max_AmbientTemp'], seasonal = 13, period=30)
        result_ambient = stl_ambient.fit()
        seasonal_ambient, trend_ambient, resid_ambient = result_ambient.seasonal, result_ambient.trend, result_ambient.resid
        estimated_ambient = trend_ambient + seasonal_ambient
        resid_mu = resid_ambient.mean()
        resid_dev = resid_ambient.std()
        
        lower = resid_mu - 3*resid_dev
        upper = resid_mu + 3*resid_dev
        
        df3 = pd.DataFrame(df['Max_AmbientTemp'])
        anomalies3 = df3[(resid_ambient < lower) | (resid_ambient > upper)]
        anomalies3=anomalies3.reset_index()
        anomalies3['Timestamp']=pd.to_datetime(anomalies3['Timestamp'])
        if not anomalies3.empty:
            anomalies3=anomalies3.resample('M', on='Timestamp').mean()
            anomalies3=anomalies3.dropna()
            anomalies3=anomalies3.reset_index()
            csv_data= anomalies3.to_csv(r'/Users/nourelgarhy/Downloads/Fault Detection App-1/STL_Anomalies/Ambient_temperature_anomalies.csv',index=False)
            dataframes_list_html.append(anomalies3.to_html(index=False))
            dataframes_list_html = u''.join(dataframes_list_html)
            return dataframes_list_html
        else:
            return 'WARNING: No Anomalies Found'
        
    def pressure():
        dataframes_list_html=[]
        stl_pressure = STL(df['Max_Pressure'], seasonal = 13, period=30)
        result_pressure = stl_pressure.fit()
        seasonal_pressure, trend_pressure, resid_pressure = result_pressure.seasonal, result_pressure.trend, result_pressure.resid
        estimated_pressure = trend_pressure + seasonal_pressure
        resid_mu = resid_pressure.mean()
        resid_dev = resid_pressure.std()
        
        lower = resid_mu - 3*resid_dev
        upper = resid_mu + 3*resid_dev
        df4 = pd.DataFrame(df['Max_Pressure'])
        anomalies4 = df4[(resid_pressure < lower) | (resid_pressure > upper)]
        
        anomalies4=anomalies4.reset_index()
        anomalies4['Timestamp']=pd.to_datetime(anomalies4['Timestamp'])
        if not anomalies4.empty:
            anomalies4=anomalies4.resample('M', on='Timestamp').mean()
            anomalies4=anomalies4.dropna()
            anomalies4=anomalies4.reset_index()
            csv_data= anomalies4.to_csv(r'/Users/nourelgarhy/Downloads/Fault Detection App-1/STL_Anomalies/pressure_anomalies.csv',index=False)
            dataframes_list_html.append(anomalies4.to_html(index=False))
            dataframes_list_html = u''.join(dataframes_list_html)
            return dataframes_list_html
        else:
            return 'WARNING: No Anomalies Found'
        
    
    wind_direction=wind_direction()
    wind_speed=wind_speed()
    humidity=humidity()
    ambient_temp=ambient_temp()
    pressure=pressure()
    print(pressure)
    return wind_direction, wind_speed, humidity, ambient_temp, pressure

 
