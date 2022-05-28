import pandas as pd

def posts():
        
    path5 = r"/Users/nourelgarhy/Downloads/Fault Detection App-1/blog/datasetss/allllll.csv"
  
    monthly_resampled_data = pd.read_csv(path5)

    posts3=[]

 
    posts4=[]

    path11 = r"/Users/nourelgarhy/Downloads/Fault Detection App-1/blog/datasetss/month1.csv"
    month1 = pd.read_csv(path11)

    path22 = r"/Users/nourelgarhy/Downloads/Fault Detection App-1/blog/datasetss/month2.csv"
    month22 = pd.read_csv(path22)

    path33 = r"/Users/nourelgarhy/Downloads/Fault Detection App-1/blog/datasetss/month3.csv"
    month33 = pd.read_csv(path33)

    path44 = r"/Users/nourelgarhy/Downloads/Fault Detection App-1/blog/datasetss/month4.csv"
    month4 = pd.read_csv(path44)
    


 # 'monthName': monthly_resampled_data['monthName'].values[i],

    for i in range(len(monthly_resampled_data)):
            x={
            'color':monthly_resampled_data['color'].values[i],
            'day':monthly_resampled_data['day'].values[i],
            'month': monthly_resampled_data['month'].values[i],
            'dayName': monthly_resampled_data['dayName'].values[i],
            'key': monthly_resampled_data['key'].values[i],
            'dayTill':monthly_resampled_data['dayTill'].values[i],
            'component':monthly_resampled_data['component'].values[i],
           

            }
            posts4.append(x)
    posts3.append(posts4)
    
    return posts4

def dictionary():
    Dict = {}

    import calendar
    dataframes_list_html = []
    path = r"/Users/nourelgarhy/Downloads/Fault Detection App-1/blog/datasetss/allllll.csv"
    
    print(path)
    print("wellllll3")
    
    temp_df = pd.read_csv(path)
    temp_df['month'] = pd.DatetimeIndex(temp_df['Timestamp']).month
    month2=pd.DataFrame()
    month2['month'] = temp_df['month'].unique()
    print(month2)
    print("-------len")
    print(len(month2))
    temp_df['key'] = temp_df.index


   
    
    for i in range(len(month2)):
        print('month22')
        print
        temp=month2.month[i]   
        

        dataframes_list_html.append(month2.to_html(index=False))
    print(dataframes_list_html)
    temp_df2 = pd.read_csv(path)

    temp_df2['month'] = pd.DatetimeIndex(temp_df2['Timestamp']).month
    temp_df2['year'] = pd.DatetimeIndex(temp_df2['Timestamp']).year
    month3=pd.DataFrame()
    month3['month'] = temp_df2['month'].unique()
    month3['year']=temp_df2['year']
    print(month3)

    rows = []
    dataframes_list_html2=[]
    dicts={}
    dicts2={}
    list=[]

    for i in range(len(month3)):
        month=month3.at[i,'month']
        year=month3.at[i,'year']

        from datetime import datetime
        x=year
        y=month
        z=1
        start_date = datetime(x,y,z)
        print(calendar.day_name[start_date.weekday()])
        Dict.update({i: calendar.day_name[start_date.weekday()],
        })
        print(Dict)

        dicts2={
            'day':i,
            'month':month,
            'dayname': calendar.day_name[start_date.weekday()],
        }
        list.append(dicts2)
        print(dicts2)

        dicts[i] = calendar.day_name[start_date.weekday()]
        rows.append(calendar.day_name[start_date.weekday()])
        
        df = pd.DataFrame (rows, columns = ['days'])
        
    
    return dicts, list