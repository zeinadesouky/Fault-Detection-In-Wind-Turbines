from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.http import JsonResponse
import pandas as pd
import io
import csv
import requests
import os
from django.contrib import messages
from django.views.generic import TemplateView

from blog.functions import dictionary, posts
from .forms import ContactForm
from IPython.display import display
from django.core.files.storage import FileSystemStorage
from .lstm_autoencoder import LSTMPrediction
from .stl import WeatherFaultDetection
# from csvvalidator import *
from tensorflow import keras
model=keras.models.load_model('./savedModel/my_model')



def home(request):
    return render(request, 'blog/home.html')


def about(request):
    return render(request, 'blog/about.html', {'title': 'About'})


# @login_required
def contact(request):
    contact_form = ContactForm()
    return render(request, 'blog/contact.html', {'contact_form': contact_form})


def help(request):
    return render(request, 'blog/help.html')



def faultDetectionView(request):

    if request.method == "GET":
        return render(request, 'blog/upload_csv.html')
    
    csv_file = request.FILES['file']

    if csv_file.size==0:
        message='Uploaded file is empty. Please try again.'
        return render(request, 'blog/upload_csv.html',{'message':message})
    elif not csv_file.name.endswith('.csv'):
       
        message='The uploaded file has to be CSV.  Please try again.'
        return render(request, 'blog/upload_csv.html',{'message':message})
    else:
        data_dict={}
        data_dict=pd.DataFrame(columns = ['Timestamp','Gen_Bear_Temp_Avg','Gen_Phase1_Temp_Avg','Hyd_Oil_Temp_Avg','HVTrafo_Phase2_Temp_Avg', 'Gear_Bear_Temp_Avg'])

        data_set = csv_file.read().decode('UTF-8')
        io_string = io.StringIO(data_set)
        next(io_string)
        reader=csv.reader(io_string, delimiter=",")
    
        for row in reader:
            for column in reader:
                data_dict=data_dict.append({

                    "Timestamp":column[0],
                    "Gen_Bear_Temp_Avg":column[1],
                    "Gen_Phase1_Temp_Avg":column[2],
                    "Hyd_Oil_Temp_Avg":column[3],
                    "HVTrafo_Phase2_Temp_Avg":column[4],
                    "Gear_Bear_Temp_Avg":column[5],
                    }, ignore_index=True)
        r1,r2,r3=FaultPrediction(data_dict)
    return render(request,'blog/results2.html',{
        'data':r1,
        'data2':r2,
        'data3':r3
        })

# LSTM UPLOAD FORM
def openDataset(request):
    message = ''
    if request.method == "GET":
        return render(request, 'blog/upload_csv_ag.html')

    if request.FILES.get("file2") is not None:
        csv_file = request.FILES['file2']        

        if not csv_file.name.endswith('.csv'):
            message='The uploaded file has to be CSV.  Please try again.'
            messages.add_message(request, messages.INFO, 'Dataset Should be .CSV file!')
        else:
            save_path = 'C:/Users/user/Desktop/Fault Detection App/LSTM_Uploaded_Datasets/'
            file_name = csv_file.name
            fs = FileSystemStorage(location=save_path)
            file = fs.save(file_name, csv_file)
            messages.add_message(request, messages.INFO, 'Dataset Uploaded Successfully!')

    else:
       
        messages.add_message(request, messages.INFO, 'No File is Uploaded!')
        #return render(request, 'blog/upload_csv_ag.html',{'message':message})       
    return render(request,'blog/upload_csv_ag.html',{'message': message})


# Weather UPLOAD FORM
def openDataset2(request):
    message = ''
    if request.method == "GET":
        return render(request, 'blog/upload_csv.html')

    if request.FILES.get("file2") is not None:
        csv_file = request.FILES['file2']        

        if not csv_file.name.endswith('.csv'):
            message='The uploaded file has to be CSV.  Please try again.'
            messages.add_message(request, messages.INFO, 'Dataset Should be .CSV file!')
            # return render(request, 'blog/upload_csv.html',{'message':message})
        else:
            save_path = '/Users/nourelgarhy/Downloads/Fault Detection App-1/blog/Uploaded_Datasets_SCADA'
            file_name = csv_file.name
            fs = FileSystemStorage(location=save_path)
            file = fs.save(file_name, csv_file)
            messages.add_message(request, messages.INFO, 'Dataset Uploaded Successfully!')

    else:
        message='no file is uploaded'
        messages.add_message(request, messages.INFO, 'No File is Uploaded!')
        # return render(request, 'blog/upload_csv.html',{'message':message})       
    return render(request,'blog/upload_csv_ag.html',{'message': message})


# @jit
def read_datasets(request):
    path = r"/Users/nourelgarhy/Downloads/Fault Detection App-1/Uploaded_Datasets/"
    
    path1, dirs, files = next(os.walk(path))
    file_count = len(files)
    print(file_count)

    dataframes_list_html = []
    file_names = []
    index = []

    delete_request = request.POST.get('delete_btn')
    
    for i in range(file_count):
        temp_df2 = pd.read_csv(path+files[i])
        print(files[i])
        dataframes_list_html.append(temp_df2.to_html(index=False))
        index.append(i)
        f =os.path.splitext(files[i])[0]
        file_names.append(f)

    
    # if 'delete_btn' in request.POST:
    #     print("gwa button")
    #     print(len(file_names))
    #     for i  in range(len(file_names)):
    #         print(request.FILES.get('delete_btn'))
    #         if(request.FILES.get('delete_btn') == path+files[i]):
    #             print("found" + file_names[i])
    #             os.remove(path + files[i])
    #             file_names.remove(files[i])
    #             return render(request,'blog/view_datasets2.html',{'files': file_names})
   
        
    # file  = request.POST.get("dataframes_list_html")
    return render(request,'blog/view_datasets.html',{'files': file_names})

def read_datasets2(request):
    path = r"/Users/nourelgarhy/Downloads/Fault Detection App-1/STL_Uploaded_Datasets/"
    
    path1, dirs, files = next(os.walk(path))
    file_count = len(files)
    print(file_count)

    dataframes_list_html = []
    file_names = []
    index = []

    delete_request = request.POST.get('delete_btn')
    
    for i in range(file_count):
        temp_df2 = pd.read_csv(path+files[i])
        print(files[i])
        dataframes_list_html.append(temp_df2.to_html(index=False))
        index.append(i)
        f =os.path.splitext(files[i])[0]
        file_names.append(f)


    # file  = request.POST.get("dataframes_list_html")
    return render(request,'blog/view_datasets2.html',{'files': file_names})

# read_datasets_jit = jit()(read_datasets)
# read_datasets_jit(requests.request)

temp_df=[]
temp_df2=[]

def one_dataset(request):
    path = r"/Users/nourelgarhy/Downloads/Fault Detection App-1/Uploaded_Datasets/"

    path1, dirs, files = next(os.walk(path))

    file_count = len(files)
    global temp_df
    dataframes_list_html = []
    exts = '.csv'

    requested_file = request.POST.get('file')
    neededFile= request.POST.get('file')
    # full_file_name = requested_file + exts
    # print(full_file_name)

    if 'file' in request.POST:    
        for i in range(file_count):
            if(files[i] == requested_file + exts):
                temp_df = pd.read_csv(path+files[i])
                dataframes_list_html.append(temp_df.to_html(index=False))
           
    return render(request, 'blog/single_dataset.html', {'dataframes':dataframes_list_html, 'needed':neededFile,})

#read specific WEATHER dataset
def one_dataset2(request):
    path = r"/Users/nourelgarhy/Downloads/Fault Detection App-1/STL_Uploaded_Datasets/"
    

    path1, dirs, files = next(os.walk(path))

    file_count = len(files)
    global temp_df2
    dataframes_list_html = []
    exts = '.csv'

    requested_file = request.POST.get('file')
    # full_file_name = requested_file + exts

    if 'file' in request.POST:    
        for i in range(file_count):
            if(files[i] == requested_file + exts):
                temp_df2 = pd.read_csv(path+files[i])
                dataframes_list_html.append(temp_df2.to_html(index=False))
           
    return render(request, 'blog/single_dataset2.html', {'dataframes':dataframes_list_html})

def wt_components(request):
    requested_file = request.POST.get('lstmModel')
    print(requested_file)
    print('sjdksdlkdsldksllsdldlskd')
    output =[]
    global temp_df
    a,b,c,d,e,f=LSTMPrediction(requested_file)
    output.append(a)
    output.append(b)
    output.append(c)
    output.append(d)
    output.append(e)
    output.append(f)
    print(output)
    return render(request, 'blog/wt_components_results.html',{'output':output})



#temp_df=[]
#temp_df2=[]

def weather(request):
    output = []
    global temp_df2
    print("------")
    print(temp_df2)
    a,b,c,d,e=WeatherFaultDetection(temp_df2)
    output.append(a)
    output.append(b)
    output.append(c)
    output.append(d)
    output.append(e)

    print(output)
    return render(request, 'blog/weather_results.html',{'output':output})

def test(request):
    return render(request, 'blog/test.html')


def dataGenBeatTime(request):
   
    dicts, list=dictionary()
    #print("00000")
    #print(list)
    #dataframes_list_html2.append(df.to_html(index=False))
    
    print("wnydudidid")

    colorss={}
    path3 = r"/Users/nourelgarhy/Downloads/Fault Detection App-1/blog/datasetss/allllll.csv"
    print(path3)
    temp_df5 = pd.read_csv(path3)

    for i in range(len(temp_df5)):
        colorss[i] = temp_df5.at[i,'color']
       
    posts4=posts()

    print(len(dicts))
    print(len(posts4))
    #print(len(monthly_resampled_data))
    return render(request, 'blog/calendar.html', { 'dayy':dicts, 'colors': posts4,
    'day2':list })

   
def color(request):
    colorss={}
    path3 = r"/Users/nourelgarhy/django1/learn/untitled folder/sub.csv"
    print(path3)
    temp_df5 = pd.read_csv(path3)

    for i in range(len(temp_df5)):
        colorss[i] = temp_df5.at[i,'color']

def all_data(request):
    import calendar
    path = r"/Users/nourelgarhy/django1/learn/untitled folder/sub444.csv"
    monthly_resampled_data = pd.read_csv(path)

    posts3=[]

    for y in range():
        posts4=[]
        for i in range(len(monthly_resampled_data)):
            
            x={
            'color':monthly_resampled_data['color'].values[i],
            'day':monthly_resampled_data['day'].values[i],
            'month': monthly_resampled_data['month'].values[i],
            'dayName': monthly_resampled_data['dayName'].values[i],

            }
            posts4.append(x)
        posts3.append(posts4)

def set_day(request):
    path = r"/Users/nourelgarhy/django1/learn/untitled folder/sub444.csv"
    m = pd.read_csv(path)
    print(m['color'].iloc[30])
    print('aaaa')

    dataframes_list_html = []
    
    print(request.POST.get('file'))
    keyy=int(request.POST.get('file'))
    print('-------')
    print(keyy)
    
    posts4=[]
    
  
    x={
            'color':m['color'].iloc[keyy],
            'day':m['month'].iloc[keyy],
            'month': m['day'].iloc[keyy],
            'dayName': m['color'].iloc[keyy],

            }
    posts4.append(x)

   


    return render(request, 'blog/chosen_day.html', {'datasingle':x})

def mapDeets(request):
    path = r"/Users/nourelgarhy/Downloads/Fault Detection App-1/blog/datasetss/allllll.csv"
    m = pd.read_csv(path)
    
    key=request.POST.get('cal')
    
    key=int(key)
    print("keeey")
    print( m['dayTill'].values[5] )
    d=m['dayTill'].values[key] 
    w=m['color'].values[key] 
    c=m['component'].values[key] 

    turbineInfo = {
      'daysTillFault': d ,
      'warningColor': w ,
      'component': c,
    }
    return render(request, 'blog/map.html', {'contact_form': turbineInfo})

    



