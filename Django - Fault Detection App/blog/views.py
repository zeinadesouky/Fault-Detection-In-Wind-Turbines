from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.http import JsonResponse
import pandas as pd
import io
import csv
import requests
from django.views.generic import TemplateView
from .forms import ContactForm
from .trial import lstm

def home(request):
    return render(request, 'blog/home.html')


def about(request):
    return render(request, 'blog/about.html', {'title': 'About'})


#@login_required
def contact(request):
    contact_form = ContactForm()
    return render(request, 'blog/contact.html', {'contact_form': contact_form})


def help(request):
    return render(request, 'blog/help.html')


def upload_file(request):

    if request.method == "GET":
        return render(request, 'blog/upload_csv.html')
    
    csv_file = request.FILES['file']

    if not csv_file.name.endswith('.csv'):
        #messages.error(request, 'THIS IS NOT A CSV FILE')
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
        results=lstm(data_dict)
    return render(request,'blog/results.html',{'data':results})

def test(request):
    return render(request, 'blog/test.html')
