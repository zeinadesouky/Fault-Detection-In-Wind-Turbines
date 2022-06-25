from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='blog-home'),
    path('about/', views.about, name='blog-about'),
    path('contact/', views.contact, name='blog-contact'),
    path('help/', views.help, name='blog-help'),
    
    path('upload_dataset/', views.openDataset, name = 'upload_csv'),
    path('view_dataset/', views.read_datasets, name = 'view_datasets'),
    path('view_one_dataset/', views.one_dataset, name='single-dataset'),
    
    path('test/', views.test, name='blog-test'),

    path('wt_components_results/',views.wt_components , name='wt_components_results'),
    
    path('upload_dataset2/', views.openDataset2, name = 'upload_csv2'),
   
    path('view_datasets2/', views.read_datasets2, name = 'view_datasets2'),
   
    path('view_one_dataset2/', views.one_dataset2, name='single-dataset2'),
    path('weather_results/', views.weather, name='weather_results'),
    
    

    path('setday/', views.mapDeets, name='set-day-view'),
    path('calendar/', views.dataGenBeatTime, name='blog-calendar'),

    



]