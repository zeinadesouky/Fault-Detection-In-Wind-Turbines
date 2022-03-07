from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='blog-home'),
    path('about/', views.about, name='blog-about'),
    path('contact/', views.contact, name='blog-contact'),
    path('help/', views.help, name='blog-help'),
    path('detect_faults/', views.upload_file, name = 'detect'),
    path('upload_dataset/', views.upload_file_ag, name = 'upload_csv'),
    path('view_dataset/', views.read_datasets, name = 'view_dataset'),
    path('test/', views.test, name='blog-test'),
]