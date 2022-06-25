import email
from django.db import models

class Contact_Response(models.Model):
    name = models.CharField(max_length=30)
    email = models.CharField(max_length=30)
    description = models.CharField(max_length=500)

class Anomalies(models.Model):
    key= models.IntegerField()
    Timestamp = models.CharField(('Timestamp'),max_length=100)
    Gen_Bear_Temp_Avg = models.FloatField((0.0))	
    Gen_Phase1_Temp_Avg = models.FloatField((0.0))	
    Hyd_Oil_Temp_Avg = models.FloatField((0.0))	
    HVTrafo_Phase2_Temp_Avg = models.FloatField((0.0))	
    Gear_Bear_Temp_Avg = models.FloatField((0.0))	
    Gear_Oil_Temp_Avg = models.FloatField((0.0))	
    loss = models.FloatField((0.0))	
    threshold = models.FloatField((0.0))	
    anomaly = models.BooleanField(default=False)
    color = models.FloatField((0.0))	
    dayTill =models.IntegerField()
    component = models.CharField(max_length=100)
    day =models.IntegerField()
    month =models.IntegerField()
    year =models.IntegerField()
    dayofyear =models.IntegerField()
    dayName  = models.CharField(max_length=100)
    monthName  = models.CharField(max_length=100)