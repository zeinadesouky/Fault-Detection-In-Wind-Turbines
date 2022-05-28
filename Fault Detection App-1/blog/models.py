import email
from django.db import models

class Contact_Response(models.Model):
    name = models.CharField(max_length=30)
    email = models.CharField(max_length=30)
    description = models.CharField(max_length=500)