import email
from unicodedata import name
from attr import field, fields
from django import forms
from .models import Contact_Response

class ContactForm(forms.ModelForm):
    name = forms.TextInput
    email = forms.TextInput
    description = forms.TextInput

    class Meta:
        model = Contact_Response
        fields = ['name', 'email', 'description']