from django.contrib import admin
from django.urls import path
from .views import responder_peticion
# Register your models here.
urlpatterns = [
    path('chatbot/', responder_peticion, name='chatbot'),
]