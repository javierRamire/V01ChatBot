from django.urls import path
from . import views

urlpatterns = [
    
    path('chatbot/', views.chatbot_response, name='chatbot_response'),
    path('',views.chatbot_page, name='home'),
]