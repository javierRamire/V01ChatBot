from django.urls import path
from . import views

urlpatterns = [
    
    path('chatbot/', views.chatbot_response_view, name='chatbot_response'),
    path('',views.chatbot_page, name='home'),
    path('status/',views.check_status, name='check_status'),
    path('consulta/', views.consulta_preguntas, name='consulta'),
    path('descarga/', views.descarga_preguntas, name='descarga_datos'),
    path('entrenar/', views.train_chatbot, name='entrenar_bot')

]