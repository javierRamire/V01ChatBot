from django.urls import path
from . import views
from . import views_adm

urlpatterns = [
    path('',views.chatbot_page, name='home'),
    path('chatbot/', views.chatbot_response_view, name='chatbot_response'),
    path('status/',views.check_status, name='check_status'),
    path('consulta/', views_adm.consulta_preguntas, name='consulta'),
    path('descarga/', views_adm.descarga_preguntas, name='descarga_datos'),
    path('entrenar/', views_adm.train_chatbot, name='entrenar_bot'),
    path('cargar-archivo/', views_adm.cargar_archivo, name='cargar_archivo'),
    path('descargar-archivo/', views_adm.descargar_archivo, name='descargar_archivo'),

]