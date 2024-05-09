from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse, HttpResponseRedirect
import subprocess
import json
import firebase_admin
from firebase_admin import firestore
import os

db = firestore.client()

def consulta_preguntas(request):
    datos = extraer_datos()
    return JsonResponse({"datos":datos})

def extraer_datos():
    consulta = db.collection("Preguntas").where("response", "==", "No").stream()
    datos = []
    documentos_id = []
    for documento in consulta:
        datos.append(documento.to_dict())
        documentos_id.append(documento.id)
    return datos, documentos_id

def descarga_preguntas(request):
    datos, documentos_id= extraer_datos()
    datos_json = json.dumps(datos, indent=4)
    response = HttpResponse(datos_json, content_type='application/json')
    response['Content-Disposition'] = 'attachment; filename="datos_firestore.json"'
    actualizar_respuesta(documentos_id)
    return response

def actualizar_respuesta(documentos_id):
    batch = db.batch()
    for doc_id in documentos_id:
        doc_ref = db.collection("Preguntas").document(doc_id)
        batch.update(doc_ref, {"response": "Si"})
    batch.commit()
    
def train_chatbot(request):
    try:
        subprocess.run(['python', 'chatbot/train_chatbot.py'])
        return JsonResponse({'status': 'success', 'message': 'Training completed successfully'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})    
            
def cargar_archivo(request):
    if request.method == 'POST' and request.FILES.get('archivo'):
        archivo_nuevo = request.FILES['archivo']
        if archivo_nuevo.name.endswith('.json'):
            ruta_archivo = os.path.join('chatbot/', archivo_nuevo.name)
            with open(ruta_archivo, 'wb+') as destino:
                for chunk in archivo_nuevo.chunks():
                    destino.write(chunk)
            return JsonResponse({'mensaje': 'Archivo cargado exitosamente'})
        else:
            return JsonResponse({'error': 'El archivo debe tener extensión .json'}, status=400)
    return render(request, 'chatbot/AdminViews.html')

def descargar_archivo(request):
    with open('chatbot/intents.json', 'rb') as archivo:
        response = HttpResponse(archivo.read(), content_type='application/json')
        response['Content-Disposition'] = 'attachment; filename="intents.json"'
        return response