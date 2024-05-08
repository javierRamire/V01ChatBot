from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import subprocess
import json
import random
import firebase_admin
from firebase_admin import credentials,firestore

cred = credentials.Certificate("C:/Users/Francisco Amador/Downloads/chatbotproyectv1-firebase-adminsdk-rjy8r-b02d097d0b.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def chatbot_page(request):
    return render(request, 'chatbot/chatbot.html')

# Cargar el modelo y otros datos necesarios
lemmatizer = WordNetLemmatizer()
max_length = 361 
model = load_model('C:/Users/Francisco Amador/mi_proyecto_djangoV2/MyprojectChatBot/chatbot/chatbot_model.h5')
intents = json.loads(open('C:/Users/Francisco Amador/mi_proyecto_djangoV2/MyprojectChatBot/chatbot/intents.json', encoding='utf-8').read())
words = pickle.load(open('C:/Users/Francisco Amador/mi_proyecto_djangoV2/MyprojectChatBot/chatbot/words.pkl', 'rb'))
classes = pickle.load(open('C:/Users/Francisco Amador/mi_proyecto_djangoV2/MyprojectChatBot/chatbot/classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    # Tokenizar el patrón: dividir las palabras en un array
    sentence_words = nltk.word_tokenize(sentence)
    # Lematizar cada palabra: crear una forma corta para la palabra
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenizar el patrón
    sentence_words = clean_up_sentence(sentence)
    # Bolsa de palabras: matriz de N palabras, matriz de vocabulario
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # Asignar 1 si la palabra actual está en la posición del vocabulario
                bag[i] = 1
                if show_details:
                    print("Encontrado en la bolsa: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    # Filtrar las predicciones por debajo de un umbral
    p = bow(sentence, words, show_details=False)
    # Ajustar la longitud del patrón de entrada si es necesario
    p = pad_sequences([p], maxlen=max_length, padding='post')[0]
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Ordenar por la fuerza de la probabilidad
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents):
    tag = ints[0]['intent']
    list_of_intents = intents['intents']
    result = "No response found"  # Asignar un valor predeterminado
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    intencion = ints[0]["intent"]
    if not ints:
        datos = {"tag": "","patterns": text,"response": ""}
        db.collection("Preguntas").add(datos)
        return "No entendí la pregunta. Por favor, reformúlala."
    if "NoEntendi" in intencion:
        datos = {"tag": intencion,"patterns": text,"response": "No"}
        db.collection("Preguntas").add(datos)
    print (ints)
    return getResponse(ints, intents)

def chatbot_response_view(request):
    if request.method == 'GET':
        message = request.GET.get('message', '')
        response_text = chatbot_response(message)  # Llama a la función chatbot_response
        # Codificar la respuesta en UTF-8
        response_text_utf8 = response_text.encode('utf-8').decode('utf-8')
        # Crear un diccionario con la respuesta
        response = {'response': response_text_utf8}
        # Devolver la respuesta como JSON
        return JsonResponse(response, json_dumps_params={'ensure_ascii': False})
    
def check_status(request):
    return JsonResponse({'status': 'ok'})

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
    # Aquí puedes agregar cualquier lógica adicional necesaria para el entrenamiento
    # Luego, ejecuta tu script de entrenamiento
    try:
        subprocess.run(['python', 'chatbot/train_chatbot.py'])
        return JsonResponse({'status': 'success', 'message': 'Training completed successfully'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})    