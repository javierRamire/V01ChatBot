from django.shortcuts import render
from django.http import JsonResponse
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import random

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
    print (ints)
    res = getResponse(ints, intents)
    return res

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