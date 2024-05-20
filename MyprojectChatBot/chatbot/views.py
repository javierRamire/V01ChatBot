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
import firebase_admin
from firebase_admin import credentials,firestore
import os
print(os.getcwd())
cred = credentials.Certificate(f"{os.getcwd()}/MyprojectChatBot/chatbot/chatbotproyectv1-firebase-adminsdk-rjy8r-b02d097d0b.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def chatbot_page(request):
    return render(request, f'{os.getcwd()}/MyprojectChatBot/templates/chatbot/chatbot.html')

# Cargar el modelo y otros datos necesarios
lemmatizer = WordNetLemmatizer()
max_length = 361 
model = load_model('MyprojectChatBot/chatbot/chatbot_model.h5')
intents = json.loads(open('MyprojectChatBot/chatbot//intents.json', encoding='utf-8').read())
words = pickle.load(open('MyprojectChatBot/chatbot//words.pkl', 'rb'))
classes = pickle.load(open('MyprojectChatBot/chatbot//classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Encontrado en la bolsa: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    p = pad_sequences([p], maxlen=max_length, padding='post')[0]
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents):
    tag = ints[0]['intent']
    list_of_intents = intents['intents']
    result = "No response found"  
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
        response_text = chatbot_response(message)  
        response_text_utf8 = response_text.encode('utf-8').decode('utf-8')
        response = {'response': response_text_utf8 }
        return JsonResponse(response, json_dumps_params={'ensure_ascii': False})
    
def check_status(request):
    return JsonResponse({'status': 'ok'})
