from django.shortcuts import render
from django.http import JsonResponse
import random
import spacy
import nltk
import numpy as np
import  string
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# Create your views here.

nlp = spacy.load('es_core_news_md')
def chatbot_page(request):
    return render(request, 'chatbot/chatbot.html')

respuestas = {
    "Hola": "¡Hola! ¿En qué puedo ayudarte?",
    "Cómo estás": "Estoy bien, gracias por preguntar.",
    "Adiós": "¡Hasta luego! Que tengas un buen día.",
    # Agrega más patrones y respuestas según tus necesidades.
}    

respuestasTop = {
    "UsuarioSiaf":"En que puedo ayudarte con los usuarios ?", 
    "Dudas":"Puedes escribir tu pregunta por favor.",
    "Formatos":" Los formatos los puedes descargar desde ",
}

respuestasOpciones = {
  "Alta":"Es alguno de estos?", 
  "Baja":"Es Alguno de estos?",
  "Modificacion":"Es Alguno de estos?",
  "Otra duda": "Que duda tienes, la puedes escribrir",
}

opciones={
    "UsuarioSiaf":["Alta", "Baja", "Modificacion","Otra duda"],
    "Alta":["Formatos","paso a seguir","dudas"],
    "Baja":["Formatos","paso a seguir","dudas"],
    "Modificacion":["Formatos","paso a seguir","dudas"],
}

referencias={
    "Formatos":"https://egresos.finanzas-puebla.mx/siaf.html",
}

def preproceso_texto(text):
    tokens = word_tokenize(text, language='spanish')
    tokens = [word.lower() for word in tokens]
    
    table= str.maketrans('','',string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words]

    tokens = [word for word in tokens if len(word) > 1]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return  lemmatized_tokens
 
def divide_secuencias(tokens, logitud):
    secuencia = []
    for i in range(0, len(tokens)-logitud+1, logitud):
        secuencias = tokens[i:i + logitud]
        secuencia.append(secuencias)       
    return secuencia

def codificacion_one_hot(secuencias_tokens, vocabulario_size):
    secuencias_codificadas = []
    for secuencia in secuencias_tokens:
        secuencia_codificada = []
        for palabra in secuencia:
            vector = np.zeros(vocabulario_size)
            vector[palabra] = 1
            secuencia_codificada.append(vector)
        secuencias_codificadas.append(secuencia_codificada)
    return secuencias_codificadas


def chatbot_response(request):
    user_input = request.GET.get('message')  # Obtén el mensaje del usuario desde la URL

    salida_preproceso = preproceso_texto(user_input)
    secuencia = divide_secuencias(salida_preproceso,5)
    
    # X_train, X_test = train_test_split(secuencia, test_size=0.2, random_state=42)
    # X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)

    # print("Tamaño del conjunto de entrenamiento:", len(X_train))
    # print("Tamaño del conjunto de validación:", len(X_val))
    # print("Tamaño del conjunto de prueba:", len(X_test))
    
     # Análisis de partes del discurso con spaCy
    doc = nlp(user_input)
    pos_tags = [(token.text, token.pos_) for token in doc]
    lemas_spacy = [token.lemma_ for token in doc]
    # Extracción de entidades con spaCy
    entities = [(entity.text, entity.label_) for entity in doc.ents]
   
    if user_input in respuestas:
        response = obtener_respuesta(user_input)
        return JsonResponse({'response': response})
   
    elif user_input in respuestasTop:
        response = obtener_Top(user_input)
        opciones = obtener_opciones(user_input)
        return JsonResponse({'response': response, 'opciones': opciones})
     
    elif user_input in respuestasOpciones :
        response = obtener_respuestaOciones(user_input)
        opciones = obtener_opciones(user_input)
        return JsonResponse({'response': response, 'opciones': opciones})
    
    elif user_input in referencias:
        response = obtener_respuesta(user_input)
        referencia = obtener_referencias(user_input)
        return JsonResponse({'response': response,'referencias':referencia})
    
    else: response= buscar_respuesta(user_input) 
    return JsonResponse({'response': response})

def obtener_respuesta(mensaje):
    # Aquí implementaremos la lógica para obtener la respuesta del chatbot.
    # Usaremos el diccionario de respuestas definido anteriormente.
    if mensaje in respuestas:
        return respuestas[mensaje]
    else:
        return "Lo siento, no entiendo tu pregunta."
    
def obtener_Top(mensaje):
    if mensaje in respuestasTop:
        return respuestasTop[mensaje]  
    else:
        return "Lo siento, no entiendo tu pregunta."       
    
def obtener_respuestaOciones(mensaje):
    if mensaje in respuestasOpciones:
        return respuestasOpciones[mensaje]
    else:
        return "Lo siento, no entiendo tu pregunta."    

def obtener_opciones(mensaje):
    if mensaje in opciones:
        return opciones[mensaje]
    else: "null"
    
def obtener_referencias(mensaje):
    if mensaje in referencias:
        return referencias[mensaje]
    else: "null"       

def buscar_respuesta(mensaje):
    respuestas = ["Lo siento, no entendí tu pregunta.",
                  "No tengo información sobre eso en este momento.",
                  "¡Esa es una excelente pregunta! Permíteme investigar y volveré contigo pronto.",
                  "No entiendo tu pregunta. ¿Puedes reformularla?",
                  "Lo siento, no entiendo tu pregunta."]
    if not mensaje:return "no se ha escrito nada"
    else: return random.choice(respuestas)