import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import json
import random


# Ejemplo de preguntas al chatbot
questions = [
    "hola",
    "Adiós",
    "¿Dónde puedo obtener los formatos de alta/modificación/inactivación?",
    "¿Qué tipos de usuario existen en el sistema?",
    "¿Cómo puedo resolver un problema de usuarios duplicados?"
]
# Cargar el modelo entrenado
model = load_model('chatbot_model_V2.h5')

# Cargar el archivo JSON que contiene los datos de entrenamiento del chatbot
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

intents = data['intents']

# Preprocesamiento de texto
def preprocess_text(text_list):
   # Concatenar todos los patrones en una sola cadena
    text = ' '.join(text_list)
    tokens = word_tokenize(text, language='spanish')
    tokens = [word.lower() for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = SnowballStemmer('spanish')
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

# Obtención de textos preprocesados
preprocessed_texts = [preprocess_text(intent['patterns']) for intent in intents]

sequence_length = 5

# Tokenización y codificación de secuencias
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_texts)
vocab_size = len(tokenizer.word_index) + 1

# Codificación de secuencias
encoded_sequences = tokenizer.texts_to_sequences(preprocessed_texts)
padded_sequences = pad_sequences(encoded_sequences, maxlen=sequence_length, padding='pre')

# Obtención de etiquetas correspondientes a cada secuencia
labels = [intent['tag'] for intent in intents]
label_dict = {label: i for i, label in enumerate(set(labels))}
y = np.array([label_dict[intent['tag']] for intent in intents])

# División en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Proceso de hacer preguntas al chatbot
def preprocess_question(question):
    tokens = word_tokenize(question, language='spanish')
    tokens = [word.lower() for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = SnowballStemmer('spanish')
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

def get_intent(question):
    preprocessed_question = preprocess_question(question)
    encoded_question = tokenizer.texts_to_sequences([preprocessed_question])
    padded_question = pad_sequences(encoded_question, maxlen=sequence_length, padding='pre')
    predicted_probabilities = model.predict(padded_question)
    predicted_intent_index = np.argmax(predicted_probabilities)
    predicted_intent = list(label_dict.keys())[predicted_intent_index]
    return predicted_intent

# Ejemplo de preguntas al chatbot
questions = [
    "¿Cómo puedo abrir una cuenta?"
]

for question in questions:
    intent = get_intent(question)
    print("Pregunta:", question)
    print("Intención predicha:", intent)
    print()
