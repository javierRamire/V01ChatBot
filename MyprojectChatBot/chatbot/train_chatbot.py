import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# Lista de palabras, clases, y documentos
words = []
classes = []
documents = []
ignore_words = ['?', '!'] 

# Inicialización del lematizador de WordNet
lemmatizer = WordNetLemmatizer()

# Cargar el archivo JSON que contiene los patrones de las intenciones
data_file = open('chatbot/intents.json', 'r', encoding='utf-8').read()
intents = json.loads(data_file)

# Preprocesamiento de los datos
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizar cada palabra
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Añadir documentos al corpus
        documents.append((w, intent['tag']))
        # Añadir a la lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematizar, convertir a minúsculas cada palabra y eliminar duplicados
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Ordenar las clases
classes = sorted(list(set(classes)))
print(len(documents), "documentos")
print(len(classes), "clases", classes)

print(len(words), "palabras lematizadas únicas", words)

# Guardar las palabras y clases en archivos pkl
pickle.dump(words,open('chatbot/words.pkl','wb'))
pickle.dump(classes,open('chatbot/classes.pkl','wb'))

# Crear datos de entrenamiento
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
# Encontrar la longitud máxima de los bags of words
max_length = max(len(doc[0]) for doc in training)

# Rellenar los bags of words para asegurar que todos tengan la misma longitud
padded_training = []
for bag, output_row in training:
    padded_bag = pad_sequences([bag], maxlen=max_length, padding='post')[0]
    padded_training.append([padded_bag, output_row])

# Convertir a un array numpy
padded_training = np.array(padded_training, dtype=object)

# Mezclar el array para evitar cualquier sesgo debido al orden de los datos
np.random.shuffle(padded_training)

# Separar características y etiquetas
train_x = np.array([x for x, _ in padded_training])
train_y = np.array([y for _, y in padded_training])

# Definir el modelo
model = Sequential()
model.add(Dense(128, input_shape=(max_length,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar el modelo
# Definir el optimizador
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Guardar el modelo
model.save('chatbot/chatbot_model.h5', hist)
print("Modelo creado")
model.summary()
