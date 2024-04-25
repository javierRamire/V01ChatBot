import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
import json

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
sequences = []
for tokens in preprocessed_texts:
    for i in range(len(tokens) - sequence_length + 1):
        sequences.append(tokens[i:i+sequence_length])

# Obtención de etiquetas correspondientes a cada secuencia
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

# Definición del modelo
embedding_dim = 128
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=sequence_length),
    LSTM(256, return_sequences=True),
    Dropout(0.5),
    LSTM(256),
    Dropout(0.5),
    Dense(len(label_dict), activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Resumen del modelo

# Entrenamiento del modelo con conjunto de validación
hist = model.fit(X_train, y_train, epochs=10, batch_size=36, validation_data=(X_val, y_val))

model.save('chatbot_model_V2.h5', hist)

model.summary()