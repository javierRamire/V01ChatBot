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


words = []
classes = []
documents = []
ignore_words = ['?', '!'] 

lemmatizer = WordNetLemmatizer()

data_file = open('intents.json', 'r', encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents to the corpus
        documents.append((w, intent['tag']))
        # Add to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lowercase each word, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sort classes
classes = sorted(list(set(classes)))
print(len(documents), "documents")
print(len(classes), "classes", classes)

print(len(words), "unique lemmatized words", words)
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# Create training data
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

# Find the maximum length of bags of words
max_length = max(len(doc[0]) for doc in training)

# Pad the bags of words to ensure they all have the same length
padded_training = []
for bag, output_row in training:
    padded_bag = pad_sequences([bag], maxlen=max_length, padding='post')[0]
    padded_training.append([padded_bag, output_row])

# Convert to numpy array
padded_training = np.array(padded_training, dtype=object)

# Shuffle the array to avoid any biases due to the order of data
np.random.shuffle(padded_training)

# Separate features and labels
train_x = np.array([x for x, _ in padded_training])
train_y = np.array([y for _, y in padded_training])


# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(max_length,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
# Define the optimizer
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5', hist)
print("Model created")
model.summary()