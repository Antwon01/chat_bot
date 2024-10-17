import random
import json
import pickle
import numpy as np
import os

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '..', 'data', 'intents.json')
model_dir = os.path.join(base_dir, '..', 'model')
model_path = os.path.join(model_dir, 'chatbot_model.h5')
words_path = os.path.join(model_dir, 'words.pkl')
classes_path = os.path.join(model_dir, 'classes.pkl')

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load intents.json
with open(data_path, 'r') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Tokenize patterns and build words and classes lists
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes using pickle
with open(words_path, 'wb') as f:
    pickle.dump(words, f)

with open(classes_path, 'wb') as f:
    pickle.dump(classes, f)

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Separate features and labels
train_x = [item[0] for item in training]
train_y = [item[1] for item in training]

# Convert to NumPy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

# Verify shapes
print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model in H5 format (correct usage)
model.save(model_path)
print('Bot is running!')