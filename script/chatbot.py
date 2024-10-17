import random
import json
import pickle
import numpy as np
import os

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '..', 'data', 'intents.json')
model_dir = os.path.join(base_dir, '..', 'model')
model_path = os.path.join(model_dir, 'chatbot_model.h5')
words_path = os.path.join(model_dir, 'words.pkl')
classes_path = os.path.join(model_dir, 'classes.pkl')

# Download necessary NLTK data (optional if already downloaded)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load intents.json
with open(data_path, 'r') as file:
    intents = json.load(file)

# Load words and classes
with open(words_path, 'rb') as f:
    words = pickle.load(f)

with open(classes_path, 'rb') as f:
    classes = pickle.load(f)

# Load the trained model
model = load_model(model_path)

def clean_up_sentence(sentence):
    """
    Tokenizes and lemmatizes the input sentence.
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    """
    Creates a bag-of-words representation of the input sentence.
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f'Found in bag: {w}')
    return np.array(bag)

def predict_class(sentence, model):
    """
    Predicts the class (intent) of the input sentence.
    """
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

def get_response(intents_list, intents_json):
    """
    Retrieves a random response from the list of possible responses for the predicted intent.
    """
    if not intents_list:
        return "I'm sorry, I didn't understand that."

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

    return "I'm sorry, I didn't understand that."

# Example interaction loop (optional)
if __name__ == "__main__":
    print("Start chatting with the bot (type 'quit' to stop)!")
    while True:
        message = input("> ")
        if message.lower() == "quit":
            print("Goodbye!")
            break

        intents_list = predict_class(message, model)
        response = get_response(intents_list, intents)
        print(response)