import os
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import sys

# Parameters
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 256

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    return lines

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return text.strip()

def create_sentence_pairs(lines):
    input_texts = []
    target_texts = []
    # Ensure there is an even number of lines
    if len(lines) % 2 != 0:
        print("Warning: Odd number of lines detected. The last line will be ignored.")
    for i in range(0, len(lines) - 1, 2):
        input_line = clean_text(lines[i])
        target_line = clean_text(lines[i + 1])
        target_line = '<START> ' + target_line + ' <END>'
        input_texts.append(input_line)
        target_texts.append(target_line)
    return input_texts, target_texts

def tokenize_sentences(input_texts, target_texts):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
    tokenizer.fit_on_texts(input_texts + target_texts)
    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)
    word_index = tokenizer.word_index
    return input_sequences, target_sequences, word_index, tokenizer

def pad_sequences_custom(sequences):
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

def save_tokenizer(tokenizer, filename):
    try:
        with open(filename, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error saving tokenizer: {e}")
        sys.exit(1)

def preprocess_data():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the 'data' directory (one level up)
    data_dir = os.path.join(script_dir, '..', 'data')
    data_dir = os.path.normpath(data_dir)

    # Construct the full path to 'conversations.txt'
    file_path = os.path.join(data_dir, 'conversations.txt')

    # Print the file path for debugging
    print(f"Looking for 'conversations.txt' at: {file_path}")

    # Load data using the dynamically constructed path
    lines = load_data(file_path)
    print(f"Number of lines loaded: {len(lines)}")

    # Preprocess data
    input_texts, target_texts = create_sentence_pairs(lines)
    print(f"Number of input_texts: {len(input_texts)}")
    print(f"Number of target_texts: {len(target_texts)}")

    input_sequences, target_sequences, word_index, tokenizer = tokenize_sentences(input_texts, target_texts)
    encoder_input_data = pad_sequences_custom(input_sequences)
    decoder_input_data = pad_sequences_custom(target_sequences)

    # Create decoder target data by shifting decoder input data by one timestep
    decoder_target_data = np.zeros_like(decoder_input_data)
    decoder_target_data[:, :-1] = decoder_input_data[:, 1:]
    decoder_target_data[:, -1] = tokenizer.word_index.get('<end>', 0)  # Use 0 if '<end>' not found

    # Limit the number of words to MAX_NUM_WORDS
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

    # Construct the path to the 'models' directory
    models_dir = os.path.join(script_dir, '..', 'models')
    models_dir = os.path.normpath(models_dir)
    os.makedirs(models_dir, exist_ok=True)

    # Save tokenizer and data
    save_tokenizer(tokenizer, os.path.join(models_dir, 'tokenizer.pkl'))
    np.save(os.path.join(models_dir, 'encoder_input_data.npy'), encoder_input_data)
    np.save(os.path.join(models_dir, 'decoder_input_data.npy'), decoder_input_data)
    np.save(os.path.join(models_dir, 'decoder_target_data.npy'), decoder_target_data)
    np.save(os.path.join(models_dir, 'num_words.npy'), np.array([num_words]))

    print('Preprocessing completed.')

if __name__ == "__main__":
    # Download NLTK data if not already present
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    preprocess_data()