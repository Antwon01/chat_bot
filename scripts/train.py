import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention
from tensorflow.keras.optimizers import Adam
import pickle

# Parameters
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 256

def load_preprocessed_data():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the 'models' directory (one level up)
    models_dir = os.path.join(script_dir, '..', 'models')
    models_dir = os.path.normpath(models_dir)

    # Load preprocessed data
    encoder_input_data = np.load(os.path.join(models_dir, 'encoder_input_data.npy'))
    decoder_input_data = np.load(os.path.join(models_dir, 'decoder_input_data.npy'))
    decoder_target_data = np.load(os.path.join(models_dir, 'decoder_target_data.npy'))
    num_words = int(np.load(os.path.join(models_dir, 'num_words.npy'))[0])

    # Load tokenizer
    with open(os.path.join(models_dir, 'tokenizer.pkl'), 'rb') as handle:
        tokenizer = pickle.load(handle)

    return encoder_input_data, decoder_input_data, decoder_target_data, num_words, tokenizer

def build_model(num_words):
    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    enc_emb = Embedding(num_words, LATENT_DIM, mask_zero=True, name='encoder_embedding')(encoder_inputs)
    encoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    dec_emb_layer = Embedding(num_words, LATENT_DIM, mask_zero=True, name='decoder_embedding')
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # Attention Layer
    attn_layer = Attention(name='attention_layer')
    attn_out = attn_layer([decoder_outputs, encoder_outputs])

    # Concatenate attention output and decoder LSTM output
    decoder_concat_input = Dense(LATENT_DIM, activation='tanh', name='decoder_dense_tanh')(attn_out)

    # Output Layer
    decoder_dense = Dense(num_words, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def train_model():
    encoder_input_data, decoder_input_data, decoder_target_data, num_words, tokenizer = load_preprocessed_data()
    model = build_model(num_words)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.2)
    model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'chatbot_model.h5'))
    print('Model training completed and saved.')

if __name__ == "__main__":
    train_model()
