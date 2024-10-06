import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Layer, Dot, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Custom Layer for tf.square without mask handling
class TFSquareLayer(Layer):
    def __init__(self, **kwargs):
        super(TFSquareLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.square(inputs)

# Custom Layer Normalization Layer without mask handling
class LayerNormalizationLayer(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super(LayerNormalizationLayer, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True,
            name='gamma'
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='beta'
        )
        super(LayerNormalizationLayer, self).build(input_shape)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

    def get_config(self):
        config = super(LayerNormalizationLayer, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

# Parameters
LATENT_DIM = 256
BATCH_SIZE = 64
EPOCHS = 100

def load_preprocessed_data():
    # Construct the path to the 'models' directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, '..', 'models')
    models_dir = os.path.normpath(models_dir)
    
    # Paths to the files
    tokenizer_path = os.path.join(models_dir, 'tokenizer.pkl')
    encoder_input_path = os.path.join(models_dir, 'encoder_input_data.npy')
    decoder_input_path = os.path.join(models_dir, 'decoder_input_data.npy')
    decoder_target_path = os.path.join(models_dir, 'decoder_target_data.npy')
    num_words_path = os.path.join(models_dir, 'num_words.npy')
    
    # Check if all files exist
    for path in [tokenizer_path, encoder_input_path, decoder_input_path, decoder_target_path, num_words_path]:
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            sys.exit(1)
    
    # Load tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Load input and target data
    encoder_input_data = np.load(encoder_input_path)
    decoder_input_data = np.load(decoder_input_path)
    decoder_target_data = np.load(decoder_target_path)
    num_words = int(np.load(num_words_path)[0])
    
    return encoder_input_data, decoder_input_data, decoder_target_data, num_words, tokenizer

def build_model(num_words):
    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs', dtype='int32')
    encoder_embedding = Embedding(input_dim=num_words, output_dim=LATENT_DIM, name='encoder_embedding')(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='encoder_lstm')(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs', dtype='int32')
    decoder_embedding = Embedding(input_dim=num_words, output_dim=LATENT_DIM, name='decoder_embedding')(decoder_inputs)
    decoder_outputs, _, _ = LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='decoder_lstm')(decoder_embedding, initial_state=encoder_states)

    # Attention mechanism
    attention_scores = Dot(axes=[2, 2], name='attention_scores')([decoder_outputs, encoder_outputs])
    attention_weights = Activation('softmax', name='attention_weights')(attention_scores)
    context_vector = Dot(axes=[2, 1], name='context_vector')([attention_weights, encoder_outputs])

    # Concatenate context vector and decoder outputs
    decoder_combined_context = Concatenate(axis=-1, name='decoder_combined_context')([context_vector, decoder_outputs])

    # Apply TensorFlow function using Custom Layer
    decoder_combined_context_tf = TFSquareLayer(name='tf_square')(decoder_combined_context)

    # Apply Layer Normalization using custom layer
    decoder_combined_context_ln = LayerNormalizationLayer(name='layer_normalization')(decoder_combined_context_tf)

    # Final Dense layer
    decoder_dense = Dense(num_words, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_combined_context_ln)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model

def train_model():
    encoder_input_data, decoder_input_data, decoder_target_data, num_words, tokenizer = load_preprocessed_data()

    # Ensure data types are correct
    encoder_input_data = encoder_input_data.astype('int32')
    decoder_input_data = decoder_input_data.astype('int32')
    decoder_target_data = decoder_target_data.astype('int32')

    # Print shapes and data types
    print("encoder_input_data shape:", encoder_input_data.shape)
    print("encoder_input_data dtype:", encoder_input_data.dtype)
    print("decoder_input_data shape:", decoder_input_data.shape)
    print("decoder_input_data dtype:", decoder_input_data.dtype)
    print("decoder_target_data shape:", decoder_target_data.shape)
    print("decoder_target_data dtype:", decoder_target_data.dtype)

    model = build_model(num_words)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')

    # Callbacks
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, '..', 'models')
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(models_dir, 'chatbot_model_best.keras'),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )

    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[checkpoint, early_stop]
    )

    # Save the trained model
    model_save_path = os.path.join(models_dir, 'chatbot_model.keras')
    model.save(model_save_path)
    print("Training completed and model saved.")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        sys.exit(1)