import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input,Add
from tensorflow.keras import layers, models

def build_lstm_model(vocab_size, embedding_dim, units):#embedding_dim 256 units 512
    image_input = Input(shape=(2048,))
    x1 = layers.Dense(embedding_dim, activation='relu')(image_input)
    x1 = layers.RepeatVector(1)(x1)
    # Image feature embedding
    caption_input = layers.Input(shape=(None,))
    x2 = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(caption_input)
    x2 = layers.LSTM(units, return_sequences=True)(x2)

    decoder = layers.concatenate([x1, x2], axis=1)
    decoder = layers.LSTM(units, return_sequences=True)(decoder)

    # Output layer
    outputs = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(decoder)

    # Create the model
    model = models.Model(inputs=[image_input, caption_input], outputs=outputs)

    return model
