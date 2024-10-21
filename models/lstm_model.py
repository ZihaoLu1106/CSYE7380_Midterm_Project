import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input

def build_lstm_model(vocab_size, embedding_dim, units):
    image_input = Input(shape=(2048,))
    caption_input = Input(shape=(None,))

    # Image feature embedding
    image_features = Dense(embedding_dim)(image_input)

    # Caption embedding
    caption_embedding = Embedding(vocab_size, embedding_dim)(caption_input)
    caption_lstm = LSTM(units)(caption_embedding)

    # Combine image and caption features
    combined = tf.keras.layers.Add()([image_features, caption_lstm])
    output = Dense(vocab_size, activation='softmax')(combined)

    model = Model(inputs=[image_input, caption_input], outputs=output)
    return model
