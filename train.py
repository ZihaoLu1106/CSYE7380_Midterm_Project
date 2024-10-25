from utils.data_loader import load_image_paths_and_captions, split_data, build_vocab
from utils.feature_extraction import initialize_densenet,get_preprocess, extract_features
from models.lstm_model import build_lstm_model
import numpy as np
import kagglehub
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
images = kagglehub.dataset_download("adityajn105/flickr30k")
captions_file = './data/captions.txt'
image_paths, captions = load_image_paths_and_captions(images, captions_file)

# Preprocess and split data
train_img_paths, test_img_paths, train_captions, test_captions = split_data(image_paths, captions)

# Build vocabulary
vocab = build_vocab(train_captions)
vocab_size = len(vocab)

#add: Define max_length for captions
max_length = 20
print("start Initialize DenseNet")
# Initialize DenseNet
densenet_model = initialize_densenet().to(device)
preprocess = get_preprocess()

# Extract features for training images
train_image_features = np.array([extract_features(img, densenet_model,preprocess,device) for img in train_img_paths])
np.save('train_image_features.npy', train_image_features)
print("Image features saved to train_image_features.npy")
# Prepare target sequences for LSTM
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_captions)
train_sequences = tokenizer.texts_to_sequences(train_captions)
y_train = pad_sequences(train_sequences, maxlen=max_length)
print("start building")
# Build LSTM model
embedding_dim = 256
units = 512
lstm_model = build_lstm_model(vocab_size, embedding_dim, units)

# Now proceed with training...
print("start compile")
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set up callbacks
checkpoint = ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model
print("start fit")
lstm_model.fit(train_image_features, y_train, 
               epochs=10, 
               batch_size=64, 
               validation_split=0.2, 
               verbose=1,  # Set verbose to 1 to see the progress
               callbacks=[checkpoint, early_stopping])

lstm_model.save('lstm_model.h5')