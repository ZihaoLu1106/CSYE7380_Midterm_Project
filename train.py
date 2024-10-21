from utils.data_loader import load_image_paths_and_captions, split_data, build_vocab
from utils.feature_extraction import initialize_densenet, extract_features
from models.lstm_model import build_lstm_model
import numpy as np
import kagglehub


# Load dataset
images = kagglehub.dataset_download("adityajn105/flickr30k")
captions_file = './data/captions.txt'
image_paths, captions = load_image_paths_and_captions(images, captions_file)

# Preprocess and split data
train_img_paths, test_img_paths, train_captions, test_captions = split_data(image_paths, captions)

# Build vocabulary
vocab = build_vocab(train_captions)
vocab_size = len(vocab)

# Initialize DenseNet
densenet_model = initialize_densenet()

# Extract features for training images
train_image_features = np.array([extract_features(img, densenet_model) for img in train_img_paths])

# Build LSTM model
embedding_dim = 256
units = 512
lstm_model = build_lstm_model(vocab_size, embedding_dim, units)

# Now proceed with training...
