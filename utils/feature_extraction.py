import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize DenseNet model (pre-trained on ImageNet)
def initialize_densenet():
    densenet = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
    return densenet

# Extract DenseNet features
def extract_features(img_path, densenet_model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = densenet_model.predict(img_array)
    return features
