import os
from sklearn.model_selection import train_test_split
import re
import string

def load_captions(filename):
    with open(filename, 'r') as file:
        captions = file.readlines()
    return captions

def preprocess_caption(caption):
    caption = caption.lower()
    caption = re.sub(f"[{string.punctuation}]", "", caption)
    caption = caption.split()
    return caption

def build_vocab(captions):
    vocab = set()
    for caption in captions:
        vocab.update(preprocess_caption(caption))
    return {word: idx for idx, word in enumerate(vocab)}

def load_image_paths_and_captions(images_folder, captions_file):
    captions_dict = {}
    with open(captions_file, 'r') as file:
        next(file)  # Skip header
        for line in file:
            img_file, caption = line.split(',', 1)
            img_file = img_file.strip()
            caption = caption.strip()
            # Add the 'Images/' subdirectory when joining the paths
            full_image_path = os.path.join(images_folder, 'Images', img_file)
            captions_dict[full_image_path] = caption
    return list(captions_dict.keys()), list(captions_dict.values())

def split_data(image_paths, captions, test_size=0.2):
    return train_test_split(image_paths, captions, test_size=test_size, random_state=42)
