import torch
from torchvision import models, transforms
from PIL import Image

# Initialize DenseNet model (pre-trained on ImageNet)
def initialize_densenet():
    model = models.densenet121(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model
def get_preprocess():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess

# Extract DenseNet features
def extract_features(img_path, densenet_model,preprocess, device):
    
    img = Image.open(img_path)
    
    # Preprocess the image
    img_t = preprocess(img)
    
    # Create a mini-batch as expected by the model
    batch_t = torch.unsqueeze(img_t, 0).to(device)
    
    # Extract features
    with torch.no_grad():  # Disable gradient calculation
        features = densenet_model(batch_t)
    print("Features extracted successfully!")
    return features.cpu().numpy()
