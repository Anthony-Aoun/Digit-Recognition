import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from train_model import CNNModel 

# Load the trained model
model = CNNModel()
model.load_state_dict(torch.load('cnn_mnist_model.pth'))  # Load the .pth model
model.eval()

def preprocess_image(img_path):
    original_img = Image.open(img_path).convert('L')  # Convert to grayscale
    img_array = np.array(original_img)
    
    # Check background brightness and invert if necessary
    if np.mean(img_array) > 128:  # Make it dark background
        img_array = 255 - img_array  # Invert colors

    # Binarize the image with a threshold
    threshold = 128
    img_array = (img_array > threshold) * 255  # White digit, black background

    # Convert back to PIL Image and apply transforms
    processed_img = Image.fromarray(np.uint8(img_array))
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(processed_img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return original_img, processed_img, img_tensor

def predict_image(img_path):
    original_img, processed_img, img_tensor = preprocess_image(img_path)
    with torch.no_grad():
        output = model(img_tensor)
        predicted_digit = torch.argmax(output, dim=1).item()
    
    # Plotting original and preprocessed images
    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(processed_img, cmap='gray')
    axes[1].set_title('Preprocessed Image')
    axes[1].axis('off')
    
    plt.suptitle(f'Predicted Digit: {predicted_digit}')
    plt.show()

# Directory containing images to test
image_folder = 'images to test'

# Iterate through images in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_folder, filename)
        predict_image(img_path)
