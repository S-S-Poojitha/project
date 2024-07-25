# scraper.py

from bing_image_downloader import downloader
from rembg import remove
from PIL import Image
import os
import glob
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np

# Configuration
base_queries = ["decorations", "phones", "bags", "watches", "shoes"]
colors = ["red", "blue", "green", "yellow", "black", "white", "pink", "purple", "orange", "brown"]
num_images = 10
output_dir = 'fashion_dataset'
target_labels = [0]  # Replace with actual target labels based on your classification needs

# Download images
def download_images():
    for base_query in base_queries:
        for color in colors:
            query = f"{base_query} {color}"
            downloader.download(
                query,
                limit=num_images,
                output_dir=output_dir,
                adult_filter_off=True,
                force_replace=False,
                timeout=60,
                verbose=True
            )

# Remove background
def remove_background(input_image_path, output_image_path):
    input_image = Image.open(input_image_path)
    output_image = remove(input_image)
    output_image.save(output_image_path)

def process_images():
    for category_dir in glob.glob(os.path.join(output_dir, '*')):
        for image_file in glob.glob(os.path.join(category_dir, '*')):
            output_path = image_file.replace(".jpg", "_no_bg.png").replace(".png", "_no_bg.png")
            remove_background(image_file, output_path)
            os.remove(image_file)  # Optionally remove the original image
            print(f"Processed {image_file} to {output_path}")

# Classify images
def classify_image(image_path, model, preprocess):
    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def filter_images():
    model = models.resnet50(pretrained=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for category_dir in glob.glob(os.path.join(output_dir, '*')):
        for image_file in glob.glob(os.path.join(category_dir, '*')):
            label = classify_image(image_file, model, preprocess)
            if label in target_labels:
                print(f"Image {image_file} classified as {label}")
            else:
                os.remove(image_file)  # Remove unwanted images

if __name__ == "__main__":
    download_images()
    process_images()
    filter_images()
