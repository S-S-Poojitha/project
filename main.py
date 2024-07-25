import streamlit as st
from PIL import Image
import numpy as np
from colorthief import ColorThief
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob

# Extract dominant color from an image
def extract_dominant_color(image_path):
    try:
        color_thief = ColorThief(image_path)
        dominant_color = color_thief.get_color(quality=1)
        return np.array(dominant_color)
    except Exception as e:
        print(f"Error extracting color from {image_path}: {e}")
        return np.array([0, 0, 0])  # Default to black if there's an error

# Load dataset and extract color features
def load_dataset_and_extract_colors(dataset_path):
    image_paths = glob.glob(os.path.join(dataset_path, '*.jpg')) + \
                  glob.glob(os.path.join(dataset_path, '*.jpeg'))
    color_features = []
    for image_path in image_paths:
        color = extract_dominant_color(image_path)
        if color is not None:
            color_features.append((image_path, color))
            print(f"Image: {image_path}, Dominant Color: {color}")  # Debug statement
        else:
            print(f"Skipping image {image_path} due to color extraction error")
    return color_features

# Define dataset path
dataset_path = 'fashion_dataset'  # Use the path where you downloaded images
dataset_images = load_dataset_and_extract_colors(dataset_path)

# Function to find similar images based on color
def find_similar_images(uploaded_image_color, dataset_images, top_n=3):
    similarities = []
    for image_path, color in dataset_images:
        similarity = cosine_similarity([uploaded_image_color], [color])[0][0]
        similarities.append((image_path, similarity))
        print(f"Comparing with {image_path}, Similarity: {similarity}")  # Debug statement
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Streamlit application
st.title('Product image search')

st.write("Upload an image to find similar images based on color from the dataset.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image_path = "uploaded_image.jpg"
    uploaded_image = Image.open(uploaded_file).convert('RGB')
    uploaded_image.save(uploaded_image_path)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Extract the dominant color of the uploaded image
    uploaded_image_color = extract_dominant_color(uploaded_image_path)
    st.write(f"Uploaded Image Color: {uploaded_image_color}")

    # Find similar images
    similar_images = find_similar_images(uploaded_image_color, dataset_images)
    
    if similar_images:
        st.write("Similar Images:")
        for image_path, similarity in similar_images:
            st.image(image_path, caption=f"{os.path.basename(image_path)} (Similarity: {similarity:.2f})", use_column_width=True)
    else:
        st.write("No similar images found.")
