# app.py

import streamlit as st
from PIL import Image
import numpy as np
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from colorthief import ColorThief

# Define functions
def extract_dominant_color(image_path, quality=10):
    try:
        color_thief = ColorThief(image_path)
        dominant_color = color_thief.get_color(quality=quality)
        return np.array(dominant_color)
    except Exception as e:
        print(f"Error extracting color from {image_path}: {e}")
        return np.array([0, 0, 0])  # Default to black if there's an error

def extract_shape_features(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            shape_features = cv2.boundingRect(contour)
            return shape_features
        return (0, 0, 0, 0)
    except Exception as e:
        print(f"Error extracting shape from {image_path}: {e}")
        return (0, 0, 0, 0)

def load_dataset_and_extract_features(dataset_path):
    image_paths = glob.glob(os.path.join(dataset_path, '**', '*_no_bg.png'), recursive=True)
    features = []
    for image_path in image_paths:
        color = extract_dominant_color(image_path)
        shape = extract_shape_features(image_path)
        features.append((image_path, color, shape))
    return features

def find_similar_images(uploaded_image_color, uploaded_image_shape, dataset_images, top_n=5):
    similarities = []
    for image_path, color, shape in dataset_images:
        color_similarity = cosine_similarity([uploaded_image_color], [color])[0][0]
        shape_similarity = np.linalg.norm(np.array(uploaded_image_shape) - np.array(shape))
        total_similarity = color_similarity - 0.1 * shape_similarity  # Adjust weighting as needed
        similarities.append((image_path, total_similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Streamlit application
st.title('Fashion Image Search by Color and Shape')

st.write("Upload an image to find similar images based on color and shape from the dataset.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image_path = "uploaded_image.jpg"
    uploaded_image = Image.open(uploaded_file).convert('RGB')
    uploaded_image.save(uploaded_image_path)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Extract the dominant color and shape of the uploaded image
    uploaded_image_color = extract_dominant_color(uploaded_image_path)
    uploaded_image_shape = extract_shape_features(uploaded_image_path)
    st.write(f"Uploaded Image Color: {uploaded_image_color}")
    st.write(f"Uploaded Image Shape: {uploaded_image_shape}")

    # Load and process dataset images
    dataset_path = 'fashion_dataset'
    dataset_images = load_dataset_and_extract_features(dataset_path)

    # Find similar images
    similar_images = find_similar_images(uploaded_image_color, uploaded_image_shape, dataset_images)
    
    if similar_images:
        st.write("Similar Images:")
        for image_path, similarity in similar_images:
            st.image(image_path, caption=f"{os.path.basename(image_path)} (Similarity: {similarity:.2f})", use_column_width=True)
    else:
        st.write("No similar images found.")
