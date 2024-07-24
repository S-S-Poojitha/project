import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset and select the first 200 images
def load_cifar10_dataset(num_images=200):
    (train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    return train_images[:num_images]

# Load a pre-trained model for feature extraction
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.resnet50.preprocess_input(image_array)
    return image_array

# Extract features from an image
def get_features(image_array):
    features = model.predict(image_array)
    return features.flatten()

# Convert CIFAR-10 images to features
def convert_cifar10_to_features(images):
    features_list = []
    for img in images:
        img = Image.fromarray(img)
        img_array = preprocess_image(img)
        features = get_features(img_array)
        features_list.append(features)
    return features_list

# Load CIFAR-10 images and compute features
cifar10_images = load_cifar10_dataset()
cifar10_features = convert_cifar10_to_features(cifar10_images)

# Function to find similar images
def find_similar_images(uploaded_image_features, dataset_features, top_n=5):
    similarities = []
    for features in dataset_features:
        similarity = cosine_similarity([uploaded_image_features], [features])[0][0]
        similarities.append(similarity)
    indices = np.argsort(similarities)[::-1][:top_n]
    return indices

# Streamlit application
st.title('CIFAR-10 Image Search')

st.write("Upload an image to find similar images from the CIFAR-10 dataset.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file).convert('RGB')
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    uploaded_image_array = preprocess_image(uploaded_image)
    uploaded_image_features = get_features(uploaded_image_array)

    similar_indices = find_similar_images(uploaded_image_features, cifar10_features)

    st.write("Similar Images:")
    for idx in similar_indices:
        similar_image = cifar10_images[idx]
        st.image(similar_image, caption=f"Similar Image {idx}", use_column_width=True)
