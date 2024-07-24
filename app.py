import streamlit as st
from google.cloud import vision
from google.cloud.vision import types
import io

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

# Streamlit app
st.title("Product Image Search")

st.write("Upload an image to search for product labels.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image file
    image = uploaded_file.read()

    # Perform label detection
    image = types.Image(content=image)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Labels:")

    for label in labels:
        st.write(label.description)
