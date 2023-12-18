import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model_path = 'C://Users//ASUS//Documents//Python Folder//Python Tutor//ProjectAkhirMultimedia//mobileNet_v2.h5'
model = load_model(model_path)

# Function to make predictions
def predict_spice(image):
    # Convert uploaded image to PIL format
    img_pil = Image.open(image)
    img_pil = img_pil.resize((224, 224))  # Resize image
    img_array = np.array(img_pil)  # Convert PIL image to array
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    return prediction

# Streamlit app
st.title('Predict Spice from Image')
st.write('Upload an image of a spice to predict its type!')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make a prediction
    prediction = predict_spice(uploaded_file)
    class_names = ['jahe_emprit', 'jahe_merah', 'jahe_putih', 'kencur', 'kunyit_hitam', 'kunyit_kuning', 'kunyit_putih', 'lengkuas', 'temulawak']
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"Predicted Spice: {predicted_class}")



