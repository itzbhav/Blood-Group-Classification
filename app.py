# Import required libraries
import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import random

# Set page configuration FIRST
st.set_page_config(page_title="Blood Group Detection", layout="wide")

# Title of the app
st.title("🩸 Blood Group Detection using LeNet Model")

# Sidebar navigation
st.sidebar.header("Navigation")
selected_option = st.sidebar.selectbox(
    "Select an option:",
    ["Home", "EDA", "Predict Blood Group"]
)

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model('bloodgroup_mobilenet_finetuned.h5')  # Ensure correct path
    return model

model = load_trained_model()

# Define class labels
class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Dataset directory (you must adjust this if needed)
DATASET_DIR = "dataset"  # Example path

# Home page
if selected_option == "Home":
    st.subheader("About the Project")
    st.write("""
        Welcome to the Blood Group Detection App! 
        
        This application uses a Deep Learning model (LeNet architecture) to detect blood groups from blood sample images.
        
        ### 🛠 Technologies Used:
        - Streamlit for Web UI
        - TensorFlow/Keras for Deep Learning
        - Image Processing with Computer Vision

        **Upload a blood sample image and predict the blood group instantly!**
    """)
    try:
        st.image("blood_home.jpg", caption="Blood Sample Analysis", use_column_width=True)
    except:
        st.warning("Home image not found. (Optional)")

# EDA page
elif selected_option == "EDA":
    st.subheader("Exploratory Data Analysis (EDA)")

    # Check if dataset exists
    if os.path.exists(DATASET_DIR):
        st.write("### 📊 Number of Images per Blood Group:")

        counts = {}
        for class_name in class_names:
            class_path = os.path.join(DATASET_DIR, class_name)
            if os.path.exists(class_path):
                counts[class_name] = len(os.listdir(class_path))
            else:
                counts[class_name] = 0

        df_counts = pd.DataFrame(list(counts.items()), columns=['Blood Group', 'Number of Images'])
        st.dataframe(df_counts)

        # Bar Chart
        st.bar_chart(df_counts.set_index('Blood Group'))

        st.write("### 🖼️ Sample Images from Each Class:")

        cols = st.columns(4)  # create 4 columns

        for idx, class_name in enumerate(class_names):
            class_path = os.path.join(DATASET_DIR, class_name)
            if os.path.exists(class_path) and len(os.listdir(class_path)) > 0:
                img_file = random.choice(os.listdir(class_path))
                img_path = os.path.join(class_path, img_file)
                img = image.load_img(img_path, target_size=(64, 64))  # resized
                with cols[idx % 4]:  # arrange in 4 columns
                    st.image(img, caption=class_name, width=150)

        st.write("### 🧩 Image Properties:")
        sample_class = class_names[0]
        sample_path = os.path.join(DATASET_DIR, sample_class, os.listdir(os.path.join(DATASET_DIR, sample_class))[0])
        sample_img = image.load_img(sample_path)
        st.write(f"- **Image shape:** {np.array(sample_img).shape}")
        st.write(f"- **Color channels:** {np.array(sample_img).shape[-1]} (RGB)")

    else:
        st.warning("Dataset not found! Please make sure the 'dataset/train' folder exists.")

# Prediction page
elif selected_option == "Predict Blood Group":
    st.subheader("Upload an Image to Predict Blood Group")

    uploaded_file = st.file_uploader("Choose a blood sample image...", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Ensure temp directory exists
        if not os.path.exists('temp'):
            os.makedirs('temp')

        # Save uploaded file temporarily
        temp_file_path = os.path.join("temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Preprocess the image
        img = image.load_img(temp_file_path, target_size=(224, 224))  # Adjust size if needed
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values

        # Predict the blood group
        with st.spinner('Predicting...'):
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]

        # Show result
        st.success(f"🧬 Predicted Blood Group: **{predicted_class}**")

        # Remove temporary file
        os.remove(temp_file_path)
