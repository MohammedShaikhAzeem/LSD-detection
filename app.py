import streamlit as st
import numpy as np
import os
import shutil
import xlsxwriter
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Flatten
from keras.models import Model
from glob import glob
from PIL import Image

# Load your model (make sure to save your model after training)
model = load_model('C:\\cattle\\lumpy_skin_model.h5')  # Use double backslashes or raw string

# Define paths
train_path = 'C:\\cattle\\cattledataset\\TrainData'
valid_path = 'C:\\cattle\\cattledataset\\TestData'

# Create an instance of ImageDataGenerator for training data
train_datagen = ImageDataGenerator(rescale=1./255)

# Load the training data
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),  # Resize images to match model input
    batch_size=32,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Streamlit UI
st.title("Cattle Disease Classification")
st.write("Upload an image of a cattle to classify its health status.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make predictions
    predictions = model.predict(processed_image)
    class_indices = {v: k for k, v in training_set.class_indices.items()}  # Get class indices from training_set
    predicted_class = class_indices[np.argmax(predictions)]
    
    st.write(f"Predicted Class: {predicted_class}")
    
    # Optionally, you can save the results to an Excel file
    if st.button("Save Results to Excel"):
        excel_dir = 'C:\\cattle'
        os.makedirs(excel_dir, exist_ok=True)
        workbook = xlsxwriter.Workbook(os.path.join(excel_dir, 'MobileImagesAnalysis-2.xlsx'))
        worksheet = workbook.add_worksheet()
        
        # Write headers
        worksheet.write(0, 0, "Image Name")
        worksheet.write(0, 1, "Predicted Class")
        
        # Write data
        worksheet.write(1, 0, uploaded_file.name)
        worksheet.write(1, 1, predicted_class)
        
        workbook.close()
        st.success("Results saved to Excel file.")

# Optionally, you can add more features like visualizing training history, confusion matrix, etc.