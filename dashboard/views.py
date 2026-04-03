import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg') # Prevents server crashes when drawing images

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.applications.efficientnet import preprocess_input

# Import your custom Grad-CAM functions
from gradcam import make_gradcam_heatmap, save_and_display_gradcam

# 1. LOAD THE MODEL ONCE WHEN THE SERVER STARTS
MODEL_PATH = "models/dr_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Dictionary to translate the AI's 0-4 output into English
DISEASE_CLASSES = {
    0: "No Diabetic Retinopathy (Healthy)",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}

def index(request):
    if request.method == 'POST' and request.FILES.get('eye_image'):
        uploaded_file = request.FILES['eye_image']
        
        # 2. SAVE THE RAW UPLOADED IMAGE
        fss = FileSystemStorage()
        saved_filename = fss.save(uploaded_file.name, uploaded_file)
        raw_image_url = fss.url(saved_filename)
        img_path = fss.path(saved_filename)
        
        # 3. PREPARE THE IMAGE FOR EFFICIENTNET
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img_array = img.astype("float32")
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # 4. MAKE THE PREDICTION
        predictions = model.predict(img_array)
        pred_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        diagnosis = DISEASE_CLASSES[pred_index]
        
        # 5. GENERATE THE XAI HEATMAP
        heatmap = make_gradcam_heatmap(img_array, model, "top_conv")
        
        # Save the heatmap so the webpage can display it
        heatmap_filename = "heatmap_" + saved_filename
        heatmap_path = fss.path(heatmap_filename)
        save_and_display_gradcam(img_path, heatmap, cam_path=heatmap_path)
        heatmap_url = fss.url(heatmap_filename)
        
        # 6. SEND EVERYTHING TO THE HTML PAGE
        return render(request, 'dashboard/index.html', {
            'diagnosis': diagnosis,
            'confidence': f"{confidence:.2f}%",
            'raw_image_url': raw_image_url,
            'heatmap_url': heatmap_url
        })

    return render(request, 'dashboard/index.html')