import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.applications.efficientnet import preprocess_input
from gradcam import make_gradcam_heatmap, save_and_display_gradcam

# -------------------------
# Load Saved Model
# -------------------------
model = tf.keras.models.load_model("models/dr_model.keras")
print("Model loaded successfully!")

# Build model once
dummy_input = np.zeros((1, 224, 224, 3))
model.predict(dummy_input)

# -------------------------
# Select Test Folder
# -------------------------
test_folder = "data/augmented_resized_V2/test/0"

images = os.listdir(test_folder)[:5]   # Take first 5 images

print("Processing 5 images...")

for idx, img_name in enumerate(images):

    img_path = os.path.join(test_folder, img_name)
    print(f"Processing: {img_path}")

    # Read image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")

    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, "top_conv")

    # Convert heatmap to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Threshold (keep strong activations)
    _, thresh = cv2.threshold(heatmap_uint8, 180, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image
    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))

    cv2.drawContours(original, contours, -1, (0,255,0), 2)

    cv2.imwrite("outputs/gradcam_with_contours.jpg", original)


    # Save with unique name
    output_path = f"outputs/gradcam_result_{idx+1}.jpg"
    save_and_display_gradcam(img_path, heatmap, cam_path=output_path)

print("All 5 Grad-CAM images generated successfully!")
