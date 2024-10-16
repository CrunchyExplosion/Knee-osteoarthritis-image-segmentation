import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Load the saved U-Net model
model_path = "MODELS/unet_model1.h5"
model = load_model(model_path)

# Recompile the model with the desired metrics
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Function to preprocess input image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(80, 80), color_mode='grayscale')
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict and post-process the segmented image
def segment_image_with_prediction(model, image_path, threshold=0.5):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    segmented_image = (prediction > threshold).astype(np.uint8)  # Apply threshold

    # Calculate percentage of area that is classified as osteoarthritis
    osteo_percentage = np.mean(segmented_image) * 100

    # Determine if osteoarthritis is present based on the segmented image
    if osteo_percentage > 0:
        prediction_text = "Yes, osteoarthritis detected"
    else:
        prediction_text = "No, osteoarthritis not detected"

    return segmented_image, prediction_text

# Example usage
input_image_path = "test_normal.png"
segmented_image, prediction_text = segment_image_with_prediction(model, input_image_path)

print(f"Prediction: {prediction_text}")

# Plot the original and segmented images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
original_image = tf.keras.preprocessing.image.load_img(input_image_path, target_size=(80, 80), color_mode='grayscale')
plt.imshow(original_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(segmented_image.squeeze(), cmap='gray')
plt.axis('off')

plt.show()
