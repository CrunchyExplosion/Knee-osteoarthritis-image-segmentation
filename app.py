from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'your_secret_key'
# Load your trained model
model_path = "MODELS/model_5_epoch.h5"
model = load_model(model_path)

# Function to process uploaded image and return segmented image and prediction
def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (80, 80))
    
    # Convert grayscale image to 1-channel image
    img_input = np.expand_dims(img_resized, axis=-1)
    # for one channel
    #img_input = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_input = np.expand_dims(img_input, axis=0) / 255.0
    
    prediction = model.predict(img_input)
    segmented_image = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255  # Adjust threshold and scale
    
    # Overlay the segmentation on the original image
    img_overlay = cv2.resize(segmented_image, (img.shape[1], img.shape[0]))
    colored_overlay = cv2.applyColorMap(img_overlay, cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(colored_overlay, 0.5, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.5, 0)
    
    # Determine prediction text based on segmented image
    osteo_percentage = np.mean(segmented_image) * 100
    if osteo_percentage > 0:
        prediction_text = "Yes, osteoarthritis detected"
    else:
        prediction_text = "No, osteoarthritis not detected"
    
    return overlayed_image, prediction_text

# Route to upload file and display result with prediction
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file:
            # Save uploaded file directly into 'static' folder as 'just_uploaded.png'
            uploaded_filename = 'just_uploaded.png'
            uploaded_file_path = os.path.join(app.static_folder, uploaded_filename)
            file.save(uploaded_file_path)
            
            # Process uploaded image and get segmented image and prediction text
            segmented_image, prediction_text = process_image(uploaded_file_path)
            
            # Save processed image as 'just_processed.png' in 'static' folder
            processed_filename = 'just_processed.png'
            processed_file_path = os.path.join(app.static_folder, processed_filename)
            cv2.imwrite(processed_file_path, segmented_image)
            
            flash('Image successfully uploaded and processed')
            return render_template('index.html', upload_wali_filename=uploaded_filename, segmented_image_filename=processed_filename, prediction_text=prediction_text)
    
    # Render template without result if GET request or upload failed
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
