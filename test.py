import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# Define paths to your reduced dataset folders
normal_folder = "/Users/apoorv/Desktop/data/normal"
osteo_folder = "/Users/apoorv/Desktop/data/osteo"

# Function to load and preprocess images with target size 80x80 and in grayscale
def load_images(folder, reduce_factor=0.5):
    image_files = glob(os.path.join(folder, '*.png'))  # Assuming images are .png format
    image_files = image_files[:int(len(image_files) * reduce_factor)]  # Reduce dataset size
    images = []
    print(f"Found {len(image_files)} images in {folder}.")
    for i, img_path in enumerate(image_files):
        if i % 100 == 0:
            print(f"Loading image {i + 1}/{len(image_files)}")
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(80, 80), color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0  # Normalize pixel values to [0, 1]
        images.append(img)
    return np.array(images)

# Load and preprocess reduced images from folders
normal_images = load_images(normal_folder, reduce_factor=0.5)
osteo_images = load_images(osteo_folder, reduce_factor=0.5)

# Check image shapes (debugging purposes)
print(f'Normal images shape: {normal_images.shape}')  # Should be (1000, 80, 80, 1)
print(f'Osteo images shape: {osteo_images.shape}')  # Should be (1000, 80, 80, 1)

# Create labels (0 for normal, 1 for osteo) and convert to one-hot encoded masks
normal_labels = np.zeros((len(normal_images), 80, 80, 1))  # Masks for normal images
osteo_labels = np.ones((len(osteo_images), 80, 80, 1))  # Masks for osteo images

# Concatenate reduced images and labels
X = np.concatenate((normal_images, osteo_images), axis=0)
y = np.concatenate((normal_labels, osteo_labels), axis=0)

# Split reduced data into train, test, and validation sets (8:1:1 split)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define U-Net model with dropout layers
def unet_model(input_shape):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck with dropout
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder
    up6 = UpSampling2D(size=(2, 2))(drop5)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Create the model with updated input shape for grayscale images
model = unet_model(input_shape=(80, 80, 1))

# Train the model with 45 epochs
history = model.fit(X_train, y_train, batch_size=16, epochs=45, validation_data=(X_valid, y_valid))

# Save the model as .h5 file
model.save("/Users/jatinbalchandani/Desktop/IBMModel/unet_model.h5")

# Plot accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Confusion matrix
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(np.uint8)  # Convert probabilities to binary predictions
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()
cm = confusion_matrix(y_test_flat, y_pred_flat)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={'fontsize': 12, 'fontweight': 'bold', 'color': 'black'},
            linewidths=1, linecolor='black', square=True)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

print("Model saved successfully as unet_model.h5")