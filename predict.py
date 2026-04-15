import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("deepfake_model.h5")

# Get image path from user
img_path = input("Enter image path: ")

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

# Output
if prediction[0][0] > 0.5:
    print("Prediction: FAKE")
    print("Confidence:", round(prediction[0][0] * 100, 2), "%")
else:
    print("Prediction: REAL")
    print("Confidence:", round((1 - prediction[0][0]) * 100, 2), "%")