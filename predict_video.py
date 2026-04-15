import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("deepfake_model.h5")

video_path = input("Enter video path: ")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Process every 20th frame (faster)
    if frame_count % 20 == 0:

        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]

        fake_score = prediction
        real_score = 1 - prediction

        print("\nFrame:", frame_count)
        print("Fake probability:", round(fake_score*100,2), "%")
        print("Real probability:", round(real_score*100,2), "%")

        # Explanation logic
        if fake_score > 0.7:
            print("Explanation: Face artifacts or unnatural blending detected.")
        elif fake_score > 0.5:
            print("Explanation: Possible lighting mismatch or facial inconsistency.")
        else:
            print("Explanation: Face structure and lighting appear natural.")

    frame_count += 1

cap.release()

print("\nVideo analysis completed.")