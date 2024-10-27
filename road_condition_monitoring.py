import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the pre-trained model
model = tf.keras.models.load_model('road_condition_model.h5')

# Define labels as per the classes used in training
labels = ['normal', 'pothole', 'crack']  # Adjust according to your classes

def alert_driver(message):
    engine.say(message)
    engine.runAndWait()

# Video capture from default camera or an IP camera
cap = cv2.VideoCapture('http://192.0.0.4:8080/video')  # Replace with your camera URL
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

# Cooldown settings to prevent repeated alerts
last_alert_time = 0
cooldown_time = 5  # Cooldown in seconds
frame_count = 0  # Frame counter

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Process every 5th frame for faster performance
    frame_count += 1
    if frame_count % 5 != 0:
        continue

    # Preprocess the frame for the model
    resized_frame = cv2.resize(frame, (150, 150))  # Ensure size matches model input
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Make predictions
    predictions = model.predict(input_frame)
    predicted_class = labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display the prediction on the video feed
    cv2.putText(frame, f'Detected: {predicted_class} ({confidence:.2f})', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Trigger alert if pothole or crack is detected
    current_time = time.time()
    if predicted_class in ['pothole', 'crack'] and confidence > 0.6:  # Adjust threshold if needed
        if current_time - last_alert_time > cooldown_time:
            alert_driver(f"{predicted_class.capitalize()} detected. Please slow down.")
            last_alert_time = current_time

    # Show the video feed
    cv2.imshow('Road Condition Monitoring', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
