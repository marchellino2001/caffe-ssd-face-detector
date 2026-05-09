import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- Configuration ---
# Ensure these three files are in the same folder as this script
PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
MASK_MODEL_PATH = "mask_detector_gpu.h5"

# Load the face detector model (SSD)
print("[INFO] Loading Face Detector...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

# Load the mask detector model (Xception)
print("[INFO] Loading Mask Detector...")
mask_model = load_model(MASK_MODEL_PATH)

# Initialize the webcam (0 is the default built-in webcam)
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    h, w = frame.shape[:2]

    # Preprocess the frame for the SSD face detection
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            # Ensure the bounding boxes fall exactly within the frame dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Extract the face Region of Interest (ROI)
            face = frame[y1:y2, x1:x2]
            
            # If the face is too small or invalid, skip it to prevent crashes
            if face.shape[0] < 10 or face.shape[1] < 10:
                continue

            # Preprocess the face for the mask detector (Xception requirements)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (299, 299))
            face_array = img_to_array(face_resized) / 255.0
            face_expanded = np.expand_dims(face_array, axis=0)

            # Predict if the face has a mask or not
            # Returns probabilities: [With Mask, Without Mask]
            predictions = mask_model.predict(face_expanded, verbose=0)[0]
            mask_prob, without_mask_prob = predictions
            
            # Determine the class label and color
            label = "Mask" if mask_prob > without_mask_prob else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # Green for Mask, Red for No Mask
            
            # Include the probability percentage in the label
            label = f"{label}: {max(mask_prob, without_mask_prob) * 100:.2f}%"

            # Display the label and bounding box on the output frame
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Show the output frame
    cv2.imshow("Face Mask Detector - Press 'q' to quit", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()