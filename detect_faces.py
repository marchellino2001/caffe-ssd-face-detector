import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.applications.xception import preprocess_input

PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
MASK_MODEL_PATH = "mask_detector_gpu.h5"

class FixedBatchNormalization(BatchNormalization):
    @classmethod
    def from_config(cls, config):
        config.pop("renorm", None)
        config.pop("renorm_clipping", None)
        config.pop("renorm_momentum", None)
        config.pop("fused", None)
        config.pop("virtual_batch_size", None)
        config.pop("adjustment", None)
        return cls(**config)

class FixedDense(Dense):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return cls(**config)

print("[INFO] Loading Face Detector...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

print("[INFO] Loading Mask Detector...")
mask_model = load_model(
    MASK_MODEL_PATH,
    compile=False,
    custom_objects={
        "BatchNormalization": FixedBatchNormalization,
        "Dense": FixedDense
    }
)

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (299, 299))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype("float32")
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            pred = mask_model.predict(face, verbose=0)[0]

            if len(pred) == 1:
                mask_prob = float(pred[0])
                label = "No Mask" if mask_prob >= 0.5 else "Mask"
                prob = mask_prob if mask_prob >= 0.5 else 1 - mask_prob
            else:
                no_mask_prob = float(pred[0])
                mask_prob = float(pred[1])
                label = "No Mask" if mask_prob > no_mask_prob else "Mask"
                prob = max(mask_prob, no_mask_prob)

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label}: {prob:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    cv2.imshow("Final Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
