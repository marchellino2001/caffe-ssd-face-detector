import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1. High-Precision Settings
INIT_LR = 1e-4 # Slow & steady for high accuracy
EPOCHS = 20    
BS = 32

# 2. Corrected Data Paths (Windows Downloads -> WSL)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", required=True)
parser.add_argument("--val_dir",   required=True)
args = parser.parse_args()
TRAIN_DIR = args.train_dir
VAL_DIR   = args.val_dir

print("[INFO] Loading data from Downloads...")

# Augment training data to prevent memorization
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    zoom_range=0.15,
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.15,
    horizontal_flip=True, 
    fill_mode="nearest"
)

# Only rescale validation data (no augmentation for a pure test)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH, 
    target_size=(299, 299), 
    batch_size=BS,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    VAL_PATH, 
    target_size=(299, 299), 
    batch_size=BS,
    class_mode="categorical"
)

# 3. Build Model with Fine-Tuning
baseModel = Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

# UNFREEZE the last 20 layers
baseModel.trainable = True
for layer in baseModel.layers[:-20]:
    layer.trainable = False

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# 4. Compile and Train
print("[INFO] Compiling model...")
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=INIT_LR), metrics=["accuracy"])

print("[INFO] Starting GPU Training...")
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

print("[INFO] Saving FINAL model...")
model.save("mask_detector_gpu.h5")
print("[SUCCESS] Model trained and saved!")