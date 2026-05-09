import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import argparse

# 1. Settings
INIT_LR = 1e-4
EPOCHS = 10  # 10 is enough for a comparison test
BS = 32

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", default="/mnt/c/Users/zed1f/Downloads/Cleaned_Data/Train")
parser.add_argument("--val_dir", default="/mnt/c/Users/zed1f/Downloads/Cleaned_Data/Validation")
args = parser.parse_args()

print("[INFO] Loading data for MobileNetV2 Comparison...")

train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode="nearest"
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    args.train_dir, target_size=(299, 299), batch_size=BS, class_mode="categorical"
)
val_data = val_datagen.flow_from_directory(
    args.val_dir, target_size=(299, 299), batch_size=BS, class_mode="categorical"
)

# 2. Build MobileNetV2 Model
print("[INFO] Loading MobileNetV2...")
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

# Unfreeze the last 20 layers for a fair comparison
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

# 3. Compile and Train
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=INIT_LR), metrics=["accuracy"])

print("[INFO] Starting MobileNetV2 Training...")
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

print("[SUCCESS] Comparison Training Complete! Check the final val_accuracy.")