import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# 1. FIX: Limit GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 2. Lower Batch Size to 8 to prevent OOM
BS = 8
TRAIN_PATH = "/mnt/c/Users/zed1f/Downloads/Cleaned_Data/Train"
VAL_PATH = "/mnt/c/Users/zed1f/Downloads/Cleaned_Data/Validation"

def test_model(model_name, base_model_class):
    # CLEAR PREVIOUS MODEL FROM GPU MEMORY
    tf.keras.backend.clear_session()
    
    print(f"\n[STARTING] Testing {model_name}...")
    
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
        TRAIN_PATH, target_size=(224, 224), batch_size=BS, class_mode="categorical")
    val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
        VAL_PATH, target_size=(224, 224), batch_size=BS, class_mode="categorical")

    base = base_model_class(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    
    # Freeze most layers to speed up and save memory
    base.trainable = False 

    x = AveragePooling2D(pool_size=(7, 7))(base.output)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x) # Smaller dense layer to save RAM
    x = Dropout(0.5)(x)
    output = Dense(2, activation="softmax")(x)
    
    model = Model(inputs=base.input, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-4), metrics=["accuracy"])
    
    # Train for only 3 epochs (enough for comparison)
    H = model.fit(train_gen, validation_data=val_gen, epochs=3, verbose=1)
    return max(H.history['val_accuracy'])

results = {}
for name, m_class in [("VGG16", VGG16), ("ResNet50", ResNet50)]:
    try:
        results[name] = test_model(name, m_class)
    except Exception as e:
        print(f"[FAILED] {name} due to: {e}")

print("\n--- FINAL COMPARISON RESULTS ---")
for name, acc in results.items():
    print(f"{name}: {acc*100:.2f}%")