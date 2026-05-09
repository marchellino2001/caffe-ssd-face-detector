import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_xception_mask_detector(input_shape=(299, 299, 3), num_classes=2):
    """
    Builds the Xception model for Face Mask Detection.
    """
    print("[INFO] Loading pre-trained Xception model...")
    # 1. Load the base Xception model (excluding the top Dense layers)
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

    # 2. Freeze the base model layers so they aren't updated during the first training phase
    for layer in base_model.layers:
        layer.trainable = False

    # 3. Construct the new head of the network
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x) # Prevents overfitting
    
    # Final layer: 2 classes (Mask vs No Mask). Using softmax for probability output.
    predictions = Dense(num_classes, activation='softmax')(x)

    # 4. Combine the base model and the new head
    model = Model(inputs=base_model.input, outputs=predictions)

    # 5. Compile the model
    # The paper mentions using the ADAM optimizer.
    print("[INFO] Compiling model...")
    opt = Adam(learning_rate=1e-4)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    return model

if __name__ == "__main__":
    # Test the architecture
    model = build_xception_mask_detector()
    model.summary()
    print("\n[SUCCESS] Model architecture is ready!")