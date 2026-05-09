import os
import cv2
import zipfile
import numpy as np
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (299, 299)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    merged = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        return False
    img = cv2.resize(img, IMG_SIZE)
    img = apply_clahe(img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    return True

def augment_images(input_dir, num_augmentations=2):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    
    for category in ['with_mask', 'without_mask']:
        in_path = os.path.join(input_dir, category)
        if not os.path.exists(in_path):
            continue
        
        images = [f for f in os.listdir(in_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        original_images = [f for f in images if 'aug' not in f]
        
        for img_name in original_images:
            img_path = os.path.join(in_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img = img.astype(np.float32) / 255.0
            img_array = np.expand_dims(img, axis=0)
            name, ext = os.path.splitext(img_name)
            
            aug_gen = datagen.flow(img_array, batch_size=1)
            for i in range(num_augmentations):
                aug_img = next(aug_gen)[0]
                aug_img = (aug_img * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(in_path, f"{name}_aug{i+1}{ext}"), aug_img)
        
        print(f"Augmented {category}: {num_augmentations} copies each")

def main():
    parser = argparse.ArgumentParser(description="Preprocess face mask dataset")
    parser.add_argument("--input", type=str, default="FaceMaskProject",
                       help="Input directory containing Train/Validation/Test folders")
    parser.add_argument("--output", type=str, default="Cleaned_Data",
                       help="Output directory for processed data")
    parser.add_argument("--augment", action="store_true",
                       help="Apply offline augmentation (not recommended if using online augmentation in training)")
    parser.add_argument("--num_aug", type=int, default=2,
                       help="Number of augmentations per image (if --augment is used)")
    args = parser.parse_args()
    
    INPUT_DIR = args.input
    OUTPUT_DIR = args.output
    DO_AUGMENT = args.augment
    NUM_AUG = args.num_aug
    
    print("=" * 50)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 50)
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} not found!")
        return
    
    print("\nStep 1: Processing images...")
    
    splits = ['Train', 'Validation', 'Test']
    
    for split in splits:
        for category in ['With Mask', 'Without Mask']:
            input_path = os.path.join(INPUT_DIR, split, category)
            
            if not os.path.exists(input_path):
                print(f"Warning: {input_path} not found")
                continue
            
            output_cat = 'with_mask' if category == 'With Mask' else 'without_mask'
            output_split = split.lower()
            output_path = os.path.join(OUTPUT_DIR, f"{output_split}_processed", output_cat)
            
            images = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Processing {len(images)} images from {split}/{category}")
            
            for img_name in images:
                img_input = os.path.join(input_path, img_name)
                img_output = os.path.join(output_path, img_name)
                process_image(img_input, img_output)
    
    print("\nStep 2: Applying augmentation to train set (if requested)...")
    if DO_AUGMENT:
        train_processed = os.path.join(OUTPUT_DIR, 'train_processed')
        if os.path.exists(train_processed):
            augment_images(train_processed, num_augmentations=NUM_AUG)
        else:
            print(f"Warning: {train_processed} not found")
    else:
        print("Skipping offline augmentation (use --augment to enable)")
    
    print("\nStep 3: Creating zip archive...")
    output_zip = f"{OUTPUT_DIR}.zip"
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files_list in os.walk(OUTPUT_DIR):
            for file in files_list:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(OUTPUT_DIR))
                zipf.write(file_path, arcname)
    
    print(f"\n✅ DONE! Output: {output_zip}")
    print("=" * 50)
    return output_zip

if __name__ == "__main__":
    output_file = main()