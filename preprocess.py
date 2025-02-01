import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
INPUT_DIR = r"C:\Users\sajal\Desktop\SEPM Project\Dataset aug"  # Change this to your dataset path
OUTPUT_DIR = r"C:\Users\sajal\Desktop\SEPM Project\preprocessed dataset"  # Where to save preprocessed images

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to apply preprocessing steps
def preprocess_image(image_path, output_path, img_size=224):
    # Read the image
    img = cv2.imread(image_path)

    # Apply Bilateral Filtering for noise reduction
    img_denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Split the image into its channels
    channels = cv2.split(img_denoised)

    # Apply CLAHE (Adaptive Histogram Equalization) to each channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = [clahe.apply(channel) for channel in channels]

    # Merge the channels back
    img_clahe = cv2.merge(channels)

    # Resize while maintaining aspect ratio using zero-padding
    h, w, _ = img_clahe.shape
    scale = img_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = cv2.resize(img_clahe, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank canvas and center the resized image
    img_padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    img_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized

    # Normalize pixel values (0-1 range)
    img_normalized = img_padded / 255.0

    # Save preprocessed image
    cv2.imwrite(output_path, (img_normalized * 255).astype(np.uint8))  # Convert back to 0-255 for saving

# Process all images in dataset
for class_folder in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, class_folder)
    output_class_path = os.path.join(OUTPUT_DIR, class_folder)

    if not os.path.isdir(class_path):
        continue  # Skip non-folder files

    os.makedirs(output_class_path, exist_ok=True)

    for img_file in tqdm(os.listdir(class_path), desc=f"Processing {class_folder}"):
        img_path = os.path.join(class_path, img_file)
        output_path = os.path.join(output_class_path, img_file)

        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            preprocess_image(img_path, output_path)
