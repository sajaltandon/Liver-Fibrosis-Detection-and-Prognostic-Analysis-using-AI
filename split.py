import os
import shutil
import random

# Define directories
train_dir = r"C:\Users\sajal\Desktop\SEPM Project\train_dataset"
test_dir = r"C:\Users\sajal\Desktop\SEPM Project\test_dataset"
input_dir = r"C:\Users\sajal\Desktop\SEPM Project\preprocessed dataset"  # Preprocessed dataset

# Create directories for training and testing sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split data for each class
for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)

    if not os.path.isdir(class_path):
        continue

    # Create subdirectories for class in train/test
    os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)

    # List images in class folder
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Shuffle the images
    random.shuffle(image_files)

    # Split into 80% training, 20% testing
    split_index = int(0.8 * len(image_files))
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]

    # Move images to training set
    for file in train_files:
        src = os.path.join(class_path, file)
        dst = os.path.join(train_dir, class_folder, file)
        shutil.copy(src, dst)

    # Move images to testing set
    for file in test_files:
        src = os.path.join(class_path, file)
        dst = os.path.join(test_dir, class_folder, file)
        shutil.copy(src, dst)

print("Dataset split completed!")
