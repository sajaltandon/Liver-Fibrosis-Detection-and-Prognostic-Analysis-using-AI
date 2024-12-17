import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Path to the dataset folder
dataset_dir = r"C:\Users\sajal\Desktop\SEPM Project\Dataset aug"

# Number of images per class
target_class_size = 2000

# Set up image data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to augment and save images
def augment_images_from_class(class_dir, target_class_size):
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
    current_count = len(images)

    if current_count >= target_class_size:
        print(f"Class {class_dir} already has {current_count} images. Skipping augmentation.")
        return

    # Calculate how many new images we need to create
    images_to_generate = target_class_size - current_count
    print(f"Augmenting {images_to_generate} images for class {class_dir}...")

    # Load the images and augment them
    for img_path in images:
        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Generate augmented images and save them
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=class_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= images_to_generate:
                break

        # Stop once we reach the target number of images
        if current_count + i >= target_class_size:
            break

# Loop over each class and balance them
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)

    if os.path.isdir(class_dir):
        augment_images_from_class(class_dir, target_class_size)
