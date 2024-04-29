import os
import shutil
import random

# Define the directories
data_dir = 'data'
images_dir = 'images'
masks_dir = 'masks'
train_images_dir = os.path.join(data_dir, 'train_images')
train_masks_dir = os.path.join(data_dir, 'train_masks')
test_images_dir = os.path.join(data_dir, 'test_images')
test_masks_dir = os.path.join(data_dir, 'test_masks')
val_images_dir = os.path.join(data_dir, 'val_images')
val_masks_dir = os.path.join(data_dir, 'val_masks')

# Create directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_masks_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)

# Get the list of all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Shuffle the list to ensure randomness
random.shuffle(image_files)

# Calculate the number of images for training, validation and testing
total_images = len(image_files)
train_count = int(total_images * 0.7)
val_count = int(total_images * 0.15)
test_count = total_images - train_count - val_count

# Move the files
for i, image_file in enumerate(image_files):
    mask_file = image_file.replace('.jpg', '_mask.gif')
    
    if i < train_count:
        # Move to training directories
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(train_images_dir, image_file))
        shutil.copy(os.path.join(masks_dir, mask_file), os.path.join(train_masks_dir, mask_file))
    elif i < train_count + val_count:
        # Move to validation directories
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(val_images_dir, image_file))
        shutil.copy(os.path.join(masks_dir, mask_file), os.path.join(val_masks_dir, mask_file))
    else:
        # Move to testing directories
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(test_images_dir, image_file))
        shutil.copy(os.path.join(masks_dir, mask_file), os.path.join(test_masks_dir, mask_file))
