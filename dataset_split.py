import os
import random
import shutil

# Define the paths to your dataset directories
dataset_dir = './new_dataset2/data'  # Replace with the actual path
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
val_dir = os.path.join(dataset_dir, 'val')  # Create a validation directory

# Create train, test, and validation directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Define the ratios for training, testing, and validation (e.g., 70% train, 20% test, 10% validation)
train_ratio = 0.7
test_ratio = 0.1
val_ratio = 0.2

# Get the list of image files in your dataset
image_files = os.listdir(images_dir)

# Shuffle the image files to randomize the split
random.shuffle(image_files)


# Calculate the number of images for each split
num_train = int(len(image_files) * train_ratio)
num_test = int(len(image_files) * test_ratio)
num_val = len(image_files) - num_train - num_test

# Split the dataset into training, testing, and validation sets
train_images = image_files[:num_train]
test_images = image_files[num_train:num_train + num_test]
val_images = image_files[num_train + num_test:]

# Move image and label files to the appropriate split directories
def move_files(image_files, dest_dir):
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        if image_file.endswith(".jpg"):
            label_path = os.path.join(labels_dir, image_file.replace('.jpg', '.txt'))
        elif image_file.endswith(".png"):
            label_path = os.path.join(labels_dir, image_file.replace('.png', '.txt'))
        print(label_path)
        # Move image and label files to the destination directory
        shutil.copy(image_path, os.path.join(dest_dir, image_file))
        if image_file.endswith(".jpg"):
            shutil.copy(label_path, os.path.join(dest_dir, image_file.replace('.jpg', '.txt')))
        elif image_file.endswith(".png"):
            shutil.copy(label_path, os.path.join(dest_dir, image_file.replace('.png', '.txt')))

move_files(train_images, train_dir)
move_files(test_images, test_dir)
move_files(val_images, val_dir)

# Create a class mapping from classes.txt
class_mapping = {}
with open(os.path.join(dataset_dir, 'classes.txt'), 'r') as class_file:
    classes = class_file.read().strip().split('\n')
    for i, class_name in enumerate(classes):
        class_mapping[i] = class_name

# Print the class mapping
print(class_mapping)
