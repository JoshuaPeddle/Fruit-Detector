'''

This code is a data preprocessing script that is used for creating a dataset to train, validate and test an image classification model. Specifically, it creates a dataset of fruit images, including Apples, Bananas, Grapes, Mangos, and Strawberries.

The script first defines the class names and number of images per class by counting the number of files in the directory for each fruit. It then creates three directories, one for each dataset split, i.e., training, validation, and testing. These directories are created if they don't exist.

The script then creates subdirectories in each of the three main directories for each fruit class. This step creates a hierarchical structure for the dataset, where each fruit class has a folder in each dataset split directory.

Next, the script collects all the image paths for each fruit class, storing them in a list. It then shuffles the paths randomly to ensure that the data is not biased in any way.

The script then calculates the size of each dataset split based on the total number of images and predefined ratios. The training dataset is the largest, comprising 97% of the total data, while the validation and testing datasets comprise 2% and 1%, respectively.

The script then creates three separate lists, one for each dataset split, containing tuples with the old and new paths for each image. The new paths are generated by concatenating the class folder, the image filename, and the corresponding dataset split directory.

Finally, the script moves each image from its old path to its new path in the appropriate dataset split directory using the os.rename() function. It then removes the old directories for each fruit class, as they are no longer needed.

The script outputs the total size of the data, as well as the size of each dataset split. Once the script finishes running, the dataset is ready to use for training, validating, and testing an image classification model.

'''


import shutil
import os
import numpy as np
from glob import glob
from tqdm import tqdm


# Copy images out of data/src into this directory, /data
# Create a directory for each class and put images in each class directory
# Run this script to create train, valid, and test directories
print("Copying images from src to data directory...")
files = os.listdir("./src")
shutil.copytree("./src", "./",dirs_exist_ok=True)

# Define class names and number of images per class
class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
n_images_per_class = len(os.listdir(f"./{class_names[0]}"))

# Define train, valid, and test directories and create them if they don't exist
train_dir = "./train"
valid_dir = "./valid"
test_dir  = "./test"

for directory in [train_dir, valid_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)

# Create subdirectories for each class in train, valid, and test directories
for name in class_names:
    for directory in [train_dir, valid_dir, test_dir]:
        class_path = os.path.join(directory, name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)

# Collect all image paths for each class
all_class_paths = [glob(f"./{name}/*") for name in class_names]

# Define training, validation, and testing size
total_size = sum([len(paths) for paths in all_class_paths])

train_ratio = 0.95
valid_ratio = 0.00
test_ratio  = 0.05

train_size = int(total_size * train_ratio)
valid_size = int(total_size * valid_ratio)
test_size  = int(total_size * test_ratio)

train_images_per_class = int(n_images_per_class * train_ratio)
valid_images_per_class = int(n_images_per_class * valid_ratio)
test_images_per_class  = int(n_images_per_class * test_ratio)

print("Total Data Size  :   {}".format(total_size))
print("Training Size    :   {}".format(train_size))
print("Validation Size  :   {}".format(valid_size))
print("Testing Size     :   {}\n".format(test_size))

# Shuffle image paths for each class
for paths in all_class_paths:
    np.random.shuffle(paths)

# Define lists of (old_path, new_path) tuples for training, validation, and testing images
train_images = [(path, os.path.join(train_dir, path.split('/')[-2], path.split('/')[-1])) for paths in all_class_paths for path in paths[:train_images_per_class]]
valid_images = [(path, os.path.join(valid_dir, path.split('/')[-2], path.split('/')[-1])) for paths in all_class_paths for path in paths[train_images_per_class: train_images_per_class + valid_images_per_class]]
test_images  = [(path, os.path.join(test_dir, path.split('/')[-2], path.split('/')[-1]))  for paths in all_class_paths for path in paths[train_images_per_class+valid_images_per_class: train_images_per_class + valid_images_per_class + test_images_per_class]]

# Move images to their new directories
for images, data_type in [(train_images, "Training"), (valid_images, "Validation"), (test_images, "Testing")]:
    for (old_path, new_path) in tqdm(images, desc=data_type + " Data"):
        os.rename(old_path, new_path)

# Remove the old directories
for directory in class_names:
    os.rmdir("./" + directory)

# Print confirmation message
print("ALL DONE!!")
