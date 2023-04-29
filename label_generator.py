# This script generates a file names labels.txt that contains the class names
# for the images in the data/src directory. The labels.txt file is used by
# other scripts to train the model.


TARGET_DIR = 'data/src'  # The location of the class directories.
LABELS_FILE = 'labels.txt'  # The name of the file to write the labels to.
 
import os
# Get the class names from the directory names in the target directory.
class_names = os.listdir(TARGET_DIR)
class_names.sort()

# Write the class names to the labels file.
with open(LABELS_FILE, 'w') as f:
    for class_name in class_names:
        f.write(class_name + '\n')