
## This program should load the training and testing data from the ./data directory
## This data is already placed in the ./data directory by the data_splitting.py program
## subfolders 'train' and 'test' have subfolders for each class inside them
## /data/
##      /train/
##              /class1/    
##              /class2/
##      /test/
##              /class1/    
##              /class2/
##
##

import tensorflow as tf
import numpy as np
import os
def load_data():
    """
    Load the training and testing data from the ./data directory
    Normalization will not be performed. No pre-processing.
    """
    # Define class names

    class_names = []

    # Load class names from ./labels.txt
    with open("./labels.txt", "r") as f:
        class_names = f.read().splitlines()


    # Define data directories
    train_dir = "./data/train"
    test_dir  = "./data/test"
    val_dir  = "./data/valid"
    # Get the number of objects in each class. This is the number of images in each class
    # This is used to define the batch size for each class
    n_objects_per_class_train = [len(os.listdir(f"{train_dir}/{name}")) for name in class_names]
    n_objects_per_class_test  = [len(os.listdir(f"{test_dir}/{name}")) for name in class_names]
    n_objects_per_class_val  = [len(os.listdir(f"{val_dir}/{name}")) for name in class_names]

    # Define data loaders
    train_loader = tf.keras.preprocessing.image.ImageDataGenerator()
    test_loader  = tf.keras.preprocessing.image.ImageDataGenerator()
    val_loader  = tf.keras.preprocessing.image.ImageDataGenerator()

    # Load data
    train_data = train_loader.flow_from_directory(train_dir, target_size=(64, 64), batch_size=int(sum(n_objects_per_class_train)), class_mode='categorical')
    test_data  = test_loader.flow_from_directory(test_dir, target_size=(64, 64), batch_size=int(sum(n_objects_per_class_test)), class_mode='categorical')
    val_data  = val_loader.flow_from_directory(val_dir, target_size=(64, 64), batch_size=int(sum(n_objects_per_class_val)), class_mode='categorical')
    train_images = train_data[0][0]
    train_labels = train_data[0][1]
    test_images = test_data[0][0]
    test_labels = test_data[0][1]
    val_images = val_data[0][0]
    val_labels = val_data[0][1]

    # Train and test labels are current one-hot encoded
    # Convert to list of integer labels
    train_labels= np.array([[x.argmax()] for x in train_labels])
    test_labels=np.array([[x.argmax()] for x in test_labels])
    val_labels=np.array([[x.argmax()] for x in val_labels])
 

    return (train_images, train_labels), (test_images, test_labels), (val_images, val_labels)


if __name__ == "__main__":
    load_data()




