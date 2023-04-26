from keras import  layers
import keras as keras

def get_data_augmentation():
    data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),        
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.1)
    ]
    )
    return data_augmentation