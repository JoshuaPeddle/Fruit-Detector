from keras import  layers
import keras as keras

def get_data_augmentation():
    data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),  
        layers.RandomRotation(0.45),
        layers.RandomContrast(0.5),
    ]
    )
    return data_augmentation