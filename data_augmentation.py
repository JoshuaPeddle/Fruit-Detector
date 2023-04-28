from keras import  layers
import keras as keras

def get_data_augmentation():
    data_augmentation = keras.Sequential(
    [
        #layers.RandomFlip("horizontal_and_vertical"),  
        #layers.RandomRotation(0.2),
        layers.RandomContrast(0.2),
    ]
    )
    return data_augmentation