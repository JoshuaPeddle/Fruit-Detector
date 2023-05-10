from keras import  layers
import keras as keras

def get_data_augmentation():
    data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),  
        layers.RandomZoom(height_factor=(0,1), fill_mode = "constant", fill_value = 1),
        layers.RandomRotation(0.45, fill_mode = "constant", fill_value = 1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness((-0.2,0.2), value_range=(0, 1.0)),
    ]
    )
    return data_augmentation