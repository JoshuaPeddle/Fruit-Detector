from keras import layers, models
import keras as keras
import tensorflow as tf


def get_model(hp, data_augmentation, img_height, img_width, class_names):
    model = keras.Sequential()
    model.add(layers.Resizing(img_height, img_width))
    model.add(data_augmentation)
    if hp:
        model.add(layers.Conv2D(hp.Int("units", min_value=32, max_value=64, step=32), (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    else:
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.15))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(len(class_names), activation='relu'))

    model.compile(keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=True),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy', 'mse', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2")]) # top 3 accuracy
    return model
