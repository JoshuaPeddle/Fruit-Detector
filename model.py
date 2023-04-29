from keras import layers, models
import keras as keras
import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization



def get_model(hp, data_augmentation, img_height, img_width, class_names):
    model = keras.Sequential()
    model.add(layers.Resizing(img_height, img_width))
    model.add(data_augmentation)

    if hp: model.add(layers.Conv2D(hp.Choice('conv1_0', [64]), (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    else:  model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    if hp: model.add(layers.Conv2D(hp.Choice('conv1_1', [64]), (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    else:  model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    if hp: model.add(layers.Conv2D(hp.Choice('conv2', [256]), (3, 3), activation=hp.Choice('act2_0', ['relu']), input_shape=(img_height, img_width, 3)))
    else: model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    if hp: model.add(layers.Conv2D(hp.Choice('conv2', [256]), (3, 3), activation=hp.Choice('act2_1', ['relu']), input_shape=(img_height, img_width, 3)))
    else: model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    if hp: model.add(layers.Dropout(hp.Choice('sparce_dropout', [0.1, 0.15, 0.25])))
    else: model.add(layers.Dropout(0.15))

    if hp: model.add(layers.Conv2D(hp.Choice('conv3_0', [512]), (3, 3), activation='relu'))
    else: model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    if hp: model.add(layers.Conv2D(hp.Choice('conv3_1', [512]), (3, 3), activation='relu'))
    else: model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    
    if hp: model.add(layers.Dense(hp.Choice('dense1', [1024]), activation='relu'))
    else: model.add(layers.Dense(256, activation='relu'))
    if hp: model.add(layers.Dense(hp.Choice('dense2', [1024]), activation='relu'))
    else: model.add(layers.Dense(256, activation='relu'))
    model.add(BatchNormalization())
    if hp: model.add(layers.Dropout(hp.Choice('dense_dropout', [0.1, 0.17, 0.25])))
    else: model.add(layers.Dropout(0.25))

    if hp: model.add(layers.Dense(len(class_names), activation=hp.Choice('finalact', ['relu'])))
    else: model.add(layers.Dense(len(class_names), activation='softmax'))
    

    if hp:
        model.compile(keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [0.001,0.0005,0.0001]), beta_1=0.9, beta_2=0.999, amsgrad=True),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy', 'mse', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2")]) # top 3 accuracy
    else:
        model.compile(keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=True),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy', 'mse', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2")]) # top 3 accuracy
    return model


def get_model2(hp, data_augmentation, img_height, img_width, class_names):
    model = keras.Sequential()
    model.add(layers.Resizing(img_height, img_width))
    model.add(data_augmentation)

    if hp: model.add(layers.Conv2D(hp.Choice('conv1', [64]), (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    else:  model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))

    if hp: model.add(layers.MaxPooling2D(pool_size=hp.Choice('polling1', [ 2])))
    else: model.add(layers.MaxPooling2D((2, 2)))

    if hp: model.add(layers.Conv2D(hp.Choice('conv2', [64, 128]), (3, 3), activation=hp.Choice('act2', ['relu']), input_shape=(img_height, img_width, 3)))
    else: model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    if hp: model.add(layers.Dropout(hp.Choice('sparce_dropout', [0.1, 0.15, 0.25])))
    else: model.add(layers.Dropout(0.15))

    if hp: model.add(layers.Conv2D(hp.Choice('conv3', [64, 128, 256, 512]), (3, 3), activation='relu'))
    else: model.add(layers.Conv2D(512, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    
    if hp: model.add(layers.Dense(hp.Choice('dense', [32, 64, 128, 256]), activation='relu'))
    else: model.add(layers.Dense(128, activation='relu'))
    model.add(BatchNormalization())
    if hp: model.add(layers.Dropout(hp.Choice('dense_dropout', [0.1, 0.17, 0.25])))
    else: model.add(layers.Dropout(0.25))

    if hp: model.add(layers.Dense(len(class_names), activation=hp.Choice('finalact', ['relu'])))
    else: model.add(layers.Dense(len(class_names), activation='softmax'))
    

    if hp:
        model.compile(keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [0.001,0.0005,0.0001]), beta_1=0.9, beta_2=0.999, amsgrad=True),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy', 'mse', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2")]) # top 3 accuracy
    else:
        model.compile(keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=True),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy', 'mse', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2")]) # top 3 accuracy
    return model
