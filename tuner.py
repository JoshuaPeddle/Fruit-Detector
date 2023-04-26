from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from data import load_data
import keras_tuner

EPOCHS = 2

#                                           LOAD IMAGES
(train_images, train_labels), (test_images, test_labels), (val_images, val_labels) = load_data()
print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

img_height = 64
img_width = 64

# Normalize pixel values to be between 0 and 1
train_images, test_images, val_images = train_images / 255.0, test_images / 255.0, val_images / 255.0

class_names = []
# Load class names from ./labels.txt
with open("./labels.txt", "r") as f:
      class_names = f.read().splitlines()



data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal"),        
    layers.RandomRotation(0.1),
    layers.RandomContrast(0.1)
  ]
)


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Resizing(img_height, img_width))
    model.add(data_augmentation)
    model.add(layers.Conv2D(hp.Int("units", min_value=32, max_value=64, step=32), (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.15))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(len(class_names)))

    model.compile(keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=True),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy', 'mse', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2")]) # top 3 accuracy

    return model

build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective=("val_accuracy"),
    max_trials=3,
    executions_per_trial=1,
    overwrite=True,
    directory="results",
    project_name="fruits",
)
print(tuner.search_space_summary())

tuner.search(train_images, train_labels, epochs=EPOCHS, 
                    validation_data=(test_images, test_labels))

# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
best_model = models[0]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
best_model.build(input_shape=(None, 64, 64))
best_model.summary()

tuner.results_summary()