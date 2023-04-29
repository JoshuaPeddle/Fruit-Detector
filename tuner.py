from data import load_data
import keras_tuner
import shutil
import matplotlib.pyplot as plt
from data_augmentation import get_data_augmentation
import tensorflow as tf
from model import get_model


tune_epochs  = 50       # How many epochs to train for
train_epochs = 50
PLOT = True       # Whether to plot 

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


data_augmentation = get_data_augmentation()

def build_model(hp):
    model = get_model(hp, data_augmentation, img_height, img_width, class_names)
    return model

build_model(keras_tuner.HyperParameters())
#shutil.rmtree('results', ignore_errors=True)
# tuner = keras_tuner.RandomSearch(
#     hypermodel=build_model,
#     objective=("val_accuracy"),
#     max_trials=50,
#     executions_per_trial=2,
#     overwrite=False,
#     directory="results",
#     project_name="fruits",

# )


tuner = keras_tuner.Hyperband( # https://keras.io/api/keras_tuner/tuners/hyperband/
    hypermodel=build_model,
    objective=("val_accuracy"),
    max_epochs=500,
    factor=3,
    hyperband_iterations=50,
    seed=None,
    overwrite=True,
    hyperparameters=None,
    tune_new_entries=True,
    allow_new_entries=True,
    max_retries_per_trial=1,
    max_consecutive_failed_trials=3,
    directory="results",
    project_name="fruits",
)


print(tuner.search_space_summary())
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4)
tuner.search(train_images, train_labels, epochs=tune_epochs, 
                    validation_data=(test_images, test_labels), callbacks=[callback])




# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
best_model = models[0]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
best_model.build(input_shape=(None, 64, 64, 3))
best_model.summary()

tuner.results_summary()


# Get the top 2 hyperparameters.
best_hps = tuner.get_best_hyperparameters(5)
# Build the model with the best hp.
model = build_model(best_hps[0])
# Fit with the entire dataset.
callback = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=5)
history = model.fit(train_images, train_labels, epochs=train_epochs, 
                    validation_data=(test_images, test_labels), callbacks=[callback])

#                                           HANDLE RESULTS
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


epochs_range = range(len(history.history['loss']))

if PLOT:
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()