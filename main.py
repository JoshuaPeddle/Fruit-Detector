import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import matplotlib.pyplot as plt
from data import load_data
import keras as keras
import numpy as np

from model import get_model
from data_augmentation import get_data_augmentation


epochs  = 50       # How many epochs to train for
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



if PLOT:
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])

    plt.show()

#                                           AUGMENTATION
data_augmentation = get_data_augmentation()

if PLOT:
    plt.figure(figsize=(10, 10))
    for i in range(9):
        augmented_image = data_augmentation(train_images[0:2])
        plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[0].numpy())
        plt.axis("off")
    plt.show()

#                                           CONFIGURE MODEL

callback = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=5)

model = get_model(False, data_augmentation, img_height, img_width, class_names)


#                                          TRAIN MODEL  

history = model.fit(train_images, train_labels, epochs=epochs, 
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


##                                          SAVE TFLITE MODEL
TF_MODEL_FILE_PATH = 'model/fruit.tflite'

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
#converter.inference_input_type = tf.uint8

tflite_model = converter.convert()

# Save the model.
with open(TF_MODEL_FILE_PATH, 'wb') as f:
  f.write(tflite_model)



# The default path to the saved TensorFlow Lite model

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

print(interpreter.get_signature_list())

classify_lite = interpreter.get_signature_runner('serving_default')
classify_lite
n= 0
try:
  while True:
    predictions_lite = classify_lite(resizing_input=val_images[n:n+1])['dense_1']
    print(predictions_lite)
    score_lite = tf.nn.softmax(predictions_lite)
    print (score_lite)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
    )
    print("Should have been " , val_labels[n], class_names[val_labels[n][0]])
    plt.imshow(val_images[n])
    plt.show()
    n+=1
except KeyboardInterrupt:
  print('interrupted!')