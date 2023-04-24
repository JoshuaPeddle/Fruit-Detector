import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from data import load_data
import tensorflow.keras as keras
import numpy as np

epochs  = 100
PLOT = True

#                                           LOAD IMAGES
(train_images, train_labels), (test_images, test_labels) = load_data()
print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

img_height = train_images.shape[1]
img_width = train_images.shape[2]

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
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
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
if PLOT:
    plt.figure(figsize=(10, 10))

    for i in range(9):
        augmented_image = data_augmentation(train_images[0:2])
        plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[0].numpy())
        plt.axis("off")

    plt.show()

#augmented_train_images = data_augmentation(train_images)
#augmented_test_images = data_augmentation(test_images)

#train_images = np.concatenate((train_images, augmented_train_images))
#train_labels = np.concatenate((train_labels, train_labels))
#test_images = np.concatenate((test_images, augmented_test_images))
#test_labels = np.concatenate((test_labels, test_labels))




#                                           CONFIGURE MODEL
model = models.Sequential()
model.add(data_augmentation)
model.add(layers.Conv2D(img_width, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.2)),
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(len(class_names)))
model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=epochs, 
                    validation_data=(test_images, test_labels))

#                                           HANDLE RESULTS
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
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
TF_MODEL_FILE_PATH = 'model.tflite'
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(TF_MODEL_FILE_PATH, 'wb') as f:
  f.write(tflite_model)



 # The default path to the saved TensorFlow Lite model

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

print(interpreter.get_signature_list())

classify_lite = interpreter.get_signature_runner('serving_default')
classify_lite

predictions_lite = classify_lite(sequential_input=test_images[0:1])['dense_1']
score_lite = tf.nn.softmax(predictions_lite)
print (score_lite)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)
plt.imshow(train_images[0])
plt.show()