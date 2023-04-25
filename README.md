

## Introduction

This repo serves as the home of my classification project. 

The current list of classified objects can be found [here](labels.txt).

## Features 

1. Create train/test/validation sets using [this script](/data/data_splitting.py).

2. Train a Tensorflow classifier based on the split data. Hyperparameters can be adjusted in (main.py)[main.py].

3. Convert the model to a TensorFlow Lite compatible tflite model.

4. Add metadata to the tflite model using the
(metadata writer)[metadata_writer_for_image_classifier.py]


## Instructions 


### 1. Split the data

First you have to split the data
go to /data directory and run the following command.

```python data_splitting.py```

Custom ratios can be chosen. The current selection can be found in the above file.

For example:
```
train_ratio = 0.75
valid_ratio = 0.01
test_ratio  = 0.24
``` 
would have a 75%/1%/24% split.

### 2. Training
Next train the model
go back to the root directory and run the following command

```python main.py```

### 3. Add metadata to the model
Finally add metadata to the model

```python ./metadata_writer_for_image_classifier.py --model_file=fruit.tflite --label_file=./labels.txt --export_directory=model_with_metadata```



## Sources
https://www.kaggle.com/datasets/chrisfilo/fruit-recognition
https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset