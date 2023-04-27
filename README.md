## Introduction

This repo serves as the home of my classification project. 

The current list of classified objects can be found [here](labels.txt).

## Features 

1. Create train/test/validation sets using [this script](/data/data_splitting.py).

2. Train a Tensorflow classifier based on the split data. 

3. Convert the model to a TensorFlow Lite compatible tflite model.

4. Add metadata to the tflite model using the
[metadata writer](metadata_writer_for_image_classifier.py)


## Instructions 


### 1. Split the data

Images are provided in the /data/src directory. They are in sub-folders named according to their class name.

So the first step is to create train, test and validation sets for training. 
Go to /data directory and run the following command. Source images are preserved so the script can be ran as many times as needed to find a good split.

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

Next train the model using the data from the previous step.
Go to the root directory and run the following command.

```python main.py```


Hyperparameters can be adjusted in [main.py](main.py).

### 3. Add metadata to the model

"TensorFlow Lite metadata provides a standard for model descriptions. The metadata is an important source of knowledge about what the model does and its input / output information." [src](https://www.tensorflow.org/lite/models/convert/metadata)

To add metadata to the model run the following command. 

```
python ./metadata_writer_for_image_classifier.py --model_file=model/fruit.tflite --label_file=./labels.txt --export_directory=model_with_metadata
```

Input image height and width must be set to the expected input size of images that will be classified using the model.

```
_MODEL_INFO = {  
    "fruit.tflite":
        ModelSpecificInfo( 
            ...        
            image_width=64,
            image_height=64,
            ...
}
```



## Sources
https://www.kaggle.com/datasets/chrisfilo/fruit-recognition
https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset

## Upcomming
Implement hyperband for hyperparameter tuning
https://www.tensorflow.org/tutorials/keras/keras_tuner