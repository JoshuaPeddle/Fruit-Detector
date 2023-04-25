
## How to use


### Split the data
First you have to split the data
go to /data directory and run the following command

```python data_splitting.py```

### Training
Next train the model
go back to the root directory and run the following command

```python main.py```

### Add metadata to the model
Finally add metadata to the model

```python ./metadata_writer_for_image_classifier.py --model_file=fruit.tflite --label_file=./labels.txt --export_directory=model_with_metadata```


## Sources
https://www.kaggle.com/datasets/chrisfilo/fruit-recognition
https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset