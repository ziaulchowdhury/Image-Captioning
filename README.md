## Image Captioning with Deep Learning

## Introduction
This is the final project of Data Science Programming ((D7054E)) course. As the topic of the project, an image captioning model is implemented by Ziaul Islam Chowdhury. The project uses Tensorflow Keras framework for creating deep neural network model to accomplish the task.

The implementation supports two datasets (Flickr8K and Flickr30K). It downloads datasets automatically from Kaggle if Kaggle API token is configured on the local machine.

## Prerequisites

A set of prerequisites shown in the following list must be fulfilled in order to run the model on local machine:
1. **Pandas**: Package for data analysis and manipulation 
2. **Kaggle**: Package for using Kaggle public API
3. **Tensorflow Keras 2**: Deep learning framework for creating models
4. **Matplotlib**: Package for plottting visualizations
5. **Numpy**: Package that supports high level functions to work with large arrays and matrices
6. **NLTK**: A suite of libraries for NLP tasks
7. **Seaborn**: Data visualization linbrary based on Matplotlib
8. **Scipy**: Package for statistical analysis

## How to configure Kaggle API token
Flickr8K and Flickr30K datasets are downloaded by using Kaggle public API which needs API token. An user must login to the Kaggle account and generate API token. The generated token must be stored in kaggle.json file inside of the *.kaggle* directory. In a MacOS, .kaggle directorry is created under user's home directory (example: *~/.kaggle*).

Please check here (https://www.kaggle.com/docs/api) for the details.

## How to test the image captioning model

1. Install required packages and frameworks
2. Please open *./test_image_captioning.py* file and adjust the  following parameters:
    1. **run_flickr_30k**: Set true to train, validate and test with Flickr30K dataset. Please note that executing Flickr30K dataset required more than 36 hours of processing time with a machine with 16 GB RAM, Intel i7 quad-core processor. Set value of run_flickr_30k = False to enable Flickr8K dataset.
    2. **num_epochs**: Adjust the number of iterations to train the model. Higher epochs requires longer processing time.
    3. **target_dir**: Directory where the dataset will be download and extracted. Example: '/Users/ziaulchowdhury/Desktop'
    4. **download_dataset**: Set to True to download dataset and perform post-processing on the unzipped directory.
