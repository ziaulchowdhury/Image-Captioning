#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Project: Data Science Programming (D7054E)

@author: Ziaul Islam Chowdhury

Created on Mar 16 2021

Luleå University of Technology
"""

import time
from os import listdir
from enum import Enum
from pathlib import Path
from collections import OrderedDict
import pickle

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model


class CnnModelType(Enum):
    ''' Supported CNN model type. Currently only VGG16 is supported. '''
    
    VGG16 = {'target_size': (224, 224)}


class CnnEncoder:
    ''' Class contains VGG16 CNN model to encode images into features '''
   
    def __init__(self, cnn_model_type: CnnModelType):
        ''' Constructor '''
        if type(cnn_model_type) != CnnModelType:
            raise TypeError(f'Invalid CNN model type: {cnn_model_type}! CnnModelType must be VGG16!')
        
        self.cnn_model_type = cnn_model_type

    def encode_features_directory(self, directory):
        '''  Extract features from the pretrained VGG16 CNN network '''
        
        print('Started creating image features by using VGG16 pretrained model ........')
        start_time = time.time()
        
        if self.cnn_model_type == CnnModelType.VGG16:
            model = VGG16()
        
        # Output layer is not required as we only need the weights of the last FC layer of the CNN
        model.layers.pop()
        
        self.cnn_encoder = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        print(f'Summary of CNN encoder model: {self.cnn_encoder.summary()}')
        
        self.image_features = OrderedDict()
        
        # Iterate through the directory, load the image, extract features (weights) from the CNN model and store in a dict
        img_num = 0
        for name in listdir(directory):
            
            filename = directory + '/' + name
            if '.DS_Store' in filename or Path(filename).is_dir():
                continue
            
            image_id = name.split('.')[0]  # Get ID of the image
            
            try:
                image_feature = self.encode_features_image(filename)
            
                # Store the feature to the dictionary
                self.image_features[image_id] = image_feature
            except Exception as e:
                print(f'Error occured while encoding features from image with id: {name}')
                print(e)
                continue
            
            img_num += 1
            print(f'{img_num} Image-ID: {image_id}')
        
        elapsed_time = time.time() - start_time
        print(f'>>> Completed creation of features of {len(self.image_features)} images. Time taken: {elapsed_time} sec!')
        
    def encode_features_image(self, filepath):
    
        image = self.preprocess_image(filepath)
            
        image_feature = self.cnn_encoder.predict(image, verbose=0)
        
        # NOTE: Flatteing of image feature is not required (image_feature.flatten())
        return image_feature
        
    def preprocess_image(self, filename):
        ''' Resizes and preprocesses the image '''
        
        image = load_img(filename, target_size=self.cnn_model_type.value['target_size'])
            
        image_arr = img_to_array(image)
        
        resized_image = image_arr.reshape((1, image_arr.shape[0], image_arr.shape[1], image_arr.shape[2]))
        
        return preprocess_input(resized_image)
    
    def dump_image_features(self, flickr30k=True):
        ''' Save the extracted image features to a pkl file '''
        
        filename = f"image_features_{'flickr30k' if flickr30k else 'flickr8k'}.pkl"
        
        print(f'Started dumping image features to {filename} ........')
        start_time = time.time()
        
        with open(filename, 'wb') as handle:
            pickle.dump(self.image_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        elapsed_time = time.time() - start_time
        print(f'>>> Completed dumping image features to {filename}. Time taken: {elapsed_time} sec!')
        
    def load_image_features_from_file(self, filename):
        ''' Loads image features from a file to self.image_features '''
        
        print(f'Started loading image features from {filename} ........')
        start_time = time.time()
        
        with open(filename, 'rb') as handle:
            self.image_features = pickle.load(handle)
        
        elapsed_time = time.time() - start_time
        print(f'>>> Completed loading image features from {filename}. Time taken: {elapsed_time} sec!')
