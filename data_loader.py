#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Project: Data Science Programming (D7054E)

@author: Ziaul Islam Chowdhury

Created on Mar 16 2021

Luleå University of Technology
"""

import kaggle
from enum import Enum
from pathlib import Path
import zipfile
import time


class DataSet(Enum):
    ''' Enum to represent which dataset to download from Kaggle '''
    
    Flickr30K = ('hsankesara/flickr-image-dataset', 'results.csv', 'flickr-image-dataset.zip')
    Flickr8K = ('adityajn105/flickr8k', 'captions.txt', 'flickr8k.zip')


class KaggleDataLoader:
    ''' Class to download dataset from Kaggle ((Flickr30K, Flickr8K options)) and to perform required post-processing '''
    
    def __init__(self, dataset_type: DataSet, target_dir='/Users/ziaulchowdhury/Desktop', should_unzip=False):
        
        if type(dataset_type) != DataSet:
            raise TypeError(f'Invalid dataset: {dataset_type}! DataSet type must be either Flickr30K or Flickr8K!')
        
        self.captions_filename = 'captions.csv'
        self.images_dirname = 'images'
        
        self.dataset_type = dataset_type
        self.target_dir = target_dir
        self.should_unzip = should_unzip
        
        self.caption_filepath = self.target_dir + '/' + self.dataset_type.value[2].split('.')[0] + '/' + self.captions_filename
    
    def download_dataset_kaggle_api(self, already_downloaded=False):
        ''' Kaggle API token must be created from Kaggle account and stored under userhome/.kaggle directory '''
        
        if already_downloaded:
            return
        
        # Authentica to Kaggle API by using local credentials
        kaggle.api.authenticate()
        
        print(f'Started downloading {self.dataset_type.value[0]} dataset from Kaggle ...')
        start_time = time.time()
        
        # Perform dataset download
        kaggle.api.dataset_download_files(self.dataset_type.value[0], path=self.target_dir, unzip=self.should_unzip)
        
        elapsed_time = time.time() - start_time
        print(f'>>> Completed downloading  {self.dataset_type.value[0]} dataset from Kaggle. Time took: {elapsed_time} sec!')
        
        self.post_processing()
    
    def post_processing(self, already_processed=False):
        ''' Rename CSV file containing captions and image names. Flickr30K captions filename is: results.csv and Flickr30K captions filename is: captions.txt '''
        
        if already_processed:
            return
        
        zip_file_location = self.target_dir + '/' + self.dataset_type.value[2]
        zip_filepath = Path(zip_file_location)
        
        if self.should_unzip is False:
            extract_path = self.target_dir + '/' + zip_filepath.stem
            with zipfile.ZipFile(zip_filepath, 'r') as zfile:
                zfile.extractall(extract_path)
        
        # Make CSV file same name (captions.csv)
        if self.dataset_type == DataSet.Flickr30K:
            filepath = self.target_dir + '/' + zip_filepath.stem + '/flickr30k_images/' + self.dataset_type.value[1]
        else:
            filepath = self.target_dir + '/' + zip_filepath.stem + '/' + self.dataset_type.value[1]
            
        caption_file_path = Path(filepath)
        caption_file_path.rename(self.target_dir + '/' + zip_filepath.stem + '/' + self.captions_filename)
        
        # In Flickr30K, images diectory is named as flickr30k_images, it will be renamed as images
        if self.dataset_type == DataSet.Flickr30K:
            img_dirpath = Path(self.target_dir + '/' + zip_filepath.stem + '/flickr30k_images/flickr30k_images')
            img_dirpath.rename(self.target_dir + '/' + zip_filepath.stem + '/' + self.images_dirname)
            
            old_img_dirpath = Path(self.target_dir + '/' + zip_filepath.stem + '/flickr30k_images')
            old_img_dirpath.rmdir()
