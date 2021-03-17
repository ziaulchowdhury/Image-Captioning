#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Project: Data Science Programming (D7054E)

@author: Ziaul Islam Chowdhury

Created on Mar 16 2021

Luleå University of Technology
"""

import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from data_loader import DataSet, KaggleDataLoader


class CaptionProcessor:
    ''' Caption processor class performs preprocessing of captions of the images '''
    
    def __init__(self, kaggle_dataloader: KaggleDataLoader):
        
        self.lemmatizer = WordNetLemmatizer()
        
        self.kaggle_dataloader = kaggle_dataloader
        
        self.read_csv_preprocess_captions()
        
        self.generate_image_caption_map()
        self.generate_rnn_image_caption_map()
    
        self.generate_vocabluary()
        
    def read_csv_preprocess_captions(self):
        ''' Reads captions.csv file to pandas dataframe, calls preprocess_captions function to preprocess captions.
        It removed unnecessary column (comment_number) when the dataset is Flickr30K '''
        
        if self.kaggle_dataloader.dataset_type == DataSet.Flickr30K:
            col_seperator = '|'
        else:
            col_seperator = ','
        
        self.image_captions_df = pd.read_csv(self.kaggle_dataloader.caption_filepath, sep=col_seperator)
        print(f'Columns of DF: {self.image_captions_df.columns}')
        
        # Remove unused column present in Flickr30K DataSet
        if self.kaggle_dataloader.dataset_type == DataSet.Flickr30K:
            self.image_captions_df.drop(columns=[' comment_number'], inplace=True, axis=1)
        
        # Make same column names in both datasets
        self.image_captions_df.columns = ['image_id', 'caption']
        
        # Sort rows by image_id so that captions of the same image are next to each other
        self.image_captions_df.sort_values(by=['image_id'], ascending=True, inplace=True)
        
        # Preprocess captions
        self.image_captions_df['caption'] = self.image_captions_df['caption'].apply(lambda caption: self.preprocess_captions(str(caption)))
        
        # Remove file extension from image_id
        self.image_captions_df['image_id'] = self.image_captions_df['image_id'].apply(lambda img_id: img_id.split('.')[0])
        
    def generate_image_caption_map(self):
        ''' Creates a map of image and it's captions in a list (one image may contain multiple captions) '''
        
        image_caption_map = {}
        for index, row in self.image_captions_df.iterrows():
            image_id = row['image_id']
            caption = row['caption']
            
            if image_id not in image_caption_map:
                image_caption_map[image_id] = []
            image_caption_map[image_id].append(caption)
            
        self.image_caption_map = image_caption_map
    
    def generate_rnn_image_caption_map(self):
        ''' Creates a map of image and it's captions in a list for RNN (contains special tags such as starttag and endtag to specify start and end of a caption '''
        
        rnn_image_caption_map = {}
         
        for image_id, captions in self.image_caption_map.items():
            rnn_image_caption_map[image_id] = []
            for caption in captions:
                
                # TODO: I didn't like to split tokens with space, hence following is commented and alternative approach is used
                # rnn_image_caption_map[image_id].append('startseq '+ ' '.join(caption)+ ' endseq')
                
                # caption_with_start_end = f'<start> {caption} <end>'
                caption_with_start_end = f'startseq {caption} endseq'
                rnn_image_caption_map[image_id].append(caption_with_start_end)
        
        self.rnn_image_caption_map = rnn_image_caption_map
    
    def preprocess_captions(self, caption):
        ''' NLP preprocessing pipeline for the captions '''
        
        pattern = r'[^\w\s]'
        caption = re.sub(pattern, '', caption)  # Exclude punctuations from caption
        tokens = caption.split()  # Tokenize captions
        tokens = [token.lower() for token in tokens]  # Convert tokens to lowercase
        
        # Perform lemmatization on tomens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Exclude numerical values
        tokens = [token for token in tokens if token.isalpha()]
        
        # Create caption by joining cleaned tokens
        caption = ' '.join(tokens)
        
        return caption
    
    def generate_vocabluary(self):
        ''' Generates vocubulary count map for each token out of image_caption_map '''
        
        # Store only tokens in the vocabulary set
        caption_vocabulary = set()
        for image_id in self.image_caption_map.keys():
            [caption_vocabulary.update(caption.split()) for caption in self.image_caption_map[image_id]]
        self.caption_vocabulary = caption_vocabulary
        
        # Only token count map
        caption_vocabulary_count_map = {}
        for image_id in self.image_caption_map.keys():
            for caption in self.image_caption_map[image_id]:
                for token in caption.split():
                    if token not in caption_vocabulary_count_map:
                        caption_vocabulary_count_map[token] = 0
                    caption_vocabulary_count_map[token] = caption_vocabulary_count_map[token] + 1
        
        # Sort by count in descending order
        caption_vocabulary_count_map = sorted(caption_vocabulary_count_map.items(), key=lambda t: t[1], reverse=True)
        self.caption_vocabulary_count_map = caption_vocabulary_count_map
    
    def save_image_captions(self, target_filepath):
        ''' Saves image captions to a file '''
        
        # Combining all captions of the images as list
        file_content_lines = []
        for image_id, captions in self.image_caption_map.items():
            for caption in captions:
                file_content_lines.append(image_id + ' ' + caption)
        
        # Add newline character to each item of the list and write to target file
        file_content = '\n'.join(file_content_lines)
        with open(target_filepath, 'w') as file:
            file.write(file_content)
