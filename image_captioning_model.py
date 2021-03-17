#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Project: Data Science Programming (D7054E)

@author: Ziaul Islam Chowdhury

Created on Mar 16 2021

Luleå University of Technology
"""

import random
import time
import numpy as np
import re
from numpy import argmax
from nltk.translate.bleu_score import corpus_bleu

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from data_preprocessing import CaptionProcessor
from cnn_encoder import CnnEncoder
from data_loader import DataSet


class ImageCaptioningModel:
    '''ImageCaptioningModel Class takes CaptionProcessor, CnnEncoder and few other parameters while creation of the instance.
      It then trains the captions by using RNN architecture (LSTM) and generates caption for the image.'''
    
    def __init__(self, caption_processor: CaptionProcessor, cnn_encoder: CnnEncoder, dropout: float = 0.1,
                 epochs: int = 1, evaluate_bleu_score: bool = False):
        ''' Mandatory parameters are CaptionProcessor, CnnEncoder. Other parameters take default value if not given. '''
        
        self.dropout = dropout
        self.caption_processor = caption_processor
        self.cnn_encoder = cnn_encoder
        
        self.train_image_ids, self.validation_image_ids, self.test_image_ids = self.split_train_val_test_image_ids()
        
        self.tokenizer = self.create_fit_tokenizer(self.train_image_ids)
        
        self.vocabulary_size = len(self.tokenizer.word_index) + 1
        
        self.sequence_max_length = self.calculate_max_sequence_length(self.train_image_ids)
        
        # Create image captioning model
        self.image_captioning_model = self.create_image_captioning_model(img_feature_shape=(1000, ))
        
        # Train image captioning model
        if caption_processor.kaggle_dataloader.dataset_type == DataSet.Flickr8K:
            self.history = self.train_image_captioning_model(epochs)
            self.plot_model_performance()
        else:
            self.train_image_captioning_model_progressive_loading(epochs)
        
        # Evaluate bleu score by using test dataset
        if evaluate_bleu_score:
            self.evaluate_image_captioning_model()
    
    def split_train_val_test_image_ids(self, should_shuffle=True, train_percentage=0.7, validation_percentage=0.15,
                                       test_percentage=0.15):
        ''' Splits the image_ids into train, validation and test image_ids. If no value given, it takes 70% image_ids
        for training, 15% for validation and 15% for testing. '''
        
        # Make a list of image ids from the rnn_image_caption_map (map of image_id and its captions)
        image_ids = list(self.caption_processor.rnn_image_caption_map.keys())
        
        if should_shuffle:
            random.shuffle(image_ids)
        
        dataset_length = len(image_ids)
        train_end_index = round(dataset_length * train_percentage)
        validation_end_index = train_end_index + round(dataset_length * validation_percentage)
        
        # Split image ids into train and test
        train_image_ids = image_ids[: train_end_index]
        validation_image_ids = image_ids[train_end_index: validation_end_index]
        test_image_ids = image_ids[validation_end_index: dataset_length]
        
        return train_image_ids, validation_image_ids, test_image_ids
    
    def create_captions_list(self, image_ids):
        
        images_captions_list = []
        
        for image_id in image_ids:
            [images_captions_list.append(caption) for caption in self.caption_processor.rnn_image_caption_map[image_id]]
            
        return images_captions_list
    
    def create_fit_tokenizer(self, image_ids):
        '''  Create tokenizer and fit it to the captions of the image_ids '''
        
        captions_to_fit = self.create_captions_list(image_ids)
        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(captions_to_fit)
        
        return tokenizer
    
    def calculate_max_sequence_length(self, image_ids):
        ''' Calculates the length of captions with words for the given image_ids '''
        
        images_captions_list = self.create_captions_list(image_ids)
        
        return max([len(images_captions.split()) for images_captions in images_captions_list])
    
    def create_caption_sequences(self, image_ids):
        ''' Creates sequences of images, input sequences and output sequences '''
        
        image_features, image_caption_sequences, output_sequences = [], [], []
        
        for image_id in image_ids:
            
            img_captions = self.caption_processor.rnn_image_caption_map[image_id]
            for img_caption in img_captions:
                
                # Encode caption into sequence
                sequence = self.tokenizer.texts_to_sequences([img_caption])[0]
                
                # Split one sequence into multiple in_sequence and out_sequence pairs
                for i in range(1, len(sequence)):
                    
                    # Split into input and output pair
                    in_sequence, out_sequence = sequence[:i], sequence[i]
                    
                    # Padding input sequence
                    in_sequence = pad_sequences([in_sequence], maxlen=self.sequence_max_length)[0]
                    
                    # Encode output sequence
                    out_sequence = to_categorical([out_sequence], num_classes=self.vocabulary_size)[0]
                    
                    # Store output sequences
                    image_features.append(self.cnn_encoder.image_features[image_id][0])
                    image_caption_sequences.append(in_sequence)
                    output_sequences.append(out_sequence)
        
        return np.array(image_features), np.array(image_caption_sequences), np.array(output_sequences)
    
    def progressive_loading_data_generator(self, image_ids):
        
        # Infinite loop
        while True:
            for image_id in image_ids:
                image_feature, image_caption_sequences, output_sequences = self.create_caption_sequences([image_id])
                yield [image_feature, image_caption_sequences], output_sequences
    
    def create_image_captioning_model(self, dropout_probability=0.5, img_feature_shape=(4096, )):
        ''' Creates TF: Keras model for image captioning '''
        
        # Image feature extractor model
        img_in = Input(shape=img_feature_shape)
        img_do = Dropout(dropout_probability)(img_in)
        img_feature = Dense(256, activation='relu')(img_do)
        
        # Caption sequence model
        cap_seq_model_in = Input(shape=(self.sequence_max_length, ))
        cap_seq_em = Embedding(self.vocabulary_size, 256, mask_zero=True)(cap_seq_model_in)
        cap_seq_do = Dropout(dropout_probability)(cap_seq_em)
        cap_sequence = LSTM(256)(cap_seq_do)
        
        # Decoder model (makes the output layer)
        decoder_img_cap_seq = add([img_feature, cap_sequence])
        decoder_img_cap_seq_dense = Dense(256, activation='relu')(decoder_img_cap_seq)
        decoder_img_cap_seq_outputs = Dense(self.vocabulary_size, activation='softmax')(decoder_img_cap_seq_dense)
        
        # Create final image captioning model containing [image, sequence] [word]
        image_captioning_model = Model(inputs=[img_in, cap_seq_model_in], outputs=decoder_img_cap_seq_outputs)
        
        # image_captioning_model.compile(loss='categorical_crossentropy', optimizer='adam')
        image_captioning_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Print model summary and plot model architecture diagram
        print(image_captioning_model.summary())
        # plot_model(image_captioning_model, to_file='model.png', show_shapes=True)
        
        return image_captioning_model
    
    def train_image_captioning_model_progressive_loading(self, epochs):
        
        print('>>> Started training image captining model in progressive loading mode ....')
        start_time = time.time()
        
        # num_steps = len(self.train_image_ids)
        
        for i in range(epochs):
            
            # Train model
            train_generator = self.progressive_loading_data_generator(self.train_image_ids)
            
            validation_generator = self.progressive_loading_data_generator(self.validation_image_ids)
            
            # Fit for one epoch
            # self.image_captioning_model.fit_generator(train_generator, epochs=1, steps_per_epoch=num_steps, verbose=1)
            # self.image_captioning_model.fit(train_generator, epochs=1, steps_per_epoch=num_steps, verbose=1)
            self.image_captioning_model.fit(train_generator, epochs=1, validation_data=validation_generator)
            
            # Save model
            filename = "image_captioning_model_{self.caption_processor.kaggle_dataloader.dataset_type.name}_{str(i)}.h5'"
            self.image_captioning_model.save(filename)
            
        elapsed_time = time.time() - start_time
        print(f'>>> Image captining model training took: {elapsed_time} sec!')
    
    def train_image_captioning_model(self, epochs):
        ''' Trains the image captioning model for the specified number of epochs '''
        
        print('>>> Started training image captining model in normal mode ....')
        start_time = time.time()
        
        # Train model
        image_features_train, image_caption_sequences_train, output_sequences_train = self.create_caption_sequences(self.train_image_ids)
        
        image_features_validation, image_caption_sequences_validation, output_sequences_validation = self.create_caption_sequences(self.validation_image_ids)
        
        history = self.image_captioning_model.fit([image_features_train, image_caption_sequences_train], output_sequences_train, epochs=epochs,
                                                  verbose=1, validation_data=([image_features_validation, image_caption_sequences_validation], output_sequences_validation))
        
        elapsed_time = time.time() - start_time
        print(f'>>> Image captining model training took: {elapsed_time} sec!')
        
        return history
    
    def evaluate_image_captioning_model(self):
        ''' Evaluates image captioning model '''
        
        print('Evaluating image captioning model and calculating BLEU score ...')
        start_time = time.time()
        
        actual_captions, predicted_captions = list(), list()
        
        # Test for complete test set
        for image_id in self.test_image_ids:
            
            # generate description
            yhat = self.generate_caption_from_image_feature(self.cnn_encoder.image_features[image_id])
            
            # store actual and predicted
            img_captions = self.caption_processor.rnn_image_caption_map[image_id]
            references = [d.split() for d in img_captions]
            actual_captions.append(references)
            predicted_captions.append(yhat.split())
        
        # calculate BLEU score
        print('BLEU-1: %f' % corpus_bleu(actual_captions, predicted_captions, weights=(1.0, 0, 0, 0)))
        print('BLEU-2: %f' % corpus_bleu(actual_captions, predicted_captions, weights=(0.5, 0.5, 0, 0)))
        print('BLEU-3: %f' % corpus_bleu(actual_captions, predicted_captions, weights=(0.3, 0.3, 0.3, 0)))
        print('BLEU-4: %f' % corpus_bleu(actual_captions, predicted_captions, weights=(0.25, 0.25, 0.25, 0.25)))
        
        elapsed_time = time.time() - start_time
        print(f'>>> Image captining model evaluation took: {elapsed_time} sec!')
        
    def decode_tokenid_to_word(self, token_id):
        
        for word, index in self.tokenizer.word_index.items():
            if index == token_id:
                return word
        return None
    
    def generate_caption_from_image_feature(self, image):
        ''' Generates captions for an image '''
        
        # seed the generation process
        # in_text = '<start>'
        in_text = 'startseq'
        
        # iterate over the whole length of the sequence
        for i in range(self.sequence_max_length):
            
            # integer encode input sequence
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            
            # pad input
            sequence = pad_sequences([sequence], maxlen=self.sequence_max_length)

            # predict next word
            yhat = self.image_captioning_model.predict([image, sequence], verbose=0)
            
            # convert probability to integer
            yhat = argmax(yhat)
            
            # map integer to word
            word = self.decode_tokenid_to_word(yhat)
            
            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += ' ' + word
            
            # stop if we predict the end of the sequence
            if word == 'endseq':  # '<end>':
                break
            
        return in_text

    def generate_caption_and_plot(self, filepath):
        ''' Generates captions of an image given in the filepath and plots it along with the generated captions '''
        
        image_feature = self.cnn_encoder.encode_features_image(filepath)
        generated_captions = self.generate_caption_from_image_feature(image_feature)
        # generated_captions = generated_captions.replace(' end', '').replace('<start>', '').replace('<end>', '')
        generated_captions = generated_captions.replace(' end', '').replace('startseq', '').replace('endseq', '')
        generated_captions = re.sub('seq$', '', generated_captions)
        
        # Plot
        fig = plt.figure(figsize=(20, 10))
        
        # Plot scaled image
        scaled_image = load_img(filepath, target_size=self.cnn_encoder.cnn_model_type.value['target_size'])
        ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
        ax.imshow(scaled_image)
        
        # Add generated captions
        ax = fig.add_subplot(1, 2, 2)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0, 0.5, generated_captions, fontsize=20, wrap=True)
        plt.show()
    
    def plot_model_performance(self):
        
        # list all data in history
        print(f'Keys present in model training history: {self.history.history.keys()}')
        
        # Plot acuracy history
        if self.history.history['accuracy']:
            plt.plot(self.history.history['accuracy'])
            
            plt.plot(self.history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
        
        # Plot loss history
        if self.history.history['loss']:
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
