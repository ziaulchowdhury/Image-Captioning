#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Project: Data Science Programming (D7054E)

@author: Ziaul Islam Chowdhury

Created on Mar 16 2021

Luleå University of Technology
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import kurtosis, skew
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import collections
from collections import defaultdict

from data_loader import KaggleDataLoader, DataSet
from data_preprocessing import CaptionProcessor


class DescriptiveAnalysis:
    ''' Performs descriptive analysis on Flick8K or Flickr30K dataset '''
    
    def __init__(self, caption_processor):
        
        self.caption_processor = caption_processor
    
    def plot_caption_length_hist(self):
        ''' Plots histogram of caption lengths and checks skewness and kurtosis of the gaussian distribution '''
        
        caption_len_df = pd.DataFrame()
        
        caption_len_df['cap_len'] = self.caption_processor.image_captions_df['caption'].apply(lambda x: len(x))
        
        plt.figure()
        sns.displot(caption_len_df, x="cap_len", binwidth=2)
        plt.title('Caption length histogram')
        plt.xlabel('Length of Captions')
        plt.ylabel('Count')
        plt.show()
        
        # Check skewness
        skewness = skew(caption_len_df['cap_len'].values)
        print(f'Skewness : {skewness}')
        
        # Check kurtosis
        kurt_value = kurtosis(caption_len_df['cap_len'].values)
        print(f'Kurtosis : {kurt_value}')
    
    def plot_most_common_words(self):
        ''' Plots top most frequent words '''
        
        vocabulary_count_df = pd.DataFrame(self.caption_processor.caption_vocabulary_count_map)
        vocabulary_count_df.columns = ['word', 'count']
        
        # Plot top 20 most common words
        top_20_words = vocabulary_count_df[vocabulary_count_df['count'] > 100].sort_values(by=['count'], ascending=False).head(20)
        plt.figure(figsize=(14, 8))
        sns.barplot(x="word", y="count", data=top_20_words)
        plt.title('Top 20 most used words')
        plt.show()
        
        # Plot top 21 to 40 most common words
        top_21to40_words = vocabulary_count_df[vocabulary_count_df['count'] > 100].sort_values(by=['count'], ascending=False).iloc[20:40]
        plt.figure(figsize=(14, 8))
        sns.barplot(x="word", y="count", data=top_21to40_words)
        plt.title('Top 20 to 40 most used words')
        plt.show()
    
    def create_caption_corpus_dict(self):
        ''' Creates corpus of captions and a dictionary containing count of word frequency '''
        
        nltk.download('stopwords')
        en_stopwords = set(stopwords.words('english'))
        
        captions_corpus = []
        
        captions = self.caption_processor.image_captions_df['caption'].str.split()
        captions_list = captions.values.tolist()
        
        captions_corpus = [word for i in captions_list for word in i]
        
        captions_stopword_dict = defaultdict(int)
        for word in captions_corpus:
            if word in en_stopwords:
                captions_stopword_dict[word] = captions_stopword_dict[word] + 1
        
        return captions_corpus, captions_stopword_dict
    
    def plot_top_ngram(self, captions_corpus, num_ngram=2, top_ngrams=10):
        ''' Plots top Ngram of the corpus  excluding stopwords'''
        
        ngrams_corpus = ngrams(captions_corpus, num_ngram)
        
        ngrams_frequency = collections.Counter(ngrams_corpus)
        
        ngrams_frequency_counter = collections.Counter(ngrams_frequency)
        
        top_ngram_words = ngrams_frequency_counter.most_common(top_ngrams)
        
        x, y = [], []
        for i in range(top_ngrams):
            x.append(' '.join(top_ngram_words[i][0]))
            y.append(top_ngram_words[i][1])
        
        # Plot bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(x=y, y=x)
        plt.title(f'Ngrams of the captions of size {num_ngram}')
        plt.xlabel('Frequency')
        plt.show()


if __name__ == '__main__':
    
    run_flickr_30k = True
    
    target_dir = '/Users/ziaulchowdhury/Desktop'
    dataset_type = DataSet.Flickr30K if run_flickr_30k else DataSet.Flickr8K
    
    # Create an instance of KaggleDataLoader
    print('................. Starting KaggleDataLoader ......................')
    kd = KaggleDataLoader(dataset_type, should_unzip=False, target_dir=target_dir)
    kd.download_dataset_kaggle_api(already_downloaded=True)
    kd.post_processing(already_processed=True)
    print('>>> Completed KaggleDataLoader !')
    
    # Create an instance of CaptionProcessor
    print('................. Starting CaptionProcessor ......................')
    cp = CaptionProcessor(kd)
    print('>>> Completed CaptionProcessor !')
    
    # Create an instance of DescriptiveAnalysis
    print('................. Starting CaptionProcessor ......................')
    da = DescriptiveAnalysis(cp)
    print('>>> Completed DescriptiveAnalysis !')
        
    da.plot_caption_length_hist()
    da.plot_most_common_words()
    captions_corpus, captions_stopword_dict = da.create_caption_corpus_dict()
    
    # Plot Bigram and tri-gram
    da.plot_top_ngram(captions_corpus, num_ngram=2, top_ngrams=20)
    da.plot_top_ngram(captions_corpus, num_ngram=3, top_ngrams=10)
