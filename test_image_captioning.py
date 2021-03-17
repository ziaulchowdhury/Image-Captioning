#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Project: Data Science Programming (D7054E)

@author: Ziaul Islam Chowdhury

Created on Mar 16 2021

Luleå University of Technology
"""

from cnn_encoder import CnnEncoder, CnnModelType
from data_loader import KaggleDataLoader, DataSet
from data_preprocessing import CaptionProcessor
from image_captioning_model import ImageCaptioningModel

#
# Configuration parameters
#
run_flickr_30k = False
num_epochs = 20
target_dir = '/Users/ziaulchowdhury/Desktop'
download_dataset = False

#
# Fixed parameters
#
flickr8k_image_dir = target_dir + '/flickr8k/Images'
flickr30k_image_dir = target_dir + '/flickr-image-dataset/Images'

if run_flickr_30k:
    flickr_image_dir = flickr30k_image_dir
    dataset_type = DataSet.Flickr30K
else:
    flickr_image_dir = flickr8k_image_dir
    dataset_type = DataSet.Flickr8K

print('................. Starting CnnEncoder ......................')
cnn_encoder = CnnEncoder(CnnModelType.VGG16)
cnn_encoder.encode_features_directory(flickr_image_dir)
cnn_encoder.dump_image_features(run_flickr_30k)
print('>>> Completed CnnEncoder !')

#
# Create an instance of KaggleDataLoader
#
print('................. Starting KaggleDataLoader ......................')
kd = KaggleDataLoader(dataset_type, should_unzip=False, target_dir=target_dir)
if download_dataset:
    kd.download_dataset_kaggle_api(already_downloaded=True)
    kd.post_processing(already_processed=True)
else:
    kd.download_dataset_kaggle_api(already_downloaded=False)
    kd.post_processing(already_processed=False)
print('>>> Completed KaggleDataLoader !')

#
# Create an instance of CaptionProcessor
#
print('................. Starting CaptionProcessor ......................')
cp = CaptionProcessor(kd)
print('>>> Completed CaptionProcessor !')

#
# ImageCaptioningModel
#
print('................. Starting ImageCaptioningModel ......................')
icm = ImageCaptioningModel(cp, cnn_encoder, epochs=num_epochs)
print('>>> Completed image captioning model training!')
print('................. Starting ImageCaptioningModel model evaluation (BLEU score) ......................')
icm.evaluate_image_captioning_model()
print('>>> Completed image captioning model evaluation (BLEU score)!')

# Check some generated captions (image ids of the Flickr8K dataset)
if not run_flickr_30k:
    image_ids = ['115684808_cb01227802', '111537217_082a4ba060', '1075716537_62105738b4', '1012212859_01547e3f17',
                '1000268201_693b08cb0e', '123889082_d3751e0350', '1302657647_46b36c0d66', '1398613231_18de248606']
    print(f'................. Started generating captions of the image_ids: {image_ids}  ......................')
    for image_id in image_ids:
        icm.generate_caption_and_plot(flickr8k_image_dir + '/' + image_id + '.jpg')
    print('>>> Completed generation of captions for the image_ids!')
