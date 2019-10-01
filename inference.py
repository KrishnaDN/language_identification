#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:53:09 2019

@author: krishna
"""

from __future__ import absolute_import
from __future__ import print_function
import os
from attention_lstm import AttentionModel
from torch.utils.data import DataLoader   
from sklearn.metrics import accuracy_score
import numpy as np
from torch import optim
import torch
from utility import utils
import os
import soundfile as sf



lang_id={'0':'english','1':'hindi','2':'Non-english-hindi'}

#######

## Model related
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")

##### MOdel specific info

device='cpu'
batch_size=32
inputbatch_size=32
input_size=80 
output_size=3
hidden_size=256
num_layers=3
model = AttentionModel(batch_size, input_size, output_size, hidden_size,num_layers).to(device)

checkpoint = torch.load('model_lstm/check_point_100',map_location='cpu')
model.load_state_dict(checkpoint['model'])

#### Data specific info

def prediction(video_link):
    #video_link = 'https://d3i3lk5iax7dja.cloudfront.net/1ef9a39f-3c6c-4e05-8096-ff146eb8aa06.mp4'
    download = 'wget '+video_link+' -O temp.mp4'
    os.system(download)
    convert  = 'ffmpeg -i temp.mp4 -f wav temp.wav'
    os.system(convert)
    downsample = 'sox temp.wav -r 16k -c 1 temp_16k.wav'
    os.system(downsample)
    audio_path='temp_16k.wav'
    
    spec = utils.load_data(audio_path)
    sample = torch.from_numpy(spec)
    features = sample.reshape([1,801,80]).to(device)
    preds = model(features)
    prediction = np.argmax(preds.detach().cpu().numpy(),axis=1)
    language = lang_id[str(prediction[0])]
    os.remove('temp.mp4')
    os.remove('temp.wav')
    os.remove('temp_16k.wav')
    
    return language
