#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 19:33:50 2022

@author: wenyilin
"""

#%% Load libraries
import numpy as np 
import pandas as pd 
import re
import string
import nltk
from nltk.corpus import stopwords
import os


#%% Loading data
dat_dir ='/Users/wenyilin/Dropbox/UCSD/2022Winter/Jerker2312/nlp-getting-started'
os.chdir(dat_dir)

train = pd.read_csv('train.csv')
train.head()
test = pd.read_csv('test.csv')
test.head()

#%% Counts of targets
train['target'].value_counts()

#%% Clean data
def clean_data(text):
    text = text.lower() # lower case
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    text = re.sub('\n', '', text) # remove separators
    text = re.sub('\r', '', text) # remove separators
    text = re.sub('\w*\d\w*', '', text) # remove words containing numbers
    return text 

# Applying the cleaning function to both test and training datasets
train['text'] = train['text'].apply(lambda x: clean_data(x))
test['text'] = test['text'].apply(lambda x: clean_data(x))

train['text'].head()

#%% Tokenization
tokenizer =  nltk.tokenize.RegexpTokenizer(r'\w+')
train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))
test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))
train['text'].head()

#%% Remove stop words
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = [w for w in text if w not in stop_words]
    return words

train['text'] = train['text'].apply(lambda x : remove_stopwords(x))
test['text'] = test['text'].apply(lambda x : remove_stopwords(x))
train.head()

#%% Stemming/Lemmatization
# may not be necessary'

#stemmer  = PorterStemmer()
#lemmatizer = WordNetLemmatizer()

#%% 
