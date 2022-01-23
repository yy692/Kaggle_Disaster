# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:58:45 2022

@author: Ruru Dai
"""
#%%
import pandas as pd
import re
import preprocessor as prep
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

'''define how to replace punctuations, and words after @ till the next space'''
REPLACE_NO_SPACE = re.compile("(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})|(#)|(@\w*)|(\*)")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:)|(\.)")


#%% Data preprocessing - CLEAN

def clean_tweets(df ):
    temp_array = []
    for line in df:
        '''use preprocessor to opt out url and lowercase words in lines'''
        prep.set_options(prep.OPT.URL)
        temp_line = prep.clean(line.lower())
        
        '''remove/replace punctuationand, and words include and after @ till the next space'''
        temp_line = REPLACE_WITH_SPACE.sub(" ", temp_line)
        temp_line = REPLACE_NO_SPACE.sub("", temp_line)
        
        '''clean spaces'''
        temp_line = re.compile("\s{2,}").sub(" ", temp_line)
        temp_line = re.compile("^\s*").sub("", temp_line)
        
        temp_array.append(temp_line)
    return temp_array        

def get_key_word(df):
    '''tokenize and remove stop words'''
    temp_array = []
    stop_words = stopwords.words('english')
    for line in df:
        token_line = word_tokenize(line)  
        filtered_line = [word for word in token_line if word not in stop_words]
        temp_array.append(filtered_line)
    return temp_array


def data_preprocess(file = 'test.csv'):
    raw_data = pd.read_csv(file)
    raw_data['clean_text'] = clean_tweets(raw_data.text)
    raw_data['token'] = get_key_word(raw_data['clean_text'])
    return raw_data

#%%

x = (data_preprocess())