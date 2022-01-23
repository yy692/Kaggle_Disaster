import re
import csv
import string
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def cut_links(line):
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, line)
    if links: 
        for l in links: line = line.replace(l[0]," ")
    return line

def cut_punctuation(line):
    #line = re.sub('[^A-Za-z0-9@]+',' ',line)
    line = re.sub('[^A-Za-z@]+',' ',line)
    line = re.sub(' +', ' ', line)
    return line

def cut_at(line):
    temp = []
    for word in line.split():
        if not word.startswith('@'):
            temp.append(word)
    return ' '.join(temp)

def cut_stopword(clean_line, language):
    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(clean_line)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)

def embedding(clean_text, tf):
    vectorizer = CountVectorizer()
    vec = vectorizer.fit_transform(clean_text)

    print(vec.toarray()[0][0:20])
    if tf:
        transformer = TfidfTransformer(smooth_idf=False)
        vec = transformer.fit_transform(vec.toarray())
    print(vec.toarray()[0][0:20])
    return vec.toarray() 

def data_loader(fname, stopword=False, language='english', tf=True):
    df = pd.read_csv(fname)
    print(df.loc[df['id'] == 891])

    # get each column of data
    text = df["text"].tolist()
    index = df["id"].to_numpy()
    target = df["target"].to_numpy()
    keyword = df["keyword"].tolist()
    location = df["location"].tolist()
    
    clean_text = []
    for line in text:
        # all lower case
        line = line.lower()
        # deal with links
        line = cut_links(line)
        # deal with punctuation
        line = cut_punctuation(line)
        # deal with @
        clean_line = cut_at(line)
        # deal with stop words
        if stopword: clean_line = cut_stopword(clean_line, language)
        clean_text.append(clean_line)

    clean_text = embedding(clean_text, tf)
    print(np.sum(target[:5000])/len(target[:5000]))
    print(np.sum(target[5000:])/len(target[5000:]))
    return index, target, clean_text

def test_SVC(ftrain, ftest, stopword=False, language='english', tf=True):
    _, yTr, xTr = data_loader(ftrain, stopword=stopword, language=language, tf=tf)
    #_, yTe, xTe = data_loader(ftest)
    print(len(yTr))
    print(len(xTr))

    yTe = yTr[5000:] 
    xTe = xTr[5000:]
    yTr = yTr[:5000] 
    xTr = xTr[:5000]

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(xTr, yTr)
    yPr = svclassifier.predict(xTe)

    print(1.0-np.sum(np.abs(yPr - yTe))/len(yTe))
    print(np.sum(yTr)/len(yTe))

if __name__=='__main__':
    _, yTr, xTr = data_loader("train.csv")
    print(len(xTr[0]))
    #test_SVC("train.csv", "test.csv", stopword=False, language='english', tf=False)
    #test_SVC("train.csv", "test.csv", stopword=True, language='english', tf=False)
    #test_SVC("train.csv", "test.csv", stopword=False, language='english', tf=True)
    #test_SVC("train.csv", "test.csv", stopword=True, language='english', tf=True)
