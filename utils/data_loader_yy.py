import re
import csv
import string
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder

def data_loader(fname):
    df = pd.read_csv(fname)

    # get each column of data
    text = df["text"].tolist()
    index = df["id"].tolist()
    target = df["target"].tolist()
    keyword = df["keyword"].tolist()
    location = df["location"].tolist()
    
    # tools
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    lib = set()
    idx_dict = {}
    idx_curr = 0
   
    result = []
    iflinks = []
    for line, kw in zip(text[:15],keyword[:15]):
        # specialize keyword
        if not kw: kw = ''
        kw = re.sub('[^A-Za-z0-9]+','',str(kw))
        kw = re.sub(' +', '', kw)
        line = line+" ##"+kw+"@@"

        # deal with links
        links = re.findall(link_regex, line)
        if links:
            for l in links: line = line.replace(l[0]," ")
            iflinks.append(1)
        else:
            iflinks.append(0)

        # deal with punctuation
        temp = ""
        for pun in string.punctuation:
            if pun in line:
                temp += " "+pun
        line = re.sub('[^A-Za-z0-9@#]+',' ',line)
        line = re.sub(' +', ' ', line)
        print(line)

        # get indexes for words
        indlist = []
        for word in (line+temp).split():
            if word in lib:
                indlist.append(idx_dict[word])
            else:
                indlist.append(idx_curr)
                idx_dict[word] = idx_curr
                idx_curr += 1
        result.append(indlist)

        feature = []

    # embedding
    for res, iflink in zip(result,iflinks):
        vector = [0]*idx_curr + [iflink]
        for idx in res:
            vector[idx] = 1
        feature.append(vector)

    print(feature)
    return np.array(index), np.array(target), np.array(feature)

def test_SVC(ftrain, ftest):
    _, yTr, xTr = data_loader(ftrain)
    #_, yTe, xTe = data_loader(ftest)

    yTe = yTr[5000:] 
    xTe = xTr[5000:]
    yTr = yTr[:5000] 
    xTr = xTr[:5000]

    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(xTr, yTr)
    yPr = svclassifier.predict(xTe)

    print(np.sum(np.abs(yPr - yTe))/len(yTe))

if __name__=='__main__':
    data_loader("train.csv")
    #test_SVC("train.csv", "test.csv")
