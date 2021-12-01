#
import os
import string
import numpy as np
import pandas as pd
from pprint import pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.datasets import fetch_20newsgroups

stopwords = stopwords.words('english')
stemmer = SnowballStemmer("english")
wnl = WordNetLemmatizer()
end_punct = [".", "?", "!"]
need_punct = ["-", "'"]
punctuation = list(string.punctuation)
for i in end_punct + need_punct:
    punctuation.remove(i)

train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers'), random_state=36)
test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers'), random_state=36)


def process(text):
    with open('temp.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    raw = []
    with open('temp.txt', 'r', encoding='utf-8') as f:
        raw = f.readlines()
    # print(text)
    text = []
    for i in raw:
        if i.find("writes:")==-1 and i.find("@")==-1:
            text.append(i)
    for i in range(len(text)):
        text[i] = text[i].replace('--', '')
        text[i] = text[i].replace('."', '')
        for p in punctuation:
            text[i] = text[i].replace(p, '')

    text = ''.join(text).split()
    for p in string.punctuation:
        while p in text:
            text.remove(p)

    doc = []
    sent = []
    for word in text:
        isEnd = False
        for p in end_punct:
            if word.endswith(p):
                isEnd = True
        if word == text[-1]:
            isEnd = True
        if (isEnd):
            sent.append(word)
            comb = ' '.join(sent)
            tokens = word_tokenize(comb)
            for stop in stopwords:
                while stop in tokens:
                    tokens.remove(stop)
            tokens = [stemmer.stem(w) for w in tokens]
            doc.append(tokens)
            sent = []
            continue
        sent.append(word)
    return np.array(doc, dtype=object)


if __name__ == "__main__":
    for i in range(len(train_data.data)):
        # print(train_data.data[i])
        corpus = process(train_data.data[i])
        path = './corpus/train/'+train_data.filenames[i].split('\\')[-2]
        name = train_data.filenames[i].split('\\')[-1]
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, name+'.npy'), corpus)
        with open(os.path.join(path, name+'.txt'), 'w', encoding='utf-8', newline='\n') as f:
            for sent in corpus:
                for word in sent:
                    f.write(word+' ')
                f.write('\n')
    for i in range(len(test_data.data)):
        # print(test_data.data[i])
        corpus = process(test_data.data[i])
        path = './corpus/test/'+test_data.filenames[i].split('\\')[-2]
        name = test_data.filenames[i].split('\\')[-1]
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, name+'.npy'), corpus)
        with open(os.path.join(path, name+'.txt'), 'w', encoding='utf-8', newline='\n') as f:
            for sent in corpus:
                for word in sent:
                    f.write(word+' ')
                f.write('\n')
