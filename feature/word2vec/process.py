import os
import string
import numpy as np
from pprint import pprint
from nltk.corpus import stopwords

corpus_path = "..\\..\\corpus"
stopwords = set(stopwords.words('english'))
train_path = "..\\..\\preprocessing\\corpus\\train"
dev_path = "..\\..\\preprocessing\\corpus\\dev"
test_path = "..\\..\\preprocessing\\corpus\\test"

npy_train = []
for root, dirs, files in os.walk(train_path):
    for f in files:
        if f.endswith('.npy'):
            npy_train.append(os.path.join(root, f))
npy_dev = []
for root, dirs, files in os.walk(dev_path):
    for f in files:
        if f.endswith('.npy'):
            npy_dev.append(os.path.join(root, f))
npy_test = []
for root, dirs, files in os.walk(test_path):
    for f in files:
            if f.endswith('.npy'):
                npy_test.append(os.path.join(root, f))

def ensure_path(path):
    if not (os.path.exists(path)):
        os.makedirs(path)

def remove_stop(doc):
    res = []
    for s in doc:
        sent = []
        for w in s:
            if w not in stopwords:
                sent.append(w)
        res.append(sent)
    return res

def remove_punct(doc):
    res = []
    for s in doc:
        sent = []
        for w in s:
            if w not in string.punctuation:
                sent.append(w)
        res.append(sent)
    return res


if __name__ == "__main__":
    # p = '\\'.join(npy_train[0].split('\\')[-3:])
    # p = os.path.join(corpus_path, p)
    # print(p)
    for f in npy_test:
        doc = np.load(f, allow_pickle=True)
        target = os.path.join(corpus_path, '\\'.join(f.split('\\')[-3:]))
        ensure_path(os.path.split(target)[0])
        res = remove_stop(doc)
        res = remove_punct(res)
        np.save(target, res)
        # test = np.load(target, allow_pickle=True)
        # pprint(test)
    for f in npy_dev:
        doc = np.load(f, allow_pickle=True)
        target = os.path.join(corpus_path, '\\'.join(f.split('\\')[-3:]))
        ensure_path(os.path.split(target)[0])
        res = remove_stop(doc)
        res = remove_punct(res)
        np.save(target, res)
    for f in npy_train:
        doc = np.load(f, allow_pickle=True)
        target = os.path.join(corpus_path, '\\'.join(f.split('\\')[-3:]))
        ensure_path(os.path.split(target)[0])
        res = remove_stop(doc)
        res = remove_punct(res)
        np.save(target, res)
