import os
import numpy as np
import string

target_path = '..\\corpus_no_num'
train_path = '.\\train'
test_path = '.\\test'
dev_path = '.\\dev'
stopwords = set(np.load("./stopwords.npy"))

npy_train = []
for root, dirs, files in os.walk(train_path):
    for f in files:
        npy_train.append(os.path.join(root, f))
npy_dev = []
for root, dirs, files in os.walk(dev_path):
    for f in files:
        npy_dev.append(os.path.join(root, f))
npy_test = []
for root, dirs, files in os.walk(test_path):
    for f in files:
        npy_test.append(os.path.join(root, f))


def ensure_path(path):
    if not (os.path.exists(path)):
        os.makedirs(path)

def remove_num(doc):
    res = []
    for s in doc:
        sent = []
        for w in s:
            if not w.isnumeric():
                sent.append(w)
        if sent:
            res.append(sent)
    return res

def remove_punct(doc):
    res = []
    for s in doc:
        sent = []
        for w in s:
            if w[0] not in string.punctuation:
                sent.append(w)
        if sent:
            res.append(sent)
    return res

def remove_stop(doc):
    res = []
    for s in doc:
        sent = []
        for w in s:
            if w not in stopwords:
                sent.append(w)
        if sent:
            res.append(sent)
    return res


if __name__ == "__main__":
    # print(npy_test)
    # print(stopwords)
    for f in npy_test:
        doc = np.load(f, allow_pickle=True)
        target = os.path.join(target_path, '\\'.join(f.split('\\')[-4:]))
        ensure_path(os.path.split(target)[0])
        res = remove_num(doc)
        res = remove_punct(res)
        res = remove_stop(res)
        np.save(target, res)
    for f in npy_dev:
        doc = np.load(f, allow_pickle=True)
        target = os.path.join(target_path, '\\'.join(f.split('\\')[-4:]))
        ensure_path(os.path.split(target)[0])
        res = remove_num(doc)
        res = remove_punct(res)
        res = remove_stop(res)
        np.save(target, res)
    for f in npy_train:
        doc = np.load(f, allow_pickle=True)
        target = os.path.join(target_path, '\\'.join(f.split('\\')[-4:]))
        ensure_path(os.path.split(target)[0])
        res = remove_num(doc)
        res = remove_punct(res)
        res = remove_stop(res)
        np.save(target, res)
