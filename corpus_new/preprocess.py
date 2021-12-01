# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:00:26 2021

@author: Zack
"""
import string
from nltk import word_tokenize
import pandas as pd

test_df = pd.read_csv('./test_raw.csv')
train_df = pd.read_csv('./train_raw.csv')
test_df = test_df.dropna()
train_df = train_df.dropna()

# %% Remove empty text, punctuations, and repeated strings

punctuation = list(string.punctuation)+["''", '""', '...', '``', '--']


def check_word(word):
    if word in punctuation:
        return False
    if len(word) == 1:
        return True
    count = 0
    prev = word[0]
    current = ''
    for i in range(len(word)):
        if i == 0:
            continue
        if count >= 3:
            if word.isupper():
                return True
            return False
        current = word[i]
        if prev in punctuation and current in punctuation:
            return False
        if prev == current:
            count += 1
        prev = current
    return True


def remove_punct(text):
    return ' '.join([word for word in word_tokenize(text) if check_word(word)])


train_df.text = train_df.text.apply(lambda x: remove_punct(x))
train_df = train_df[train_df.text.astype(bool)]
test_df.text = test_df.text.apply(lambda x: remove_punct(x))
test_df = test_df[test_df.text.astype(bool)]

train_df.to_csv('train_no_punct.csv', index=False)
test_df.to_csv('test_no_punct.csv', index=False)

# %%
