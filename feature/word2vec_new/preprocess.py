# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 11:13:12 2021

@author: Zack
"""

import pandas as pd
import string
import numpy as np
import re


# %% Sep text by punct
train_df = pd.read_csv('../../corpus_new/train_no_num_0.csv')
test_df = pd.read_csv('../../corpus_new/test_no_num_0.csv')

punct = list(string.punctuation)


def check_word(word):
    if len(word) == 1:
        return True
    count = 0
    prev = word[0]
    current = ''
    for i in range(len(word)):
        if i == 0:
            continue
        if count >= 3:
            return False
        current = word[i]
        if prev == current:
            count += 1
        prev = current
    return True


def replace_punct(text):
    for i in punct:
        text = text.replace(i, ' ')
    text = ' '.join([word for word in text.split() if check_word(word)])
    return text


train_df.text = train_df.text.apply(lambda x: replace_punct(x))
train_df = train_df[train_df.text.astype(bool)]
test_df.text = test_df.text.apply(lambda x: replace_punct(x))
test_df = test_df[test_df.text.astype(bool)]

train_df.to_csv('train_no_punct_1.csv', index=False)
test_df.to_csv('test_no_punct_1.csv', index=False)


# %% Remove ST (advanced)
test_df = pd.read_csv('./test_no_punct_1.csv')
train_df = pd.read_csv('./train_no_punct_1.csv')

stopword = set(np.load('./stopwords.npy'))


def remove_st(text, stopword):
    return ' '.join([word for word in text.split() if word not in stopword])


train_df.text = train_df.text.apply(lambda x: remove_st(x, stopword))
train_df = train_df[train_df.text.astype(bool)]
test_df.text = test_df.text.apply(lambda x: remove_st(x, stopword))
test_df = test_df[test_df.text.astype(bool)]

train_df.to_csv('train_no_st_1.csv', index=False)
test_df.to_csv('test_no_st_1.csv', index=False)


# %% Remove num (naive)
test_df = pd.read_csv('./test_no_st_1.csv')
train_df = pd.read_csv('./train_no_st_1.csv')


def remove_num(text):
    return ' '.join([word for word in text.split() if not word.isnumeric()])


test_df.text = test_df.text.apply(lambda x: remove_num(x))
test_df = test_df[test_df.text.astype(bool)]

train_df.text = train_df.text.apply(lambda x: remove_num(x))
train_df = train_df[train_df.text.astype(bool)]

train_df.to_csv('train_no_num_1.csv', index=False)
test_df.to_csv('test_no_num_1.csv', index=False)


# %% Remove num (progressive)
test_df = pd.read_csv('./test_no_num_1.csv')
train_df = pd.read_csv('./train_no_num_1.csv')


def remove_num(text):
    return ' '.join([word for word in text.split() if not word[0].isnumeric()])


test_df.text = test_df.text.apply(lambda x: remove_num(x))
test_df = test_df[test_df.text.astype(bool)]

train_df.text = train_df.text.apply(lambda x: remove_num(x))
train_df = train_df[train_df.text.astype(bool)]

train_df.to_csv('train_no_num_2.csv', index=False)
test_df.to_csv('test_no_num_2.csv', index=False)


# %% Remove customize words 1
test_df = pd.read_csv('./test_no_num_2.csv')
train_df = pd.read_csv('./train_no_num_2.csv')

no_num = re.compile(r'.*\d+.*')
no_abc = re.compile(r'.*abcde.*')
re_list = [no_num, no_abc]


def check_word(word, re_list):
    for r in re_list:
        if re.match(r, word):
            return False
    return True


def remove_cust(text):
    return ' '.join([word for word in text.split() if check_word(word, re_list)])


test_df.text = test_df.text.apply(lambda x: remove_cust(x))
test_df = test_df[test_df.text.astype(bool)]

train_df.text = train_df.text.apply(lambda x: remove_cust(x))
train_df = train_df[train_df.text.astype(bool)]

train_df.to_csv('train_cust_0.csv', index=False)
test_df.to_csv('test_cust_0.csv', index=False)


# %% Remove customize words 2
test_df = pd.read_csv('./test_cust_0.csv')
train_df = pd.read_csv('./train_cust_0.csv')

no_aaa = re.compile(r'^a+$')
re_list = [no_aaa]


def check_word(word, re_list):
    for r in re_list:
        if re.match(r, word):
            return False
    return True


def remove_cust(text):
    return ' '.join([word for word in text.split() if check_word(word, re_list)])


test_df.text = test_df.text.apply(lambda x: remove_cust(x))
test_df = test_df[test_df.text.astype(bool)]

train_df.text = train_df.text.apply(lambda x: remove_cust(x))
train_df = train_df[train_df.text.astype(bool)]

train_df.to_csv('train_cust_1.csv', index=False)
test_df.to_csv('test_cust_1.csv', index=False)

