# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:08:41 2021

@author: Zack
"""
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

test_df = pd.read_csv('./test_no_punct.csv')
train_df = pd.read_csv('./train_no_punct.csv')

# %% Remove simple stopwords, apply gentle lemmatizer
wnl = WordNetLemmatizer()
stopword = stopwords.words('english')


def lemmatize(tokens):
    return [wnl.lemmatize(word) for word in tokens]


def check_st(tokens, stopword):
    lower = [word.lower() for word in tokens]
    return [word for word in lower if word not in stopword]


test_df.text = test_df.text.apply(lambda x: ' '.join(
    check_st(lemmatize(x.split()), stopword)))
test_df = test_df[test_df.text.astype(bool)]

train_df.text = train_df.text.apply(lambda x: ' '.join(
    check_st(lemmatize(x.split()), stopword)))
train_df = train_df[train_df.text.astype(bool)]

train_df.to_csv('train_no_st_0.csv', index=False)
test_df.to_csv('test_no_st_0.csv', index=False)
