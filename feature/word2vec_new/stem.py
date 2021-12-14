# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 22:25:10 2021

@author: Zack
"""

from nltk.stem.snowball import SnowballStemmer
import pandas as pd

stemmer = SnowballStemmer("english")


test_df = pd.read_csv('./train_cust_1.csv')
train_df = pd.read_csv('./test_cust_1.csv')


def stemming(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])


test_df.text = test_df.text.apply(lambda x: stemming(x))
test_df = test_df[test_df.text.astype(bool)]

train_df.text = train_df.text.apply(lambda x: stemming(x))
train_df = train_df[train_df.text.astype(bool)]

train_df.to_csv('train_stem_1.csv.csv', index=False)
test_df.to_csv('test_stem_1.csv', index=False)
