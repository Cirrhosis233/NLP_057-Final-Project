# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:00:26 2021

@author: Zack
"""

import pandas as pd

test_df = pd.read_csv('./test_raw.csv')
train_df = pd.read_csv('./train_raw.csv')
test_df = test_df.dropna()
train_df = train_df.dropna()

#%%
from nltk import word_tokenize
import string

punctuation = list(string.punctuation)+["''", '""', '...', '``']

tokens = word_tokenize(text)
clean = [word for word in tokens if word not in punctuation]
