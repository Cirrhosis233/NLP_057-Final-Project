# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:37:38 2021

@author: Zack
"""

import pandas as pd

test_df = pd.read_csv('./test_no_st_0.csv')
train_df = pd.read_csv('./train_no_st_0.csv')

# %% Remove simple numbers


def remove_num(tokens):
    return ' '.join([word for word in tokens if not word.isnumeric()])


test_df.text = test_df.text.apply(lambda x: remove_num(x.split()))
test_df = test_df[test_df.text.astype(bool)]

train_df.text = train_df.text.apply(lambda x: remove_num(x.split()))
train_df = train_df[train_df.text.astype(bool)]

train_df.to_csv('train_no_num_0.csv', index=False)
test_df.to_csv('test_no_num_0.csv', index=False)
