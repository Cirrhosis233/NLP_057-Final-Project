# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 22:30:36 2021

@author: Zack
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec

data = pd.read_csv('./train_stem_1.csv')
test_df = pd.read_csv('./test_stem_1.csv')
all_text = data.text.append(test_df.text, ignore_index=True)

word_list = []

for doc in all_text:
    tokens = [word for word in doc.split()]
    word_list.append(tokens)

# np.save('word_list.npy', word_list)

word2vec_model = Word2Vec(word_list, vector_size=300,
                          sg=1, hs=1, window=9, min_count=1)
word2vec_model.save('word2vec_model.w2v')


# %%
from gensim.models.keyedvectors import KeyedVectors

google_model = KeyedVectors.load_word2vec_format(
    "./Pre_w2v/GoogleNews-vectors-negative300.bin", binary=True)

# %%
# word2vec_model = Word2Vec.load("word2vec_model.w2v")
print(word2vec_model.wv.most_similar('machin'))
print()
print(google_model.most_similar('machin'))
print()
print(google_model.most_similar('machine'))
