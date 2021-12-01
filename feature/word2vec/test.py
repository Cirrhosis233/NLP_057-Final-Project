import numpy as np
import gensim.models as gm

word_list = np.load("./train_word_list.npy", allow_pickle=True)
doc = gm.doc2vec.TaggedDocument(word_list)
doc2vec_model = gm.Doc2Vec(doc, vector_size=200, dm=0, hs=1, window=8, min_count=1)

