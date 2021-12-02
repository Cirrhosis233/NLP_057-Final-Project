# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 22:55:20 2021

@author: Zack
"""

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
# from sklearn.neighbors import KNeighborsClassifier


# data = pd.read_csv('../../../corpus_new/train_no_num_0.csv')
data = pd.read_csv('../train_cust_1.csv')
test_df = pd.read_csv('../test_cust_1.csv')
all_text = data.text.append(test_df.text, ignore_index=True)

# %% Generate tf-idf vectorizer

tfidf_model = TfidfVectorizer(min_df=1)
tfidf_model.fit(all_text)
tfidf_vocab = set(tfidf_model.get_feature_names_out())
idf_dict = dict(zip(tfidf_vocab, list(tfidf_model.idf_)))

# %% TF-IDF vector
X = tfidf_model.fit_transform(data.text)
test_X = tfidf_model.fit_transform(test_df.text)


# %% Generate word list
word_list = []

for doc in data.text:
    tokens = [word for word in doc.split()]
    word_list.append(tokens)
for doc in test_df.text:
    tokens = [word for word in doc.split()]
    word_list.append(tokens)

np.save('word_list.npy', word_list)

word_list_length = []
for i in word_list:
    word_list_length.append(len(i))
print("Avg length:", np.sum(word_list_length)/len(word_list_length))
# print("Suggested window size:", np.sqrt(total_size/14654)/2)


# %% Train word2vec by self
# word_list = np.load("./word_list.npy", allow_pickle=True)
# word2vec_model = Word2Vec(word_list)
word2vec_model = Word2Vec(word_list, vector_size=300,
                          sg=1, hs=1, window=8, min_count=1)
word2vec_model.save('word2vec_model.w2v')


# %% Generate doc vectors based on tf-idf weighted w2v
# word2vec_model = Word2Vec.load("word2vec_model.w2v")

def get_docVector(cutWords, word2vec_model, tfidf_vocab, idf_dict):
    i = 0
    word_set = set(word2vec_model.wv.index_to_key)
    article_vector = np.zeros((word2vec_model.layer1_size))
    for cutWord in cutWords:
        if cutWord in word_set and cutWord in tfidf_vocab:
            w2v_vec = word2vec_model.wv.get_vector(cutWord)
            idf = idf_dict.get(cutWord)
            vec = np.multiply(w2v_vec, idf)
            article_vector = np.add(article_vector, vec)
            i += 1
    if i == 0:
        return article_vector
    cutWord_vector = np.divide(article_vector, i)
    return cutWord_vector

# test = get_docVector(data.text[0].split(), word2vec_model, tfidf_vocab, idf_dict)
X = [get_docVector(doc.split(), word2vec_model, tfidf_vocab, idf_dict) for doc in data.text]

np.save('X.npy', X)


# %% Split train and dev, LabelEncoder
# X = np.load('X.npy')
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(data['class'])
# y = labelEncoder.fit_transform(data['topic'])

train_X, dev_X, train_y, dev_y = train_test_split(X, y, test_size=0.2)


# %% LR
logistic_model = LogisticRegression(
    multi_class="multinomial", solver="newton-cg", max_iter=1000)
logistic_model.fit(train_X, train_y)
joblib.dump(logistic_model, 'logistic.model')


# %% Cross-valid LR
cv_split = ShuffleSplit(n_splits=5, train_size=0.8, test_size=0.2)
score_ndarray = cross_val_score(logistic_model, X, y, cv=cv_split)
print(score_ndarray)
print(score_ndarray.mean())


# %% Report (dev)
y_pred = logistic_model.predict(dev_X)
print(labelEncoder.inverse_transform([[x] for x in range(6)]))
print(classification_report(dev_y, y_pred))


# %% Report (test)
# test_df = pd.read_csv('../../../corpus_new/test_no_num_0.csv')
test_X = [get_docVector(doc.split(), word2vec_model, tfidf_vocab, idf_dict) for doc in test_df.text]
test_y = labelEncoder.transform(test_df['class'])
y_pred = logistic_model.predict(test_X)
print(labelEncoder.inverse_transform([[x] for x in range(6)]))
print(classification_report(test_y, y_pred))


# %% SVM
svm_model = SVC(C=3, kernel="rbf",
                decision_function_shape="ovo", probability=False)
svm_model.fit(train_X, train_y)
joblib.dump(svm_model, 'svm.model')


# %% Cross-valid SVM
cv_split = ShuffleSplit(n_splits=5, train_size=0.8, test_size=0.2)
score_ndarray = cross_val_score(svm_model, X, y, cv=cv_split)
print(score_ndarray)
print(score_ndarray.mean())


# %% Report (dev)
y_pred = svm_model.predict(dev_X)
print(labelEncoder.inverse_transform([[x] for x in range(6)]))
print(classification_report(dev_y, y_pred))


# %% Report (test)
# test_df = pd.read_csv('../../../corpus_new/test_no_num_0.csv')
# test_X = [get_docVector(doc.split(), word2vec_model) for doc in test_df.text]
# test_y = labelEncoder.transform(test_df['class'])
y_pred = svm_model.predict(test_X)
print(labelEncoder.inverse_transform([[x] for x in range(6)]))
print(classification_report(test_y, y_pred))


# %% KNN
# knn_model = KNeighborsClassifier()
# knn_model.fit(train_X, train_y)
# joblib.dump(knn_model, 'knn.model')


# # %% Cross-valid SVM
# cv_split = ShuffleSplit(n_splits=5, train_size=0.8, test_size=0.2)
# score_ndarray = cross_val_score(knn_model, X, y, cv=cv_split)
# print(score_ndarray)
# print(score_ndarray.mean())


# # %% Report (dev)
# y_pred = knn_model.predict(dev_X)
# print(labelEncoder.inverse_transform([[x] for x in range(6)]))
# print(classification_report(dev_y, y_pred))


# # %% Report (test)
# test_df = pd.read_csv('../../../corpus_new/test_no_num_0.csv')
# test_X = [get_docVector(doc.split(), word2vec_model) for doc in test_df.text]
# test_y = labelEncoder.transform(test_df['class'])
# y_pred = knn_model.predict(test_X)
# print(labelEncoder.inverse_transform([[x] for x in range(6)]))
# print(classification_report(test_y, y_pred))