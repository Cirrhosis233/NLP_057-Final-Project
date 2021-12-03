# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:09:44 2021

@author: Zack
"""

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.neighbors import KNeighborsClassifier

#%%
# data = pd.read_csv('../../../corpus_new/train_no_num_0.csv')
data = pd.read_csv('../train_cust_1.csv')
test_df = pd.read_csv('../test_cust_1.csv')
all_text = data.text.append(test_df.text, ignore_index=True)


# %% Generate word list
word_list = []

for doc in all_text:
    tokens = [word for word in doc.split()]
    word_list.append(tokens)

np.save('word_list.npy', word_list)

word_list_length = []
for i in word_list:
    word_list_length.append(len(i))
print("Avg length:", np.sum(word_list_length)/len(word_list_length))


# %% Train word2vec by self
# word_list = np.load("./word_list.npy", allow_pickle=True)
# word2vec_model = Word2Vec(word_list)
word2vec_model = Word2Vec(word_list, vector_size=300,
                          sg=1, hs=1, window=75, min_count=1)
word2vec_model.save('word2vec_model.w2v')


# %% Generate doc vectors
google_model = KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin", binary=True)

#%%
word2vec_model = Word2Vec.load("../w2v_models/75/word2vec_model.w2v")


def get_docVector(cutWords, word2vec_model):
    i = 0
    word_set = set(word2vec_model.wv.index_to_key)
    article_vector = np.zeros((word2vec_model.layer1_size))
    for cutWord in cutWords:
        if cutWord in word_set:
            article_vector = np.add(
                article_vector, word2vec_model.wv.get_vector(cutWord))
            i += 1
    if i == 0:
        return article_vector
    cutWord_vector = np.divide(article_vector, i)
    return cutWord_vector


def get_docVector_pre(cutWords, word2vec_model):
    i = 0
    article_vector = np.zeros(300)
    for cutWord in cutWords:
        if cutWord in word2vec_model:
            article_vector = np.add(
                article_vector, word2vec_model[cutWord])
            i += 1
    if i == 0:
        return article_vector
    cutWord_vector = np.divide(article_vector, i)
    return cutWord_vector


# X = [get_docVector(doc.split(), word2vec_model) for doc in data.text]
# X = [get_docVector_pre(doc.split(), google_model) for doc in data.text]
X = [np.append(get_docVector_pre(doc.split(), google_model), get_docVector(
    doc.split(), word2vec_model)) for doc in data.text]

np.save('X_pre.npy', X)

test_X = [np.append(get_docVector_pre(doc.split(), google_model), get_docVector(
    doc.split(), word2vec_model)) for doc in test_df.text]

np.save('X_test_pre.npy', test_X)


# %% Split train and dev, LabelEncoder
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
# test_X = [np.append(get_docVector_pre(doc.split(), google_model), get_docVector(
#     doc.split(), word2vec_model)) for doc in test_df.text]
# test_y = labelEncoder.transform(test_df['class'])
y_pred = svm_model.predict(test_X)
print(labelEncoder.inverse_transform([[x] for x in range(6)]))
print(classification_report(test_y, y_pred))


# %% Evaluation

# %%
'''
Evaluates a model performance.
:parameter
    :param y_test: array
    :param predicted: array
    :param predicted_prob: array
    :param figsize: tuple - plot setting
'''
def evaluate_multi_classif(y_test, predicted, predicted_prob, figsize=(15,5)):
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values
    
    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob, multi_class="ovo")
    print("Accuracy:",  round(accuracy,2))
    print("Auc:", round(auc,2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))
    
    ## Plot confusion matrix
    # plt.figure(figsize = (20,5))
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, yticklabels=classes, title="Confusion matrix")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.yticks(rotation=0)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i], predicted_prob[:,i])
        ax[0].plot(fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(fpr, tpr)))
    ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], xlabel='False Positive Rate', 
              ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)
    
    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(y_test_array[:,i], predicted_prob[:,i])
        ax[1].plot(recall, precision, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(recall, precision)))
    ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()


# %%
data = pd.read_csv('../train_cust_1.csv')
test_df = pd.read_csv('../test_cust_1.csv')
y = data['class']
test_y = test_df['class']


# %%
X = np.load("../w2v_models/75/X_pre.npy")
test_X = np.load("../w2v_models/75/X_test_pre.npy")


# %% LR
logistic_model = LogisticRegression(
    multi_class="multinomial", solver="newton-cg", max_iter=1000)
logistic_model.fit(X, y)
pred = logistic_model.predict(test_X)
pred_prob = logistic_model.predict_proba(test_X)


# %% SVM
svm_model = SVC(C=3, kernel="rbf",
                decision_function_shape="ovo", probability=True)
svm_model.fit(X, y)
pred = svm_model.predict(test_X)
pred_prob = svm_model.predict_proba(test_X)


# %%
evaluate_multi_classif(test_y, pred, pred_prob)
