import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score
import os

# This file is an experiment to measure the time saving by LSA for SVM. 

dir1='/Users/leung/Desktop/Junior_1/NLP/NLP_057-Final-Project/feature/word2vec_new/TFIDF_w2v' 
os.chdir(dir1)

# data = pd.read_csv('../../../corpus_new/train_no_num_0.csv')
data = pd.read_csv('../train_cust_1.csv')
test_df = pd.read_csv('../test_cust_1.csv')
all_text = data.text.append(test_df.text, ignore_index=True)

# %% Generate tf-idf vectorizer
tfidf_model = TfidfVectorizer(min_df=1)
tfidf_model.fit(all_text)
# tfidf_vocab = set(tfidf_model.get_feature_names_out())
# idf_dict = dict(zip(tfidf_vocab, list(tfidf_model.idf_)))

# %% TF-IDF vector
X = tfidf_model.fit_transform(data.text)

# Don't touch test_X until the final testing process
test_X = tfidf_model.fit_transform(test_df.text)



# %% Split train and dev, LabelEncoder
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(data['class'])


x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2)

print(np.shape(x_train))

# tfidf_vec = TfidfVectorizer(use_idf=True, norm='l2')
# transformed_x_train = tfidf_vec.fit_transform(x_train)
# transformed_x_dev = tfidf_vec.transform(x_dev)

# dim is a hyperparameter.
dim = 200
svd = TruncatedSVD(n_components=dim)
print('TF-IDF output shape:', x_train.shape)
x_train_svd = svd.fit_transform(x_train)
x_dev_svd = svd.transform(x_dev)
print('LSA output shape:', x_train_svd.shape)
explained_variance = svd.explained_variance_ratio_.sum()
print("Sum of explained variance ratio: %d%%" % (int(explained_variance * 100)))

import time
start_time = time.time()

# SVM model with LSA
print(np.shape(y_dev))
# C=10, kernel='linear',  gamma='auto', decision_function_shape='ovo'
SVM_Classifier = svm.SVC()
SVM_Classifier.fit(x_train_svd, y_train)
predictions_SVM = SVM_Classifier.predict(x_dev_svd)
print(classification_report(y_dev,predictions_SVM)) 

# cv = KFold(n_splits=10, shuffle=True)    
# cv_score = cross_val_score(SVM_Classifier, x_dev_svd, y_dev, cv=cv, scoring='accuracy')
# print("SVM LSA 10-fold CV Accuracy: %0.4f (+/- %0.4f)" % (cv_score.mean(), cv_score.std() * 2))

print("--- %s seconds ---" % (time.time() - start_time))
lsaTime = time.time() - start_time  # Record time spend for lsa

# SVM model with TF-IDF
# C=10, kernel='linear',  gamma='auto', decision_function_shape='ovo'
SVM_Classifier = svm.SVC()
SVM_Classifier.fit(x_train, y_train)
predictions_SVM = SVM_Classifier.predict(x_dev)
print(classification_report(y_dev,predictions_SVM)) 

# cv = KFold(n_splits=10, shuffle=True)    
# cv_score = cross_val_score(SVM_Classifier, x_dev, y_dev, cv=cv, scoring='accuracy')
# print("SVM TF-IDF 10-fold CV Accuracy: %0.4f (+/- %0.4f)" % (cv_score.mean(), cv_score.std() * 2))

print("--- %s seconds ---" % (time.time() - start_time))
tfidfTime = time.time() - start_time    #record time spent for tfidf
diff = tfidfTime - lsaTime

print("The time LSA saves is: %s" % diff)