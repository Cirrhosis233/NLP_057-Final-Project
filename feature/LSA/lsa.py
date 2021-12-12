import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
import os
import zipfile


# This file is an experiment to compare the performance of pure TFIDF and LSA 
# for different Machine Learning models: Logistic regression, SVM, and Random Forest. 

dir1='/Users/leung/Desktop/Junior_1/NLP/NLP_057-Final-Project/feature/word2vec_new/TFIDF_w2v' 
os.chdir(dir1)

# data = pd.read_csv('../../../corpus_new/train_no_num_0.csv')
data = pd.read_csv('../train_cust_1.csv')
test_df = pd.read_csv('../test_cust_1.csv')
# all_text = data.text.append(test_df.text, ignore_index=True)

# %% Generate tf-idf vectorizer
tfidf_model = TfidfVectorizer(min_df=1)
tfidf_model.fit(data['text'])
# tfidf_vocab = set(tfidf_model.get_feature_names_out())
# idf_dict = dict(zip(tfidf_vocab, list(tfidf_model.idf_)))

# %% TF-IDF vector
X = tfidf_model.fit_transform(data['text'])

# Don't touch test_X until the final testing process
test_X = tfidf_model.transform(test_df['text'])
print(np.shape(X))
print(np.shape(test_X))


# %% Split train and dev, LabelEncoder
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(data['class'])
test_Y = labelEncoder.fit_transform(test_df['class'])

# x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2)

# Using test_X and test_Y for final testing !!!
x_train = X
y_train = y
x_dev = test_X
y_dev = test_Y


print(np.shape(x_train))

# tfidf_vec = TfidfVectorizer(use_idf=True, norm='l2')
# transformed_x_train = tfidf_vec.fit_transform(x_train)
# transformed_x_dev = tfidf_vec.transform(x_dev)

# Based on the documentation. dim = 100 is recommanded for LSA
dim = 200
svd = TruncatedSVD(n_components=dim)
print('TF-IDF output shape:', x_train.shape)
x_train_svd = svd.fit_transform(x_train)
x_dev_svd = svd.transform(x_dev)
print('LSA output shape:', x_train_svd.shape)
explained_variance = svd.explained_variance_ratio_.sum()
print("Sum of explained variance ratio: %d%%" % (int(explained_variance * 100)))

# LR model with LSA
lr_model = LogisticRegression(multi_class="multinomial", solver='newton-cg',n_jobs=-1)
lr_model.fit(x_train_svd, y_train)

score = lr_model.score(x_dev_svd, y_dev)
print('LSA Accurary:', score)

predictions_LR = lr_model.predict(x_dev_svd)
print(classification_report(y_dev,predictions_LR))

cv = KFold(n_splits=10, shuffle=True)    
cv_score = cross_val_score(lr_model, x_dev_svd, y_dev, cv=cv, scoring='accuracy')
print("LSA 10-fold CV Accuracy: %0.4f (+/- %0.4f)" % (cv_score.mean(), cv_score.std() * 2))

# ROC AUC score 
y_prob = lr_model.predict_proba(x_dev_svd)

macro_roc_auc_ovo = roc_auc_score(y_dev, y_prob, multi_class="ovo", average="macro")
weighted_roc_auc_ovo = roc_auc_score(
    y_dev, y_prob, multi_class="ovo", average="weighted"
)
macro_roc_auc_ovr = roc_auc_score(y_dev, y_prob, multi_class="ovr", average="macro")
weighted_roc_auc_ovr = roc_auc_score(
    y_dev, y_prob, multi_class="ovr", average="weighted"
)
print(
    "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
)
print(
    "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
)

print('\n------------------------------------------')


# LR model with TF-IDF
lr_model = LogisticRegression(multi_class="multinomial", solver='newton-cg',n_jobs=-1)
lr_model.fit(x_train, y_train)
score = lr_model.score(x_dev, y_dev)
print('TF-IDF Accurary:', score)
predictions_LR = lr_model.predict(x_dev)
print(classification_report(y_dev,predictions_LR))

cv = KFold(n_splits=10, shuffle=True)    
cv_score = cross_val_score(lr_model, x_dev, y_dev, cv=cv, scoring='accuracy')
print("TF-IDF 10-fold CV Accuracy: %0.4f (+/- %0.4f)" % (cv_score.mean(), cv_score.std() * 2))

# ROC AUC score 
y_prob = lr_model.predict_proba(x_dev)

macro_roc_auc_ovo = roc_auc_score(y_dev, y_prob, multi_class="ovo", average="macro")
weighted_roc_auc_ovo = roc_auc_score(
    y_dev, y_prob, multi_class="ovo", average="weighted"
)
macro_roc_auc_ovr = roc_auc_score(y_dev, y_prob, multi_class="ovr", average="macro")
weighted_roc_auc_ovr = roc_auc_score(
    y_dev, y_prob, multi_class="ovr", average="weighted"
)
print(
    "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
)
print(
    "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
)


print('\n------------------------------------------')

# SVM model with LSA
print(np.shape(y_dev))
SVM_Classifier = svm.SVC(C=10, kernel='linear',  gamma='auto', decision_function_shape='ovo', probability=True)
SVM_Classifier.fit(x_train_svd, y_train)
score = SVM_Classifier.score(x_dev_svd, y_dev)
print('SVM LSA Accurary:', score)
predictions_SVM = SVM_Classifier.predict(x_dev_svd)
print(classification_report(y_dev,predictions_SVM)) 

cv = KFold(n_splits=10, shuffle=True)    
cv_score = cross_val_score(SVM_Classifier, x_dev_svd, y_dev, cv=cv, scoring='accuracy')
print("SVM LSA 10-fold CV Accuracy: %0.4f (+/- %0.4f)" % (cv_score.mean(), cv_score.std() * 2))

# ROC AUC score 
y_prob = SVM_Classifier.predict_proba(x_dev_svd)

macro_roc_auc_ovo = roc_auc_score(y_dev, y_prob, multi_class="ovo", average="macro")
weighted_roc_auc_ovo = roc_auc_score(
    y_dev, y_prob, multi_class="ovo", average="weighted"
)
macro_roc_auc_ovr = roc_auc_score(y_dev, y_prob, multi_class="ovr", average="macro")
weighted_roc_auc_ovr = roc_auc_score(
    y_dev, y_prob, multi_class="ovr", average="weighted"
)
print(
    "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
)
print(
    "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
)

print('\n------------------------------------------')

# SVM model with TF-IDF
SVM_Classifier = svm.SVC(C=10, kernel='linear',  gamma='auto', decision_function_shape='ovo', probability=True)
SVM_Classifier.fit(x_train, y_train)
score = SVM_Classifier.score(x_dev, y_dev)
print('SVM TF-IDF Accurary:', score)
predictions_SVM = SVM_Classifier.predict(x_dev)
print(classification_report(y_dev,predictions_SVM)) 

cv = KFold(n_splits=10, shuffle=True)    
cv_score = cross_val_score(SVM_Classifier, x_dev, y_dev, cv=cv, scoring='accuracy')
print("SVM TF-IDF 10-fold CV Accuracy: %0.4f (+/- %0.4f)" % (cv_score.mean(), cv_score.std() * 2))

# ROC AUC score 
y_prob = SVM_Classifier.predict_proba(x_dev)

macro_roc_auc_ovo = roc_auc_score(y_dev, y_prob, multi_class="ovo", average="macro")
weighted_roc_auc_ovo = roc_auc_score(
    y_dev, y_prob, multi_class="ovo", average="weighted"
)
macro_roc_auc_ovr = roc_auc_score(y_dev, y_prob, multi_class="ovr", average="macro")
weighted_roc_auc_ovr = roc_auc_score(
    y_dev, y_prob, multi_class="ovr", average="weighted"
)
print(
    "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
)
print(
    "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
)

print('\n------------------------------------------')

# # RF model with LSA
# print(np.shape(y_dev))
# rf = RandomForestClassifier(n_estimators=837, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=False)
# rf.fit(x_train_svd, y_train)
# score = rf.score(x_dev_svd, y_dev)
# print('RF LSA Accurary:', score)
# predictions_SVM = rf.predict(x_dev_svd)
# print(classification_report(y_dev,predictions_SVM)) 

# cv = KFold(n_splits=10, shuffle=True)    
# cv_score = cross_val_score(rf, x_dev_svd, y_dev, cv=cv, scoring='accuracy')
# print("RF LSA 10-fold CV Accuracy: %0.4f (+/- %0.4f)" % (cv_score.mean(), cv_score.std() * 2))

# # RF model with TF-IDF
# rf = RandomForestClassifier(n_estimators=837, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=False)
# rf.fit(x_train, y_train)
# score = rf.score(x_dev, y_dev)
# print('RF TF-IDF Accurary:', score)
# predictions_SVM = rf.predict(x_dev)
# print(classification_report(y_dev,predictions_SVM)) 

#     # 10-fold cv below takes a very long time to run. 
#     # You can comment the below code when doing repeated experiment
#     # This also indicates that LSA can save a lot of run time for complex ML model. 
# cv = KFold(n_splits=10, shuffle=True)    
# cv_score = cross_val_score(rf, x_dev, y_dev, cv=cv, scoring='accuracy')
# print("RF TF-IDF 10-fold CV Accuracy: %0.4f (+/- %0.4f)" % (cv_score.mean(), cv_score.std() * 2))