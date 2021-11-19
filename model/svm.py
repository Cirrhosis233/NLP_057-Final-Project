#!/usr/bin/env python
# coding: utf-8

# In[107]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  svm
from sklearn.metrics import accuracy_score
import os
import zipfile


# In[108]:


all_files = []
with zipfile.ZipFile('NLP_057-Final-Project-main.zip') as z:
    for filename in z.namelist():
        all_files.append(filename)


# In[109]:


train_file =[x for x in all_files if (('NLP_057-Final-Project-main/preprocessing/corpus/train/' in x) & (x[-4:] =='.txt'))]
len(train_file)

test_file =[x for x in all_files if (('NLP_057-Final-Project-main/preprocessing/corpus/test/' in x) & (x[-4:] =='.txt'))]
len(test_file)
train_lvl = [x.split('/train/')[1].split('/')[0] for x in train_file]
test_lvl = [x.split('/test/')[1].split('/')[0] for x in test_file]
cats = ['alt.atheism', 'sci.space']


# In[110]:


train_indx = [i for i, val in enumerate(train_lvl) if val in cats]
test_indx = [i for i, val in enumerate(test_lvl) if val in cats]


# In[111]:


train_lvl1 =  [train_lvl[i] for i in train_indx]
train_file1 =  [train_file[i] for i in train_indx]

test_lvl1 =  [test_lvl[i] for i in test_indx]
test_file1 =  [test_file[i] for i in test_indx]

x_train =[]
with zipfile.ZipFile('NLP_057-Final-Project-main.zip') as z:
    for f in train_file1:
        file = z.open(f, 'r')
        text = file.read().strip()
        file.close()
        x_train.append(text)

x_test =[]
with zipfile.ZipFile('NLP_057-Final-Project-main.zip') as z:
    for f in test_file1:
        file = z.open(f, 'r')
        text = file.read().strip()
        file.close()
        x_test.append(text)        
        
y_train= [0 if x == 'alt.atheism'  else 1 for x in train_lvl1]
y_test = [0 if x == 'alt.atheism'  else 1 for x in test_lvl1]        


# In[112]:



vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf =True, stop_words = 'english')
train_corpus_tf_idf = vectorizer.fit_transform(x_train)
test_corpus_tf_idf = vectorizer.transform(x_test)


# In[113]:



SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(train_corpus_tf_idf,y_train)
predictions_SVM = SVM.predict(test_corpus_tf_idf)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)


# In[ ]:




