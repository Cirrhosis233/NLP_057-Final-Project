
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  svm
from sklearn.metrics import accuracy_score
import os
import zipfile

dir1='D:\\nlp' 
os.chdir(dir1)

def collect_train_test_data():
    zip_fn ='NLP_057-Final-Project-main.zip'
    cats = ['alt.atheism', 'sci.space']
    all_files = []
    with zipfile.ZipFile(zip_fn) as z:
        for filename in z.namelist():
            all_files.append(filename)

    train_file =[x for x in all_files if (('NLP_057-Final-Project-main/preprocessing/corpus/train/' in x) & (x[-4:] =='.txt'))]
    train_lvl = [x.split('/train/')[1].split('/')[0] for x in train_file]
    train_indx = [i for i, val in enumerate(train_lvl) if val in cats]
    train_lvl1 =  [train_lvl[i] for i in train_indx]
    train_file1 =  [train_file[i] for i in train_indx]
    x_train =[]
    with zipfile.ZipFile(zip_fn) as z:
        for f in train_file1:
            file = z.open(f, 'r')
            text = file.read().strip()
            file.close()
            x_train.append(text)
    y_train= [0 if x == 'alt.atheism'  else 1 for x in train_lvl1]

    dev_file =[x for x in all_files if (('NLP_057-Final-Project-main/preprocessing/corpus/dev/' in x) & (x[-4:] =='.txt'))]
    dev_lvl = [x.split('/dev/')[1].split('/')[0] for x in dev_file]
    dev_indx = [i for i, val in enumerate(dev_lvl) if val in cats]
    dev_lvl1 =  [dev_lvl[i] for i in dev_indx]
    dev_file1 =  [dev_file[i] for i in dev_indx]
    x_dev =[]
    with zipfile.ZipFile(zip_fn) as z:
        for f in dev_file1:
            file = z.open(f, 'r')
            text = file.read().strip()
            file.close()
            x_dev.append(text)
    y_dev = [0 if x == 'alt.atheism'  else 1 for x in dev_lvl1]
    
    
    test_file =[x for x in all_files if (('NLP_057-Final-Project-main/preprocessing/corpus/test/' in x) & (x[-4:] =='.txt'))]
    test_lvl = [x.split('/test/')[1].split('/')[0] for x in test_file]
    test_indx = [i for i, val in enumerate(test_lvl) if val in cats]
    test_lvl1 =  [test_lvl[i] for i in test_indx]
    test_file1 =  [test_file[i] for i in test_indx]
    x_test =[]
    with zipfile.ZipFile(zip_fn) as z:
        for f in test_file1:
            file = z.open(f, 'r')
            text = file.read().strip()
            file.close()
            x_test.append(text)     
    y_test = [0 if x == 'alt.atheism'  else 1 for x in test_lvl1]
    return x_train, y_train, x_dev, y_dev, x_test, y_test

if __name__ == "__main__":
    x_train, y_train, x_dev, y_dev, x_test, y_test  = collect_train_test_data()
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf =True, stop_words = 'english')
    train_corpus_tf_idf = vectorizer.fit_transform(x_train)
    dev_corpus_tf_idf = vectorizer.transform(x_dev)
    test_corpus_tf_idf = vectorizer.transform(x_test)

    SVM = svm.SVC(C=.5, kernel='linear', gamma='auto')
    SVM.fit(train_corpus_tf_idf,y_train)
    predictions_SVM = SVM.predict(dev_corpus_tf_idf)
    print("SVM Accuracy Score for dev with C = 0.5 -> ",accuracy_score(predictions_SVM, y_dev)*100)

    SVM = svm.SVC(C=1, kernel='linear',  gamma='auto')
    SVM.fit(train_corpus_tf_idf,y_train)
    predictions_SVM = SVM.predict(dev_corpus_tf_idf)
    print("SVM Accuracy Score for dev with C = 1 -> ",accuracy_score(predictions_SVM, y_dev)*100)
    
    SVM = svm.SVC(C=10, kernel='linear',  gamma='auto')
    SVM.fit(train_corpus_tf_idf,y_train)
    predictions_SVM = SVM.predict(dev_corpus_tf_idf)
    print("SVM Accuracy Score for dev with C = 10 -> ",accuracy_score(predictions_SVM, y_dev)*100)
    
    SVM = svm.SVC(C=10, kernel='linear',  gamma='auto')
    SVM.fit(train_corpus_tf_idf,y_train)
    predictions_SVM = SVM.predict(test_corpus_tf_idf)
    print("SVM Accuracy Score for test from the best model -> ",accuracy_score(predictions_SVM, y_test)*100)



