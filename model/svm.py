import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  svm
from sklearn.metrics import accuracy_score
import os
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
import warnings
dir1='/Users/leung/Desktop/Junior_1/NLP' 
os.chdir(dir1)

zip_fn ='NLP_057-Final-Project.zip'

grp1 = [ 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x']
grp2 = ['misc.forsale']

grp3 = [ 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc']


grp4 = [  'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey']
   
grp5 =['alt.atheism',
 'soc.religion.christian',
 'talk.religion.misc']
grp6 =[ 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space']

all_files = []
with zipfile.ZipFile(zip_fn) as z:
    for filename in z.namelist():
        all_files.append(filename)

train_file =[x for x in all_files if (('NLP_057-Final-Project/preprocessing/corpus/train/' in x) & (x[-4:] =='.txt'))]
train_lvl = [x.split('/train/')[1].split('/')[0] for x in train_file]
x_train =[]
with zipfile.ZipFile(zip_fn) as z:
    for f in train_file:
        file = z.open(f, 'r')
        text = file.read().strip()
        file.close()
        x_train.append(text)

train_grp_lvl = []
for l in train_lvl:
    if l in grp1:
        level = 0
    elif l in grp2:
        level = 1
    elif l in grp3:
        level = 2
    elif l in grp4:
        level = 3        
    elif l in grp5:
        level = 4   
    else:
        level = 5
    train_grp_lvl.append(level)


vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(x_train)
vectors_train.shape

dev_file =[x for x in all_files if (('NLP_057-Final-Project/preprocessing/corpus/dev/' in x) & (x[-4:] =='.txt'))]
dev_lvl = [x.split('/dev/')[1].split('/')[0] for x in dev_file]
dev_grp_lvl = []
for l in dev_lvl:
    if l in grp1:
        level = 0
    elif l in grp2:
        level = 1
    elif l in grp3:
        level = 2
    elif l in grp4:
        level = 3        
    elif l in grp5:
        level = 4   
    else:
        level = 5
    dev_grp_lvl.append(level)

x_dev =[]

with zipfile.ZipFile(zip_fn) as z:
    for f in dev_file:
        file = z.open(f, 'r')
        text = file.read().strip()
        file.close()
        x_dev.append(text)
vectors_dev = vectorizer.transform(x_dev)
SVM = svm.SVC()
SVM.fit(vectors_train,train_grp_lvl)
predictions_SVM = SVM.predict(vectors_dev)
print(classification_report(dev_grp_lvl,predictions_SVM)) 

