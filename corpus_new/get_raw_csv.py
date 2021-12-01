# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:08:51 2021

@author: Zack
"""
from sklearn.datasets import fetch_20newsgroups


newsgroups_all = fetch_20newsgroups(subset='all',
                                    remove=('headers', 'footers', 'quotes'))

target_names = newsgroups_all.target_names


#%%
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(newsgroups_all.data,
                                                    newsgroups_all.target,
                                                    test_size=0.2,
                                                    stratify=newsgroups_all.target)


#%%

new_class = {
    "computer": {'comp.graphics', 'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
            'comp.windows.x'},
    
    "miscellaneous": {'misc.forsale'},
    
    "politics": {'talk.politics.guns', 'talk.politics.mideast',
            'talk.politics.misc'},
    
    "recreation": {'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
              'rec.sport.hockey'},
    
    "religion": {'alt.atheism', 'soc.religion.christian',
                 'talk.religion.misc'},
    
    "science": {'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'}
    }


train_targetname = [target_names[i] for i in train_target]
train_class = []
for name in train_targetname:
    for key in new_class:
        if name in new_class.get(key):
            train_class.append(key)
test_targetname = [target_names[i] for i in test_target]
test_class = []
for name in test_targetname:
    for key in new_class:
        if name in new_class.get(key):
            test_class.append(key)


#%%
import pandas as pd
    
train_dict = {
    "class": train_class,
    "topic": train_targetname,
    "text": train_data
    }
train_df = pd.DataFrame(train_dict)
train_df.to_csv("train_raw.csv", index=False)

test_dict = {
    "class": test_class,
    "topic": test_targetname,
    "text": test_data
    }
test_df = pd.DataFrame(test_dict)
test_df.to_csv("test_raw.csv", index=False)
    
