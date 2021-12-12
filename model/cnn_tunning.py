import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csr import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer

import os

# This file was written by Zhenyuan Liang


# For padding, parameter setting, model builindg, compiling and training, refer to 
# https://www.kaggle.com/faressayah/20-news-groups-classification-prediction-cnns 
# written by FARES SAYAH

# obtain data and do LSA dimension reduction
dir1='/Users/leung/Desktop/Junior_1/NLP/NLP_057-Final-Project/model' 
os.chdir(dir1)


data = pd.read_csv('../corpus_new/train_cust_1.csv')
test_df = pd.read_csv('../corpus_new/test_cust_1.csv')
all_text = data.text.append(test_df.text, ignore_index=True)

# %% Generate tf-idf vectorizer
# tfidf_model = TfidfVectorizer(min_df=1)
# tfidf_model.fit(all_text)
# tfidf_vocab = set(tfidf_model.get_feature_names_out())
# idf_dict = dict(zip(tfidf_vocab, list(tfidf_model.idf_)))

# %% TF-IDF vector
X = data.text

# Don't touch test_X until the final testing process
test_X = test_df.text



# %% Split train and dev, LabelEncoder


labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(data['class'])


x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2)

# Tokenization
tokenizer = Tokenizer(num_words=x_train.shape[0])
tokenizer.fit_on_texts(list(x_train))
x_train = tokenizer.texts_to_sequences(x_train)
x_dev = tokenizer.texts_to_sequences(x_dev)
# test_X = tokenizer.texts_to_sequences(test_X)
# ————————————————
# Refer to CSDN for this tokenizer code only: 
# 版权声明：本文为CSDN博主「ZesenChen」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/ZesenChen/article/details/84347553

print('\n-------------------------------------------------------')


train_inputs = x_train
test_inputs = x_dev


# train_inputs = csr_matrix.todense(train_inputs).tolist()
# test_inputs = csr_matrix.todense(test_inputs).tolist()

print(type(train_inputs))
print(type(y_train))

# Padding 

MAX_LEN = max([len(sentence) for sentence in train_inputs])
train_inputs = tf.keras.preprocessing.sequence.pad_sequences(train_inputs,
                                                             value=0,
                                                             padding="post",
                                                             maxlen=MAX_LEN)

test_inputs = tf.keras.preprocessing.sequence.pad_sequences(test_inputs,
                                                            value=0,
                                                            padding="post",
                                                            maxlen=MAX_LEN)

print('reach')
print(type(train_inputs))
print(type(y_train))

# Build CNN Model
class DCNN(tf.keras.Model):
    
    def __init__(self, vocab_size, emb_dim, nb_filters, FFN_units, nb_classes,
                 dropout_rate, training=False, name="dcnn"):
        super(DCNN, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocab_size, emb_dim)
        self.bigram = layers.Conv1D(filters=nb_filters, kernel_size=2, padding="valid", activation="relu")
        self.trigram = layers.Conv1D(filters=nb_filters, kernel_size=3, padding="valid", activation="relu")
        self.fourgram = layers.Conv1D(filters=nb_filters, kernel_size=4, padding="valid", activation="relu")
        self.pool = layers.GlobalMaxPool1D() # no training variable so we can
                                             # use the same layer for each
                                             # pooling step
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1, activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes, activation="softmax")
        
    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)
        
        merged = tf.concat([x_1, x_2, x_3], axis=-1) # (batch_size, 3 * nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)
        
        return output


# Setting Parameters
# hp = kt.HyperParameters()

# VOCAB_SIZE = np.shape(x_train)[0]

# EMB_DIM = hp.Int("dim", min_value=50, max_value=500, step=32)
# NB_FILTERS = hp.Int("units", min_value=50, max_value=500, step=32)
# FFN_UNITS = hp.Int("units", min_value=32, max_value=512, step=32)
# NB_CLASSES = len(set(y_train))

# DROPOUT_RATE = hp.Float("dr", min_value=0.1, max_value=0.9, sampling="log")

# BATCH_SIZE = 32
# NB_EPOCHS = 5


#  Compile and Train the Model
def buildModel(hp):

    VOCAB_SIZE = np.shape(x_train)[0]

    EMB_DIM = hp.Int("dim", min_value=50, max_value=500, step=32)
    NB_FILTERS = hp.Int("filters", min_value=50, max_value=500, step=32)
    FFN_UNITS = hp.Int("units", min_value=32, max_value=512, step=32)
    NB_CLASSES = len(set(y_train))

    DROPOUT_RATE = hp.Float("dr", min_value=0.1, max_value=0.9, sampling="log")

    BATCH_SIZE = 32
    NB_EPOCHS = 5
    Dcnn = DCNN(vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, nb_filters=NB_FILTERS,
                FFN_units=FFN_UNITS, nb_classes=NB_CLASSES,
                dropout_rate=DROPOUT_RATE)

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    Dcnn.compile(loss="sparse_categorical_crossentropy",
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics=["sparse_categorical_accuracy"])
    return Dcnn


# print(type(train_inputs))
# print(type(y_train))
rows_len = np.shape(x_train)[0]

for i in range(rows_len):
    x_train[i] = np.sort(x_train[i])

# Dcnn.fit(train_inputs,
#          y_train,
#          batch_size=BATCH_SIZE,
#          epochs=NB_EPOCHS)

# results = Dcnn.evaluate(test_inputs, y_dev, batch_size=BATCH_SIZE)
# print(results)

hp = kt.HyperParameters()
buildModel(hp)

# Random Search for Hyperparameter Tunning. Inspired by the example in keras documentation
tuner = kt.RandomSearch(
    hypermodel= buildModel,
    objective="val_sparse_categorical_accuracy",
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
    )
# You can print a summary of the search space:
print(tuner.search_space_summary())
print('reach')
# The call to search has the same signature as model.fit()
tuner.search(train_inputs, y_train, epochs=3, validation_data=(test_inputs, y_dev))
# When search is over, you can retrieve the best model(s):
models = tuner.get_best_models(num_models=2)
print(models)
# Or print a summary of the results:
print(tuner.results_summary())