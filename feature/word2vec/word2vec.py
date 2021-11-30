import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report

train_path = "..\\..\\corpus_no_num\\train"
dev_path = "..\\..\\corpus_no_num\\dev"
# test_path = "..\\..\\corpus\\test"
npy_dev = []
for root, dirs, files in os.walk(dev_path):
    for f in files:
        npy_dev.append(os.path.join(root, f))
npy_train = []
for root, dirs, files in os.walk(train_path):
    for f in files:
        npy_train.append(os.path.join(root, f))


def get_docVector(doc, word2vec_model):
    cutWords = [word for sent in doc for word in sent]
    i = 0
    word_set = set(word2vec_model.wv.index_to_key)
    article_vector = np.zeros((word2vec_model.layer1_size))
    for cutWord in cutWords:
            if cutWord in word_set:
                    article_vector = np.add(article_vector, word2vec_model.wv.get_vector(cutWord))
                    i += 1
    if i == 0:
        return article_vector
    cutWord_vector = np.divide(article_vector, i)
    return cutWord_vector


if __name__ == "__main__":
    #==============================================================
    #* train_word_list.npy generate
    # res = []
    # start = True
    # temp = []
    # for f in npy_train:
    #     doc = np.load(f, allow_pickle=True)
    #     if doc.ndim != 1:
    #         temp.append(doc.tolist()[0])
    #         continue
    #     if start:
    #         res = doc
    #         # print(np.shape(res))
    #         start = False
    #         continue
    #     res = np.append(res, doc, axis=0)
    # temp = np.array(temp, dtype=object)
    # res = np.append(res, temp, axis=0)
    # np.save("train_word_list.npy", res)
    #==============================================================
    #* word_list.npy generate
    # res = np.load("train_word_list.npy", allow_pickle=True)
    # temp = []
    # for f in npy_dev:
    #     doc = np.load(f, allow_pickle=True)
    #     if doc.ndim != 1:
    #         temp.append(doc.tolist()[0])
    #         continue
    #     res = np.append(res, doc, axis=0)
    # temp = np.array(temp, dtype=object)
    # res = np.append(res, temp, axis=0)
    # np.save("word_list.npy", res)
    #==============================================================
    #* word2vec_model_train.w2v generate
    # word_list = np.load("./train_word_list.npy", allow_pickle=True)
    # word2vec_model = Word2Vec(word_list)
    # word2vec_model.save('word2vec_model.w2v')
    #==============================================================
    #* word2vec_model.w2v generate
    # word_list = np.load("./word_list.npy", allow_pickle=True)
    # word2vec_model = Word2Vec(word_list, vector_size=300, sg=1, min_count=1)
    # word2vec_model.save('word2vec_model.w2v')
    #==============================================================
    #* word2vec_model test
    # word2vec_model = Word2Vec.load("./word2vec_model.w2v")
    # print(word2vec_model.wv.most_similar('soccer'))
    #==============================================================
    #* Generate y (label list) for train and dev set
    # class_dev = [f.split('\\')[-3] for f in npy_dev]
    # class_train = [f.split('\\')[-3] for f in npy_train]
    # labelEncoder = LabelEncoder()
    # train_y = labelEncoder.fit_transform(class_train)
    # dev_y = labelEncoder.fit_transform(class_dev)
    # np.save("train_y.npy", train_y)
    # np.save("dev_y.npy", dev_y)
    # print(train_y)
    #==============================================================
    #* Generate X for train and dev set
    # word2vec_model = Word2Vec.load("./word2vec_model.w2v")
    # dev_X = [get_docVector(np.load(f, allow_pickle=True), word2vec_model) for f in npy_dev]
    # np.save("dev_X.npy", dev_X)
    # train_X = [get_docVector(np.load(f, allow_pickle=True), word2vec_model) for f in npy_train]
    # np.save("train_X.npy", train_X)
    #==============================================================
    #* Train and Logistic model
    # train_X = np.load("./train_X.npy")
    # train_y = np.load("./train_y.npy")
    # dev_X = np.load("./dev_X.npy")
    # dev_y = np.load("./dev_y.npy")
    # logistic_model = LogisticRegression(max_iter=1000)
    # logistic_model.fit(train_X, train_y)
    # print(logistic_model.score(dev_X, dev_y))
    # joblib.dump(logistic_model, 'logistic.model')
    #==============================================================
    #* Evaluation Logistic model
    logistic_model = joblib.load('logistic.model')
    labelEncoder = LabelEncoder()
    class_dev = [f.split('\\')[-3] for f in npy_dev]
    dev_X = np.load("./dev_X.npy")
    dev_y = labelEncoder.fit_transform(class_dev)
    y_pred = logistic_model.predict(dev_X)
    print(labelEncoder.inverse_transform([[x] for x in range(6)]))
    print(classification_report(dev_y, y_pred))
    #==============================================================
    #* Train and Logistic model
    # train_X = np.load("./train_X.npy")
    # train_y = np.load("./train_y.npy")
    # dev_X = np.load("./dev_X.npy")
    # dev_y = np.load("./dev_y.npy")
    # logistic_model = LogisticRegression(max_iter=1000)
    # logistic_model.fit(train_X, train_y)
    # print(logistic_model.score(dev_X, dev_y))
    # joblib.dump(logistic_model, 'logistic.model')
