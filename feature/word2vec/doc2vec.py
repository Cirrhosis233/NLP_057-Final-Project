import os
import numpy as np
from gensim.models import Doc2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report
from tqdm import tqdm


train_path = "..\\..\\corpus_no_num\\train"
dev_path = "..\\..\\corpus_no_num\\dev"
test_path = "..\\..\\corpus_no_num\\test"

npy_dev = []
for root, dirs, files in os.walk(dev_path):
    for f in files:
        npy_dev.append(os.path.join(root, f))
npy_train = []
for root, dirs, files in os.walk(train_path):
    for f in files:
        npy_train.append(os.path.join(root, f))
npy_test = []
for root, dirs, files in os.walk(test_path):
    for f in files:
        npy_test.append(os.path.join(root, f))


def get_docVector(doc, doc2vec_model):
    cutWords = [word for sent in doc for word in sent]
    i = 0
    word_set = set(doc2vec_model.wv.index_to_key)
    article_vector = np.zeros((doc2vec_model.layer1_size))
    for cutWord in cutWords:
        if cutWord in word_set:
            article_vector = np.add(
                article_vector, doc2vec_model.wv.get_vector(cutWord))
            i += 1
    if i == 0:
        return article_vector
    cutWord_vector = np.divide(article_vector, i)
    return cutWord_vector


if __name__ == "__main__":
    # ==============================================================
    # * doc2vec_model_train.d2v generate
    word_list = np.load("./train_word_list.npy", allow_pickle=True)
    doc2vec_model = Doc2Vec(word_list, vector_size=200, dm=0, hs=1, window=8, min_count=1)
    # doc2vec_model.build_vocab([x for x in tqdm(word_list)])
    # doc2vec_model = doc2vec_model.train([x for x in tqdm(word_list)], total_examples=len(word_list), epochs=10)
    doc2vec_model.save('doc2vec_model.w2v')
    # ==============================================================
    # * doc2vec_model test
    doc2vec_model = Doc2Vec.load("./doc2vec_model.w2v")
    print(doc2vec_model.infer_vector(['soccer']))
    # # ==============================================================
    # # * Generate y (label list) for train and dev and test set
    # # class_dev = [f.split('\\')[-3] for f in npy_dev]  #! DEV
    # class_train = [f.split('\\')[-3] for f in npy_train]
    # class_test = [f.split('\\')[-3] for f in npy_test]  #! TEST
    # labelEncoder = LabelEncoder()
    # train_y = labelEncoder.fit_transform(class_train)
    # # dev_y = labelEncoder.fit_transform(class_dev)  #! DEV
    # test_y = labelEncoder.fit_transform(class_test)  #! TEST
    # np.save("train_y.npy", train_y)
    # # np.save("dev_y.npy", dev_y)  #! DEV
    # np.save("test_y.npy", test_y)  #! TEST
    # # print(train_y)
    # # ==============================================================
    # # * Generate X for train and dev and test set
    # doc2vec_model = Doc2Vec.load("./doc2vec_model.w2v")
    # # dev_X = [get_docVector(np.load(f, allow_pickle=True), doc2vec_model) for f in npy_dev]  #! DEV
    # # np.save("dev_X.npy", dev_X)  #! DEV
    # test_X = [get_docVector(np.load(f, allow_pickle=True), doc2vec_model) for f in npy_test]  #! TEST
    # np.save("test_X.npy", test_X)  #! TEST
    # train_X = [get_docVector(np.load(f, allow_pickle=True), doc2vec_model) for f in npy_train]
    # np.save("train_X.npy", train_X)
    # # ==============================================================
    # # * Train and Logistic model
    # train_X = np.load("./train_X.npy")
    # train_y = np.load("./train_y.npy")
    # # dev_X = np.load("./dev_X.npy")
    # # dev_y = np.load("./dev_y.npy")
    # # test_X = np.load("./test_X.npy")
    # # test_y = np.load("./test_y.npy")
    # logistic_model = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=1000)
    # logistic_model.fit(train_X, train_y)
    # # print(logistic_model.score(dev_X, dev_y))
    # joblib.dump(logistic_model, 'logistic.model')
    # # ==============================================================
    # # * Evaluation Logistic model
    # logistic_model = joblib.load('logistic.model')
    # labelEncoder = LabelEncoder()
    # # class_dev = [f.split('\\')[-3] for f in npy_dev]
    # # dev_X = np.load("./dev_X.npy")
    # # dev_y = labelEncoder.fit_transform(class_dev)
    # # y_pred = logistic_model.predict(dev_X)
    # class_test = [f.split('\\')[-3] for f in npy_test]
    # test_X = np.load("./test_X.npy")
    # test_y = labelEncoder.fit_transform(class_test)
    # y_pred = logistic_model.predict(test_X)
    # print(labelEncoder.inverse_transform([[x] for x in range(6)]))
    # # print(classification_report(dev_y, y_pred))
    # print(classification_report(test_y, y_pred))
    # # ==============================================================
    # # * Train and SVM model
    # train_X = np.load("./train_X.npy")
    # train_y = np.load("./train_y.npy")
    # # dev_X = np.load("./dev_X.npy")
    # # dev_y = np.load("./dev_y.npy")
    # # test_X = np.load("./test_X.npy")
    # # test_y = np.load("./test_y.npy")
    # svm_ovo = SVC(C=10, kernel="rbf", decision_function_shape="ovo", probability=True)
    # svm_ovo.fit(train_X, train_y)
    # # print(logistic_model.score(dev_X, dev_y))
    # joblib.dump(svm_ovo, 'SVM.model')
    # # ==============================================================
    # # * Evaluation SVM model
    # svm_ovo = joblib.load('SVM.model')
    # labelEncoder = LabelEncoder()
    # # class_dev = [f.split('\\')[-3] for f in npy_dev]
    # # dev_X = np.load("./dev_X.npy")
    # # dev_y = labelEncoder.fit_transform(class_dev)
    # # y_pred = svm_ovo.predict(dev_X)
    # class_test = [f.split('\\')[-3] for f in npy_test]
    # test_X = np.load("./test_X.npy")
    # test_y = labelEncoder.fit_transform(class_test)
    # y_pred = svm_ovo.predict(test_X)
    # print(labelEncoder.inverse_transform([[x] for x in range(6)]))
    # # print(classification_report(dev_y, y_pred))
    # print(classification_report(test_y, y_pred))