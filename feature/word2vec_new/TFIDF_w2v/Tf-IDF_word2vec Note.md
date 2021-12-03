# TF-IDF_word2vec Note



## window size = 9

### LR:

```
Accuracy: 0.83
Auc: 0.96
Detail:
               precision    recall  f1-score   support

     computer       0.86      0.89      0.88       953
miscellaneous       0.74      0.57      0.64       191
     politics       0.78      0.82      0.80       510
   recreation       0.86      0.88      0.87       751
     religion       0.85      0.79      0.82       478
      science       0.79      0.80      0.79       764

     accuracy                           0.83      3647
    macro avg       0.81      0.79      0.80      3647
 weighted avg       0.83      0.83      0.83      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\tfidf_9_lr_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\tfidf_9_lr_fig.png)

### SVM:

```
Accuracy: 0.85
Auc: 0.97
Detail:
               precision    recall  f1-score   support

     computer       0.87      0.93      0.90       953
miscellaneous       0.80      0.64      0.71       191
     politics       0.77      0.84      0.80       510
   recreation       0.88      0.88      0.88       751
     religion       0.88      0.79      0.83       478
      science       0.84      0.81      0.82       764

     accuracy                           0.85      3647
    macro avg       0.84      0.81      0.82      3647
 weighted avg       0.85      0.85      0.85      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\tfidf_9_svm_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\tfidf_9_svm_fig.png)



## window size = 38

### LR:

```
Accuracy: 0.84
Auc: 0.97
Detail:
               precision    recall  f1-score   support

     computer       0.87      0.90      0.89       953
miscellaneous       0.75      0.61      0.67       191
     politics       0.78      0.82      0.80       510
   recreation       0.87      0.88      0.87       751
     religion       0.85      0.82      0.84       478
      science       0.81      0.80      0.80       764

     accuracy                           0.84      3647
    macro avg       0.82      0.80      0.81      3647
 weighted avg       0.84      0.84      0.84      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\tfidf_38_lr_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\tfidf_38_lr_fig.png)

### SVM:

```
Accuracy: 0.86
Auc: 0.97
Detail:
               precision    recall  f1-score   support

     computer       0.88      0.93      0.90       953
miscellaneous       0.81      0.66      0.73       191
     politics       0.79      0.85      0.82       510
   recreation       0.89      0.88      0.89       751
     religion       0.87      0.81      0.84       478
      science       0.84      0.83      0.84       764

     accuracy                           0.86      3647
    macro avg       0.85      0.83      0.84      3647
 weighted avg       0.86      0.86      0.86      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\tfidf_38_svm_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\tfidf_38_svm_fig.png)



## window size = 75

### LR:

```
Accuracy: 0.84
Auc: 0.97
Detail:
               precision    recall  f1-score   support

     computer       0.87      0.90      0.89       953
miscellaneous       0.75      0.62      0.68       191
     politics       0.78      0.82      0.80       510
   recreation       0.86      0.87      0.87       751
     religion       0.86      0.81      0.84       478
      science       0.81      0.80      0.81       764

     accuracy                           0.84      3647
    macro avg       0.82      0.80      0.81      3647
 weighted avg       0.84      0.84      0.84      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\tfidf_75_lr_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\tfidf_75_lr_fig.png)

### SVM:

```
Accuracy: 0.85
Auc: 0.97
Detail:
               precision    recall  f1-score   support

     computer       0.87      0.92      0.89       953
miscellaneous       0.83      0.67      0.74       191
     politics       0.78      0.83      0.81       510
   recreation       0.90      0.89      0.89       751
     religion       0.87      0.80      0.83       478
      science       0.83      0.82      0.82       764

     accuracy                           0.85      3647
    macro avg       0.85      0.82      0.83      3647
 weighted avg       0.85      0.85      0.85      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\tfidf_75_svm_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\tfidf_75_svm_fig.png)

