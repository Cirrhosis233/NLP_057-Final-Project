# IDF_word2vec Note



## window size = 9

### LR:

```
Accuracy: 0.83
Auc: 0.96
Detail:
               precision    recall  f1-score   support

     computer       0.87      0.90      0.89       953
miscellaneous       0.74      0.63      0.68       191
     politics       0.77      0.81      0.79       510
   recreation       0.87      0.88      0.87       751
     religion       0.83      0.78      0.80       478
      science       0.80      0.79      0.80       764

     accuracy                           0.83      3647
    macro avg       0.81      0.80      0.80      3647
 weighted avg       0.83      0.83      0.83      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\idf_9_lr_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\idf_9_lr_fig.png)

### SVM:

```
Accuracy: 0.86
Auc: 0.98
Detail:
               precision    recall  f1-score   support

     computer       0.88      0.94      0.91       953
miscellaneous       0.84      0.68      0.75       191
     politics       0.77      0.85      0.81       510
   recreation       0.89      0.90      0.89       751
     religion       0.89      0.79      0.84       478
      science       0.85      0.82      0.83       764

     accuracy                           0.86      3647
    macro avg       0.85      0.83      0.84      3647
 weighted avg       0.86      0.86      0.86      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\idf_9_svm_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\idf_9_svm_fig.png)



## window size = 38

### LR:

```
Accuracy: 0.84
Auc: 0.97
Detail:
               precision    recall  f1-score   support

     computer       0.88      0.89      0.88       953
miscellaneous       0.67      0.63      0.65       191
     politics       0.78      0.81      0.80       510
   recreation       0.88      0.88      0.88       751
     religion       0.85      0.84      0.84       478
      science       0.82      0.81      0.81       764

     accuracy                           0.84      3647
    macro avg       0.81      0.81      0.81      3647
 weighted avg       0.84      0.84      0.84      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\idf_38_lr_fig.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\idf_38_lr_fig.png)

### SVM:

```
Accuracy: 0.87
Auc: 0.98
Detail:
               precision    recall  f1-score   support

     computer       0.88      0.93      0.91       953
miscellaneous       0.81      0.70      0.75       191
     politics       0.79      0.85      0.82       510
   recreation       0.91      0.89      0.90       751
     religion       0.89      0.83      0.86       478
      science       0.87      0.84      0.86       764

     accuracy                           0.87      3647
    macro avg       0.86      0.84      0.85      3647
 weighted avg       0.87      0.87      0.87      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\idf_38_svm_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\idf_38_svm_fig.png)



## window size = 75

### LR:

```
Accuracy: 0.84
Auc: 0.96
Detail:
               precision    recall  f1-score   support

     computer       0.87      0.90      0.88       953
miscellaneous       0.72      0.64      0.68       191
     politics       0.77      0.82      0.79       510
   recreation       0.88      0.87      0.88       751
     religion       0.84      0.81      0.82       478
      science       0.81      0.80      0.81       764

     accuracy                           0.84      3647
    macro avg       0.82      0.81      0.81      3647
 weighted avg       0.84      0.84      0.84      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\idf_75_lr_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\idf_75_lr_fig.png)

### SVM:

```
Accuracy: 0.87
Auc: 0.98
Detail:
               precision    recall  f1-score   support

     computer       0.88      0.93      0.91       953
miscellaneous       0.84      0.70      0.77       191
     politics       0.80      0.86      0.83       510
   recreation       0.89      0.89      0.89       751
     religion       0.89      0.82      0.85       478
      science       0.86      0.83      0.85       764

     accuracy                           0.87      3647
    macro avg       0.86      0.84      0.85      3647
 weighted avg       0.87      0.87      0.87      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\idf_75_svm_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\TFIDF_w2v\idf_75_svm_fig.png)

