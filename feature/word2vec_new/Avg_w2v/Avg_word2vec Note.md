# Avg_word2vec Note



## window size = 9

### LR:

```
Accuracy: 0.84
Auc: 0.97
Detail:
               precision    recall  f1-score   support

     computer       0.88      0.91      0.89       953
miscellaneous       0.80      0.63      0.70       191
     politics       0.78      0.84      0.81       510
   recreation       0.88      0.88      0.88       751
     religion       0.87      0.79      0.83       478
      science       0.80      0.81      0.80       764

     accuracy                           0.84      3647
    macro avg       0.84      0.81      0.82      3647
 weighted avg       0.84      0.84      0.84      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\Avg_w2v\9_lr_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\Avg_w2v\9_lr_fig.png)

### SVM:

```
Accuracy: 0.86
Auc: 0.98
Detail:
               precision    recall  f1-score   support

     computer       0.88      0.94      0.91       953
miscellaneous       0.84      0.69      0.76       191
     politics       0.78      0.85      0.81       510
   recreation       0.90      0.89      0.90       751
     religion       0.89      0.79      0.84       478
      science       0.84      0.82      0.83       764

     accuracy                           0.86      3647
    macro avg       0.85      0.83      0.84      3647
 weighted avg       0.86      0.86      0.86      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\Avg_w2v\9_svm_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\Avg_w2v\9_svm_fig.png)



## window size = 38

### LR:

```
Accuracy: 0.85
Auc: 0.97
Detail:
               precision    recall  f1-score   support

     computer       0.90      0.90      0.90       953
miscellaneous       0.78      0.65      0.71       191
     politics       0.78      0.83      0.80       510
   recreation       0.89      0.88      0.88       751
     religion       0.86      0.83      0.84       478
      science       0.82      0.82      0.82       764

     accuracy                           0.85      3647
    macro avg       0.84      0.82      0.83      3647
 weighted avg       0.85      0.85      0.85      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\Avg_w2v\38_lr_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\Avg_w2v\38_lr_fig.png)

### SVM:

```
Accuracy: 0.87
Auc: 0.98
Detail:
               precision    recall  f1-score   support

     computer       0.89      0.93      0.91       953
miscellaneous       0.80      0.70      0.75       191
     politics       0.78      0.86      0.82       510
   recreation       0.90      0.89      0.90       751
     religion       0.90      0.82      0.86       478
      science       0.87      0.84      0.85       764

     accuracy                           0.87      3647
    macro avg       0.86      0.84      0.85      3647
 weighted avg       0.87      0.87      0.87      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\Avg_w2v\38_svm_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\Avg_w2v\38_svm_fig.png)



## window size = 75

### LR:

```
Accuracy: 0.85
Auc: 0.97
Detail:
               precision    recall  f1-score   support

     computer       0.88      0.91      0.90       953
miscellaneous       0.78      0.65      0.71       191
     politics       0.78      0.84      0.81       510
   recreation       0.88      0.88      0.88       751
     religion       0.87      0.82      0.84       478
      science       0.82      0.82      0.82       764

     accuracy                           0.85      3647
    macro avg       0.84      0.82      0.83      3647
 weighted avg       0.85      0.85      0.85      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\Avg_w2v\75_lr_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\Avg_w2v\75_lr_fig.png)

### SVM:

```
Accuracy: 0.87
Auc: 0.98
Detail:
               precision    recall  f1-score   support

     computer       0.88      0.93      0.91       953
miscellaneous       0.84      0.70      0.76       191
     politics       0.80      0.85      0.83       510
   recreation       0.90      0.90      0.90       751
     religion       0.89      0.82      0.85       478
      science       0.86      0.83      0.85       764

     accuracy                           0.87      3647
    macro avg       0.86      0.84      0.85      3647
 weighted avg       0.87      0.87      0.87      3647
```

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\Avg_w2v\75_svm_cm.png)

![](D:\Data\NLP_057-Final-Project\feature\word2vec_new\Avg_w2v\75_svm_fig.png)

