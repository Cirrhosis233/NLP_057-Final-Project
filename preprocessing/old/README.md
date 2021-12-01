# Corpus Manual

### raw:

- Contains raw 20-news groups corpus set.

### corpus:

- Contains preprocessed corpus, divided into training and testing datasets, ratio of 7 : 3

### How to use .npy file

```python
numpy.load('sample.npy', allow_pickle=True)
```

- You will get nested list of corpus directly, no need to read from .txt.



— By Letian Ye