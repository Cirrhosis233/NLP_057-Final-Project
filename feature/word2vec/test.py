import numpy as np

path = "..\\..\\corpus\\dev\\alt.atheism\\53068.npy"

doc = np.load(path, allow_pickle=True)

# print(doc)

print([word for sent in doc for word in sent])
