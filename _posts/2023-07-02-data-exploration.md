---
layout: post
title: Data Exploration and Preprocessing
categories: 
  - Mathoverflow Tag Recommendation
date: 2023-07-01 00:00:03

---

We take a look at the data which comes from the quarterly [Stack Exchange data dump](https://archive.org/details/stackexchange).  We explore the data to understand how it is structured, explore the co-occurance of labels to see that it is reasonable to reduce the scope of our problem from thousands of labels to just the 32 arxiv tag labels, filter and normalize the data, and select a vocabulary of 20000 words from the cleaned data. We also create testing, validation, and training sets which are balanced with respect to label representation.



```python
import pandas as pd
import numpy as np
```


```python
X_train = pd.read_csv('X_train', index_col= 0)
X_valid = pd.read_csv('X_valid', index_col= 0)
X_test  = pd.read_csv('X_test', index_col= 0)
y_train = pd.read_csv('y_train', index_col= 0)
y_valid = pd.read_csv('y_valid', index_col= 0)
y_test  = pd.read_csv('y_test', index_col= 0)
```


```python
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape
```




    ((74796, 1), (74796, 32), (8301, 1), (8301, 32), (9162, 1), (9162, 32))


