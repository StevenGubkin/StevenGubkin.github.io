---
layout: post
title: Distilbert with a Simple Classifier Head
categories: 
  - Mathoverflow Tag Recommendation
date: 2023-07-01 00:00:01
---

We summarize the work done in [this Colab notebook](https://colab.research.google.com/drive/1spmsNKc4q7B5m48Hp2BSo9-MNVQx36yX?usp=sharing).

In the notebook we:

* tokenize the text data of each post.
* run "distilbert-base-uncased" and extract the last hidden state for each post to obtain a tensor of shape (number of observations, number of tokens, embedding_dimesion).
* We take the mean of these over the tokens to obtain a tensor of shape (number of observations, embedding_dimension).  This is what we train our classification head on.
* The classification head is the following shallow neural network:

```python
model = torch.nn.Sequential(
    torch.nn.Linear(embedding_dimension,embedding_dimension),
    torch.nn.Dropout(p=0.1),
    torch.nn.ReLU(),
    torch.nn.Linear(embedding_dimension,num_labels),
    torch.nn.Sigmoid()
)

optimizer = torch.optim.RMSprop(
    model.parameters(), lr = 0.0001
    )

loss = torch.nn.BCELoss()
```

* Training this model for ~1000 epochs gives a result which is slightly better than our PMI model:  the top 3 arxiv labels recommended by the pytorch model contain one of the actual tags 85% of the time, as opposed to the 83% success rate we got with the PMI model.

