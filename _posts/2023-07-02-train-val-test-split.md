---
layout: post
title: Multilabel Stratified Split
categories: 
  - Mathoverflow Tag Recommendation
date: 2023-07-01 00:00:03
---

Most train/valid/test split tools are not optimized for multilabel problems.  The tool MultilabelStratifiedShuffleSplit from iterstrat.ml_stratifiers (see the [github page](https://github.com/trent-b/iterative-stratification)) implements the algorithm from [Konstantinos Sechidis, Grigorios Tsoumakas & Ioannis Vlahavas (2011)](https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10).

The issue is this:  multi-**class** stratified sampling is easy.  We can just take a random sample of each class to create our split.

Multi-**label** stratification is harder (and somewhat ill-defined).  If we have $$L$$ labels, then there are $$2^{L} - 1$$ possible ways that labels can co-occur with each other.  Ideally we would like each of these $$2^{L} - 1$$ possibilities represented in equal proportion in a split.  This is likely impossible unless we have a very small number of labels.

The paper relaxes this requirement, and instead just asks for proportional representation of each label.  This is a little tricky, since when we grab an example to put in a split, that example has multiple labels attached to it.  If you try to achieve proportional representation in one label, you may find that you have accidentally unbalanced another label.  The solution is to start with the rarest labels, adjusting the running total of each label available as we go.  This will initially create unbalanced splits for the more common labels, but since they are more common we have more opportunities to correct this inbalance later.

```python
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

#Creating a split of data into training set (70% of data) and the rest (30% of data).
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state = 42)
train, not_train = next(msss.split(df['Title_Body'],df[arxiv_tags]))
df_train = df.iloc[train].copy()
df_not_train = df.iloc[not_train].copy()

#Splitting the 30% of the remaining data into our validation and testing sets for a 70/20/10 split.
msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3333, random_state = 42)
valid, test = next(msss2.split(df_not_train['Title_Body'],df_not_train[arxiv_tags]))
df_valid = df_not_train.iloc[valid]
df_test = df_not_train.iloc[test]

#Creating pytorch datasets from these pandas DataFrames
from datasets import Dataset, DatasetDict

ds_train = Dataset.from_pandas(df_train)
ds_valid = Dataset.from_pandas(df_valid)
ds_test = Dataset.from_pandas(df_test)
ds = DatasetDict()
ds['train'] = ds_train
ds['valid'] = ds_valid
ds['test'] = ds_test

ds.save_to_disk('mathoverflow-tags-dataset')
```

This was then uploaded to HuggingFace [here](https://huggingface.co/datasets/stevengubkin/mathoverflow_text_arxiv_labels).