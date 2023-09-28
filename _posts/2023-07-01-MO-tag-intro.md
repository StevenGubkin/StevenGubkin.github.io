---
layout: post
title: Introduction to the MathOverflow Tag Recommendation Problem
categories: 
  - Mathoverflow Tag Recommendation
date: 2023-07-01 00:00:05
---

Here is the first paragraph of a [recent post](https://mathoverflow.net/q/450133/1106) on the front page of MathOverflow:

>Let $$X$$ be a connected qcqs scheme. We say that $$X$$ is a (étale) $$K(\pi,1)$$ if for every locally constant constructible abelian sheaf $$\mathscr{F}$$ on $$X$$ and every geometric point $$\overline{x}$$ the natural morphisms
>
>$$\mathrm{H}^i(\pi_1(X,\overline{x}),\mathscr{F}_x) \to \mathrm{H}^i_\text{ét}(X,\mathscr{F})$$
>
>are isomorphisms for all $$i\geq 0$$. (See section 4 on \[P. Achinger's Wild Ramification and $$K(\pi,1)$$ spaces\]\[1\] for a quick review on this.)

Could you guess which tags might be used for this post based on the text?  Personally, I might guess "algebraic-geometry" and "étale-cohomology".

In fact, the tags being used are "ag.algebraic-geometry", "at.algebraic-topology", "etale-cohomology", and "derham-cohomology".

This is a multilabel text classification problem.  In this sequence of posts we will explore several different ways to tackle this problem with machine learning.

- In [Data Exploration and Preprocessing]({% post_url 2023-07-02-data-exploration %})
 we take a look at the data which comes from the quarterly [Stack Exchange data dump](https://archive.org/details/stackexchange).  We explore the data to understand how it is structured. We clean the data. We also create testing, validation, and training sets which are balanced with respect to label representation.
- In [Multilabel Stratified Split]({% post_url 2023-07-02-train-val-test-split %}) we explain why making a stratified split for a multilabel problem is a little tricky, and outline a solution provided by [this paper](https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10).  Thankfully the algorithm was already implemented [here](https://github.com/trent-b/iterative-stratification).  We save the resulting train/val/test split as a [HuggingFace dataset](https://huggingface.co/datasets/stevengubkin/mathoverflow_text_arxiv_labels).
- In [PMI Model]({% post_url 2023-07-03-pmi-baseline-model %}) we give a first attempt at the classification problem.  We select a vocabulary of 10000 words from the cleaned data. We compute the pointwise mutual information between words in our vocabulary and arxiv labels.  Multiplying this matrix with a one-hot encoded document vector gives us a score for each label.  We use the highest scoring labels as our predicted labels.  One of the top 3 recommended labels is a correct label 83% of the time for the validation set.
- In [Distilbert with a Simple Classifier Head]({% post_url 2023-07-04-distilbert-simple-head %}) we train a shallow neural network classification head on top of “distilbert-base-uncased”.  It performs a little better than our PMI model. 
