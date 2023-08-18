---
layout: post
title: Introduction to the MathOverflow Tag Recommendation Problem
categories: 
  - Mathoverflow Tag Recommendation
date: 2023-07-01 00:00:04
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
 we take a look at the data which comes from the quarterly [Stack Exchange data dump](https://archive.org/details/stackexchange).  We explore the data to understand how it is structured.  We select a vocabulary of 10000 words from the cleaned data. We also create testing, validation, and training sets which are balanced with respect to label representation.
- In [Baseline Model]({% post_url 2023-07-03-pmi-baseline-model %}) we give a first attempt at the classification problem.  In the baseline model we compute the pointwise mutual information between words in our vocabulary and arxiv labels.  Multiplying this matrix with a one-hot encoded document vector gives us a score for each label.  We use the highest scoring labels as our predicted labels.
- In [Encoder-Decoder]({% post_url 2023-07-04-encT5 %}) we implement a model similar to ["EncT5: A Framework for Fine-tuning T5 as Non-autoregressive Models"](https://aclanthology.org/2023.findings-acl.360.pdf) by Frederick Liu, Terry Huang, Shihang Lyu, Siamak Shakeri, Hongkun Yu, and Jing Li.  
