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
 we take a look at the data which comes from the quarterly [Stack Exchange data dump](https://archive.org/details/stackexchange).  We explore the data to understand how it is structured, explore the co-occurance of labels to see that it is reasonable to reduce the scope of our problem from thousands of labels to just the 32 arxiv tag labels, filter and normalize the data, and select a vocabulary of 10000 words from the cleaned data. We also create testing, validation, and training sets which are balanced with respect to label representation.
- In [Baseline Model]({% post_url 2023-07-03-baseline-model %}) we a first attempt at the classification problem.  In the baseline model we train 32 binary classification models (one for each arxiv tag label) and concatenate these models.  Each model has a very simple architecture:  $$10000 \stackrel{\textrm{dense}}{\longrightarrow} 100 \stackrel{\textrm{ReLU}}{\longrightarrow} 100 \stackrel{\textrm{dense}}{\longrightarrow} 1 \stackrel{\textrm{sigmoid}}{\longrightarrow} 1$$.
- In [Multilabel Reasoner]({% post_url 2023-07-04-multilabel-reasoner %}) we implement the model proposed in ["A novel reasoning mechanism for multi-label text classification"](https://doi.org/10.1016/j.ipm.2020.102441) by Ran Wang, Robert Ridley, Xi'ao Su, Weiguang Qu, and Xinyu Dai.
