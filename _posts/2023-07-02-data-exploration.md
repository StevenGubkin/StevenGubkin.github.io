---
layout: post
title: Data Exploration and Preprocessing
categories: 
  - Mathoverflow Tag Recommendation
date: 2023-07-01 00:00:03

---

We take a look at the data which comes from the quarterly [Stack Exchange data dump](https://archive.org/details/stackexchange).  We explore the data to understand how it is structured, explore the co-occurance of labels to see that it is reasonable to reduce the scope of our problem from thousands of labels to just the 32 arxiv tag labels, filter and normalize the data, and select a vocabulary of 20000 words from the cleaned data. We also create testing, validation, and training sets which are balanced with respect to label representation.
