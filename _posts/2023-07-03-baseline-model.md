---
layout: post
title: Baseline Model
categories: 
  - Mathoverflow Tag Recommendation
date: 2023-07-01 00:00:02

---

We give a first attempt at the classification problem.  In the baseline model we train 32 binary classification models (one for each arxiv tag label) and concatenate these models.  Each model has a very simple architecture:  $$10000 \stackrel{\textrm{dense}}{\longrightarrow} 100 \stackrel{\textrm{ReLU}}{\longrightarrow} 100 \stackrel{\textrm{dense}}{\longrightarrow} 1 \stackrel{\textrm{sigmoid}}{\longrightarrow} 1$$.