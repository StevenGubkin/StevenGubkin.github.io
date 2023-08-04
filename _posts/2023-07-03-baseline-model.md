---
layout: post
title: Baseline Model
categories: 
  - Mathoverflow Tag Recommendation
date: 2023-07-01 00:00:02

---

We give two first attempts at the classification problem.  The first baseline model is a very simple neural network using two 64 dimensional hidden layer with relu activation, a 32 dimensional output layer with sigmoid activation, and using binary cross entropy as our loss function.  The second baseline model uses a random forest.