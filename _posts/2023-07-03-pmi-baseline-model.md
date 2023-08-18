---
layout: post
title: Baseline Model
categories: 
  - Mathoverflow Tag Recommendation
date: 2023-07-01 00:00:02

---

In this post we will give a baseline model which uses [pointwise mutual information]().

We have $$N$$ documents using a vocabulary with words $$w_i$$.  Each document is tagged with a subset of the labels $$l_j$$.

The pointwise mutual information between a word $$w$$ and a label $$l$$ is defined by 

$$
\textrm{pmi}(w ; l) = \log \left( \frac{p(w \vert l)}{p(w)} \right)
$$

Let's break this down one step at a time. 

$$p(w)$$ answers the question "If I choose a word at random from any document, what is the chance that my word is $$w$$?"  

$$p(w \vert l)$$ answers the question "If I choose a word at random from a document with label $$l$$, what is the chance that my word is $$w$$?"

So $$\frac{p(w \vert l)}{p(w)}$$ answers the question "How many times more likely am I to see $$w$$ when choosing a random word from a document with label $$l$$ than I am to see it when choosing at random from all of the documents"?  This number will be more than $$1$$ when $$w$$ occurs more frequently than expected in the context of label $$l$$ and less than $$1$$ otherwise.

The reason we take the logarithm is the same reason we always take logarithms:  because we want to convert products into sums.  We will use a [naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) assumption in our classification algorithm, and assume independence of the words.  Then

$$
\begin{align*}
\textrm{pmi}(w_1w_2 ; l) 
&= \log \left( \frac{p(w_1w_2 \vert l)}{p(w_1 w_2)} \right)\\
&= \log \left( \frac{p(w_1 \vert l)p(w_2 \vert l)}{p(w_1)p(w_2)}\right)\\
&= \log \log \left( \frac{p(w_1 \vert l)}{p(w)} \right) + \log \left( \frac{p(w_2 \vert l)}{p(w)} \right)\\
&= \textrm{pmi}(w_1; l) + \textrm{pmi}(w_2;l)
\end{align*}
$$

where we use the assumption of independence to split the joint probabilities into the products of the individual probabilities.

So the logarithm gives us the nice feature that adding the $$\textrm{pmi}$$ score for each word in a document will give rise to a meaningful score.  It will also move these scores into a range which is more human interpretable.

Here we can see the top PMI scoring words for three different labels in our training data:

![Top PMI scoring words for some labels](/assets/images/top-pmi-scores.png)

If you know anything about these fields they seem reasonable.  For instance, [Émile Picard](https://en.wikipedia.org/wiki/%C3%89mile_Picard) is a mathematician who made many contributions to Complex Analysis, and his theorems are often referenced by his name.  It makes sense that "Picard" is occuring much more frequently in the cv.complex-variables tagged posts than the corpus as a whole.

We can also look at the most negative scoring words for the same three labels:

![Bottom PMI scoring words for some labels](/assets/images/bot-pmi-scores.png)

Apparently 

* Category Theorists don't like talking about differential geometry ('geodesic', 'hyperbolic'), complex numbers ('hermitian'), or number theory ('digits').
* Complex Analysts don't like talking about category theory ('categories', 'colimits'), logic ('cardinals', 'definable'), or algebraic topology ('simplicial').
* Differential Geometers don't like talking about logic ('cardinals', 'forcing'), number theory ('primes'), or category theory ('enriched').

So if we see a document which has the words 'curvature' and 'overdetermined' and which doesn't have the words 'cardinals' we can be pretty sure that it is a differential geometry question.

It is also interesting to look in the other direction:  what tags have positive $$\textrm{pmi}$$ for a given word?

![PMI scores for the words 'ample', 'cardinal', and 'also'](/assets/images/pmi-for-words.png)

The word 'ample' ends up having a positive pmi association with the tags 'ag.algebraic-geometry' and 'sg.symplectic-geometry', but is more strongly associated with 'ag.algebraic-geometry'.

The word 'cardinal' ends up having a positive pmi association with the tags 'gn.general-topology' and 'lo.logic', but is more strongly associated with 'lo.logic'.

The word 'also' ends up having pmi scores very close to 0, since it's usage is not associated with any particular label.

You can see the full implementation details of compute these $$\textrm{pmi}$$ scores in the associated Jupyter notebook, but I think there are a few tips and tricks which are worth noting here even if you do not want to wade through the full notebook.

In our training data we have 74796 documents.  We one-hot encode these documents using our vocabulary of 10000 words into a (74796,10000) matrix $$X$$.  Similarly, our labels are one-hot encoded into a (74796, 32) matrix $$y$$.

The first thing worth noting is that the matrix $$\textrm{coocur} = X^{\top} y$$ is the matrix of word-label co-occurrence counts!  A column of $$X$$ is a vector indexed by document and recording whether the word corresponding to that column is present or absent.  A column of $$y$$ is a vector indexed by document and recording whether the label corresponding to that column is present or absent.  When we dot those together, we add $$1$$ to the sum whenever the word and labal are both present in the same document, and we add $$0$$ otherwise.  $$X^\top y$$ dots each column of $$X$$ with each column of $$y$$ to create a co-occurrence count matrix  $$\textrm{coocur}$$ of shape (10000,32).  Using pytorch to perform this matrix multiplication is way faster than the naive iterated for-loop solution.

Let $$\textrm{word_counts}$$ be the $$(10000)$$ vector which contains a count of the frequency of each word across the entire corpus.
 
Remember that 

$$
\textrm{pmi}(w_i; l_j) = \log\left(\frac{p(w_i \vert l_j)}{p(w_i)}\right)
$$

I am estimating these using the frequency counts we obtained above. 

$$
p(w_i \vert l_j) \approx \frac{\textrm{coocur[i,j]}}{\textrm{coocur[:, $l_j$].sum()}}
$$

$$
p(w_i) \approx \frac{\textrm{word_counts[i]}}{\textrm{word_counts.sum()}}
$$

so we have

$$
\textrm{pmi}(w_i,l_j) = \log\left(\frac{\textrm{coocur[i,j]} \times \textrm{word_counts.sum()}}{\textrm{word_counts[i]} \times \textrm{coocur[:, $l_j$].sum()}} \right)
$$

All of this can be efficiently calculated using pytorch broadcasting rules as follows:

```python
coocur = torch.transpose(X) @ y
prob_ratios = ((coocur*word_counts.sum())/(coocur.sum(dim =0)))/(word_counts.unsqueeze(1))
pmi = torch.log(prob_ratios)
pmi[pmi == float('-inf')] = 0
```

Note that without the last line $$\textrm{pmi}$$ would have lots of entries being equal to $$-\infty$$!  This happens when $$\textrm{word}_i$$ never appears in a document with $$\textrm{label}_j$$.  In our prediction algorithm, we will apply the $$\textrm{pmi}$$ matrix to a one-hot encoded document vector to obtain a score.  Words with positive $$\textrm{pmi}$$ for a given label will contribute positively to the score for that label while negative $$\textrm{pmi}$$ will contribute negatively. 

The presense of a word $$w_i$$ in the document with $$\textrm{pmi}[i,j] = -\infty$$ will always give a score of $$-\infty$$ to that label.  This is undesirable.  Just because we never saw the word "elephant" labeled with "combinatorics" in our training data doesn't mean we should rule out a new document having the label "combinatorics" just because it uses the word "elephant"!

Multilabel prediction algorithm evaluation is tricky business.  We will do a much more detailed analysis later, but for now we will compute our "accuracy" as the fraction of the time that our **top** predicted label for each document is actually one of the labels attached to that document.

On our data we have

```python
pmi_train_predictions = X_train_tensor @ pmi
pmi_valid_predictions = X_valid_tensor @ pmi
indices_train =torch.max(pmi_train_predictions,1,keepdim=True)[1]
indices_valid = torch.max(pmi_valid_predictions,1,keepdim=True)[1]
train_accuracy = torch.take_along_dim(y_train,indices_train,dim=1).sum()/y_train.shape[0]
valid_accuracy = torch.take_along_dim(y_valid,indices_valid,dim=1).sum()/y_valid.shape[0]
print(train_accuracy, valid_accuracy)
```

```
tensor(0.6754) tensor(0.6185)
```

This is not so bad!  It definitely outperforms 'extremely dumb' baseline strategy of choosing the most common label ('ag.algebraic-geometry'), which would have an accuracy of 

$$
\frac{12288}{74796} \approx 0.165
$$

We can also test it on some examples from the training data. Here I am just taking a training document at random, feeding it into the $$\textrm{pmi}$$ matrix, and then selecting which three tags have the highest score.

* [Link to full question](https://mathoverflow.net/q/33945/1106)

* Tokenized text: "spectrum ring entire functions let ring entire functions functions  holomorphic maximal ideals besides obvious ones anyone give concise description..."

* The actual tags:  
  * ag.algebraic-geometry 
  * cv.complex-variables 
  * ra.rings-and-algebras

* The predicted tags: 
  * ag.algebraic-geometry
  * cv.complex-variables
  * ac.commutative-algebra 

This does a pretty food job! We might even argue that ac.commutative-algebra was a better pick than ra.rings-and-algebras since the ring of entire functions is commutative.
