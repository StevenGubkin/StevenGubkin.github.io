---
layout: post
title: PMI Model
categories: 
  - Mathoverflow Tag Recommendation
date: 2023-07-01 00:00:02

---

Andrej Karpathy makes a distinction between what he calls [software 1.0 and software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35).  Software 1.0 consists of explicit instructions for transforming inputs into desired outputs.  Software 2.0 is machine learning: we provide a model with a ton of parameters and minimize a loss function.  The trained model then transforms inputs into desired outputs in a way which performs well on the training data, and which (we hope!) will generalize to novel data.

In this post I explain how I wrote the best "software 1.0" code I could to solve our classification problem. The model utilizes [pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information).

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
&= \log \left( \frac{p(w_1 \vert l)}{p(w)} \right) + \log \left( \frac{p(w_2 \vert l)}{p(w)} \right)\\
&= \textrm{pmi}(w_1; l) + \textrm{pmi}(w_2;l)
\end{align*}
$$

where we use the assumption of independence to split the joint probabilities into the products of the individual probabilities.

So the logarithm gives us the nice feature that adding the $$\textrm{pmi}$$ score for each word in a document will give rise to a meaningful score.  It will also move these scores into a range which is more human interpretable.

You can see the full implementation details of compute these $$\textrm{pmi}$$ scores in the associated Jupyter notebook, but I think there are a few tips and tricks which are worth noting here even if you do not want to wade through the full notebook.

In our training data we have $$N$$ documents.  We one-hot encode these documents using our vocabulary of 10000 words into a (N,10000) matrix $$X$$.  Similarly, our labels are one-hot encoded into a (N, 32) matrix $$y$$.

The first thing worth noting is that the matrix 

$$\textrm{coocur} = X^{\top} y$$

 is the matrix of word-label co-occurrence counts!  A column of $$X$$ is a vector indexed by document and recording whether the word corresponding to that column is present or absent.  A column of $$y$$ is a vector indexed by document and recording whether the label corresponding to that column is present or absent.  When we dot those together, we add $$1$$ to the sum whenever the word and labal are both present in the same document, and we add $$0$$ otherwise.  $$X^\top y$$ dots each column of $$X$$ with each column of $$y$$ to create a co-occurrence count matrix  $$\textrm{coocur}$$ of shape (10000,32).  Using pytorch to perform this matrix multiplication is way faster than the naive iterated for-loop solution.

Let $$\textrm{word_counts}$$ be the $$(10000)$$ vector which contains a count of the frequency of each word across the entire corpus.
 
Remember that 

$$
\textrm{pmi}(w_i; l_j) = \log\left(\frac{p(w_i \vert l_j)}{p(w_i)}\right)
$$

I am estimating these using the frequency counts we obtained above. 

$$
p(w_i \vert l_j) \approx \frac{\textrm{coocur[i,j]}}{\textrm{coocur[:, $j$].sum()}}
$$

$$
p(w_i) \approx \frac{\textrm{word_counts[i]}}{\textrm{word_counts.sum()}}
$$

so we have

$$
\textrm{pmi}(w_i,l_j) = \log\left(\frac{\textrm{coocur[i,j]} \times \textrm{word_counts.sum()}}{\textrm{word_counts[i]} \times \textrm{coocur[:, $j$].sum()}} \right)
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

Multilabel prediction algorithm evaluation is tricky business.  We will do a more detailed analysis later, but for now we will compute our "accuracy" as the fraction of the time that any one of our **top 3** predicted labels for each document is actually one of the labels attached to that document.

On our data we have

```python
pmi_valid_predictions = X_valid_tensor @ pmi
top_pred_in_label_set = 0
for i in range(y_valid.shape[0]):
  top3 = torch.topk(pmi_valid_predictions[i],3).indices.numpy()
  if y_valid[i, top3[0]] == 1 or y_valid[i, top3[1]] == 1 or y_valid[i, top3[2]] == 1:
    top_pred_in_label_set += 1
top_pred_in_label_set
print(top_pred_in_label_set/y_valid.shape[0])
```

```
0.8329116973858571
```

This is not so bad!  You could think of this as saying that our model is giving at least one relevant suggestion 83% of the time.   It definitely outperforms 'extremely dumb' baseline strategy of always recommending the 3 most common labels.  We can compute how well that baseline strategy would do as follows:

```python
most_common_labels = torch.topk(y_valid.sum(axis = 0), k=3).indices.numpy()
top_pred_in_label_set = 0
for i in range(y_valid.shape[0]):
  if y_valid[i, most_common_labels[0]] == 1 or y_valid[i, most_common_labels[1]] == 1 or y_valid[i, most_common_labels[2]] == 1:
    top_pred_in_label_set += 1
top_pred_in_label_set
print(top_pred_in_label_set/y_valid.shape[0])
```

```
0.3927237682206963
```

So the baseline strategy of always recommending the 3 most common labels would give at least one relevant tag 39% of the time.

We can also test it on some examples from the training data. Here I am just taking a training document at random, feeding it into the $$\textrm{pmi}$$ matrix, and then selecting which three tags have the highest score.

* [Link to full question](https://mathoverflow.net/q/33945/1106)

* Question text:  
> What is the spectrum of the ring of entire functions?
>
> Let $$\mathcal{O}(\mathbb{C})$$ be the ring of entire functions, that is, those functions $$f : \mathbb{C} \to \mathbb{C}$$ which are holomorphic for all $$z \in \mathbb{C}$$. For each $$z_0 \in \mathbb{C}$$.
>
> Are there any other maximal ideals in $$\mathcal{O}(\mathbb{C})$$ besides these obvious ones?
> 
> If anyone can give a concise description of $$\text{Spec }\mathcal{O}(\mathbb{C})$$, that would be extremely helpful. I'm trying to understand wether or not knowing the closed subset $$V(f)$$ of $$\text{Spec }\mathcal{O}(\mathbb{C})$$ of ideals containing $$f$$ gives you more information about $$f$$ than simply knowing the vanishing set of $$f$$ in the classical sense.

* Tokenized text: "spectrum ring entire functions let ring entire functions functions  holomorphic maximal ideals besides obvious ones anyone give concise description..."

* The actual tags:  
  * ag.algebraic-geometry 
  * cv.complex-variables 
  * ra.rings-and-algebras

* The predicted tags: 
  * ag.algebraic-geometry
  * cv.complex-variables
  * ac.commutative-algebra 

This does a pretty good job! We might even argue that ac.commutative-algebra was a better pick than ra.rings-and-algebras since the ring of entire functions is commutative.

We can also use the PMI score matrix to do some fun analysis of the MathOverflow corpus!

Here we can see the top PMI scoring words for three different labels in our training data:

ct.category-theory	rosicky	shulmans	makkai	pseudofunctors	cocompletion	rosický	adámek	kellys	trimble	zhen
cv.complex-variables	picards	ahlfors	remmert	removable	titchmarsh	grauert	blaschke	liouvilles	continuations	abscissa
dg.differential-geometry	deane	kuiper	curvatures	civita	overdetermined	bryants	patodi	bryant	rotationally	michor

| arxiv tag                | 1       | 2        | 3          | 4              | 5              | 6       | 7        | 8          | 9             | 10       |
|--------------------------|---------|----------|------------|----------------|----------------|---------|----------|------------|---------------|----------|
|       ct.category-theory | rosicky | shulmans |     makkai | pseudofunctors |   cocompletion | rosický |   adámek |     kellys |       trimble |     zhen |
|     cv.complex-variables | picards |  ahlfors |    remmert |      removable |     titchmarsh | grauert | blaschke | liouvilles | continuations | abscissa |
| dg.differential-geometry |   deane |   kuiper | curvatures |         civita | overdetermined | bryants |   patodi |     bryant |  rotationally |   michor |

If you know anything about these fields they seem reasonable.  For instance, [Émile Picard](https://en.wikipedia.org/wiki/%C3%89mile_Picard) is a mathematician who made many contributions to Complex Analysis, and his theorems are often referenced by his name.  It makes sense that "Picard" is occuring much more frequently in the cv.complex-variables tagged posts than the corpus as a whole.

We can also look at the most negative scoring words for the same three labels:

| arxiv tag                | 1          | 2          | 3          | 4         | 5            | 6           | 7         | 8       | 9         | 10         |
|--------------------------|------------|------------|------------|-----------|--------------|-------------|-----------|---------|-----------|------------|
|       ct.category-theory |   geodesic | hyperbolic | eigenvalue | geodesics | discriminant |       ricci | hermitian |      cm |    motion |     digits |
|     cv.complex-variables | categories |  cardinals |     models | definable |     colimits |   separable |  theories | boolean |     étale | simplicial |
| dg.differential-geometry |  cardinals |     primes |    boolean |        zf |       turing | presentable |        pa | forcing | definable |   enriched |

Apparently 

* Category Theorists don't like talking about differential geometry ('geodesic', 'hyperbolic'), complex numbers ('hermitian'), or number theory ('digits').
* Complex Analysts don't like talking about category theory ('categories', 'colimits'), logic ('cardinals', 'definable'), or algebraic topology ('simplicial').
* Differential Geometers don't like talking about logic ('cardinals', 'forcing'), number theory ('primes'), or category theory ('enriched').

So if we see a document which has the words 'curvature' and 'overdetermined' and which doesn't have the words 'cardinals' we can be pretty sure that it is a differential geometry question.

It is also interesting to look in the other direction:  what tags have positive $$\textrm{pmi}$$ for a given word?

|          |   ac |      ag |   ap |   gm |      gn |   gr |      lo |   kt |   rt |      sg |  sp |
|---------:|-----:|--------:|-----:|-----:|--------:|-----:|--------:|-----:|-----:|--------:|----:|
|    ample | -0.5 | **1.4** | -2.2 | -1.4 |    -3.1 | -3.2 |    -3.3 | -0.3 | -1.7 | **0.3** | 0.0 |
| cardinal | -0.8 |    -2.6 |  0.0 | -1.1 | **0.6** | -0.9 | **1.6** | -2.9 | -2.5 |    -3.0 | 0.0 |
|     also |  0.1 |     0.0 | -0.0 | -0.1 |     0.1 |  0.1 |     0.0 |  0.0 |  0.0 |    -0.1 | 0.1 |

The word 'ample' ends up having a positive pmi association with the tags 'ag.algebraic-geometry' and 'sg.symplectic-geometry', but is more strongly associated with 'ag.algebraic-geometry'.

The word 'cardinal' ends up having a positive pmi association with the tags 'gn.general-topology' and 'lo.logic', but is more strongly associated with 'lo.logic'.

The word 'also' ends up having pmi scores very close to 0, since it's usage is not associated with any particular label.


