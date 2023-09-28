---
layout: post
title: Data Exploration and Preprocessing
categories: 
  - Mathoverflow Tag Recommendation
date: 2023-07-01 00:00:04

---

We take a look at the data which comes from the quarterly [Stack Exchange data dump](https://archive.org/details/stackexchange).  We explore the data to understand how it is structured and clean the data.

The data from the datadump has 323005 rows and 22 columns.  Of these columns, the only ones relevant to our classification problem are:

* 'Id': A unique identifier for each post.
* 'PostTypeId': Equal to 1 if the post is a question, 2 if it is an answer.
* 'Score':  The cumulative number of upvotes/downvotes the post has recieved.  Relevant because we do not want to train on low quality posts.
* 'Body':  The markdown code for the post.
* 'Title':  The title of the post.
* 'Tags':  The tags of the post (example:  \<gr.group-theory\>\<lie-groups\>).  Note:  only questions posts have tags.
* 'ParentId':  If the post is an answer, this is the Id of the question it is answering.  We will need this to assign tags to answers posts.

We replace Tags with a comma seperated list

```python
df['Tags'] = df['Tags'].str.replace('>',', ').str.replace('<','').str[:-2]
```

We want to use the questions, but they are currently not labeled with tags. The next cell creates dictionaries which convert back and forth between index, Id, and ParentId.   We then assign, to each answer post, the tags associated with the question post it is answering:

$$
\textrm{index} \to \textrm{ParentId} \to \textrm{index} \to \textrm{Tags} 
$$
```python
# These are all fairly self-explanatory given their names.
index_to_Id = df[df['PostTypeId'] == 1]['Id'].to_dict()
Id_to_index = dict(map(reversed, index_to_Id.items()))
index_to_ParentId = df[df['PostTypeId'] == 2]['ParentId'].astype(int).to_dict()
ParentId_to_index = dict(map(reversed, index_to_ParentId.items()))

# Assigning, to each answer post, the tags associated with the question post it is answering.
df_copy = df.copy()
for i in df.index:
    if df['PostTypeId'].loc[i] == 2:
        df_copy['Tags'].loc[i] = df['Tags'].loc[Id_to_index[index_to_ParentId[i]]]

# Only keep posts with score greater than 5.
df = df_copy[df_copy['Score'] > 5]
```

We extract the text from each 'Body' using:

```python
df['Body'] = df['Body'].parallel_apply(lambda text: BeautifulSoup(text, 'html.parser').get_text())
```

parallel_apply is a nifty tool which parallelizes application of functions on all of your available CPUs.  It also gives a handy progress bar.


The following function replaces any tex enclosed by $, $$, or \begin{...} environments with [UNK], in preperation for BERT tokenization.  
TexSoup fails on some rare examples, and it doesn't seem worth the time to try to fix it.  This is a cludge, but it doesn't throw away too many examples.

```python
!pip install TexSoup
from TexSoup import TexSoup

def delete_math(text):
    try:
        soup = TexSoup(text, tolerance = 1)
        for stuff in soup:
            if '$' in str(stuff) or '\\begin' in str(stuff):
                soup.replace(stuff, '[UNK]')
        return str(soup)
    except:
        return 'soup failed'

df['Body'] = df['Body'].parallel_apply(delete_math)
```

We apply the same treatment to the title, and the concatenate the cleaned title and body into one string Title_Body

```python
df['Title_Body'] = df['Title'] + ' ' + df['Body']
```

I only chose to retain those examples with at least 200 characters in the body of the post:

```python
df = df[df['Body'].str.len() > 200]
```

There are hundreds of tags.  I limit myself to only the following arxiv tags for this project:

```python
arxiv_tags = ['ac.commutative-algebra',
 'ag.algebraic-geometry',
 'ap.analysis-of-pdes',
 'at.algebraic-topology',
 'ca.classical-analysis-and-odes',
 'co.combinatorics',
 'ct.category-theory',
 'cv.complex-variables',
 'dg.differential-geometry',
 'ds.dynamical-systems',
 'fa.functional-analysis',
 'gm.general-mathematics',
 'gn.general-topology',
 'gr.group-theory',
 'gt.geometric-topology',
 'ho.history-overview',
 'it.information-theory',
 'kt.k-theory-and-homology',
 'lo.logic',
 'mg.metric-geometry',
 'mp.mathematical-physics',
 'na.numerical-analysis',
 'nt.number-theory',
 'oa.operator-algebras',
 'oc.optimization-and-control',
 'pr.probability',
 'qa.quantum-algebra',
 'ra.rings-and-algebras',
 'rt.representation-theory',
 'sg.symplectic-geometry',
 'sp.spectral-theory',
 'st.statistics']
```

We then one-hot encode the arxiv tag:

```python
# making a column of zeros for each arxiv tag 
df[arxiv_tags] = 0

# setting the value equal to 1 if the tag appears in the 'Tags' column.
for i in df.index:
    for j in range(len(arxiv_tags)):
        if arxiv_tags[j] in df.loc[i,'Tags']:
            df.loc[i, arxiv_tags[j]] = 1
```

Not all of these posts actually use any arxiv labels!  We only want to retain posts which use at least one arxiv label.

```python
df = df[df[arxiv_tags].sum(axis = 1) != 0]

df.to_csv('data_with_unks')
```



