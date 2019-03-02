---
layout: post
title: Let's implement the CART Algorithm
date: 2019-03-01 10:00:00 {{site.timezone}}
mathjax: true
excerpt_separator: <!--more-->
tags: [decision-trees, machine-learning, Python, code]
comments: true
feature: https://images.unsplash.com/photo-1515879218367-8466d910aaa4?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80
---

This is Part 3 of my decision trees series. This time around we are going to code a decision tree in Python. So I'm going to try to make this code as understandable as possible, but if you are not familiar with [Object Oriented Programming (OOP)](https://en.wikipedia.org/wiki/Object-oriented_programming) or [recursion](https://en.wikipedia.org/wiki/Recursion_(computer_science)) you might have a tougher time.  
<!--more-->
To make data handling easier, we are going to be using the wonderful `pandas` package, so if you don't know how to use it I highly recommend you learn to by reading [their intro to pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html). *(there are also very good tutorials by [Chris Albon](https://chrisalbon.com/) but they are more focused on a specific feature)*, but I'll just quickly go over the most important points. In `pandas` tabular data is stored in a `DataFrame` and these objects allow us to have named columns and rows and to easily subset data with these names. Ok so let's create a `DataFrame` from a subset of our iris data.

~~~python
>>> import pandas as pd
>>> iris_df = pd.DataFrame(iris_data, columns = column_names)
>>> iris_df
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) species
0                5.1               3.5                1.4               0.2  setosa
1                4.9               3.0                1.4               0.2  setosa
2                4.7               3.2                1.3               0.2  setosa
3                4.6               3.1                1.5               0.2  setosa
4                5.0               3.6                1.4               0.2  setosa
~~~
Ok so we have our `DataFrame`, and now we can select just the species column for example with:
~~~python
>>> iris_df['species']
0    setosa
1    setosa
2    setosa
3    setosa
4    setosa
Name: species, dtype: object
~~~
Or we can get the samples which have $$sepal\ width \leq 3.1$$:
~~~python
>>> iris_df[iris_df['sepal width (cm)'] <= 3.1]
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) species
1                4.9               3.0                1.4               0.2  setosa
3                4.6               3.1                1.5               0.2  setosa
~~~
So it is easy to see how this makes understanding what the code is doing easier and that's why I'm using `DataFrame`s instead of arrays and indices. This is however far from all `pandas` can do *(duh...)*, and it's just the basics of the basics of the basics.   

# Representing a tree. 

So what is a tree?  
A tree is a set of nodes, so we need to define a `Node` object which will store some properties about the node, as well as it's left and right children nodes.  

What does a node of our decision tree need?  
A pandas DataFrame to represent the training data at this node, the target feature *(in the case of our iris example that would be the "species" feature)*. We need to have a left and right child as well, and if the node is a leaf node it needs to be able to give a prediction. This gives us the following `Node` class definition, where we feed the dataframe as well as the target feature upon initialization.  

~~~python
class Node:

    def __init__(self, data, target):
        self.data = data # Dataframe with features and target
        self.target = target # name of target
        self.left = None
        self.right = None
        self.prediction = None
~~~

<br/>

Ok so now we have our `Node` class, how do we represent a tree? where do we keep all the nodes we will need. Well actually we do not need to create a `Tree` class:  
since each node stores its left and right children, we can access any and all nodes of the tree from the root node. So the whole tree can just be represented by the root node. For example if we want to get the *versicolor* node in our simple tree we can access it from the root node with: `root.right.left`
![small tree]({{site.baseurl}}/assets/images/simple_tree.svg)  

# getting the splits
Ok, so we can just concentrate ourselves on the `Node` class. So if you remember how the algorithm works *(if not [here is the post]({{site.baseurl}}{%link _posts/2019-02-27-the-CART-algorithm.markdown%}) where I explain it)*, we are going to need a way to find splits in our data. As we said in previous parts, there are categorical and numerical splits, so we need a way to determine if a feature is categorical or numerical, fortunately `pandas` has us covered, and we can make a simple function:
~~~python
from pandas.api.types import is_categorical, is_string_dtype, is_bool

def check_categorical(data):
    truth = is_categorical(data) | is_string_dtype(data) | is_bool(data)
    return truth
~~~

So now that we can distinguish the two we can write this function that gets all possible splits in our dataset, and returns them as a dictionnary

~~~python
def get_splits(self):
    features = self.data.columns.drop(self.outcome)
    all_splits = {} 

    for feature in features:

        if check_categorical(self.data[feature]):
            all_splits.update(self.get_categorical_splits(feature))
        else:
            all_splits.update(self.get_numerical_splits(feature))

    return all_splits
~~~

As you can see gets all the features in our dataset (except the outcome), loops over them and checks if the feature is categorical or numerical. Depending on th feature type it calls a given splitting method. So how do we get our categorical and numerical splits? For the numerical splits we have this function :  
~~~python
def get_numerical_splits(self, feature):
    splits = {}
    uniques = self.data[feature].unique()
    for value in uniques:
        if value != max(uniques):
            splits[(feature, value, 'numerical')] = self.data[self.data[feature] <= value]
    return splits
~~~
This returns all possible numerical splits in a dictionnary where the key is a `tuple` of the feature name, the value on which the split is done and the type of split, and for value the data that goes to the left side of the split *(the data that respects the split condition)*.  
For categorical features we are not going to follow exactly what I said in [part 2](), indeed the total number of splits is: $$2^{k-1} - 1$$, with $$k$$ the possible values of our feature, this can get huge very quickly. For example, a categorical feature with 25 levels (25 brands of car, or 25 different languages, whatever...), which can be easily attained in some datasets, would result in $$33554432$$ splits to evaluate, and that's just for one feature in one node. So this can get out of hand very quickly and slow our program to a crawl. So I'm going to make an executive decision here and say we will only consider splits made by a single level, for example `brand = Ford` and eliminate all splits made by combinations of levels: `(brand = Ford) or (brand = chevrolet)`. This brings us back to a nice $$k$$ possible splits. So we can add this method to get categorical splits: 
~~~python
def get_categorical_splits(self, feature):
    splits = {}
    for unique in self.data[feature].unique():
        splits[(feature, unique, 'categorical')] = self.data[self.data[feature] == unique]
    return splits
~~~

Ok all done, so now at a single node we can get all the dataset splits we want to evaluate by calling the `get_splits()` method. 

# Computing impurity
Next we need a way to calculate the impurity of a split, in our case is going to be the Gini index. For reminder the Gini index $$G$$ for a node $$t$$ is defined as:  
$$
G(t) = 1 - \sum^k_{i=1} p_i^2
$$

Where $$p_i$$ is the proportion of samples of class $$i$$ in the node data, and $$k$$ is the number of different classes. So let's add a method to our `Node` class to do just that:  

~~~python
def get_gini(self, data):
    proportions = data[self.outcome].value_counts(normalize=True)
    return 1 - (proportions ** 2).sum() # the ** applies to all elements of the column 
~~~

`value_counts()` is a `pandas` method that gets all unique values in a given column and returns their counts, the `normalize` option makes it return proportions instead of counts. The next step is computing the decrease in impurity $$\Delta i$$ (see part 2 for formula). 

~~~python
def get_delta_i(self, subset):
    gini = self.get_gini(self.data)

    left = subset 
    right = self.data.drop(subset.index, axis=0)

    p_left = len(left) / len(self.data)
    p_right = 1 - p_left

    sub_left = p_left * self.get_gini(left)
    sub_right = p_right * self.get_gini(right)

    return gini - sub_left - sub_right
~~~

Here `subset` is the data in the entries of the dictionnary given by `get_splits()`, so it's just the left side of the splits. To get the right side we take the whole data of the node and get rid of all the rows that are in the left split (`subset`).  
So now we can get the best split by looping over all possible splits, ad returning the one with the highest value of $$\Delta i$$:  
~~~python
def get_best_split(self):
    all_splits = self.get_splits()
    delta_is = {}

    for key, split in all_splits.items():
        delta_is[key] = self.get_delta_i(split)

    return max(delta_is, key=delta_is.get)
~~~

