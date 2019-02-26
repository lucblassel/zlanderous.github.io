---
layout: post
title:  "What are decision trees?"
date:   2019-02-25 19:53:20 +0100
---

The first subject I want to tackle on this page is decision trees. What are they? How do they work? How can I make one?  
I am planning to make a small series, rangin from explaining the concept, to implementing a decision tree inference algorithm and hopefully all the way up to implementing Random Forests.  
All right let's get started.  


# Example data
We are going to use, as an example, de Fisher Iris dataset (more info [here](https://en.wikipedia.org/wiki/Iris_flower_data_set)) and we are going to build a tree that can separate the different types of irises (classes), which is called a classification tree. 

This dataset is a simple, standard dataset in machine learning and it is easy to understand. 
There are  150 samples *(ie. different flowers)* of irises, and each of these has 4 features and one label. The features are [petal](https://en.wikipedia.org/wiki/Petal) length and width, as well as [sepal](https://en.wikipedia.org/wiki/Sepal) length and width, all in centimeters. The label of each observation is simply 
the species of the observed iris, which can be *iris setosa*, *iris virginica* or *iris versicolor*. The goal of our tree is to help us determine what the caracteristics of these different species are, and how we can differentiate them. 
So let's look at a small subset of our data:  

| sepal length (x1) | sepal width (x2) | petal length (x3) | petal width (x4) |   species    |
|:-----------------:|:----------------:|:-----------------:|:----------------:|:------------:|
|        6.7        |        3.0       |        5.2        |        2.3       |  *virginica* |
|        5.0        |        3.2       |        1.2        |        0.2       |   *setosa*   |
|        5.5        |        2.5       |        4.0        |        1.3       | *versicolor* |
|        6.0        |        3.0       |        4.8        |        1.8       |  *virginica* |
|        4.6        |        3.1       |        1.5        |        0.2       |   *setosa*   |

A tree built on this subset (which we call the **training** or **learning** set), can look something like this:  
![a simple decision tree]({{site.baseurl}}/assets/images/simple_tree.svg)

Where the data is split at each node according to a condition on a feature, for example: *is the petal length lower or equal to 1.7cm?*  
This notion of splitting the data leads us quite well into our next section of seeing trees as partitions. 

# Trees are partitions
This is a quite fundamental concept of decision trees. We can imagine our dataset as a multidimensional space, and each internal node of the tree *(ie. a node with a condition not a terminal, leaf node)* is a partition that splits the space into two subspaces.  
Here our dataset is 4 dimensional, however it is a little complicated for use humans to visually understand 4 dimensions, so let's imagine our dataset with only two dimensions. We will restrict our dataset to only petal length and width *(the 2 features which were used in the simple decision tree above)*. Since it is only 2 dimensions we can easily represent it as a plane:  

![the iris dataset]({{site.baseurl}}/assets/images/iris_dataset_base.svg)

If we went all the way to have pure nodes we would get this tree which is overfitted:

![an overfitted tree]({{site.baseurl}}/assets/images/overfitted_tree.svg)