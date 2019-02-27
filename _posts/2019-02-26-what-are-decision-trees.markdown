---
layout: post
mathjax: true
title:  "What are decision trees?"
date:   2019-02-26 12:53:20 +0100
excerpt_separator: <!--more-->
---

The first subject I want to tackle on this page is decision trees. What are they? How do they work? How can I make one?  
I am planning to make a small series, rangin from explaining the concept, to implementing a decision tree inference algorithm and hopefully all the way up to implementing Random Forests.  
All right let's get started.  
<!--more-->

# Example data
We are going to use, as an example, the Fisher Iris dataset (more info [here](https://en.wikipedia.org/wiki/Iris_flower_data_set)) and we are going to build a tree that can separate the different types of irises (classes), which is called a classification tree. 

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
{:refdef: style="text-align: center;"}
![a simple decision tree]({{site.baseurl}}/assets/images/simple_tree.svg)
{:refdef}
Where the data is split at each node according to a condition on a feature, for example: *is the petal length lower or equal to 1.7cm?*  
This notion of splitting the data leads us quite well into our next section of seeing trees as partitions. 

# Trees are partitions
This is a quite fundamental concept of decision trees. We can imagine our dataset as a multidimensional space, and each internal node of the tree *(ie. a node with a condition not a terminal, leaf node)* is a partition that splits the space into two subspaces.  
Here our dataset is 4 dimensional, however it is a little complicated for use humans to visually understand 4 dimensions, so let's imagine our dataset with only two dimensions. We will restrict our dataset to only petal length and width *(the 2 features which were used in the simple decision tree above)*. Since it is only 2 dimensions we can easily represent it as a plane:  

{:refdef: style="text-align: center;"}
![the iris dataset]({{site.baseurl}}/assets/images/iris_dataset_base.svg)
{:refdef}

Now let's represent the splits in our decision trees as lines that separate the plane into 2 sub-splanes:  
For the first split, we can draw a vertical line that shows  $petal\ length = 1.9$ that corresponds to the first split of our tree. As we can see in the figure below that perfectly separates *iris setosa* form the other two:  

{:refdef: style="text-align: center;"}
![the first split]({{site.baseurl}}/assets/images/iris_dataset_split_1.svg)
{:refdef}  

Now we can draw our second split, the horizontal line representing $petal\ width = 1.7$. This split only divides the *right* subspace of our first split, this is called [recursive partitioning](https://en.wikipedia.org/wiki/Recursive_partitioning). As you can see below, this separates our two remaining species, *versicolor* and *virginica* fairly well. However, near the boundary of this second split, we can see some of our *versicolor* flowers end un on the *virginica* side and *vice-versa*. 

{:refdef: style="text-align: center;"}
![the second split]({{site.baseurl}}/assets/images/iris_dataset_split_2.svg)
{:refdef}  

*Why don't we keep partitioning until there are no stragglers ?* you might ask.  
To uderstand that let's take a look at what the tree would look like if we kept splitting the dataset until each subspace was only filled with one species:  

{:refdef: style="text-align: center;"}
![an overfitted tree]({{site.baseurl}}/assets/images/overfitted_tree.svg)
{:refdef}  
As you can see this tree is a lot bigger and more complicated to take in, and it has splits that are very close to one another like $petal\ length = 4.9$ and $petal\ length = 4.8$ 

{:refdef: style="text-align: center;"}
![ovrfitted partitioning]({{site.baseurl}}/assets/images/iris_splits_overfit.svg)
{:refdef}  
*(N.b, you might have noticed in middle-top partition there appears to be only a sample of virginica, so why was is separated from the middle-right partition which is also virginica? In reality, because of the low precision of the dataset measurements, there are 2 versicolor and 1 virginica that have the same values for petal length and width, making them indistinguishable in the plane)*  

The decision tree we have here is very specific to our present dataset, it splits as much as possible to **fit** our **training data**, and what it is doing is called **overfitting**. This means that our tree is so specific to the data it was given, that if we get new samples *(ie. petal lengths and widths for new flowers not in the dataset)*, and we cycle them through the tree they might not end up detected as the right species. This is one of the reasons we want to restrict our decision tree to generalize it.  

There are a couple ways to restrict the tree, either by specifying a maximum depth value *(how many splits in a row you can do)*, or a threshold value *(if a split has a species that makes up more than 90% of it's samples we can call it "pure" and stop splitting for examples)* or by **pruning** the tree, meaning we make a decision tree that is av precise as possible, very overfitted, and then, according to a set of rules, we remove the branches and nodes that are too specific.  

## conclusion
OK so that was a quick introduction to trees, and my goal was to make you understand how a decision tree works, that it is just a set of nested partitions. Here we restricted ourselves to 2-D but it is easy to see how this carries to 3-D, we have a volume instead of a plane, and splits are surfaces instead of lines.  
Stay tuned for [part 2](https://zlanderous.github.io/2019/02/27/the-CART-algorithm.html) where I will go explain the CART algorithm for building these decision tres and implement it in `Python`.