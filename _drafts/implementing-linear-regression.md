---
layout: post
title: Implementing linear regression ?
mathjax: true
excerpt_separator: <!--more-->
tags: [linear model, machine-learning, Python, code, regression]
comments: true
feature: https://upload.wikimedia.org/wikipedia/commons/e/ed/Residuals_for_Linear_Regression_Fit.png
---

Today I want to explain linear regression. It is one of the simplest statistical learning models and can be implemented in only a couple lines of Python code, in an efficient manner. Being so simple however does not mean it is not useful, in fact it can be very practical to explore relationships between features in a dataset and make predictions on a target value. Therefore I think it's important to understand how the method works and how the different parameters have an effect on the outcome. 

<!-- more -->

# What is linear regression ?
Like it's name implies it is a regression method, therefore it learns a relationship between input features and a continuous numerical value, for example, given the number of rooms in an apartment and the crime rate in its neighborhood, guess the value of that apartment. The *linear* means that the inferred relationship between features and target value is linear: the target value is equal to the sum of the feature values multiplied by coefficients. What the method learns is those coefficients in order to fit to the training data as closely as possible.  

### What does it mean in practice ?
Ok, let's start. Let's assume, for simplicity's sake, that we have only one feature $x$ associated to a target value $y$ that is equal to $2\cdot x$ with noise added. If we plot this dataset we get this figure, with the red dotted line corresponding to the line of equation $ y = 2\cdot x$.

![example dataset]({{site.baseurl}}/assets/images/linear_regression/example_dataset_linear.svg)

The linear regression will make the assumption that the underlying equation of this dataset is:  

$$
y = \theta\cdot x
$$

and if the algorithm is implemented correctly it will return $ \theta \approx 2$.  

This is all well and good but what if the equation that generated the dataset is no longer $y = 2\cdot x$ but is affine and $y = 2\cdot x + 3$?  
In this case our model cannot get a good fit to the data if it assumes that the equation is of the form  $y = \theta\cdot x$. To take this into account we usually add a bias term to the features that is always equal to $1$ and learn 2 parameters $\theta_0$ for the bias and $\theta_1$ for the $x$ feature, this way the regression fits the following equation:  

$$
y = \theta_0\cdot 1 + \theta_1\cdot x
$$

and if the methods works well we should find $\theta_0\approx 3$ and $\theta_1\approx 2$.  

This works as well if we have several features $x_1$ and $x_2$ for example, in this case we fit the equation $y = \theta_0\cdot 1 + \theta_1\cdot x_1 + \theta_2\cdot x_2$, and so on and so forth.  
The general notation for this if we have $n$ features $x_1, x_2, \cdots, x_n$ becomes:  

$$
y = \theta_0 + \sum_{i=1}^{n} \theta_i\cdot x_i
$$

### How do we compute $\theta$ values ?
Ok so first let's agree on some notation, we have:  
 - $X$ the set of features for every training example, it is a table *(ie. matrix)* where each row is an example and each column is a feature. The dataset presented in above has 15 points and only one feature $x$ so in this case $X$ has 15 rows and only 1 column.
 - $y$ is a column *(ie. vector)* containing the target value for each training example. 
 - $x^{(i)}$ is the $i_{th}$ training example *(ie. the $i^{th}$ row of $X$)*, alternatively $y^{(i)}$ is the $i^{th}$ target value. 
 - $x_{j}$ represents the $j^{th}$ feature, *(ie. the $j^{th}$ column of $X$)*.
 - $\theta_i$ is the coefficient associated to $x_i$
 - $\Theta$ is a list *(ie. row vector)* containing all the coefficients $\theta_i$ for each feature $x_i$.
 - $\hat{y}$ is the column *(ie. vector)* of target values predicted by the linear regression model, and $\hat{y}^{(i)}$ its $i^{th}$ value. 

The way linear regression works is by finding the coefficients that result in the minimal distance between each target value and it's corresponding prediction, this distance is represented in 


# How can we implement linear regression ?