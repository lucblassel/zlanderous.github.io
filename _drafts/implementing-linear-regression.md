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

<!--more-->

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
 - $m$ the number of training examples and $n$ the number of features. 

The way linear regression works is by finding the coefficients that result in the minimal distance between each target value and it's corresponding prediction, this distance is represented in the following figure as green lines. As we can see the distance between the prediction *(dotted line)* and real values *(blue points)* is much lower when $\theta = 2$ *(which is the coefficient that generated the data)* on the left than when $\theta = 6$ on the right.  

![distance to fitted line]({{site.baseurl}}/assets/images/linear_regression/square_distance.svg)

In actuality what we want to minimize is the average of the sum of square distances, which will be our cost function $C$, which we can write this way:  

$$
    C = \frac{1}{2m}\sum^{m}_{i=1}(\hat{y}^{(i)} - y^{(i)})^2 \\
    with\ \ \hat{y}^{(i)} = \theta_0 + \sum_{j=1}^n \theta_j\cdot x^{(i)}_j\\

$$

How do we find the minimal value for $C$, the first thing to note here is that the only values we have any control over are the $\hat{y}^{(i)}$, and within that the only values we can change are the $\theta$ values, so we change these coefficients until we find the minimal value of $C$.  
Let's take our first dataset again where the target values $y$ are roughly twice the single feature $x$, without any bias. Therefore our linear regression algorithm only has to find one $\theta$. We can compute the cost for several different values of $\theta$ and see which one has the best cost which is show in the next figure:  
![cost in function on theta]({{site.baseurl}}/assets/images/linear_regression/cost_gradient_simple.svg)  
Here we can see that the cost is minimal when $\theta=2$ which is what we want. However it is not very practical, nor efficient, to compute the cost for many values of $\theta$, especially once we have more than one feature. So we need a way to avoid having to compute all costs and only the ones that will help us guess the minimum correctly, to do this we use *gradient descent*.  

For those that have done some algebra before, you'll know everything about derivatives and gradients, for the others I'll do a very quick recap. The derivative of a function $f(x)$ with regards to $x$, $\frac{df}{\delta x}$, describes the rate of change of $f$. So when $\frac{df}{\delta x}$ is positive, it means $f$ is growing when $x$ grows and when $\frac{df}{\delta x}$ is negative, $f$ is decreasing when $x$ grows.  
If you have a function that depends on several variables, like $g(x_1, x_2)$ we can compute several partial derivatives, each with regards to one of the variables of our function. In our case these partial derivatives would be: $\frac{\delta g}{\delta x_1}$ and $\frac{\delta g}{\delta x_2}$. The gradient is simply a vector containing all the partial derivatives of our function.  

If we compute the gradient of our cost function $C$ with regards to the different $\theta$ coefficients, we can tell how we need to adjust a particular $\theta$ value to lower the cost.  
For example, let's assume we have our gradient for $C$ composed of all the $\frac{\delta C}{\delta \theta_j}$. Let's imagine then that $\frac{\delta C}{\delta \theta_1} > 0$, this means that our cost is increasing with $\theta_1$, meaning that if we want to decrease $C$ we must also decrease $\theta_1$ by a certain amount. By doing this for all the $\theta_j$ we can decrease $C$ and if we repeat this step a large enough number of times we can converge to the minimal value of $C$. We know that when we reach the minimum of $C$, then the gradient value should all be equal to 0.  

If we try to summarize the steps that we will have to implement for linear regression we have:  
 1. compute the cost $C$ with all the $\theta_j$ values
 2. compute the gradient of $C$ with respect to all the $\theta_j$
 3. adjust all $\theta_j$ values according to the corresponding partial derivative: if $\frac{\delta C}{\delta \theta_j} < 0$ increase $\theta_j$ a little otherwise decrease it.  
   
We then repeat these steps with the adjusted values for all the $\theta_j$ until we have a gradient that is equal to 0, or more commonly if we reach a predefined maximum number of iterations.  

And that's it, this is the principle behind linear regression. However before starting the implementation there are a few things I need to explain still.

#### How do we adjust $\theta$ ?

Just above I mentioned that if $\frac{\delta C}{\delta \theta_j} < 0$ we need to increase $\theta_j$, but by how much do we increase it?  
The basic idea is that we subtract the partial derivative value from the current $\theta_j$ value, therefore if the partial derivative is negative by subtracting a negative to $\theta_j$ we increase it like we must. However in practice we multiply the partial derivative by a learning rate $\alpha$ that we must choose before subtracting it. This learning rate will allow us to adjust how fast our regression "learns", it's choice is very important, if $\alpha$ is too small we will barely change the $\theta$ values and our regression will find the minimum very slowly, and inversely if $\alpha$ is too big our cost will jump around everywhere and we might not find the minimum at all and our cost might actually increase as we adjust the $\theta$ values. I'll come back to this later. 

#### Can we write the math in another way?

In this part I will show you how to write all the math in matrix form, because this will make the programming part a lot easier and more efficient to run *(This means that you have to know the matrix math basics to understand this)*.  
As we said earlier we have our training examples $X$, where each example $x^{(i)}$ is a row and where each feature $x_j$ is a column. Therefore if we have $m$ examples and $n$ features, then $X$ is an $m\times n$ matrix. We also have our vector of target values $y$ which is an $m\times 1$ vector. If you recall earlier we had our predicted value for $x^{(i)}$:  

$$
\hat{y}^{(i)} = \theta_0 + \sum_{j=1}^{n}\theta_j\cdot x^{(i)}_j
$$

if we add a feature to 

# How can we implement linear regression ?

