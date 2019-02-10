# MACHINE LEARNING
---

##  DATA PROCESSING

![day 1](https://user-images.githubusercontent.com/17926361/51560346-4c023f00-1eaa-11e9-854f-3ce6fefd059b.jpg)

For [code](https://github.com/subhashgowda/100daysofml/blob/master/Data%20Processing/Datapreprocessing.ipynb) and  [dataset](https://github.com/subhashgowda/100daysofml/blob/master/Data%20Processing/Data.csv) <--- Click

------
### REGRESSION

-> Regression is used when the prediction have "infinite posibilities".

Types of regression

>> Simple Linear Regression

>> Multiple Linear Regression 

>> Polynomial Regression

------
### SIMPLE LINEAR REGRSSION

![day 2](https://user-images.githubusercontent.com/17926361/51587733-ca40fe80-1f07-11e9-8f7d-3dc959eab890.jpg)

Clik here for [Code](https://github.com/subhashgowda/100daysofml/blob/master/Simple%20linear%20regression/Simplelinearregression.ipynb) and [dataset](https://github.com/subhashgowda/100daysofml/blob/master/Simple%20linear%20regression/studentscores.csv)

SLR is used, when we have a "single input attribute" and we want to use linearity between variables.

2 Variables, Dependent variable (predicting) and independent variable / exploratory variable(observed)

Simple Linear Regression follows linear equation 
 
   >                                   Y = m x + C
   
   Y = line, Output variable to be predict
   
   x = input variable
   
   m = slope
    
   C = intercept
   
>> A line plot through variables, must be "passing through intercept and mean of (x,Y) cordinate, then that line is known as line of **best** **fit.** 

>> The goal is to find the best estimates for the coefficients to mininmize the errors in predicting y from x.

 **Slope**
  
  How x translates into Y value before bias.

>>    b1 / m = (Sum((x-mean(x)* (y-mean(y)))/(Sum((x-mean(x)^2))

**Intercept**

Point that cuts through x axis is intercept

>>   C = mean(y)-m(mean(x))

###### Assumptions of Linear regression


  :one: Model should be **Linear**
  
  :two: Errors should be **Independent**
  
  :three: Error terms should be **normally distributed**
  
  :four: **Homoscedacity** :Const variance on error terms
  
 -------
 ### MULTIPLE LINEAR REGRESSION
 
 ![day 3](https://user-images.githubusercontent.com/17926361/51655635-87dcf780-1fc3-11e9-9543-be2431c8e8c8.jpg)
 
 For greater numbers of independent variables, visual understanding is more abstract. For p independent variables, the data points (x1, x2, x3 …, xp, y) exist in a p + 1 -dimensional space. What really matters is that the linear model (which is p -dimensional) can be represented by the p + 1 coefficients β0, β1, …, βp so that y is approximated by the equation y = β0 + β1*x1 +....

 Click here for [code](https://github.com/subhashgowda/100daysofml/blob/master/Multiple%20linear%20regression/Multiple%20linear%20regression.ipynb) and [dataset](https://github.com/subhashgowda/100daysofml/blob/master/Multiple%20linear%20regression/50_Startups.csv)

-------
### CLASSIFICATION


-------
### LOGISTIC REGRESSION

![day 4](https://user-images.githubusercontent.com/17926361/51655768-25382b80-1fc4-11e9-9af5-842d68258864.jpg)

Click here for [Code](https://github.com/subhashgowda/100daysofml/blob/master/Logistic%20regression/Logisticregession.ipynb) and [dataset](https://github.com/subhashgowda/100daysofml/blob/master/Logistic%20regression/Social_Network_Ads.csv)

![data](https://user-images.githubusercontent.com/17926361/51655771-2701ef00-1fc4-11e9-9df0-509fa0b28e17.PNG)



--------


### K NEAREST NEIGHBOUR

![day 7](https://user-images.githubusercontent.com/17926361/51655923-ce7f2180-1fc4-11e9-92d3-50928e2a463c.jpg)


Click here for [Code](https://github.com/subhashgowda/100daysofml/blob/master/K%20Nearest%20Neighbours/Knearestneighbors.ipynb) and [Dataset](https://github.com/subhashgowda/100daysofml/blob/master/K%20Nearest%20Neighbours/Social_Network_Ads.csv)

--------
### SUPPORT VECTOR MACHINES/REGRESSION

![day 12](https://user-images.githubusercontent.com/17926361/51656917-c0cb9b00-1fc8-11e9-80a6-234805a3c114.jpg)

Click here for [code](https://github.com/subhashgowda/100daysofml/blob/master/Support%20Vector%20Regression/svr.py) and [Dataset](https://github.com/subhashgowda/100daysofml/blob/master/Support%20Vector%20Regression/Position_Salaries.csv).

------

### DECISION TREE 

![day 23](https://user-images.githubusercontent.com/17926361/51657127-8d3d4080-1fc9-11e9-8f32-aefc44e5d5ce.jpg)

Click here for [code](https://github.com/subhashgowda/100daysofml/blob/master/Decision%20tree%20Regression/decision_tree_regression.py) and [Dataset](https://github.com/subhashgowda/100daysofml/blob/master/Decision%20tree%20Regression/Position_Salaries.csv) for **regression**.

Click here for [code](https://github.com/subhashgowda/100daysofml/blob/master/Decision%20Tree%20Classification/decision_tree_classification.py) and [Dataset](https://github.com/subhashgowda/100daysofml/blob/master/Decision%20Tree%20Classification/Social_Network_Ads.csv) for **Classififcation**.

-------

### RANDOM FOREST

![day 33](https://user-images.githubusercontent.com/17926361/51657139-94fce500-1fc9-11e9-9d5c-7f836b57189d.jpg)

Click here for [code](https://github.com/subhashgowda/100daysofml/blob/master/Random%20forest%20regression/random_forest_regression.py) and [Dataset](https://github.com/subhashgowda/100daysofml/blob/master/Random%20forest%20regression/Position_Salaries.csv) for **regression**.
Click here for [code](https://github.com/subhashgowda/100daysofml/blob/master/Random%20Forest%20Classification/random_forest_classification.py) and [Dataset](https://github.com/subhashgowda/100daysofml/blob/master/Random%20Forest%20Classification/Social_Network_Ads.csv) for ckassifier.

-------
### KERNAL SVM

Click here for [code](https://github.com/subhashgowda/100daysofml/blob/master/KernelSVM/kernel_svm.py) and [Dataset](https://github.com/subhashgowda/100daysofml/blob/master/KernelSVM/Social_Network_Ads.csv).

------
### NAIVE BAYES

Click here for [code](https://github.com/subhashgowda/100daysofml/blob/master/Naive%20Bayes/naive_bayes.py) and [Dataset](https://github.com/subhashgowda/100daysofml/blob/master/Naive%20Bayes/Social_Network_Ads.csv).

------

### CLUSTERING

------
### K MEANS 

![day 43](https://user-images.githubusercontent.com/17926361/51657141-962e1200-1fc9-11e9-99d4-8c2b1c9c23e9.jpg)

An unsupervised learning algorithm (meaning there are no target labels) that allows you to identify similar groups or clusters of data points within your data. 

Algorithm
1. We randomly initialize the K starting centroids. Each data point is assigned to its nearest centroid.
2. The centroids are recomputed as the mean of the data points assigned to the respective cluster.
3. Repeat steps 1 and 2 until we trigger our stopping criteria.

optimizing for and the answer is usually Euclidean distance or squared Euclidean distance to be more precise. Data points are assigned to the cluster closest to them or in other words the cluster which minimizes this squared distance. We can write this more formally as:

![](https://cdn-images-1.medium.com/max/1600/1*UVJKdowZ9CHxvrII1IYolw.png)

Kmeans Visualize

We have defined k = 2 so we are assigning data to one of two clusters at each iteration. Figure (a) corresponds to the randomly initializing the centroids. In (b) we assign the data points to their closest cluster and in Figure c we assign new centroids as the average of the data in each cluster. This continues until we reach our stopping criteria (minimize our cost function J or for a predefined number of iterations). Hopefully, the explanation above coupled with the visualization has given you a good understanding of what K means is doing. 

![1_dpglfqy3obgpgubyqk9hiq](https://user-images.githubusercontent.com/17926361/52537070-55374b00-2d88-11e9-9416-e244d7f24faf.gif)

Click here for [code](https://github.com/subhashgowda/100daysofml/blob/master/Kmeans/kmeans.py) and [Dataset](https://github.com/subhashgowda/100daysofml/blob/master/Kmeans/Mall_Customers.csv).

-------

### Hierarchical Clustering 

![day 54](https://user-images.githubusercontent.com/17926361/51657143-975f3f00-1fc9-11e9-83d9-e4a836c568de.jpg)

Click here for [code](https://github.com/subhashgowda/100daysofml/blob/master/Hierarchical%20Clustering/hc.py) and [Dataset](https://github.com/subhashgowda/100daysofml/blob/master/Hierarchical%20Clustering/Mall_Customers.csv).

------

