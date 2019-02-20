# MACHINE LEARNING
   Based on Avain Jain's 100 days of ml code**
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
## DEEP LEARNING

Deep learning is a sub-field of machine learning dealing with algorithms inspired by the structure 
and function of the brain called artificial neural networks.
Deep learning algorithms are similar to how nervous system structured where each neuron connected each other and passing information.
Geoffrey Hinton is known as "father of Neural Networks".

![1_jvbomzzzouv7rhu3ergbrw](https://user-images.githubusercontent.com/17926361/53065082-2d668680-34f0-11e9-8baa-b22dcc3f82eb.jpeg)

Deep learning models work in layers and a typical model atleast have three layers. Each layer accepts the information from previous and pass it on to the next one.

![1_io_kq3nkdwnhng6rgjlidq](https://user-images.githubusercontent.com/17926361/53066479-1460d400-34f6-11e9-9b42-3337e8e9a5fd.png)


![deep-learning-ai-machine-matrix2](https://user-images.githubusercontent.com/17926361/53068700-6528fa80-34ff-11e9-93fa-c03c414058ce.gif)

Deep learning models tend to perform well with amount ofdata wheras old machine learning models stops improving after a saturation point.
                   
 ![1_oesktupu54xd_gp7uwuocw](https://user-images.githubusercontent.com/17926361/53067091-ae298080-34f8-11e9-8abc-ee733d1f3046.png)
 
 #### 1.Activation function
 Activation functions are functions that decide, given the inputs into the node, what should be the node’s output? Because it’s the activation function that decides the actual output, we often refer to the outputs of a layer as its “activations”.
        
 One of the simplest activation functions is the Heaviside step function. This function returns a 0 if the linear combination is less than 0. It returns a 1 if the linear combination is positive or equal to zero.
        
![1_xwblayqrqdehev-j3j6t2q](https://user-images.githubusercontent.com/17926361/53069452-129d0d80-3502-11e9-9a9f-8d3beb715e3d.png)
        
The output unit returns the result of f(h), where h is the input to the output unit.

![0_kethx4mtzfu8_0se](https://user-images.githubusercontent.com/17926361/53070625-ca7fea00-3505-11e9-91c3-14519d089c9f.png)


![iicbq](https://user-images.githubusercontent.com/17926361/53070555-90164d00-3505-11e9-8f00-f4da1187b763.gif)

Some Activation functions are listed below:

1. A Threshold function is also known as a Step function. Here we set a threshold value and if the Y value(output) is greater than the threshold value, the function is activated and fired, else it is not fired.


![main-qimg-ff5c2723500aa15e26b6fb1d9dd16534](https://user-images.githubusercontent.com/17926361/53071101-3c0c6800-3507-11e9-9981-98a7dff14ca4.png)


2. Sigmoid function - Sigmoid is another very common activation function which is used to predict the probability as an output. The output of this function always lies between 0 and 1. Sigmoid is used in hidden layers as well as in the output layers where the target is binary.
 
 ![image](https://user-images.githubusercontent.com/17926361/53071156-6c540680-3507-11e9-9323-c8e44089d4e1.png)
 
3. ReLU function - ReLU is one of the most widely used activation function. It stands for Rectified Linear Unit. It gives an output of X, if X is positive and 0 otherwise. ReLU is often used in the hidden layers.

![image](https://user-images.githubusercontent.com/17926361/53071243-a58c7680-3507-11e9-9faf-ed6f4933cea6.png)

4.Hyperbolic / Tanh function - Tanh function is similar to a Sigmoid function but is bound between the range (-1, 1). It is also used in the hidden layers as well as in the output layer.

![image](https://user-images.githubusercontent.com/17926361/53071321-e3899a80-3507-11e9-9d92-28a812ce306f.png)

5. Softmax function - Softmax function is generally used in the output layer. It converts every output to been in the range of 0 and 1, just like the Sigmoid function. But it divides each output such that the total sum of the outputs is equal to 1.

![image](https://user-images.githubusercontent.com/17926361/53071441-40855080-3508-11e9-8bb1-23b04a3b730d.png)

#### 2. Weights

When input data comes into a neuron, it gets multiplied by a weight value that is assigned to this particular input. For example, the neuron above university example have two inputs, tests for test scores and grades, so it has two associated weights that can be adjusted individually.

Use of weights

These weights start out as random values, and as the neural network learns more about what kind of input data leads to a student being accepted into a university, the network adjusts the weights based on any errors in categorization that the previous weights resulted in. This is called training the neural network.

Remember we can associate weight as m(slope) in the orginal linear equation.

y = mx+b

#### 3.Bias
Weights and biases are the learnable parameters of the deep learning models.

Bias represented as b in the above linear equation.

![image](https://user-images.githubusercontent.com/17926361/53071711-06687e80-3509-11e9-9d84-ef8a058f24fb.png)



#### 4. Neural Networks

Deep learning is a sub-field of machine learning dealing with algorithms inspired by the structure and function of the brain called artificial neural networks. I will explain here how we can construct a simple neural network from the example. 

**Artificial neural network**


It is truly said that the working of ANN takes its roots from the neural network residing in human brain. ANN operates on something referred to as Hidden layers. These hidden layers are similar to neurons. Each of these hidden layers is a transient form which has a probabilistic behavior. A grid of such hidden layers act as a bridge between the input and the output.

![image](https://user-images.githubusercontent.com/17926361/53076659-31a59a80-3516-11e9-8ced-4eab4e705605.png)

Let’s try to understand what the above diagram actually means. We have a vector of three inputs and we intend to find the probability that the output event will fall into class 1 or class 2. For this prediction we need to predict a series of hidden classes in between (the bridge). The vector of the three inputs in some combination predicts the probability of activation of hidden nodes from 1 – 4. The probabilistic combination of hidden state 1-4 are then used to predict the activation rate of hidden nodes 5-8. These hidden nodes 5-8 in turn are used to predict hidden nodes 9-12, which finally predicts the outcome. The intermediate latent states allows the algorithm to learn from every prediction.

![image](https://user-images.githubusercontent.com/17926361/53076870-c27c7600-3516-11e9-94e9-3ba656b10278.png)


![image](https://user-images.githubusercontent.com/17926361/53077046-2e5ede80-3517-11e9-9d5f-40d384a6e04c.png)


**Back propagation**


**Gradient Descent**




