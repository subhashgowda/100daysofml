# 100 DAYS OF ML
---

## DAY1 : DATA PROCESSING

![day 1](https://user-images.githubusercontent.com/17926361/51560346-4c023f00-1eaa-11e9-854f-3ce6fefd059b.jpg)

For [code](https://github.com/subhashgowda/100daysofml/blob/master/Data%20Processing/Datapreprocessing.ipynb) and  [dataset](https://github.com/subhashgowda/100daysofml/blob/master/Data%20Processing/Data.csv) <--- Click

---
### REGRESSION

-> Regression is used when the prediction have "infinite posibilities".

Types of regression

>> Simple Linear Regression

>> Multiple Linear Regression 

>> Polynomial Regression

---
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
  
  
