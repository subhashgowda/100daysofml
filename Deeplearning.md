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
