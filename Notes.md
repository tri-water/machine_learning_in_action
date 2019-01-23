### Steps for Developing a Machine Learning Applications

1. Collect data
2. Prepare data
3. Analyse data
4. Train the algorithm
5. Test the algorithm
6. Use it

### Test the Performance of a classifier
1. Error rate: the number of mis-classifying over the total number of tests

### Information gain
#### Definition of the information for symbol $x_i$
$$l(x_i)=\log_{2}{p(x_i)}$$
where $p(x_i)$ is the probability of choosing this class
Shannon entropy, which is the expected value of all the information of all posible values of the classes:
$$H=-\sum_{i=1}^{n} p(x_i) \log_2{p(x_i)}$$
The higher the entropy, the more mixed up the data is.

#### Sigmoid function
$$\sigma(z)=\frac{1}{1+e^{-z}}$$
The input to the sigmoid function:
$$z=w_0x_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$$
Cost function:

cross entropy: $l\times [-\sigma (z)] + (1-l)[-(1-\sigma (z))]$

where $l$ is the true class labels (0 or 1)

### Support vector machines
#### linearly separable
If there are two groups of data, and the data points are separated enough that you could draw a straight line on the figure with all the points of one class on one side of the line and all the points of the other class on the other side of the line, then we say the data is linearly separable.

The decision boundary used to separate the dataset is called a *separating hyperplane*.

SVMs are binary classifiers. You'll need to write a little more code to use an SVM on a problem with more than two classes.

The target for the optimisation is:
1. find the points with the smallest margin
2. when the points with the smallest margin are found, maximize that margin
   
This can be written as:

$$\operatorname{arg\, max}_{w, b}\{\operatorname{min}_n(label\cdot(w^T+b))\cdot\frac{1}{\parallel w \parallel}\}$$

