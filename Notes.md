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