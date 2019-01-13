import tensorflow as tf
import os
import numpy as np

def load_dataset():
    """load data set from testSet.txt

    Returns:
        data_mat and label_mat: np.mat
    """
    cwd = os.getcwd()
    with open(cwd + r'\data\testSet.txt', 'r') as file:
        samples = file.readlines()

    samples = np.mat([data.split('\t') for data in samples], dtype=float)

    data_mat = samples[:, :2]
    label_mat = samples[:, 2]
    return data_mat, label_mat


def sigmoid(in_X):
    return 1.0/(1 + tf.exp(-in_X))


def cross_entropy(sigma, labels):
    return tf.reduce_mean(-labels*tf.log(sigma) \
                          - (1 - labels) * tf.log(1 - sigma))


def grad_ascent(data_mat, class_labels, epoch=150, learning_rate=0.0025,
                decay_rate = 0.97):
    """calculate the weights and bias for logistic regression using TensorFlow
    functions

    Params:
    =======
        data_mat: np.mat, the features of the samples
        class_labels: np.mat, the classes of the samples
        epoch: much many iterations to run
    Returns:
    ========
        weights and bias: np.ndarray
    """
    m, n = data_mat.shape
    tf_data_mat = tf.placeholder(dtype=tf.float32, shape=[None, n], name='data_mat')
    tf_class_labels = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='labels')
    W = tf.Variable(tf.ones([n, 1])/100, name='weight', dtype=tf.float32)
    bias = tf.Variable(tf.zeros(1), name='bias', dtype=tf.float32)
    in_X = tf.matmul(tf_data_mat, W) + bias
    sigma = sigmoid(in_X)

    cost = cross_entropy(sigma, tf_class_labels)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100, decay_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = data_mat.shape[0]
        
        for i in range(epoch):
            batch_idx = 0
            while(batch_idx < data_mat.shape[0]):
                sub_data_mat = data_mat[batch_idx: batch_idx + batch_size, :]
                sub_labels = class_labels[batch_idx: batch_idx + batch_size, :]
                feed_dict = {tf_data_mat: sub_data_mat, tf_class_labels: sub_labels}
                if i == 0:
                    cost_out = sess.run([cost], feed_dict=feed_dict)
                    print('start cost:', cost_out)
                _, cost_out = sess.run([optimizer, cost], feed_dict=feed_dict)
                batch_idx += batch_size
        weight_out, bias_out = sess.run([W, bias])
        print('end cost after {0} epoches:'.format(str(epoch)), cost_out)

    return weight_out, bias_out


def plot_best_fit():
    """Test the sigmoid regression process using data from testSet.txt"""
    
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_dataset()
    weight, bias = grad_ascent(data_mat, label_mat, 500, 0.1)
    
    n = data_mat.shape[0] # Number of samples

    label_array = np.squeeze(np.array(label_mat))

    data_mat1 = np.asarray(data_mat[label_array==1, :])
    data_mat2 = np.asarray(data_mat[label_array==0, :])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_mat1[:, 0], data_mat1[:, 1], s=30, c='red', marker='s')
    ax.scatter(data_mat2[:, 0], data_mat2[:, 1], s=30, c='green')
    x = np.arange(-3., 3., 0.1)
    y = (-bias - weight[0]*x)/weight[1]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classify(X_in, weights, bias):
    prob = X_in.dot(weights) + bias
    pred = prob.copy()
    pred[prob>0.5] = 1
    pred[prob<=0.5] = 0
    return pred


def read_horse_data(file_name):
    cwd = os.getcwd()
    with open(cwd + r'/data/' + file_name, 'r') as file:
        samples = file.readlines()
    samples = [sample.strip().split('\t') for sample in samples]
    samples = np.mat(samples, dtype=float)
    data_mat = samples[:, :-1]
    label_mat = samples[:, -1]
    return data_mat, label_mat


def colicTest():
    training_set, training_labels = read_horse_data('horseColicTraining.txt')
    weight, bias = grad_ascent(training_set, training_labels, epoch=10000)
    test_set, test_labels = read_horse_data('horseColicTest.txt')
    labels_pred = np.asarray(classify(test_set, weight, bias))

    true_preds = labels_pred == test_labels
    num_true = np.sum(true_preds)
    error_rate = 1- num_true / len(test_labels)
    print('the error rate of this test is: {0:.2%}'.format(error_rate))
    return error_rate



if __name__ == '__main__':
    plot_best_fit()