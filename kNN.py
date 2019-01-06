import numpy as np
import operator
import pandas as pd
import os


def create_dataset():
    group = np.array([[1.0, 1.1],
                      [1.0, 1],
                      [0., 0.],
                      [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(x_in, data_set, labels, k):
    x_in = np.array(x_in)
    data_set = np.array(data_set)
    # Calculate distance
    sq_diffs = (data_set - x_in)**2
    distances = np.sqrt(sq_diffs.sum(axis=1))
    # Voting with lowest k distances
    classes = pd.DataFrame.from_dict({'distances': distances,
                                      'labels': labels})
    classes.sort_values(by='distances', ascending=True, inplace=True)
    classes = classes.iloc[:k, :]
    # the counts of each labels in the smallest k distances
    counts = classes['labels'].value_counts()
    # Return the majority class
    return counts.idxmax()


def file2matrix(loc, file_name):
    with open(loc + '/' + file_name, 'r') as file:
        return_mat = list()
        return_labels = list()
        for line in file.readlines():
            line = line.strip().split('\t')
            return_mat.append(line[:-1])
            return_labels.append(line[-1])

    try:
        return_mat = np.array(return_mat, dtype=np.float32)
    except ValueError:
        return_mat = np.array(return_mat, dtype=object)
    return_labels = np.array(return_labels)
    return return_mat, return_labels


def auto_norm(dataset):
    """normalize the data to values between 0 and 1."""

    min_vals = dataset.min(axis=0)
    max_vals = dataset.max(axis=0)
    ranges = max_vals
    norm_data_set = (dataset - min_vals)/ranges
    return norm_data_set, ranges, min_vals


def dating_class_test(loc, file_name):
    ho_ratio = 0.1
    dating_data_mat, dating_labels = file2matrix(loc, file_name)
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_samples = int(ho_ratio * m)
    error_count = 0
    for i in range(num_test_samples):
        label_pred = classify(norm_mat[i, :], norm_mat[num_test_samples:, :], 
                              dating_labels[num_test_samples:], 3)
        print('The predicted label: {0}, the true label: {1}'.format(label_pred,
                                                                     dating_labels[i]))
        if label_pred != dating_labels[i]:
            error_count += 1
    print('The total error rate is: {0:%}'.format(error_count/num_test_samples))


def img2vector(loc, file_name):
    with open(loc + '/' + file_name, 'r') as file:
        return_vector = []
        for line in file.readlines():
            line = list(line.strip())
            return_vector += line
    return_vector = np.array(return_vector, dtype=np.int32)
    return return_vector


def hand_writing_class_test(testing_loc, training_loc):
    training_files = [file for file in os.listdir(training_loc) 
                if os.path.isfile(os.path.join(training_loc, file))]
    
    training_mat = list()
    training_labels = list()

    for file in training_files:
        training_mat.append(img2vector(training_loc, file))
        training_labels.append(int(file[0]))
    training_mat = np.array(training_mat)
    training_labels = np.array(training_labels)

    testing_files = [file for file in os.listdir(testing_loc) 
                if os.path.isfile(os.path.join(training_loc, file))]
    testing_mat = list()
    testing_labels = list()

    for file in testing_files:
        testing_mat.append(img2vector(testing_loc, file))
        testing_labels.append(int(file[0]))
    testing_mat = np.array(testing_mat)
    testing_labels = np.array(testing_labels)

    error_counts = 0
    num_testing = len(testing_labels)

    for i in range(num_testing):
        label_pred = classify(testing_mat[i], training_mat, training_labels, 3)
        label_true = testing_labels[i]

        print('predicted label: {0}, true label: {1}'.format(label_pred, label_true))
        if label_pred != label_true:
            error_counts += 1
    print('error rates: {0: .2%}'.format(error_counts/num_testing))


    

