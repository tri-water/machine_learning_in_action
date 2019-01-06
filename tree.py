import numpy as np
import pandas as pd
import pickle


def cal_shannon_ent(dataset):
    if not (type(dataset) is np.ndarray):
        dataset = np.array(dataset, dtype=object)
    # Counts the unique classes
    classes = dataset[:, -1]
    unique, counts = np.unique(classes, return_counts=True)
    label_counts = pd.Series(counts, index=unique)
    # Calculate probability and information for each class
    probs = label_counts / label_counts.sum()
    infos = np.log2(probs)
    # Calculate the Shannon entropy of the dataset
    ahannonEnt = -(infos*probs).sum()
    return ahannonEnt


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def split_data_set(dataset, col, value):
    """Split the dataset on a feature

    params:
    ======
    dataset: the dataset we'll split
    col: the feature we'll split on
    value: the value of the feature to return
    """
    if not (type(dataset) is np.ndarray):
        dataset = np.array(dataset, dtype=object)
    dataset = dataset[dataset[:, col]==value]
    # Delete the col that we split on
    return np.delete(dataset, col, axis=1)


def choose_best_feature_to_split(dataset):
    if not (type(dataset) is np.ndarray):
        dataset = np.array(dataset, dtype=object)
    num_features = dataset.shape[1] - 1
    base_entropy = cal_shannon_ent(dataset)

    best_info_gain = 0.
    best_feature = -1

    for i in range(num_features):
        feature_list = dataset[:, i]
        unique_values = np.unique(feature_list)
        new_entropy = 0.

        # Calculate the new entropy of the subdatasets splited on
        # feature i
        for value in unique_values:
            sub_dataset = split_data_set(dataset, i, value)
            prob = sub_dataset.shape[0] / dataset.shape[0]
            new_entropy += prob*cal_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        # Update the best feature_and the best_info_gain
        if info_gain > best_info_gain:
            best_feature = i
            best_info_gain = info_gain
    return best_feature


def majority_count(class_list):
    """return the class that occurs with the greatest frequency"""

    uniques, counts = np.unique(class_list, return_counts=True)
    counts_series = pd.Series(counts, index=uniques)
    counts_series.sort_values(inplace=True)
    return counts_series.index[-1]


def create_tree(dataset, labels):
    if not (type(dataset) is np.ndarray):
        dataset = np.array(dataset, dtype=object)
    if not (type(labels) is list):
        labels = list(labels)

    if len(np.unique(dataset[:, -1])) == 1:
        # If only one class is left, return the class
        return dataset[0, -1]
    if dataset.shape[1] == 1:
        # If no feature is left, return the class occuring most frequently
        return majority_count(dataset)

    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    # delete the labels used in this node
    del labels[best_feature]

    best_feature_values = np.unique(dataset[:, best_feature])

    tree = {best_feature_label: {}}

    for value in best_feature_values:
        # create trees for each sub-dataset
        sub_dataset = split_data_set(dataset, best_feature, value)
        tree[best_feature_label][value] = create_tree(sub_dataset, labels.copy())

    return tree


def classify(input_tree, labels, input_sample):
    if not (type(labels is list)):
        labels = list(labels)

    label = list(input_tree)[0]
    
    feature = labels.index(label)
    value = input_sample[feature]
    sub_tree = input_tree[label][value]

    if type(sub_tree) is dict:
        return classify(sub_tree, labels, input_sample)
    else:
        return sub_tree


def save_tree(input_tree, file_name):
    with open(file_name, 'wb') as pickle_out:
        pickle.dump(input_tree, pickle_out)


def load_tree(file_name):
    with open(file_name, 'rb') as pickle_in:
        tree = pickle.load(pickle_in)
    return tree



if __name__ == '__main__':
    from tree_plotter import retreve_tree
    data_loc = r'C:\Users\gao_x\OneDrive\tensorflow_study\machine_learning_in_action\data'
    file_name = 'lenses.txt'

    with open(data_loc + '/' + file_name) as file:
        lense_data = [line.strip().split('\t') for line in file.readlines()]

    lense_feature_labels = ['age', 'prescript', 'astigmatic', 'tearRate']

    lense_tree = create_tree(lense_data, lense_feature_labels)
    print(lense_tree)

