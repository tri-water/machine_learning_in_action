import pandas as pd
import numpy as np
import os

def load_dataset():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'help',
                    'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not

    return posting_list, class_vec


def create_vocab_list(dataset):
    vocab_set = set([])
    for document in dataset:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words_2_vec(vocab_list: list, input_sets):
    # If the word in vocab_list is also in input_set,
    # set the value of the corresponding position in the return_vec to 1

    if not(type(vocab_list) is list):
        vocab_list = list(vocab_list)
    if type(input_sets) is not list:
        input_sets = list(input_sets)
    if (type(input_sets[0]) is not list):
        input_sets = [input_sets]

    return_mat = np.zeros((len(input_sets), len(vocab_list)))

    for (i, input_set) in enumerate(input_sets):
        for word in input_set:
            if word in vocab_list:
                return_mat[i, vocab_list.index(word)] = 1
            else:
                print('the word: {0} in sample {1} is not in my Vocabulary'.format(word, str(i)))
    return return_mat


def train_NB(train_matrix, train_category):
    counts = np.zeros((len(train_matrix[0]), len(np.unique(train_category))))
    
    for i, sample in enumerate(train_matrix):
        category = train_category[i]
        counts[:, category] = (counts[:, category] + np.array(sample))
    
    total_counts = counts.sum(axis=0)
    # Given the document class, the probability of the word i
    # Add 1 to the nominators and 2 to the denominators
    # to avoid probability of 0
    p_vects = (counts + 1) / (total_counts[np.newaxis, :] + 2)
    # p_abusive: the probability that a document is an abusive document
    p_abusive = np.sum(train_category) / len(train_category)
    # take the natural logarithm of the probabilities
    # to covert multiplications to sum to avoid round-off error
    p_vects = np.log(p_vects)

    return p_vects, p_abusive


def classify_NB(vector2classify, p_vects, p_abusive):
    classes = list()
    for i in range(vector2classify.shape[0]):

        p1 = np.sum(vector2classify[i]*p_vects[:, 1]) + np.log(p_abusive)
        p0 = np.sum(vector2classify[i]*p_vects[:, 0]) + np.log(1-p_abusive)

        if p1 > p0:
            classes.append(1)
        else:
            classes.append(0)
    return classes


def testing_NB():
    posting_list, classes_list = load_dataset()
    my_vocal_list = create_vocab_list(posting_list)
    train_mat = set_of_words_2_vec(my_vocal_list, posting_list)
    p_vects, p_abusive = train_NB(train_mat, classes_list)
    test_entry = [['love', 'my', 'dalmation'], ['stupid', 'garbage']]
    test_mat = set_of_words_2_vec(my_vocal_list, test_entry)
    print(test_entry, 'classified as: ', classify_NB(test_mat, p_vects, p_abusive))
    test_entry = ['stupid', 'garbage']
    test_mat = set_of_words_2_vec(my_vocal_list, test_entry)
    print(test_entry, 'classified as: ', classify_NB(test_mat, p_vects, p_abusive))


def text_parse(big_string):
    import re
    list_of_tokens = re.split('\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok)>2]


def spam_test():
    """ Predict is an email is a spam email"""
    doc_list = []
    cls_list = []

    cwd = os.getcwd()

    for i in range(1, 26):
        with open(cwd + r'\data\email\ham\{0}.txt'.format(str(i)), 'r') as email:
            list_of_tokens = text_parse(email.read())
            doc_list.append(list_of_tokens)
            cls_list.append(0)
        with open(cwd + r'\data\email\spam\{0}.txt'.format(str(i)), 'r') as email:
            list_of_tokens = text_parse(email.read())
            doc_list.append(list_of_tokens)
            cls_list.append(1)

    my_vocal_list = create_vocab_list(doc_list)
    # Randomly select 40 emails for the training set  
    training_set_idx = np.random.choice(range(len(doc_list)), 40, replace=False)
    training_set = list()
    training_cls = list()

    for idx in reversed(sorted(training_set_idx)):
        training_set.append(doc_list[idx])
        training_cls.append(cls_list[idx])
        del doc_list[idx]
        del cls_list[idx]
    # The left emails are used as the test set
    test_set = doc_list
    test_cls = cls_list

    # Create token lists for taining set
    training_mat = set_of_words_2_vec(my_vocal_list, training_set)
    # Get the probability matrix
    p_vec, p_abusive = train_NB(training_mat, training_cls)
    # Create token lists for testing set
    test_mat = set_of_words_2_vec(my_vocal_list, test_set)
    # Classify the test set
    test_cls_pred = classify_NB(test_mat, p_vec, p_abusive)
    # Calculate the error rate
    errors = np.array(test_cls) != np.array(test_cls_pred)
    error_rate = errors.sum()/len(test_cls)
    # Print the error message(s)
    for i, error in enumerate(errors):
        if error:
            pred = test_cls_pred[i]
            actual = test_cls[i]
            print('error in predicting: ', test_set[i], 
                  '\npredicted:', pred, 'actual:', actual)

    print('the error rate is: {0:.2%}'.format(error_rate))

