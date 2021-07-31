import numpy as np
import csv
import json
from math import log
import pickle

def import_training_data(train_X_file_path, train_Y_file_path):
    X = np.genfromtxt(train_X_file_path, delimiter='\n', dtype=str)
    Y = np.genfromtxt(train_Y_file_path, delimiter=',', dtype=np.int32)
    return X, Y


def remove_special_chars_other_than_space(X):
    X_without_spl_chars = []
    for x in X:
        i = 0
        x_with_no_spl_chars = ""
        for i in range(len(x)): 
            if (ord(x[i]) >= ord('A') and
                ord(x[i]) <= ord('Z') or 
                ord(x[i]) >= ord('a') and 
                ord(x[i]) <= ord('z') or
                ord(x[i]) == ord(' ')):
                x_with_no_spl_chars += x[i] 
        X_without_spl_chars.append(x_with_no_spl_chars)

    return X_without_spl_chars
    

def preprocess(X):
    X = remove_special_chars_other_than_space(X)
    
    for i in range(len(X)):
        X[i] =' '.join(X[i].split())
        X[i] = X[i].lower()
    
    return X


def class_wise_words_frequency_dict(X, Y):
    class_wise_frequency_dict = dict()
    for i in range(len(X)):
        words = X[i].split()
        y = Y[i]
        for word in words:
            if y not in class_wise_frequency_dict:
                class_wise_frequency_dict[y] = dict()
            
            if word not in class_wise_frequency_dict[y]:
                class_wise_frequency_dict[y][word] = 0
            
            class_wise_frequency_dict[y][word] += 1
    
    return class_wise_frequency_dict


def calculate_class_wise_denominator(X, Y, class_wise_frequency_dict, smoothing_parameter):
    classes = list(set(Y))
    ls = []
    for cl in classes:
        ls += set(class_wise_frequency_dict[cl].keys())
    len_vocab = len(set(ls))
    #print(len_vocab)
    class_wise_denominators = dict()
    for cl in classes:
        no_of_words = sum(class_wise_frequency_dict[cl].values())
        class_wise_denominators[cl] = no_of_words + (len_vocab * smoothing_parameter)
    return class_wise_denominators


def save_model(class_wise_frequency_dict, class_wise_denominator, prior_probabilities, classes, smoothing_parameter):
    model = dict()
    model["trained_class_wise_frequency_dict"] = class_wise_frequency_dict
    model["trained_class_wise_denominator"] = class_wise_denominator
    model["trained_prior_probabilities"] = prior_probabilities
    model["trained_classes"] = classes
    model["laplace_parameter"] = smoothing_parameter
    pickle.dump(model, open('MODEL_FILE.sav', 'wb'))


def compute_likelihood(x, c, class_wise_frequency_dict, class_wise_denominators, laplace_smoothing):
    denominator = class_wise_denominators[c]
    log_likelihood = 0
    for word in x.split():
        if word in class_wise_frequency_dict[c]:
            p = (class_wise_frequency_dict[c][word] + laplace_smoothing)/denominator
        else:
            p = laplace_smoothing/denominator
        log_likelihood += log(p)
    return log_likelihood


def predict_class(x, classes, class_wise_frequency_dict, class_wise_denominators, prior_probabilities, laplace_smoothing):
    P = []
    C = []
    for c in classes:
        p = log(prior_probabilities[c]) + compute_likelihood(x, c, class_wise_frequency_dict, class_wise_denominators, laplace_smoothing)
        P.append(p)
        C.append(c)
    index = np.argmax(P)
    return C[index]


def calculate_accuracy(pred_Y, actual_Y):
    correct = 0
    for i in range(len(pred_Y)):
        correct += (pred_Y[i] == actual_Y[i])
    return correct/len(pred_Y)


def train_and_validate(X, Y, test_start, test_end):
    testing_X_data = X[test_start : test_end]
    testing_Y_data = Y[test_start : test_end]

    train_X = X.copy()
    train_Y = Y.copy()
    train_X[test_start : test_end] = []
    train_Y[test_start : test_end] = []

    training_X_data = train_X
    training_Y_data = train_Y

    class_wise_frequency_dict = class_wise_words_frequency_dict(training_X_data, training_Y_data)
    
    classes = list(set(training_Y_data))

    smoothing_parameter = 1

    prior_probabilities = dict()
    for c in classes:
        prior_probabilities[c] = (sum(y == c for y in training_Y_data))/len(training_Y_data)

    class_wise_denominator = calculate_class_wise_denominator(training_X_data, training_Y_data, class_wise_frequency_dict, smoothing_parameter)

    pred_Y = []
    for x in testing_X_data:
        pred_Y.append(predict_class(x, classes, class_wise_frequency_dict, class_wise_denominator, prior_probabilities, smoothing_parameter))

    acc = calculate_accuracy(pred_Y, testing_Y_data)
    return acc


def train(X ,Y):
    X = preprocess(X)
    Y = list(Y)
    """
    testing = int(len(X) * 0.25)
    accuracy = 0
    accuracy += train_and_validate(X, Y, 0, testing)
    accuracy += train_and_validate(X, Y, testing, 2*testing)
    accuracy += train_and_validate(X, Y, 2*testing, 3*testing)
    accuracy += train_and_validate(X, Y, 3*testing, 4*testing)
    #accuracy += train_and_validate(X, Y, 4*testing, 5*testing)
    
    avg_accuracy = accuracy/4
    print(avg_accuracy)
    """
    class_wise_frequency_dict = class_wise_words_frequency_dict(X, Y)
    
    classes = list(set(Y))

    smoothing_parameter = 1

    prior_probabilities = dict()
    for c in classes:
        prior_probabilities[c] = (sum(y == c for y in Y))/len(Y)

    class_wise_denominator = calculate_class_wise_denominator(X, Y, class_wise_frequency_dict, smoothing_parameter)
    
    save_model(class_wise_frequency_dict, class_wise_denominator, prior_probabilities, classes, smoothing_parameter)


if __name__ == '__main__':
    X, Y = import_training_data('train_X_nb.csv', 'train_Y_nb.csv')
    train(X, Y)