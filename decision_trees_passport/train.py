import csv
import numpy as np
import sys
import pickle
import csv


class Node:
    def __init__(self, predicted_class, depth):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.depth = depth
        self.left = None
        self.right = None


def import_training_set(train_X_file_path, train_Y_file_path):
    train_X = np.genfromtxt(train_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    train_Y = np.genfromtxt(train_Y_file_path, delimiter=',', dtype=np.float64)
    return train_X, train_Y


def calculate_gini_index(Y_subsets):
    ginies = []
    total_instances = sum(len(y) for y in Y_subsets)
    for y in Y_subsets:
        classes = sorted(set(y))
        no_of_instances = len(y)
        probabilities = [y.count(c)/no_of_instances for c in classes]
        gini = 1
        for p in probabilities:
            gini -= p**2
        ginies.append(gini)
    
    gini_index = 0
    i = 0
    for y in Y_subsets:
        gini_index += len(y)/total_instances * ginies[i]
        i += 1
    return gini_index


def split_data_set(data_X, data_Y, feature_index, threshold):
    left_X = []
    right_X = []
    left_Y = []
    right_Y = []
    i = 0
    for instance in data_X:
        if instance[feature_index] < threshold:
            left_X.append(instance)
            left_Y.append(data_Y[i])
        else:
            right_X.append(instance)
            right_Y.append(data_Y[i])
        i += 1
    
    return left_X, left_Y, right_X, right_Y


def get_best_split(X, Y):
    X = np.array(X)
    best_gini_index = 99999
    best_feature = 0
    best_threshold = 0
    for i in range(len(X[0])):
        thresholds = set(sorted(X[ : , i]))
        for t in thresholds:
            left_X, left_Y, right_X, right_Y = split_data_set(X, Y, i, t)
            if len(left_X) == 0 or len(right_X) == 0:
                continue
            gini_index = calculate_gini_index([left_Y, right_Y])
            if gini_index < best_gini_index:
                best_gini_index, best_feature, best_threshold = gini_index, i, t
    
    return best_feature, best_threshold





def construct_tree(X, Y, max_depth, min_size, depth):
    classes = list(set(Y))
    predicted_class = classes[np.argmax([np.sum(Y==c) for c in classes])]
    node = Node(predicted_class, depth)

    if len(set(Y)) == 1:
        return node
    
    if depth >= max_depth:
        return node
    
    if len(Y) <= min_size:
        return node
    
    feature_index, threshold = get_best_split(X, Y)
    
    if feature_index is None or threshold is None:
        return node
    
    node.feature_index = feature_index
    node.threshold = threshold
    
    left_X, left_Y, right_X, right_Y = split_data_set(X, Y, feature_index, threshold)

    node.left = construct_tree(np.array(left_X), np.array(left_Y), max_depth, min_size, depth+1)
    node.right = construct_tree(np.array(right_X), np.array(right_Y), max_depth, min_size, depth+1)
    return node


def train_model(X, Y):
    root_of_tree = construct_tree(X, Y, 10, 1, 0)
    pickle.dump(root_of_tree, open('MODEL_FILE.sav', 'wb'))


if __name__ == '__main__':
    X, Y = import_training_set('train_X_de.csv', 'train_Y_de.csv')
    train_model(X, Y)