import numpy as np
import csv
import sys
from sklearn.svm import SVC
import pickle


def import_training_data(train_X_file_path, train_Y_file_path):
    X = np.genfromtxt(train_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt(train_Y_file_path, delimiter=',', dtype=np.float64)
    return X, Y


def min_max_normalize(X, column_indices):
    for i in column_indices:
        column = X[ : , i]
        max_val = np.max(column)
        min_val = np.min(column)
        column = (column - min_val)/(max_val - min_val)
        X[ : , i] = column
    return X


def replace_null_with_mean(X, column_indices):
    
    new_X = np.zeros((len(X), 1))
    start = 0
    for curr_index in column_indices:
        new_X = np.append(new_X, X[ : , start:curr_index], axis = 1)
     
        start = curr_index + 1
        column = X[ : , curr_index]
        column_mean = np.nanmean(column)

        column[np.isnan(column)] = column_mean
        column = column.reshape((len(X), 1))
        new_X = np.append(new_X, column, axis = 1)
    
    new_X = np.append(new_X, X[ : , start:], axis = 1)
    new_X = new_X[ : , 1 : ]
    return new_X


def preprocess(X):
    X = replace_null_with_mean(X, [0,1,2,3,4,5,6,7,8,9,10])
    #X = min_max_normalize(X, [0,1,2,3,4,5,6,7,8,9,10])

    return X


def train_model(X, Y):
    X = preprocess(X)

    clf = SVC(C = 1000, kernel='rbf')
    clf.fit(X, Y)

    pickle.dump(clf, open('MODEL_FILE.sav', 'wb'))
    


if __name__ == '__main__':
    X, Y = import_training_data('train_X_svm.csv', 'train_Y_svm.csv')
    train_model(X, Y)