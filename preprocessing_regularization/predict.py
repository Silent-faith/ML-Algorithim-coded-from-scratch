import numpy as np
import csv
import sys

from validate import validate


def sigmoid(Z):
    sigmoid_of_Z = 1 / (1 + np.exp(-Z))

    return sigmoid_of_Z


def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights


def apply_one_hot_encoding(X):
    nominal_labels = list(set(X))
    nominal_labels.sort()
    nominal_labels = np.array(nominal_labels)
    #print(nominal_labels)
    one_hot_encoded_array = []
    for label in X:
        one_hot_encoded_array.append(list(np.where(nominal_labels == label, 1, 0)))
    return one_hot_encoded_array


def convert_given_cols_to_one_hot(X, column_indices):
    one_hot_encoded_X = np.zeros((len(X), 1))
    
    start = 0
    for curr_index in column_indices:
        one_hot_encoded_X = np.append(one_hot_encoded_X, X[ : , start:curr_index], axis = 1)

        start = curr_index + 1
        column = X[ : , curr_index]
        encoded = apply_one_hot_encoding(column)
        #print(np.shape(encoded))
        one_hot_encoded_X = np.append(one_hot_encoded_X, encoded, axis = 1)
    
    one_hot_encoded_X = np.append(one_hot_encoded_X, X[ : , start:], axis = 1)
    one_hot_encoded_X = one_hot_encoded_X[ : , 1 : ]
    return one_hot_encoded_X


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


def replace_null_with_zero(X, column_indices):
    new_X = np.zeros((len(X), 1))
    
    start = 0
    for curr_index in column_indices:
        new_X = np.append(new_X, X[ : , start:curr_index], axis = 1)

        start = curr_index + 1
        column = X[ : , curr_index]

        column[np.isnan(column)] = 0
        column = column.reshape((len(X), 1))
        new_X = np.append(new_X, column, axis = 1)
    
    new_X = np.append(new_X, X[ : , start:], axis = 1)
    new_X = new_X[ : , 1 : ]
    return new_X


def preprocess(X):
    X = replace_null_with_mean(X, [1, 2, 4, 5, 6])
    X = replace_null_with_zero(X, [0, 3])
    X = min_max_normalize(X, [1,2,4,5,6])

    X = convert_given_cols_to_one_hot(X, [0, 3])

    #print(np.shape(X))
    return X


def predict_target_values(X, weights):
    W = weights[:len(weights)-1]
    b = weights[len(weights)-1]

    W = W.reshape((len(W), 1))
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    A = A.flatten()
    A = np.where(A>=0.5, 1, 0)
    return A
    

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    test_X = preprocess(test_X)
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_pr.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_pr.csv") 