# 0.7996654462820352
# 0.7996654462820352

import numpy as np
import csv

def import_training_data(train_X_file_path, train_Y_file_path):
    X = np.genfromtxt(train_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt(train_Y_file_path, delimiter=',', dtype=np.int32)
    return X, Y


def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()


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

def initialize_weights(n):
    w = np.zeros((n,1))
    return w, 0


def sigmoid(Z):
    sigmoid_of_Z = 1 / (1 + np.exp(-Z))

    return sigmoid_of_Z


def compute_cost(X, Y, W, b):
    m = len(X)
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    A[A==1] = 0.99999
    A[A==0] = 0.00001
    loss = np.dot(Y.T, np.log(A)) + np.dot((1-Y.T), np.log(1-A))
    cost = (-1/m) * np.sum(loss)
    return cost


def compute_gradient_of_cost_function(X, Y, W, b):
    m = len(X)
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    dz = A-Y

    dw = (1/m) * (np.dot(X.T, dz))
    db = (1/m) * (np.sum(dz))
    return dw, db


def optimize_weights(X, Y, W, b, learning_rate, cost_diff_bw_iterations):
    previous_iteration_cost = 0
    iter_no = 0
    while True:
        iter_no += 1
        dw, db = compute_gradient_of_cost_function(X, Y, W, b)
        W = W - learning_rate * dw
        b = b - learning_rate * db
        cost = compute_cost(X, Y, W, b)

        if iter_no % 1000 == 0:
            print(iter_no, cost)
      
        if iter_no > 1 and (previous_iteration_cost - cost) < 0:
            print("cost is increasing!!!")
            break
  
        if abs(previous_iteration_cost - cost) <= cost_diff_bw_iterations:
            print(iter_no, previous_iteration_cost, cost)
            break
        
        previous_iteration_cost = cost
    
    return W, b


def train_model(X, Y):
    X = preprocess(X)
    n = len(X[0])
    m = len(X)
    Y = np.reshape(Y, (m, 1))
    #print(X[ : 15, 4: 7])
    W, b = initialize_weights(n)
    
    W, b = optimize_weights(X, Y, W, b, 3.91, 0.0000001)
    
    final_model = np.vstack((W,b))
    save_model(final_model, 'WEIGHTS_FILE.csv')  


if __name__ == '__main__':
    X, Y = import_training_data('train_X_pr.csv', 'train_Y_pr.csv')
    train_model(X, Y)