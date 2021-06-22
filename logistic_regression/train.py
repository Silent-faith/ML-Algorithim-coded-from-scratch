import csv
import math
import numpy as np

def import_dataset(train_X_file_path, train_Y_file_path):
    X = np.genfromtxt(train_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt(train_Y_file_path, delimiter=',', dtype=np.int32)
    return X, Y


def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()


def get_class_training_data(X, Y, class_label):
    class_X = np.copy(X)
    class_Y = np.copy(Y)
    class_Y = np.where(class_Y == class_label, 1, 0)
    return class_X, class_Y


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


def train_class_1(X, Y):
    X, Y = get_class_training_data(X, Y, 0)
    n = len(X[0])
    m = len(X)
    Y = np.reshape(Y, (m, 1))

    W, b = initialize_weights(n)
    W, b = optimize_weights(X, Y, W, b, 0.0075, 0.000001)

    final_model = np.vstack((W,b))
    
    save_model(final_model, 'Weights_for_class_1.csv')    


def train_class_2(X, Y):
    X, Y = get_class_training_data(X, Y, 1)
    n = len(X[0])
    m = len(X)
    Y = np.reshape(Y, (m, 1))

    W, b = initialize_weights(n)
    W, b = optimize_weights(X, Y, W, b, 0.007, 0.000001)

    final_model = np.vstack((W,b))
    
    save_model(final_model, 'Weights_for_class_2.csv')


def train_class_3(X, Y):
    X, Y = get_class_training_data(X, Y, 2)
    n = len(X[0])
    m = len(X)
    Y = np.reshape(Y, (m, 1))

    W, b = initialize_weights(n)
    W, b = optimize_weights(X, Y, W, b, 0.0072, 0.000001)

    final_model = np.vstack((W,b))
    
    save_model(final_model, 'Weights_for_class_3.csv')


def train_class_4(X, Y):
    X, Y = get_class_training_data(X, Y, 3)
    n = len(X[0])
    m = len(X)
    Y = np.reshape(Y, (m, 1))

    W, b = initialize_weights(n)
    W, b = optimize_weights(X, Y, W, b, 0.0065, 0.000001)

    final_model = np.vstack((W,b))
    
    save_model(final_model, 'Weights_for_class_4.csv')


if __name__ == '__main__':
    X, Y = import_dataset('train_X_lg_v2.csv', 'train_Y_lg_v2.csv')
    train_class_1(X, Y)
    print()
    train_class_2(X, Y)
    print()
    train_class_3(X, Y)
    print()
    train_class_4(X, Y)