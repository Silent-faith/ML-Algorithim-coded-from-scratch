# importing the libraries
import numpy as np
import csv

#function for importing the data into the file
def import_data() :
    X = np.genfromtxt('train_X_lr.csv', delimiter = ',', dtype = np.float64, skip_header = 1)
    Y = np.genfromtxt('train_Y_lr.csv', delimiter = ',', dtype = np.float64)
    return X,Y


def compute_cost(X,Y, W) :
    Y_pred = np.dot(X, W)
    diffrence = Y - Y_pred
    mse = np.sum(np.square(diffrence))
    cost = mse/(2*len(X))
    return cost

#funtion for computing the gradient descent of cost function
def compute_gradient_descent_of_cost_function(X, Y, W) :
    Y_pred = np.dot(X, W)
    diffrence =  Y_pred-Y
    dW = np.dot(diffrence.T, X)/len(X)
    return dW.T

# optimizing the weights
def optimizing_the_weights_using_gradient_descent(X, Y, W, learning_rate) :
    previous_itr_cost = 0
    iter_no = 0
    while (True):
        iter_no += 1
        dW = compute_gradient_descent_of_cost_function(X, Y, W)
        W = W - (learning_rate * dW)
        cost = compute_cost(X, Y, W)
        if iter_no%100000 == 0 :
            print("number of itteration :", iter_no,"cost :", cost)
        if abs(previous_itr_cost - cost) < 0.00001  :
            print()
            print("---------------------------------")
            print("number of itteration :", iter_no,"cost :", cost)
            break
        previous_itr_cost = cost
    return W

def train_model(X, Y) :
    X = np.insert(X, 0, 1, axis = 1)
    Y = Y.reshape(len(X), 1)
    W = np.zeros((X.shape[1], 1))
    W = optimizing_the_weights_using_gradient_descent(X, Y, W, 0.0001)
    return W

def save_model(weights, weights_file_name) :
     with open(weights_file_name, 'w') as file :
         wr = csv.writer(file)
         wr.writerows(weights)
         file.close()

if __name__ == "__main__" :
    X,Y = import_data()
    weights = train_model(X, Y)
    save_model(weights, "WEIGHTS_FILE.csv")
