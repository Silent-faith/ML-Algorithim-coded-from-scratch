import numpy as np
import csv
import sys
import pickle

from validate import validate


def import_data(test_X_file_path, model_file_path, train_X_file_path, train_Y_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    hyperparameters = np.genfromtxt(model_file_path, delimiter=',', dtype=np.float64)
    train_X = np.genfromtxt(train_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    train_Y = np.genfromtxt(train_Y_file_path, delimiter=',', dtype=np.float64)
    
    return test_X, hyperparameters, train_X, train_Y


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def compute_ln_norm_distance(vector1, vector2, n):
    vector_len = len(vector1)
    #print(np.shape(vector1))
    #print(np.shape(vector2))
    distance = 0
    for i in range(vector_len):
        distance += (abs(vector1[i] - vector2[i])) ** n
    
    return distance ** (1/n)


def find_k_nearest_neighbours(train_X, test_example, k, n):
    dist_indices_pairs = []
    for i in range(len(train_X)):
        distance = compute_ln_norm_distance(train_X[i], test_example, n)
        dist_indices_pairs.append([distance, i])
    
    dist_indices_pairs.sort(key = lambda x : (x[0],x[1]))
    nearest_indices = [i[1] for i in dist_indices_pairs]
    k_nearest_indices = nearest_indices[:k]

    nearest_distances = [i[0] for i in dist_indices_pairs]
    k_nearest_distances = nearest_distances[:k]

    return k_nearest_indices, k_nearest_distances


def compute_similarity(knn_distances, tau):
    knn_distances = np.array(knn_distances)
    knn_distances = knn_distances.reshape(len(knn_distances), 1)
    similarity = np.exp(-((knn_distances) ** 2)/(2 * (tau**2)))
    return similarity


def compute_gradient_of_cost_function(X, Y, W, similarity):
    k = len(X)
    diff = np.dot(X, W) - Y
    gradient = (1/k) * np.dot(similarity.T, diff)
    return gradient


def compute_cost(X, Y, W, similarity):
    Y_prediction = np.dot(X, W)
    mse = np.sum(np.dot(similarity.T, np.square(Y_prediction - Y)))
    cost = mse/(2*len(X))
    return cost





def optimize_weights_using_gradient_descent(X, Y, similarity, test_x, W, learning_rate, cost_diff):
    previous_iteration_cost = 0
    iter_no = 0

    while True:
        iter_no += 1
        dW = compute_gradient_of_cost_function(X, Y, W, similarity)

        W = W - learning_rate*dW

        cost = compute_cost(X, Y, W, similarity)

        #if iter_no % 1000 == 0:
         #   print(iter_no, cost)
      
        if iter_no > 1 and (previous_iteration_cost - cost) < 0:
            print("cost is increasing!!!")
            break
  
        if abs(previous_iteration_cost - cost) <= cost_diff:
            #print(iter_no, previous_iteration_cost, cost)
            break
        
        previous_iteration_cost = cost

    return W


def predict_target_values(train_X, train_Y, test_X, hyperparameters):
    k, tau, learning_rate = int(hyperparameters[0]), hyperparameters[1], float(hyperparameters[2])
    cost_diff = 0.0001
    pred_Y = []
    for x in test_X:
        W = np.zeros((train_X.shape[1], 1))

        knn_list, knn_distances = find_k_nearest_neighbours(train_X, x, k, 2)
        kn_training_examples_X = train_X[knn_list]
        kn_training_examples_Y = train_Y[knn_list]

        similarity = compute_similarity(knn_distances, tau)

        W = optimize_weights_using_gradient_descent(kn_training_examples_X, kn_training_examples_Y, similarity, x, W, learning_rate, cost_diff)
    
        prediction = np.dot(x.T, W)
        pred_Y.append(prediction)
        #print(prediction)
    
    return pred_Y


def predict(test_X_file_path):

    # Load Data
    test_X, hyperparameters, train_X, train_Y = import_data(test_X_file_path, 'MODEL_FILE.csv', 'train_X_re.csv', 'train_Y_re.csv')
    #print(np.shape(train_Y))
    #print(np.shape(train_X))
    
    train_X = np.insert(train_X, 0, 1, axis = 1)
    train_Y = train_Y.reshape(len(train_X), 1)
    test_X = np.insert(test_X, 0, 1, axis = 1)

    #print(np.shape(train_Y))
    #print(np.shape(train_X))
    # Predict Target Variables
    pred_Y = predict_target_values(train_X, train_Y, test_X, hyperparameters)
    
    #print(pred_Y[1])
    pred_Y = np.array(pred_Y)
    #print(np.shape(pred_Y))

    
    write_to_csv_file(pred_Y, "predicted_test_Y_re.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_re.csv") 