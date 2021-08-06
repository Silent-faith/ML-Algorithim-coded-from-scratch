import numpy as np
import math
import csv

def import_training_data(train_X_file_path, train_Y_file_path):
    train_X = np.genfromtxt(train_X_file_path, delimiter=',', dtype = np.float64, skip_header=1)
    train_Y = np.genfromtxt(train_Y_file_path, delimiter=',', dtype = np.float64)
    return train_X, train_Y


def calculate_mse(predicted_Y, actual_Y):
    total_cases = len(actual_Y)
    sqrd_diff = 0
    
    for i in range(total_cases):
        sqrd_diff += (actual_Y[i] - predicted_Y[i])**2
    
    mse = sqrd_diff/total_cases
    return mse


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


def train_and_validate(X, Y, test_start, test_end, k, tau, learning_rate):
    testing_X_data = X[test_start : test_end, : ]
    testing_Y_data = Y[test_start : test_end]

    train_X = np.copy(X)
    train_Y = np.copy(Y)

    training_X_data = np.delete(train_X, slice(test_start, test_end), axis=0)
    training_Y_data = np.delete(train_Y, slice(test_start, test_end), axis=0)
    #print(len(training_X_data))
    #print(len(testing_X_data))
    
    cost_diff = 0.0001
    
    pred_Y = []

    testing_index = 0

    for x in testing_X_data:
        #print('testing_index', testing_index)
        testing_index += 1

        W = np.zeros((training_X_data.shape[1], 1))
    
        knn_list, knn_distances = find_k_nearest_neighbours(training_X_data, x, k, 2)
        kn_training_examples_X = training_X_data[knn_list]
        kn_training_examples_Y = training_Y_data[knn_list]
        
        similarity = compute_similarity(knn_distances, tau)
        #print(similarity)

        W = optimize_weights_using_gradient_descent(kn_training_examples_X, kn_training_examples_Y, similarity, x, W, learning_rate, cost_diff)
        
        prediction = np.dot(x.T, W)
        pred_Y.append(prediction)

    mse = calculate_mse(pred_Y, testing_Y_data)
    return mse


def train_model(X, Y):
    tau = 3
    learning_rate = 0.00001

    X = np.insert(X, 0, 1, axis = 1)
    Y = Y.reshape(len(X), 1)

    testing_ratio = 0.2
    testing = math.floor(len(X) * testing_ratio)

    mse_k_pairs = []

    max_value_of_k = len(X) - testing

    for k in range(5, 15):
        mse = 0

        mse += train_and_validate(X, Y, 0, testing, k, tau, learning_rate)
        mse += train_and_validate(X, Y, testing, 2*testing, k, tau, learning_rate)
        mse += train_and_validate(X, Y, 2*testing, 3*testing, k, tau, learning_rate)
        mse += train_and_validate(X, Y, 3*testing, 4*testing, k, tau, learning_rate)
        mse += train_and_validate(X, Y, 4*testing, 5*testing, k, tau, learning_rate)
        avg_mse = mse/5
        print('k', k, 'avg_mse', avg_mse)
        mse_k_pairs.append([avg_mse, k])
    

    min_mse = min(mse_k_pairs, key = lambda x : x[0])
    index = mse_k_pairs.index(min_mse)
    best_k = mse_k_pairs[index][1]
    print('min_mse', mse_k_pairs[index][0])
    print('best_k', best_k)

    hyperparameters = [best_k, tau, learning_rate]
    return hyperparameters


def save_model(hyperparameters, file_name):
    with open(file_name, 'w') as hyperparameter_file:
        wr = csv.writer(hyperparameter_file)
        wr.writerows(hyperparameters)
        hyperparameter_file.close()


if __name__ == '__main__':
    train_X, train_Y = import_training_data('train_X_re.csv', 'train_Y_re.csv')
    hyperparameters = train_model(train_X, train_Y)
    hyperparameters = np.reshape(hyperparameters, (len(hyperparameters), 1))
    
    save_model(hyperparameters, 'MODEL_FILE.csv')
