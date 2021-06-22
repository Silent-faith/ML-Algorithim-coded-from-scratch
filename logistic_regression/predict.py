import numpy as np
import csv
import sys

from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_lg.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_weights(test_X_file_path, weights1_file_path, weights2_file_path, weights3_file_path, weights4_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights1 = np.genfromtxt(weights1_file_path, delimiter=',', dtype=np.float64)
    weights2 = np.genfromtxt(weights2_file_path, delimiter=',', dtype=np.float64)
    weights3 = np.genfromtxt(weights3_file_path, delimiter=',', dtype=np.float64)
    weights4 = np.genfromtxt(weights4_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights1, weights2, weights3, weights4

def sigmoid(Z):
    sigmoid_of_Z = 1 / (1 + np.exp(-Z))

    return sigmoid_of_Z


def predict_target_values(X, weights):
    W = weights[:len(weights)-1]
    b = weights[len(weights)-1]

    W = W.reshape((len(W), 1))
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    A = A.flatten()
    return A


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights1, weights2, weights3, weights4 = import_data_and_weights(test_X_file_path, "Weights_for_class_1.csv", "Weights_for_class_2.csv", "Weights_for_class_3.csv", "Weights_for_class_4.csv")

    pred_Y = []

    pred_Y_1 = predict_target_values(test_X, weights1)
    pred_Y_2 = predict_target_values(test_X, weights2)
    pred_Y_3 = predict_target_values(test_X, weights3)
    pred_Y_4 = predict_target_values(test_X, weights4)
    #print(pred_Y_1, pred_Y_2, pred_Y_3, pred_Y_4, sep = '\n\n')
    for i in range(len(pred_Y_1)):
        maximum = max(pred_Y_1[i], pred_Y_2[i], pred_Y_3[i], pred_Y_4[i])
        #print(maximum)
        if maximum == pred_Y_1[i]:
            pred_Y.append(0)
        elif maximum == pred_Y_2[i]:
            pred_Y.append(1)
        elif maximum == pred_Y_3[i]:
            pred_Y.append(2)
        elif maximum == pred_Y_4[i]:
            pred_Y.append(3)
        
    #print(pred_Y)
    pred_Y = np.array(pred_Y)
    write_to_csv_file(pred_Y, "predicted_test_Y_lg.csv")



if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_lg_v2.csv") 


# python predict.py train_X_lg_v2.csv