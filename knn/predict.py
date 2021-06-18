import numpy as np
import csv
import sys
from helpers import get_best_k_using_validation_set, calculate_accuracy, classify_points_using_knn, find_k_nearest_neighbors, compute_ln_norm_distance
from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data(test_X_file_path, train_X_file_path, train_Y_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    train_X = np.genfromtxt(train_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    train_Y = np.genfromtxt(train_Y_file_path, delimiter=',', dtype=np.int32)
    return test_X, train_X, train_Y


def predict_target_values(train_X, train_Y, test_X):
    validation_split_percent = 30
    n = 2
    k = get_best_k_using_validation_set(train_X, train_Y, validation_split_percent, n)
    predicted_classes = classify_points_using_knn(train_X, train_Y, test_X, k, n)
    return predicted_classes


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = np.array(pred_Y)
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, train_X, train_Y = import_data(test_X_file_path, "train_X_knn.csv", "train_Y_knn.csv")
    pred_Y = predict_target_values(train_X, train_Y, test_X)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv")
