import numpy as np
import csv
import sys
import pickle
from validate import validate
from train import Node

def predict_class(root, X):
    if root.left is None and root.right is None:
        return root.predicted_class
    
    if X[root.feature_index] >= root.threshold:
        return predict_class(root.right, X)
    
    else:
        return predict_class(root.left, X)


def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    model = pickle.load(open(model_file_path, 'rb'))
    return test_X, model


def predict_target_values(test_X, model):
    pred_Y = []
    for x in test_X:
        pred_Y.append(predict_class(model, x))
    
    return pred_Y


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, model = import_data_and_model(test_X_file_path, 'MODEL_FILE.sav')
    pred_Y = predict_target_values(test_X, model)
    pred_Y = np.array(pred_Y)
    write_to_csv_file(pred_Y, "predicted_test_Y_de.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_de.csv") 