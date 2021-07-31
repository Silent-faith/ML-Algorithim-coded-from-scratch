import numpy as np
import csv
import sys
import json
import pickle
from validate import validate
from math import log

def import_data_and_model(test_X_file_path, model_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter='\n', dtype=str)
    model = pickle.load(open(model_file_path, 'rb'))
    return test_X, model


def compute_likelihood(x, c, class_wise_frequency_dict, class_wise_denominators, laplace_smoothing):
    denominator = class_wise_denominators[c]
    log_likelihood = 0
    for word in x.split():
        if word in class_wise_frequency_dict[c]:
            p = (class_wise_frequency_dict[c][word] + laplace_smoothing)/denominator
        else:
            p = laplace_smoothing/denominator
        log_likelihood += log(p)
    return log_likelihood


def predict_class(x, classes, class_wise_frequency_dict, class_wise_denominators, prior_probabilities, laplace_smoothing):
    P = []
    C = []
    for c in classes:
        p = log(prior_probabilities[c]) + compute_likelihood(x, c, class_wise_frequency_dict, class_wise_denominators, laplace_smoothing)
        P.append(p)
        C.append(c)
    index = np.argmax(P)
    return C[index]


def predict_target_values(test_X, model):
    pred_Y = []
    for x in test_X:
        pred_Y.append(predict_class(x, model['trained_classes'], model['trained_class_wise_frequency_dict'], model['trained_class_wise_denominator'], model['trained_prior_probabilities'], model['laplace_parameter']))
    
    return pred_Y


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = np.array(pred_Y)
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def remove_special_chars_other_than_space(X):
    X_without_spl_chars = []
    for x in X:
        i = 0
        x_with_no_spl_chars = ""
        for i in range(len(x)): 
            if (ord(x[i]) >= ord('A') and
                ord(x[i]) <= ord('Z') or 
                ord(x[i]) >= ord('a') and 
                ord(x[i]) <= ord('z') or
                ord(x[i]) == ord(' ')):
                x_with_no_spl_chars += x[i] 
        X_without_spl_chars.append(x_with_no_spl_chars)

    return X_without_spl_chars
    

def preprocess(X):
    X = remove_special_chars_other_than_space(X)
    
    for i in range(len(X)):
        X[i] =' '.join(X[i].split())
        X[i] = X[i].lower()
    
    return X


def predict(test_X_file_path):
    # Load Model Parameters
    """
    Parameters of Naive Bayes include Laplace smoothing parameter, Prior probabilites of each class and values related to likelihood computation.
    """
    test_X, model = import_data_and_model(test_X_file_path, "MODEL_FILE.sav")
    test_X = preprocess(test_X)
    pred_Y = predict_target_values(test_X, model)
    write_to_csv_file(pred_Y, "predicted_test_Y_nb.csv")    


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_nb.csv") 