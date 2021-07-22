# ML-Algorithim-coded-from-scratch

In this repositry I had tried to code some machine learning algorthime from scratch.This repositry inlucdde following algorhims : - 

# 1. Linear Regression 



In each algothim we have following files. 

**Training.py** : - This File can be use for trainning your model.

**Validate.py** :- This File is use for validating your model with the existing model of sklearn or well known libraries. 

**predict.py** : - this file is use for predicting the values of new data on the basis of the model you had trained. 

**Weights.csv** : - This file stores the weights of the model trained So that they can be use to predict the value of the new data. 


# 2. K nearest Neighbours  

It contains all the files similar to the Linear regression. There is only one change that it does not have any **trainning.py** as KNN is a instance based learning algorithim So there is no require of traning the model So instead of **training.py** it has a  **helper.py** file which contains all the helping function which are needed for the prediction of the data in the KNN algorithim. It contains function for computing the distance between two vectors, finding k neares neighbours, finding the accuracy, finding the best value of the k, predicting the class etc. 

More improvement can be dine in this algorithim like here we are comparing the test vector with all the training examples but instead of that we could built somethin likhe Kd trees, local sensitive hashing or Inverted Index So that the number of the comparision could be reduced. 


# 3. Logistic Regression :- 

The structure of the code is similar to the linear regression the only change is that instead of using the **Mean Squared Error(MSE)** we have used the **Log Loss** . We done so beacuse MSE does not form a convex function for the logistic regression cost function So we use the Log Loss So that we could get the global minimum with this I have implemented the multiclass classification using logistic Regression for this I had used One vs all project technique.


# 4. Preprocessing :- 
In this I have eimplementedd some preprocessing codes for rmemoving the null values in the Data from the mean of the column or by same standered data like 0, With this I had also implemented the scaling of the data by min-max scaler and encoded the non categorical feature to one hot encoding after that I implemented the Logistic regression classifer to predict the cataegory.

# 5. SVM :- 
For SVM sklearn library is used 

# 6 Descision Tree : 

Here trainning.py file is use for traning the descision tree model and after training of the model the weights of the model is stored using a pickel libray in python . 


