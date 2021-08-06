# ML-Algorithim-coded-from-scratch

In this repositry I had tried to code some machine learning algorthime from scratch.This repositry inlucdde following algorhims : - 

# Linear Regression 



In each algothim we have following files. 

**Training.py** : - This File can be use for trainning your model.

**Validate.py** :- This File is use for validating your model with the existing model of sklearn or well known libraries. 

**predict.py** : - this file is use for predicting the values of new data on the basis of the model you had trained. 

**Weights.csv** : - This file stores the weights of the model trained So that they can be use to predict the value of the new data. 


# K nearest Neighbours  

It contains all the files similar to the Linear regression. There is only one change that it does not have any **trainning.py** as KNN is a instance based learning algorithim So there is no require of traning the model So instead of **training.py** it has a  **helper.py** file which contains all the helping function which are needed for the prediction of the data in the KNN algorithim. It contains function for computing the distance between two vectors, finding k neares neighbours, finding the accuracy, finding the best value of the k, predicting the class etc. 

More improvement can be dine in this algorithim like here we are comparing the test vector with all the training examples but instead of that we could built somethin likhe Kd trees, local sensitive hashing or Inverted Index So that the number of the comparision could be reduced. 


#  Logistic Regression :- 

The structure of the code is similar to the linear regression the only change is that instead of using the **Mean Squared Error(MSE)** we have used the **Log Loss** . We done so beacuse MSE does not form a convex function for the logistic regression cost function So we use the Log Loss So that we could get the global minimum with this I have implemented the multiclass classification using logistic Regression for this I had used One vs all project technique.


# Preprocessing :- 
In this I have eimplementedd some preprocessing codes for rmemoving the null values in the Data from the mean of the column or by same standered data like 0, With this I had also implemented the scaling of the data by min-max scaler and encoded the non categorical feature to one hot encoding after that I implemented the Logistic regression classifer to predict the cataegory.

#  SVM :- 
It has two diffrent folder in which one is coded using the Sklearn library and the other folder have sklearn coded from scratch 

#  Descision Tree : 

Here trainning.py file is use for traning the descision tree model and after training of the model the weights of the model is stored using a pickel libray in python . 


# Regression : - 

This folder contain the knn regression applied to the data the idea of applying KNN regression is similar to KNN clasification the only change in the metrieces to choose the threshold and the attribute value and at the end the value is assogned as the average value of the n neighbours. Similar idea could be applied to the Decision tree also to get the Descision tree regression model .

