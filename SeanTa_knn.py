#!/usr/bin/env python
# coding: utf-8

# In[230]:


#-------------------------------------------------------------------------
# AUTHOR: Sean Ta
# FILENAME: SeanTa_knn.py
# SPECIFICATION: Assignment#2, Question 3e
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
             db.append (row)

# set count of number of predictions to 0
wrong_prediction = 0
right_prediction = 0

#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X and remove the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]]. Convert values to float to avoid warning messages

    #transform the original training classes to numbers and add them to the vector Y. Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...]. Convert values to float to avoid warning messages

    #--> add your Python code here
    # X =
    # Y =
    X = []
    Y = []
    
    minus_plus= {
    "-": 1,
    "+": 2} # transform class labels to 1,2

    for data in db:
        X.append([float(data[0]), float(data[1])])
        Y.append(float(minus_plus[data[2]]))
        
    testSample = X[i],Y[i] # test sample is X,Y of the specific instance we are removing/leaving out
    del X[i], Y[i] # removing this instance 


    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample[0]]) # predicting on the instance that we removed

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if (class_predicted == testSample[1]) == False: # if the prediction is False, then we add 1 to wrong_prediction count
        wrong_prediction+=1
    else: # if prediction is True, we add 1 to the right_prediction count
        right_prediction+=1
        

# #--> add your Python code here
error_rate = wrong_prediction / (wrong_prediction+right_prediction) # error rate = number of wrong predictions/ total predictions
print(error_rate)


# In[ ]:




