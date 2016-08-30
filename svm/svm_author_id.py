#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy as np
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#clf = SVC(kernel="linear")
clf = SVC(C=10000,kernel="rbf")

#########################################################
### Minimize the training set ###
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "training time of ", len(labels_train), "training emails:", round(time()-t0, 3), "s"

t1 = time()
labels_predict = clf.predict(features_test)
print "predicting time of", len(labels_test), "emails:", round(time()-t1, 3), "s"

accuracy = accuracy_score(labels_predict,labels_test)

### print out accuracy
print "Accuracy of Naive Bayes algrorithm for email author classification:", accuracy
#########################################################

### print out the 10th, 26th and 50th prediction results
answer = [labels_predict[10], labels_predict[26], labels_predict[50]]
print "the 10th, 26th and 50th prediction results are", answer
#########################################################

### print out how many Chris (1) has been predicted ###
number = len(labels_predict[labels_predict[:]==1])
print "number of Chris:", number
