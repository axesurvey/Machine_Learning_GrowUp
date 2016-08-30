#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

clf = GaussianNB()

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


