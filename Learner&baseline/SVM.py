# cross validation inside of grid search, without data balance， using SVM
#find the best parameter, train again on the whole train set and test on evaluation set
#read file 'feature', this file will keep updating
# Author: Yuan

import nltk
import codecs
import random
from nltk.corpus import names
from nltk.metrics.scores import (f_measure, precision, recall)
import csv
import pandas as pd
import numpy as np
from itertools import islice
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib


# import balancing_data    # default: the file is under the same folder
df = pd.read_csv('D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\ModelInput.csv', sep=',',encoding='gbk')

dataset = []
outputsets = []

dataset = np.array(df, dtype=str)
random.shuffle(dataset)  # randomly order the records

# turn the data in a (samples, feature) matrix:
n_samples = len(dataset)
# seperate data into feature set and targetset
X = pd.read_csv('D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\ModelInput.csv', sep=',',
                usecols=['positionInSentence_x', 'wordsInSentence_x', 'wordLen_x','positionInSentence_y', 'wordsInSentence_y', 'wordLen_y'],encoding='gbk')
y = pd.read_csv('D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\ModelInput.csv', sep=',',
                usecols=['Label'],encoding='gbk')

#filling all missing data with 0
X=X.fillna(0)
# Split the dataset in train and evaluation set (only use cross validation in train set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
# transfer from type dataframe to array
X_train = np.array(X_train)
X_test = np.array(X_test)
# reduce dimension of target set
y_train = np.array(y_train).flatten()
y_test = np.array(y_test).flatten()


# Set the parameters by cross-validation
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,0.05,1e-1,0.15,0.2],
                     'C': [130,133,136,140]},
                    {'kernel': ['linear'], 'C': [10,50,100]}]


#scores = ['f1','precision', 'recall'] #also possible to filter model using precision, recall
scores = ['f1']
# tuning hyper-parameters with precision and recall
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=10,
                       scoring='%s_weighted' % score)
    clf.fit(X_train, y_train)
    #save svm model for all parameters, load back the model using #clf = joblib.load('allSVMmodel.pkl')
    joblib.dump(clf, 'allSVMmodel.pkl')

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))

    # train the best model on the whole trainset, test on evaluation set
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    #if the best kernel is linear, there is no gamma parameter
    if clf.best_params_['kernel']=='rbf':
        clf_best = svm.SVC(C=clf.best_params_['C'], kernel=clf.best_params_['kernel'],gamma=clf.best_params_['gamma'])
    else:
        clf_best = svm.SVC(C=clf.best_params_['C'], kernel=clf.best_params_['kernel'])
    clf_best.fit(X_train, y_train)
    # save the best svm model, load back the model using #clf_best = joblib.load('bestSVMmodel.pkl')
    joblib.dump(clf_best, 'bestSVMmodel.pkl')
    y_true, y_pred = y_test, clf_best.predict(X_test)
    print(classification_report(y_true, y_pred))

    # write the output file
    import os
    # delete if the file already exists (make sure only data for this run is stored)
    # if os.path.exists("D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\output_%s.csv" % score):
    # os.remove("D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\output_%s.csv" % score)

    ## export the prediction into csv file, change to your path
    # save = pd.DataFrame(outputsets)
    # fieldnames = ['relation', 'feature1', 'feature2', 'feature3', 'feature4','prediction']
    # save.to_csv('D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\output_%s.csv" % score, header=fieldnames,index=False, index_label=fieldnames, sep=',', encoding='utf-8')

