#This file is for subtask 1.1
# cross validation inside of grid search, without data balance， using SVM
# her we use the whole data for CV and train on the whole data, but no final performance since test data is not given
# find the best parameter, train again on the whole train set and test on evaluation set
# read file 'ModelInput', this file will keep updating
# Author: Yuan

import nltk
import codecs
import random
from nltk.corpus import names
from nltk.metrics.scores import (f_measure, precision, recall)
import csv
from scipy import sparse
import pandas as pd
import numpy as np
from itertools import islice
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
path = 'D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\Multi_class\\Task1.1\\'
# seperate data into feature set and targetset
#X is feature set, y is label set, dataset is the merge of the both 2
X = pd.read_csv(path+'ModelInput.csv', sep=',',
                      usecols=[12,*range(15,21), *range(33,39)], encoding = 'gbk')
y = pd.read_csv(path+'ModelInput.csv', sep=',',
                usecols=['label_index'], encoding='gbk')
dataset = pd.read_csv(path+'ModelInput.csv', sep=',',
                      usecols=[12,*range(15,21), *range(33,39)], encoding = 'gbk')

# use the whole tf-idf terms
df1 = sparse.load_npz(path+"tfidf_sparse.npz")
tfidfs=pd.DataFrame(df1.toarray())
#merge the dataset with tfidf
dataset = pd.concat([dataset, tfidfs], axis=1)
X = pd.concat([X, tfidfs], axis=1)

#delete the data which label_index=-1 (no relation)
dataset = dataset[dataset.label_index!=-1]
X = X[X.label_index!=-1]
y = y[y.label_index !=-1]
X.drop(['label_index'],axis=1, inplace=True)

#print(dataset.columns.values)

# filling all missing data with 0
X = X.fillna(0)
dataset = dataset.fillna(0)

# transfer from type dataframe to array
X_train = np.array(X)
dataset = np.array(dataset)

# reduce dimension of target set
y_train = np.array(y).flatten()

# Set the parameters by cross-validation
tuned_parameters = [{ 'C': [0.001,0.01,0.1,1,10]}]

# scores = ['f1','precision', 'recall'] #also possible to filter model using precision, recall
scores = ['f1']
# tuning hyper-parameters with precision and recall
#from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
#myown_f1 = make_scorer(f1_score,average='binary',pos_label = 1)
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    #lin_clf = svm.LinearSVC()
    #lin_clf.fit(X, Y)
    #LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              #intercept_scaling=1, loss='squared_hinge', max_iter=1000,
              #multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              #verbose=0)
    #dec = lin_clf.decision_function([[1]])
    #dec.shape[1]

    LinearSVC(class_weight='balanced')
    clf = GridSearchCV(svm.LinearSVC(),tuned_parameters, cv=10,
                       scoring='f1_macro')
    clf.fit(X_train, y_train)
    # save svm model for all parameters, load back the model using #clf = joblib.load('allSVMmodel.pkl')
    #joblib.dump(clf, 'allSVMmodel.pkl')

    print("Best parameters set found:")
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"% (mean_score, scores.std() * 2, params))
    print()

    # if the best kernel is linear, there is no gamma parameter
    #if clf.best_params_['kernel'] == 'rbf':
    #   clf_best = svm.SVC(C=clf.best_params_['C'], kernel=clf.best_params_['kernel'], gamma=clf.best_params_['gamma'])
    #else:
    #    clf_best = svm.SVC(C=clf.best_params_['C'], kernel=clf.best_params_['kernel'])
    clf_best = svm.SVC(C=clf.best_params_['C'])

    # train the best model on the whole trainset
    clf_best.fit(X_train, y_train)
    # save the best svm model, load back the model using #clf_best = joblib.load('bestSVMmodel.pkl')
    import time
    a=time.time()
    joblib.dump(clf_best, 'bestSVMmodelSVM0108.pkl')


# get the classification report for the best parameter
# k-fold classification on the whole dataset
kf = KFold(n_splits=10)  # set k=10

y_true = []
y_pred = []

for train, test in kf.split(dataset):
    x_f = []
    x_l = []
    test_set = []
    pred = []
    for i in train:
        x_f.append(dataset[i][1:])
        x_l.append(dataset[i][0])
    x_f = np.array(pd.DataFrame(x_f))
    clf_best.fit(x_f, x_l)
    # collect all the real label and predicted label
    for j in test:
        y_true.append(dataset[j][0])
        test_set.append(dataset[j][1:])
    test_set = np.array(pd.DataFrame(test_set))
    # print classification report
    pred = clf_best.predict(test_set)
    y_pred.extend(pred)
print(classification_report(y_true, y_pred))
print('finish')

# write the output file
import os
# delete if the file already exists (make sure only data for this run is stored)
# if os.path.exists("D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\output_%s.csv" % score):
# os.remove("D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\output_%s.csv" % score)

## export the prediction into csv file, change to your path
# save = pd.DataFrame(outputsets)
# fieldnames = ['relation', 'feature1', 'feature2', 'feature3', 'feature4','prediction']
# save.to_csv('D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\output_%s.csv" % score, header=fieldnames,index=False, index_label=fieldnames, sep=',', encoding='utf-8')


