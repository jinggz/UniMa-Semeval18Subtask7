
# coding: utf-8

# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
#from __future__ import division
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA


# In[14]:


Doc=pd.read_csv("D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\Instances2.0.csv",encoding="gbk")


sentence=Doc.Instances
Sentences=[]
for sentences in sentence:
    sentences=sentences.replace("[","")
    sentences=sentences.replace("]","")
    sentences=sentences.replace("'","")
    sentences=sentences.replace(" ","")
    sentences=sentences.replace(","," ")
    Sentences.append(sentences)

tokenize = lambda doc: doc.lower().split(" ")
stopWords = stopwords.words('english')
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)

sklearn_representation = sklearn_tfidf.fit_transform(Sentences)

vectorizer = CountVectorizer(stop_words = stopWords)
#print vectorizer
transformer = TfidfTransformer()

trainVectorizerArray = vectorizer.fit_transform(Sentences).toarray()


# cross validation inside of grid search, without data balance， using SVM
#her we use the whole data for CV and train on the whole data, but no final performance since test data is not given
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
#X = pd.read_csv('D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\ModelInput_woStopWords.csv', sep=',',
                #usecols=['positionInSentence_x', 'wordsInSentence_x', 'wordLen_x','positionInSentence_y', 'wordsInSentence_y', 'wordLen_y'],encoding='gbk')
y = pd.read_csv('D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\ModelInput_woStopWords.csv', sep=',',
                usecols=['Label'],encoding='gbk')
#filling all missing data with 0
#X=X.fillna(0)

X_train = trainVectorizerArray

# reduce dimension of target set
y_train = np.array(y).flatten()

# Set the parameters by cross-validation
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,0.1],
                     'C': [0.1,10]}
                    ]

#scores = ['f1','precision', 'recall'] #also possible to filter model using precision, recall
scores = ['f1']
# tuning hyper-parameters with precision and recall
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score)
    clf.fit(X_train, y_train)
    #save svm model for all parameters, load back the model using #clf = joblib.load('allSVMmodel.pkl')
    #joblib.dump(clf, 'allSVMmodelWOsplit.pkl')

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    # train the best model on the whole trainset
    #if the best kernel is linear, there is no gamma parameter
    if clf.best_params_['kernel']=='rbf':
        clf_best = svm.SVC(C=clf.best_params_['C'], kernel=clf.best_params_['kernel'],gamma=clf.best_params_['gamma'])
    else:
        clf_best = svm.SVC(C=clf.best_params_['C'], kernel=clf.best_params_['kernel'])
    clf_best.fit(X_train, y_train)
    # save the best svm model, load back the model using #clf_best = joblib.load('bestSVMmodel.pkl')
    #joblib.dump(clf_best, 'bestSVMmodelWOsplit.pkl')

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


