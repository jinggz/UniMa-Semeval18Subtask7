# cross validation with data balance option
# use data PostaggingFile.csv
# Author: Yuan, Jingyi

import nltk
import codecs
from nltk.metrics.scores import (f_measure, precision, recall)
import random
import csv
import pandas as pd
import numpy as np
from itertools import islice
from sklearn.model_selection import KFold
from sklearn import metrics
import balancing_data    # default: the file is under the same folder

df = pd.read_csv('F:\\Data Science\\Team Project\\svn\\Teamprojekt2017\\Project-code\\ReadDataAndPostagging\\PostaggingFile.csv', sep=',')

dataset = []
outputsets = []

# generate feature sets from a instance
def gen_features(line):
    features = {'postag_1': line[1], 'postag_2':line[2], 'entity1':line[3], 'entity2': line[4], 'reverse': line[5]}
    return features

def gen_labeledset(dataset):
    featuresets = []
    for record in dataset:
        label = record[0]
        featuresets = featuresets + [(gen_features(record), label)] # join feature sets with label
    return featuresets

#dataset for performance calculation
predictionSet=[]
relationSet=[]

#random.shuffle(dataset)     # randomly order the records
dataset = np.array(df, dtype=str)
kf = KFold(n_splits=10) #set k=10
#k-fold validation, for each iteration,generate train/test set and implement NB
for train, test in kf.split(dataset):
    train_set=[]
    test_set=[]
    for i in train:
        train_set.append(dataset[i])    # datatype: 'list'
    train_balanced = balancing_data.balancingData(train_set) # dtype: 'list'
    for j in test:
        test_set.append(dataset[j])
        # get the 'real' label for each test data
        real_relation = dataset[j][0]
        relationSet.append(real_relation)

      # use NaiveBayes algorithm to train the training set

#   classifier = nltk.NaiveBayesClassifier.train(gen_labeledset(train_set)) # not use balancing
    classifier = nltk.NaiveBayesClassifier.train(gen_labeledset(train_balanced)) # use balancing

    for case in test_set:
        prediction = classifier.classify(gen_features(case))
        row = np.insert(case, 6, prediction)
        outputsets.append(row.tolist())
        predictionSet.append(prediction)

import os

# delete if the file already exists (make sure only data for this run is stored)
if os.path.exists("F:\\Data Science\\Team Project\\dataset\\output_NBCV.csv"):
    os.remove("F:\\Data Science\\Team Project\\dataset\\output_NBCV.csv")

## export the prediction into csv file, change to your path
save = pd.DataFrame(outputsets)
fieldnames = ['relation', 'postag1', 'postag2', 'entity1', 'entity2', 'reverse', 'prediction']
save.to_csv('F:\\Data Science\\Team Project\\dataset\\output_NBCV.csv', header=fieldnames,index=False, index_label=fieldnames, sep=',', encoding='utf-8')


# evalutaion part
print('Accuracy = ', nltk.classify.accuracy(classifier, gen_labeledset(test_set)))
print('Recall=',metrics.recall_score(relationSet,predictionSet, average='weighted'))
print('Precision=',metrics.precision_score(relationSet,predictionSet, average='weighted'))
print('F1=',metrics.f1_score(relationSet,predictionSet, average='weighted'))


