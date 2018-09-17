# cross validation without data balance (will add later)
# use data PostaggingFile.csv
# Author: Yuan (NB from Jingyi)
#update 1101 add calculate performance

import nltk
import codecs
from nltk.corpus import names
from nltk.metrics.scores import (f_measure, precision, recall)
import random
import csv
import pandas as pd
from pandas import DataFrame
from itertools import islice
from sklearn.model_selection import KFold
from sklearn import metrics

dir= "D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\" #replace by your path
fn = "PostaggingFile.csv"

df = codecs.open(dir+fn, "r", "UTF-8")

dataset = []
outputsets = []

# generate feature sets from a instance
def gen_features(line):
    sl = line.split(',')
    features = {'postag_1': sl[1], 'postag_2':sl[2], 'entity1':sl[3], 'entity2': sl[4], 'reverse': sl[5]}
    return features

def gen_labeledset(dataset):
    featuresets = []
    for record in dataset:
        label = record.split(',')[0]
        featuresets = featuresets + [(gen_features(record), label)] # join feature sets with label
    return featuresets

for line in islice(df, 1, None):
    dataset = dataset + [line]
random.shuffle(dataset)     # randomly order the records

#dataset for performance calculation
predictionSet=[]
relationSet=[]

kf = KFold(n_splits=10) #set k=10
#k-fold validation, for each iteration,generate train/test set and implement NB
for train, test in kf.split(dataset):
    train_set=[]
    test_set=[]
    for i in train:
        train_set.append(dataset[i])
    for j in test:
        test_set.append(dataset[j])
        #get the 'real' label for each test data
        real_relation = dataset[j].split(',')[0]
        relationSet.append(real_relation)
    # use NaiveBayes algorithm to train the training set
    classifier = nltk.NaiveBayesClassifier.train(gen_labeledset(train_set))
    for case in test_set:
        prediction = classifier.classify(gen_features(case))
        predictionSet.append(prediction)
        outputsets = outputsets + [case[:-2] + ',' + prediction]

## export the prediction into csv file, change to your path
import os
#delete if the file already exists (make sure only data for this run is stored)
if os.path.exists("D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\output_kfold.csv"):
    os.remove("D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\output_kfold.csv")
fw = codecs.open("D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\output_kfold.csv","a+","UTF-8" )
fieldnames = ['relation', 'postag1', 'postag2', 'entity1', 'entity2', 'reverse', 'prediction']
writer = csv.writer(fw)
writer.writerow(fieldnames)
for row in outputsets:
    writer.writerow(row.split(','))
fw.close()

#print acc, recall, precision, F1
print('Accuracy = ', nltk.classify.accuracy(classifier, gen_labeledset(test_set)))
print('Recall=',metrics.recall_score(relationSet,predictionSet, average='weighted'))
print('Precision=',metrics.precision_score(relationSet,predictionSet, average='weighted'))
print('F1=',metrics.f1_score(relationSet,predictionSet, average='weighted'))

