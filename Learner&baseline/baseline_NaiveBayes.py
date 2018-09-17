# simple Naive Bayes as baseline
# use postagging as features
# no cross-validation
# Author: Jingyi

import nltk
import codecs
from nltk.corpus import names
import random
import csv
import pandas as pd
from itertools import islice

# generate feature sets from a instance
def gen_features(line):
    sl = line.split(',')
    features = {'postag_1': sl[1], 'postag_2':sl[2], 'entity1':sl[3], 'entity2': sl[4], 'reverse': sl[5]}
    return features

dir = "F:\\Data Science\\Team Project\\dataset\\"
fn = "PostaggingFile.csv"

df = codecs.open(dir+fn, "r", "UTF-8")

dataset = []

outputsets = []
for line in islice(df, 1, None):
    dataset = dataset + [line]
random.shuffle(dataset)     # randomly order the records
train_set, test_set = dataset[:1000], dataset[1001:]  # split data into training set and test set, the size is defined in []

def gen_labeledset(dataset):
    featuresets = []
    for record in dataset:
        label = record.split(',')[0]
        featuresets = featuresets + [(gen_features(record), label)] # join feature sets with label
    return featuresets

# use NaiveBayes algorithm to train the training set
classifier = nltk.NaiveBayesClassifier.train(gen_labeledset(train_set))

for case in test_set:
    prediction = classifier.classify(gen_features(case))
    outputsets = outputsets + [case[:-2] + ',' + prediction]

## export the prediction into csv file
fw = codecs.open("F:\\Data Science\\Team Project\\dataset\\output.csv","a+","UTF-8" )
fieldnames = ['relation', 'postag1', 'postag2', 'entity1', 'entity2', 'reverse', 'prediction']
writer = csv.writer(fw)
writer.writerow(fieldnames)
for row in outputsets:
    writer.writerow(row.split(','))
fw.close()



#testline1 ="USAGE,NNS,NNS,information retrieval techniques,keywords,TRUE"
#testline2 = "USAGE,NN,NNS,oral communication,indices,FALSE"
#print(testline1)
#print(classifier.classify(gen_features(testline1)))
#print(testline2)
#print(classifier.classify(gen_features(testline2)))
print(nltk.classify.accuracy(classifier, gen_labeledset(test_set)))
#classifier.show_most_informative_features(5)