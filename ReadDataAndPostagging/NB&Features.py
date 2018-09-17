#NaiveBayes and Most informative features
# Author: Sara
import nltk
import codecs
from nltk.corpus import names
import random
import csv
import pandas as pd
from itertools import islice

def gen_features(line):
    sl = line.split(',')
    features = {'postag_1': sl[1], 'postag_2':sl[2], 'entity1':sl[3], 'entity2': sl[4], 'reverse': sl[5]}
    return features
file_name = "PostaggingFile.csv"
df = codecs.open(file_name, "r", "UTF-8")
featuresets=[]
for line in islice(df, 1, None):

    label = line.split(',')[0]
    featuresets = featuresets + [(gen_features(line), label)]
train_set =featuresets[:800]
test_set = featuresets[801:]
classifier=nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier,test_set)*100)
classifier.show_most_informative_features(10)
