# simple Naive Bayes as baseline
# use postagging as features
# cross-validation
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
xml = codecs.open("PostaggingFile.csv", "r", "UTF-8")
featuresets=[]
for line in islice(xml, 1, None):

    label = line.split(',')[0]
    a = featuresets + [(gen_features(line), label)]
def cross_validate(data_files, folds):
    if len(data_files) % folds != 0:
        raise ValueError(
            "invalid number of folds ({}) for the number of "
            "documents ({})".format(folds, len(data_files))
        )
    fold_size = len(data_files) // folds
    for split_index in range(0, len(data_files), fold_size):
        training = data_files[split_index:split_index + fold_size]
        testing = data_files[:split_index] + data_files[split_index + fold_size:]
        yield training, testing
        print(training)
ss=cross_validate(a,2)
classifier = nltk.NaiveBayesClassifier.train(ss)

