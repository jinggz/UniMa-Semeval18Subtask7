# coding: utf-8
# build majority baseline
# Author: Jingyi

# for this baseline, the full data set is used for both training and testing
import pandas as pd

def majority_vote(train,test):
    prediction=train['relation'].value_counts().idxmax()
    predicted = [ prediction for i in range(len(test))]
    return predicted


train = pd.read_csv('F:\\Data Science\\Team Project\\dataset\\relation1.1.csv')
class_feq = train['relation'].value_counts()    # get the statistics of the classes
test = train.reindex(columns=[*train.columns.tolist(), 'predicted'])    # add column for predicted value
predictions = majority_vote(train, test)
test['predicted'] = predictions
#print(test.head())
#print(test.describe())
print(class_feq)
print(predictions)