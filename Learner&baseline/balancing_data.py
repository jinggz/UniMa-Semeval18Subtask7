# Naive random over-sampling
# input: imbalanced data  dtype: 'list'
# output: rebalanced data  dtype: 'list
# Author: Jingyi
import pandas as pd
import numpy as np


#df = pd.read_csv('F:\\Data Science\\Team Project\\svn\\Teamprojekt2017\\Project-code\\ReadDataAndPostagging\\PostaggingFile.csv', sep=',')

def balancingData(train):
    # split the dataset into x: feature set and y: label set
    train = np.array(train)
    X = np.array(train[:, 1:6])# parameter in [] should CHANGE when number or order of features change!!!
    y = np.array(train[:, 0])     # parameter in [] should CHANGE when order of label change!!!
    from collections import Counter
    #print(sorted(Counter(y).items())) # imbalanced classes
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(X, y)
    #print(sorted(Counter(y_resampled).items())) # rebalanced classes
    y_resampled = y_resampled[:, np.newaxis]
    # join the feature and label sets and transform to dtype 'list'
    train_resampled = np.concatenate((y_resampled, X_resampled), axis=1)
    train_resampled = train_resampled.tolist()
    #return X_resampled,y_resampled
    return train_resampled



