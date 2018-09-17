# for cross validation
# Author: Jingyi
import codecs
import csv
import numpy as np
from itertools import islice
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd


dir = "F:\\Data Science\\Team Project\\dataset\\"
fn = "PostaggingFile.csv"

df = codecs.open(dir+fn, "r", "UTF-8")
next(df)
data1 = np.loadtxt(df, dtype=str, delimiter=',', usecols=(0))
data = np.array(data1)
data_label=data.tolist()
df = codecs.open(dir+fn, "r", "UTF-8")
next(df)
data2 = np.loadtxt(df,dtype=str, delimiter=',', usecols=(1,2,3,4,5))
data = np.array(data2)
data_feature=data.tolist()


data_feature=data.tolist()
X_train, X_test, y_train, y_test = train_test_split(data_feature, data_label, test_size=0.6, random_state=1)




