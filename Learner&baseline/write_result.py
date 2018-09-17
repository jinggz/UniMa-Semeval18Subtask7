# writing result for submission
# write prediction for test data
# Author: Yuan

import pandas as pd
import nltk
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

path = 'D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\Multi_class\\Task1.1\\'
dataset = pd.read_csv(path + 'ModelInput.csv', sep=',',
                      usecols=[5, 12, *range(15, 21), *range(33, 39)], encoding='gbk')

# use the whole tf-idf terms
df1 = sparse.load_npz(path + "tfidf_sparse.npz")
tfidfs = pd.DataFrame(df1.toarray())
# merge the dataset with tfidf
dataset = pd.concat([dataset, tfidfs], axis=1)

# delete the data which label_index=-1 (no relation)
dataset = dataset[dataset.label_index != -1]

# filling all missing data with 0
dataset = dataset.fillna(0)
dataset = np.array(dataset)

clf_best = joblib.load(path + 'bestSVMmodelknn0107.pkl')

x_f = []
x_l = []
y_pred = []
entity_pair = []

for row in dataset:
    x_f.append(row[2:])  # the first index is entity id pair, 2 is label_index, the rest are features
    x_l.append(row[1])
    entity_pair.append(row[0])

x_f = np.array(pd.DataFrame(x_f))
clf_best.fit(x_f, x_l)

y_pred=clf_best.predict(x_f)

# merge entity pairs and prediction
#con = np.concatenate((entity_pair, y_pred))
entity_pair = pd.DataFrame(entity_pair)
y_pred = pd.DataFrame(y_pred)

y_pred.replace(to_replace=1, value='USAGE', inplace=True)
y_pred.replace(to_replace=2, value='MODEL-FEATURE', inplace=True)
y_pred.replace(to_replace=3, value='PART_WHOLE', inplace=True)
y_pred.replace(to_replace=4, value='COMPARE', inplace=True)
y_pred.replace(to_replace=5, value='RESULT', inplace=True)
y_pred.replace(to_replace=6, value='TOPIC', inplace=True)

print(x_f)

#merge entity pair with prediction
con = pd.concat([entity_pair, y_pred], axis=1)

#edit the file name
file = path+'task1.1_knn0108.csv'
con.to_csv(file)