

# writing result for submission
# write prediction for test data
# based on Yuan's code
# Author: Jingyi

import pandas as pd
import nltk
from nltk.metrics.scores import (f_measure, precision, recall)
import csv
from scipy import sparse
import pandas as pd
import numpy as np
import codecs
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

SelectedFeatures=[]
with codecs.open("F:\\Data Science\\Team Project\\tp\\SelectedFeaturesIndex.txt", "r", encoding='utf-8') as df0:
    for line in df0.readlines():
        SelectedFeatures.append(int(line))

df1 = sparse.load_npz("F:\\Data Science\\Team Project\\tp\\tfidf_sparse.npz")
df2 = pd.read_csv('F:\\Data Science\\Team Project\\tp\\ModelInput.csv' + 'ModelInput.csv', sep=',',
                      usecols=[13, *range(15, 32), *range(33, 49)], encoding='gbk')
y = pd.read_csv('F:\\Data Science\\Team Project\\tp\\ModelInput.csv', encoding='gbk',
                          usecols=['Label'])
entityids = pd.read_csv('F:\\Data Science\\Team Project\\tp\\ModelInput.csv', sep=',',
                usecols=[3,4])
entityids = np.array(entityids)
# dataset = pd.read_csv('F:\\Data Science\\Team Project\\tp\\ModelInput.csv', sep=',',
#                       usecols=['Label', 'positionInSentence_x', 'wordsInSentence_x', 'wordLen_x',
#                                'positionInSentence_y', 'wordsInSentence_y', 'wordLen_y'], encoding='gbk')

#tfidfs=pd.DataFrame(df1.toarray()).iloc[:,:10] # only use parts of tf-idf terms
tfidfs=pd.DataFrame(df1.toarray()) # uncomment it to use the whole tf-idf terms
#dataset=pd.concat([y,X])
#print(X.iloc[1,:])

# filling all missing data with 0
df2 = df2.fillna(0)

X = pd.concat([df2, tfidfs], axis=1)
X_selected= X.iloc[SelectedFeatures]
# reduce dimension of target set
x_l = np.array(y).flatten()

clf_best = joblib.load( 'F:\\Data Science\\Team Project\\tp\\SVM_best_0105.pkl.pkl')

# for row in X:
#     x_f.append(row[2:])  # the first index is entity id pair, 2 is label_index, the rest are features
#     x_l.append(row[1])
#     entity_pair.append(row[0])


clf_best.fit(X_selected, x_l)
'''only for testing'''
y_pred=clf_best.predict(X_selected)

# # delete if the file already exists (make sure only data for this run is stored)
# if os.path.exists(outf):
#     os.remove(outf)
#
for i, j in entityids, y_pred:
    if (y_pred == 1):
        linestr = "ANY("+i[0]+","+i[1]+")"
        print(linestr)
        with codecs.open('F:\\Data Science\\Team Project\\tp\\task2', 'wb',) as f:
            f.write(linestr)
            f.write('\n')

print("finish")