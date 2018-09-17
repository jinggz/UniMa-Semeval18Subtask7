# subtask 2 -- relation extraction
# apply different models on test data
# structure is based on svm, varied by different models
# Author: Jingyi
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
import codecs
import os
from sklearn.neighbors import KNeighborsClassifier

path = "F:\\Data Science\\Team Project\\tp\\"

# This function will be used to slice features when manually entering feature names is costly.
def slice_features(allfeatures, usedfeatures):
    if len(usedfeatures)<100:
        allfeatures = allfeatures.fillna(0)
        return allfeatures.values[:, usedfeatures]
    else:
        tfidf = sparse.load_npz("tfidf_0120_clean.npz")
        tfidf=tfidf.toarray()
        # filling all missing data with 0
        allfeatures = allfeatures.fillna(0)
        allfeatures = np.hstack((np.array(allfeatures), tfidf))
        return allfeatures[:, usedfeatures]

'''Part 1: Read Training set'''
# as the selected features are not many, read the selected features directly by entering the names of features
X = pd.read_csv(path+'ModelInput0124.csv',
                  usecols=['positionInSentence_y',
'positionInSentence_x',
'distLastEntity_y',
'distNextEntity_x',
'wordLen_y',
'posTag01_y = 15.0',
'posTag03_x = 3.0',
'firstLetterInt_y',
'posTag02_x = 17.0',
'relativePosition_x',
'posTag01_y = 17.0',
'posTag01_y = 10.0'], encoding='gbk')
y = pd.read_csv(path + 'ModelInput0124.csv', encoding='gbk',
                          usecols=['LabelInput'])

#best_features=np.loadtxt(path+'Results\\SelectedFeaturesIndices011804.txt',delimiter='\r\n',dtype=int)
#X = slice_features(X, best_features)
X=X.fillna(0)
X=np.array(X)
'''Part 2: Read Best Model and Train'''
best_model=joblib.load(path+'Results\\bestModelLinearSVC0126.pkl')
best_model.fit(X, y.values.flatten())

'''Part 3: Read Test Set'''
X_test = pd.read_csv(path+'testdata\\TestInput2.csv',
                  usecols=['positionInSentence_y',
'positionInSentence_x',
'distLastEntity_y',
'distNextEntity_x',
'wordLen_y',
'posTag01_y = 15.0',
'posTag03_x = 3.0',
'firstLetterInt_y',
'posTag02_x = 17.0',
'relativePosition_x',
'posTag01_y = 17.0',
'posTag01_y = 10.0'], encoding='gbk')
# y_test = pd.read_csv(path + 'testdata\\TestInput2.csv', encoding='gbk',
#                           usecols=['LabelInput'])

#X_test = slice_features(X_test,best_features)
X_test=X_test.fillna(0)
X_test=np.array(X_test)
y_pred = best_model.predict(X_test)

'''Part 4: Write Output'''

entityids = pd.read_csv(path+'testdata\\TestInput2.csv', sep=',',
                usecols=['Entity Pair'])
entityids = np.array(entityids).flatten()
print("xshape", X_test.shape)
print("yshape", len(y_pred))
print("idpairshape",entityids.shape )
# delete if the file already exists (make sure only data for this run is stored)
if os.path.exists(path+ 'Results\\2_extractionSubmit_LinearSVC0127.txt'):
    os.remove(path+ 'Results\\2_extractionSubmit_LinearSVC0127.txt')

#zip(entityids,y_pred)
for pair, pred in zip(entityids,y_pred):
    if (pred == 1):
        linestr = "ANY"+ pair
        with codecs.open(path + 'Results\\2_extractionSubmit_LinearSVC0127.txt', 'a', encoding='utf-8') as f:
            f.write(linestr)
            f.write('\r\n')
#print(f1_score(y_test.values, y_pred, average='binary'))
print("finish")

