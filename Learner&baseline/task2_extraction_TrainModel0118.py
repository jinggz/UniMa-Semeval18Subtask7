# subtask 2 - Relation Extraction
# Pipeline for Model Learning
# including feature selection, grid search, cross validation
# reused in classification: change learner in Line74, scoring in Line80, and other related parameter
# Author: Jingyi

import pandas as pd
import numpy as np
import os
from scipy import sparse
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pickle
from sklearn.externals import joblib
import codecs
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier

path = 'F:\\Data Science\\Team Project\\tp\\'
# for svm
X = pd.read_csv(path+'ModelInput0124.csv',
                  usecols=[ 'means', 'positionInSentence_x', 'wordsInSentence_x',
 'wordLen_x', 'distLastEntity_x', 'distNextEntity_x', 'relativePosition_x',
 'lastLetterInt_x', 'firstLetterInt_x', 'lastVerbDistance_x',
 'nextVerbDistance_x', 'hasWikiAricle_x', 'positionInSentence_y',
 'wordsInSentence_y', 'wordLen_y', 'distLastEntity_y', 'distNextEntity_y',
 'relativePosition_y', 'lastLetterInt_y', 'firstLetterInt_y',
 'lastVerbDistance_y', 'nextVerbDistance_y', 'hasWikiAricle_y',
 'posTag01_x = 9.0', 'posTag01_x = 11.0', 'posTag01_x = 6.0',
 'posTag01_x = 18.0', 'posTag01_x = 17.0', 'posTag01_x = 13.0',
 'posTag01_x = 10.0', 'posTag01_x = 5.0', 'posTag01_x = 3.0',
 'posTag01_x = 15.0', 'posTag01_x = 4.0', 'posTag01_x = 7.0',
 'posTag01_x = 16.0', 'posTag02_x = 11.0', 'posTag02_x = 9.0',
 'posTag02_x = -2.0', 'posTag02_x = 6.0', 'posTag02_x = 17.0',
 'posTag02_x = 5.0', 'posTag02_x = 2.0', 'posTag02_x = 18.0',
 'posTag02_x = 3.0', 'posTag02_x = 4.0', 'posTag02_x = 10.0',
 'posTag02_x = 16.0', 'posTag02_x = 14.0', 'posTag03_x = 2.0',
 'posTag03_x = 11.0', 'posTag03_x = -2.0', 'posTag03_x = 5.0',
 'posTag03_x = 9.0', 'posTag03_x = 17.0', 'posTag03_x = 6.0',
 'posTag03_x = 13.0', 'posTag03_x = 18.0', 'posTag03_x = 10.0',
 'posTag03_x = 4.0', 'posTag03_x = 14.0', 'posTag03_x = 16.0',
 'posTag03_x = 3.0', 'posTag03_x = -1.0', 'posTag04_x = 11.0',
 'posTag04_x = -2.0', 'posTag04_x = 9.0', 'posTag04_x = 6.0',
 'posTag04_x = 5.0', 'posTag04_x = 17.0', 'posTag04_x = 18.0',
 'posTag04_x = 14.0', 'posTag04_x = 16.0', 'posTag04_x = 2.0',
 'posTag04_x = 4.0', 'posTag04_x = 15.0', 'posTag04_x = 10.0',
 'posTag05_x = -2.0', 'posTag05_x = 11.0', 'posTag05_x = 9.0',
 'posTag05_x = 6.0', 'posTag05_x = 14.0', 'posTag05_x = 10.0',
 'posTag05_x = 3.0', 'posTag05_x = 4.0', 'posTag05_x = 2.0',
 'posTag05_x = 5.0', 'posTag05_x = 17.0', 'posTag05_x = -1.0',
 'posTag06_x = -2.0', 'posTag06_x = 9.0', 'posTag06_x = 11.0',
 'posTag06_x = 5.0', 'posTag06_x = 6.0', 'posTag06_x = 4.0',
 'posTag01_y = 9.0', 'posTag01_y = 11.0', 'posTag01_y = 6.0',
 'posTag01_y = 18.0', 'posTag01_y = 17.0', 'posTag01_y = 5.0',
 'posTag01_y = 13.0', 'posTag01_y = -1.0', 'posTag01_y = 10.0',
 'posTag01_y = 3.0', 'posTag01_y = 16.0', 'posTag01_y = 15.0',
 'posTag01_y = 7.0', 'posTag02_y = -2.0', 'posTag02_y = 9.0',
 'posTag02_y = 11.0', 'posTag02_y = 17.0', 'posTag02_y = 6.0',
 'posTag02_y = 18.0', 'posTag02_y = 5.0', 'posTag02_y = 2.0',
 'posTag02_y = 13.0', 'posTag02_y = 10.0', 'posTag02_y = 4.0',
 'posTag02_y = 16.0', 'posTag02_y = 3.0', 'posTag02_y = -1.0',
 'posTag02_y = 14.0', 'posTag03_y = -2.0', 'posTag03_y = 5.0',
 'posTag03_y = 9.0', 'posTag03_y = 2.0', 'posTag03_y = 11.0',
 'posTag03_y = 17.0', 'posTag03_y = 6.0', 'posTag03_y = 18.0',
 'posTag03_y = 10.0', 'posTag03_y = 4.0', 'posTag03_y = 3.0',
 'posTag03_y = -1.0', 'posTag04_y = -2.0', 'posTag04_y = 11.0',
 'posTag04_y = 9.0', 'posTag04_y = 5.0', 'posTag04_y = 6.0',
 'posTag04_y = 17.0', 'posTag04_y = 14.0', 'posTag04_y = 16.0',
 'posTag04_y = 2.0', 'posTag04_y = 10.0', 'posTag04_y = 13.0',
 'posTag04_y = 3.0', 'posTag04_y = 4.0', 'posTag04_y = 18.0',
 'posTag05_y = -2.0', 'posTag05_y = 9.0', 'posTag05_y = 11.0',
 'posTag05_y = 6.0', 'posTag05_y = 14.0', 'posTag05_y = 17.0',
 'posTag05_y = 10.0', 'posTag05_y = 5.0', 'posTag05_y = 18.0',
 'posTag05_y = 2.0', 'posTag06_y = -2.0', 'posTag06_y = 5.0',
 'posTag06_y = 9.0', 'posTag06_y = 11.0', 'posTag06_y = 6.0',
 'posTag06_y = 17.0', 'posTag06_y = -1.0','hasUse_x','hasUse_y','hasOf_x','hasUse_y'], encoding='gbk')
# for knn
# X = pd.read_csv(path+'ModelInput0124.csv',
#                   usecols=[ 'means', 'positionInSentence_x', 'wordsInSentence_x',
#  'wordLen_x', 'distLastEntity_x', 'distNextEntity_x', 'relativePosition_x',
#  'lastLetterInt_x', 'firstLetterInt_x', 'lastVerbDistance_x',
#  'nextVerbDistance_x', 'hasWikiAricle_x', 'positionInSentence_y',
#  'wordsInSentence_y', 'wordLen_y', 'distLastEntity_y', 'distNextEntity_y',
#  'relativePosition_y', 'lastLetterInt_y', 'firstLetterInt_y',
#  'lastVerbDistance_y', 'nextVerbDistance_y', 'hasWikiAricle_y',
#   'posTag01_x',    'posTag02_x',    'posTag03_x',    'posTag04_x', 'posTag05_x',    'posTag06_x',
#    'posTag07_x',    'posTag08_x',   'posTag09_x','posTag01_y',	'posTag02_y',	'posTag03_y',	'posTag04_y',
#    'posTag05_y',	'posTag06_y',	'posTag07_y',	'posTag08_y',	'posTag09_y'], encoding='gbk')
y = pd.read_csv(path + 'ModelInput0124.csv', encoding='gbk',
                          usecols=['LabelInput'])

tfidf = sparse.load_npz("tfidf_0120_clean.npz")
tfidf=tfidf.toarray() # uncomment it to use the whole tf-idf terms

# filling all missing data with 0
X = X.fillna(0)

X=np.array(X)
fs_shape=X.shape[1]
X = np.hstack((np.array(X), tfidf))

# reduce dimension of target set
y = y.values.flatten()

#store indices of features for svm
features=list(range(fs_shape))
tfidf = list(range(fs_shape, X.shape[1]))
#store indices of features for knn
#features=list(range(35))
#tfidf = list[range(41, X.shape[1])]
#print(tfidf)

# tuned parameter
features.append(tfidf)
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 0.01, 0.5, 1, 10 ],
#                 'C': [1, 10, 100]},
#             {'kernel': ['linear'], 'C': [0.1, 1, 10]}]
tuned_parameters = [{'C': [0.1,1,10], 'class_weight': ['balanced', None]}]
#tuned_parameters = [{'n_neighbors': [3,5],'weights': ['distance','uniform']}]
#learner = svm.SVC() # the model for tuning
learner = LinearSVC()
#learner = KNeighborsClassifier()

'''Sequential Forward Feature Selection'''
def calc_score(X, y, indices, tuned_parameters,estimator):
    # run grid search in learner for feature subset
    clf = GridSearchCV(estimator, tuned_parameters, cv=10,
             scoring = myown_f1)
    clf.fit(X[:, indices], y)
    clf_best = clf.best_estimator_
    score = clf.best_score_
    model = pickle.dumps(clf_best)
    return score, model

# define a customized f1 score, after experiment, it is performed better than default binary fi-score
myown_f1 = make_scorer(f1_score,average='binary',pos_label = 1)

'''Main Method'''
SelectedFeatures = []   # store the final indices of best feature set
k = 30  # maximum number of features
lastscore = -1  # store the final best score
#last2score = -1 # store the last but one score
bestmodel = ''  # store the final best model
while features:
    print("while loop:")
    maxs = -1   # store the best score for one outer loop with the same number of features
    maxmodel = '' # store the best model for one outer loop with the same number of features
    maxfs = -1  # store the best feature indices for one outer loop with the same number of features
    for i in range(0, len(features)):
        # tested feature set
        testfs = []
        testfs.extend(SelectedFeatures)
        if type(features[i]) is not int:
            testfs.extend(features[i])
        else:
            testfs.append(features[i])
        #        print(len(testfs))

        # get the optimal model and performance for the tested feature set
        score, model = calc_score(X, y, testfs, tuned_parameters, learner)
        if score > maxs:
            maxs = score
            maxmodel = model
            maxfs = i

    if maxs <= lastscore:
        break

    lastscore = maxs
    bestmodel = maxmodel
    if type(features[maxfs]) is not int:
        SelectedFeatures.extend(features[maxfs])
    else:
        SelectedFeatures.append(features[maxfs])
    #    print(maxfs)
    #    print(len(SelectedFeatures))
    del features[maxfs]


'''result display'''
print("The best F1 score is: ", lastscore )
with open(path+ "Results\\score0125svm_.txt", "w", encoding='utf-8') as f:
        f.write(str(lastscore))
print("The indices of best features are: ", SelectedFeatures)
np.savetxt(path+ 'Results\\SelectedFeaturesIndices0125svm.txt',SelectedFeatures,fmt='%i',delimiter='\r\n')
clf_best = pickle.loads(bestmodel)
print("The best model is: ", clf_best)
# save the best svm model, load back the model using #clf_best = joblib.load('bestSVMmodel.pkl')
joblib.dump(clf_best, path+'Results\\bestMode0125svm.pkl')

print('Finish')
