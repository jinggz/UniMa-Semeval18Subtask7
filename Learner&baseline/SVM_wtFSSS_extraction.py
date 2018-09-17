# SVM for relation extraction
# include feature selection, grid search
# Author: Jingyi

import pandas as pd
import numpy as np
from itertools import combinations
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import metrics
import pickle
from sklearn.externals import joblib
from sklearn.metrics import f1_score, make_scorer
import codecs

df1 = sparse.load_npz("tfidf_sparse.npz")
df2 = pd.read_csv('F:\\Data Science\\Team Project\\tp\\ModelInput.csv' + 'ModelInput.csv', sep=',',
                      usecols=[13, *range(15, 32), *range(33, 49)], encoding='gbk')
y = pd.read_csv('F:\\Data Science\\Team Project\\tp\\ModelInput.csv', encoding='gbk',
                          usecols=['Label'])
# dataset = pd.read_csv('F:\\Data Science\\Team Project\\tp\\ModelInput.csv', sep=',',
#                       usecols=['Label', 'positionInSentence_x', 'wordsInSentence_x', 'wordLen_x',
#                                'positionInSentence_y', 'wordsInSentence_y', 'wordLen_y'], encoding='gbk')

tfidfs=pd.DataFrame(df1.toarray()).iloc[:,:10] # only use parts of tf-idf terms
#tfidfs=pd.DataFrame(df1.toarray()) # uncomment it to use the whole tf-idf terms
#dataset=pd.concat([y,X])
#print(X.iloc[1,:])

# filling all missing data with 0
df2 = df2.fillna(0)

X = pd.concat([df2, tfidfs], axis=1, join='inner')

# transfer from type dataframe to array
#X_train = np.array(X)

# reduce dimension of target set
y_train = np.array(y).flatten()
#store indices of features, hard coding
posInSentence = [1,18]
wordsInSentence= [2,19]
relativePos=[4,21]
wordLen=[3,20]
tfidf=[i for i in range(33, X.shape[1])]
#print(tfidf)

features=[posInSentence,wordsInSentence,wordLen,relativePos,0,5,6,7,8,9,10,11,12,13,14,15,15, 22,24,25,26,27,28,29,30,31,32,32,tfidf]

# use squential forward feature subset selection combined with grid search and SVM model

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 0.01, 0.5, 1, 10 ],
                'C': [1, 10, 100]},
            {'kernel': ['linear'], 'C': [0.1, 1, 10]}]
svc = svm.SVC()

'''squential forward feature subset selection'''
def calc_score(X, y, indices, tuned_parameters,estimator):
    print("# Tuning hyper-parameters for F1")
    # run grid search in learner for feature subset
    clf = GridSearchCV(estimator, tuned_parameters, cv=5,
             scoring = myown_f1)
    temp = X.iloc[ : , indices]
    #print(temp.shape)
    X_ = np.array(temp)
    clf.fit(X_ , y)

    #if the best kernel is linear, there is no gamma parameter
    if clf.best_params_['kernel']=='rbf':
        clf_best = svm.SVC(C=clf.best_params_['C'], kernel=clf.best_params_['kernel'],gamma=clf.best_params_['gamma'])
    else:
        clf_best = svm.SVC(C=clf.best_params_['C'], kernel=clf.best_params_['kernel'])
    #clf_best.fit(X, y)
    #save svm model for all parameters, load back the model using #clf = joblib.load('allSVMmodel.pkl')
    #joblib.dump(clf, 'allSVMmodelWOsplit.pkl')
    score = clf.best_score_
    model = pickle.dumps(clf_best)
    return score, model

# define a customized f1 score
myown_f1 = make_scorer(f1_score,average='binary',pos_label = 1)

SelectedFeatures = []   # store the final indices of best feature
k = 5
previousscore = -1  # store the final best performance
bestmodel = ''  # store the final best model
while features:
    #    print("while loop:")
    maxs = -1
    maxmodel = ''
    maxfs = -1
    for i in range(0, len(features)):
        # tested feature set
        testfs = []
        testfs.extend(SelectedFeatures)
        testfs.extend(features[i])

        #        print(len(testfs))

        # get the optimal model and performance for the tested feature set
        score, model = calc_score(X, y_train, testfs, tuned_parameters, svc)
        if score > maxs:
            maxs = score
            maxmodel = model
            maxfs = i

    if maxs > previousscore:
        previousscore = maxs
        bestmodel = maxmodel
    else:
        break
    SelectedFeatures.extend(features[maxfs])
    #    print(maxfs)
    #    print(len(SelectedFeatures))
    del features[maxfs]


'''result display'''
print("The best F1 score is: ", previousscore )
print("The indices of best features are: ", SelectedFeatures)
with codecs.open("SelectedFeaturesIndex.txt", "wb", encoding='utf-8') as fp:
    for item in SelectedFeatures:
        fp.write("%s\n" % item)

clf_best = pickle.loads(bestmodel)
print("The best model is: ", clf_best)
# save the best svm model, load back the model using #clf_best = joblib.load('bestSVMmodel.pkl')
joblib.dump(clf_best, 'bestSVMmodel1.pkl')


