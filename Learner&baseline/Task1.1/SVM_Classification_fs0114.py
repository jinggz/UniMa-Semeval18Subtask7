# Subtask 1.1
# SVM with feature selection, grid search
# Author: Jingyi (Yuan modified)


import pandas as pd
import numpy as np
from itertools import combinations
from scipy import sparse
import matplotlib.pyplot as plt
from nltk.metrics.scores import (f_measure, precision, recall)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics
import pickle
from sklearn.externals import joblib
from sklearn.metrics import f1_score, make_scorer

path = 'D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\Multi_class\\Task1.1\\'
# seperate data into feature set and targetset
# X is feature set, y is label set, dataset is the merge of the both 2
X = pd.read_csv(path + 'ModelInput.csv', sep=',',
                usecols=[10, 11, *range(13, 31), *range(32, 50)], encoding='gbk')
y = pd.read_csv(path + 'ModelInput.csv', sep=',',
                usecols=['label_index'], encoding='gbk')
#dataset = pd.read_csv(path + 'ModelInput.csv', sep=',',
  #                    usecols=[10, 11, *range(13, 31), *range(32, 50)], encoding='gbk')

# print(X.columns.values)

# use the whole tf-idf terms
df1 = sparse.load_npz(path + "tfidf_sparse.npz")
tfidfs = pd.DataFrame(df1.toarray())
# merge the dataset with tfidf
#dataset = pd.concat([dataset, tfidfs], axis=1)
X = pd.concat([X, tfidfs], axis=1)

# delete the data which label_index=-1 (no relation)
#dataset = dataset[dataset.label_index != -1]
X = X[X.label_index != -1]
y = y[y.label_index != -1]
X.drop(['label_index'], axis=1, inplace=True)
X=X[:1213]

# print(X.columns.values)

# filling all missing data with 0
X = X.fillna(0)
#dataset = dataset.fillna(0)

# reduce dimension of target set
y_train = np.array(y).flatten()

# store indices of features
means = [0]
posInSentence = [1, 19]
wordsInSentence = [2, 20]
wordLen = [3, 21]
relativePos = [4, 22]
posTag01 = [5, 23]
posTag02 = [6, 24]
posTag03 = [7, 25]
posTag04 = [8, 26]
posTag05 = [9, 27]
posTag06 = [10, 28]
posTag07 = [11, 29]
posTag08 = [12, 30]
posTag09 = [13, 31]
posTag10 = [14, 32]
posTag11 = [15, 33]
posTag12 = [16, 34]
lastLetterInt = [17, 35]
firstLetterInt = [18, 36]
tfidf = [i for i in range(37, X.shape[1] - 1)]
# print(tfidf)

features = [means, posInSentence, wordsInSentence, wordLen, relativePos, posTag01, posTag02, posTag03, posTag04,
            posTag05,posTag06, posTag07, posTag08, posTag09, posTag10, posTag11, posTag12, lastLetterInt, firstLetterInt, tfidf]

# use squential forward feature subset selection combined with grid search and SVM model

# Set the parameters by cross-validation
tuned_parameters = [{'C': [0.001, 0.01]}]
LinearSVC(class_weight='balanced')
svc = svm.LinearSVC()

'''squential forward feature subset selection'''


def calc_score(X, y, indices, tuned_parameters, estimator):
    print("# Tuning hyper-parameters for F1-macro")
    # run grid search in learner for feature subset
    clf = GridSearchCV(estimator, tuned_parameters, cv=5,
                       scoring='f1_macro')  # for relation classification,we use criteria F1 macro
    temp = X.iloc[:, indices]
    # print(temp.shape)
    X_ = np.array(temp)
    clf.fit(X_, y)

    # if the best kernel is linear, there is no gamma parameter
    clf_best = svm.LinearSVC(C=clf.best_params_['C'])

    # clf_best.fit(X, y)
    # save svm model for all parameters, load back the model using #clf = joblib.load('allSVMmodel.pkl')
    # joblib.dump(clf, 'allSVMmodelWOsplit.pkl')
    score = clf.best_score_
    model = pickle.dumps(clf_best)
    return score, model


# define a customized f1 score
# myown_f1 = make_scorer(f1_score,average='binary',pos_label = 1)

SelectedFeatures = []  # store the final indices of best feature
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
print("The best F1 score is: ", previousscore)
print("The indices of best features are: ", SelectedFeatures)
clf_best = pickle.loads(bestmodel)
print("The best model is: ", clf_best)
# save the best svm model, load back the model using #clf_best = joblib.load('bestSVMmodel.pkl')
joblib.dump(clf_best, 'bestSVMmodelSVM0114.pkl')

# if selected index contains tfidf, then merge it with it, else just select the normal features
if len(SelectedFeatures) > 1000:
    # tfidf matrix's width is 3975
    a = len(SelectedFeatures) - 3975
    SelectedFeatures = SelectedFeatures[:a]
    dataset = dataset.iloc[1:, SelectedFeatures]
    dataset = pd.concat([dataset, tfidfs], axis=1, join='inner')  # noted with the entity id
else:
    dataset = dataset.iloc[1:, SelectedFeatures + 1]

'''
# get the classification report for the best parameter
# k-fold classification on the whole dataset
kf = KFold(n_splits=10)  # set k=10

y_true = []
y_pred = []

dataset = dataset.fillna(0)
dataset = np.array(dataset)

for train, test in kf.split(dataset):
    x_f = []
    x_l = []
    train_set = []
    test_set = []
    pred = []
    for i in train:
        x_f.append(dataset[i][1:])
        x_l.append(dataset[i][0])
    x_f = np.array(pd.DataFrame(x_f))
    clf_best.fit(x_f, x_l)
    # collect all the real label and predicted label
    for j in test:
        y_true.append(dataset[j][0])
        test_set.append(dataset[j][1:])
    test_set = np.array(pd.DataFrame(test_set))
    # print classification report
    pred = clf_best.predict(test_set)
    y_pred.extend(pred)
print(classification_report(y_true, y_pred))
print('finish')
'''