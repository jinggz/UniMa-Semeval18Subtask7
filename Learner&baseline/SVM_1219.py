#this file include grid search, feature selection, reading tfidf
#Author: Jingyi
#Edited: Yuan

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


path = "D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\"
df1 = sparse.load_npz("tfidf_sparse.npz")
df2 = pd.read_csv(path+'ModelInput.csv', sep=',',
                usecols=['positionInSentence_x', 'wordsInSentence_x', 'wordLen_x','relativePosition_x', 'posTag_x','positionInSentence_y',
                         'wordsInSentence_y', 'wordLen_y','relativePosition_y','posTag_y'], encoding='gbk')
y = pd.read_csv(path+'ModelInput.csv', sep=',',
                usecols=['Label'], encoding='gbk')
# dataset = pd.read_csv('F:\\Data Science\\Team Project\\tp\\ModelInput.csv', sep=',',
#                       usecols=['Label', 'positionInSentence_x', 'wordsInSentence_x', 'wordLen_x',
#                                'positionInSentence_y', 'wordsInSentence_y', 'wordLen_y'], encoding='gbk')

#tfidfs=pd.DataFrame(df1.toarray()).iloc[:,:10] # only use parts of tf-idf terms
tfidfs=pd.DataFrame(df1.toarray()) # uncomment it to use the whole tf-idf terms
#dataset=pd.concat([y,X])
#print(X.iloc[1,:])

# filling all missing data with 0
df2 = df2.fillna(0)

X = pd.concat([df2, tfidfs], axis=1, join='inner') # noted with the entity id

# transfer from type dataframe to array
#X_train = np.array(X)

# reduce dimension of target set
y_train = np.array(y).flatten()
#store indices of features
posInSentence = [1,6]
wordsInSentence= [2,7]
relativePos=[4,9]
wordLen=[3,8]
posTag=[5,10]
tfidf=[i for i in range(9, X.shape[1])]
print(tfidf)

features=[posInSentence,wordsInSentence,wordLen,relativePos,posTag,tfidf]

# use squential forward feature subset selection combined with grid search and SVM model

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3],
                'C': [1, 10]}]
svc = svm.SVC()

'''squential forward feature subset selection'''
def calc_score(X, y, indices, tuned_parameters,estimator):
    print("# Tuning hyper-parameters for F1")
    # run grid search in learner for feature subset
    clf = GridSearchCV(estimator, tuned_parameters, cv=2,
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
clf_best = pickle.loads(bestmodel)
print("The best model is: ", clf_best)
# save the best svm model, load back the model using #clf_best = joblib.load('bestSVMmodel.pkl')
joblib.dump(clf_best, 'bestSVMmodel1.pkl')



dataset = pd.read_csv(path+'ModelInput.csv', sep=',',
                usecols=['Label','positionInSentence_x', 'wordsInSentence_x', 'wordLen_x','relativePosition_x','posTag_x', 'positionInSentence_y',
                         'wordsInSentence_y', 'wordLen_y','relativePosition_y','posTag_y'], encoding='gbk')
#if selected index contains tfidf, then merge it with it, else just select the normal features
if len(SelectedFeatures)>1000:
    #tfidf matrix's width is 3987
    a = len(SelectedFeatures)-3987
    SelectedFeatures = SelectedFeatures[:a]
    dataset = dataset.iloc [1:,SelectedFeatures]
    dataset = pd.concat([dataset, tfidfs], axis=1, join='inner') # noted with the entity id
else: dataset = dataset.iloc [1:,SelectedFeatures+1]

# get the classification report for the best parameter
# k-fold classification on the whole dataset
kf = KFold(n_splits=5)  # set k=10

y_true = []
y_pred = []

for train, test in kf.split(dataset):
    x_f = []
    x_l = []
    train_set = []
    test_set = []
    pred = []
    for i in train:
        train_set.append(dataset[i])
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
