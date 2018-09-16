# coding: utf-8
# input: training samples(array X[n_samples, n_features]), and class labels (array y[n_samples])
# output: n x k-dimensional feature set
# Author: Jingyi
# reference: https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#overview

# Initialize test data, should be replaced by our real data
#############another test#####################
# from sklearn.datasets import load_breast_cancer
# breast_cancer=load_breast_cancer()
# from sklearn import svm
# X = breast_cancer.data[:100, :5]
# y = breast_cancer.target[:100]
################################################
#X.shape

import pandas as pd
import numpy as np
from sklearn import svm
######## load input file ########
inf = 'F:\\Data Science\\Team Project\\tp\\ModelInput.csv'
outf = 'medi-results\\output_SVM_integrated.csv' # file to output the predications
# seperate data into feature set and targetset
X = pd.read_csv(inf, sep=',',
                usecols=['positionInSentence_x', 'wordsInSentence_x', 'wordLen_x', 'positionInSentence_y', 'wordsInSentence_y', 'wordLen_y']) #选择要用的特征
y = pd.read_csv(inf, sep=',',
                usecols=['Label'])
entityId = pd.read_csv(inf, sep=',',
                usecols=['Entity1ID', 'Entity2ID'])

#filling all missing data with 0
X=X.fillna(0)
# transfer from type dataframe to array
X = np.array(X)

# reduce dimension of target set
y = np.array(y).flatten()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.33, random_state=1)


from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# classifier for learning features
svm1 = svm.SVC(kernel='linear', C=1)

# Sequential Backward Floating Selection with Cross Validation
sffs = SFS(svm1,
           k_features=(1,3),    # If k_features is set to to a tuple (min_k, max_k), the size of the returned feature subset is within max_k to min_k
           forward=False,
           floating=True,
           scoring='f1',
           cv=5,
           )
sffs = sffs.fit(X_train, y_train)

# Use scikit-learn's GridSearch to tune the hyperparameters inside and outside the SequentialFeatureSelector


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Define a pipeline combining feature selection and estimator used in selection
pipe = Pipeline([('sfs', sffs),
                 ('svm1', svm1)])

# Dictionary with parameters names (string) as keys and lists of parameter settings to try as values
param_grid = [
     {'sfs__estimator__kernel': ['rbf'], 'sfs__estimator__gamma': [1e-3, 1e-4],
                     'sfs__estimator__C': [1, 10]},
                    {'sfs__estimator__kernel': ['linear'], 'sfs__estimator__C': [1, 10]}
     #'sfs__estimator__kernel': ['linear'], 'sfs__estimator__C': [1, 10]}
     #'sfs__cv': [5, 8, 10]}
]
#scores = ['f1','precision', 'recall']
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='f1',
                  #scoring='%s_weighted'%f1,
                  n_jobs=1,
                  cv=5,
                  refit=True,
                  )

# run gridearch
gs = gs.fit(X_train, y_train)

print('Best features:', gs.best_estimator_.steps[0][1].k_feature_idx_)
print('CV Score:')
print(sffs.k_score_)

# visualize the results using matplotlib figures.
# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
# import matplotlib.pyplot as plt
# fig1 = plot_sfs(sffs.get_metric_dict(), kind='std_dev')
#
# plt.ylim([0.8, 1])
# plt.title('Sequential Forward Selection (w. StdDev)')
# plt.grid()
# plt.show()

#Generate the new subsets based on the selected features
#Note that the transform call is equivalent to
#X_train[:, sfs1.k_feature_idx_]

gs.best_estimator_.steps
gs.best_params_
print('Best svc:', gs.best_estimator_.steps[1][1])
print('Best svc_c:', gs.best_estimator_.steps[1][1].C)
if gs.best_estimator_.steps[1][1].kernel=='rbf':
    svm_best = svm.SVC(C=gs.best_estimator_.steps[1][1].C, kernel=gs.best_estimator_.steps[1][1].kernel, gamma=gs.best_estimator_.steps[1][1].gamma)
else:
    svm_best = svm.SVC(C=gs.best_estimator_.steps[1][1].C, kernel=gs.best_estimator_.steps[1][1].kernel)
#svm_best = svm.SVC(kernel='linear', C=gs.best_estimator_.steps[1][1].C)
X_train_sfs = X_train[:, gs.best_estimator_.steps[0][1].k_feature_idx_]
X_test_sfs = X_test[:, gs.best_estimator_.steps[0][1].k_feature_idx_]

# Fit the estimator using the new feature subset
# and make a prediction on the test data
svm_best.fit(X_train_sfs, y_train)
y_pred = svm_best.predict(X_test_sfs)

from sklearn import metrics
print("F1: ", metrics.f1_score(y_test,y_pred, average='weighted'))
print("Precision: ", metrics.precision_score(y_test,y_pred, average='weighted'))
print("Recall: ", metrics.recall_score(y_test,y_pred, average='weighted'))

