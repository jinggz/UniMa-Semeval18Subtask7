
# coding: utf-8
# integrated SVM model with feature selection, grid search and cross-validation
# will keep updating
# Author: Jingyi


# In[6]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#          X, y, test_size=0.33, random_state=1)


# In[42]:


from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
import pickle
from sklearn.externals import joblib
from sklearn.metrics import f1_score, make_scorer

# define a customized f1 score
myown_f1 = make_scorer(f1_score,average='binary',pos_label = 1)

# sequential backward feature selection with embedded classifier
class SBS():
    def __init__(self, estimator, k_features,tuned_parameters): 
        self.estimator = clone(estimator)         
        self.k_features = k_features
        self.tuned_parameters = tuned_parameters    

    def fit(self, X, y):
        dim = X.shape[1]
        self.indices_ = tuple(range(dim)) 
        self.subsets_ = [self.indices_]
        self.models_ = []
        score, model = self._calc_score(X, y, 
                                 self.indices_, tuned_parameters)
        self.scores_ = [score]
        self.models_ = [model]
        while dim > self.k_features :
            scores = []
            subsets = []
            models = []
            for p in combinations(self.indices_, r=dim - 1):
                score, model = self._calc_score(X, y, p, tuned_parameters)

                scores.append(score)
                subsets.append(p)
                models.append(model)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            self.models_.append(models[best])
            self.scores_.append(scores[best])
            dim -= 1
#             if self.scores_[-1] <= self.scores_[-2] and dim <= self.k_features:
#                  break
             
        self.k_score = self.scores_[-1] # put the scoring of k feature set
        self.k_model = self.models_[-1] # put the best model trained by k features
        self.k_subset = self.subsets_[-1] # put the k feature set
        # alternatively, best_xx stores the best feature subsets in the range [k,dim]
        best = np.argmax(self.scores_)
        self.best_subset= self.subsets_[best]
        self.best_score = self.scores_[best]
        self.best_model = self.models_[best]

        return self

    def transform(self, X):
        return X[:, self.indices_]
    def _calc_score(self, X, y, indices, tuned_parameters):
        print("# Tuning hyper-parameters for F1")
        # run grid search in learner for feature subset
        clf = GridSearchCV(self.estimator, tuned_parameters, cv=2,
                scoring = myown_f1)
        clf.fit(X[:, indices], y)
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
        


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
import numpy as np
inf = 'F:\\Data Science\\Team Project\\tp\\ModelInput.csv'
outf = 'medi-results\\output_SVM_integrated.csv' # file to output the predications
# selecting features
k_features = 2
# selecting tuned parameters for Grid Search
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 0.01, 0.5, 1, 10 ],
                'C': [1, 10, 100]},
            {'kernel': ['linear'], 'C': [0.1, 1, 10]}]
#svc = svm.SVC(class_weight={1: 3})
svc = svm.SVC()
######## load input file ########


# seperate data into feature set and targetset
X = pd.read_csv(inf, sep=',',
                usecols=['positionInSentence_x', 'wordsInSentence_x', 'wordLen_x', 'positionInSentence_y', 'wordsInSentence_y', 'wordLen_y']) # select features wanted to import
y = pd.read_csv(inf, sep=',',
                usecols=['Label'])
entityId = pd.read_csv(inf, sep=',',
                usecols=['Entity1ID', 'Entity2ID'])

#filling all missing data with 0
X=X.fillna(0)
# transfer from type dataframe to array
X_train = np.array(X)

# reduce dimension of target set
y_train = np.array(y).flatten()
        
#         X_train, X_test, y_train, y_test = \
#             train_test_split(X, y, test_size=self.test_size,
#                              random_state=self.random_state)
##############

# use squential backward feature subset selection combined with grid search and SVM model
sbs = SBS(svc, k_features, tuned_parameters)
sbs.fit(X_train, y_train)


print("The best F1 score is: ", sbs.k_score)
print("The k-features are: ", sbs.k_subset)

clf_best = pickle.loads(sbs.k_model)
print("The best model is: ", clf_best)
clf_best.fit(X_train[:, sbs.k_subset], y_train)
#clf_best.fit(X_test[:, sbs.k_subset], y_test)
# save the best svm model, load back the model using #clf_best = joblib.load('bestSVMmodel.pkl')
joblib.dump(clf_best, 'bestSVMmodelWOsplit.pkl')
# save the indexs of feature subsets
import codecs
with codecs.open("featuresubset.txt", "wb", encoding='utf-8') as fp:   
    for item in sbs.k_subset:
        fp.write("%s\n" % item)

# In[12]: Apply model: using seperate test dataset
#
# y_true, y_pred = y_train, clf_best.predict(X_test[:, sbs.k_subset])
# print(classification_report(y_true, y_pred))


# ######## write predication file ########
# y_true, y_pred = y_train, clf_best.predict(X_train[:, sbs.k_subset])
# print(classification_report(y_true, y_pred))

# # delete if the file already exists (make sure only data for this run is stored)
# if os.path.exists(outf):
#     os.remove(outf)
#
# save = pd.DataFrame(y_pred, entityId)
# #fieldnames = ['relation', 'feature1', 'feature2', 'feature3', 'feature4','prediction']
# save.to_csv(outf, sep=',', encoding='utf-8')
# # still need to convert to the format according to the orgnization
# # example: MODEL-FEATURE(H01-1041.10, H01-1041.11)
# ###############

########## result visualization ##############
# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('F1 score')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
#plt.savefig('./sbs2.png', dpi=300)
plt.show()






