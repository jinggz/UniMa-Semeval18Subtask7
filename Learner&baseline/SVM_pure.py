import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

path = "F:\\Data Science\\Team Project\\tp\\"
# dataset = pd.read_csv(path+'ModelInput_Stemmed.csv', sep=',',
#                 usecols=['Label', 'positionInSentence_x', 'wordsInSentence_x', 'wordLen_x','positionInSentence_y',
#                          'wordsInSentence_y', 'wordLen_y'], encoding='gbk')
dataset = pd.read_csv(path+'ModelInput.csv', sep=',',
                usecols=['Label','positionInSentence_x', 'wordsInSentence_x', 'wordLen_x','relativePosition_x','posTag_x', 'positionInSentence_y',
                         'wordsInSentence_y', 'wordLen_y','relativePosition_y','posTag_y'], encoding='gbk')
# #if selected index contains tfidf, then merge it with it, else just select the normal features
# if len(SelectedFeatures)>1000:
#     #tfidf matrix's width is 3987
#     a = len(SelectedFeatures)-3987
#     SelectedFeatures = SelectedFeatures[:a]
#     dataset = dataset.iloc [1:,SelectedFeatures]
#     dataset = pd.concat([dataset, tfidfs], axis=1, join='inner') # noted with the entity id
# else: dataset = dataset.iloc [1:,SelectedFeatures+1]

# get the classification report for the best parameter
# k-fold classification on the whole dataset
kf = KFold(n_splits=10)  # set k=10
dataset = dataset.fillna(0)
dataset=np.array(dataset)

y_true = []
y_pred = []
#clf_best=svm.SVC(C=10,kernel='rbf', gamma=0.01)
clf_best = GaussianNB()
for train, test in kf.split(dataset):
    x_f = []
    x_l = []
    train_set = []
    test_set = []
    pred = []
    for i in train:
#        print(dataset.iloc[i,:])
 #       train_set.append(dataset[i])
        x_f.append(dataset[i][1:])
        x_l.append(dataset[i][0])
    x_f = np.array(pd.DataFrame(x_f))
    #clf_best.fit(x_f, x_l)
    clf_best.partial_fit(x_f, x_l, np.unique(x_l))
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
