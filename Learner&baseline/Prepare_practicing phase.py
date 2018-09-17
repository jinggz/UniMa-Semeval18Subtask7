# prepare train and test set for practicing phase
# Input1: ModelInput.csv;
# Input2: Abstract list provided by Codelab, link: https://lipn.univ-paris13.fr/~gabor/semeval2018task7/training-eval.txt
# Output: test data contain abstract from input 2, and train data rest
# Author: Jingyi

import codecs
import pandas as pd

path = "F:\\Data Science\\Team Project\\tp\\"
textid=[]

with codecs.open(path+'1.1practise.txt', encoding='utf-8') as f:
    for line in f.readlines():
        textid.append(line.split('\r')[0].split(' ')[1])

data = pd.read_csv(path+'ModelInput.csv', encoding='gbk')
data=data[data.label_index != -1]
data = data.reset_index(drop=True)
testData =[]
trainData =[]
for i in range(data.shape[0]):
    for j in range(len(textid)):
        record=str(data.Entity1ID[i])
        #print(record)
        #print(record.find(textid[j]))
        if record.find(textid[j]) != -1:
            testData.append(data.iloc[i,:])
            break
        elif j==49 and record.find(textid[j]) == -1:
            trainData.append(data.iloc[i,:])

pd.DataFrame(testData).to_csv(path+'task1.1testPractise.csv', encoding='gbk',index=False)
pd.DataFrame(trainData).to_csv(path+'task1.1trainPractise.csv', encoding='gbk',index=False)


print('finish')