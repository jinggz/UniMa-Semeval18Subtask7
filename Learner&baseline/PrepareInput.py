# read feature.csv and instances0701.csv,means.csv file
## prepare for SVM to detect relation
# Author: Yuan

import csv
import codecs
import pandas as pd
import numpy as np
import os
from pandas.core.frame import DataFrame

path='D:\\3nd Semester\\Semantic Relation Extraction\\MyCode\\PrepareInput\\Clean\\'

file1 = path+'Instances0120.csv'
df = pd.read_csv(file1, encoding="gbk",usecols=['Instances','LabelInput','Pairs','Reverse','detail'])

# split Entity id into 2 columns, Entity1ID, Entity2ID
pairs = pd.read_csv(file1, encoding="gbk",
                    usecols=['Pairs'])
pairs = np.array(pairs)

entity2 = []
entity1 = []
entitypair = []
#writeentity = []
#remove the puntuations eg [,'...]
for i in range(0, len(pairs)):
    id1 = pairs[i][0].split(',')[0]
    id1 = id1.strip().lstrip('[\'').rstrip('\'')
    id2 = pairs[i][0].split(',')[1]
    id2 = id2.strip().lstrip('\'').rstrip('\']').rstrip('\'')
    entity1.append(id1)
    entity2.append(id2)
    #in submission format
    entitypair.append('('+id1+','+id2+')')
entity1 = np.array(entity1)
entity2 = np.array(entity2)
entitypair = np.array(entitypair)
#insert columns into the csv file
dic1={"Entity1ID": entity1}
ids1=DataFrame(dic1)
df.insert(2,'Entity1ID',ids1)
dic2={"Entity2ID": entity2}
ids2=DataFrame(dic2)
df.insert(3,'Entity2ID',ids2)
dic3={"EntityPair": entitypair}
idspair = DataFrame(dic3)
df.insert(4,'Entity Pair',idspair)

df['Reverse'].replace(to_replace='FALSE',value=0, inplace=True)
df['Reverse'].replace(to_replace='none',value=-1, inplace=True)
df['Reverse'].replace(to_replace='TRUE',value=1, inplace=True)

#add lables including relation type
#add column, change class labels to index
df4=pd.read_csv(file1, encoding="gbk", usecols=['detail'])
df4.replace(to_replace='none', value=-1, inplace=True)
df4.replace(to_replace='USAGE', value=1, inplace=True)
df4.replace(to_replace='MODEL-FEATURE', value=2, inplace=True)
df4.replace(to_replace='PART_WHOLE', value=3, inplace=True)
df4.replace(to_replace='COMPARE', value=4, inplace=True)
df4.replace(to_replace='RESULT', value=5, inplace=True)
df4.replace(to_replace='TOPIC', value=6, inplace=True)
df4.rename(columns={'detail':'label_index'}, inplace=True)
df = pd.concat([df, df4], axis=1)

df.drop(['Pairs'], axis=1, inplace=True)

print('File Instances is ready')

#merge with feature "means"
filemeans = path+'means0120.csv'
dfmeans = pd.read_csv(filemeans, encoding="gbk",usecols=['means'])
df.insert(8,'means',dfmeans)
df.to_csv(path+'tryinstance.csv')



#join features with entity1 and entity2
file2 = path+'Features.csv'
#df2 = pd.read_csv(file2,encoding='Windows-1252')
'''features
['word' 'abstractNr' 'sentenceNr' 'positionInSentence' 'entityIdentifyer'
 'relationWith' 'relationType' 'relationReverse' 'lastLetter'
 'lastButOneLetter' 'lastButTwoLetter' 'lastButThreeLetter'
 'lastButFourLetter' 'firstLetter' 'secondLetter' 'thirdLetter'
 'fourthLetter' 'fifthLetter' 'wordsInSentence' 'lastWord' 'lastButOneWord'
 'lastButTwoWord' 'nextWord' 'nextButOneWord' 'nextButTwoWord' 'wordLen'
 'distLastEntity' 'distNextEntity' 'relativePosition' 'posTag01' 'posTag02'
 'posTag03' 'posTag04' 'posTag05' 'posTag06' 'posTag07' 'posTag08'
 'posTag09' 'posTag10' 'posTag11' 'posTag12' 'posTag13' 'posTag14'
 'posTag15' 'lastLetterInt' 'lastButOneLetterInt' 'lastButTwoLetterInt'
 'lastButThreeLetterInt' 'lastButFourLetterInt' 'firstLetterInt'
 'secondLetterInt' 'thirdLetterInt' 'fourthLetterInt' 'fifthLetterInt'
 'lastWordVector' 'lastButOneWordVector' 'lastButTwoWordVector'
 'nextWordVector' 'nextButOneWordVector' 'nextButTwoWordVector' 'lastVerb'
 'lastVerbVector' 'lastVerbDistance' 'nextVerb' 'nextVerbVector'
 'nextVerbDistance' 'hasWikiAricle']
 '''
df2 = pd.read_csv(file2,usecols=['entityIdentifyer','positionInSentence','wordsInSentence','wordLen','relativePosition',
                                 'distLastEntity','distNextEntity','posTag01','posTag02','posTag03','posTag04','posTag05','posTag06',
                                 'firstLetterInt','lastLetterInt','lastVerbDistance','nextVerbDistance','hasWikiAricle',
                                 'hasOf','hasUse','lastVerbVector','nextVerbVector'])
df2.rename(columns={'entityIdentifyer': 'Entity1ID'}, inplace=True)
#df.join(df2.set_index('Entity1ID'), on='Entity1ID')
df=df.merge(right=df2, how='left', left_on='Entity1ID', right_on='Entity1ID')
#df2.rename(columns={'Entity1ID':'Entity2ID','word_1': 'word_2','positionInSentence_1':'positionInSentence_2','wordsInSentence_1':'wordsInSentence_2','wordLen_1':'wordLen_2'}, inplace=True)
df2.rename(columns={'Entity1ID':'Entity2ID'}, inplace=True)
df=df.merge(right=df2, how='left', left_on='Entity2ID', right_on='Entity2ID')



#merge with posTag sparse matrix
fileposTag=path+'posTagBin0121.csv'
posTag = pd.read_csv(fileposTag)
posTag.drop(['att1'], axis=1, inplace=True)
df = pd.concat([df, posTag], axis=1)


def Vector2Column(vector):
    verb = np.array(vector)

    data = []
    emptyList = [np.nan] * 300
    # try = lastVerb[3][0].split('[ ')[1].split(']')[0].split()
    for i in range(0, df1.shape[0]):
        j = 0
        s = verb[i]
        # print(type(s[0]))

        if type(s[0]) == str:
            # print(s[0].split('[')[1].split(']')[0].split()[0])
            data.append(s[0].split('[')[1].split(']')[0].split())
            for j in range(0, len(data[i])):
                data[i][j] = float(data[i][j])
                # print(type(data[i][j]))
                ##print(float(s[0].split('[')[1].split(']')[0].split()[j]))
                # data[i][j]=float(s[0].split('[')[1].split(']')[0].split()[j])
                # print(data[i])
        else:
            # data[i]=str
            data.append(emptyList)
            # print('empty')

    data = np.array(data)
    data = pd.DataFrame(data)
    return data

#convert verb vector to 300 columns
lastVerbVector_x=pd.DataFrame(df['lastVerbVector_x'])
lvb_x=Vector2Column(lastVerbVector_x)
df = pd.concat([df, lvb_x], axis=1)
lastVerbVector_y=pd.DataFrame(df['lastVerbVector_y'])
lvb_y=Vector2Column(lastVerbVector_y)
df = pd.concat([df, lvb_y], axis=1)
nextVerbVector_x=pd.DataFrame(df['nextVerbVector_x'])
nvb_x=Vector2Column(lastVerbVector_x)
df = pd.concat([df, nvb_x], axis=1)
nextVerbVector_y=pd.DataFrame(df['nextVerbVector_y'])
nvb_y=Vector2Column(nextVerbVector_y)
df = pd.concat([df, nvb_y], axis=1)

df.drop(['lastVerbVector_x','lastVerbVector_y','nextVerbVector_x','nextVerbVector_y'], axis=1, inplace=True)
print(df.shape)

#output the final modelInput
file3 = path+'ModelInput0124.csv'
df.to_csv(file3,index=False)
print('Entity1 and Entity2 features are joined')
