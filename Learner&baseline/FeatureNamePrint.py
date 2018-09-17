# find corresponding feature names from feature indices
# coding: utf-8
# In[1]:


import pandas as pd
import numpy as np
import codecs


# In[2]:


path='F:\\Data Science\\Team Project\\tp\\'


# In[2]:

# reading for SVM
df2 = pd.read_csv(path+'ModelInput0124.csv',usecols=['hasUse_x','hasUse_y','hasOf_x','hasOf_y','means', 'positionInSentence_x', 'wordsInSentence_x',
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
 'posTag06_y = 17.0', 'posTag06_y = -1.0'],
                   encoding='gbk')

# In[3]:

# reading for KNN
# y = pd.read_csv(path + 'ModelInput0124.csv', encoding='gbk',
#                           usecols=['hasUse_x', 'hasUse_y', 'hasOf_x', 'hasOf_y', 'means', 'positionInSentence_x', 'wordsInSentence_x',
#  'wordLen_x', 'distLastEntity_x', 'distNextEntity_x', 'relativePosition_x',
#  'lastLetterInt_x', 'firstLetterInt_x', 'lastVerbDistance_x',
#  'nextVerbDistance_x', 'hasWikiAricle_x', 'positionInSentence_y',
#  'wordsInSentence_y', 'wordLen_y', 'distLastEntity_y', 'distNextEntity_y',
#  'relativePosition_y', 'lastLetterInt_y', 'firstLetterInt_y',
#  'lastVerbDistance_y', 'nextVerbDistance_y', 'hasWikiAricle_y',
#   'posTag01_x', 'posTag02_x', 'posTag03_x', 'posTag04_x', 'posTag05_x', 'posTag06_x',
# 'posTag01_y',	'posTag02_y',	'posTag03_y',	'posTag04_y', 'posTag05_y','posTag06_y'])

# In[5]


a=df2.columns.values.tolist()
#b=y.columns.values.tolist()

# In[7]:

# In[7]:

fs=np.loadtxt(path+'Results\\SelectedFeaturesIndices0126LinearSVC.txt',delimiter='\r\n',dtype=int)

for i in fs:
    print("'"+a[i]+"', ")
    






