
# coding: utf-8

# In[49]:


import nltk
import pandas as pd
import csv
from pandas import DataFrame
import  xml.dom.minidom
import csv
from pandas import DataFrame
from itertools import islice
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics


# In[3]:


df=pd.read_csv("PostaggingFile.csv")


# In[10]:


Relation=df.relation


# In[14]:


RelationSet=['USAGE','RESULT','MDEL-FEATURE','PART_WHOLE','TOPIC','COMPARE']


# In[18]:


Prediction=[]
for i in range(len(Relation)):
    Prediction.append(random.choice(RelationSet)) #randomly choose a prediction lable


# In[21]:


Performance={
    "Relation" : Relation,
    "Prediction" : Prediction
}


# In[53]:


Performance=DataFrame(Performance)
Performance.to_csv("RandomlyPerformance.csv") #out put the performance file


# In[28]:


Relation= np.array(Relation)         # change dataframe to list
Relation=Relation.tolist()


# In[31]:


Accuracy=accuracy_score(Relation,Prediction)


# In[40]:


Recall=metrics.recall_score(Relation,Prediction,average='weighted')


# In[44]:


F1=metrics.f1_score(Relation,Prediction, average='weighted')

