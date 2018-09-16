
# coding: utf-8

# In[1]:


import  xml.dom.minidom
import csv
from pandas import DataFrame
import pandas as pd
from xml.etree import ElementTree as ET
from xml.dom.minidom import parse
import string


# In[20]:


df=pd.read_csv("abstract.csv",encoding="gbk")
Abstract=[]


# In[21]:


for i in range(len(df)):
    Abstract.append(df.x[i].split("."))
Sentence=[]
for i in range(len(df)):
    for j in range(len(Abstract[i])):
        Sentence.append(Abstract[i][j].split(" "))
    
    
    


# In[23]:


Sentence[0]

