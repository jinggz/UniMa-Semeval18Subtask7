
# coding: utf-8

# In[5]:


import nltk
import pandas as pd
import csv
from pandas import DataFrame
import  xml.dom.minidom
import csv
from pandas import DataFrame


# In[6]:


df=pd.read_csv("abstract1.csv")


# In[7]:


Tag=[]                                     ##creat a list for result of pos tagging
                             ##creat a list for dictionary of each abstract's  tag
for i in range(350):                      #do pos tagging
    text=nltk.word_tokenize(df.x[i])
    Tag.append(nltk.pos_tag(text))


# In[8]:


# for j in range(350):                     ##store the result to dictionary TagDictionary
   # TagDict={}                           #TagDict is a templete dictionary store tagg for each abstract
    #length=len(Tag[j])
   # for i in range(length):                #add TagDict to final dictionary TagDictionary
       # TagDict[Tag[j][i][0]]=Tag[j][i][1]
        #TagDictionary.append(TagDict)
   


# In[9]:


dom = xml.dom.minidom.parse('1.1.text.xml')        #read xml file
root = dom.documentElement              


# In[10]:


itemlist = root.getElementsByTagName('text')      #get the elments list of text


# In[11]:


TextId={}                                       #store the textID in a Dictionary
for i in range(350):
    item=itemlist[i]
    TextId[item.getAttribute("id")]=i


# In[241]:


relation=pd.read_csv("Rt.csv")


# In[242]:


for h in range(1228):                                  #find the text where the selected entity belongs to
    Find=relation.entity1[h].split(".")
    TID=TextId[Find[0]]
    TagD=Tag[TID]
    length=len(TagD)
    TagDict={} 
    for m in range(length):                           #based on the word of entity, find the tag
        TagDict[TagD[m][0]]=TagD[m][1]
        
    word1=relation.word1[h].split(' ')
    word2=relation.word2[h].split(' ')
    m1=len(word1)
    m2=len(word2)
    word1=word1[m1-1]
    word2=word2[m2-1]
    relation.entity1[h]=TagDict[word1]
    relation.entity2[h]=TagDict[word2]
    
    
    


# In[243]:


relation.to_csv("PostaggingFile.csv")

