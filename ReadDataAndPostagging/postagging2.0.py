
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import csv
from pandas import DataFrame
import  xml.dom.minidom
import csv
from pandas import DataFrame


# In[20]:


df=pd.read_csv("abstract1.csv")
Tag=[]                                     ##creat a list for result of pos tagging
                             ##creat a list for dictionary of each abstract's  tag
for i in range(350):                      #do pos tagging
    text=nltk.word_tokenize(df.x[i])
    Tag.append(nltk.pos_tag(text))
    
dom = xml.dom.minidom.parse('1.1.text.xml')        #read xml file
root = dom.documentElement  
itemlist = root.getElementsByTagName('text')  
TextId={}                                       #store the textID in a Dictionary
for i in range(350):
    item=itemlist[i]
    TextId[item.getAttribute("id")]=i
    
Entity=pd.read_csv("EntityPreprocessed.csv")
TagResult=[]


# In[ ]:


for i in range(len(Entity)):                      #find the textID of selected entity
    Find=Entity.EntityID[i].split(".")
    TID=TextId[Find[0]]    
    TagD=Tag[TID]
    length=len(TagD)
    TagDict={}
    for m in range(length):                      #get the tag dictionary of the text
        TagDict[TagD[m][0]]=TagD[m][1]
    
    content=Entity.Content[i].split(' ')
    Tagging=[]
    for j in range(len(content)):                #get the tag of this entity
        Tagging.append(TagDict[content[j]])
    
    EntityTag=','.join(Tagging)
    TagResult.append(EntityTag) 
    
    
    


# In[25]:


TagResult={"Tag": TagResult}                    #combine the entity file and tag file


# In[26]:


TagResult=DataFrame(TagResult)


# In[28]:


Entity["Tag"]=TagResult


# In[30]:


Entity.to_csv("EntityTagging.csv")

