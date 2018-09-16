
# coding: utf-8

# In[267]:


import  xml.dom.minidom
import csv
from pandas import DataFrame
import pandas as pd
from xml.etree import ElementTree as ET
from xml.dom.minidom import parse
import string


# In[228]:


dom = xml.dom.minidom.parse('1.1.text.xml')        #read xml file
root = dom.documentElement                          #get all the element from xml file


# In[229]:


dom = parse("1.1.text.xml")
abstract=[]


# In[230]:


for node in dom.getElementsByTagName('abstract'):
    f=node.toxml()
    abstract.append(f)
    


# In[231]:


dom = xml.dom.minidom.parse('1.1.text.xml')        #read xml file
root = dom.documentElement  
itemlist = root.getElementsByTagName('text')  
TextId={}                                       #store the textID in a Dictionary
for i in range(350):
    item=itemlist[i]
    TextId[i]=item.getAttribute("id")


# In[233]:


for i in range(len(TextId)):                       #split the abstract into sentence wise 
    t=abstract[i].replace("<abstract> ","")
    t=t.replace(" </abstract>","")
    tt=TextId[i]
    tf=tt+"."
    ttt=tt+":"
    t=t.replace(tf,ttt)
    tm=t.split(".")
    abstract[i]=tm
    


# In[ ]:


print(abstract)

