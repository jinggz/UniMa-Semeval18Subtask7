
# coding: utf-8

# In[1]:


import  xml.dom.minidom
import csv
from pandas import DataFrame
import pandas as pd
from xml.etree import ElementTree as ET
from xml.dom.minidom import parse
import string
import nltk


# In[2]:


dom = xml.dom.minidom.parse('1.1.text.xml')        #read xml file
root = dom.documentElement                          #get all the element from xml file


# In[3]:


dom = parse("1.1.text.xml")
abstract=[]


# In[4]:


for node in dom.getElementsByTagName('abstract'):
    f=node.toxml()
    abstract.append(f)
    


# In[5]:


dom = xml.dom.minidom.parse('1.1.text.xml')        #read xml file
root = dom.documentElement  
itemlist = root.getElementsByTagName('text')  
TextId={}                                       #store the textID in a Dictionary
for i in range(350):
    item=itemlist[i]
    TextId[i]=item.getAttribute("id")


# In[6]:


for i in range(len(TextId)):                       #split the abstract into sentence wise 
    t=abstract[i].replace("<abstract> ","")
    t=t.replace(" </abstract>","")
    tt=TextId[i]
    tf=tt+"."
    ttt=tt+":"
    t=t.replace(tf,ttt)
    tm=t.split(".")
    abstract[i]=tm
    


# In[7]:


def secondSpilt(tests):
    for i in range(len(tests)):
        tests[i]=tests[i].split("<entity id=")
        
    return tests
    


# In[8]:


def finalSplit(tts):
    tfs=[]
    for i in range(len(tts)):
        if tts[i].find(">")<0:
            tfs.append(tts[i].split(" "))
        if tts[i].find(">")>0:
            tfs.append(tts[i])
    
    return tfs
 
        
   
    

        


# In[9]:


def FinalSentences(Final):
    FinalSentence=[]
    for i in range(len(Final)):
        for j in range(len(Final[i])):
            if type(Final[i][j])==list:
                FinalSentence.extend(Final[i][j])
               
            else:
                FinalSentence.append(Final[i][j])
                
                
    return FinalSentence
           


# In[10]:


#for i in range(len(Final)):
#    for j in range(len(Final[i])):
#            FinalSentence.extend(Final[i][j])
 #           FinalSentence.remove("")
 #       else:
 #           FinalSentence.append(Final[i][j])
 #           FinalSentence.remove("")
#           


# In[11]:


#OneAbstract=abstract[0]
#OneSentence=OneAbstract[0]
#OneSentence=OneSentence.split("</entity>")
#Seconds=secondSpilt(OneSentence)
#Final=[]
#for i in range(len(Seconds)):
#    Final.append(finalSplit(Seconds[i]))
#FinalSentence=FinalSentences(Final)


# In[12]:


AbstractWise=[]


# In[13]:


for i in range(len(abstract)):
    oneabstract=abstract[i]
    FinalResult=[]
    for j in range(len(oneabstract)):
        onesentence=oneabstract[j]
        onesentence=onesentence.split("</entity>")
        Seconds=secondSpilt(onesentence)
        Final=[]
        for h in range(len(Seconds)):
            Final.append(finalSplit(Seconds[h]))
        FinalSentence=FinalSentences(Final)
        FinalResult.append(FinalSentence)
    AbstractWise.append(FinalResult)


# In[14]:


df=pd.read_csv("abstract1.csv",encoding="gbk")
Tag=[]                                     ##creat a list for result of pos tagging
                             ##creat a list for dictionary of each abstract's  tag
for i in range(350):                      #do pos tagging
    text=nltk.word_tokenize(df.abstract[i])
    Tag.append(nltk.pos_tag(text))


# In[15]:


for i in range(len(AbstractWise)):
    for j in range(len(AbstractWise[i])):
        while "" in AbstractWise[i][j]:
            AbstractWise[i][j].remove("")
        
    


# In[18]:


#TagResult=[]
#for i in range(len(AbstractWise)):
 #   TagAbstract=[]    
  #  TagD=Tag[TID]
   # length=len(TagD)
    #TagDict={} 
    #for m in range(length): 
     #   TagDict[TagD[m][0]]=TagD[m][1]
    #for j in range(len（AbstractWise[i]):
     #   TagSentence=[]
      #  Sentences=Abstract[i][j]
       # for h in range(len(Sentences)):
        #    if (Sentences[h].find(">")>0):
                
        
        

