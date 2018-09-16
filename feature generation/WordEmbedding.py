
# coding: utf-8

# In[21]:


import  xml.dom.minidom
import csv
from pandas import DataFrame
import pandas as pd
from xml.etree import ElementTree as ET
from xml.dom.minidom import parse
import string
import nltk
import copy
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


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
    t=abstract[i].replace("<abstract> ","")        #replace the "."in the entity id with ":"
    t=t.replace(" </abstract>","")
    tt=TextId[i]
    tf=tt+"."
    ttt=tt+":"
    t=t.replace(tf,ttt)
    tm=t.split(".")
    abstract[i]=tm
    


# In[7]:


def secondSpilt(tests):                            #function for second step split 
    for i in range(len(tests)):                    #split by "<entity id="
        tests[i]=tests[i].split("<entity id=")
        
    return tests
    


# In[8]:


def finalSplit(tts):                            #function for third step split
    tfs=[]
    for i in range(len(tts)):                   #split by ">"
        if tts[i].find(">")<0:
            tfs.append(tts[i].split(" "))
        if tts[i].find(">")>0:
            tfs.append(tts[i])
    
    return tfs
 
        
   
    

        


# In[9]:


def FinalSentences(Final):                     #function for form the final sentence
    FinalSentence=[]
    for i in range(len(Final)):
        for j in range(len(Final[i])):
            if type(Final[i][j])==list:
                FinalSentence.extend(Final[i][j])
               
            else:
                FinalSentence.append(Final[i][j])
                
                
    return FinalSentence
           


# In[10]:


AbstractWise=[]


# In[11]:


for i in range(len(abstract)):                   #process for split the xml file to sentence wise
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


# In[12]:


for i in range(len(AbstractWise)):             #process for remove empty space
    for j in range(len(AbstractWise[i])):
        while "" in AbstractWise[i][j]:
            AbstractWise[i][j].remove("")
        
    


# In[13]:


TagforWord=copy.deepcopy(AbstractWise)        #prepare a list for store the entity id 
for i in range(len(TagforWord)):
    for j in range(len(TagforWord[i])):
        for h in range(len(TagforWord[i][j])):
            TagforWord[i][j][h]="context"



# In[14]:


def Tokenization(AbstractWise,TagforWord):
    for i in range(len(AbstractWise)):              #process for tolkenaise the entities 
        for j in range(len(AbstractWise[i])):       #and store the id in TagforWord list
            for h in range(len(AbstractWise[i][j])):
                Sentence=AbstractWise[i][j]
                TagforWords=TagforWord[i][j]
                Word=Sentence[h]
                if (Word.find(">")>0):
                    n=Word.find(">")
                    Word1=Word[n+1:]
                    Word2=Word[:n]
                    Word1=Word1.split(" ")
                    Sentence[h]=Word1[0]
                    TagforWords[h]=Word2
                
                    if(len(Word)>1):
                        for m in range(len(Word1)-1):
                            Sentence.insert(h+m+1,Word1[m+1])
                            TagforWords.insert(h+m+1,Word2)
                        
                    AbstractWise[i][j]=Sentence
                    TagforWord[i][j]=TagforWords
    Result=[]
    Result.append(AbstractWise)
    Result.append(TagforWord)
    return Result


# In[15]:


Result=Tokenization(AbstractWise,TagforWord)
Result=Tokenization(Result[0],Result[1])
Result=Tokenization(Result[0],Result[1])


# In[16]:


sentences=[]                                      #store all the text in sentence wise
AbstractWise=Result[0]
for i in range(len(AbstractWise)):
    for j in range(len(AbstractWise[i])):
        sentences.append(AbstractWise[i][j])
    



# In[48]:


model = Word2Vec(sentences, min_count=1)          #train the word embedding model based on the abstract data
wordvector=[]


# In[52]:


for i in range(len(sentences)):                  #store the word vector in sentence wise
    sentencevector=[]
    for j in range(len(sentences[i])):
        onewordvector=[]
        vector=model.wv[sentences[i][j]]
        for h in range(100):
            onewordvector.append(vector[h])
        sentencevector.append(onewordvector)
    wordvector.append(sentencevector)
        
    

