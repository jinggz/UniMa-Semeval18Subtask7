
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
import copy
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import tensorflow
import  random
from itertools import combinations


# In[2]:


with open('1.1.test.relations.txt', 'r') as f:  
    dataRelation1 = f.readlines()
with open('1.2.test.relations.txt', 'r') as f:  
    dataRelation2 = f.readlines()


# In[3]:


Relation1=[] 
Relation2=[]
for line in dataRelation1:  
    m=line.replace('\n',"")
    Relation1.append(m)
for line in dataRelation2:  
    m=line.replace('\n',"")
    Relation2.append(m)


# In[4]:


RDetail1=[]
Reverse1=[]
Entity11=[]
Entity12=[]

for relation in Relation1:
    t=relation.split(",")
    if(len(t)>2):
        t1=t[0].split("(")
        RDetail1.append(t1[0])
        Entity11.append(t1[1])
        Entity12.append(t[1].replace(")",""))        
        Reverse1.append(1)
    else:
        t1=t[0].split("(")
        RDetail1.append(t1[0])
        Entity11.append(t1[1])
        Entity12.append(t[1].replace(")","")) 
        Reverse1.append(0)

RDetail2=[]
Reverse2=[]
Entity21=[]
Entity22=[]
for relation in Relation2:
    t=relation.split(",")
    if(len(t)>2):
        t1=t[0].split("(")
        RDetail2.append(t1[0])
        Entity21.append(t1[1])
        Entity22.append(t[1].replace(")","")) 
        Reverse2.append(1)
    else:
        t1=t[0].split("(")
        RDetail2.append(t1[0])
        Entity21.append(t1[1])
        Entity22.append(t[1].replace(")","")) 
       
        Reverse2.append(0)
        
        
        


# In[204]:


dom = xml.dom.minidom.parse('1.2.text.xml')        #read xml file
root = dom.documentElement                          #get all the element from xml file


# In[205]:


dom = parse("1.2.text.xml")
abstract=[]
title=[]


# In[206]:


for node in dom.getElementsByTagName('abstract'):
    f=node.toxml()
    abstract.append(f)


# In[207]:


for node in dom.getElementsByTagName('title'):
    f=node.toxml()
    title.append(f)


# In[208]:



itemlist = root.getElementsByTagName('text')  
TextId={}                                       #store the textID in a Dictionary
for i in range(150):
  item=itemlist[i]
  TextId[i]=item.getAttribute("id")


# In[209]:


for i in range(len(TextId)):                       #split the abstract into sentence wise 
    t=abstract[i].replace("<abstract> ","")        #replace the "."in the entity id with ":"
    t=t.replace(" </abstract>","")
    t=t.replace("\n</abstract>","")
    t=t.replace("<abstract>\n","")
    t=t.replace("<abstract>","")
    tt=TextId[i]
    tf=tt+"."
    ttt=tt+":"
    t=t.replace(tf,ttt)
    tm=t.split(". ")
    abstract[i]=tm   


# In[210]:


for i in range(len(TextId)):                       #split the abstract into sentence wise 
    t=title[i].replace("<title> ","")        #replace the "."in the entity id with ":"
    t=t.replace(" </title>","")
    t=t.replace("\n</title>","")
    t=t.replace("<title>\n","")
    t=t.replace("<title>","")
    tt=TextId[i]
    tf=tt+"."
    ttt=tt+":"
    t=t.replace(tf,ttt)
    tm=t.split(". ")
    title[i]=tm 


# In[211]:


def secondSpilt(tests):                            #function for second step split 
    for i in range(len(tests)):                    #split by "<entity id="
        tests[i]=tests[i].split("<entity id=")
        
    return tests


# In[212]:


def finalSplit(tts):                            #function for third step split
    tfs=[]
    for i in range(len(tts)):                   #split by ">"
        if tts[i].find(">")<0:
            tfs.append(tts[i].split(" "))
        if tts[i].find(">")>0:
            tfs.append(tts[i])
    
    return tfs        


# In[213]:


def FinalSentences(Final):                     #function for form the final sentence
    FinalSentence=[]
    for i in range(len(Final)):
        for j in range(len(Final[i])):
            if type(Final[i][j])==list:
                FinalSentence.extend(Final[i][j])             
            else:
                FinalSentence.append(Final[i][j])                
    return FinalSentence   


# In[214]:


AbstractWise=[]
titleWise=[]


# In[215]:


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


# In[216]:


for i in range(len(title)):                   #process for split the xml file to sentence wise
    oneabstract=title[i]
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
    titleWise.append(FinalResult)


# In[217]:


for i in range(len(AbstractWise)):             #process for remove empty space
    for j in range(len(AbstractWise[i])):
        while "" in AbstractWise[i][j]:
            AbstractWise[i][j].remove("")


# In[218]:


for i in range(len(titleWise)):             #process for remove empty space
    for j in range(len(titleWise[i])):
        while "" in titleWise[i][j]:
            titleWise[i][j].remove("")


# In[219]:


TagforWord=copy.deepcopy(AbstractWise)                     #prepare a list for store the entity id 
WordEmbedding=copy.deepcopy(AbstractWise)
for i in range(len(TagforWord)):
    for j in range(len(TagforWord[i])):
        for h in range(len(TagforWord[i][j])):
            TagforWord[i][j][h]="context"


# In[220]:


TagforWordT=copy.deepcopy(titleWise)                     #prepare a list for store the entity id 
WordEmbeddingT=copy.deepcopy(titleWise)
for i in range(len(TagforWordT)):
    for j in range(len(TagforWordT[i])):
        for h in range(len(TagforWordT[i][j])):
            TagforWordT[i][j][h]="context"


# In[221]:


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


# In[222]:


Result=Tokenization(AbstractWise,TagforWord)
Result=Tokenization(Result[0],Result[1])
Result=Tokenization(Result[0],Result[1])


# In[223]:


Result2=Tokenization(titleWise,TagforWordT)
Result2=Tokenization(Result2[0],Result2[1])
Result2=Tokenization(Result2[0],Result2[1])


# In[224]:


sentences=[]                                      #store all the text in sentence wise
AbstractWise=Result[0]
for i in range(len(AbstractWise)):
    for j in range(len(AbstractWise[i])):
        sentences.append(AbstractWise[i][j])


# In[225]:


titleWise=Result2[0]
for i in range(len(titleWise)):
    for j in range(len(titleWise[i])):
        sentences.append(titleWise[i][j])


# In[226]:


for i in range(150):                             
    for j in range(len(WordEmbedding[i])):
        for h in range(len(WordEmbedding[i][j])):
                Sentence=WordEmbedding[i][j]               
                Word=Sentence[h]
                if (Word.find(">")>0):
                    WordEmbedding[i][j][h]="<entity id="+Word+"</entity>"


# In[227]:


for i in range(150):                             
    for j in range(len(WordEmbeddingT[i])):
        for h in range(len(WordEmbeddingT[i][j])):
                Sentence=WordEmbeddingT[i][j]               
                Word=Sentence[h]
                if (Word.find(">")>0):
                    WordEmbeddingT[i][j][h]="<entity id="+Word+"</entity>"


# In[228]:


Sentences=[]                                     #store all the text in sentence wise
for i in range(len(WordEmbedding)):
    for j in range(len(WordEmbedding[i])):
        Sentences.append(WordEmbedding[i][j])


# In[229]:


for i in range(len(WordEmbeddingT)):
    for j in range(len(WordEmbeddingT[i])):
        Sentences.append(WordEmbeddingT[i][j])


# In[230]:


len(Sentences)


# In[231]:


SelectedSentences=[]                          #select sentence contain at least two entities
for i in range(len(Sentences)):
    count=0
    for j in range(len(Sentences[i])):
        if (Sentences[i][j].find(">")>1):
            count=count+1
    if(count>1):
        SelectedSentences.append(Sentences[i])
        
        


# In[232]:


Instances=[]                               #generate sentences based on every pair of entities
for i in range(len(SelectedSentences)):
    EntityPosition=[]
    Combination=[]
    if(len(SelectedSentences[i])<3):
        Instances.append(SelectedSentences[i])
    else:
        
        for j in range(len(SelectedSentences[i])):
            if (SelectedSentences[i][j].find("<")>-1):
                EntityPosition.append(j)
        for h in combinations(EntityPosition,2):
            Combination.append(h)
        for r in range(len(Combination)):
            head=int(Combination[r][0])
            end=int(Combination[r][1])
            NewSentence=[SelectedSentences[i][head]]
            for f in range(end-head):
                if(SelectedSentences[i][head+f+1].find("<")==-1):
                     NewSentence.append(SelectedSentences[i][head+f+1])
            NewSentence.append(SelectedSentences[i][end])
               
            Instances.append(NewSentence)
        
    
    
        
    


# In[233]:


len(Instances)


# In[234]:


In2=copy.deepcopy(Instances)                                      #clean the tag of entity
for i in range(len(Instances)):
    for j in range(len(Instances[i])):
        Instances[i][j]=Instances[i][j].replace(":",".")
        Instances[i][j]=Instances[i][j].replace("<entity id=","")
        Instances[i][j]=Instances[i][j].replace("</entity>","")
        Instances[i][j]=Instances[i][j].replace(">"," ")
        
        


# In[235]:


for i in range(len(Instances)):                                 #split entity
    for j in range(len(Instances[i])):
        Instances[i][j]=Instances[i][j].split(" ")


# In[236]:


Distance=[]
Distance2=[]


# In[237]:


Epairs=[]                                                       #store entity id and distance
for i in range(len(Instances)):
    ps=[]
    Distancet=[]
    Distancett=[]
    for j in range(len(Instances[i][0])-1):
        Distancet.append(0)
    l=len(Instances[i])
    for h in range(l-1+len(Instances[i][l-1])-2):
        Distancet.append(h+1)
    e1=Instances[i][0][0]
    ps.append(e1)
    for m in range(len(Instances[i])-2+len(Instances[i][0])-1):
        tlength=len(Instances[i])+len(Instances[i][0])-3
        
        Distancett.append(m-tlength)
    for n in range(len(Instances[i][l-1])-1):
        Distancett.append(0)
        
    
    e2=Instances[i][l-1][0]
    ps.append(e2)
    Epairs.append(ps)
    Distance.append(Distancet)
    Distance2.append(Distancett)


# In[238]:


for i in range(len(Distance)):
    ss=sum(Distance[i])
    for j in range(len(Distance[i])):
        Distance[i][j]=Distance[i][j]/ss


# In[239]:


for i in range(len(Distance2)):
    ss=sum(Distance2[i])
    for j in range(len(Distance2[i])):
        Distance2[i][j]=-Distance2[i][j]/ss


# In[240]:


for i in range(len(Instances)):                             #Insert the whole text of entity back to sentences                                
    Instances[i][0]=Instances[i][0][1:]


# In[241]:


for i in range(len(Instances)):
    l=len(Instances[i])
    Instances[i][l-1]=Instances[i][l-1][1:]


# In[242]:


for i in range(len(Instances)):
    onelist=[]
    for j in range(len(Instances[i])):
        onelist.extend(Instances[i][j])
    Instances[i]=onelist


# In[243]:


for i in range(len(Instances)):
    if("," in Instances[i]):
        Instances[i].remove(",")      
        
    


# In[244]:


for i in range(len(Epairs)):
    Epairs[i][0]=Epairs[i][0][1:-1]
    Epairs[i][1]=Epairs[i][1][1:-1]


# In[245]:


LabelInput=[]
LabelDetail=[]
Ls=[]
Reverse=[]
selectedInstances=[]
Rs=[]
Es=[]

#recognize positive or negative sentences
for i in range(len(Epairs)):
    m=0
    t=0
    for j in range(len(Entity11)):
        if(Epairs[i][0]==Entity11[j] and Epairs[i][1]==Entity12[j]):
            m=m+1
            t=j
            
    if(m>0):
        LabelInput.append(1)
        LabelDetail.append(RDetail2[t])
        Ls.append(RDetail2[t])
        Reverse.append(Reverse2[t])
        Rs.append(Reverse2[t])
        selectedInstances.append(Instances[i])
        Es.append(Epairs[i])
        
    else:
        LabelInput.append(0)
        LabelDetail.append('none')
        Reverse.append('none')
        
        
        
        


# In[153]:


c={"Instances":Instances,"Pairs":Epairs}


# In[247]:


c={"Instances":Instances,"Label":LabelInput,"Reverse":Reverse,"Detail":LabelDetail,"Epairs":Epairs}


# In[248]:


c=DataFrame(c)


# In[249]:


c.to_csv("1.1.test.instances.csv")

