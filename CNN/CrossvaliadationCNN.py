
# coding: utf-8

# In[272]:


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
from sklearn.model_selection import KFold
import numpy as np


# In[273]:


with open('1.1.relations.txt', 'r') as f:  
    dataRelation1 = f.readlines()
with open('1.2.relations.txt', 'r') as f:  
    dataRelation2 = f.readlines()


# In[274]:


Relation1=[] 
Relation2=[]
for line in dataRelation1:  
    m=line.replace('\n',"")
    Relation1.append(m)
for line in dataRelation2:  
    m=line.replace('\n',"")
    Relation2.append(m)


# In[275]:


#RDetail1=[]
Reverse1=[]
Entity11=[]
Entity12=[]

for relation in Relation1:
    t=relation.split(",")
    if(len(t)>2):
        t1=t[0].split("(")
        #RDetail1.append(t1[0])
        Entity11.append(t1[1])
        Entity12.append(t[1].replace(")",""))        
        Reverse1.append(1)
    else:
        t1=t[0].split("(")
        #RDetail1.append(t1[0])
        Entity11.append(t1[1])
        Entity12.append(t[1].replace(")","")) 
        Reverse1.append(0)

#RDetail2=[]
Reverse2=[]
Entity21=[]
Entity22=[]
for relation in Relation2:
    t=relation.split(",")
    if(len(t)>2):
        t1=t[0].split("(")
        #RDetail2.append(t1[0])
        Entity21.append(t1[1])
        Entity22.append(t[1].replace(")","")) 
        Reverse2.append(1)
    else:
        t1=t[0].split("(")
        #RDetail2.append(t1[0])
        Entity21.append(t1[1])
        Entity22.append(t[1].replace(")","")) 
        
        Reverse2.append(0)
        
        
        


# In[207]:


dom = xml.dom.minidom.parse('1.1.text.xml')        #read xml file
root = dom.documentElement                          #get all the element from xml file


# In[208]:


dom = parse("1.1.text.xml")
abstract=[]


# In[209]:


for node in dom.getElementsByTagName('abstract'):
    f=node.toxml()
    abstract.append(f)


# In[210]:



itemlist = root.getElementsByTagName('text')  
TextId={}                                       #store the textID in a Dictionary
for i in range(350):
  item=itemlist[i]
  TextId[i]=item.getAttribute("id")


# In[211]:


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


# In[212]:


def secondSpilt(tests):                            #function for second step split 
    for i in range(len(tests)):                    #split by "<entity id="
        tests[i]=tests[i].split("<entity id=")
        
    return tests


# In[213]:


def finalSplit(tts):                            #function for third step split
    tfs=[]
    for i in range(len(tts)):                   #split by ">"
        if tts[i].find(">")<0:
            tfs.append(tts[i].split(" "))
        if tts[i].find(">")>0:
            tfs.append(tts[i])
    
    return tfs        


# In[214]:


def FinalSentences(Final):                     #function for form the final sentence
    FinalSentence=[]
    for i in range(len(Final)):
        for j in range(len(Final[i])):
            if type(Final[i][j])==list:
                FinalSentence.extend(Final[i][j])             
            else:
                FinalSentence.append(Final[i][j])                
    return FinalSentence   


# In[215]:


AbstractWise=[]


# In[216]:


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


# In[217]:


for i in range(len(AbstractWise)):             #process for remove empty space
    for j in range(len(AbstractWise[i])):
        while "" in AbstractWise[i][j]:
            AbstractWise[i][j].remove("")


# In[218]:


TagforWord=copy.deepcopy(AbstractWise)                     #prepare a list for store the entity id 
WordEmbedding=copy.deepcopy(AbstractWise)
for i in range(len(TagforWord)):
    for j in range(len(TagforWord[i])):
        for h in range(len(TagforWord[i][j])):
            TagforWord[i][j][h]="context"


# In[219]:


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


# In[220]:


Result=Tokenization(AbstractWise,TagforWord)
Result=Tokenization(Result[0],Result[1])
Result=Tokenization(Result[0],Result[1])


# In[221]:


sentences=[]                                      #store all the text in sentence wise
AbstractWise=Result[0]
for i in range(len(AbstractWise)):
    for j in range(len(AbstractWise[i])):
        sentences.append(AbstractWise[i][j])


# In[222]:


for i in range(350):                             
    for j in range(len(WordEmbedding[i])):
        for h in range(len(WordEmbedding[i][j])):
                Sentence=WordEmbedding[i][j]               
                Word=Sentence[h]
                if (Word.find(">")>0):
                    WordEmbedding[i][j][h]="<entity id="+Word+"</entity>"


# In[223]:


Sentences=[]                                     #store all the text in sentence wise
for i in range(len(WordEmbedding)):
    for j in range(len(WordEmbedding[i])):
        Sentences.append(WordEmbedding[i][j])


# In[224]:


SelectedSentences=[]                          #select sentence contain at least two entities
for i in range(len(Sentences)):
    count=0
    for j in range(len(Sentences[i])):
        if (Sentences[i][j].find(">")>1):
            count=count+1
    if(count>1):
        SelectedSentences.append(Sentences[i])
        
        


# In[225]:


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
        
    
    
        
    


# In[226]:


In2=copy.deepcopy(Instances)                                      #clean the tag of entity
for i in range(len(Instances)):
    for j in range(len(Instances[i])):
        Instances[i][j]=Instances[i][j].replace(":",".")
        Instances[i][j]=Instances[i][j].replace("<entity id=","")
        Instances[i][j]=Instances[i][j].replace("</entity>","")
        Instances[i][j]=Instances[i][j].replace(">"," ")
        
        


# In[227]:


for i in range(len(Instances)):                                 #split entity
    for j in range(len(Instances[i])):
        Instances[i][j]=Instances[i][j].split(" ")


# In[228]:


Distance=[]
Distance2=[]


# In[229]:


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


# In[230]:


for i in range(len(Distance)):
    ss=sum(Distance[i])
    for j in range(len(Distance[i])):
        Distance[i][j]=Distance[i][j]/ss


# In[231]:


for i in range(len(Distance2)):
    ss=sum(Distance2[i])
    for j in range(len(Distance2[i])):
        Distance2[i][j]=-Distance2[i][j]/ss


# In[232]:


for i in range(len(Instances)):                             #Insert the whole text of entity back to sentences                                
    Instances[i][0]=Instances[i][0][1:]


# In[233]:


for i in range(len(Instances)):
    l=len(Instances[i])
    Instances[i][l-1]=Instances[i][l-1][1:]


# In[234]:


for i in range(len(Instances)):
    onelist=[]
    for j in range(len(Instances[i])):
        onelist.extend(Instances[i][j])
    Instances[i]=onelist


# In[235]:


for i in range(len(Instances)):
    if("," in Instances[i]):
        Instances[i].remove(",")      
        
    


# In[236]:


df=pd.read_csv("PNInstances.csv",encoding="gbk")
PN=df.Instances
Neww=[]
for i in range(len(PN)):
    New=[]
    m=PN[i]
    m=m[1:-1]
    m=m.split(",")
    for j in range(len(m)):
        It=m[j][1:-1]
        It=It.replace("""'""","")
        New.append(It)
    if(len(New)<=90):        
        Neww.append(New)


# In[237]:


for i in range(len(Neww)):
    Instances.append(Neww[i])


# In[238]:


filename = 'acl_vectors_glove_300d.txt.word2vec'           #load the pre-trained model from Glove
model = KeyedVectors.load_word2vec_format(filename, binary=False)
glove_input_file = 'acl_vectors_glove_300d.txt'
word2vec_output_file = 'acl_vectors_glove_300d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)


# In[239]:


wordvector=[]
for i in range(len(Instances)):                  #store the word vector in sentence wise
    sentencevector=[]
    for j in range(len(Instances[i])):
        onewordvector=[]
        vector=[]
        if(str.lower(Instances[i][j]) in model.vocab):   # if word is existing in the model then get the vector 
            vector=model.wv[str.lower(Instances[i][j])]
        else:                                            #else randomly initial vector
            for x in range(300):
                vector.append(random.uniform(-1,1))            
        for h in range(300):
            onewordvector.append(vector[h])
        sentencevector.append(onewordvector)
    wordvector.append(sentencevector)


# In[240]:


paddingvector=[]                           #creat padding vector
for i in range(300):
    paddingvector.append(0)


# In[241]:


for i in range(len(wordvector)):          #extend vector to max length
    if(len(wordvector[i])<90):
        l=len(wordvector[i])
        more=90-l
        for j in range(more):
            wordvector[i].append(paddingvector)


# In[242]:


for i in range(len(Epairs)):
    Epairs[i][0]=Epairs[i][0][1:-1]
    Epairs[i][1]=Epairs[i][1][1:-1]


# In[243]:


LabelInput=[]
LabelDetail=[]
Reverse=[]
selectedInstances=[]

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
       
        Reverse.append(Reverse2[t])
        selectedInstances.append(Instances[i])
        
    else:
        LabelInput.append(0)
        LabelDetail.append('none')
        Reverse.append('none')
for i in range(len(Neww)):
    LabelInput.append(1)
        
        
        
        


# In[43]:


for i in range(len(Neww)):
    Epairs.append(["N","N"])


# In[69]:


Label=[]
for m in LabelInput:
    if(m==1):
        Label.append([1,0])
    else:
        Label.append([0,1])


# In[74]:


kf = KFold(n_splits=5)  # perpare data for cross validation in ten fold 
trainD=[]
testD=[]


# In[75]:


for train,test in kf.split(wordvector):
    trainD.append(train)
    testD.append(test)


# In[76]:


import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# In[97]:


model = Sequential()


# In[98]:


input_shape = (90, 300) 


# In[99]:


embedding_dim = 300            #parameters of the cnn model         
filter_sizes = (3,10)
num_filters = 64
dropout_prob = (0.5, 0.8)
hidden_dims = 20
batch_size = 64
num_epochs =25


# In[100]:


model_input = Input(shape=input_shape)                            # the convolution layer and pooling layer
z = model_input
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]


# In[101]:


import keras.backend as K                                 #concatenate layer and do the classification
from keras import metrics
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])



# In[105]:


Evaluation=[0,0,0]                        #Evaluation part in crossvaliadation
EP=[]       
for i in range(5): #10 fold crossvaliadation                
    test_label=[]
    c=0  #true number how many instances of each class
    p=0  #prediction of each class 
    tp=0 # correct prediction number of each class  
     
    
    for j in range(len(testD[i])):
        test_label.append(Label[testD[i][j]])
    for q in range(len(test_label)):
        c=c+test_label[q][0]
    x=[]
    for t in range(len(trainD[i])):
        x.append(wordvector[trainD[i][t]])
        if(Label[trainD[i][t]]==1):
                  x.append(wordvector[trainD[i][t]])
                  x.append(wordvector[trainD[i][t]])                  
    y=[]
    for u in range(len(trainD[i])):
        y.append(Label[trainD[i][u]])
        if(Label[trainD[i][u]]==1):
                  y.append(Label[trainD[i][u]])
                  y.append(Label[trainD[i][u]])
    
    tx=[] 
    for w in range(len(testD[i])):
        tx.append(wordvector[testD[i][w]])
    ty=[] 
    for e in range(len(testD[i])):
        ty.append(Label[testD[i][e]])
    pp=[]
    for v in range(len(testD[i])):
        pp.append(Epairs[testD[i][v]])
    
    x1=np.array(x)
    y1=np.array(y)
    tx1=np.array(tx)
    ty1=np.array(ty)
    model = Sequential()
    model_input = Input(shape=input_shape)                            # the convolution layer and pooling layer
    z = model_input
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(2, activation="sigmoid")(z)

    model = Model(model_input, model_output)
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])

    model.fit(x1,y1, batch_size=batch_size, epochs=num_epochs, # fit model
          validation_data=(tx1,ty1), verbose=2)
    result=model.predict(tx1, batch_size=32, verbose=0)
    pi=[]
    for i in range(len(result)):
        mm=result[i].tolist()
        if(mm[0]>mm[1]):
            
            p=p+1
            pi.append(pp[i])
            if(test_label[i][0]==1):
                tp=tp+1 
    EP.append(pi)
            
         
        
                             #calculate the precision recall and F1 score  f1=2*p*r/(p+r)
    if(p==0):
        p=1
    if((tp/p+tp/c)!=0):            
        Evaluation[0]=tp/p+Evaluation[0]
        Evaluation[1]=tp/c+Evaluation[1]
        Evaluation[2]=2*tp/p*tp/c/(tp/p+tp/c)+ Evaluation[2]
    else:
        Evaluation[0]=tp/p+Evaluation[0]
        Evaluation[1]=tp/c+Evaluation[1]
        Evaluation[2]=2*tp/p*tp/c/(0.00000001)+Evaluation[2]
   


# In[124]:


from keras.models import model_from_json
import numpy
import os
model_json = model.to_json()
with open("modelCNNfinal.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelCNNfinal.h5")


# In[207]:


json_file = open('modelCNNfinal.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelCNNfinal.h5")

