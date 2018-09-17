
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


dom = xml.dom.minidom.parse('1.1.text.xml')        #read xml file
root = dom.documentElement                          #get all the element from xml file


# In[3]:


dom = parse("1.1.text.xml")
abstract=[]


# In[4]:


for node in dom.getElementsByTagName('abstract'):
    f=node.toxml()
    abstract.append(f)


# In[6]:



itemlist = root.getElementsByTagName('text')  
TextId={}                                       #store the textID in a Dictionary
for i in range(350):
  item=itemlist[i]
  TextId[i]=item.getAttribute("id")


# In[7]:


for i in range(len(TextId)):                       #split the abstract into sentence wise 
    t=abstract[i].replace("<abstract> ","")        #replace the "."in the entity id with ":"
    t=t.replace(" </abstract>","")
    t=t.replace("\n</abstract>","")
    t=t.replace("<abstract>\n","")
    tt=TextId[i]
    tf=tt+"."
    ttt=tt+":"
    t=t.replace(tf,ttt)
    tm=t.split(".")
    abstract[i]=tm   


# In[9]:


def secondSpilt(tests):                            #function for second step split 
    for i in range(len(tests)):                    #split by "<entity id="
        tests[i]=tests[i].split("<entity id=")
        
    return tests


# In[10]:


def finalSplit(tts):                            #function for third step split
    tfs=[]
    for i in range(len(tts)):                   #split by ">"
        if tts[i].find(">")<0:
            tfs.append(tts[i].split(" "))
        if tts[i].find(">")>0:
            tfs.append(tts[i])
    
    return tfs        


# In[11]:


def FinalSentences(Final):                     #function for form the final sentence
    FinalSentence=[]
    for i in range(len(Final)):
        for j in range(len(Final[i])):
            if type(Final[i][j])==list:
                FinalSentence.extend(Final[i][j])             
            else:
                FinalSentence.append(Final[i][j])                
    return FinalSentence   


# In[12]:


AbstractWise=[]


# In[13]:


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


# In[14]:


for i in range(len(AbstractWise)):             #process for remove empty space
    for j in range(len(AbstractWise[i])):
        while "" in AbstractWise[i][j]:
            AbstractWise[i][j].remove("")


# In[15]:


TagforWord=copy.deepcopy(AbstractWise)                     #prepare a list for store the entity id 
WordEmbedding=copy.deepcopy(AbstractWise)
for i in range(len(TagforWord)):
    for j in range(len(TagforWord[i])):
        for h in range(len(TagforWord[i][j])):
            TagforWord[i][j][h]="context"


# In[16]:


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


# In[17]:


Result=Tokenization(AbstractWise,TagforWord)
Result=Tokenization(Result[0],Result[1])
Result=Tokenization(Result[0],Result[1])


# In[18]:


sentences=[]                                      #store all the text in sentence wise
AbstractWise=Result[0]
for i in range(len(AbstractWise)):
    for j in range(len(AbstractWise[i])):
        sentences.append(AbstractWise[i][j])


# In[19]:


for i in range(350):                             
    for j in range(len(WordEmbedding[i])):
        for h in range(len(WordEmbedding[i][j])):
                Sentence=WordEmbedding[i][j]               
                Word=Sentence[h]
                if (Word.find(">")>0):
                    WordEmbedding[i][j][h]="<entity id="+Word+"</entity>"


# In[20]:


Sentences=[]                                     #store all the text in sentence wise
for i in range(len(WordEmbedding)):
    for j in range(len(WordEmbedding[i])):
        Sentences.append(WordEmbedding[i][j])


# In[21]:


SelectedSentences=[]                          #select sentence contain at least two entities
for i in range(len(Sentences)):
    count=0
    for j in range(len(Sentences[i])):
        if (Sentences[i][j].find(">")>1):
            count=count+1
    if(count>1):
        SelectedSentences.append(Sentences[i])
        
        


# In[22]:


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
        
    
    
        
    


# In[23]:


In2=copy.deepcopy(Instances)                                      #clean the tag of entity
for i in range(len(Instances)):
    for j in range(len(Instances[i])):
        Instances[i][j]=Instances[i][j].replace(":",".")
        Instances[i][j]=Instances[i][j].replace("<entity id=","")
        Instances[i][j]=Instances[i][j].replace("</entity>","")
        Instances[i][j]=Instances[i][j].replace(">"," ")
        
        


# In[24]:


for i in range(len(Instances)):                                 #split entity
    for j in range(len(Instances[i])):
        Instances[i][j]=Instances[i][j].split(" ")


# In[25]:


Distance=[]
Distance2=[]


# In[26]:


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


# In[27]:


for i in range(len(Distance)):
    ss=sum(Distance[i])
    for j in range(len(Distance[i])):
        Distance[i][j]=Distance[i][j]/ss


# In[28]:


for i in range(len(Distance2)):
    ss=sum(Distance2[i])
    for j in range(len(Distance2[i])):
        Distance2[i][j]=-Distance2[i][j]/ss


# In[29]:


for i in range(len(Instances)):                             #Insert the whole text of entity back to sentences                                
    Instances[i][0]=Instances[i][0][1:]


# In[30]:


for i in range(len(Instances)):
    l=len(Instances[i])
    Instances[i][l-1]=Instances[i][l-1][1:]


# In[31]:


for i in range(len(Instances)):
    onelist=[]
    for j in range(len(Instances[i])):
        onelist.extend(Instances[i][j])
    Instances[i]=onelist


# In[33]:


for i in range(len(Instances)):
    if("," in Instances[i]):
        Instances[i].remove(",")      
        
    


# In[324]:


filename = 'acl_vectors_glove_300d.txt.word2vec'           #load the pre-trained model from Glove
model = KeyedVectors.load_word2vec_format(filename, binary=False)
glove_input_file = 'acl_vectors_glove_300d.txt'
word2vec_output_file = 'acl_vectors_glove_300d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)


# In[325]:


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


# In[326]:


for i in range(len(wordvector)):
    for j in range(len(wordvector[i])):
        wordvector[i][j].append(Distance[i][j])
        wordvector[i][j].append(Distance2[i][j])


# In[327]:


paddingvector=[]                           #creat padding vector
for i in range(302):
    paddingvector.append(0)


# In[328]:


for i in range(len(wordvector)):          #extend vector to max length
    if(len(wordvector[i])<67):
        l=len(wordvector[i])
        more=67-l
        for j in range(more):
            wordvector[i].append(paddingvector)


# In[329]:


Label=pd.read_csv("relation1.1.csv",encoding="gbk") #load label


# In[295]:


entity1=Label.entity1
entity2=Label.entity2


# In[296]:


for i in range(len(Epairs)):
    Epairs[i][0]=Epairs[i][0][1:-1]
    Epairs[i][1]=Epairs[i][1][1:-1]


# In[46]:


LabelInput=[]
LabelDetail=[]
Reverse=[]

#recognize positive or negative sentences
for i in range(len(Epairs)):
    m=0
    t=0
    for j in range(len(entity1)):
        if(Epairs[i][0]==entity1[j] and Epairs[i][1]==entity2[j]):
            m=m+1
            t=j
            
    if(m>0):
        LabelInput.append(1)
        LabelDetail.append(Label.relation[t])
        Reverse.append(Label.reverse[t])
        
    else:
        LabelInput.append(0)
        LabelDetail.append('none')
        Reverse.append('none')
        
        
        
        


# In[81]:


#c={ "Instances":Instances, "detail":LabelDetail, "Reverse":Reverse, "LabelInput":LabelInput, "Pairs":Epairs}


# In[82]:


#c=DataFrame(c)


# In[47]:


#c.to_csv("Instances0701.csv")


# In[332]:


Label=[]
for m in LabelInput:
    c=[m]
    Label.append(c)
    


# In[333]:


trainD=wordvector[:6800]
trainl=Label[:6800]
st=[]
sl=[]


# In[334]:


for i in range(6800):
    if trainl[i][0]==1:
        trainD.append(trainD[i])
        trainl.append(trainl[i])
        trainD.append(trainD[i])
        trainl.append(trainl[i])
       


# In[335]:


#for i in range(6800):
#    if trainl[i][0]==1:
#        st.append(trainD[i])
#        sl.append(trainl[i])
        


# In[336]:


#i=0
#s=0
#while s<1090:
#    i=i+1
#    if trainl[i][0]==0:
#        st.append(trainD[i])
#        sl.append(trainl[i])
#        s=s+1

   
    
    


# In[338]:


trainD=np.array(trainD)


# In[339]:


trainl=np.array(trainl)


# In[340]:


import numpy as np
#Label=np.array(Label)              #change Label and sentences in array layout
TrainData=np.array(wordvector)


# In[308]:


import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence


# In[60]:


from sklearn.cross_validation import train_test_split


# In[341]:


model = Sequential()


# In[342]:



test_x=TrainData[6801:]
test_label=Label[6801:]


# In[343]:


input_shape = (67, 302) 


# In[344]:


len(trainD[0][0])


# In[410]:


embedding_dim = 302             #parameters of the cnn model         
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 302
batch_size = 64
num_epochs = 18


# In[411]:


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


# In[412]:


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



# In[413]:


model.fit(trainD,trainl, batch_size=batch_size, epochs=num_epochs, # fit model
          validation_data=(test_x, test_label), verbose=2)


# In[109]:


from keras.models import Sequential             # save model
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


# In[110]:


# load json and create model              #load model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])
score = loaded_model.evaluate(test_x, test_label, verbose=0)


# In[400]:


result_cnn=model.predict(test_x, batch_size=32, verbose=0) # get the prediction


# In[401]:


PP=0                                            # calculate the evaluation matrix
PN=0
TP=0
TN=0
for i in range(len(result_cnn)):
    if result_cnn[i][0]>0.5:
        PP=PP+1
        if test_label[i][0]==1:
            TP=TP+1
    if result_cnn[i][0]<0.5:
        PN=PN+1
        if test_label[i][0]==0:
            TN=TN+1
               
        
    
        
    
    


# In[403]:


PP


# In[404]:


PN


# In[405]:


TP


# In[406]:


TN

