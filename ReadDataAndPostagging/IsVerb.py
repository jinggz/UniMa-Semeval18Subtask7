
# coding: utf-8

# In[23]:


#Author:Sara
#if its verb return 1 if not -1
#for using as a featur
# coding: utf-8


# In[24]:


from nltk import pos_tag
from numpy import array
import numpy as np
import nltk
#nltk.download('tagsets')


# In[25]:


tagslist=[]
words=[]
#tagslist=nltk.help.upenn_tagset()
file1 = open("abstractsara.csv")
line = file1.read()# Use this to read file content as a stream:
words = line.split()
tagged_sent = pos_tag(words) 
#print(tagged_sent)
words, tags = zip(*tagged_sent)
words=words
Tags=list(set(tags))
Tags=sorted(Tags)
#print(Tags)
#VB=Tags.index('VB')
#VB=list('VB','VBZ','VBD','VBG',')


# In[34]:


def posTag(sss):
    
    d=nltk.pos_tag([sss])
    words1, tags1 = zip(*d)
    for i in tags1:
        i=i.replace(",","")
        #print(i)
        #need this way to work, but this word is too important to have a false negative
        if(sss.startswith('use')):
            return 1
            
        if i.startswith('VB'):
             return(1)
        else:
             return(-1)
             
        


# In[36]:


#print(posTag('is'))
#print(posTag('be'))
#print(posTag('lying'))
#print(posTag('use'))

