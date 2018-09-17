#Author:Sara
# coding: utf-8

# In[2]:


from nltk import pos_tag
from numpy import array
import numpy as np
import nltk
#nltk.download('tagsets')


# In[3]:


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
print(Tags)   


# In[7]:


def posTag(sss):
    
    d=nltk.pos_tag([sss])
    words1, tags1 = zip(*d)
    for i in tags1:
        i=i.replace(",","")
        if i in Tags:
            return(Tags.index(i))
        else:
            #the word is not exist
            return(-1)


# In[6]:


print(posTag('Oral'))

