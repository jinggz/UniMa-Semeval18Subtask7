
# coding: utf-8

# In[ ]:


#Help class for Features.py
#This is used to cache the results of the wikipedia queries, because the
#queries need a lot of time. This file uses da data base to store the
#words which had already a query and the results for them. So number of the
#expensive wikipedia queries can be reduced.
#If the data base does not exist, it will be created.
#Could also be used to cache other data, but than the filename of the data base
#should be changed.
#Use szenario: If you want to know the value for a word, use get(word). If the
#value is already there, you get it. If not, do the wikipedia request and than
#use insert(word, value) to store the result for later use. After everything is
#done, call close() to close the connection to the data base and so avoid
#recource leaks.
#Author: Thorsten


# In[1]:


import sqlite3


# In[24]:


#open the data base or creates it if it doesn't exist.
conn = sqlite3.connect('words.db')
c = conn.cursor()


# In[25]:


#create table if does not exist, necessary if the data base is new created.
ob=c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='words';")
conn.commit()
ob=ob.fetchall()
if(len(ob)==0):
    c.execute('Create table words (word text, value integer)')
    conn.commit()
    ob=c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='words';")
    conn.commit()


# In[26]:


#Ask the data base if the word has an entry.
#Input: word: string, the word which should looked up.
#Output: boolean value if word exists in the data base,
#        this value says if word has a wikipedia article (or whatever you want to store).
#    None if the word has no entry in the data base jet.
def get(word):
    ob=c.execute("select value from words where word=='"+word+"';")
    conn.commit()
    ob=ob.fetchall()
    if(len(ob)==0):
        return None
    if(ob[0][0]==0):
        return False
    if(ob[0][0]==1):
        return True
    print('Error, wrong value in the data base for ', word, ':')
    print(ob[0])
    sleep() #should crash the programm because of bad error


# In[27]:


#Insert a new word and its value to the data base.
#Input: word: sting, the word which should be added as key.
#    value: boolean, the value for this word.
def insert(word, value):
    if(value==False):
        value='0'
    else:
        value='1'
    ob=c.execute("insert into words values ('"+word+"', "+value+");")
    conn.commit()


# In[28]:


#Close the connection to the data base. May be needed to connect with other programs.
def close():
    conn.close()


# In[30]:


#for testing
#insert('communication', True)
#print(get('communication'))

