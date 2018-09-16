
# coding: utf-8

# In[19]:

#Input: A sentence including its entity tags
#Output: List:
#    First element: dictionary containing:
#        the indexes of the entities as key
#        the entity identifyer as value
#    Second element: given sentence with entities removed
#Note: This output is the input for the feature generator
#Author: Thorsten
def interpretEntities(sentence):
    positions=set()
    newSentence=sentence[:]
    tags={}
    for x in range(len(sentence)):
        #Use: If the word is tagged as entity, the tags are at the beginning and at the end.
        #check if it is an entity
        if(len(sentence[x])<23):
           #too short for an entity, encoding needs at least 21 characters
            continue
        if(sentence[x][:12]=='<entity id="'):
            #entity beginning tag found
            #now search the end
            y=12
            while(len(sentence[x])>y and sentence[x][y]!='>'):
                y+=1
            y+=1
            tags.update({x: sentence[x][12:y-2]})
            #now search the ending tag
            z=len(sentence[x])-9 #index of beginning of the ending tag
            if(sentence[x][-9:]=='</entity>'):
                #entity tag complete, if not error
                positions.update({x})
                newSentence[x]=newSentence[x][y:z]
    return [tags, newSentence]


# In[21]:

#for testing
#se=['<entity id="H01-1001.1">Oral communication</entity>', 'is', 'ubiquitous', 'and', 'carries', 'important', 'information', 'yet', 'it', 'is', 'also', 'time', 'consuming', 'to', 'document']
#sentence=interpretEntities(se)
#print(sentence)

