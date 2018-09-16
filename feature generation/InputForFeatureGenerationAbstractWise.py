
# coding: utf-8

# In[ ]:


#Creates the input for the feature generator.
#Input: Filename of the source file to be loaded: string
#Output: 3-dimensional list containing:
#1st dimension: abstract
#2nd dimension: sentence
#3rd dimension: word (including entity tags)
#Autor: Zhonghao (code itself) and Thorsten (making it to a function)
def inputForFeatureGenerationAbstractWise(filename):
    import  xml.dom.minidom
    import csv
    from pandas import DataFrame
    import pandas as pd
    from xml.etree import ElementTree as ET
    from xml.dom.minidom import parse
    import string
    
    dom = xml.dom.minidom.parse(filename)        #read xml file
    root = dom.documentElement                          #get all the element from xml file
    
    dom = parse(filename)
    abstract=[]
    
    for node in dom.getElementsByTagName('abstract'):
        f=node.toxml()
        abstract.append(f)
    
    dom = xml.dom.minidom.parse(filename)        #read xml file
    root = dom.documentElement  
    itemlist = root.getElementsByTagName('text')  
    TextId={}                                       #store the textID in a Dictionary
    for i in range(350):
        item=itemlist[i]
        TextId[i]=item.getAttribute("id")
        
    for i in range(len(TextId)):                       #split the abstract into sentence wise 
        t=abstract[i].replace("<abstract> ","")
        t=t.replace(" </abstract>","")
        tt=TextId[i]
        tf=tt+"."
        ttt=tt+":"
        t=t.replace(tf,ttt)
        tm=t.split(".")
        abstract[i]=tm
    
    AbstractWise=[]
    
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
        
    for i in range(350):
        for j in range(len(AbstractWise[i])):
            for h in range(len(AbstractWise[i][j])):
                    Sentence=AbstractWise[i][j]              
                    Word=Sentence[h]
                    if (Word.find(">")>0):
                        AbstractWise[i][j][h]="<entity id="+Word+"</entity>"
                        
    return AbstractWise


# In[ ]:


#help function for inputForFeatureGenerationAbstractWise
def secondSpilt(tests):
    for i in range(len(tests)):
        tests[i]=tests[i].split("<entity id=")
        
    return tests
    


# In[ ]:


#help function for inputForFeatureGenerationAbstractWise
def finalSplit(tts):
    tfs=[]
    for i in range(len(tts)):
        if tts[i].find(">")<0:
            tfs.append(tts[i].split(" "))
        if tts[i].find(">")>0:
            tfs.append(tts[i])
    
    return tfs
 


# In[ ]:


#help function for inputForFeatureGenerationAbstractWise
def FinalSentences(Final):
    FinalSentence=[]
    for i in range(len(Final)):
        for j in range(len(Final[i])):
            if type(Final[i][j])==list:
                FinalSentence.extend(Final[i][j])
               
            else:
                FinalSentence.append(Final[i][j])
                
                
    return FinalSentence
           


# In[ ]:


#for testing
#inputForFeatureGenerationAbstractWise()[0][0]

