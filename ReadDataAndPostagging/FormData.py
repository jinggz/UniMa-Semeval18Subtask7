import  xml.dom.minidom
import csv
from pandas import DataFrame
import pandas as pd
dom = xml.dom.minidom.parse('1.1.text.xml')        #read xml file
root = dom.documentElement                          #get all the element from xml file
itemlist = root.getElementsByTagName('entity')     #get all the element named "entity"
item = itemlist[0]                                 #get first entity
un=item.getAttribute("id")                         #get id's value of first entity
cc=dom.getElementsByTagName('entity')             #get all elemennts value from xml file
c1=cc[0]
dict={un:c1.firstChild.data}                       #creat dictionary for entity number and text
for i in range(5258):
    j=i+1
    item=itemlist[j]
    un=item.getAttribute("id")
    c1=cc[j]
    dict[un]=c1.firstChild.data
#######################################################output the csv file of dictionary################################3
EntityNumber=[]
for i in range(5259):                            # get all the entity number
    item=itemlist[i]
    un=item.getAttribute("id")
    EntityNumber+=[un]

EntityText=[]
for i in range(5259):                           #get all the entity text
    c1=cc[i]
    EntityText+=[c1.firstChild.data]

CSVfile=DataFrame(EntityText,EntityNumber)
CSVfile.to_csv("EntityFile.csv")
##########################################################replace id with real words in file relation1.1.csv##################
import pandas as pd
import csv
df = pd.read_csv('relation1.1.csv')


for i in range(1228):    #replace entity1
    t=df.entity2[i]
    t2=dict[t]
    df.entity2[i]=t2


for i in range(1228):   #replace entity2
    t=df.entity1[i]
    t2=dict[t]
    df.entity1[i]=t2

df.to_csv("Realtion.csv")





