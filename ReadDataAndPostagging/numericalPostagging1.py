from nltk import pos_tag
from numpy import array
import numpy as np
import nltk
#nltk.download('tagsets')
tagslist=[]
words=[]
vector=[]
#tagslist=nltk.help.upenn_tagset()
file1 = open("abstractsara.csv")
line = file1.read()# Use this to read file content as a stream:
words = line.split()
tagged_sent = pos_tag(words) 
#print(tagged_sent)
words, tags = zip(*tagged_sent)
words=words
Tags=list(set(tags))
#print(Tags)
#Tags=['NN','NNS','IN','JJ','CC','NNP','VBG','DT','CD','VB','VBN','VBP','EX','FW','JJR','JJS','LS','MD','NNPS']
matches = [x for x in tags if x in Tags]

for r in range(len(tags)):
     for r in matches:
        vector.append(True)
     else:
         vector.append(False)
         a=np.asarray(vector)
                        
         print(a)   
""""
out = 0
for bit in vector:
   out = (out << 1) | bit
print(out)    

"""         
         
        
