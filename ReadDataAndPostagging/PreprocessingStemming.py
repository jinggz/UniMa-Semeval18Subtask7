# remove Stopwords
# Stemming
# remove Uppercase
#Title&Abstract as input
# Author: Sara

import io
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#word_tokenize accepts a string as an input, not a file.
ps=PorterStemmer()
stop_words = set(stopwords.words('english'))
file1 = open("TitleAndAbstract.csv")
line = file1.read()# Use this to read file content as a stream:
words = line.split()
if os.path.exists('Stemmedfile.txt'):
   os.remove('Stemmedfile.txt')
for r in words:
    if not r in stop_words:
        r=r.lower()
        st=ps.stem(r)
        appendFile = open('Stemmedfile.txt','a')
        appendFile.write(" "+st)
        appendFile.close()
        
