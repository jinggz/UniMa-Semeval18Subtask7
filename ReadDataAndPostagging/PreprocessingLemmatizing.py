# remove Stopwords
# Lemmatization
# remove Uppercase
#Title&Abstract as input
# Author: Sara

from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
import io
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#word_tokenize accepts a string as an input, not a file.
lemmatizer=WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
file1 = open("TitleAndAbstract.csv")
line = file1.read()# Use this to read file content as a stream:
words = line.split()
if os.path.exists('Lemmatizeredfile.txt'):
   os.remove('Lemmatizeredfile.txt')
for r in words:
    if not r in stop_words:
        r=r.lower()
        lm=lemmatizer.lemmatize(r)
        appendFile = open('Lemmatizeredfile.txt','a')
        appendFile.write(" "+lm)
        appendFile.close()
       



