# generate TF-IDF vector for texts in between two entities
# corpus: texts in between two entities plus words of entities
# output: dense format of vector matrix
# used package: sklearn.TfidfVectorizer
# based on work of Sara and Zhonghao
# author: Jingyi

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA

path = "F:\\Data Science\\Team Project\\tp\\"
Doc=pd.read_csv(path+"Instances0701.csv",encoding="gbk")
sentence=Doc.Instances
Sentences=[]
# transform texts in order to be tokenized
for sentences in sentence:
    sentences=sentences.replace("[","")
    sentences=sentences.replace("]","")
    sentences=sentences.replace("'","")
    sentences=sentences.replace(" ","")
    sentences=sentences.replace(","," ")
    Sentences.append(sentences)

# generate TF-IDF vector
vectoriser = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words='english', analyzer='word', lowercase=True)
tfidfVectors = vectoriser.fit_transform(Sentences)
#tfidfVectors_sparse = tfidfVectors.toarray()

from scipy import sparse
#Save
sparse.save_npz(path+'tfidf_0117.npz', tfidfVectors)
#Load
#data = sparse.load_npz(path+ "tfidf_0117.npz")
print(tfidfVectors.shape)
print(tfidfVectors.toarray())

# ids=pd.read_csv(path+'ModelInput.csv', usecols=['Entity Pair'])
# ids=np.array(ids).flatten()
# import os
# import codecs
# if os.path.exists('instance_tfidf_0117.txt'):
#      os.remove('instance_tfidf_0117.txt')
# #save_test=save.iloc[ : , :10 ]
# for entitypair, vec in zip(ids,tfidfVectors_sparse):
#         with codecs.open(path + 'instance_tfidf_0117.txt', 'a', encoding='utf-8') as f:
#             f.write(entitypair+str(vec))
#             f.write('\r\n')