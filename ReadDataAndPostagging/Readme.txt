FormData.py is file that read data from xml file.

Postagging.py is file that do the pos tagging.

TitleAndAbstract.R is file that get the clean text of title and abstract.

1.1.text.xml is file contain original xml text.

abstract1.csv is clean text of all abstract.

Entity.csv is all the entity id and entity word.

PostaggingFile.csv is the data contain pos tagging reslut.(used for naive bayes baseline)

RelatioinV3.csv is all the relations, and the entity id has been replaced by the entity word.

Rt.csv is file used for PostaggingFile.csv. This file contain both entity id and entity word.

TitleAndAbstract.csv is file contain clean text of title and abstract.

Postagging2.0.py is the file get all tags of entities.
---------------------------------------------------------------------------------------------------------
NB&Features.py is the file that creat features and NB with given featureset  

PreprocessingLemmatizing.py is the preprocessing file that use TitleAndAbstract.csv  and remove stopwords, Lemmatization, remove Uppercase. The 'Lemmatizeredfile.txt' is the output.

PreprocessingStemming.py is the preprocessing file that use TitleAndAbstract.csv  and remove stopwords, Stemming, remove Uppercase. The 'Stemmedfile.txt' is the output.

Features.py is the file that use vectorize and countVectorizer, convert uppercase,removed words less than two letters(like 'a'), remove punctuations and duplicates and implement some features. it used "InterpretEntities.py" to discover entity list.

namedEntityRecognition.py is the file that use TitleAndAbstract.csv as an input, tokenize sentences,remove stopwords, stemming, pos tagging and seeks to locate and classify elements in text into pre-defined categories.The idea is to have the machine immediately be able to pull out "entities" like people, places, things, locations, monetary figures, and more.The 'NameEntityRecognition.txt' is the output.
-----------------------------------------------------------------------------------------------------------
CrossValidation_Yuan.py is the file that implement k-fold cross validation for NB. Read: PostaggingFile and output output_kfold.
-----------------------------------------------------------------------------------------------------------
