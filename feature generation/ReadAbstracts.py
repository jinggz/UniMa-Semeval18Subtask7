
# coding: utf-8

# In[1]:


#Read the given XML file (given from the project) and parse it.
#Replacement of "InputForFeatureGenerationAbstractWise.py".
#Author: Thorsten


# In[2]:


import xml.sax as sax


# In[3]:


#Parses an XML file containing the abstracts like given for the contest.
#Input: Filename or, if the file is in another folder, path to the file and name.
#    Example: '1.1.text.xml' (when file in the same folder).
#Output: all abstracts of the file in a 3-dim list: abstracts[abstractNr][sentenceNr][wordNr]
#    Note: The XML-tag for the intities is passed through because it's needed later.
#Author: Thorsten
def parse(file):
    parser=sax.make_parser()
    r=Reader()
    parser.setContentHandler(r)
    parser.parse(file)
    return r.getAbstracts()


# In[25]:


#Help class needed for the parse function, see the documentation of xml.sax for details.
#Author: Thorsten
class Reader(sax.handler.ContentHandler):
    def __init__(self):
        self.abstracts=[]
        self.inEntity=False
    def startElement(self, name, attrs):
        if(name=='doc'):
            return
        if(name=='text'):
            self.abstracts.append([])
            return
        if(name=='title'):
            self.abstracts[len(self.abstracts)-1].append([])
            return
        if(name=='abstract'):
            self.abstracts[len(self.abstracts)-1].append([])
            return
        if(name=='SectionTitle'):
            return
        if(name=='entity'):
            #little tricky, but doing it right would need huge adaptions in Features.py.
            self.inEntity=True
            i=len(self.abstracts)-1
            j=len(self.abstracts[i])-1
            self.abstracts[i][j].append('<entity id="'+attrs.getValue('id')+'">')
            return
        print('XML-Parser: unknown name ', name)
    def endElement(self, name):
        if(name=='entity'):
            self.inEntity=False
            #i=len(self.abstracts)-1
            #j=len(self.abstracts[i])-1
            #k=len(self.abstracts[i][j])-1
            #self.abstracts[i][j][k]=self.abstracts[i][j][k]+'</entity>'
    def characters(self, content):
        content=content.strip()
        if(len(content)==0):
            return
        i=len(self.abstracts)-1
        j=len(self.abstracts[i])-1
        k=len(self.abstracts[i][j])-1
        if(self.inEntity):
            #self.abstracts[i][j][k]=self.abstracts[i][j][k]+content
            self.abstracts[i][j][k]=self.abstracts[i][j][k]+content+'</entity>'
            self.inEntity=False
            return
        sentences=content.split('.')
        for x in range(len(sentences)):
            words=sentences[x].strip().split(' ')
            for w in words:
                self.abstracts[i][j].append(w.strip())
            if(x<len(sentences)-1):
                self.abstracts[i].append([])
                i=len(self.abstracts)-1
                j=len(self.abstracts[i])-1
    def getAbstracts(self):
        return self.abstracts


# In[24]:


#Code for testing
#abstracts=parse('1.1.text.xml')

#for i in range(len(abstracts)):
#    for j in range(len(abstracts[i])):
#        print(abstracts[i][j])
#    print()

#print(abstracts[0][1])
#print(len(abstracts))

