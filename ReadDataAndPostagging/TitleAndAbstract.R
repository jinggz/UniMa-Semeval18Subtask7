#install.packages("XML")
library(XML)
library(methods)
url<-'D:\\2017s\\AdminLTE-2.4.2 (1)\\AdminLTE-2.4.2\\dist\\img\\1.1.text.xml'
xmldoc<-xmlParse(url)
rootNote<-xmlRoot(xmldoc)
data<-xmlSApply(rootNote,function(x)xmlSApply(x,xmlValue))
data<-data.frame(t(data),rownames=F)


