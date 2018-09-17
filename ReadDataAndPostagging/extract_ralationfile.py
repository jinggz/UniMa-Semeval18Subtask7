#coding: utf-8
#extract the file of relations and read into a csv file
# column 'REVERSE' represents if 2 entities are in reverse order in the text
# Author: Jingyi


import codecs
import csv
import os

dir = "F:\\Data Science\\Team Project\\dataset\\"
file1 = "F:\\Data Science\\Team Project\\dataset\\1.2.relations.txt" # should be replaced by your own dir
output1 = "F:\\Data Science\\Team Project\\dataset\\relation1.2.csv" # output file

# function to check if a file exists
def file_exists(filename):
    try:
        with open(filename) as f:
            f.close()
            return True
    except IOError:
            return False

if file_exists(file1):
    fr = codecs.open(file1, "r", "UTF-8")
else:
    print("Input file is not found.")

fw = codecs.open(output1, "w+", "UTF-8")
fieldnames = ['entity1', 'entity2', 'reverse', 'relation']  # add the header to output csv file
writer=csv.writer(fw)
writer.writerow(fieldnames)

for line in fr:
    relation = line.split('(')[0]   # first split using '('
    tempstr = line.split('(')[1]    # get the string after '('
    tempstr=tempstr.rstrip(')\n')   # remove the line break at the end of string
    if (tempstr.find("REVERSE") != -1):     # check if the entities are tagged as 'reverse order'; if true, it would split into 3 strings.
        entity1 = tempstr.split(',')[0]
        entity2 = tempstr.split(',')[1]
        reverse=True
    else:
        entity1, entity2 = tempstr.split(',')
        reverse=False
    writer.writerow([entity1, entity2, reverse, relation])
fr.close()
fw.close()

# print for testing
#while line:
    #print(line)




