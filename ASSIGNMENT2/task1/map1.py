#!/usr/bin/python
import os
import sys
key=()
dic = {}
k, d = None, None
def spill(dic):
    for key, count in dic.items():
        k, d = key[0], key[1]
        print("{0}\t{1}\t{2}".format(k,d,count)) #output each word and related doc_id and frequence

for line in sys.stdin:
    
    files = os.environ["mapreduce_map_input_file"]
    
    doc_id = files.split('/')[-1]
    words = line.strip().split() #get words in each line
    for word in words:
        key = (word, doc_id) #word, doc_id consist of a composite key in the dictionary
        dic[key] = dic.get(key,0)+1 #frequency of word occurrence in each doc as the value of this dictionary
        
spill(dic)



