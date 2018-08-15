#!/usr/bin/python

import sys

n=0 #the number of documents

list_doc={} #creat a dictionary to store doc ID and frequency
k, doc, freq=None, None, None #previous word, doc ID and frequency
for line in sys.stdin:
    word, doc_id, count = line.strip().split('\t')
    count=int(count)
    if k == word:
        
        list_doc[doc_id]=list_doc.get(doc_id, 0)+count #considering that doc_id will repeat in different mappers
        
    else:
        if k:
            n=len(list_doc) #get the final number of doc
            
            c=str(sorted(list_doc.items())).replace('[','{').replace(']','}').replace("'", " ") #convert to expected output format
            print("{0} : {1} : {2}".format(k,n,c))
            list_doc.clear() #clear unnecessary data after output
        
        k, doc, freq=word, doc_id, count
        list_doc[doc]=list_doc.get(doc, 0)+freq
        

n=len(list_doc)
c=str(sorted(list_doc.items())).replace('[','{').replace(']','}').replace("'", " ")
print("{0} : {1} : {2}".format(k,n,c))
