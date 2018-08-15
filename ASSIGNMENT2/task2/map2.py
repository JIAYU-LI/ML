#!/usr/bin/python

import sys
import re
typeID, rowID, count = None,None,None
top_10 = []
for line in sys.stdin:
    
    
    typeID=re.findall(r'(?<=PostTypeId=")(\d+)', line) #find the unique typeId in each line
    typeID =int(" ".join(typeID))

    
    if typeID==1: #find and output "ViewCount" and "row Id" in question line 
        rowID=re.findall(r'(?<=row Id=")(\d+)', line)
        rowID=int(" ".join(rowID))

        count=re.findall(r'(?<=ViewCount=")(\d+)', line)
        count=int(" ".join(count))
        
        top_10.append((count, rowID))

    if len(top_10)>=200: #set 200 boundary to dictionary
        top_10.sort(reverse=True)
        del top_10[10:] #delete the latter 190 in the sorted list so as to remain the top 10 

top_10.sort(reverse=True) 
del top_10[10:]   
    
for i in top_10:
    c, I = i[0],i[1] #output viewcount and question ID separated by a single space
    print("{} {}".format(c, I))
    
   
