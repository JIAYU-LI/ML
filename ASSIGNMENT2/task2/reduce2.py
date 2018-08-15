#!/usr/bin/python

import sys
 
top_10=[]
for line in sys.stdin:
     
    count,ID=line.strip().split()
    count, ID = int(count), int(ID)
    top_10.append((count,ID)) #obtain top 10 from each mapper and store them as tuples in a list

top_10.sort(reverse=True)
del top_10[10:]   #delete the latter 190 in the sorted list so as to remain the top 10 
    
for i in top_10:
    c, I = i[0],i[1] #output viewcount and question ID separated by a single space
    print("{} {}".format(c, I))
