#!/usr/bin/python

import sys

k = None
v=()
for line in sys.stdin:
    ID, mc = line.strip().split("\t")[0], line.strip().split("\t")[1:]  #get ID, mark and course from the output of task7
    
    value = tuple(mc)  #convert mark and course into tuple
    if k != ID:  #find out each ID's corresponding mark and course
        
        if v!=():
            
            print("{0}-->{1}".format(k, v))
        else:
            if k:
               
                print("{0}-->".format(k))  #only output ID if this ID does have marks
        k = ID
        v=value
    else:
        if v!=() and value !=():  #only output exist value rather than none
            v = "{0} {1}".format(v, value)

if ID == k:  #write the last line to output
    if v!=():
            
        print("{0}-->{1}".format(k, v))
    else:
         
        print("{0}-->".format(k))
    
