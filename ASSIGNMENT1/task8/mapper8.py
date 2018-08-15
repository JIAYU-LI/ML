#!/usr/bin/python

import sys

average = 0.0 
k = None
for line in sys.stdin:
    ID, courseM = line.strip().split("-->")[0], line.strip().split("-->")[1:]
    
    m = " ".join(courseM)  #deal with obtained (mark, course) in order to calculate the number of tuples
    m = m.replace("'", " ")
    ele = m.strip().replace(')', " ").replace('(', " ").replace(',', " ")
    l = ele.split()
    n = len(l)/2  #pick mark from (mark, course)
    if n >=3:
        v = 0
        for i in range(len(l)):
            if i%2 == 0:
                
                l[i] = int(l[i])  #convert default string into int in order to sum marks
                v += l[i]
        k = ID
        if v:
            v=float(v)
    
            average = v/n  #compute the average mark
   
            print("{0}\t{1}".format(average, ID))  
            
    

    
