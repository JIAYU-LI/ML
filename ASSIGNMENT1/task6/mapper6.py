#!/usr/bin/python

import sys

k1, a = None, []
total = 0.0

for line in sys.stdin:
    
     
    seq, count = line.strip().split("\t")  #obtain four-token and corresponding value from outputs of task4
    count = float(count)  #convert type of count from default string to float

    k_3 = seq.strip().split()[0:3] #select the first three tokens
    k_3st = " ".join(k_3) #convert three-token into string
     
    
    if k1 != k_3st: 
        if k1:
        
            print("{0}\t{1}\t{2}".format(k1, a, total))
        del a[:]   
        k1 = k_3st 
        
        a.append(count)  #group values into a list
        total = count  
    else:
        
        a.append(count)
        total += count  #sum values with the same three-token

if k1 == k_3st:
    
    print("{0}\t{1}\t{2}".format(k1, a, total))

