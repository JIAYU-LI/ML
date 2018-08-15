#!/usr/bin/python

import sys
k = " "
v = " "

for line in sys.stdin:
    
    keyM, count = line.strip().split("\t")   #obtain four-token and corresponding value from outputs of task4
    k1 = "".join(keyM)
    	
    if k1!= v:  #change previous value to new key for the sake of latter sorting
        
        k = count
	v = k1
        

        print("{0}\t{1}".format(k, v))
