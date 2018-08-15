#!/usr/bin/python

import sys

maxB = 0  #set the initial max value
maxW = 0

for line in sys.stdin:
	
	
    bytes, words = line.strip().split("\t") #get the number of bytes and tokens from outputs of mapper3
	
	
    b = int(bytes) #convert the default string type into integer in order to make comparison latter
    w = int(words)
	
			
    if maxB == 0 and maxW ==0:
        maxB, maxW = b, w	

print("{0}\t{1}".format(maxB, maxW))
 
	 
