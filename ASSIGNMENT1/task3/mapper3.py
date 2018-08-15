#!/usr/bin/python

import sys

for line in sys.stdin:

    key = line.strip()  #Remove trailing characters
    words = line.strip().split() #get words in each line
    bytes = len(key) #get total bytes of each line
    tokens = len(words) #get the number of words in each line
    print("{0}\t{1}".format(bytes, tokens))
	
