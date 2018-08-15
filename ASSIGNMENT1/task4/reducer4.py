#!/usr/bin/python

import sys

key, total = None, 0
for line in sys.stdin:

    keyM, count = line.strip().split("\t")  #obtain four-token and value 1 from outputs of mapper4
    count = int(count)  #convert the type of count for the sake of latter calculation
   

    if key!= keyM:  #output four-token and its total number
	if total != 0:
	    print("{0}\t{1}".format(key,total))  #output four-token and its total value
        key, total = keyM, count
	
    else:
        total += count

if keyM == key:  #write the last unique line to stdout
    print("{0}\t{1}".format(key,total))

