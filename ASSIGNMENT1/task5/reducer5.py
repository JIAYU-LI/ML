#!/usr/bin/python

import sys
 
count = 0
for line in sys.stdin:
    l = line.strip()  # Remove trailing characters
    count +=1  #count the number of each input

    if count <=25: #output the first 25 frequent four-token sequence and their values
        print(line)
    

