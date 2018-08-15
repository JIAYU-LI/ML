#!/usr/bin/python

import sys

for line in sys.stdin:
    
    four_token = None  #set initial four-token sequence

    words = line.strip().split()  #obtain tokens in each line
    
	
    if len(words) >= 4:  #only select lines with more that four tokens
        for j in range(0, len(words)-3):

            four_token = words[j:j+4]
            k = " ".join(four_token)  #convert four-token into string
	    print("{0}\t{1}".format(k, 1))

