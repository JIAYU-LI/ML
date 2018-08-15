#!/usr/bin/python

import sys
key = None
for line in sys.stdin:
	keyL, count = line.strip()[:-2], line.strip()[-1]
    	count = int(count)  #convert the default type of 'count' in order to pick lines

	if count == 1:  #convert non-lower version line into lower version
		key = keyL.lower()
		print(key)
	else:
		key = keyL  #output previous lower-version lines
		print(key)
	
