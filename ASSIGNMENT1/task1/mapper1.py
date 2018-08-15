#!/usr/bin/python

import sys
for line in sys.stdin:
	line = line.strip()
		
	if line.islower() is False:
		
		print(line + "\t1")  #label the line that needs to be converted into lower version
	else:
		print(line + "\t0")
		

			
