#!/usr/bin/python

import sys

k1, k2, v1, v2 = None, None, None, None

for line in sys.stdin:

    s_m = line.strip().split()[0]

    if s_m != "student":  #output ID and existed course and mark
        ID, course, mark = line.strip().split()[1], line.strip().split()[2], line.strip().split()[3]
        ID, mark = int(ID), int(mark)
		
    
        k1, v1, v2 = ID, course, mark
 	print("{0}\t{1}\t{2}".format(k1, v2, v1))
	
    else:  #output ID with no mark and course records
        ID, name = line.strip().split()[1], line.strip().split()[2]
        k2 = ID
    
        print("{0}\t{1}\t{2}".format(k2, " ", " "))
