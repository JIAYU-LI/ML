#!/usr/bin/python

import sys
lowest = 0.0
stu = None
for line in sys.stdin:
    
    average, ID = line.strip().split("\t")
    average = float(average)
    if lowest == 0.0:  #select the first row since lines have been sorted
        lowest = average
        stu = ID
    
    else:
        if average == lowest and ID != stu:
            stu = "{0}\t{1}".format(stu, ID)  #pick ID with the same lowest mark

print'%s%f\t%s%s'%("Scores:", lowest, "ID:", stu)  


