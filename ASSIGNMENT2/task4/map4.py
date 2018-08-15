#!/usr/bin/env python
import sys
import random
samp = None
n_line = 0
for line in sys.stdin:
    num = random.randint(0, n_line) #select a random number from range(0, number of lines)
    
    if num == 0: #print the line that satisfies the specific condition 0(at least one line is printed since we chose a random number from (0, 0) at first
        samp = line.strip()
    n_line += 1 #sum number of lines in order to ensure equal probability

print("{0}\t{1}".format(samp, n_line)) #print selected line and total number of lines in each mapper
