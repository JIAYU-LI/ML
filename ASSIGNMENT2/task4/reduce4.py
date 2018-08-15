#!/usr/bin/env python
import sys
import random
#select one single line from outputs of mappers according to diff weights
n_line, total = 0, 0
samp = None
for line in sys.stdin:
    l, n = line.split('\t')
    n = int(n)
    
    for i in range(n): #loop more time if n is bigger
        total = i+n_line
        num = random.randint(0, total) #ensure probability of 1/n
    
        if num == 0:
            samp = l
    n_line += n 

print(samp)
