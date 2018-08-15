#!/usr/bin/env python
import sys
import random
samp = [] 
n_line = 0
orig_stdout = sys.stdout
f = open('output5.txt', 'w')
sys.stdout = f

for line in sys.stdin:


    if n_line < 100:
        samp.append(line.strip()) #Firstly fill 100 lines into a list
    else:
        index_samp = random.randint(0,n_line) #select a random number from (0, number of lines)
        if index_samp < 100: #ensure probability of 100/n
            samp[index_samp] = line.strip() #selected lines are filled into the previous list
    n_line += 1
for i in samp:
    print(i)        

sys.stdout = orig_stdout
f.close()
