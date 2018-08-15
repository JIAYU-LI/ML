#!/usr/bin/env python
import sys
import commands
from bitarray import bitarray
import mmh3

m=10*int(commands.getoutput('cat /afs/inf.ed.ac.uk/group/teaching/exc/ex2/part3/webLarge.txt |wc -l')) #bits per key=9.6, here set 10*n
bitarr=bitarray('0')*m
n_hash = 7 #compute with probability of 1%, got 6.6

orig_stdout = sys.stdout
f = open('output6.txt', 'w')
sys.stdout = f

for line in sys.stdin:      
    count=0 
    for i in range(n_hash): #loop 7 times for a hash function mmh3
        #get bit number in the array through mmhs hash function and use % in order to ensure bit number < m
        result_hash = mmh3.hash("{}_{}".format(i,line.strip())) % m 
        if bitarr[result_hash]==0: #output lines that satisfy the condition of 0 value
            count=1
            bitarr[result_hash]=1 #convert value to 1 with regards to unexisted lines
            
    if count!=0: #unique line if failling to encounter 0 value in the array
        
        print(line)

sys.stdout = orig_stdout
f.close()      

