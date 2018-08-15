#!/usr/bin/env python
import sys
import commands
import mmh3
import os
import mmap

orig_stdout = sys.stdout
f = open('output7.txt', 'w')
sys.stdout = f

def memory_map(filename, access=mmap.ACCESS_WRITE): #build a shared memory to map previous created file 'bitarray'
    size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDWR)
    return mmap.mmap(fd, size, access=access)
bitarray=memory_map('bitarray')

#bits per key=9.6, here set 10*n
m=10*int(commands.getoutput('cat /afs/inf.ed.ac.uk/group/teaching/exc/ex2/part3/webLarge.txt |wc -l')) 

n_hash = 7 #compute with probability of 1%, got 6.6

for line in sys.stdin:      
    count=0 
    for i in range(n_hash): #loop 7 times for a hash function mmh3
        #get bit number in the array through mmhs hash function and use % in order to ensure bit number < m
        result_hash = mmh3.hash("{}_{}".format(i,line)) % m 
        if bitarray[result_hash]==0: #output lines that satisfy the condition of 0 value
            #count=1
            bitarray[result_hash]=1 #convert value to 1 with regards to unexisted lines
for i in bitarray: 
    print(i)

sys.stdout = orig_stdout
f.close() 
