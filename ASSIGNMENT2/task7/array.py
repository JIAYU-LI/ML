#!/usr/bin/env python
import commands

m=10*int(commands.getoutput('cat /afs/inf.ed.ac.uk/group/teaching/exc/ex2/part3/webLarge.txt |wc -l'))

with open('bitarray', 'wb') as b: #create a file with size m and be used to store binary output
    b.seek(m-1)
    b.write(b'\x00')
