#!/usr/bin/env python
import sys
import commands

window_size = 1000 #(1/error parameter) = 1000
dic={}
n_line=0
del_key=[]

orig_stdout = sys.stdout
f = open('output8.txt', 'w')
sys.stdout = f

def spill(dic):
    for key, value in dic.items():
        #output lines that satisfy specific condition 1%
        if value>=1*int(commands.getoutput('cat /afs/inf.ed.ac.uk/group/teaching/exc/ex2/part4/queriesLarge.txt | wc -l'))/100:
            print(key)
    
          
for line in sys.stdin:      
    que = line.strip()

    dic[que]=dic.get(que, 0)+1 #count each line read from stdin
    n_line+=1
    
    if n_line==window_size: #count number of lines until arrive at window size
        for key in dic.keys():
            dic[key]=int(dic[key])-1 #minus 1 and delete the line's frequency of only 1
            if dic[key] == 0:
                del_key.append(key)
        for i in del_key: #store lines that need to be deleted in a list in order to delete in the dictionary
            del [i]
        del_key=[]
        n_line=0
        
for key in dic.keys(): #in case for total lines/windows size != int
    dic[key]=int(dic[key])-1
    if dic[key] == 0:
        del_key.append(key)
for i in del_key:
    del [i]
del_key=[]
spill(dic)

sys.stdout = orig_stdout
f.close()              


