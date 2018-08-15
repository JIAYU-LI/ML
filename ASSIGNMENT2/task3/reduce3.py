#!/usr/bin/python

import sys
import re
k, v = None, None 
dic={} 
maxQ=0
key2, seq_ID=None, None

for line in sys.stdin:
    match_id, q_o = line.strip().split('\t') #extract elements from outputs of mappers
    
    if k != match_id:
        k, v = match_id, q_o
    else:
        if 'P' in q_o:
            ID = re.findall(r'(\d+)', q_o)[0]
            dic.setdefault(int(ID), []).append(int(v)) #create a dictionary which regards owner ID as key and a list of question ID as value
        else:
            ID2 = re.findall(r'(\d+)', v)[0]
            dic.setdefault(int(ID2), []).append(int(q_o))
    

for key, value in dic.items():
    
    if len(value)>=maxQ: #find the element in the dictionary with the longest value = the person who answers the most questions
        maxQ = len(value)
        key2 =  key 
        
        seq_ID=str(sorted(value)).lstrip('[').rstrip(']') #sort the list of question ID and convert the list to string
print("{0} ->  {1}".format(key2, seq_ID))







