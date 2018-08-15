#!/usr/bin/python3

import sys
    
key, total = None, 0
st = " "

for line in sys.stdin:
    
    k, count = line.strip().split("\t")[:-1], line.strip().split("\t")[-1]
    st = " ".join(k)
    count = int(count)
    if st != key:
        if total == 1:
            print(key)
        key, total = st, count
		
    else:
        total += count
	

if total == 1:
    print(key)
    
