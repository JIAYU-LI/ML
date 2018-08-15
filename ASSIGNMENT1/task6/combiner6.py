#!/usr/bin/python

import sys

for line in sys.stdin:

    k1, v, s = line.strip().split("\t")
    s = float(s)
    v = v.strip('[]')
    a = v.split(',')
    total = len(a)

    for i in range(total):
        a[i] =float(a[i])
        p=a[i]/s
        
        print("{0}\t{1}".format(k1, p))
