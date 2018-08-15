#!/usr/bin/python

import sys
import math

key, H = None, 0.0

for line in sys.stdin:
    k2, p = line.strip().split("\t")
    p = float(p)

    if k2 != key:
	key = k2
        H = math.log(p, 2)*(-p)
    else:
	H += math.log(p, 2)*(-p)

    
    print("{0}\t{1}".format(key, H))
