#!/usr/bin/python3

import sys
for line in sys.stdin:
    l = line.strip()
    print("{0}\t{1}".format(l, 1))  #label each line with 1
