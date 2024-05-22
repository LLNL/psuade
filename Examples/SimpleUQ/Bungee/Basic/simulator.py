#!/usr/bin/python
import sys
import string

# get input information
infile = open(sys.argv[1], "r")
lineIn = infile.readline()
ncols  = lineIn.split()
nInps  = eval(ncols[0])

# read inputs
X = []
for ii in range(nInps):
   lineIn = infile.readline()
   ncols  = lineIn.split()
   ddata  = eval(ncols[0])
   X.append(ddata)
infile.close()

# compute
g = 9.8
kel = 1.5
Y = X[0] - 2.0 * X[1] * g / (kel * X[2]);

# write output
outfile = open(sys.argv[2], "w")
outfile.write("%e\n" % Y)
outfile.close()

