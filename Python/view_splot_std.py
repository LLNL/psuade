# ********************************************************************
# Copyright (c) 2021   Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the PSUADE team.
# All rights reserved.
#
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice,
# disclaimer, contact information and the GNU Lesser General Public 
# License.
#
# This is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License (as published by 
# the Free Software Foundation) version 2.1 dated February 1999.
#
# This software is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF 
# MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public 
# License along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 
# USA
# *******************************************************************
# This file contains Python codes that read a PSUADE standard file
# and display a X-Y scatter plot
# *******************************************************************
import os
import sys
import numpy as np 
try:
   import matplotlib.pyplot as plt
   MATPLOTLIB = True
except ImportError:
   print('INFO: matplotlib unavailable')
   MATPLOTLIB = False
   sys.exit(0)

print('************************************************************')
print('This function extracts data from a file in PSUADE standard')
print('format and creates a 1D scatter plot.')
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   dafile = raw_input('Enter name of the data file (in standard format) : ')
else:
   dafile = input('Enter name of the data file (in standard format) : ')

# first search for dimensions in X and Y
with open(dafile, 'r') as infile:
   lineIn = infile.readline()
   cols = lineIn.split()
   if len(cols) != 3:
      print('ERROR: the first line should have 3 integers.')
      print('       Namely, nSamples nInputs nOutputs.')
      sys.exit(0)
   nSams = int(cols[0])
   nInps = int(cols[1])
   nOuts = int(cols[2])
   if nSams <= 0 or nInps <= 0 or nOuts <= 0:
      print('ERROR: Invalid nSamples, nInputs, or nOutputs.')
      sys.exit(0)
   X = np.zeros([nInps, nSams])
   Y = np.zeros([nOuts, nSams])
   for ii in range(nSams):
      lineIn = infile.readline()
      cols = lineIn.split()
      if len(cols) != nInps+nOuts:
         print('ERROR: When reading line ' + str(ii+2))
         sys.exit(0)
      for jj in range(nInps):
         X[jj][ii] = float(cols[jj])
      for jj in range(nOuts):
         Y[jj][ii] = float(cols[jj+nInps])
 
# ask for information
print('Sample size    = ' + str(nSams))
print('Number inputs  = ' + str(nInps))
print('Number outputs = ' + str(nOuts))
if sys.version_info[0] < 3:
   inpStr = raw_input('Which input should be used as X axis : ')
else:
   inpStr = input('Which input should be used as X axis : ')
if sys.version_info[0] < 3:
   outStr = raw_input('Which output should be used as Y axis : ')
else:
   outStr = input('Which output should be used as Y axis : ')
try:
   inpInd = int(inpStr)
   outInd = int(outStr)
   if inpInd <= 0 or inpInd > nInps:
      print('ERROR: Wrong input index ' + inpStr)
      sys.exit(0)
   if outInd <= 0 or outInd > nOuts:
      print('ERROR: Wrong output index ' + outStr)
      sys.exit(0)
except:
   print('ERROR encountered in getting which input or output.')
   sys.exit(0)

# extract the proper columns to XX and YY
inpInd = inpInd - 1
outInd = outInd - 1
XX = np.zeros(nSams)
for ii in range(nSams):
   XX[ii] = X[inpInd][ii]
YY = np.zeros(nSams)
for ii in range(nSams):
   YY[ii] = Y[outInd][ii]

# plot
fig, ax = plt.subplots()
plt.scatter(XX, YY, s=20, marker='*')
xlabel = 'X' + inpStr
plt.xlabel(xlabel, fontsize=14, fontweight='bold')
ylabel = 'Y' + outStr
plt.ylabel(ylabel, fontsize=14, fontweight='bold')
title = '1D Scatter Plot'
plt.title(title, fontsize=14, fontweight='bold')
for axis in ['top', 'bottom', 'left', 'right']:
   ax.spines[axis].set_linewidth(4)
#plt.grid(b=True)

plt.savefig('psu_splot_std.png')
print('One-input-one-output scatter plot file is in psu_splot_std.png')

plt.show()

