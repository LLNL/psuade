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
# This file contains Python codes that read a sample data file and
# display a histogram 
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

try:
   from mpl_toolkits.mplot3d import axes3d, Axes3D
   MPL = True
except:
   print('INFO: mpl_toolkits unavailable')
   print('      Will create 2D contour plot instead')
   MPL = False

# get data file
print('************************************************************')
print('This function reads from a sample file and plots histogram')
print('using the data. The data file should be in this format:') 
print('Line 1: <n values>')
print('Line 2: <n values>')
print('....')
print('If n > 1, you will be asked to select a column to plot.')
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   dafile = raw_input('Enter data file name : ')
else:
   dafile = input('Enter data file name : ')

try:
   with open(dafile, 'r') as infile:
      allLines = infile.readlines()
      nLines = len(allLines)
      if nLines < 50:
         print('ERROR: < 50 lines ==> will not generate histogram')
         sys.exit(0)
except:
   print('ERROR: in reading ' + dafile)
   sys.exit(0)
   
lineIn = allLines[0]
cols = lineIn.split()
ncols = len(cols)
if ncols < 1:
   print('ERROR: the first line has no entry.')
   sys.exit(0)
      
for ii in range(nLines):
   lineIn = allLines[ii]
   nc = len(cols)
   if nc < ncols:
      break

if ii < 50:
   print('ERROR: < 50 valid lines ==> will not generate histogram')
   sys.exit(0)

nrows = ii + 1
ncols = nc
print('INFO : number of rows detected = ' + str(nrows))
print('       number of cols detected = ' + str(ncols))
col = 0
if ncols > 1:
   col = 0
   while col > ncols or col <= 0:
      if sys.version_info[0] < 3:
         colStr = raw_input('Which column to select : ')
      else:
         colStr = input('Which column to select : ')
      try:
         col = int(colStr) - 1
      except:
         print('ERROR: non-integer entry')

X = np.zeros(nrows)
for ii in range(nrows):
   lineIn = allLines[ii]
   cols = lineIn.split()
   try:
      X[ii] = float(cols[col])
   except:
      print('ERROR: non-numeric value entered in line ' + str(ii+1))
      sys.exit(0)
    
# plot
fig, ax = plt.subplots()
plt.hist(X, bins=20)
xlabel = 'Input'
plt.xlabel(xlabel, fontsize=14, fontweight='bold')
ylabel = 'Count'
plt.ylabel(ylabel, fontsize=14, fontweight='bold')
title = 'Histogram'
plt.title(title, fontsize=14, fontweight='bold')
for axis in ['top', 'bottom', 'left', 'right']:
   ax.spines[axis].set_linewidth(4)
#plt.grid(visible=True)
plt.savefig('psu_hist.png')
print('Histogram has been stored in psu_hist.png')

plt.show()
dmean = np.mean(X)
dstd  = np.std(X)
print('Sample mean    = ' + str(dmean))
print('Sample std dev = ' + str(dstd))
print('Histogram plot file is now in pythonhist.png')

