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
# This file contains Python codes that read matlabua.m created by
# PSUADE and display the histogram
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

# get matlabua.m-like file
print('************************************************************')
print('This function reads from a matlabua.m-like file and plots')
print('the histogram.') 
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   uafile = raw_input('Enter matlabua.m-like file name : ')
else:
   uafile = input('Enter matlabua.m-like file name : ')
if not os.path.isfile(uafile):
   print('ERROR: data file not found.')
   sys.exit(1)

try:
   with open(uafile, 'r') as infile:
      allLines = infile.readlines()
except:
   print('ERROR: in reading ' + uafile)
   sys.exit(0)

# extract the data
lineNum = 0
while lineNum < len(allLines):
   lineIn = allLines[lineNum]
   sInd = lineIn.find('Y = [')
   if sInd >= 0:
      YArray = []
      lineNum = lineNum + 1
      while 1:
         lineIn = allLines[lineNum]
         cols = lineIn.split()
         if cols[0] == '];':
            break
         else:
            YArray.append(float(cols[0]))
            lineNum = lineNum + 1
      break
   lineNum = lineNum + 1
leng = len(YArray)
if leng < 10:
   print('ERROR: data length < 10')
   sys.exit(0)
 
# plot
fig, ax = plt.subplots()
nhist = 10
counts, bins = np.histogram(YArray,nhist)
probs = 1.0 * counts / len(YArray)
plt.hist(bins[:-1], bins, weights=probs)
pstr = 'Probability'
plt.ylabel(pstr, fontsize=13,fontweight='bold')
title = 'Probability Distribution'
plt.title(title, fontsize=13,fontweight='bold')
plt.xlabel('Y', fontsize=13,fontweight='bold')
xmax = np.max(YArray)
xmin = np.min(YArray)
xstep = (xmax-xmin)/2
plt.xticks(np.arange(xmin,xmax+0.01*xstep,xstep))
ymax = np.max(probs)
ymin = 0
ystep = (ymax-ymin)/2
plt.yticks(np.arange(ymin,ymax+0.01*ystep,ystep))
#plt.grid(b=True)
for axis in ['top', 'bottom', 'left', 'right']:
   ax.spines[axis].set_linewidth(4)

saveFile = 'psu_ua.png'
plt.savefig(saveFile)
print('The plot has been saved to ' + saveFile)

plt.show()

