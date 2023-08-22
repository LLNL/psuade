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
# This file contains Python codes that read matlabaeua.m created by
# PSUADE and display the p-box using matplotlib
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

# get matlabaeua.m-like file
print('************************************************************')
print('This function extracts information from the matlabaeua.m')
print('file (or the equivalent) and plots a number of cumulative')
print('distributions that form a p-box.')
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   aefile = raw_input('Enter name of the matlabaeua.m-like file : ')
else:
   aefile = input('Enter name of the matlabaeua.m-like file : ')
if not os.path.isfile(aefile):
   print('ERROR: data file not found.')
   sys.exit(1)

# first read the entire file
try:
   with open(aefile, 'r') as infile:
      allLines = infile.readlines()
except:
   print('ERROR: fail to read file ' + aefile)
   sys.exit(1)

# first search for the vector 
lineNum = 0
fig, ax = plt.subplots()
while lineNum < len(allLines):
   lineIn = allLines[lineNum]
   sInd = lineIn.find(' = [')
   if sInd >= 0:
      print('INFO: CDF array found.')
      Y1Array = []
      lineNum = lineNum + 1
      while 1:
         lineIn = allLines[lineNum]
         cols = lineIn.split()
         if cols[0] == '];':
            break
         else:
            Y1Array.append(float(cols[0]))
            lineNum = lineNum + 1
      # display
      leng = len(Y1Array)
      YY = np.arange(leng) / (leng-1)   
      Y1Array = np.sort(Y1Array)
      plt.plot(Y1Array, YY)
   lineNum = lineNum + 1

for axis in ['top', 'bottom', 'left', 'right']:
   ax.spines[axis].set_linewidth(6)
#plt.grid(visible=True)
xlabel = 'Output Values'
plt.xlabel(xlabel, fontsize=14, fontweight='bold')
ylabel = 'Cumulative Probability'
plt.ylabel(ylabel, fontsize=14, fontweight='bold')
title = 'p-Box of CDFs from Aleatory-epistemic Uncertainty Analysis'
plt.title(title, fontsize=14, fontweight='bold')

plt.savefig('psu_aeua.png')
print('p-box plot file is now in psu_aeua.png')
plt.show()
   
