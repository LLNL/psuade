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
# This file contains Python codes that read matlabrsplot1.m created 
# by PSUADE and display the 1D response surface plot using matplotlib
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

# get matlabrsplot1.m-like file
print('************************************************************')
print('This function extracts information from the matlabrsplot1.m')
print('file (or same but renamed file) and creates a 2D (1-input,')
print('1-output) response surface plot.')
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   rsfile = raw_input('Enter name of the matlabrsplot1.m-like file : ')
else:
   rsfile = input('Enter name of the matlabrsplot1.m-like file : ')

# first search for dimensions in X and A
try:
   with open(rsfile, 'r') as infile:
      allLines = infile.readlines()
   xlabel = 'X'
   ylabel = 'Y'
   lineNum = 0
   while lineNum < len(allLines):
      lineIn = allLines[lineNum]
      sInd = lineIn.find('A = [')
      if sInd >= 0:
         AArray = []
         lineNum = lineNum + 1
         while 1:
            lineIn = allLines[lineNum]
            cols = lineIn.split()
            if cols[0] == '];':
               break
            else:
               AArray.append(float(cols[0]))
               lineNum = lineNum + 1
      sInd = lineIn.find('X = [')
      if sInd >= 0:
         XArray = []
         lineNum = lineNum + 1
         while 1:
            lineIn = allLines[lineNum]
            cols = lineIn.split()
            if cols[0] == '];':
               break
            else:
               XArray.append(float(cols[0]))
               lineNum = lineNum + 1
      sInd = lineIn.find('xlabel')
      if sInd >= 0:
         for kk in range(len(lineIn)):
            if lineIn[sInd+kk+8] == "'":
               break
         xlabel = lineIn[sInd+8:sInd+kk+8]
      sInd = lineIn.find('ylabel')
      if sInd >= 0:
         for kk in range(len(lineIn)):
            if lineIn[sInd+kk+8] == "'":
               break
         ylabel = lineIn[sInd+8:sInd+kk+8]
      lineNum = lineNum + 1
except:
   print('ERROR: problem reading ' + rsfile)
   sys.exit(0)
 
# plot
fig, ax = plt.subplots()
plt.plot(XArray, AArray, 'b-', linewidth=2)
for axis in ['top', 'bottom', 'left', 'right']:
   ax.spines[axis].set_linewidth(4)
#plt.grid(b=True)
plt.xlabel(xlabel, fontsize=14, fontweight='bold')
plt.ylabel(ylabel, fontsize=14, fontweight='bold')
title = 'One-Input-One-Output Response Surface Plot'
plt.title(title, fontsize=14, fontweight='bold')
plt.savefig('psu_rs1.png')
print('RS1 plot file is now in psu_rs1.png')

plt.show()

