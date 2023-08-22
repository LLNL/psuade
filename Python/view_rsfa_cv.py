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
# This file contains Python codes that read RSFA_CV_err.m created by
# PSUADE and display the parity plot using matplotlib
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

# get RSFA_CV_err.m-like file
print('************************************************************')
print('This function extracts information from the RSFA_CV_err.m')
print('file (or the equivalent) and generates a parity plot.')
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   rsfile = raw_input('Enter name of the RSFA_CV_err.m-like file : ')
else:
   rsfile = input('Enter name of the RSFA_CV_err.m-like file : ')
if not os.path.isfile(rsfile):
   print('ERROR: data file not found.')
   sys.exit(1)

# first read the entire file
try:
   with open(rsfile, 'r') as infile:
      allLines = infile.readlines()
except:
   print('ERROR: fail to read file ' + rsfile)
   sys.exit(1)

# search for and read A matrix
lineNum = 0
while lineNum < len(allLines):
   lineIn = allLines[lineNum]
   sInd = lineIn.find('A = [')
   if sInd >= 0:
      XArray = []
      YArray = []
      lineNum = lineNum + 1
      while 1:
         lineIn = allLines[lineNum]
         cols = lineIn.split()
         if cols[0] == '];':
            break
         else:
            XArray.append(float(cols[1]))
            YArray.append(float(cols[2]))
            lineNum = lineNum + 1
   lineNum = lineNum + 1

# and then display parity plot  
fig, ax = plt.subplots()
plt.scatter(XArray, YArray, s=10, marker='*')
xlabel = 'True Value'
plt.xlabel(xlabel, fontsize=14, fontweight='bold')
ylabel = 'Predicted Value'
plt.ylabel(ylabel, fontsize=14, fontweight='bold')
title = 'Response Surface Fit Parity Plot'
plt.title(title, fontsize=14, fontweight='bold')
for axis in ['top', 'bottom', 'left', 'right']:
   ax.spines[axis].set_linewidth(6)
#plt.grid(b=True)

plt.savefig('psu_rsfa_cv.png')
print('RS parity plot file is now in psu_rsfa_cv.png')
plt.show()

