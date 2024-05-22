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
# by PSUADE and display the 1D plots for the H(Y|X) that are used
# to compute E[H(Y|X)] 
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
print('This function extracts information from the rsent2_data.m')
print('------------------------------------------------------------')
rsfile = "rsent2_data.m"

# first read all data
try:
   with open(rsfile, 'r') as infile:
      allLines = infile.readlines()
except:
   print('ERROR: problem reading ' + rsfile)
   sys.exit(0)

# count how many sets of data
count = 0
lineNum = 0
while lineNum < len(allLines):
    lineIn = allLines[lineNum]
    sInd = lineIn.find('IP')
    if sInd >= 0:
        count = count + 1
    lineNum = lineNum + 1
print('Number of plots = ' + str(count))

# read and plot
fig, ax = plt.subplots()
lineNum = 0
current = 0
while lineNum < len(allLines):
    lineIn = allLines[lineNum]
    sInd = lineIn.find('IP')
    if sInd >= 0:
        current = current + 1
        plt.subplot(count, 1, current)
        lineNum = lineNum + 1
        XArray = []
        YArray = []
        while 1:
            lineIn = allLines[lineNum]
            cols = lineIn.split()
            ncol = len(cols)
            if cols[0] == '];':
               break
            elif ncol > 1:
               XArray.append(float(cols[0]))
               YArray.append(float(cols[ncol-1]))
               lineNum = lineNum + 1

        print('Ymax = ' + str(np.max(YArray)))
        plt.plot(XArray, YArray, 'b*', linewidth=2)
        for axis in ['top', 'bottom', 'left', 'right']:
           ax.spines[axis].set_linewidth(4)
        xlabel = 'X'
        ylabel = 'Conditional entropy'
        plt.xlabel(xlabel, fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        title = 'Conditional Entropy (for computing E[H(Y|X)])' + str(current)
        plt.title(title, fontsize=14, fontweight='bold')
    lineNum = lineNum + 1

plt.show()

