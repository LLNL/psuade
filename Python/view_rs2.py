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
# This file contains Python codes that read matlabrsplot2.m created 
# by PSUADE and display the 2D response surface plot using matplotlib
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

# get matlabrsplot2.m-like file
print('************************************************************')
print('This function extracts information from the matlabrsplot2.m')
print('file (or same but renamed file) and creates a 3D (2-input,')
print('1-output) response surface plot or a 2D contour plot if the')
print('MPL library is not available.')
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   rsfile = raw_input('Enter name of matlabrsplot2.m-like file : ')
else:
   rsfile = input('Enter name of matlabrsplot2.m-like file : ')

# first search for dimensions in X and Y
try:
   with open(rsfile, 'r') as infile:
      while 1:
         lineIn = infile.readline()
         sInd = lineIn.find('reshape')
         if sInd >= 0:
            for kk1 in range(len(lineIn)):
               if lineIn[sInd+kk1] == ",":
                  break
            kk1 = kk1 + 1
            for kk2 in range(len(lineIn)):
               if lineIn[sInd+kk1+kk2] == ",":
                  break
            xStr = lineIn[sInd+kk1:sInd+kk1+kk2]
            kk2 = kk2 + 1
            for kk3 in range(len(lineIn)):
               if lineIn[sInd+kk1+kk2+kk3] == ")":
                  break
            yStr = lineIn[sInd+kk1+kk2:sInd+kk1+kk2+kk3]
            xdim = int(xStr)
            ydim = int(yStr)
            break
except:
   print('ERROR: problem reading ' + rsfile)
   sys.exit(0)
 
# now read A matrix and X and Y vectors
xlabel = 'X'
ylabel = 'Y'
title = 'title'
with open(rsfile, 'r') as infile:
   doneFlag = 0
   while doneFlag < 6:
      lineIn = infile.readline()
      cols = lineIn.split()
      if len(cols) >= 2 and cols[0] == 'A' and cols[2] == '[':
         vecData = np.zeros([xdim,ydim])
         for ii in range(xdim):
            for jj in range(ydim):
               lineIn = infile.readline()
               cols = lineIn.split()
               vecData[jj][ii] = float(cols[0])
         doneFlag = doneFlag + 1
      if len(cols) >= 2 and cols[0] == 'x' and cols[2] == '[':
         vecX = np.zeros(xdim)
         for ii in range(xdim):
            lineIn = infile.readline()
            cols = lineIn.split()
            vecX[ii] = float(cols[0])
         doneFlag = doneFlag + 1
      if len(cols) >= 2 and cols[0] == 'y' and cols[2] == '[':
         vecY = np.zeros(ydim)
         for ii in range(ydim):
            lineIn = infile.readline()
            cols = lineIn.split()
            vecY[ii] = float(cols[0])
         doneFlag = doneFlag + 1
      if len(cols) >= 1:
         sInd = lineIn.find('xlabel')
         if sInd >= 0:
            for kk in range(len(lineIn)):
               if lineIn[sInd+kk+8] == "'":
                  break
            xlabel = lineIn[sInd+8:sInd+kk+8]
            doneFlag = doneFlag + 1
         sInd = lineIn.find('ylabel')
         if sInd >= 0:
            for kk in range(len(lineIn)):
               if lineIn[sInd+kk+8] == "'":
                  break
            ylabel = lineIn[sInd+8:sInd+kk+8]
            doneFlag = doneFlag + 1
         sInd = lineIn.find('title')
         if sInd >= 0:
            for kk in range(len(lineIn)):
               if lineIn[sInd+kk+7] == "'":
                  break
            title = lineIn[sInd+7:sInd+kk+7]
            doneFlag = doneFlag + 1
# plot
fig = plt.figure(figsize=(5,5))
if MPL:
   ax = Axes3D(fig)
   ax.contourf(vecX, vecY, vecData, xdim)
   title = 'Two-Parameter Response Surface Plot'
else:
   fig, ax = subplots()
   plt.contourf(vecX, vecY, vecData, xdim)
   for axis in ['top', 'bottom', 'left', 'right']:
      ax.spines[axis].set_linewidth(4)
   title = 'Two-Parameter Response Surface Contour Plot'

plt.xlabel(xlabel, fontsize=14, fontweight='bold')
plt.ylabel(ylabel, fontsize=14, fontweight='bold')
plt.title(title, fontsize=14, fontweight='bold')
plt.savefig('psu_rs2.png')
print('RS2 plot file is now in psu_rs2.png')

plt.show()

