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
# This file contains Python codes that read matlabsp2.m created by
# PSUADE and display the X-Y scatter plot
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
print('This function takes the matlabsp2.m file created by PSUADE')
print('and plot the scatter plot using matplotlib.')
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   spfile = raw_input('Enter name of matlabsp2.m-like file : ')
else:
   spfile = input('Enter name of matlabsp2.m-like file : ')

# first search for dimensions in X
try:
   with open(spfile, 'r') as infile:
      # search for X = [
      while 1:
         lineIn = infile.readline()
         sInd = lineIn.find('X = [')
         if sInd >= 0:
            break
      # count until ]
      count = 0
      while 1:
         lineIn = infile.readline()
         sInd = lineIn.find('];')
         if sInd >= 0:
            break
         count = count + 1
except:
   print('ERROR: ' + spfile + ' file not found.')
   sys.exit(0)

# now read X matrix 
xlabel = 'X'
ylabel = 'Y'
title = 'title'
with open(spfile, 'r') as infile:
   # search for X = [
   while 1:
      lineIn = infile.readline()
      sInd = lineIn.find('X = [')
      if sInd >= 0:
         break
   # read in X
   X = np.zeros([3, count])
   for ii in range(count):
      lineIn = infile.readline()
      cols = lineIn.split()
      if len(cols) < 3:
         print('ERROR: in reading matlabsp2.m')
      for jj in range(3):
         X[jj][ii] = float(cols[jj])

# plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[0], X[1], X[2], marker='x')
xlabel = 'X'
ylabel = 'Y'
zlabel = 'Z'
ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
ax.set_zlabel(zlabel, fontsize=14, fontweight='bold')
title = 'Scatter Plot'
plt.title(title, fontsize=14, fontweight='bold')
#plt.grid(b=True)
for axis in ['top', 'bottom', 'left', 'right']:
   ax.spines[axis].set_linewidth(4)
plt.show()
plt.savefig('psu_sp2.png')
print('Scatter plot file is now in psu_sp2.png')

