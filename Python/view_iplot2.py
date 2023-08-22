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
# This file contains Python codes that read matlabiplt2.m created by
# PSUADE and display the 2-input scatter plot using matplotlib
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
print('This function takes the matlabiplt2.m file created by PSUADE')
print('and plot the scatter plot using matplotlib.')
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   ipfile = raw_input('Enter name of matlabiplt2.m-like file : ')
else:
   ipfile = input('Enter name of matlabiplt2.m-like file : ')

# first search for dimensions in X
try:
   with open(ipfile, 'r') as infile:
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
   print('ERROR: ' + ipfile + ' file not found.')
   sys.exit(0)

if count == 0:
   print('ERROR: no data found in ' + ipfile + ' file.')
   sys.exit(0)

# now read X matrix 
with open(ipfile, 'r') as infile:
   # search for X = [
   while 1:
      lineIn = infile.readline()
      sInd = lineIn.find('X = [')
      if sInd >= 0:
         break
   # read in X
   X = np.zeros([2, count])
   for ii in range(count):
      lineIn = infile.readline()
      cols = lineIn.split()
      if len(cols) < 3:
         print('ERROR: in reading ' + ipfile)
      for jj in range(2):
         X[jj][ii] = float(cols[jj])

# plot
fig, ax = plt.subplots()
plt.scatter(X[0], X[1], s=20, marker='*')
xlabel = 'X'
plt.xlabel(xlabel, fontsize=14, fontweight='bold')
ylabel = 'Y'
plt.ylabel(ylabel, fontsize=14, fontweight='bold')
title = '2-Input Scatter Plot'
plt.title(title, fontsize=14, fontweight='bold')
for axis in ['top', 'bottom', 'left', 'right']:
   ax.spines[axis].set_linewidth(6)
#plt.grid(visible=True)

plt.savefig('psu_iplt2.png')
print('2-Input Scatter plot is now in psu_iplt2.png')

plt.show()

