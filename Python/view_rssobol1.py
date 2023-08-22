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
# This file contains Python codes that read matlabrssobol1.m created 
# by PSUADE and display the sensitivity results
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

# get matlabrssobol1.m-like file
print('************************************************************')
print('This function reads from a matlabrssobol1.m file and plots')
print('the sensitivity results.') 
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   datafile = raw_input('Enter matlabrssobol1.m-like file name : ')
else:
   datafile = input('Enter matlabrssobol1.m-like file name : ')
if not os.path.isfile(datafile):
   print('ERROR: data file not found.')
   sys.exit(1)

try:
   with open(datafile, 'r') as infile:
      allLines = infile.readlines()
except:
   print('ERROR: in reading ' + datafile)
   sys.exit(0)

#------------------------------------------------------------------
# read matlabrssobol1.m created by psuade
#------------------------------------------------------------------
with open(datafile, 'r') as infile:
   doneFlag = 0
   while doneFlag < 2:
      lineIn = infile.readline()
      cols = lineIn.split()
      if len(cols) >= 2 and cols[0] == 'Mids' and cols[2] == '[':
         meanData = []
         nInputs = 0
         while 1:
            lineIn = infile.readline()
            cols = lineIn.split()
            if len(cols) > 0 and cols[0] == '];':
               break
            elif len(cols) > 0:
               meanData.append(float(cols[0]))
               nInputs = nInputs + 1
         doneFlag = doneFlag + 1
      if len(cols) >= 2 and cols[0] == 'Str' and cols[1] == '=':
         varNames = []
         tlist = cols[2].split(',')
         for ii in range(nInputs):
            vname = tlist[ii]
            if ii == 0:
               tname = vname[2:len(vname)-1]
            elif ii == nInputs-1:
               tname = vname[1:len(vname)-3]
            else:
               tname = vname[1:len(vname)-1]
            varNames.append(tname)
         doneFlag = doneFlag + 1

#------------------------------------------------------------------
# plot
#------------------------------------------------------------------
if len(varNames) == nInputs and len(meanData) == nInputs:
   fig, ax = plt.subplots()
   plt.bar(range(nInputs),meanData, 0.5)
   plt.xticks(range(nInputs), varNames, rotation=45, fontweight='bold')
   plt.ylabel("Sobol' Sensitivity Index", fontsize=13, fontweight='bold')
   plt.title("Sobol' Main Effect Plot", fontsize=15, fontweight='bold')
   for axis in ['top', 'bottom', 'left', 'right']:
      ax.spines[axis].set_linewidth(4)
   #plt.grid(b=True)
   saveFile = 'psu_rssobol1.png'
   plt.savefig(saveFile)
   print('The plot has been saved to ' + saveFile)
   plt.show()
else:
   print('ERROR: Insufficient information for plotting')

