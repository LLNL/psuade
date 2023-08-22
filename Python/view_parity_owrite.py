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
# This file contains Python codes that read a data file created by 
# the owrite command and display the parity plot of 2 selected
# outputs
# *******************************************************************
import os
import sys
import numpy as np 
import matplotlib.pyplot as plt

# Read owrite file 
print('************************************************************')
print("This function takes a data file created by PSUADE's owrite")
print('command and uses 2 selected outputs to create a parity plot.')
print('(That is, a scatter plot with the X-axis representing one of')
print('the outputs and the Y-axis another. This command is useful') 
print('for visualizing the differences between 2 outputs. E.g. the')
print('X-axis may correspond to simulation outputs and the Y-axis')
print('the predicted output based on response surface interpolation.')
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   ofile = raw_input('Name of data file (owrite format) : ')
else:
   ofile = input('Name of data file (owrite format) : ')

if not os.path.isfile(ofile):
   print('ERROR: file ' + ofile + ' not found.')
   sys.exit(0)

try:
   with open(ofile, 'r') as infile:
      oData = infile.readlines();
      lineIn = oData[0]
      cols = lineIn.split()
      if len(cols) != 2:
         print('ERROR: when reading data file.')
         print('Note: The first line should be <nSamples, nOutputs>')
         sys.exit(0)
      nData = int(cols[0]) 
      nOuts = int(cols[1]) 
      if nData <= 1 or nOuts < 2:
         print('ERROR: when reading data file.')
         print('   nSamples = ' + str(nData) + ' (must be > 1)')
         print('   nOutputs = ' + str(nOuts) + ' (must be > 1)')
         sys.exit(0)

      print('There are ' + str(nOuts) + ' outputs.')
      if sys.version_info[0] < 3:
         XStr = raw_input('Which output should be used as X-axis ? ')
      else:
         XStr = input('Which output should be used as X-axis ? ')
      try:
         XInd = int(XStr)
      except:
         print('ERROR: Invalid input.')
         sys.exit(1)
      if XInd < 1 or XInd > nOuts:
         print('ERROR: Invalid input.')
         sys.exit(1)
      XInd = XInd - 1
      if sys.version_info[0] < 3:
         YStr = raw_input('Which output should be used as Y-axis ? ')
      else:
         YStr = input('Which output should be used as Y-axis ? ')
      try:
         YInd = int(YStr)
      except:
         print('ERROR: Invalid input.')
         sys.exit(1)
      if YInd < 1 or YInd > nOuts:
         print('ERROR: Invalid input.')
         sys.exit(1)
      YInd = YInd - 1

      sData  = np.zeros(nData)
      pData  = np.zeros(nData)
      for ii in range(nData):
         lineIn = oData[ii+1]
         cols = lineIn.split()
         if len(cols) != nOuts+1:
            print('ERROR: when reading data file.')
            print('   Line ' + str(ii+2) + ' : incorrect data')
            sys.exit(0)

         sData[ii] = float(cols[XInd+1])
         pData[ii] = float(cols[YInd+1])

except:
   print('ERROR: when reading the data file.')
   sys.exit(1)

###########################################################
# parity plot
# ---------------------------------------------------------
fig, ax = plt.subplots()
plt.subplot(1,1,1)
plt.plot(sData,pData,'kx',markersize=12)

dmax1 = np.max(sData)
dmax2 = np.max(pData)
dmin1 = np.min(sData)
dmin2 = np.min(pData)
dmin  = np.min([dmin1, dmin2])
dmax  = np.max([dmax1, dmax2])
plt.plot([dmin, dmax],[dmin, dmax],'b-')
dstep = (dmax - dmin) / 2
plt.xticks(np.arange(dmin,dmax+0.01*dstep,dstep),fontsize=13,fontweight='bold')
plt.yticks(np.arange(dmin,dmax+0.01*dstep,dstep),fontsize=13,fontweight='bold')
plt.xlim([dmin, dmax])
plt.ylim([dmin, dmax])
xStr = 'Output ' + str(XInd+1)
plt.xlabel(xStr,fontsize=13,fontweight='bold')
yStr = 'Output ' + str(YInd+1)
plt.ylabel(yStr,fontsize=13,fontweight='bold')
#plt.grid(b=True)
# thicken the border lines
for axis in ['top', 'bottom', 'left', 'right']:
   ax.spines[axis].set_linewidth(6)
tStr = 'Parity Plot'
plt.title(tStr,fontsize=14,fontweight='bold')
saveFile = 'parity.png'
plt.savefig(saveFile)
plt.show()
print('Parity plot can now be found in parity.png.')

