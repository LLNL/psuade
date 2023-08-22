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
# This file contains Python codes that read MCMCPostSample created by
# PSUADE and display the posterior histogram and heat plots
# *******************************************************************
import os
import sys
import numpy as np 
import matplotlib.pyplot as plt

# Read mcmc_predict output file ==> [Pmeans,Pstdvs,Pmins,Pmaxs]
print('************************************************************')
print("This function takes the mcmcpredict.out file from PSUADE's")
print('mcmc_predict command, compares it to another user-provided')
print('sample file (same sample size) with the true data, and')
print('creates a parity plot of true versus predicted results with')
print('uncertainty bounds.')
print('------------------------------------------------------------')
print('In order to run this function, users are to enter: ')
print('1. a mcmc_predict output file (and do not change it)')
print('2. a sample file with true outputs (format: MCMC experiment file')
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   predictfile = raw_input('mcmc_predict output file : ')
else:
   predictfile = input('mcmc_predict output file : ')

if not os.path.isfile(predictfile):
   print('ERROR: file ' + predictfile + ' not found.')
   sys.exit(0)

try:
   with open(predictfile, 'r') as infile:
      predictData = infile.readlines();
      nlines = len(predictData) - 1
      Pmeans = np.zeros(nlines)
      Pstdvs = np.zeros(nlines)
      Pmins  = np.zeros(nlines)
      Pmaxs  = np.zeros(nlines)
      nPdata = 0
      for ii in range(nlines):
         lineIn = predictData[ii+1]
         cols = lineIn.split()
         if len(cols) == 4 and (cols[0][0] != '#' or cols[0][0] != 'm'):
            Pmeans[nPdata] = float(cols[0])
            Pstdvs[nPdata] = float(cols[1])
            Pmins[nPdata]  = float(cols[2])
            Pmaxs[nPdata]  = float(cols[3])
            nPdata = nPdata + 1 
except:
   print('ERROR: when reading mcmc_predict output file.')
   print('       Check format (did you change it after mcmc_predict?)')
   sys.exit(1)
print('Number of prediction samples = ' + str(nPdata))

# Read actual sample ==> std format
print('************************************************************')
print('Next, enter a validation sample file that has true outputs.')
print('This file should be the same as the file you enter when you')
print('run rsmcmc. You will be asked to select the column to be used')
print('for comparison.')
print('This file should have ' + str(nPdata) + ' sample points.')
print('------------------------------------------------------------')
if sys.version_info[0] < 3:
   expfile = raw_input('Enter validation sample file : ')
else:
   expfile = input('Enter validation sample file : ')

if not os.path.isfile(expfile):
   print('ERROR: file ' + expfile + ' not found.')
   sys.exit(0)

try:
   with open(expfile, 'r') as infile:
      expData = infile.readlines();
      nlines = len(expData)
      for ii in range(nlines):
         lineIn = expData[ii]
         cols = lineIn.split()
         if len(cols) > 0 and cols[0][0] != '#':
            nExp = int(cols[0])
            if nExp != nPdata:
               print('ERROR: invalid experiment file data size') 
               print('       actual   data size = ' + str(nExp))
               print('       expected data size = ' + str(nPdata))
               sys.exit(1)
            else:
               nIns  = int(cols[2])
               nOuts = int(cols[1]) * 2
               break
      nextLine = ii
      print('Number of sample  in this file = ' + (str(nExp)))
      print('Number of inputs  in this file = ' + (str(nIns)))
      print('Number of outputs in this file = ' + (str(nOuts)))
      if nOuts > 1:
         print('Which output to select ? ')
         if sys.version_info[0] < 3:
           outIDStr = raw_input('Select output : ')
         else:
           outIDStr = input('Select output : ')
         outID = int(outIDStr) 
      else:
         outID = 1
      if outID < 1 or outID > nOuts:
         print('ERROR: wrong output selection')
         sys.exit(1)
      Emeans = np.zeros(nPdata)
      count = 0
      InpData = np.zeros([nPdata, nIns])
      for ii in range(nlines):
         if ii >= nextLine:
            lineIn = expData[ii]
            cols = lineIn.split()
            if len(cols) == nIns + nOuts + 1:
               Emeans[count] = float(cols[nIns+outID])
               for kk in range(nIns):
                  InpData[count][kk] = float(cols[kk+1])
               count = count + 1
except:
   print('ERROR: when reading validation sample file.')
   print('       Check format (did you change it after rsmcmc?)')
   sys.exit(1)

###########################################################
# parity plot
# ---------------------------------------------------------
fig, ax = plt.subplots()
plt.subplot(1,2,1)
plt.plot(Emeans[0:nPdata],Pmeans[0:nPdata],'kx')

# the following code may be used for specific purposes
# for ii in range(nPdata):
#    if InpData[ii][2] > 300:
#       plt.plot(Emeans[ii],Pmeans[ii],'rx',markersize=12)
#    else:
#       plt.plot(Emeans[ii],Pmeans[ii],'bx',markersize=12)

for ii in range(nPdata):
   xx = [Emeans[ii], Emeans[ii]]
   yy = [Pmeans[ii]-Pstdvs[ii]*2, Pmeans[ii]+Pstdvs[ii]*2] 
   plt.plot(xx,yy,'b-')
#  if InpData[ii][2] > 300:
#     plt.plot(xx,yy,'r-')
#  else:
#     plt.plot(xx,yy,'b-')
dmax1 = np.max(Emeans[0:nPdata])
dmax2 = np.max(Pmeans[0:nPdata])
dmin1 = np.min(Emeans[0:nPdata])
dmin2 = np.min(Pmeans[0:nPdata])
dmin  = np.min([dmin1, dmin2])
dmax  = np.max([dmax1, dmax2])
plt.plot([dmin, dmax],[dmin, dmax],'b-')
dstep = (dmax - dmin) / 2
plt.xticks(np.arange(dmin,dmax+0.01*dstep,dstep),fontsize=13,fontweight='bold')
plt.yticks(np.arange(dmin,dmax+0.01*dstep,dstep),fontsize=13,fontweight='bold')
plt.xlim([dmin, dmax])
plt.ylim([dmin, dmax])
plt.xlabel('Experiment',fontsize=13,fontweight='bold')
plt.ylabel('Prediction',fontsize=13,fontweight='bold')
#plt.grid(b=True)
# thicken the border lines
for axis in ['top', 'bottom', 'left', 'right']:
   ax.spines[axis].set_linewidth(6)
tStr = 'Parity Plot'
plt.title(tStr,fontsize=14,fontweight='bold')

###########################################################
# histogram plot of error
# ---------------------------------------------------------
# put the figure on the right
plt.subplot(1,2,2)
# generate histogram of errors with 10 bins
nhist = 10
counts, bins = np.histogram(Emeans-Pmeans,nhist)
probs = 1.0 * counts / len(Emeans)
plt.hist(bins[:-1], bins, weights=probs)
# compute error mean
dmean = 0
for ii in range(len(Emeans)):
   if Emeans[ii] > 0:
      dmean = dmean + (Emeans[ii]-Pmeans[ii])/Emeans[ii]
   elif Emeans[ii] < 0:
      dmean = dmean - (Emeans[ii]-Pmeans[ii])/Emeans[ii]
   else:
      dmean = dmean - (Emeans[ii]-Pmeans[ii])
dmean = dmean / len(Emeans)
print('(Experiment - Prediction) Error mean (normalized) = ' + \
      str(dmean))

# draw a red line at error = 0
print('Red line : error = 0')
plt.ylim([0,np.max(probs)])
xx = [0, 0]
yy = [0, np.max(probs)]
plt.plot(xx,yy,'r-',linewidth=2)
xmax = np.max(Emeans-Pmeans)
xmin = np.min(Emeans-Pmeans)
plt.xlim([xmin,xmax])
# only use 3 xticks to prevent crowdiness
xstep = (xmax-xmin)/2
plt.xticks(np.arange(xmin,xmax+0.01*xstep,xstep),fontsize=13,fontweight='bold')
ymax = np.max(probs)
ymin = 0
plt.ylim([ymin, ymax])
ystep = (ymax-ymin)/2
# only use 3 yticks to prevent crowdiness
plt.yticks(np.arange(ymin,ymax+0.01*ystep,ystep),fontsize=13,fontweight='bold')
plt.xlabel('Experimental - Prediction mean',fontsize=13,fontweight='bold')
plt.ylabel('Probability',fontsize=13,fontweight='bold')
tStr = 'Prediction Error Statistics'
plt.title(tStr,fontsize=14,fontweight='bold')
#plt.grid(b=True)
# thicken the border lines
for axis in ['top', 'bottom', 'left', 'right']:
   ax.spines[axis].set_linewidth(6)

#plt.axis('off')
saveFile = 'psu_mcmcpredict.png'
plt.savefig(saveFile)
print('The plot has been saved to ' + saveFile)
plt.show()

