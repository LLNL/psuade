#!/usr/bin/python
# NOTE: Make sure the above Python path is correct.
import os
import sys
import string
import shutil
import math
import numpy as np
#######################################################
# USER SPECIFIC SECTION 
#======================================================
modelFile   = "kpcaModel"
expDataFile = "MeasurementData"
simOutFile  = "solution.vec"
simulator   = None
psuadeExe   = None
#======================================================
# NOTE: The following file is for storing the inputs
#       to the simulation (maybe high-dimensional) and
#       after dimension recovery from calibration inputs.
postXFile   = "PosteriorX"
#======================================================
# NOTE: The following file is for storing the simulation
#       outputs every time it is run (each line contains
#       outputs for one simulation.
postYFile   = "PosteriorY"
#======================================================
# NOTE: The following flag specifies whether postXFile
#       and postYFile will be written to (default 1). 
postFlag = 1
#======================================================
# END USER SPECIFIC SECTION 
#######################################################
#######################################################
# Obtain field dimension and reduced dimension 
#======================================================
def getInfoFromModel():
   infile = open(modelFile, "r")
   lineIn = infile.readline()
   lineIn = infile.readline()
   nCols = lineIn.split()
   fieldDim = eval(nCols[0])
   rdim = 0
   while 1:
      lineIn  = infile.readline()
      if lineIn == "":
         break
      nCols = lineIn.split()
      if (len(nCols) > 2):
         if nCols[0] == "rdim":
            rdim = eval(nCols[2])
   infile.close()
   return [fieldDim, rdim]
#######################################################
# perform KPCA inversion 
#======================================================
def performInversion(inpFileName,outFileName,fieldDim):
   outfile = open("ps.script", "w")
   outfile.write("kpca_inverse\n")
   outfile.write("%s\n" % modelFile)
   outfile.write("%s\n" % inpFileName)
   outfile.write("psTmp\n")
   outfile.write("q\n")
   outfile.close()
   sysCmd = psuadeExe
   sysCmd = sysCmd + " < ps.script > log"
   os.system(sysCmd)
   os.remove("log")
   os.remove("ps.script")
   infile  = open("psTmp", "r")
   outfile = open(outFileName, "w")
   outX    = open("PosteriorX", "a")
   while 1:
      lineIn  = infile.readline()
      if lineIn == "":
         break
      nCols = lineIn.split()
      if len(nCols) == 1:
         ddata = float(nCols[0])
         ddata = np.exp(ddata)
         outfile.write("%e\n" % ddata)
         outX.write("%e " % ddata)
      else:
         for ii in range(len(nCols)):
            ddata = float(nCols[ii])
            ddata = np.exp(ddata)
            outfile.write("%e\n" % ddata)
            outX.write("%e " % ddata)
         break
   infile.close()
   outfile.close()
   outX.close()
   if os.path.isfile('psTmp'):
      os.remove('psTmp')

#######################################################
# run application program 
# NOTE: make sure the simulation call sequence is right
#       for your application.
#======================================================
def runSimulation(pFile):
   sysCmd = simulator + " 2 parameters.txt " + pFile + ' '
   sysCmd = sysCmd + expDataFile + " grad cost > log"
   os.system(sysCmd)
   return
#######################################################
# create likelihood 
#======================================================
def genOutputFile(outFileName, likelihood):
   print("     COST = %e" % likelihood)
   oFile = open(outFileName, "w")
   oFile.write("%e\n" % likelihood)
   oFile.close()
   if (postFlag == 1): 
      aFile = open(postXFile, "a")
      ddata = - 2.0 * math.log(likelihood)
      aFile.write("%e\n" % ddata)
      aFile.close()
   return
#######################################################
# MAIN 
#======================================================
# first check to see if psuade and simulation model 
# exist or not
os.system('which psuade > psTemp')
psuadeExe = None
with open('psTemp','r') as infile:
   psuadeExe = infile.readline()
psuadeExe = psuadeExe[0:len(psuadeExe)-1]
if os.path.isfile('psTemp'):
   os.remove('psTemp')
if not os.access(psuadeExe, os.X_OK):
   print('ERROR: psuade executable not found.')
   sys.exit(0)

os.system('which adjoint.2016 > psTemp')
simulator = None
with open('psTemp','r') as infile:
   simulator = infile.readline()
simulator = simulator[0:len(simulator)-1]
if os.path.isfile('psTemp'):
   os.remove('psTemp')
if not os.access(simulator, os.X_OK):
   print('ERROR: Geocentric executable not found.')
   sys.exit(0)

if len(sys.argv) < 3:
   print("This script requires 2 parameters: ")
   print("(1) the sample in U space")
   print("(2) the file to which likelihood is written")
   exit()

# NOTE: The following line is application specific.
#       fieldDim is just the dimension of the inputs.
#       rdim is the dimension of the reduced space.
fieldDim, rdim = getInfoFromModel()
# input file has sample from inference (in reduced space)
inpFile = sys.argv[1]

# perform inversion (from reduced to physical space)
simInpFile = "reconstructedDataFile"
performInversion(inpFile, simInpFile, fieldDim)

# run simulation
runSimulation(simInpFile)

# compute likelihood
infile1 = open(simOutFile, "r")
infile2 = open(expDataFile, "r")
if (postFlag == 1): 
   outfile = open(postYFile, "a")
rmsErr = 0;
# NOTE: it is assume simulation output and measurement
#       files have the same length.
d1 = np.zeros(4050)
d2 = np.zeros(4050)
ind = 0
while 1:
   lineIn = infile1.readline()
   if lineIn == "":
      break
   nCols  = lineIn.split()
   ddata  = eval(nCols[0])
   d1[ind] = ddata
   if (postFlag == 1): 
      outfile.write("%e " % ddata)
   lineIn = infile2.readline()
   nCols  = lineIn.split()
   d2[ind] = eval(nCols[0])
   ddata  = ddata - eval(nCols[0])
   rmsErr = rmsErr + ddata * ddata
   ind = ind + 1
rmsErr = 0.5 * rmsErr
infile1.close()
infile2.close()
rmsErr = 0
for ii in range(22):
   for jj in range(22):
      ind = (ii * 2 + 1) * 45 + jj * 2 + 1 
      ddata  = d1[ind] - d2[ind]
      rmsErr = rmsErr + ddata * ddata
rmsErr = 0.5 * rmsErr

if (postFlag == 1): 
  outfile.write("\n")
  outfile.close()
print("     RMS = ", rmsErr)
likelihood = math.exp(-rmsErr)

costFileName = sys.argv[2]
genOutputFile(costFileName, likelihood)
if os.path.isfile('grad'):
   os.remove('grad')
if os.path.isfile('cost'):
   os.remove('cost')
if os.path.isfile('log'):
   os.remove('log')
if os.path.isfile('reconstructedDataFile'):
   os.remove('reconstructedDataFile')

