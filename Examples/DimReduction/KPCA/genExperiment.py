#!/bin/python
import os
import sys
import string
import shutil
import glob
import numpy as np

#######################################################
# select which to use
#======================================================
def selectSnapshot(snapFileIn, snapFileOut, num):
   # read sample input (in low dimension)
   infile = open(snapFileIn, "r")
   # skip the the num line
   skipLine = num - 1
   if (skipLine > 0):
      for ii in range(skipLine):
         lineIn = infile.readline()
   lineIn = infile.readline()
   outfile = open(snapFileOut, "w")
   outfile.write(lineIn)
   infile.close()
   outfile.close()
   return

#######################################################
# create MeasurementData
#======================================================
def genOneExperiment(snapFile):
   infile = open(snapFile, "r")
   pFile = "snap.WF"
   outfile = open(pFile, "w")
   while 1:
      lineIn  = infile.readline()
      nCols = lineIn.split()
      if len(nCols) == 1:
         ddata = float(nCols[0])
         ddata = np.exp(ddata)
         outfile.write("%e\n" % ddata)
      else:
         for ii in range(len(nCols)):
            ddata = float(nCols[ii])
            outfile.write("%e\n" % np.exp(ddata))
         break
   infile.close()
   outfile.close()

   sysCmd = "/home/tong10/Projects/Geocentric/solvers/"
   sysCmd = sysCmd + "adjoint-elasticity-v2/bin/"
   sysCmd = sysCmd + "adjoint.2016 2 parameters.txt "
   sysCmd = sysCmd + pFile + " " + pFile
   sysCmd = sysCmd + " gradFile costFile > log"
   os.system(sysCmd)
   os.remove("costFile")
   os.remove("gradFile")
   os.remove("log")
   os.remove("snap.WF")
   os.rename("solution.vec", "MeasurementData")
   return

#######################################################
# Main program file 
#######################################################

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
geoExe = None
with open('psTemp','r') as infile:
   geoExe = infile.readline()
geoExe = geoExe[0:len(geoExe)-1]
if os.path.isfile('psTemp'):
   os.remove('psTemp')
if not os.access(geoExe, os.X_OK):
   print('ERROR: Geocentric executable not found.')
   sys.exit(0)

# get ensemble snapshots
snapFileIn = input("Name of snapshot ensemble file: ")
if not os.path.isfile(snapFileIn):
  print("File ", snapFileIn, " does not exist")
  exit()

# get KPCA model
kpcaModelFile = input("Name of KPCA model file : ")
if not os.path.isfile(kpcaModelFile):
  print("File ", kpcaModelFile, " does not exist")
  exit()

snapNum = input("Which snapshot to be the solution (1-based): ")
snapNum = int(snapNum)

print("Select Snapshot %d ==> Snapshot%d" % (snapNum, snapNum))
snapFileOut = "Snapshot" + str(snapNum) + ".1"
snap = selectSnapshot(snapFileIn, snapFileOut, snapNum)

# run 10 iterations forward/backward because the source
# image in MCMC is not perfect (smooth edge) and we do
# not want to compare the recovered image against the
# perfect one (rough edges)
for ii in range(10):
  print("Iteration " + str(ii+1))
  outfile = open("ps.script", "w")
  outfile.write("kpca_forward\n")
  outfile.write("%s\n" % kpcaModelFile)
  inpFileName = " Snapshot" + str(snapNum) + "." + str(ii+1)
  outfile.write("%s\n" % inpFileName)
  outFileName = " SnapshotXi" + str(snapNum) + "." + str(ii+1)
  outfile.write("%s\n" % outFileName)
  outfile.write("q\n")
  outfile.close()
  sysCmd = psuadeExe
  sysCmd = sysCmd + " < ps.script > log"
  print("    ForwardKPCA")
  os.system(sysCmd)
  os.remove("log")
  os.remove("ps.script")

  print("    InverseKPCA")
  outfile = open("ps.script", "w")
  outfile.write("kpca_inverse\n")
  outfile.write("%s\n" % kpcaModelFile)
  inpFileName = " SnapshotXi" + str(snapNum) + "." + str(ii+1)
  outfile.write("%s\n" % inpFileName)
  outFileName = " Snapshot" + str(snapNum) + "." + str(ii+2)
  outfile.write("%s\n" % outFileName)
  outfile.write("q\n")
  outfile.close()
  sysCmd = psuadeExe
  sysCmd = sysCmd + " < ps.script > log"
  os.system(sysCmd)
  os.remove("log")
  os.remove("ps.script")

dfile = "Snapshot" + str(snapNum) + ".10" 
sfile = "Snapshot" + str(snapNum) 
shutil.copy(dfile,sfile)
dfile = "SnapshotXi" + str(snapNum) + ".10" 
sfile = "SnapshotXi" + str(snapNum) 
shutil.copy(dfile,sfile)

print("Generate experimental data in MeasurementData")
snapFile = "Snapshot" + str(snapNum)
genOneExperiment(snapFile)

dfile = "SnapshotXi" + str(snapNum) + ".10" 
if os.path.isfile(dfile):
   os.rename(dfile, 'SrcSnapshotXi')
dfile = "Snapshot" + str(snapNum) + ".10" 
if os.path.isfile(dfile):
   os.rename(dfile, 'SrcSnapshot')

fileList = glob.glob('Snapshot*')
for dfile in fileList:
   try:
      os.remove(dfile)
   except OSError:
      print('ERROR while deleting Snapshot files.')
if os.path.isfile('parameters.txt'):
   os.remove('parameters.txt')

