// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC. 
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team.
// All rights reserved.
//
// Please see the COPYRIGHT_and_LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public License.
//
// PSUADE is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License (as published by the Free Software
// Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Functions for the Generalized Morris one-at-a-time class 
//**/  (Latin hypercube idea not feasible due to column permutation)
// AUTHOR : CHARLES TONG
// DATE   : 2006
// ************************************************************************
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//**/using namespace std;

#include "sysdef.h"
#include "PsuadeUtil.h"
#include "pData.h"
#include "PsuadeData.h"
#include "Psuade.h"
#include "GMOATSampling.h"
#define PABS(x) ((x) > 0 ? (x) : -(x))

//*************************************************************************
//* Constructor
//*------------------------------------------------------------------------
GMOATSampling::GMOATSampling() : Sampling()
{
  samplingID_  = PSUADE_SAMP_GMOAT;
  //**/ default number of levels
  P_  = 4;
}

//*************************************************************************
//* Copy Constructor added by Bill Oliver
//*------------------------------------------------------------------------
GMOATSampling::GMOATSampling(const GMOATSampling & gms) : Sampling()
{
  samplingID_  = gms.samplingID_;
  vecInpLevels_ = gms.vecInpLevels_;
  vecInpSubset_ = gms.vecInpSubset_;
  matInitX_     = gms.matInitX_;
  printLevel_ = gms.printLevel_;
  samplingID_ = gms.samplingID_;
  nSamples_ = gms.nSamples_;
  nOutputs_ = gms.nOutputs_;
  randomize_ = gms.randomize_;
  P_ = gms.P_;
  nInputs_ = gms.nInputs_;
  nReplications_ = gms.nReplications_;
  vecLBs_ = gms.vecLBs_;
  vecUBs_ = gms.vecUBs_;
  vecSamInps_ = gms.vecSamInps_;
  vecSamOuts_ = gms.vecSamOuts_;
  vecSamStas_ = gms.vecSamStas_;
}

//*************************************************************************
//* destructor 
//*------------------------------------------------------------------------
GMOATSampling::~GMOATSampling()
{
}

//*************************************************************************
//* initialize the sampling data
//*------------------------------------------------------------------------
int GMOATSampling::initialize(int initLevel)
{
  int    ii, ii2, rr, ss, nReps, nn, index, idata, nlevels;
  int    kk1, kk2, maxSamples;
  int    maxReps=5000, base1, base2, nSub;
  double ddata, delta, rdata, maxDist;
  double dDist;
  char   *cString, winput1[500], winput2[500];
  char   partitionFile[500], pString[500], pString2[500];
  FILE   *fp;
  psIVector vecCounts;
  psMatrix  matBS, matBRan;
  psVector  vecXT, vecRanges;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  if (nSamples_ == 0)
  {
    printf("GMOATSampling::initialize ERROR - nSamples = 0.\n");
    exit(1);
  }
  if (nInputs_ == 0)
  {
    printf("GMOATSampling::initialize ERROR - input not set up.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ clean up and initialize
  //**/ ----------------------------------------------------------------
  nReps = nSamples_ / (nInputs_ + 1);
  if ((nReps * (nInputs_+1)) != nSamples_) 
  {
    printf("GMOATSampling : nSamples should be multiples of nInputs+1.\n");
    printf("                nSamples reset to be 10*(nInputs+1).\n");
    nSamples_ = 10 * (nInputs_ + 1);
  }
  if (initLevel != 0) return 0;

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 4)
  {
    printf("GMOATSampling::initialize: nSamples  = %d\n", nSamples_);
    printf("GMOATSampling::initialize: nInputs   = %d\n", nInputs_);
    printf("GMOATSampling::initialize: nOutputs  = %d\n", nOutputs_);
    printf("GMOATSampling: initialize: numLevels = %d\n", P_);
    if (randomize_ != 0)
         printf("GMOATSampling::initialize: randomize on\n");
    else printf("GMOATSampling::initialize: randomize off\n");
    for (ii2 = 0; ii2 < nInputs_; ii2++)
      printf("    GMOATSampling input %3d = [%e %e]\n", ii2+1,
             vecLBs_[ii2], vecUBs_[ii2]);
  }

  //**/ ----------------------------------------------------------------
  //**/ if nInputs > 100, use fast version
  //**/ ----------------------------------------------------------------
  if (nInputs_ > 100)
  {
    printf("GMOATSampling: nInputs > 100, use fast version.\n");
    initializeHighDimension();
    return 0;
  }

  //**/ ----------------------------------------------------------------
  //**/ set default number of levels
  //**/ ----------------------------------------------------------------
  if (nReps > maxReps) maxReps = nReps;
  vecInpLevels_.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++) vecInpLevels_[ii] = P_;

  //**/ ----------------------------------------------------------------
  //**/ adjust P for inputs
  //**/ ----------------------------------------------------------------
  cString = psConfig_.getParameter("GMOAT_P");
  if (cString != NULL)
  {
    sscanf(cString, "%s %s %d",winput1,winput2,&P_);
    P_ = P_ / 2 * 2;
    if (P_ <= 0 || P_ > 100) P_ = 4;
    printf("GMOATSampling: P set to %d (config)\n", P_);
    for (ii = 0; ii < nInputs_; ii++) vecInpLevels_[ii] = P_;
  }
  cString = psConfig_.getParameter("GMOAT_partition_file");
  if (cString != NULL)
  {
    sscanf(cString, "%s %s %s",winput1,winput2,partitionFile);
    printf("GMOATSampling: use GMOAT input partition file %s.\n",
           partitionFile);
    fp = fopen(partitionFile, "r");
    if (fp != NULL)
    {
      fscanf(fp, "%d", &ss);
      if (ss <= 0 || ss >= nInputs_)
      {
        printf("GMOATSampling: invalid GMOAT input partition file.\n");
        printf("              The first line should be nInputs.\n");
        fclose(fp);
        fp = NULL;
      }
      else
      {
        vecInpSubset_.setLength(nInputs_);
        for (ii = 0; ii < nInputs_; ii++) vecInpSubset_[ii] = 0;
        for (ii = 0; ii < ss; ii++)
        {
          fscanf(fp, "%d", &ii2);
          if (ii2 < 1 || ii2 > nInputs_)
          {
            printf("GMOATSampling: invalid input partition file.\n");
            printf("               invalid input index %d.\n", ii);
            if(fp != NULL) fclose(fp);
            fp = NULL;
            break;
          }
          else vecInpSubset_[ii2-1] = 1;
        }
      }
      if (fp != NULL) fclose(fp);
    }
  }
  if (psConfig_.SamExpertModeIsOn())
  {
    printf("GMOATSampling: default number of levels = %d.\n", P_);
    snprintf(pString,100,
             "Change number of levels for individual inputs ? (y or n) ");
    getString(pString, winput1);
    if (winput1[0] == 'y')
    {
      ii = nInputs_+1;
      snprintf(pString,100,
              "Which input to change ? (1 - %d, 0 to end) ",nInputs_);
      while (ii > 0)
      {
        ii = getInt(0, nInputs_, pString);
        if (ii == 0) break;
        snprintf(pString2,100,
                "Enter number of levels for input %d (2 - 1000) : ",ii);
        vecInpLevels_[ii-1] = getInt(2, 1000, pString2);
      }
    }
  }
  if (printLevel_ > 2) 
  {
    for (ii = 0; ii < nInputs_; ii++)
      printf("GMOATSampling: input %3d has %4d levels\n",ii+1,
             vecInpLevels_[ii]);
  }
 
  //**/ ----------------------------------------------------------------
  //**/ fetch input subset
  //**/ ----------------------------------------------------------------
  if (psConfig_.SamExpertModeIsOn())
  {
    printf("GMOAT has an option for an input partition file.\n");
    printf("It is used in case you need to separate the inputs\n");
    printf("into two sets so that for each MOAT path members of\n");
    printf("the selected subset will be varied first.\n");
    printf("This option is generally not needed, but maybe useful\n");
    printf("for parallel computing.\n");
    snprintf(pString,100,"Do you have a MOAT input partition file ? (y or n) ");
    getString(pString, winput1);
    if (winput1[0] == 'y')
      printf("GMOAT INFO: this option is available only in configFile.\n");
  }

  //**/ ---------------------------------------------------------------
  //**/ initialize initial points (use Latin hypercube, cannot use OA 
  //**/ since OA depends on nInputs) 
  //**/ ----------------------------------------------------------------
  int **iPtr2;
  maxLevels_ = 0;
  for (ii = 0; ii < nInputs_; ii++)
    if (vecInpLevels_[ii] > maxLevels_) maxLevels_ = vecInpLevels_[ii];
  maxLevels_ += maxReps;
  matInitX_.setFormat(2);
  matInitX_.setDim(nInputs_,maxLevels_);
  iPtr2 = matInitX_.getIMatrix2D();
  for (ii = 0; ii < nInputs_; ii++)
  {
    nlevels = vecInpLevels_[ii] - 1;
    for (ii2 = 0; ii2 < maxReps; ii2+=nlevels)
      generateRandomIvector(nlevels, &(iPtr2[ii][ii2]));
  }

  //**/ ----------------------------------------------------------------
  //**/ build the B matrix, and allocate for the other
  //**/ ----------------------------------------------------------------
  maxSamples = maxReps * (nInputs_ + 1);
  if (nSamples_ > maxSamples) maxSamples = nSamples_;
  matBS.setFormat(2);
  matBS.setDim(maxSamples, nInputs_);

  //**/ ----------------------------------------------------------------
  //**/ preparation for generating the samples
  //**/ ----------------------------------------------------------------
  allocSampleData();
  vecRanges.setLength(nInputs_);
  for (ii = 0;  ii < nInputs_;  ii++) 
    vecRanges[ii] = vecUBs_[ii] - vecLBs_[ii];

  //**/ ----------------------------------------------------------------
  //**/ generate a MOAT sample
  //**/ ----------------------------------------------------------------
  double **BS = matBS.getMatrix2D();
  printf("GMOATSampling::generate: finding path 1 (out of %d)\n",nReps);
  for (rr = 0; rr < maxReps; rr++) generate(&BS[rr*(nInputs_+1)],rr);
  for (rr = 1; rr < nReps; rr++)
  {
    printf("GMOATSampling::generate: finding path %d (out of %d)\n",
           rr+1, nReps);
    maxDist = 0.0;
    index = rr;
    base1 = (rr - 1) * (nInputs_ + 1);
    for (ss = rr; ss < maxReps; ss++)
    {
      dDist = 0;
      base2 = ss * (nInputs_ + 1);
      for (kk1 = 0; kk1 <= nInputs_; kk1++)
      {
        for (kk2 = 0; kk2 <= nInputs_; kk2++)
        {
          for (ii = 0; ii < nInputs_; ii++)
          {
            ddata = BS[base1+kk1][ii] - BS[base2+kk2][ii];
            dDist += ddata * ddata;
          }
        }
      }
      if (dDist > maxDist)
      {
        maxDist = dDist;
        index = ss;
      }
    }
    if (index != rr)
    {
      base1 = rr * (nInputs_ + 1);
      base2 = index * (nInputs_ + 1);
      for (kk1 = 0; kk1 <= nInputs_; kk1++)
      {
        for (ii = 0; ii < nInputs_; ii++)
        {
          ddata = BS[base1+kk1][ii];
          BS[base1+kk1][ii] = BS[base2+kk1][ii];
          BS[base2+kk1][ii] = ddata;
        }
      }
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ first distribution checking
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 1)
  {
    vecCounts.setLength(nSamples_);
    vecXT.setLength(nSamples_);
    for (ii = 0; ii < nInputs_; ii++)
    {
      nlevels = vecInpLevels_[ii] - 1;
      for (ss = 0; ss <= nlevels; ss++) vecCounts[ss] = 0;
      for (ss = 0; ss < nSamples_; ss+=(nInputs_+1))
      {
        for (ii2 = 0; ii2 < nInputs_+1; ii2++) vecXT[ii2] = BS[ii2+ss][ii];
        sortDbleList(nInputs_+1, vecXT.getDVector());
        nn = (int) ((vecXT[0] + 1.0e-8) * nlevels);
        vecCounts[nn]++;
        nn = (int) ((vecXT[nInputs_] + 1.0e-8) * nlevels);
        vecCounts[nn]++;
      }
      for (ii2 = 0; ii2 <= nlevels; ii2++)
        printf("GMOAT: input %3d - level = %2d, # replications = %3d\n",
               ii+1, ii2, vecCounts[ii2]);
    }
  }
   
  //**/ ----------------------------------------------------------------
  //**/ convert the sample to application ranges
  //**/ (the randomized version samples different spatial frequencies)
  //**/ ----------------------------------------------------------------
  if (vecInpSubset_.length() > 0)
  {
    nSub = 0;
    for (ii = 0; ii < nInputs_; ii++)
      if (vecInpSubset_[ii] == 1) nSub++;
  }
  else nSub = nInputs_;

  if (randomize_ == 1)
  {
    printf("You are turning on randomFlag in GMOAT.\n");
    printf("Please confirm (y or n): \n");
    scanf("%s", winput1);
    if (winput1[0] != 'y')
    {
      printf("Turning off randomFlag.\n");
      randomize_ = 0;
    }
  }
  if (randomize_ == 0)
  {
    for (ss = 0; ss < nSamples_; ss+=(nInputs_+1))
    {
      for (ii = 0; ii <= nInputs_; ii++)
      {
        for (ii2 = 0; ii2 < nInputs_; ii2++)
        {
          ddata = BS[ss+ii][ii2];
          ddata = ddata * vecRanges[ii2] + vecLBs_[ii2];
          vecSamInps_[(ss+ii)*nInputs_+ii2] = ddata;
        }
      }
      for (ii = nSub+1; ii <= nInputs_; ii++)
        vecSamStas_[ss+ii] = 1;
    }
  }    
  else
  {
    matBRan.setFormat(2);
    matBRan.setDim(nInputs_, maxLevels_);
    double **ranMat = matBRan.getMatrix2D();
    for (ii = 0; ii < nInputs_; ii++)
    {
      delta = 0.5 / (double) (vecInpLevels_[ii] - 1);
      for (ii2 = 0; ii2 < vecInpLevels_[ii]; ii2++)
        ranMat[ii][ii2] = (PSUADE_drand() - 0.5) * delta;
    } 
    for (ss = 0; ss < nSamples_; ss+=(nInputs_+1))
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = BS[ss][ii];
        ddata = ddata * vecRanges[ii] + vecLBs_[ii];
        vecSamInps_[ss*nInputs_+ii] = ddata;
      }
      for (ii = 0; ii < nInputs_; ii++)
      {
        for (ii2 = 0; ii2 < nInputs_; ii2++)
        {
          if (BS[ss+ii+1][ii2] != BS[ss+ii][ii2])
          {
            index = ii2; 
            break;
          }
        }
        for (ii2 = 0; ii2 < nInputs_; ii2++)
          vecSamInps_[(ss+ii+1)*nInputs_+ii2] = 
                vecSamInps_[(ss+ii)*nInputs_+ii2];
        ddata = BS[ss+ii+1][index];
        idata = (int) (ddata * (vecInpLevels_[ii] - 1) + 1.0e-5);
        rdata = ranMat[ii][idata];
        if (ddata > 0.0 && ddata < 1.0) ddata += rdata;
        vecSamInps_[(ss+ii+1)*nInputs_+index] = ddata * vecRanges[index] + 
                                            vecLBs_[index];
      }
      for (ii = nSub+1; ii <= nInputs_; ii++) vecSamStas_[ss+ii] = 1;
    }
  }    

  //**/ ----------------------------------------------------------------
  //**/ repair if needed to
  //**/ ----------------------------------------------------------------
  if (repair(NULL, 0) != 0)
  {
    if (checkSample(nInputs_, nSamples_, vecSamInps_.getDVector()) != 0)
    {
      printf("GMOATSampling : generated sample is not MOAT.\n");
      exit(1);
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ count how many times each level is visited
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 1)
  {
    vecXT.setLength(nSamples_);
    for (ii = 0; ii < nInputs_; ii++)
    {
      for (ii2 = 0; ii2 < nSamples_; ii2++)
        vecXT[ii2] = vecSamInps_[ii2*nInputs_+ii];
      sortDbleList(nSamples_, vecXT.getDVector());
      nn = 1;
      for (ii2 = 1; ii2 < nSamples_; ii2++)
      {
        if (vecXT[ii2] != vecXT[ii2-1])
        {
          printf("GMOAT: input %3d - level = %12.4e, # visits = %4d\n",
                 ii+1, vecXT[ii2-1], nn);
          nn = 1;
        } else nn++;
      }
      printf("GMOAT: input %3d - level = %12.4e, # visits = %4d\n",
             ii+1, vecXT[ii2-1], nn);
    }
  }
  return 0;
}

//*************************************************************************
//* generate the BS matrix
//*------------------------------------------------------------------------
int GMOATSampling::generate(double **BS, int index)
{
  int    ss, ii, ii2, idata, nSub;
  double delta;
  psIVector vecPerm, vecSubset;
  psVector  vecD, vecX;
  psMatrix  matB, matB2;

  //**/ ----------------------------------------------------------------
  //**/ build the B matrix, and allocate for the other
  //**/ ----------------------------------------------------------------
  matB.setFormat(2);
  matB.setDim(nInputs_+1, nInputs_);
  double **B = matB.getMatrix2D();
  for (ii = 0; ii <= nInputs_; ii++)
  {
    for (ii2 = 0; ii2 < ii; ii2++) B[ii][ii2] = 1.0;
    for (ii2 = ii; ii2 < nInputs_; ii2++) B[ii][ii2] = 0.0;
  }
  vecD.setLength(nInputs_);
  vecX.setLength(nInputs_);
  vecPerm.setLength(nInputs_);
  matB2.setFormat(2);
  matB2.setDim(nInputs_+1, nInputs_);
  double **B2 = matB2.getMatrix2D();

  //**/ ----------------------------------------------------------------
  //**/ initialize the D matrix (+-delta/2 on diagonal)
  //**/ ----------------------------------------------------------------
  for (ii = 0; ii < nInputs_; ii++)
  {
    vecD[ii] = PSUADE_drand();
    if (vecD[ii] > 0.5) vecD[ii] = 1.0;
    else                vecD[ii] = -1.0;
  }

  //**/ -----------------------------------------------------------------
  //**/ initialize the X vector
  //**/ ----------------------------------------------------------------
#if 1 
  int **initX = matInitX_.getIMatrix2D();
  for (ii = 0; ii < nInputs_; ii++)
  {
    idata = initX[ii][index];
    vecX[ii] = ((double) idata) / (double) (vecInpLevels_[ii] - 1);
  }
#else
  int imax = P_ - 1;
  for (ii = 0; ii < nInputs_; ii++)
  {
    vecX[ii] = PSUADE_drand();
    idata = (int) (vecX[ii] * imax);
    if (idata >= imax) idata--;
    vecX[ii] = (double) idata / (double) (P_ - 1);
  }
#endif

  //**/ ----------------------------------------------------------------
  //**/ initialize the permute (random permutation) matrix
  //**/ (if subset file exists, separate into two subsets)
  //**/ ----------------------------------------------------------------
  if (vecInpSubset_.length() == 0)
  {
    generateRandomIvector(nInputs_, vecPerm.getIVector());
  }
  else
  {
    vecSubset.setLength(nInputs_);
    nSub = 0;
    for (ii = 0; ii < nInputs_; ii++)
      if (vecInpSubset_[ii] == 1) nSub++;
    generateRandomIvector(nSub, vecSubset.getIVector());
    generateRandomIvector(nInputs_-nSub, vecPerm.getIVector());
    for (ii = 0; ii < nInputs_; ii++)
      vecSubset[nSub+ii] = vecPerm[ii] + nSub;

    ss = 0;
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (vecInpSubset_[ii] == 1)
      {
        vecPerm[ii] = vecSubset[ss];
        ss++;
      }
    }
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (vecInpSubset_[ii] != 1)
      {
        vecPerm[ii] = vecSubset[ss];
        ss++;
      }
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ form the BS matrix (BS = J_1 x + [(2B - J) P D + J] G
  //**/                        = J_1 x + [(2B P - J) D + J] G
  //**/                        = J_1 x + [(2B' - J) D + J] G
  //**/ where J_1 is a (k+1)-by-1 matrix of all ones
  //**/ where x is a 1-by-k row vector
  //**/ where G is a k-by-k diagonal matrix with delta_i entries
  //**/ where D is a k-by-k diagonal matrix of -1 or 1
  //**/ where P is a permutation matrix
  //**/ where J is a (k+1)-by-k matrix of all ones
  //**/ ----------------------------------------------------------------
  for (ii = 0; ii <= nInputs_; ii++)
    for (ii2 = 0; ii2 < nInputs_; ii2++)
      B2[ii][ii2] = B[ii][vecPerm[ii2]];
  for (ii = 0; ii <= nInputs_; ii++)
  {
    for (ii2 = 0; ii2 < nInputs_; ii2++)
    {
      delta = 1.0 / ((double) vecInpLevels_[ii2] - 1.0);
      BS[ii][ii2] = vecX[ii2]+delta/2*((2.0*B2[ii][ii2]-1.0)*vecD[ii2]+1.0);
    }
  }
  return 0;
}

//*************************************************************************
// refine the sample space (not supported)
//**/ (no point of doing this here, since one can create a new GMOAT sample
//**/  and concatenate with the old one)
//-------------------------------------------------------------------------
int GMOATSampling::refine(int, int, double, int, double *)
{
  printf("GMOATSampling: refine not implemented and not needed.\n");
  printf("               You can create a new GMOAT sample and\n");
  printf("               concatenate with the old one.\n");
  return 1;
}

//*************************************************************************
//* repair a MOAT design due to constraints
//  (substitute a subset of inputs with another set of patterns from a
//   repair file) 
//*------------------------------------------------------------------------
int GMOATSampling::repair(char *fname, int start)
{
  int    nPatterns, nInps, nSets, ii, jj, kk, pindex, iindex;
  char   inStr[500], winput1[500], winput2[500], *cString, repairFile[500];
  double ddata;
  FILE   *fp;
  psVector  vecPatterns, vecOneSam;
  psIVector vecInpList;
 
  //**/ -------------------------------------------------------------------
  //**/ initial error checking
  //**/ -------------------------------------------------------------------
  if (start / (nInputs_+1) * (nInputs_ + 1) != start)
  {
    printf("GMOATSampling : start should be multiples of nInputs+1.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ ask users for a repair file
  //**/ ----------------------------------------------------------------
  fp = NULL;
  if (fname != NULL)
  {
    fp = fopen(fname, "r");
    if (fp != NULL) printf("GMOAT repair file found = %s.\n", fname);
  }
  if (fp == NULL)
  {
    cString = psConfig_.getParameter("GMOAT_repair_file");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %s",winput1,winput2,repairFile);
      fp = fopen(repairFile, "r");
      if (fp != NULL)
        printf("GMOAT repair file found = %s.\n", repairFile);
    }
  }
  if (fp == NULL)
  {
    printf("GMOAT repair: no repair file found.\n");
    return 1;
  }

  //**/ -------------------------------------------------------------------
  //**/ read the first line
  //**/ -------------------------------------------------------------------
  fscanf(fp, "%s", inStr);
  if (strcmp(inStr, "BEGIN"))
  {
    printf("GMOATSampling : wrong format in repair file.\n");
    printf("First  line : BEGIN\n");
    printf("Second line : nPatterns nInputs.\n");
    printf("Third  line : a list of input IDs (1-based).\n");
    printf("Fourth line : (and on) set of patterns.\n");
    printf("Last   line : END\n");
    fclose(fp);
    exit(1);
  }

  //**/ -------------------------------------------------------------------
  //**/ read the second line and check errors
  //**/ -------------------------------------------------------------------
  fscanf(fp, "%d %d", &nPatterns, &nInps);
  if (nPatterns <= 0 || nInps <= 0)
  {
    printf("GMOATSampling : nPatterns or nInps <= 0.\n");
    fclose(fp);
    exit(1);
  }
  nSets = nPatterns / (nInps + 1);
  if (nSets*(nInps+1) != nPatterns)
  {
    printf("GMOATSampling ERROR: nPatterns must be multiples of nInputs+1\n");
    fclose(fp);
    exit(1);
  }
  printf("nPatterns = %d involving %d inputs\n", nPatterns, nInps);

  //**/ -------------------------------------------------------------------
  //**/ read the input parameter list
  //**/ -------------------------------------------------------------------
  vecInpList.setLength(nInps);
  for (ii = 0;  ii < nInps; ii++)
  {
    fscanf(fp, "%d", &kk);
    vecInpList[ii] = kk;
    if (vecInpList[ii] <= 0 || vecInpList[ii] > nInputs_)
    {
      printf("GMOATSampling ERROR: input index out of range (%d,%d)\n",
             vecInpList[ii], nInputs_);
      fclose(fp);
      exit(1);
    }
    for (jj = 0; jj < ii; jj++)
    {
      if (vecInpList[ii] == vecInpList[jj])
      {
        printf("GOATSampling ERROR: repeated index (%d)\n",vecInpList[ii]);
        fclose(fp);
        exit(1);
      }
    }
  }

  //**/ -------------------------------------------------------------------
  //**/ read the pattern data
  //**/ -------------------------------------------------------------------
  vecPatterns.setLength(nPatterns * nInps);
  for (ii = 0;  ii < nPatterns; ii++)
  {
    for (jj = 0; jj < nInps; jj++)
    {
      fscanf(fp, "%lg", &ddata);
      vecPatterns[ii*nInps+jj] = ddata;
    }
  }
  fscanf(fp, "%s", inStr);
  fclose(fp);

  //**/ the file must have an "END" after all pattern data
  if (strcmp(inStr, "END"))
  {
    printf("GMOATSampling : wrong format in repair file.\n");
    printf("The file should end with END\n");
    exit(1);
  }

  //**/ -------------------------------------------------------------------
  //**/ check to see if the incoming patterns are indeed MOAT
  //**/ -------------------------------------------------------------------
  if (checkSample(nInps, nPatterns, vecPatterns.getDVector()) != 0)
  {
    printf("GMOATSampling : pattern in repair file is not MOAT.\n");
    exit(1);
  }

  //**/ -------------------------------------------------------------------
  //**/ search and replace
  //**/ -------------------------------------------------------------------
  vecOneSam.setLength(nInps);
  for (ii = start; ii < nSamples_; ii+=nInputs_+1)
  {
    //**/ randomly choose a base pattern
    //**/pindex = (PSUADE_rand() % nSets) * (nInps + 1);
    pindex = (ii / (nInputs_+1)) * (nInps + 1);

    //**/ copy the base pattern to the base sample
    for (kk = 0; kk < nInps; kk++)
    {
      iindex = vecInpList[kk] - 1;
      vecOneSam[kk] = vecSamInps_[ii*nInputs_+iindex];
      vecSamInps_[ii*nInputs_+iindex] = vecPatterns[pindex*nInps+kk];
    }

    //**/ substitute the rest of the current set
    for (jj = 1; jj <= nInputs_; jj++)
    {
      //**/ see if the current sample point involves changing the
      //**/ input to be repaired from the previous sample point
      for (kk = 0; kk < nInps; kk++)
      {
        iindex = vecInpList[kk] - 1;
        if (vecSamInps_[(ii+jj)*nInputs_+iindex] != vecOneSam[kk]) break; 
      }

      //**/ if so, increment the pointer to the repair pattern
      if (kk < nInps) pindex++;

      //**/ copy the current repair pattern to the updated sample point 
      for (kk = 0; kk < nInps; kk++)
      {
        iindex = vecInpList[kk] - 1;
        vecOneSam[kk] = vecSamInps_[(ii+jj)*nInputs_+iindex];
        vecSamInps_[(ii+jj)*nInputs_+iindex] = vecPatterns[pindex*nInps+kk]; 
      }
    }
  }

  //**/ -------------------------------------------------------------------
  //**/ final check to see if the repaired sample is indeed an MOAT sample
  //**/ -------------------------------------------------------------------
  if (checkSample(nInputs_, nSamples_, vecSamInps_.getDVector()) != 0)
  {
    printf("GMOATSampling : repaired file is not MOAT.\n");
    exit(1);
  }
  return 0;
}

//*************************************************************************
//* merge two MOAT designs
//*------------------------------------------------------------------------
int GMOATSampling::merge()
{
  int    nInps1, nOuts1, nSamp1, nInps2, nSamp2, nOutputs, nReps, count; 
  int    samplingMethod, ii, ii2, rr, nSamples, nInputs, cnt1, cnt2;
  double *sampleInputs1, *sampleInputs2;
  char   **inpNames1, **inpNames2, **inpNames, file1[500], file2[500];
  char   pString[500];
  FILE   *fp1, *fp2;
  PsuadeData *psuadeIO1, *psuadeIO2;
  pData      pPtr1, pINames1;
  pData      pPtr2, pINames2;
  psIVector vecSamStas;
  psVector  vecSamInps, vecSamOuts;

  //**/ -------------------------------------------------------------------
  //**/ read first data file
  //**/ -------------------------------------------------------------------
  snprintf(pString,100,"Please enter the name of the first MOAT datafile: ");
  getString(pString, file1);
  file1[strlen(file1)-1] = '\0';
  if ((fp1=fopen(file1,"r")) == NULL)
  {
    printf("GMOATSampling ERROR: File %s not found.\n", file1);
    return 1;
  }
  else fclose(fp1);
  psuadeIO1 = new PsuadeData();
  psuadeIO1->setOutputLevel(0);
  if (psuadeIO1->readPsuadeFile(file1) != 0)
  {
    printf("GMOAT ERROR: problem with reading file %s.\n", file1);
    delete psuadeIO1;
    return 1;
  }
  psuadeIO1->getParameter("input_ninputs", pPtr1);
  nInps1 = pPtr1.intData_;
  psuadeIO1->getParameter("input_names", pINames1);
  inpNames1 = pINames1.strArray_;
  psuadeIO1->getParameter("output_noutputs", pPtr1);
  nOuts1 = pPtr1.intData_;
  psuadeIO1->getParameter("method_sampling", pPtr1);
  samplingMethod = pPtr1.intData_;
  if (samplingMethod != PSUADE_SAMP_GMOAT)
  {
    printf("GMOAT Merge ERROR: data1 is not GMOAT.\n");
    delete psuadeIO1;
    pINames1.clean();
    return 1;
  }
  assert(psuadeIO1->getParameter("method_nsamples", pPtr1) == 0);
  nSamp1 = pPtr1.intData_;
  assert(psuadeIO1->getParameter("input_sample", pPtr1) == 0);
  sampleInputs1 = pPtr1.dbleArray_;
  if (checkSample2(nInps1, nSamp1, sampleInputs1) != 0)
  {
    printf("GMOAT Merge ERROR: first sample is not GMOAT.\n");
    delete psuadeIO1;
    pINames1.clean();
    return 1;
  }

  //**/ -------------------------------------------------------------------
  //**/ read second data file
  //**/ -------------------------------------------------------------------
  snprintf(pString,100,"Please enter the name of the second MOAT datafile: ");
  getString(pString, file2);
  file2[strlen(file2)-1] = '\0';
  if ((fp2=fopen(file2,"r")) == NULL)
  {
    printf("GMOAT ERROR : File %s not found.\n", file2);
    delete psuadeIO1;
    pINames1.clean();
    return 1;
  }
  else fclose(fp2);
                                                                  
  psuadeIO2 = new PsuadeData();
  psuadeIO2->setOutputLevel(0);
  if (psuadeIO2->readPsuadeFile(file2) != 0)
  {
    printf("GMOAT ERROR : problem with reading file %s.\n", file2);
    delete psuadeIO1;
    delete psuadeIO2;
    pINames1.clean();
    return 1;
  }
  assert(psuadeIO2->getParameter("input_ninputs", pPtr2) == 0);
  nInps2 = pPtr2.intData_;
  assert(psuadeIO2->getParameter("input_names", pINames2) == 0);
  inpNames2 = pINames2.strArray_;
  assert(psuadeIO2->getParameter("method_sampling", pPtr2) == 0);
  samplingMethod = pPtr2.intData_;
  if (samplingMethod != PSUADE_SAMP_GMOAT)
  {
    printf("GMOAT Merge ERROR : data2 is not GMOAT.\n");
    delete psuadeIO1;
    delete psuadeIO2;
    pINames1.clean();
    pINames2.clean();
    return 1;
  }
  assert(psuadeIO2->getParameter("method_nsamples", pPtr2) == 0);
  nSamp2 = pPtr2.intData_;
  assert(psuadeIO2->getParameter("input_sample", pPtr2) == 0);
  sampleInputs2 = pPtr2.dbleArray_;
  if (checkSample2(nInps2, nSamp2, sampleInputs2) != 0)
  {
    printf("GMOAT Merge ERROR : second sample is not GMOAT.\n");
    delete psuadeIO1;
    delete psuadeIO2;
    pINames1.clean();
    pINames2.clean();
    return 1;
  }

  //**/ -------------------------------------------------------------------
  //**/ error checking
  //**/ -------------------------------------------------------------------
  nReps = nSamp1 / (nInps1 + 1);
  if (nReps != (nSamp2 / (nInps2 + 1)))
  {
    printf("GMOAT Merge ERROR : different number of replications.\n");
    delete psuadeIO1;
    delete psuadeIO2;
    pINames1.clean();
    pINames2.clean();
    return 1;
  }

  //**/ -------------------------------------------------------------------
  //**/ combine the two samples
  //**/ -------------------------------------------------------------------
  nSamples = nSamp1 + nSamp2 - nReps;
  nInputs  = nInps1 + nInps2;
  nOutputs = nOuts1;
  vecSamInps.setLength(nSamples*nInputs);
  vecSamOuts.setLength(nSamples*nOutputs);
  vecSamStas.setLength(nSamples);

  for (rr = 0; rr < nReps; rr++)
  {
    count = rr * (nInputs + 1);
    cnt1  = rr * (nInps1 + 1);
    cnt2  = rr * (nInps2 + 1);
    for (ii = 0; ii < nInps1; ii++)
      vecSamInps[count*nInputs+ii] = sampleInputs1[cnt1*nInps1+ii];
    for (ii = 0; ii < nInps2; ii++)
      vecSamInps[count*nInputs+nInps1+ii] = sampleInputs2[cnt2*nInps2+ii];
    for (ii = 0; ii < nInps1; ii++)
    {
      count++;
      cnt1++;
      for (ii2 = 0; ii2 < nInps1; ii2++)
        vecSamInps[count*nInputs+ii2] = sampleInputs1[cnt1*nInps1+ii2];
      for (ii2 = 0; ii2 < nInps2; ii2++)
        vecSamInps[count*nInputs+nInps1+ii2] = 
               sampleInputs2[cnt2*nInps2+ii2];
    }
    for (ii = 0; ii < nInps2; ii++)
    {
      count++;
      cnt2++;
      for (ii2 = 0; ii2 < nInps1; ii2++)
        vecSamInps[count*nInputs+ii2] = sampleInputs1[cnt1*nInps1+ii2];
      for (ii2 = 0; ii2 < nInps2; ii2++)
        vecSamInps[count*nInputs+nInps1+ii2] = 
               sampleInputs2[cnt2*nInps2+ii2];
    }
  }
  for (ii = 0; ii < nSamples; ii++) vecSamStas[ii] = 0;
  for (ii = 0; ii < nSamples*nOutputs; ii++)
    vecSamOuts[ii] = PSUADE_UNDEFINED;
  inpNames = new char*[nInputs];
  for (ii = 0; ii < nInps1; ii++)
  {
    inpNames[ii] = new char[100]; 
    strcpy(inpNames[ii], inpNames1[ii]);
  }
  for (ii = 0; ii < nInps1; ii++)
  {
    inpNames[nInps1+ii] = new char[100]; 
    strcpy(inpNames[nInps1+ii], inpNames2[ii]);
  }
  psuadeIO1->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                   vecSamInps.getDVector(),inpNames,NULL,NULL,NULL,NULL);
  psuadeIO1->updateOutputSection(nSamples,nOutputs,vecSamOuts.getDVector(),
                                 vecSamStas.getIVector(),NULL);
  psuadeIO1->updateMethodSection(-1,nSamples,-1,-1,-1);
  psuadeIO1->writePsuadeFile(NULL,0);
   
  //**/ -------------------------------------------------------------------
  //**/ clean up 
  //**/ -------------------------------------------------------------------
  delete psuadeIO1;
  delete psuadeIO2;
  pINames1.clean();
  pINames2.clean();
  for (ii = 0; ii < nInputs; ii++) delete [] inpNames[ii];
  delete [] inpNames;
  return 0;
}

//*************************************************************************
//* check whether the sample is MOAT (return 0 if yes)
//*------------------------------------------------------------------------
int GMOATSampling::checkSample(int nInputs, int nSamples, double *X)
{
  int    ss, ii, ii2, nDiff;
  double xtemp1, xtemp2;
  psIVector vecErrFlags;

  vecErrFlags.setLength(nInputs);
  for (ss = 0; ss < nSamples; ss+=(nInputs+1))
  {
    for (ii = 0; ii < nInputs; ii++) vecErrFlags[ii] = 0;
    for (ii = 1; ii <= nInputs; ii++)
    {
      for (ii2 = 0; ii2 < nInputs; ii2++)
      {
        xtemp1 = X[(ss+ii-1)*nInputs+ii2];
        xtemp2 = X[(ss+ii)*nInputs+ii2];
        if (xtemp1 != xtemp2) vecErrFlags[ii2]++;
      }
    }
    nDiff = 0;
    for (ii = 0; ii < nInputs; ii++) nDiff += vecErrFlags[ii];
    if (nDiff != nInputs) return 1;
  }
  return 0;
}

//*************************************************************************
//* check whether the sample is MOAT (return 0 if yes)
//  (input X is in a different format)
//*------------------------------------------------------------------------
int GMOATSampling::checkSample2(int nInputs, int nSamples, double *X)
{
  int    ss, ii, ii2, nDiff;
  double xtemp1, xtemp2;
  psIVector vecErrFlags;

  vecErrFlags.setLength(nInputs);
  for (ss = 0; ss < nSamples; ss+=(nInputs+1))
  {
    for (ii = 0; ii < nInputs; ii++) vecErrFlags[ii] = 0;
    for (ii = 1; ii <= nInputs; ii++)
    {
      for (ii2 = 0; ii2 < nInputs; ii2++)
      {
        xtemp1 = X[(ss+ii-1)*nInputs+ii2];
        xtemp2 = X[(ss+ii)*nInputs+ii2];
        if (xtemp1 != xtemp2) vecErrFlags[ii2]++;
      }
    }
    nDiff = 0;
    for (ii = 0; ii < nInputs; ii++) nDiff += vecErrFlags[ii];
    if (nDiff != nInputs) return 1;
  }
  return 0;
}

//*************************************************************************
//* initialize the sampling data
//*------------------------------------------------------------------------
int GMOATSampling::initializeHighDimension()
{
  int nReps, ii, rr, ind1, ind2, kk, count, ii2, nn, currBin;
  psVector  vecRanges, vecS1, vecS2, vecXT;
  psIVector vecSI, vecBins;

  //**/ ----------------------------------------------------------------
  //**/ preparation for generating the sample
  //**/ ----------------------------------------------------------------
  nReps = nSamples_ / (nInputs_ + 1);
  allocSampleData();
  vecRanges.setLength(nInputs_);
  for (ii = 0;  ii < nInputs_;  ii++) 
    vecRanges[ii] = vecUBs_[ii] - vecLBs_[ii];

  //**/ ----------------------------------------------------------------
  //**/ generate an GMOAT sample
  //**/ ----------------------------------------------------------------
  vecS1.setLength(nInputs_);
  vecS2.setLength(nInputs_);
  vecSI.setLength(nInputs_);
  for (rr = 0; rr < nReps; rr++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      ind1 = PSUADE_rand() % P_; 
      vecS1[ii] = vecRanges[ii] / (P_ - 1) * ind1 + vecLBs_[ii]; 
      kk = PSUADE_rand() % 2;
      if (kk == 0) ind2 = ind1 - 1;
      else         ind2 = ind1 + 1;
      if      (ind2 < 0)    ind2 = ind2 + 2;
      else if (ind2 > P_-1) ind2 = ind2 - 2;
      vecS2[ii] = vecRanges[ii] / (P_ - 1) * ind2 + vecLBs_[ii]; 
    }
    for (ii = 0; ii < nInputs_; ii++)
      vecSamInps_[(rr*(nInputs_+1))*nInputs_+ii] = vecS1[ii];
    for (ii = 0; ii < nInputs_; ii++) vecSI[ii] = ii;
    count = nInputs_;
    for (kk = 0; kk < nInputs_; kk++)
    {
      ind1 = PSUADE_rand() % count; 
      ind2 = vecSI[ind1];
      for (ii = ind1; ii < count-1; ii++) vecSI[ii] = vecSI[ii+1];
      count--;
      for (ii = 0; ii < nInputs_; ii++)
        vecSamInps_[(rr*(nInputs_+1)+kk+1)*nInputs_+ii] = 
               vecSamInps_[(rr*(nInputs_+1)+kk)*nInputs_+ii]; 
      vecSamInps_[(rr*(nInputs_+1)+kk+1)*nInputs_+ind2] = vecS2[ind2]; 
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ count how many times each level is visited
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 2)
  {
    vecXT.setLength(2*nReps);
    vecBins.setLength(P_);
    for (ii = 0; ii < P_; ii++) vecBins[ii] = 0;
    for (ii = 0; ii < nInputs_; ii++)
    {
      for (rr = 0; rr < nReps; rr++)
      {
        vecXT[rr*2] = vecSamInps_[(rr*(nInputs_+1))*nInputs_+ii];
        for (ii2 = 1; ii2 < nInputs_+1; ii2++)
        {
          if (vecSamInps_[(rr*(nInputs_+1)+ii2)*nInputs_+ii] != vecXT[2*rr])
          {
            vecXT[rr*2+1] = vecSamInps_[(rr*(nInputs_+1)+ii2)*nInputs_+ii];
            break;
          }
        }
      }
      sortDbleList(2*nReps, vecXT.getDVector());
      nn = 1;
      currBin = 0;
      for (ii2 = 1; ii2 < 2*nReps; ii2++)
      {
        if (vecXT[ii2] != vecXT[ii2-1])
        {
          printf("GMOAT: input %3d - level = %12.4e, # times = %4d\n",
                 ii+1, vecXT[ii2-1], nn);
          vecBins[currBin++] += nn;
          nn = 1;
        } else nn++;
      }
      printf("GMOAT: input %3d - level = %12.4e, # times = %4d\n",
             ii+1, vecXT[ii2-1], nn);
      vecBins[currBin++] += nn;
    }
    for (ii = 0; ii < P_; ii++) 
      printf("GMOAT: frequency of visit to bin %5d = %d\n",ii+1,vecBins[ii]);
  }
  return 0;
}

// ************************************************************************
// equal operator modified by Bill Oliver
// ------------------------------------------------------------------------
GMOATSampling& GMOATSampling::operator=(const GMOATSampling &gms)
{
  if (this == &gms) return *this;

  int maxReps = 5000;
  int nlevels;
  samplingID_  = gms.samplingID_;
  P_ = gms.P_;
  //**/ number of levels for each input
  nInputs_ = gms.nInputs_;
  vecInpLevels_ = gms.vecInpLevels_; 
  vecInpSubset_ = gms.vecInpSubset_; 
  matInitX_ = gms.matInitX_; 
  printLevel_ = gms.printLevel_;
  samplingID_ = gms.samplingID_;
  nSamples_ = gms.nSamples_;
  nOutputs_ = gms.nOutputs_;
  randomize_ = gms.randomize_;
  nReplications_ = gms.nReplications_;
  vecLBs_ = gms.vecLBs_;
  vecUBs_ = gms.vecUBs_;
  vecSamInps_ = gms.vecSamInps_; 
  vecSamOuts_ = gms.vecSamOuts_; 
  vecSamStas_ = gms.vecSamStas_; 
  return (*this);
}

