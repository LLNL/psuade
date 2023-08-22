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
// Functions for the Morris one-at-a-time class (improved) 
// AUTHOR : CHARLES TONG
// DATE   : 2004 (updated in 2006 and 2007)
//*------------------------------------------------------------------------
//**/ added features
//**/    - adjust P based on input file specification
//**/    - add randomness (turned off)
//**/    - refinement (adding more replicates)
//**/    - repair
//**/    - subsetting (for telescoping to a small set of inputs) 
//**/    - merge two MOAT with 2 different sets of inputs
//**/    - use Campolongo's space filling technique
// ************************************************************************
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "pData.h"
#include "pData.h"
#include "PsuadeData.h"
#include "FuncApprox.h"
#include "Psuade.h"
#include "MOATSampling.h"
#include "PrintingTS.h"
#define PABS(x) ((x) > 0 ? (x) : -(x))

//*************************************************************************
//* Constructor
//*------------------------------------------------------------------------
MOATSampling::MOATSampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_MOAT;
  P_ = 4;
}

//*************************************************************************
//* Copy Constructor added by Bill Oliver
//*------------------------------------------------------------------------
MOATSampling::MOATSampling(const MOATSampling & ms) : Sampling()
{
  samplingID_ = ms.samplingID_;
  P_ = ms.P_;
  nInputs_ = ms.nInputs_;
  vecInpSubset_ = ms.vecInpSubset_;

  //**/MetisSampling inherits from Sampling so include the parent class 
  //**/data members
  printLevel_ = ms.printLevel_;
  samplingID_ = ms.samplingID_;
  nSamples_ = ms.nSamples_;
  nOutputs_ = ms.nOutputs_;
  randomize_ = ms.randomize_;
  nReplications_ = ms.nReplications_;
  vecLBs_ = ms.vecLBs_;
  vecUBs_ = ms.vecUBs_;
  vecSamInps_ = ms.vecSamInps_;
  vecSamOuts_ = ms.vecSamOuts_;
  vecSamStas_ = ms.vecSamStas_;
}

//*************************************************************************
//* destructor 
//*------------------------------------------------------------------------
MOATSampling::~MOATSampling()
{
}

//*************************************************************************
//* initialize the sampling data
//*------------------------------------------------------------------------
int MOATSampling::initialize(int initLevel)
{
  int    ii, ii2, rr, ss, nReps, nn, randomize, currBin, nSub=0;
  int    kk1, kk2, maxReps=500, maxSamples, index, base1, base2, setFlag=0;
  double ddata, maxDist, dDist;
  char   *cString, partitionFile[200], winput1[200], winput2[200];
  FILE*  fp;
  psVector vecRanges;
  psMatrix matBS;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  if (nSamples_ == 0)
  {
    printOutTS(PL_ERROR,
         "MOATSampling::initialize ERROR - nSamples = 0.\n");
    exit(1);
  }
  if (nInputs_ == 0)
  {
    printOutTS(PL_ERROR,
         "MOATSampling::initialize ERROR - input not set up.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ set up 
  //**/ ----------------------------------------------------------------
  randomize = (randomize_ & 1);
  if (nSamples_/(nInputs_+1) * (nInputs_+1) != nSamples_) 
  {
    printOutTS(PL_INFO,
         "MOATSampling: nSamples should be multiples of nInputs+1.\n");
    printOutTS(PL_INFO,
         "              nSamples reset to be 10*(nInputs+1).\n");
    nSamples_ = 10 * (nInputs_ + 1);
  }

  //**/ ----------------------------------------------------------------
  //**/ If initLevel is 0, do not need to generate sample. 
  //**/ This is used when an old sample is to be loaded for refinement. 
  //**/ ----------------------------------------------------------------
  if (initLevel != 0) return 0;

  //**/ ----------------------------------------------------------------
  //**/ check for subset information from a file
  //**/ Subsetting is useful when a few inputs are to be examined more
  //**/ closely while all the others are left alone.
  //**/ ----------------------------------------------------------------
  cString = psConfig_.getParameter("MOAT_P");
  if (cString != NULL)
  {
    sscanf(cString, "%s %s %d",winput1,winput2,&P_);
    P_ = P_ / 2 * 2;
    if (P_ <= 0 || P_ > 100) P_ = 4;
    printOutTS(PL_INFO,"MOATSampling: P set to %d (config)\n", P_);
    setFlag = 1;
  }
  cString = psConfig_.getParameter("MOAT_partition_file");
  if (cString != NULL)
  {
    sscanf(cString, "%s %s %s",winput1,winput2,partitionFile);
    printOutTS(PL_INFO,
       "MOATSampling: use MOAT input partition file %s.\n",
       partitionFile);
    fp = fopen(partitionFile, "r");
    if (fp != NULL)
    {
      fscanf(fp, "%d", &ss);
      if (ss <= 0 || ss >= nInputs_)
      {
        printOutTS(PL_INFO,
             "MOATSampling: invalid MOAT input partition file.\n");
        printOutTS(PL_INFO,
             "              The first line should be nInputs.\n");
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
            printOutTS(PL_INFO,
                 "MOATSampling: invalid input partition file.\n");
            printOutTS(PL_INFO,
                 "               invalid input index %d.\n", ii);
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
  if (psConfig_.SamExpertModeIsOn() && setFlag == 0)
  {
    printOutTS(PL_INFO,"MOATSampling: the current P is %d.\n", P_);
    snprintf(winput1,100,"Please choose a new P: (4 - 10, even) ");
    P_ = getInt(4, 10, winput1);
    P_ = P_ / 2 * 2;
  }

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 4)
  {
    printOutTS(PL_INFO,"MOATSampling: initialize: nSamples  = %d\n", 
               nSamples_);
    printOutTS(PL_INFO,"MOATSampling: initialize: nInputs   = %d\n", 
               nInputs_);
    printOutTS(PL_INFO,"MOATSampling: initialize: nOutputs  = %d\n", 
               nOutputs_);
    printOutTS(PL_INFO,"MOATSampling: initialize: numLevels = %d\n",P_);
    if (randomize != 0)
         printOutTS(PL_INFO,"MOATSampling: initialize: randomize on\n");
    else printOutTS(PL_INFO,"MOATSampling: initialize: randomize off\n");
  }

  //**/ ----------------------------------------------------------------
  //**/ if nInputs > 100, use fast version
  //**/ ----------------------------------------------------------------
  if (nInputs_ > 100)
  {
    printOutTS(PL_INFO,"MOATSampling: nInputs > 100, use fast version.\n");
    initializeHighDimension();
    return 0;
  }

  //**/ ----------------------------------------------------------------
  //**/ allocate pattern matrix
  //**/ ----------------------------------------------------------------
  nReps = nSamples_ / (nInputs_ + 1);
  if (nReps > maxReps) maxReps = nReps;
  maxSamples = (nInputs_ + 1) * maxReps;
  matBS.setFormat(2);
  matBS.setDim(maxSamples,nInputs_);

  //**/ ----------------------------------------------------------------
  //**/ generate an MOAT sample
  //**/ ----------------------------------------------------------------
  double **BS;
  BS = matBS.getMatrix2D();
  if (printLevel_ > 0)
    printOutTS(PL_INFO,
         "MOATSampling::generate: finding path 1 (out of %d)\n",nReps);
  for (rr = 0; rr < maxReps; rr++) generate(&BS[rr*(nInputs_+1)]);

  //**/ ----------------------------------------------------------------
  //**/ path selection
  //**/ ----------------------------------------------------------------
  for (rr = 1; rr < nReps; rr++) 
  {
    if (printLevel_ > 0)
      printOutTS(PL_INFO,
         "MOATSampling::generate: finding path %d (out of %d)\n",rr+1,
         nReps);
    maxDist = 0;
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
  //**/ preparation for generating the sample
  //**/ ----------------------------------------------------------------
  vecSamInps_.setLength(nSamples_*nInputs_);
  vecSamOuts_.setLength(nSamples_*nOutputs_);
  vecSamStas_.setLength(nSamples_);
  vecRanges.setLength(nInputs_);
  for (ii = 0;  ii < nInputs_;  ii++) 
    vecRanges[ii] = vecUBs_[ii] - vecLBs_[ii];

  //**/ ----------------------------------------------------------------
  //**/ convert the sample to application ranges
  //**/ (the randomized version samples different spatial frequencies)
  //**/ ----------------------------------------------------------------
  if (vecInpSubset_.length() > 0)
  {
    nSub = 0;
    for (ii = 0; ii < nInputs_; ii++) if (vecInpSubset_[ii] == 1) nSub++;
  }
  else nSub = nInputs_;

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
    for (ii = nSub+1; ii <= nInputs_; ii++) vecSamStas_[ss+ii] = 1;
  }
  for (ss = 0; ss < nSamples_*nOutputs_; ss++)
    vecSamOuts_[ss] = PSUADE_UNDEFINED;

  //**/ ----------------------------------------------------------------
  //**/ repair if needed to
  //**/ (for constrained MOAT)
  //**/ ----------------------------------------------------------------
  if (repair(NULL,0) != 0)
  {
    if (checkSample(nInputs_, nSamples_, vecSamInps_.getDVector()) != 0)
    {
      printOutTS(PL_ERROR,
           "MOATSampling: generated sample is not MOAT.\n");
      exit(1);
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ count how many times each level is visited
  //**/ ----------------------------------------------------------------
  psVector  vecXT;
  psIVector vecBins;
  if (printLevel_ > 2)
  {
    vecXT.setLength(2*nReps);
    vecBins.setLength(P_);
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
          printf("MOAT: input %3d - level = %12.4e, # times = %4d\n",
                 ii+1, vecXT[ii2-1], nn);
          vecBins[currBin++] += nn;
          nn = 1;
        } else nn++;
      }
      printf("MOAT: input %3d - level = %12.4e, # times = %4d\n",
             ii+1, vecXT[ii2-1], nn);
      vecBins[currBin++] += nn;
    }
    for (ii = 0; ii < P_; ii++) 
      printf("MOAT: frequency of visit to bin %5d = %d\n",ii+1,vecBins[ii]);
  }
  return 0;
}

//*************************************************************************
//* generate the BS matrix
//*------------------------------------------------------------------------
int MOATSampling::generate(double **BS)
{
  int    ss, ii, ii2, idata, nSub, imax;
  double delta;
  psMatrix matB1, matB2;
  psVector vecD, vecX;
  psIVector vecPerm, vecSubset;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  delta = P_ / ((double) (2*P_) - 2.0);

  //**/ ----------------------------------------------------------------
  //**/ build the B matrix, and allocate for the other
  //**/ ----------------------------------------------------------------
  matB1.setFormat(2);
  matB1.setDim(nInputs_+1, nInputs_);
  for (ii = 0; ii <= nInputs_; ii++)
  {
    for (ii2 = 0; ii2 < ii; ii2++) matB1.setEntry(ii,ii2,1.0);
    for (ii2 = ii; ii2 < nInputs_; ii2++) matB1.setEntry(ii,ii2,0.0);
  }
  vecD.setLength(nInputs_);
  vecX.setLength(nInputs_);
  vecPerm.setLength(nInputs_);
  matB2.setFormat(2);
  matB2.setDim(nInputs_+1, nInputs_);

  //**/ ----------------------------------------------------------------
  //**/ initialize the D matrix (+-1 on diagonal)
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
#if 0 //**/
  for (ii = 0; ii < nInputs_; ii++) //**/
  { //**/
    X[ii] = PSUADE_drand(); //**/
    idata = P_ / 2; //**/
    ddata = ((double) idata) / (double) (P_ - 1); //**/
    idata = (int) (X[ii] * ddata * ((double) (P_ - 1))); //**/
    if (idata == P_/2) idata--; //**/
    X[ii] = (double) idata / (double) (P_ - 1); //**/
  } //**/
#endif //**/
  imax = (P_ - 1) / 2;
  for (ii = 0; ii < nInputs_; ii++)
  {
    vecX[ii] = PSUADE_drand();
    idata = (int) (vecX[ii] * (imax + 1));
    if (idata > imax) idata--;
    vecX[ii] = (double) idata / (double) (P_ - 1);
  }
   
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
    for (ii = 0; ii < nInputs_; ii++) if (vecInpSubset_[ii] == 1) nSub++;
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
  //**/ form the BS matrix (BS = (J x + delta/2[(2B - J)D + J]) P)
  //**/ ----------------------------------------------------------------
  double **B1 = matB1.getMatrix2D();
  double **B2 = matB2.getMatrix2D();

  for (ii = 0; ii <= nInputs_; ii++)
    for (ii2 = 0; ii2 < nInputs_; ii2++)
      B2[ii][ii2] = vecX[ii2]+delta/2*((B1[ii][ii2]*2-1.0)*vecD[ii2]+1.0);
  for (ii = 0; ii <= nInputs_; ii++)
    for (ii2 = 0; ii2 < nInputs_; ii2++)
      BS[ii][ii2] = B2[ii][vecPerm[ii2]];

  return 0;
}

//*************************************************************************
//* refine the sample space
//**/ (Refine here means only to increase the number of paths by a factor of
//**/  2. The reason is that there is no reason to refine from previous
//**/  patterns. If one wants to do multiscale MOAT analysis, you can just
//**/  create from scratch another MOAT pattern.)
//**/  (Also this version of MOAT is not suitable for multiscale due to
//**/   the use of equilibrated distribution (delta > half of range))
//*------------------------------------------------------------------------
int MOATSampling::refine(int refineRatio, int randomize, double thresh,
                         int nSamples, double *sampleErrors)
{
  int    ss, ii, ii2, jj, rr, nReps, nTimes, newNSamples;
  double ddata;
  psVector  vecRanges, vecNewSamInps, vecNewSamOuts;
  psIVector vecNewSamStas;
  psMatrix  matBS;

  //**/ -------------------------------------------------------------------
  //**/ unused parameters
  //**/ -------------------------------------------------------------------
  (void) randomize;
  (void) thresh;
  (void) nSamples;
  (void) sampleErrors;

  //**/ -------------------------------------------------------------------
  //**/ check to see if the sample is MOAT 
  //**/ -------------------------------------------------------------------
  if (vecInpSubset_.length() > 0)
  {
    printOutTS(PL_INFO,"MOATSampling: refine is not available due to the\n");
    printOutTS(PL_INFO,"              use of selective replications.\n");
    return 0;
  }
  if (refineRatio != 2)
    printOutTS(PL_INFO,"MOATSampling WARNING: refinement ratio set to 2.\n");

  nTimes = 2;

  //**/ ----------------------------------------------------------------
  //**/ allocate space for new sample data
  //**/ ----------------------------------------------------------------
  newNSamples = nSamples_ * nTimes;
  vecNewSamInps.setLength(newNSamples*nInputs_);
  vecNewSamOuts.setLength(newNSamples*nOutputs_);
  vecNewSamStas.setLength(newNSamples);
  for (ss = 0;  ss < newNSamples; ss++)
  {
    for (jj = 0; jj < nOutputs_; jj++)
      vecNewSamOuts[ss*nOutputs_+jj] = PSUADE_UNDEFINED;
  }

  //**/ ----------------------------------------------------------------
  //**/ copy the old samples
  //**/ ----------------------------------------------------------------
  for (ss = 0; ss < nSamples_; ss++) 
  {
    for (ii = 0; ii < nInputs_; ii++)
      vecNewSamInps[ss*nInputs_+ii] = vecSamInps_[ss*nInputs_+ii];
    for (jj = 0; jj < nOutputs_; jj++)
      vecNewSamOuts[ss*nOutputs_+jj] = vecSamOuts_[ss*nOutputs_+jj];
    vecNewSamStas[ss] = 1;
  }

  //**/ ----------------------------------------------------------------
  //**/ allocate the temporary BS matrix
  //**/ ----------------------------------------------------------------
  matBS.setFormat(2);
  matBS.setDim(nSamples_, nInputs_);

  //**/ ----------------------------------------------------------------
  //**/ preparation for generating the samples
  //**/ ----------------------------------------------------------------
  nReps  = nSamples_ / (nInputs_ + 1);
  vecRanges.setLength(nInputs_);
  for (ii = 0;  ii < nInputs_;  ii++) 
    vecRanges[ii] = vecUBs_[ii] - vecLBs_[ii];

  //**/ ----------------------------------------------------------------
  //**/ repeat the sample generation
  //**/ ----------------------------------------------------------------
  double **BS = matBS.getMatrix2D();
  for (rr = 0; rr < nReps; rr++) generate(&BS[rr*(nInputs_+1)]);
  for (ss = nSamples_; ss < newNSamples; ss+=(nInputs_+1))
  {
    for (ii = 0; ii <= nInputs_; ii++)
    {
      for (ii2 = 0; ii2 < nInputs_; ii2++)
      {
        ddata = BS[ss-nSamples_+ii][ii2];
        ddata = ddata * vecRanges[ii2] + vecLBs_[ii2];
        vecNewSamInps[(ss+ii)*nInputs_+ii2] = ddata;
      }
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ randomized version (do not use for now)
  //**/ ----------------------------------------------------------------
#if 0 //**/
  if (randomize_ != 0) //**/
  { //**/
    delta = 0.5 / (double) (P_ - 1); //**/
    for (ss = nSamples_; ss < newNSamples; ss+=(nInputs_+1)) //**/
    { //**/
      for (ii = 0; ii < nInputs_; ii++) //**/
      { //**/
        ddata = BS[ss-nSamples_][ii]; //**/
        ddata = ddata * vecRanges[ii] + lowerBounds_[ii]; //**/
        vecSamInps_[ss*nInputs_+ii] = ddata; //**/
      } //**/
      for (ii = 0; ii < nInputs_; ii++) //**/
      { //**/
        for (ii2 = 0; ii2 < nInputs_; ii2++) //**/
        { //**/
          if (BS[ss-nSamples_+ii+1][ii2] != BS[ss-nSamples_+ii][ii2]) //**/
          { //**/
            index = ii2; //**/
            break; //**/
          } //**/
        } //**/
        for (ii2 = 0; ii2 < nInputs_; ii2++)    //**/
          vecSamInps_[(ss+ii+1)*nInputs_+ii2] = //**/
            vecSamInps_[(ss+ii)*nInputs_+ii2];  //**/
        ddata = BS[ss-nSamples_+ii+1][index]; //**/
        rdata = (PSUADE_drand() - 0.5) * delta; //**/
        if ((ddata+rdata) < 0.0 || (ddata+rdata) > 1.0) ddata -= rdata; //**/
        else                                            ddata += rdata; //**/
        ddata = ddata * vecRanges[index] + vecLBs_[index]; //**/
        vecSamInps_[(ss+ii+1)*nInputs_+index] = ddata; //**/
      } //**/
    } //**/
  } //**/
#endif //**/

  //**/ -------------------------------------------------------------------
  //**/ revise internal variables
  //**/ -------------------------------------------------------------------
  nSamples_ = nSamples_ * nTimes;
  vecSamInps_ = vecNewSamInps;
  vecSamOuts_ = vecNewSamOuts;
  vecSamStas_ = vecNewSamStas;

  //**/ -------------------------------------------------------------------
  //**/ check to see if the final sample is indeed MOAT 
  //**/ -------------------------------------------------------------------
  if (checkSample(nInputs_, nSamples_, vecSamInps_.getDVector()) != 0)
  {
    printOutTS(PL_ERROR,"MOATSampling: refined sample is not MOAT.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 4)
  {
    printOutTS(PL_INFO,"MOATSampling::refine: nSamples = %d\n",nSamples_);
    printOutTS(PL_INFO,"MOATSampling::refine: nInputs  = %d\n",nInputs_);
    printOutTS(PL_INFO,"MOATSampling::refine: nOutputs = %d\n",nOutputs_);
    if (randomize != 0)
         printOutTS(PL_INFO,"MOATSampling::refine: randomize on\n");
    else printOutTS(PL_INFO,"MOATSampling::refine: randomize off\n");
  }

  //**/ ----------------------------------------------------------------
  //**/ repair if needed to
  //**/ ----------------------------------------------------------------
  if (repair(NULL, nSamples_/nTimes) != 0)
  {
    if (checkSample(nInputs_, nSamples_, vecSamInps_.getDVector()) != 0)
    {
      printOutTS(PL_ERROR,"MOATSampling: refined sample is not MOAT.\n");
      exit(1);
    }
  }
  return 0;
}

//*************************************************************************
//* repair a MOAT design due to constraints
//**/  (substitute a subset of inputs with another set of patterns from 
//**/   psuade_moat_repair_file) 
//*------------------------------------------------------------------------
int MOATSampling::repair(char *fname, int start)
{
  int    nPatterns, nInps, nSets, ii, jj, kk, pindex, iindex;
  char   inStr[200], *cString, winput1[200], winput2[200], repairFile[200];
  double dtemp;
  FILE   *fp=NULL;
  psVector  vecPatterns, vecOneSam;
  psIVector vecInpList;

  //**/ -------------------------------------------------------------------
  //**/ initial error checking
  //**/ -------------------------------------------------------------------
  if (start / (nInputs_+1) * (nInputs_ + 1) != start)
  {
    printOutTS(PL_ERROR,
         "MOATSampling: start should be multiples of nInputs+1.\n");
    exit(1);
  }
  
  //**/ -------------------------------------------------------------------
  //**/ check to see if the repair file exists
  //**/ -------------------------------------------------------------------
  fp = NULL;
  if (fname != NULL)
  {
    fp = fopen(fname, "r");
    if (fp != NULL) printf("MOAT repair file found = %s.\n", fname);
  }
  if (fp == NULL)
  {
    cString = psConfig_.getParameter("MOAT_repair_file");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %s",winput1,winput2,repairFile);
      fp = fopen(repairFile, "r");
      if (fp != NULL)
        printf("MOAT repair file found = %s.\n", repairFile);
    }
  }
  if (fp == NULL)
  {
    if (fname != NULL)
      printOutTS(PL_INFO,"MOAT repair: repair file not found.\n");
    return 1;
  }
  
  //**/ -------------------------------------------------------------------
  //**/ read the first line
  //**/ -------------------------------------------------------------------
  fscanf(fp, "%s", inStr);
  if (strcmp(inStr, "BEGIN"))
  {
    printOutTS(PL_ERROR,"MOATSampling: wrong format in repair file.\n");
    printOutTS(PL_ERROR,"First  line : BEGIN\n");
    printOutTS(PL_ERROR,"Second line : nPatterns nInputs.\n");
    printOutTS(PL_ERROR,"Third  line : a list of input IDs (1-based).\n");
    printOutTS(PL_ERROR,"Fourth line : (and on) set of patterns.\n");
    printOutTS(PL_ERROR,"Last   line : END\n");
    fclose(fp);
    exit(1);
  }

  //**/ -------------------------------------------------------------------
  //**/ read the second line and check errors
  //**/ -------------------------------------------------------------------
  fscanf(fp, "%d %d", &nPatterns, &nInps);
  if (nPatterns <= 0 || nInps <= 0)
  {
    printOutTS(PL_ERROR,"MOATSampling: nPatterns or nInps <= 0.\n");
    fclose(fp);
    exit(1);
  }
  nSets = nPatterns / (nInps + 1);
  if (nSets*(nInps+1) != nPatterns)
  {
    printOutTS(PL_ERROR,
         "MOATSampling: nPatterns should be multiples of nInputs+1.\n");
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
      printOutTS(PL_ERROR,
           "MOATSampling ERROR: input index out of range (%d,%d)\n",
           vecInpList[ii], nInputs_);
      fclose(fp);
      exit(1);
    }
    for (jj = 0; jj < ii; jj++)
    {
      if (vecInpList[ii] == vecInpList[jj])
      {
        printOutTS(PL_ERROR,
             "MOATSampling ERROR: repeated index (%d)\n",vecInpList[ii]);
        fclose(fp);
        return 1;
      }
    }
  }

  //**/ -------------------------------------------------------------------
  //**/ read the pattern data
  //**/ -------------------------------------------------------------------
  vecPatterns.setLength(nPatterns*nInps);
  for (ii = 0;  ii < nPatterns; ii++)
  {
    for (jj = 0; jj < nInps; jj++) 
    {
      fscanf(fp, "%lg", &dtemp);
      vecPatterns[ii*nInps+jj] = dtemp;
    }
  }
  fscanf(fp, "%s", inStr);
  fclose(fp);

  //**/ the file must have an "END" after all pattern data
  if (strcmp(inStr, "END"))
  {
    printOutTS(PL_ERROR,
               "MOATSampling ERROR: wrong format in repair file.\n");
    printOutTS(PL_ERROR,"The file should end with END\n");
    exit(1);
  }

  //**/ --------------------------------------------------------------
  //**/ check to see if the incoming patterns are indeed MOAT
  //**/ --------------------------------------------------------------
  if (checkSample(nInps, nPatterns, vecPatterns.getDVector()) != 0)
  {
    printOutTS(PL_ERROR,
         "MOATSampling ERROR: pattern in repair file is not MOAT.\n");
    exit(1);
  }

  //**/ --------------------------------------------------------------
  //**/ search and replace
  //**/ --------------------------------------------------------------
  vecOneSam.setLength(nInps);
  for (ii = start; ii < nSamples_; ii+=nInputs_+1)
  {
    //**/ randomly choose a base pattern
    //**/ pindex = (PSUADE_rand() % nSets) * (nInps + 1);
    //**/ sequentially
    pindex = ii / (nInputs_ + 1) * (nInps + 1);

    //**/ copy the base pattern to the base sample
    for (kk = 0; kk < nInps; kk++)
    {
      iindex = vecInpList[kk] - 1;
      vecOneSam[kk] = vecSamInps_[ii*nInps+iindex];
      vecSamInps_[ii*nInps+iindex] = vecPatterns[pindex*nInps+kk];
    }

    //**/ substitute the rest of the current set
    for (jj = 1; jj <= nInputs_; jj++)
    {
      //**/ see if the current sample point involves changing the
      //**/ input to be repaired from the previous sample point
      for (kk = 0; kk < nInps; kk++)
      {
        iindex = vecInpList[kk] - 1;
        if (vecSamInps_[(ii+jj)*nInps+iindex] != vecOneSam[kk]) break; 
      }

      //**/ if so, increment the pointer to the repair pattern
      if (kk < nInps) pindex++;

      //**/ copy the current repair pattern to the updated sample point 
      for (kk = 0; kk < nInps; kk++)
      {
        iindex = vecInpList[kk] - 1;
        vecOneSam[kk] = vecSamInps_[(ii+jj)*nInps+iindex];
        vecSamInps_[(ii+jj)*nInps+iindex] = vecPatterns[pindex*nInps+kk]; 
      }
    }
  }

  //**/ --------------------------------------------------------------
  //**/ final check to see if the repaired sample is an MOAT sample
  //**/ --------------------------------------------------------------
  if (checkSample(nInputs_, nSamples_, vecSamInps_.getDVector()) != 0)
  {
    printOutTS(PL_ERROR,
               "MOATSampling ERROR: repaired file is not MOAT.\n");
    exit(1);
  }
  return 0;
}

//*************************************************************************
//* merge two MOAT designs (FOR 2 SETS OF INPUTS)
//*------------------------------------------------------------------------
int MOATSampling::merge()
{
  int    nInps1, nOuts1, nSamp1, nInps2, nSamp2, nOutputs, nReps, count; 
  int    samplingMethod, ii, ii2, rr, nSamples, nInputs, cnt1, cnt2;
  double *sampleInputs1, *sampleInputs2;
  char   **inpNames1, **inpNames2, **inpNames, file1[500], file2[500];
  char   pString[500];
  FILE   *fp1, *fp2;
  pData  pPtr1, pINames1;
  pData  pPtr2, pINames2;
  PsuadeData *psuadeIO1, *psuadeIO2;
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
    printOutTS(PL_ERROR,"ERROR : File %s not found.\n", file1);
    return 1;
  }
  else fclose(fp1);
  psuadeIO1 = new PsuadeData();
  psuadeIO1->setOutputLevel(0);
  if (psuadeIO1->readPsuadeFile(file1) != 0)
  {
    printOutTS(PL_ERROR,
         "MOAT ERROR : problem with reading file %s.\n", file1);
    delete psuadeIO1;
    return 1;
  }
  assert(psuadeIO1->getParameter("input_ninputs", pPtr1) == 0);
  nInps1 = pPtr1.intData_;
  assert(psuadeIO1->getParameter("input_names", pINames1) == 0);
  inpNames1 = pINames1.strArray_;
  assert(psuadeIO1->getParameter("output_noutputs", pPtr1) == 0);
  nOuts1 = pPtr1.intData_;
  assert(psuadeIO1->getParameter("method_sampling", pPtr1) == 0);
  samplingMethod = pPtr1.intData_;
  if (samplingMethod != PSUADE_SAMP_MOAT &&
      samplingMethod != PSUADE_SAMP_GMOAT)
  {
    printOutTS(PL_ERROR,"MOAT Merge ERROR : data1 is not MOAT.\n");
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
    printOutTS(PL_ERROR,
         "MOAT Merge ERROR : first sample is not MOAT.\n");
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
    printOutTS(PL_ERROR,"MOAT ERROR : File %s not found.\n", file2);
    delete psuadeIO1;
    pINames1.clean();
    return 1;
  }
  else fclose(fp2);
                                                                  
  psuadeIO2 = new PsuadeData();
  psuadeIO2->setOutputLevel(0);
  if (psuadeIO2->readPsuadeFile(file2) != 0)
  {
    printOutTS(PL_ERROR,
         "MOAT ERROR : problem with reading file %s.\n", file2);
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
  if (samplingMethod != PSUADE_SAMP_MOAT &&
      samplingMethod != PSUADE_SAMP_GMOAT)
  {
    printOutTS(PL_ERROR,"MOAT Merge ERROR : data2 is not MOAT.\n");
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
    printOutTS(PL_ERROR,"MOAT Merge ERROR : second sample is not MOAT.\n");
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
    printOutTS(PL_ERROR,
         "MOAT Merge ERROR : different number of replications.\n");
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
  pINames1.clean();
  pINames2.clean();
  delete psuadeIO1;
  delete psuadeIO2;
  for (ii = 0; ii < nInputs; ii++) delete [] inpNames[ii];
  delete [] inpNames;
  return 0;
}

//*************************************************************************
//* check whether the sample is MOAT (return 0 if yes)
//*------------------------------------------------------------------------
int MOATSampling::checkSample(int nInputs, int nSamples, double *X)
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
//**/  (input X is in a different format)
//*------------------------------------------------------------------------
int MOATSampling::checkSample2(int nInputs, int nSamples, double *X)
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
//* generate sample with constraints
//*------------------------------------------------------------------------
int MOATSampling::genRepair(int nInputs, double *lbounds, double *ubounds)
{
  int    status, nFiles, faFlag, kk, jj, ii, nPaths, currP, nTrials;
  int    count, trial, iInd, ind, ind2;
  double currX1, currX2, currX1F, currX2F, filterRange, currY1, currY2;
  double sLo, sHi, dtemp, ddata, *dPtr;
  char   pString[1000], winput[1000];
  FILE   *fp;
  FuncApprox **faPtrs;
  PsuadeData *ioPtr=NULL;
  pData      pPtr, pLower, pUpper;
  psVector   vecLThreshs, vecUThreshs, vecMoat, vecWT;
  psIVector  vecStates, vecIndSet;

  status = 0;
  snprintf(pString,100,"How many constraint data files are there (1-10)? ");
  nFiles = getInt(1, 10, pString);
  faFlag = 2;
  faPtrs = new FuncApprox*[nFiles];
  vecLThreshs.setLength(nFiles);
  vecUThreshs.setLength(nFiles);
  for (kk = 0; kk < nFiles; kk++)
  {
    snprintf(pString,100,"Enter name of file #%d : ", kk+1);
    getString(pString, winput);
    ioPtr = new PsuadeData;
    status = ioPtr->readPsuadeFile(winput);
    if (status != 0)
    {
      printOutTS(PL_ERROR,"moatgen READ ERROR: file = %s\n", winput);
      exit(1);
    }
    ioPtr->getParameter("input_ninputs", pPtr);
    jj = pPtr.intData_;
    if (jj != nInputs)
    {
      printOutTS(PL_ERROR,"moatgen ERROR: nInputs mismatch.\n");
      exit(1);
    }
    pLower.clean();
    ioPtr->getParameter("input_lbounds", pLower);
    for (ii = 0; ii < nInputs; ii++)
    {
      if (lbounds[ii] != pLower.dbleArray_[ii])
      {
        printOutTS(PL_ERROR,
             "MOAT genRepair ERROR: lower bound mismatch.\n");
        exit(1);
      }
    }
    pUpper.clean();
    ioPtr->getParameter("input_ubounds", pUpper);
    for (ii = 0; ii < nInputs; ii++)
    {
      if (ubounds[ii] != pUpper.dbleArray_[ii])
      {
        printOutTS(PL_ERROR,"moatgen ERROR: upper bound mismatch.\n");
        exit(1);
      }
    }
    faPtrs[kk] = genFAInteractive(ioPtr, faFlag);
    if (faPtrs[kk] == NULL) {printf("ERROR detected.\n"); exit(1);}
    faPtrs[kk]->setOutputLevel(printLevel_);
    snprintf(pString,100,"Constraint %d lower bound : ",kk+1);
    vecLThreshs[kk] = getDouble(pString);
    snprintf(pString,100,"Constraint %d upper bound : ",kk+1);
    vecUThreshs[kk] = getDouble(pString);
    if (vecLThreshs[kk] >= vecUThreshs[kk])
    {
      printf("ERROR : lower bound >= upper bound.\n");
      exit(1);;
    }
    delete ioPtr;
  }
  snprintf(pString,100,"Please enter the number of paths to search: ");
  nPaths = getInt(1, 1000, pString);
  snprintf(pString,100,"Please enter P (resolution: try 4-10) : ");
  currP = getInt(4, 10, pString);
  snprintf(pString,100,"Please enter the number of trials (> 100) : ");
  nTrials = getInt(101, 10000000, pString);
  vecMoat.setLength(nPaths*(nInputs+1)*nInputs_);
  vecWT.setLength(nInputs);
  vecIndSet.setLength(nInputs);
  count = 0;
  for (ii = 0; ii < nPaths; ii++)
  {
    trial = 0; 
    while (trial < nTrials)
    {
      iInd = count;
      trial++;
      for (jj = 0; jj < nInputs; jj++)
      {
        ind = PSUADE_rand() % currP;
        dtemp = ind * (ubounds[jj] - lbounds[jj]) / (currP - 1.0);
        vecMoat[iInd*nInputs+jj] = lbounds[jj] + dtemp;
      }
      for (kk = 0; kk < nFiles; kk++)
      {
        dPtr = vecMoat.getDVector();
        dtemp = faPtrs[kk]->evaluatePoint(&(dPtr[iInd*nInputs]));
        if (dtemp < vecLThreshs[ii] || dtemp > vecUThreshs[ii]) break;
      }
      if (kk != nFiles) continue;

      //**/ randomize the selection of next input
      generateRandomIvector(nInputs, vecIndSet.getIVector());

      //**/ construct the path
      for (jj = 0; jj < nInputs; jj++)
      {
        iInd++;
        //**/ always start from last point
        for (kk = 0; kk < nInputs; kk++)
          vecMoat[iInd*nInputs+kk] = vecMoat[(iInd-1)*nInputs+kk];

        //**/ pick the next input
        ind2 = vecIndSet[jj];

        //**/ see if both end within feasible region
        ddata = vecMoat[iInd*nInputs+ind2]; 
        currX1F = - PSUADE_UNDEFINED;
        currX2F =   PSUADE_UNDEFINED;
        for (kk = 0; kk < nFiles; kk++)
        {
          filterRange = vecUThreshs[kk] - vecLThreshs[kk];
          vecMoat[iInd*nInputs+ind2] = lbounds[ind2];
          dPtr = vecMoat.getDVector();
          currY1 = faPtrs[kk]->evaluatePoint(&(dPtr[iInd*nInputs]));
          vecMoat[iInd*nInputs+ind2] = ubounds[ind2];
          currY2 = faPtrs[kk]->evaluatePoint(&(dPtr[iInd*nInputs]));
          currX1 = lbounds[ind2];
          currX2 = ubounds[ind2];

          //**/ if not, the search range is empty
          if (currY2 >= vecLThreshs[kk] && currY1 >= vecUThreshs[kk])
            currX1 = currX2 = 0.0;
          else if (currY2 <= vecLThreshs[kk] && currY1 <= vecLThreshs[kk])
            currX1 = currX2 = 0.0;
          else if (currY2 > currY1)
          {
            //**/ binary search
            if (currY2 <= vecUThreshs[kk]) currX2 = ubounds[ind2];
            else
            {
              sLo = lbounds[ind2];
              sHi = ubounds[ind2];
              while (PABS((currY2-vecUThreshs[kk])/filterRange)>1e-4)
              {
                vecMoat[iInd*nInputs+ind2] = 0.5 * (sLo + sHi);
                dPtr = vecMoat.getDVector();
                currY2=faPtrs[kk]->evaluatePoint(&(dPtr[iInd*nInputs]));
                if (currY2 > vecUThreshs[kk]) sHi = 0.5 * (sLo+sHi);
                else                          sLo = 0.5 * (sLo+sHi);
              }
              currX2 = vecMoat[iInd*nInputs+ind2];
            }
            if (currY1 >= vecLThreshs[kk]) currX1 = lbounds[ind2];
            else
            {
              sLo = lbounds[ind2];
              sHi = ubounds[ind2];
              while (PABS((currY1-vecLThreshs[kk])/filterRange)>1e-4)
              {
                vecMoat[iInd*nInputs+ind2] = 0.5 * (sLo + sHi);
                dPtr = vecMoat.getDVector();
                currY1=faPtrs[kk]->evaluatePoint(&(dPtr[iInd*nInputs]));
                if (currY1 < vecLThreshs[kk]) sLo = 0.5 * (sLo+sHi);
                else                          sHi = 0.5 * (sLo+sHi);
              }
              currX1 = vecMoat[iInd*nInputs+ind2];
            }
          }
          else
          {
            if (currY1 <= vecUThreshs[kk]) currX1 = lbounds[ind2];
            else
            {
              sLo = lbounds[ind2];
              sHi = ubounds[ind2];
              while (PABS((currY1-vecUThreshs[kk])/filterRange)>1e-4)
              {
                vecMoat[iInd*nInputs+ind2] = 0.5 * (sLo + sHi);
                dPtr = vecMoat.getDVector();
                currY1=faPtrs[kk]->evaluatePoint(&(dPtr[iInd*nInputs]));
                if (currY1 > vecUThreshs[kk]) sLo = 0.5 * (sLo+sHi);
                else                          sHi = 0.5 * (sLo+sHi);
              }
              currX1 = vecMoat[iInd*nInputs+ind2];
            }
            if (currY2 >= vecLThreshs[kk]) currX2 = ubounds[ind2];
            else
            {
              sLo = lbounds[ind2];
              sHi = ubounds[ind2];
              while (PABS((currY2-vecLThreshs[kk])/filterRange)>1e-4)
              {
                vecMoat[iInd*nInputs+ind2] = 0.5 * (sLo + sHi);
                dPtr = vecMoat.getDVector();
                currY2=faPtrs[kk]->evaluatePoint(&(dPtr[iInd*nInputs]));
                if (currY2 < vecLThreshs[kk]) sHi = 0.5 * (sLo+sHi);
                else                          sLo = 0.5 * (sLo+sHi);
              }
              currX2 = vecMoat[iInd*nInputs+ind2];
            }
          }
          if (PABS(currX2-currX1)<0.1*(ubounds[ind2]-lbounds[ind2])) 
            break;
          if (currX1 > currX1F) currX1F = currX1;
          if (currX2 < currX2F) currX2F = currX2;
        }
        vecMoat[iInd*nInputs+ind2] = ddata;
        if (kk != nFiles) break;
        vecWT[ind2] = PABS(currX2F - currX1F) / (currP-1.0);
        vecMoat[iInd*nInputs+ind2] += vecWT[ind2];
        if (vecMoat[iInd*nInputs+ind2] > ubounds[ind2])
          dtemp = vecLThreshs[kk] - 1.0;
        else
        {
          for (kk = 0; kk < nFiles; kk++)
          {
            vecMoat[iInd*nInputs+ind2] = ddata;
            vecMoat[iInd*nInputs+ind2] += vecWT[ind2];
            dPtr = vecMoat.getDVector();
            dtemp = faPtrs[kk]->evaluatePoint(&(dPtr[iInd*nInputs]));
            if (dtemp < vecLThreshs[kk] || dtemp > vecUThreshs[kk]) 
            {
              vecMoat[iInd*nInputs+ind2] -= 2.0 * vecWT[ind2];
              if (vecMoat[iInd*nInputs+ind2] < lbounds[ind2])
                break;
              dtemp = faPtrs[kk]->evaluatePoint(&(dPtr[iInd*nInputs]));
              if (dtemp < vecLThreshs[kk] || dtemp > vecUThreshs[kk]) 
                break;
            }
          }
          vecMoat[iInd*nInputs+ind2] = ddata;
          if (kk != nFiles) break;
        }
      }
      if (jj == nInputs) 
      {
        count += (nInputs + 1);
        if (printLevel_ > 2)
          printOutTS(PL_INFO,
             "MOAT genRepair: path %d (out of %d) found.\n",ii+1,nPaths);
        break; 
      }
      else
      {
        if (printLevel_ > 2)
          printOutTS(PL_INFO,
               "Current path fails (%d out of max %d).\n",trial, nTrials); 
      }
    }
    if (trial >= nTrials)
    {
      printOutTS(PL_INFO,"moatgen fails to find all possible paths.\n");
      printOutTS(PL_INFO,"Suggestion: try a larger P than %d.\n", currP);
      break;
    }
  }
  for (ii = 0; ii < nPaths; ii++)
  {
    for (kk = 0; kk < nFiles; kk++)
    {
      dPtr = vecMoat.getDVector();
      dtemp = faPtrs[kk]->evaluatePoint(&(dPtr[ii*nInputs]));
      if (dtemp < vecLThreshs[kk] || dtemp > vecUThreshs[kk])
      printOutTS(PL_ERROR,
           "MOAT genRepair:sample %d fails final test (%e <? %e <? %e).\n",
           ii, vecLThreshs[kk], dtemp, vecUThreshs[kk]);
    }
  }
  for (kk = 0; kk < nFiles; kk++) delete faPtrs[kk];
  delete [] faPtrs;
  if (trial >= nTrials)
  {
    printOutTS(PL_ERROR,"MOAT genRepair FAILS.\n");
    return 0; 
  }
  fp = fopen("MOAT_repair_file", "w");
  // Add a check for NULL by Bill Oliver
  if(fp != NULL)
  {
    fprintf(fp, "BEGIN\n");
    fprintf(fp, "%d %d\n", nPaths*(nInputs+1), nInputs);
    for (ii = 0; ii < nInputs; ii++) fprintf(fp, "%d ", ii+1);
    fprintf(fp, "\n");
    for (ii = 0; ii < nPaths*(nInputs+1); ii++)
    {
      for (jj = 0; jj < nInputs; jj++)
        fprintf(fp, "%e ", vecMoat[ii*nInputs+jj]);
      fprintf(fp, "\n");
    }
    fprintf(fp, "END\n");
    fclose(fp);
  }
  count = nPaths * (nInputs + 1); 
  vecWT.setLength(count*nInputs);
  vecStates.setLength(count);
  for (ii = 0; ii < count; ii++) vecStates[ii] = 1;
  for (ii = 0; ii < count; ii++)
    for (jj = 0; jj < nInputs; jj++)
      vecWT[ii*nInputs+jj] = vecMoat[ii*nInputs+jj];
  printOutTS(PL_INFO,"MOAT genRepair: check for repeated sample points.\n");
  for (ii = 0; ii < count; ii++)
  {
    status = compareSamples(ii,count,nInputs, vecWT.getDVector(), 
                            vecStates.getIVector());
    if (status >= 0)
      printOutTS(PL_INFO,
           "MOAT genRepair check: sample %d and %d are identical.\n",
           ii+1,status+1);
  }
  printOutTS(PL_INFO,
       "MOAT genRepair: repair file created in MOAT_repair_file.\n");
  printOutTS(PL_INFO,"         Make sure to change the input indices.\n");
  return 0;
}

//*************************************************************************
//* generate sample with response/surface constraints
//*------------------------------------------------------------------------
int MOATSampling::genRepair(PsuadeData *psIO)
{
  int    status, sInd, faFlag, jj, ii, kk, ind, nPaths, currP, nTrials;
  int    outputID, count, trial, iInd, ind2, nInputs, nOutputs, nSamples;
  double threshL, threshU, dtemp, ddata, currX1, currX2, filterRange, currY1;
  double currY2, sLo, sHi, Ymax, Ymin, *sampleOutputs, *iLowerB, *iUpperB;
  double *dPtr;
  char   pString[1000];
  FILE   *fp;
  FuncApprox *faPtr;
  pData      pPtr, pLower, pUpper;
  psIVector  vecIndSet, vecStates;
  psVector   vecWT, vecU, vecL, vecMoat;

  faFlag = 3;
  faPtr = genFAInteractive(psIO, faFlag);
  if (faPtr == NULL) 
  {
    printOutTS(PL_ERROR,"ERROR detected.\n"); 
    return 0;
  }
  faPtr->setOutputLevel(printLevel_);
  psIO->getParameter("ana_outputid", pPtr);
  outputID = pPtr.intData_;
  Ymax = - 1.0e35;
  Ymin =   1.0e35;
  psIO->getParameter("input_ninputs", pPtr);
  nInputs = pPtr.intData_;
  psIO->getParameter("output_noutputs", pPtr);
  nOutputs = pPtr.intData_;
  psIO->getParameter("method_nsamples", pPtr);
  nSamples = pPtr.intData_;
  psIO->getParameter("output_sample", pPtr);
  sampleOutputs  = pPtr.dbleArray_;

  pLower.clean();
  psIO->getParameter("input_lbounds", pLower);
  iLowerB = pLower.dbleArray_;
  pUpper.clean();
  psIO->getParameter("input_ubounds", pUpper);
  iUpperB = pUpper.dbleArray_;
  for (sInd = 0; sInd < nSamples; sInd++)
  {
    if (vecSamOuts_[sInd*nOutputs+outputID] > Ymax)
      Ymax = vecSamOuts_[sInd*nOutputs+outputID];
    if (vecSamOuts_[sInd*nOutputs+outputID] < Ymin)
      Ymin = vecSamOuts_[sInd*nOutputs+outputID];
  }
  snprintf(pString,100,
          "Please enter the lower bound constraint (Ymin=%e) : ",Ymin);
  threshL = getDouble(pString);
  snprintf(pString,100,
          "Please enter the upper bound constraint (Ymax=%e) : ",Ymax);
  threshU = getDouble(pString);
  if (threshL >= threshU)
  {
    printf("ERROR : lower bound >= upper bound.\n");
    // Cleanup by Bill Oliver
    delete faPtr;
    return 0;
  }
  snprintf(pString,100,"Please enter the number of paths to search: ");
  nPaths = getInt(1, 1000, pString);
  snprintf(pString,100,"Please enter P (resolution: try 4-10) : ");
  currP = getInt(4, 10, pString);
  snprintf(pString,100,"Please enter the number of trials (> 100) : ");
  nTrials = getInt(101, 10000000, pString);
  vecMoat.setLength(nPaths*(nInputs+1)*nInputs);
  vecWT.setLength(nInputs);
  vecIndSet.setLength(nInputs);
  count = 0;
  filterRange = threshU - threshL;
  for (ii = 0; ii < nPaths; ii++)
  {
    trial = 0; 
    while (trial < nTrials)
    {
      iInd = count;
      trial++;
      for (jj = 0; jj < nInputs; jj++)
      {
        ind = PSUADE_rand() % currP;
        dtemp = ind * (iUpperB[jj] - iLowerB[jj]) / (currP - 1.0);
        vecMoat[iInd*nInputs_+jj] = iLowerB[jj] + dtemp;
      }
      dPtr = vecMoat.getDVector();
      dtemp = faPtr->evaluatePoint(&(dPtr[iInd*nInputs_]));
      if (dtemp < threshL || dtemp > threshU) continue;

      //**/ randomize the selection of next input
      generateRandomIvector(nInputs, vecIndSet.getIVector());

      //**/ construct the path
      for (jj = 0; jj < nInputs; jj++)
      {
        iInd++;
        //**/ always start from last point
        for (kk = 0; kk < nInputs; kk++)
          vecMoat[iInd*nInputs_+kk] = vecMoat[(iInd-1)*nInputs_+kk];

        //**/ pick the next input
        ind2 = vecIndSet[jj];

        //**/ see if both end within feasible region
        ddata = vecMoat[iInd*nInputs_+ind2]; 
        vecMoat[iInd*nInputs_+ind2] = iLowerB[ind2];
        dPtr = vecMoat.getDVector();
        currY1 = faPtr->evaluatePoint(&(dPtr[iInd*nInputs_]));
        vecMoat[iInd*nInputs_+ind2] = iUpperB[ind2];
        currY2 = faPtr->evaluatePoint(&(dPtr[iInd*nInputs_]));
        currX1 = iLowerB[ind2];
        currX2 = iUpperB[ind2];
        vecMoat[iInd*nInputs_+ind2] = ddata;

        //**/ if not, the search range is empty
        if (currY2 >= threshU && currY1 >= threshU)
          currX1 = currX2 = 0.0;
        else if (currY2 <= threshL && currY1 <= threshL)
          currX1 = currX2 = 0.0;
        else if (currY2 > currY1)
        {
          //**/ binary search
          if (currY2 <= threshU) currX2 = iUpperB[ind2];
          else
          {
            sLo = iLowerB[ind2];
            sHi = iUpperB[ind2];
            while (PABS((currY2-threshU)/filterRange)>0.0001)
            {
              vecMoat[iInd*nInputs_+ind2] = 0.5 * (sLo + sHi);
              dPtr = vecMoat.getDVector();
              currY2 = faPtr->evaluatePoint(&(dPtr[iInd*nInputs_]));
              if (currY2 > threshU) sHi = 0.5 * (sLo + sHi);
              else                  sLo = 0.5 * (sLo + sHi);
            }
            currX2 = vecMoat[iInd*nInputs_+ind2];
          }
          if (currY1 >= threshL) currX1 = iLowerB[ind2];
          else
          {
            sLo = iLowerB[ind2];
            sHi = iUpperB[ind2];
            while (PABS((currY1-threshL)/filterRange)>0.0001)
            {
              vecMoat[iInd*nInputs_+ind2] = 0.5 * (sLo + sHi);
              dPtr = vecMoat.getDVector();
              currY1 = faPtr->evaluatePoint(&(dPtr[iInd*nInputs_]));
              if (currY1 < threshL) sLo = 0.5 * (sLo + sHi);
              else                  sHi = 0.5 * (sLo + sHi);
            }
            currX1 = vecMoat[iInd*nInputs_+ind2];
          }
        }
        else
        {
          if (currY1 <= threshU) currX1 = iLowerB[ind2];
          else
          {
            sLo = iLowerB[ind2];
            sHi = iUpperB[ind2];
            while (PABS((currY1-threshU)/filterRange)>0.0001)
            {
              vecMoat[iInd*nInputs_+ind2] = 0.5 * (sLo + sHi);
              dPtr = vecMoat.getDVector();
              currY1 = faPtr->evaluatePoint(&(dPtr[iInd*nInputs_]));
              if (currY1 > threshU) sLo = 0.5 * (sLo + sHi);
              else                  sHi = 0.5 * (sLo + sHi);
            }
            currX1 = vecMoat[iInd*nInputs_+ind2];
          }
          if (currY2 >= threshL) currX2 = iUpperB[ind2];
          else
          {
            sLo = iLowerB[ind2];
            sHi = iUpperB[ind2];
            while (PABS((currY2-threshL)/filterRange)>0.0001)
            {
              vecMoat[iInd*nInputs_+ind2] = 0.5 * (sLo + sHi);
              dPtr = vecMoat.getDVector();
              currY2 = faPtr->evaluatePoint(&(dPtr[iInd*nInputs_]));
              if (currY2 < threshL) sHi = 0.5 * (sLo + sHi);
              else                  sLo = 0.5 * (sLo + sHi);
            }
            currX2 = vecMoat[iInd*nInputs_+ind2];
          }
        }
        if (PABS(currX2-currX1)<0.1*(iUpperB[ind2]-iLowerB[ind2])) 
          break;
        vecMoat[iInd*nInputs_+ind2] = ddata;
        vecWT[ind2] = PABS(currX2 - currX1) / (currP-1.0);
        vecMoat[iInd*nInputs_+ind2] += vecWT[ind2];
        if (vecMoat[iInd*nInputs_+ind2] > iUpperB[ind2])
          dtemp = threshL - 1.0;
        else 
        {
          dPtr = vecMoat.getDVector();
          dtemp = faPtr->evaluatePoint(&(dPtr[iInd*nInputs_]));
        }
        if (dtemp < threshL || dtemp > threshU) 
        {
          vecMoat[iInd*nInputs_+ind2] -= 2.0 * vecWT[ind2];
          if (vecMoat[iInd*nInputs_+ind2] < iLowerB[ind2])
            break;
          dPtr = vecMoat.getDVector();
          dtemp = faPtr->evaluatePoint(&(dPtr[iInd*nInputs_]));
          if (dtemp < threshL || dtemp > threshU) 
            break;
        }
      }
      if (jj == nInputs) 
      {
        count += (nInputs + 1);
        if (printLevel_ > 2)
          printOutTS(PL_INFO,"moatgen: path %d (out of %d) found.\n", 
               ii+1, nPaths);
        break; 
      }
      else
      {
        if (printLevel_ > 2)
          printOutTS(PL_INFO,"Current path fails (%d out of max %d).\n",
                     trial, nTrials); 
      }
    }
    if (trial >= nTrials)
    {
      printOutTS(PL_INFO,
            "MOAT genRepair FAILS to find all possible paths.\n");
      printOutTS(PL_INFO,
            "Suggestion: try a larger P than %d.\n", currP);
    }
  }
  for (ii = 0; ii < nPaths; ii++)
  {
    dPtr = vecMoat.getDVector();
    dtemp = faPtr->evaluatePoint(&(dPtr[ii*nInputs_]));
    if (dtemp < threshL || dtemp > threshU)
      printOutTS(PL_INFO,
          "MOAT genRepair:sample %d fails final test (%e <? %e <? %e).\n",
          ii, threshL, dtemp, threshU);
  }
  if (trial >= nTrials)
  {
    delete faPtr;
    return 0;
  }
  fp = fopen("MOAT_repair_file", "w");
  // add a check for NULL by Bill Oliver
  if(fp != NULL)
  {
    fprintf(fp, "BEGIN\n");
    fprintf(fp, "%d %d\n", nPaths*(nInputs+1), nInputs);
    for (ii = 0; ii < nInputs; ii++) fprintf(fp, "%d ", ii+1);
    fprintf(fp, "\n");
    for (ii = 0; ii < nPaths*(nInputs+1); ii++)
    {
      for (jj = 0; jj < nInputs; jj++)
        fprintf(fp, "%e ", vecMoat[ii*nInputs_+jj]);
      fprintf(fp, "\n");
    }
    fprintf(fp, "END\n");
    fclose(fp);
  }
  count = nPaths * (nInputs + 1); 
  vecWT.setLength(count*nInputs);
  vecStates.setLength(count);
  for (ii = 0; ii < count; ii++) vecStates[ii] = 1;
  for (ii = 0; ii < count; ii++)
    for (jj = 0; jj < nInputs; jj++)
      vecWT[ii*nInputs+jj] = vecMoat[ii*nInputs_+jj];
  printOutTS(PL_INFO,
       "MOAT genRepair: check for repeated sample points.\n");
  for (ii = 0; ii < count; ii++)
  {
    status = compareSamples(ii,count,nInputs, vecWT.getDVector(), 
                            vecStates.getIVector());
    if (status >= 0)
      printf("moatgen check: sample %d and %d are identical.\n",
             ii+1,status+1);
  }
  printOutTS(PL_INFO,
       "MOAT genRepair: repair file created in MOAT_repair_file.\n");
  printOutTS(PL_INFO,
       "                Make sure to change the input indices.\n");
  delete faPtr;
  return 0;
}

//*************************************************************************
//* initialize the sampling data
//*------------------------------------------------------------------------
int MOATSampling::initializeHighDimension()
{
  int nReps, ii, rr, ind1, ind2, kk, count, ii2, nn, currBin;
  psVector  vecRanges, vecS1, vecS2, vecXT;
  psIVector vecSI, vecBins;

  //**/ ----------------------------------------------------------------
  //**/ preparation for generating the sample
  //**/ ----------------------------------------------------------------
  nReps = nSamples_ / (nInputs_ + 1);
  vecSamInps_.setLength(nSamples_*nInputs_);
  vecSamOuts_.setLength(nSamples_*nOutputs_);
  vecSamStas_.setLength(nSamples_);

  vecRanges.setLength(nInputs_);
  for (ii = 0;  ii < nInputs_;  ii++) 
    vecRanges[ii] = vecUBs_[ii] - vecLBs_[ii];

  //**/ ----------------------------------------------------------------
  //**/ generate an MOAT sample
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
      if (kk == 0) ind2 = ind1 - P_ / 2;
      else         ind2 = ind1 + P_ / 2;
      if      (ind2 < 0)    ind2 = ind2 + P_;
      else if (ind2 > P_-1) ind2 = ind2 - P_;
      vecS2[ii] = vecRanges[ii] / (P_ - 1) * ind2 + vecLBs_[ii]; 
    }
    for (ii = 0; ii < nInputs_; ii++)
      vecSamInps_[rr*(nInputs_+1)*nInputs_+ii] = vecS1[ii];
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
        vecXT[rr*2] = vecSamInps_[rr*(nInputs_+1)*nInputs_+ii];
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
          printOutTS(PL_INFO,
               "MOAT: input %3d - level = %12.4e, # times = %4d\n",
               ii+1, vecXT[ii2-1], nn);
          vecBins[currBin++] += nn;
          nn = 1;
        } else nn++;
      }
      printOutTS(PL_INFO,
           "MOAT: input %3d - level = %12.4e, # times = %4d\n",
           ii+1, vecXT[ii2-1], nn);
      vecBins[currBin++] += nn;
    }
    for (ii = 0; ii < P_; ii++) 
      printOutTS(PL_INFO,
           "MOAT: frequency of visit to bin %5d = %d\n",ii+1,vecBins[ii]);
  }
  return 0;
}

// ************************************************************************
// equal operator modified by Bill Oliver
// ------------------------------------------------------------------------
MOATSampling& MOATSampling::operator=(const MOATSampling & ms)
{
  if(this == &ms) return *this;
  samplingID_ = ms.samplingID_;
  P_ = ms.P_;
  nInputs_ = ms.nInputs_;
  vecInpSubset_ = ms.vecInpSubset_;

  //**/MetisSampling inherits from Sampling so include the parent class 
  //**/data members
  printLevel_ = ms.printLevel_;
  samplingID_ = ms.samplingID_;
  nSamples_ = ms.nSamples_;
  nOutputs_ = ms.nOutputs_;
  randomize_ = ms.randomize_;
  nReplications_ = ms.nReplications_;
  vecLBs_ = ms.vecLBs_;
  vecUBs_ = ms.vecUBs_;
  vecSamInps_ = ms.vecSamInps_;
  vecSamOuts_ = ms.vecSamOuts_;
  vecSamStas_ = ms.vecSamStas_;
  return (*this);
}

