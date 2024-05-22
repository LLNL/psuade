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
// Functions for the fractional factorial sampling class 
// AUTHOR : CHARLES TONG
// DATE   : 2005
// ************************************************************************
#include <stdio.h>
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "FractFactSampling.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
FractFactSampling::FractFactSampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_FF4;
  resolution_ = 4;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
FractFactSampling::~FractFactSampling()
{
}

// ************************************************************************
// initialize the sampler data
// ------------------------------------------------------------------------
int FractFactSampling::initialize(int initLevel)
{
  int maxInputs, nInputCnt, ii, increment, count, sampleID, inputID;
  int ss, nterms, kk;
  psIVector vecIndices;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  //if (nSamples_ == 0)
  //{
  //  printf("FractFactSampling::initialize ERROR - nSamples = 0.\n");
  //  exit(1);
  //}
  if (nInputs_ == 0)
  {
    printf("FractFactSampling::initialize ERROR - input not set up.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ clean up and allocate new storage
  //**/ ----------------------------------------------------------------
  vecSamInps_.setLength(0);
  vecSamOuts_.setLength(0);
  vecSamStas_.setLength(0);
  if (initLevel != 0) return 0;

  //**/ ----------------------------------------------------------------
  //**/ need to compute the sample size
  //**/ ----------------------------------------------------------------
  if (nInputs_ <= resolution_-1) nInputCnt = nInputs_;
  else
  {
    nInputCnt = resolution_ - 2;
    maxInputs = 0;
    while (maxInputs < nInputs_)
    {
      nInputCnt++;
      increment = resolution_ - 2;
      maxInputs = nInputCnt;
      while (nInputCnt-increment > 0)
      {
        increment++;
        count = 1;
        for (ii = nInputCnt; ii > nInputCnt-increment; ii--)
          count = count * ii / (nInputCnt - ii + 1); 
        maxInputs += count;
      }
    }
  }
  if (nInputCnt > 12)
  {
    printf("FractFactSampling initialize ERROR - nInputs > %d not",
           nInputs_);
    printf(" supported.\n");
    exit(1);
  }
  nSamples_ = (int) pow(2.0001, (double) nInputCnt);
  allocSampleData();

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 3)
  {
    printf("FractFactSampling initialize: nSamples = %d\n", nSamples_);
    printf("FractFactSampling initialize: nInputs  = %d\n", nInputs_);
    printf("FractFactSampling initialize: nOutputs = %d\n", nOutputs_);
    if (printLevel_ > 4)
      for (inputID = 0; inputID < nInputs_; inputID++)
        printf("    FractFactSampling input %3d = [%e %e]\n",inputID+1,
               vecLBs_[inputID], vecUBs_[inputID]);
  }

  //**/ ----------------------------------------------------------------
  //**/ first fill the sample pattern using generators
  //**/ ----------------------------------------------------------------
  for (ii = 0; ii < nInputCnt; ii++)
  {
    increment = (int) pow(2.000001, ii+1);
    for (sampleID = 0; sampleID < nSamples_; sampleID+=increment)
    {
      for (ss = 0; ss < increment/2; ss++)
        vecSamInps_[(sampleID+ss)*nInputs_+ii] = -1;
      for (ss = 0; ss < increment/2; ss++)
        vecSamInps_[(sampleID+increment/2+ss)*nInputs_+ii] = 1;
    }
  }
  nterms = resolution_ - 1;
  vecIndices.setLength(12);
  for (ii = 0; ii < nterms; ii++) vecIndices[ii] = ii;
  count = nInputCnt;
  while (count < nInputs_)
  {
    for (sampleID = 0; sampleID < nSamples_; sampleID++)
    {
      vecSamInps_[sampleID*nInputs_+count] = 
          vecSamInps_[sampleID*nInputs_+vecIndices[0]] * 
          vecSamInps_[sampleID*nInputs_+vecIndices[1]] * 
          vecSamInps_[sampleID*nInputs_+vecIndices[2]]; 
      for (kk = 3; kk < nterms; kk++)
        vecSamInps_[sampleID*nInputs_+count] *= 
          vecSamInps_[sampleID*nInputs_+vecIndices[kk]]; 
    }
    kk = nterms - 1;
    while (kk >= 0 && vecIndices[kk] >= nInputCnt+kk-nterms) kk--;
    if (kk < 0)
    {
      nterms++;
      for (ii = 0; ii < nterms; ii++) vecIndices[ii] = ii;
    }
    else
    {
      vecIndices[kk]++;
      for (ii = kk+1; ii < nterms; ii++) 
        vecIndices[ii] = vecIndices[ii-1] + 1;
    }
    count++;
  }
   
  //**/ ----------------------------------------------------------------
  //**/ generate the actual sample data
  //**/ ----------------------------------------------------------------
  for (sampleID = 0; sampleID < nSamples_; sampleID++)
  {
    for (inputID = 0; inputID < nInputs_; inputID++) 
    {
      if (vecSamInps_[sampleID*nInputs_+inputID] == -1)
        vecSamInps_[sampleID*nInputs_+inputID] = vecLBs_[inputID];
      else if (vecSamInps_[sampleID*nInputs_+inputID] == 1)
        vecSamInps_[sampleID*nInputs_+inputID] = vecUBs_[inputID];
    }
  }
  return 0;
}

// ************************************************************************
// refine the sample space
// ------------------------------------------------------------------------
int FractFactSampling::refine(int refineRatio, int randomize, double thresh,
                               int nSamples, double *sampleErrors)
{
  (void) refineRatio;
  (void) randomize;
  (void) thresh;
  (void) nSamples;
  (void) sampleErrors;

  printf("FractFactSampling::refine ERROR - not available.\n");
  exit(1);
  return 0;
}

// ************************************************************************
// set internal scheme
// ------------------------------------------------------------------------
int FractFactSampling::setParam(char *sparam)
{
  char winput[1001];
  sscanf(sparam, "%s", winput);
  if (!strcmp(winput, "setResolution"))
  {
    sscanf(sparam, "%s %d", winput, &resolution_);
    if (resolution_ != 4 && resolution_ != 5) resolution_ = 4;
    if (resolution_ == 4) samplingID_ = PSUADE_SAMP_FF4;
    else                  samplingID_ = PSUADE_SAMP_FF5;
  }
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
FractFactSampling& FractFactSampling::operator=(const FractFactSampling &)
{
  printf("FractFactSampling operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

