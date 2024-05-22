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
// Functions for the replicated fractional factorial sampling class 
// AUTHOR : CHARLES TONG
// DATE   : 2012
// ************************************************************************
#include <stdio.h>
#include <sstream>
//**/using namespace std;

#include "Psuade.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "RFractFactSampling.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
RFractFactSampling::RFractFactSampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_RFF4;
  resolution_ = 4;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
RFractFactSampling::~RFractFactSampling()
{
}

// ************************************************************************
// initialize the sampler data
// ------------------------------------------------------------------------
int RFractFactSampling::initialize(int initLevel)
{
  int    maxInputs, nInputCnt, ii, increment, count, nn;
  int    ss, iii, nterms, kk, rr, nReps, offset;
  double width;
  char   pString[500];
  psIVector vecIndices, vecIT;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  if (nSamples_ == 0)
  {
    printf("RFractFactSampling initialize ERROR: nSamples = 0.\n");
    exit(1);
  }
  if (nInputs_ == 0)
  {
    printf("RFractFactSampling initialize ERROR: input not set up.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ clean up and allocate new storage
  //**/ ----------------------------------------------------------------
  vecIndices.setLength(12);
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
    printf("RFractFactSampling::initialize ERROR - nInputs > %d not",
           nInputs_);
    printf(" supported.\n");
    exit(1);
  }
  nSamples_ = (int) pow(2.0001, (double) nInputCnt);
  nReps = 1;
  printf("Replicated FF INFO: nSamples for one replication = %d\n",
         nSamples_);
  snprintf(pString,100,"How many replications ? (1 - 10000) : ");
  nReps = getInt(1, 10000, pString);

  nSamples_ *= nReps;
  allocSampleData();

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 3)
  {
    printf("RFractFactSampling initialize: nSamples = %d\n",nSamples_);
    printf("RFractFactSampling initialize: nInputs  = %d\n",nInputs_);
    printf("RFractFactSampling initialize: nOutputs = %d\n",nOutputs_);
    if (printLevel_ > 4)
      for (ii = 0; ii < nInputs_; ii++)
        printf("    RFractFactSampling input %3d = [%e %e]\n", ii+1,
               vecLBs_[ii], vecUBs_[ii]);
  }

  //**/ ----------------------------------------------------------------
  //**/ first fill the sample pattern using generators
  //**/ ----------------------------------------------------------------
  vecIT.setLength(nInputs_);
  for (rr = 0; rr < nReps; rr++)
  {
    generateRandomIvector(nInputs_, vecIT.getIVector());
    offset = rr * nSamples_ / nReps;
    for (iii = 0; iii < nInputCnt; iii++)
    {
      ii = vecIT[iii];
      increment = (int) pow(2.000001, iii+1);
      for (nn = 0; nn < nSamples_/nReps; nn+=increment)
      {
        for (ss = 0; ss < increment/2; ss++)
          vecSamInps_[(offset+nn+ss)*nInputs_+ii] = -1;
        for (ss = 0; ss < increment/2; ss++)
          vecSamInps_[(offset+nn+increment/2+ss)*nInputs_+ii] = 1;
      }
    }
    nterms = resolution_ - 1;
    for (ii = 0; ii < nterms; ii++) vecIndices[ii] = ii;
    count = nInputCnt;
    while (count < nInputs_)
    {
      for (nn = 0; nn < nSamples_/nReps; nn++)
      {
        vecSamInps_[(offset+nn)*nInputs_+vecIT[count]] = 
             vecSamInps_[(offset+nn)*nInputs_+vecIT[vecIndices[0]]] * 
             vecSamInps_[(offset+nn)*nInputs_+vecIT[vecIndices[1]]] * 
             vecSamInps_[(offset+nn)*nInputs_+vecIT[vecIndices[2]]]; 
        for (kk = 3; kk < nterms; kk++)
          vecSamInps_[(offset+nn)*nInputs_+vecIT[count]] *= 
              vecSamInps_[(offset+nn)*nInputs_+vecIT[vecIndices[kk]]]; 
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
  }
   
  //**/ ----------------------------------------------------------------
  //**/ generate the actual sample data
  //**/ ----------------------------------------------------------------
  for (rr = 0; rr < nReps; rr++)
  {
    offset = rr * nSamples_ / nReps;
    kk = PSUADE_rand() % 2;
    if (nReps == 1) kk = 0;
    for (nn = 0; nn < nSamples_/nReps; nn++)
    {
      for (ii = 0; ii < nInputs_; ii++) 
      {
        if (nReps == 1) width = (vecUBs_[ii] - vecLBs_[ii]); 
        else            width = 0.75 * (vecUBs_[ii] - vecLBs_[ii]);
        if (vecSamInps_[(offset+nn)*nInputs_+ii] == -1)
        {
          if (kk == 0) 
               vecSamInps_[(offset+nn)*nInputs_+ii] = vecLBs_[ii];
          else vecSamInps_[(offset+nn)*nInputs_+ii] = vecLBs_[ii]+width/3.0;
        }
        else if (vecSamInps_[(offset+nn)*nInputs_+ii] == 1)
        {
          if (kk == 0) 
               vecSamInps_[(offset+nn)*nInputs_+ii] = vecLBs_[ii] + width;
          else vecSamInps_[(offset+nn)*nInputs_+ii] = vecUBs_[ii];
        }
      }
    }
  }
  return 0;
}

// ************************************************************************
// refine the sample space
// ------------------------------------------------------------------------
int RFractFactSampling::refine(int, int, double, int, double *)
{
  printf("RFractFactSampling refine ERROR: not available.\n");
  exit(1);
  return 0;
}

// ************************************************************************
// set internal scheme
// ------------------------------------------------------------------------
int RFractFactSampling::setParam(char *sparam)
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
RFractFactSampling& RFractFactSampling::operator=(const RFractFactSampling&)
{
  printf("RFractFactSampling operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

