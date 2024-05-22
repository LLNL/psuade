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
// Functions for the Central Composite sampling class 
// AUTHOR : CHARLES TONG
// DATE   : 2005
// ************************************************************************
#include <stdio.h>
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "CentralCompositeSampling.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
CentralCompositeSampling::CentralCompositeSampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_CCI4;
  scheme_ = 0;
  resolution_ = 4;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
CentralCompositeSampling::~CentralCompositeSampling()
{
}

// ************************************************************************
// initialize the sampler data
// ------------------------------------------------------------------------
int CentralCompositeSampling::initialize(int initLevel)
{
  int    nSamples, inputID, inputID2, sampleID, outputID;
  double alpha;
  Sampling *samplePtr;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  //if (nSamples_ == 0)
  //{
  //  printf("CentralCompositeSampling initialize ERROR: nSamples = 0.\n");
  //  exit(1);
  //}
  if (nInputs_ == 0)
  {
    printf("CentralCompositeSampling initialize ERROR: no inputs.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  if (initLevel == 1) return 0;

  //**/ ----------------------------------------------------------------
  //**/ create base sampling object and create sample matrix
  //**/ ----------------------------------------------------------------
  if (resolution_ == 4)
       samplePtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_FF4);
  else if (resolution_ == 5)
       samplePtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_FF5);
  else
  {
    samplePtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_FACT);
    nSamples = 1;
    for (inputID = 0; inputID < nInputs_; inputID++) nSamples *= 2;
  }
  samplePtr->setInputBounds(nInputs_,vecLBs_.getDVector(),
                            vecUBs_.getDVector());
  samplePtr->setOutputParams(nOutputs_);
  samplePtr->setSamplingParams(nSamples, 1, 0);
  samplePtr->initialize(0);
  nSamples = samplePtr->getNumSamples();

  //**/ ----------------------------------------------------------------
  //**/ retrieve the intermediate sample
  //**/ ----------------------------------------------------------------
  psVector  vecDX, vecDY;
  psIVector vecDS;
  vecDX.setLength(nSamples*nInputs_);
  vecDY.setLength(nSamples*nOutputs_);
  vecDS.setLength(nSamples);
  samplePtr->getSamples(nSamples,nInputs_,nOutputs_,vecDX.getDVector(), 
                        vecDY.getDVector(), vecDS.getIVector());

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  nSamples_ = nSamples + 2 * nInputs_ + 1;
  if (printLevel_ > 4)
  {
    printf("CentralCompositeSampling initialize: nSamples = %d\n",
           nSamples_);
    printf("CentralCompositeSampling initialize: nInputs  = %d\n",
           nInputs_);
    printf("CentralCompositeSampling initialize: nOutputs = %d\n",
           nOutputs_);
    for (inputID = 0; inputID < nInputs_; inputID++)
      printf("   CentralCompositeSampling input %3d = [%e %e]\n",
             inputID+1, vecLBs_[inputID], vecUBs_[inputID]);
  }

  //**/ ----------------------------------------------------------------
  //**/ generate final sample
  //**/ ----------------------------------------------------------------
  allocSampleData();
  if (scheme_ == 0) alpha = pow((double) nSamples, 0.25);
  else              alpha = 1.0;

  for (sampleID = 0; sampleID < nSamples; sampleID++)
  {
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      //**/ no change for CCF and CCC (shrink for CCI)
      if (alpha == 1.0)
         vecSamInps_[sampleID*nInputs_+inputID] = 
              vecDX[sampleID*nInputs_+inputID];
      else
      {
        if (vecDX[sampleID*nInputs_+inputID]==vecLBs_[inputID])
          vecSamInps_[sampleID*nInputs_+inputID] = 0.5 * 
               (vecLBs_[inputID] + vecUBs_[inputID]) + 
               0.5/alpha*(vecLBs_[inputID]-vecUBs_[inputID]); 
        else
          vecSamInps_[sampleID*nInputs_+inputID] = 0.5 *  
               (vecLBs_[inputID] + vecUBs_[inputID]) + 
               0.5/alpha*(vecUBs_[inputID]-vecLBs_[inputID]); 
      }
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ add the star pattern (expand only for CCC)
  //**/ ----------------------------------------------------------------
  if (scheme_ == 2) alpha = pow((double) nSamples, 0.25);
  else              alpha = 1.0;

  for (inputID = 0; inputID < nInputs_; inputID++)
  {
    for (inputID2 = 0; inputID2 < nInputs_; inputID2++)
    {
      if (inputID == inputID2)
      {
        vecSamInps_[(nSamples+2*inputID)*nInputs_+inputID2] = 0.5 *  
              (vecLBs_[inputID] + vecUBs_[inputID]) + 
              0.5*alpha*(vecLBs_[inputID]-vecUBs_[inputID]); 
        vecSamInps_[(nSamples+2*inputID+1)*nInputs_+inputID2] = 0.5 * 
              (vecLBs_[inputID] + vecUBs_[inputID]) + 
              0.5*alpha*(vecUBs_[inputID]-vecLBs_[inputID]); 
      }
      else
      {
        vecSamInps_[(nSamples+2*inputID)*nInputs_+inputID2] = 0.5 * 
                         (vecLBs_[inputID2] + vecUBs_[inputID2]); 
        vecSamInps_[(nSamples+2*inputID+1)*nInputs_+inputID2] = 0.5 * 
                         (vecLBs_[inputID2] + vecUBs_[inputID2]); 
      }
    }
  }
  for (inputID = 0; inputID < nInputs_; inputID++)
    vecSamInps_[(nSamples_-1)*nInputs_+inputID] = 0.5 * 
                    (vecLBs_[inputID] + vecUBs_[inputID]);
  delete samplePtr;
  return 0;
}

// ************************************************************************
// refine the sample space
// ------------------------------------------------------------------------
int CentralCompositeSampling::refine(int,int,double, int, double *)
{
  printf("CentralCompositeSampling::refine ERROR - not available.\n");
  exit(1);
  return 0;
}

// ************************************************************************
// set internal scheme
// ------------------------------------------------------------------------
int CentralCompositeSampling::setParam(char *sparam)
{
  char winput[501];
  sscanf(sparam, "%s", winput);
  if (!strcmp(winput, "setResolution"))
  {
    sscanf(sparam, "%s %d", winput, &resolution_);
    if (resolution_ != 4 && resolution_ != 5 && resolution_ != 0)
      resolution_ = 4;
  }
  else if (!strcmp(winput, "setScheme"))
  {
    sscanf(sparam, "%s %d", winput, &scheme_);
    if (scheme_ < 0 && scheme_ > 2) scheme_ = 0;
  }
  if (resolution_ == 0 && scheme_ == 0) samplingID_ = PSUADE_SAMP_CCIF;
  if (resolution_ == 4 && scheme_ == 0) samplingID_ = PSUADE_SAMP_CCI4;
  if (resolution_ == 5 && scheme_ == 0) samplingID_ = PSUADE_SAMP_CCI5;
  if (resolution_ == 0 && scheme_ == 1) samplingID_ = PSUADE_SAMP_CCFF;
  if (resolution_ == 4 && scheme_ == 1) samplingID_ = PSUADE_SAMP_CCF4;
  if (resolution_ == 5 && scheme_ == 1) samplingID_ = PSUADE_SAMP_CCF5;
  if (resolution_ == 0 && scheme_ == 2) samplingID_ = PSUADE_SAMP_CCCF;
  if (resolution_ == 4 && scheme_ == 2) samplingID_ = PSUADE_SAMP_CCC4;
  if (resolution_ == 5 && scheme_ == 2) samplingID_ = PSUADE_SAMP_CCC5;
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
CentralCompositeSampling& CentralCompositeSampling::operator=
                                  (const CentralCompositeSampling &)
{
  printf("CentralCompositeSampling operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

