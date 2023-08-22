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
// Functions for the Sobol's one-at-a-time class 
// AUTHOR : CHARLES TONG
// DATE   : 2004
// ************************************************************************
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "SobolSampling.h"
#include "Sampling.h"
#include "MCSampling.h"
#include "LPtauSampling.h"
#include "LHSampling.h"

// fix delta h for Sobol instead of random
//#define PSUADE_SAL_GRID

// ************************************************************************
// Constructor
// ------------------------------------------------------------------------
SobolSampling::SobolSampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_SOBOL;
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
SobolSampling::~SobolSampling()
{
}

// ************************************************************************
// initialize the sampling data
// ------------------------------------------------------------------------
int SobolSampling::initialize(int initLevel)
{
  int    iD, iD2, inputID, nReps, sampleCount, iR;
  double ddata;
  psMatrix matM1, matM2;
  psVector vecRanges;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  if (nSamples_ == 0)
  {
    printf("SobolSampling::initialize ERROR - nSamples = 0.\n");
    exit(1);
  }
  if (nInputs_ == 0)
  {
    printf("SobolSampling::initialize ERROR - input not set up.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ clean up
  //**/ ----------------------------------------------------------------
  if (nInputs_ > 25)
  {
    printf("SobolSampling : can only handle nInputs <= 25\n");
    exit(1);
  }
  if (nSamples_ / (nInputs_ + 2) * (nInputs_ + 2) != nSamples_) 
  {
    printf("SobolSampling : nSamples must be multiples of nInputs+2.\n");
    nSamples_ = (nSamples_ / (nInputs_ + 2) + 1) * (nInputs_ + 2);
    printf("SobolSampling : nSamples has been changed to be %d\n",nSamples_);
  }
  if (initLevel != 0) return 0;

  //**/ ----------------------------------------------------------------
  //**/ allocate space for matrix
  //**/ ----------------------------------------------------------------
  allocSampleData();

  nReps = nSamples_ / (nInputs_ + 2);

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 4)
  {
    printf("SobolSampling::initialize: nSamples = %d\n", nSamples_);
    printf("SobolSampling::initialize: nInputs  = %d\n", nInputs_);
    printf("SobolSampling::initialize: nOutputs = %d\n", nOutputs_);
    for (inputID = 0; inputID < nInputs_; inputID++)
      printf("    SobolSampling input %3d = [%e %e]\n", inputID+1,
             vecLBs_[inputID], vecUBs_[inputID]);
  }

  //**/ ----------------------------------------------------------------
  //**/ allocate space for the matrices
  //**/ ----------------------------------------------------------------
  matM1.setFormat(2);
  matM1.setDim(nReps, nInputs_);
  matM2.setFormat(2);
  matM2.setDim(nReps, nInputs_);
  double **M1Mat = matM1.getMatrix2D();
  double **M2Mat = matM2.getMatrix2D();

  //**/ ----------------------------------------------------------------
  //**/ preparation for generating the samples
  //**/ ----------------------------------------------------------------
  vecRanges.setLength(nInputs_);
  for (iD = 0;  iD < nInputs_;  iD++) 
    vecRanges[iD] = vecUBs_[iD] - vecLBs_[iD];

  //**/ ----------------------------------------------------------------
  //**/ repeat the sample generation
  //**/ ----------------------------------------------------------------
  sampleCount = 0;
  generate(M2Mat, nReps);
#ifdef PSUADE_SAL_GRID
  //**/ M1 and M2 are generated on a fixed grid (nReps/2 grid points)
  ddata  = 1.0 / (double) (nReps/2 - 1);
  for (iD = 0; iD < nReps; iD++)
  {
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata2 = PSUADE_drand();
      if (ddata2 > 0.5) ddata2 = 1.0;
      else              ddata2 = -1.0;
      if ((M2Mat[iD][iD2]+ddata2*ddata)>1.0 || 
          (M2Mat[iD][iD2]+ddata2*ddata)< 0.0)
           M1Mat[iD][iD2] = M2Mat[iD][iD2] - ddata2*ddata;
      else M1Mat[iD][iD2] = M2Mat[iD][iD2] + ddata2*ddata;
    }
  }
#endif
#ifdef PSUADE_SAL_GRID
  //**/ M1 generated from M2 by adding fixed (for all inputs) but 
  //**/ random (from sample point to sample point) perturbations
  for (iD = 0; iD < nReps; iD++)
    for (iD2 = 0; iD2 < nInputs_; iD2++)
      M1Mat[iD][iD2] = M2Mat[iD][iD2];
  for (iD = 0; iD < nReps; iD++)
  {
    ddata = 0.25 * PSUADE_drand();
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata2 = PSUADE_drand();
      if (ddata2 > 0.5) ddata2 = 1.0;
      else              ddata2 = -1.0;
      if ((M2Mat[iD][iD2]+ddata2*ddata)>1.0 || 
          (M2Mat[iD][iD2]+ddata2*ddata)< 0.0)
           M1Mat[iD][iD2] = M2Mat[iD][iD2] - ddata2*ddata;
      else M1Mat[iD][iD2] = M2Mat[iD][iD2] + ddata2*ddata;
    }
  }
#endif
#if 1
   //**/ M1 generated from M2 by generating 2 LPtau samples
  int iOne=1;
  psVector  vecLBs, vecUBs, vecSamInps, vecSamOuts;
  psIVector vecSamStas;
  vecLBs.setLength(2*nInputs_);
  vecUBs.setLength(2*nInputs_);
  for (iD = 0; iD < 2*nInputs_; iD++) vecLBs[iD] = 0.0;
  for (iD = 0; iD < 2*nInputs_; iD++) vecUBs[iD] = 1.0;
  //Sampling *sampler = (Sampling *) new MCSampling();
  Sampling *sampler = (Sampling *) new LPtauSampling();
  sampler->setInputBounds(2*nInputs_,vecLBs.getDVector(),vecUBs.getDVector());
  sampler->setOutputParams(iOne);
  sampler->setSamplingParams(nReps, iOne, iOne);
  sampler->initialize(0);
  vecSamInps.setLength(nReps*2*nInputs_);
  vecSamOuts.setLength(nReps);
  vecSamStas.setLength(nReps);
  sampler->getSamples(nReps,2*nInputs_,iOne,vecSamInps.getDVector(), 
                      vecSamOuts.getDVector(), vecSamStas.getIVector());
  for (iD = 0; iD < nReps; iD++)
    for (iD2 = 0; iD2 < nInputs_; iD2++) 
      M1Mat[iD][iD2] = vecSamInps[iD*2*nInputs_+iD2];
  for (iD = 0; iD < nReps; iD++)
    for (iD2 = 0; iD2 < nInputs_; iD2++) 
      M2Mat[iD][iD2] = vecSamInps[iD*2*nInputs_+nInputs_+iD2];
  delete sampler;
#endif
  for (iR = 0; iR < nReps; iR++)
  {
    //**/ first point in the block: totally from M2
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata = M2Mat[iR][iD2];
      ddata = ddata * vecRanges[iD2] + vecLBs_[iD2];
      vecSamInps_[sampleCount*nInputs_+iD2] = ddata;
    }
    sampleCount++;
    //**/ middle points in the block: from first point with a
    //**/ single modification from M1
    for (iD = 0; iD < nInputs_; iD++)
    {
      //**/ copy from previous point
      for (iD2 = 0; iD2 < nInputs_; iD2++)
      {
        ddata = M2Mat[iR][iD2];
        ddata = ddata * vecRanges[iD2] + vecLBs_[iD2];
        vecSamInps_[sampleCount*nInputs_+iD2] = ddata;
      }
      //**/ modify the iD-th input from M1
      ddata = M1Mat[iR][iD];
      ddata = ddata * vecRanges[iD] + vecLBs_[iD];
      vecSamInps_[sampleCount*nInputs_+iD] = ddata;
      sampleCount++;
    }
    //**/ last point in the block: totally from M1
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata = M1Mat[iR][iD2];
      ddata = ddata * vecRanges[iD2] + vecLBs_[iD2];
      vecSamInps_[sampleCount*nInputs_+iD2] = ddata;
    }
    sampleCount++;
  }
  return 0;
}

// ************************************************************************
// generate the BS matrix
// ------------------------------------------------------------------------
int SobolSampling::generate(double **inMat, int size)
{
  int iD, iD2, nmax;
#ifdef PSUADE_SAL_GRID
  int idata;
#endif

  nmax = size / 2;
  for (iD = 0; iD < size; iD++)
  {
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
#ifdef PSUADE_SAL_GRID
      idata = (int) (PSUADE_drand() * nmax);
      if (idata == nmax) idata = nmax - 1;
      inMat[iD][iD2] = (double) idata / (double) (nmax - 1);
#else
      inMat[iD][iD2] = PSUADE_drand();
#endif
    }
  }
  return 0;
}

// ************************************************************************
// refine the sample space
// ------------------------------------------------------------------------
int SobolSampling::refine(int refineRatio, int randomize, double thresh,
                          int nSamples, double *sampleErrors)
{
  int    iD, iD2, sampleCount, nReps, iR, nLevels;
  double ddata;
  psVector  vecNewSamInps, vecNewSamOuts, vecRanges;
  psIVector vecNewSamStas;
  psMatrix  matM1, matM2;

  //**/ ----------------------------------------------------------------
  //**/ unused parameters
  //**/ ----------------------------------------------------------------
  (void) randomize;
  (void) thresh;
  (void) nSamples;
  (void) sampleErrors;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  nLevels = refineRatio;
  nReps = nSamples_ * (nLevels - 1) / (nInputs_ + 2);

  //**/ ----------------------------------------------------------------
  //**/ allocate space for new sample data
  //**/ ----------------------------------------------------------------
  // First do some defensive programming and range checking by Bill Oliver
  if(nSamples_*nLevels <= 0)
  {
    printf("nSamples_*nLevels <= 0 in file %s line %d\n",__FILE__,__LINE__);
    exit(1);
  }
  vecNewSamInps.setLength(nSamples_*nLevels*nInputs_);
  vecNewSamOuts.setLength(nSamples_*nLevels*nOutputs_);
  vecNewSamStas.setLength(nSamples_*nLevels);
  for (iD = 0;  iD < nSamples_*nLevels; iD++)
  {
    vecNewSamStas[iD] = 0;
    for (iD2 = 0; iD2 < nOutputs_; iD2++)
      vecNewSamOuts[iD*nOutputs_+iD2] = PSUADE_UNDEFINED;
  }

  //**/ ----------------------------------------------------------------
  //**/ copy the old samples
  //**/ ----------------------------------------------------------------
  for (iD = 0;  iD < nSamples_; iD++) 
  {
    for (iD2 = 0; iD2 < nInputs_; iD2++)
      vecNewSamInps[iD*nInputs_+iD2] = vecSamInps_[iD*nInputs_+iD2];
    for (iD2 = 0; iD2 < nOutputs_; iD2++)
      vecNewSamOuts[iD*nOutputs_+iD2] = vecSamOuts_[iD*nOutputs_+iD2];
    vecNewSamStas[iD] = 1;
  }

  //**/ ----------------------------------------------------------------
  //**/ allocate temporary matrices
  //**/ ----------------------------------------------------------------
  matM1.setFormat(2);
  matM1.setDim(nReps, nInputs_);
  matM2.setFormat(2);
  matM2.setDim(nReps, nInputs_);
  double **M1Mat = matM1.getMatrix2D();
  double **M2Mat = matM2.getMatrix2D();

  //**/ ----------------------------------------------------------------
  //**/ preparation for generating the samples
  //**/ ----------------------------------------------------------------
  vecRanges.setLength(nInputs_);
  for (iD = 0;  iD < nInputs_;  iD++) 
    vecRanges[iD] = vecUBs_[iD] - vecLBs_[iD];

  //**/ ----------------------------------------------------------------
  //**/ repeat the sample generation
  //**/ ----------------------------------------------------------------
  sampleCount = nSamples_;
  generate(M2Mat, nReps);
#ifdef PSUADE_SAL_GRID
  //**/ M1 and M2 are generated on a fixed grid (nReps/2 grid points)
  ddata  = 1.0 / (double) (nReps/2 - 1);
  for (iD = 0; iD < nReps; iD++)
  {
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata2 = PSUADE_drand();
      if (ddata2 > 0.5) ddata2 = 1.0;
      else              ddata2 = -1.0;
      if ((M2Mat[iD][iD2]+ddata2*ddata)>1.0 || 
          (M2Mat[iD][iD2]+ddata2*ddata)< 0.0)
           M1Mat[iD][iD2] = M2Mat[iD][iD2] - ddata2*ddata;
      else M1Mat[iD][iD2] = M2Mat[iD][iD2] + ddata2*ddata;
    }
  }
#endif
#ifdef PSUADE_SAL_GRID
  //**/ M1 generated from M2 by adding fixed (for all inputs) but 
  //**/ random (from sample point to sample point) perturbations
  for (iD = 0; iD < nReps; iD++)
    for (iD2 = 0; iD2 < nInputs_; iD2++)
      M1Mat[iD][iD2] = M2Mat[iD][iD2];
  for (iD = 0; iD < nReps; iD++)
  {
    ddata = 0.25 * PSUADE_drand();
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata2 = PSUADE_drand();
      if (ddata2 > 0.5) ddata2 = 1.0;
      else              ddata2 = -1.0;
      if ((M2Mat[iD][iD2]+ddata2*ddata)>1.0 || 
          (M2Mat[iD][iD2]+ddata2*ddata)< 0.0)
           M1Mat[iD][iD2] = M2Mat[iD][iD2] - ddata2*ddata;
      else M1Mat[iD][iD2] = M2Mat[iD][iD2] + ddata2*ddata;
    }
  }
#endif
#if 1
  //**/ M1 generated from M2 by generating 2 samples
  int iOne=1;
  psVector  vecLBs, vecUBs, vecSamInps, vecSamOuts;
  psIVector vecSamStas;
  vecLBs.setLength(2*nInputs_);
  vecUBs.setLength(2*nInputs_);
  for (iD = 0; iD < 2*nInputs_; iD++) vecLBs[iD] = 0.0;
  for (iD = 0; iD < 2*nInputs_; iD++) vecUBs[iD] = 1.0;
  Sampling *sampler = (Sampling *) new MCSampling();
  sampler->setInputBounds(2*nInputs_,vecLBs.getDVector(),vecUBs.getDVector());
  sampler->setOutputParams(iOne);
  sampler->setSamplingParams(nReps*nLevels, iOne, iOne);
  sampler->initialize(0);
  vecSamInps.setLength(nReps*2*nInputs_);
  vecSamOuts.setLength(nReps);
  vecSamStas.setLength(nReps);
  sampler->getSamples(nReps*nLevels,nInputs_,iOne,vecSamInps.getDVector(), 
                      vecSamOuts.getDVector(), vecSamStas.getIVector());
  for (iD = 0; iD < nReps*(nLevels-1); iD++)
    for (iD2 = 0; iD2 < nInputs_; iD2++) 
      M1Mat[iD][iD2] = vecSamInps[(nReps+iD)*2*nInputs_+iD2];
  for (iD = 0; iD < nReps*(nLevels-1); iD++)
    for (iD2 = 0; iD2 < nInputs_; iD2++) 
      M2Mat[iD][iD2] = vecSamInps[(nReps+iD)*2*nInputs_+nInputs_+iD2];
  delete sampler;
#endif
  for (iR = 0; iR < nReps; iR++)
  {
    //**/ first point in the block: totally from M2
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata = M2Mat[iR][iD2];
      ddata = ddata * vecRanges[iD2] + vecLBs_[iD2];
      vecNewSamInps[sampleCount*nInputs_+iD2] = ddata;
    }
    sampleCount++;
    //**/ middle points in the block: from first point 
    //**/ single modification from M1
    for (iD = 0; iD < nInputs_; iD++)
    {
      for (iD2 = 0; iD2 < nInputs_; iD2++)
      {
        ddata = M2Mat[iR][iD2];
        ddata = ddata * vecRanges[iD2] + vecLBs_[iD2];
        vecNewSamInps[sampleCount*nInputs_+iD2] = ddata;
      }
      ddata = M1Mat[iR][iD];
      ddata = ddata * vecRanges[iD] + vecLBs_[iD];
      vecNewSamInps[sampleCount*nInputs_+iD] = ddata;
      sampleCount++;
    }
    //**/ last point in the block: totally from M2
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata = M1Mat[iR][iD2];
      ddata = ddata * vecRanges[iD2] + vecLBs_[iD2];
      vecNewSamInps[sampleCount*nInputs_+iD2] = ddata;
    }
    sampleCount++;
  }

  //**/ -------------------------------------------------------------------
  //**/ revise internal variables
  //**/ -------------------------------------------------------------------
  nSamples_ = nSamples_ * nLevels;
  vecSamInps_ = vecNewSamInps;
  vecSamOuts_ = vecNewSamOuts;
  vecSamStas_ = vecNewSamStas;

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 4)
  {
    printf("SobolSampling::refine: nSamples = %d\n", nSamples_);
    printf("SobolSampling::refine: nInputs  = %d\n", nInputs_);
    printf("SobolSampling::refine: nOutputs = %d\n", nOutputs_);
    for (iD2 = 0; iD2 < nInputs_; iD2++)
      printf("    SobolSampling input %3d = [%e %e]\n", iD2+1,
             vecLBs_[iD2], vecUBs_[iD2]);
  }
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
SobolSampling& SobolSampling::operator=(const SobolSampling &)
{
  printf("SobolSampling operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

