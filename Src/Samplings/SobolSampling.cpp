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
// ------------------------------------------------------------------------
// initialize  - for computing first-order and total-order indices
// initialize2 - for computing second-order indices
// initialize3 - for computing group-order indices
// VecM1_ and VecM2_ may be loaded from outside using setM1M2
// VecM1_ and VecM2_ may be generated locally by calling createM1M2
//               (for inputs with uniform distribution)
// ************************************************************************
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "SobolSampling.h"
#include "Sampling.h"
#include "MCSampling.h"
#include "LPtauSampling.h"
#include "LHSampling.h"
#include "Psuade.h"
#include "PsuadeConfig.h"
#include "PrintingTS.h"
#include "pdfData.h"
#include "PDFManager.h"

// fix delta h for Sobol instead of random (for refine only)
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
  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  assert((nSamples_ > 0) && 
         "SobolSampling::initialize ERROR - nSamples <= 0.");
  assert((nInputs_ > 0) && 
         "SobolSampling::initialize ERROR - nInputs <= 0.");

  //**/ ----------------------------------------------------------------
  //**/ call different initializer if other than first order
  //**/ ----------------------------------------------------------------
  if (order_ == 2) return initialize2(initLevel);
  if (order_ == 3) 
  {
    assert((order_ == 1) && 
       "SobolSampling::init ERROR - call initialize3 for group order.");
  }

  //**/ ----------------------------------------------------------------
  //**/ revise sample size
  //**/ ----------------------------------------------------------------
  if (nSamples_ / (nInputs_ + 2) * (nInputs_ + 2) != nSamples_) 
  {
    printf("SobolSampling : nSamples must be multiples of nInputs+2.\n");
    nSamples_ = (nSamples_ / (nInputs_ + 2) + 1) * (nInputs_ + 2);
    printf("SobolSampling : nSamples has been changed to be %d\n",
           nSamples_);
  }
  if (initLevel != 0) return 0;

  //**/ ----------------------------------------------------------------
  //**/ allocate space for matrix
  //**/ ----------------------------------------------------------------
  allocSampleData();

  int nReps = nSamples_ / (nInputs_ + 2);

  //**/ ----------------------------------------------------------------
  //**/ check to make sure the seed samples are okay
  //**/ ----------------------------------------------------------------
  if (VecM1_.length()/nInputs_ != nReps && VecM1_.length() > 0)
  {
    printf("SobolSampling ERROR: Invalid loaded M1 length.\n");
    printf("                     Loaded M1 length   = %d.\n",
           VecM1_.length()/nInputs_);
    printf("                     Expected M1 length = %d.\n",
           nReps);
    exit(1);
  }
  if (VecM2_.length()/nInputs_ != nReps && VecM2_.length() > 0)
  {
    printf("SobolSampling ERROR: Invalid loaded M2 length.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  int iD, iD2;
  if (printLevel_ > 4)
  {
    printf("SobolSampling::initialize: nSamples = %d\n", nSamples_);
    printf("SobolSampling::initialize: nInputs  = %d\n", nInputs_);
    printf("SobolSampling::initialize: nOutputs = %d\n", nOutputs_);
    for (iD = 0; iD < nInputs_; iD++)
      printf("    SobolSampling input %3d = [%e %e]\n", iD+1,
             vecLBs_[iD], vecUBs_[iD]);
  }

  //**/ ----------------------------------------------------------------
  //**/ allocate space for the M1 and M2 matrices
  //**/ ----------------------------------------------------------------
  psMatrix matM1, matM2;
  matM1.setFormat(2);
  matM1.setDim(nReps, nInputs_);
  matM2.setFormat(2);
  matM2.setDim(nReps, nInputs_);
  double **M1Mat = matM1.getMatrix2D();
  double **M2Mat = matM2.getMatrix2D();

  //**/ ----------------------------------------------------------------
  //**/ if seed samples have been created or loaded, use them
  //**/ Otherwise, assume uniform distribution and create them
  //**/ (put into M1Mat and M2Mat)
  //**/ ----------------------------------------------------------------
  if (VecM2_.length()/nInputs_ == nReps)
  {
    if (psConfig_.InteractiveIsOn() && printLevel_ > 0)
      printf("SobolSampling: Uses loaded M1 and M2.\n");
    for (iD = 0; iD < nReps; iD++)
    {
      for (iD2 = 0; iD2 < nInputs_; iD2++)
      {
        M1Mat[iD][iD2] = VecM1_[iD*nInputs_+iD2];
        M2Mat[iD][iD2] = VecM2_[iD*nInputs_+iD2];
      }
    }
  }
  else
  {
    //**/ Seed samples have not been created, that means uniform
    //**/ distributions are assumed. Generate M1/M2 and using 1 
    //**/ set of LPtau samples (2*nInputs)
    //**/ Note: This is the correct way (instead of nInputs with
    //**/       double sample size
    psIVector vecPTypes;
    vecPTypes.setLength(nInputs_);
    pdfData pdfObj;
    pdfObj.nInputs_  = nInputs_;
    pdfObj.nSamples_ = nReps;
    pdfObj.VecPTypes_.load(nInputs_, vecPTypes.getIVector());
    pdfObj.VecLBs_.load(nInputs_, vecLBs_.getDVector());
    pdfObj.VecUBs_.load(nInputs_, vecUBs_.getDVector());
    createM1M2(pdfObj);
    for (iD = 0; iD < nReps; iD++)
    {
      for (iD2 = 0; iD2 < nInputs_; iD2++)
      {
        M1Mat[iD][iD2] = VecM1_[iD*nInputs_+iD2];
        M2Mat[iD][iD2] = VecM2_[iD*nInputs_+iD2];
      }
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ now create Sobol' samples from M1Mat and M2Mat
  //**/ ----------------------------------------------------------------
  int    sampleCount = 0, iR;
  double ddata;
  for (iR = 0; iR < nReps; iR++)
  {
    //**/ first point in the block: totally from M2
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata = M2Mat[iR][iD2];
      vecSamInps_[sampleCount*nInputs_+iD2] = ddata;
    }
    sampleCount++;
    //**/ middle points in the block: from M2 with a
    //**/ single modification from M1
    for (iD = 0; iD < nInputs_; iD++)
    {
      //**/ copy from M2 
      for (iD2 = 0; iD2 < nInputs_; iD2++)
      {
        ddata = M2Mat[iR][iD2];
        vecSamInps_[sampleCount*nInputs_+iD2] = ddata;
      }
      //**/ modify the iD-th input from M1
      ddata = M1Mat[iR][iD];
      vecSamInps_[sampleCount*nInputs_+iD] = ddata;
      sampleCount++;
    }
    //**/ last point in the block: totally from M1
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata = M1Mat[iR][iD2];
      vecSamInps_[sampleCount*nInputs_+iD2] = ddata;
    }
    sampleCount++;
  }
  return 0;
}

// ************************************************************************
// initialize the sampling data
// ------------------------------------------------------------------------
int SobolSampling::initialize2(int initLevel)
{
  //**/ ----------------------------------------------------------------
  //**/ revise sample size
  //**/ ----------------------------------------------------------------
  int blkSize = (nInputs_ - 1) * (nInputs_) / 2 + 2;
  int nReps = nSamples_ / blkSize;
  if (nReps * blkSize != nSamples_ || nReps < 100)
  {
    printf("SobolSampling: nSamples must be multiples of %d.\n",blkSize);
    printf("               nSamples must be >= %d.\n",blkSize*100);
    if (nReps < 100) nReps = 100;
    nSamples_ = nReps * blkSize;
    printf("SobolSampling : nSamples has been changed to be %d\n",
            nSamples_);
  }
  if (initLevel != 0) return 0;

  //**/ ----------------------------------------------------------------
  //**/ allocate space for matrix
  //**/ ----------------------------------------------------------------
  allocSampleData();

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  int iD, iD2;
  if (printLevel_ > 4)
  {
    printf("SobolSampling initialize2: nSamples = %d\n", nSamples_);
    printf("SobolSampling initialize2: nInputs  = %d\n", nInputs_);
    printf("SobolSampling initialize2: nOutputs = %d\n", nOutputs_);
    for (iD = 0; iD < nInputs_; iD++)
      printf("    SobolSampling input %3d = [%e %e]\n", iD+1,
             vecLBs_[iD], vecUBs_[iD]);
  }

  //**/ ----------------------------------------------------------------
  //**/ allocate space for the matrices
  //**/ ----------------------------------------------------------------
  psMatrix matM1, matM2;
  matM1.setFormat(2);
  matM1.setDim(nReps, nInputs_);
  matM2.setFormat(2);
  matM2.setDim(nReps, nInputs_);
  double **M1Mat = matM1.getMatrix2D();
  double **M2Mat = matM2.getMatrix2D();
  double ddata;

  //**/ ----------------------------------------------------------------
  //**/ check to see if the seed matrices have been loaded
  //**/ ----------------------------------------------------------------
  if (VecM1_.length()/nInputs_ != nReps && VecM1_.length() > 0)
  {
    printf("SobolSampling ERROR: Invalid loaded M1 length.\n");
    printf("                     Loaded M1 length   = %d.\n",
           VecM1_.length()/nInputs_);
    printf("                     Expected M1 length = %d.\n",
           nReps);
    exit(1);
  } 
  if (VecM2_.length()/nInputs_ != nReps && VecM2_.length() > 0)
  {
    printf("SobolSampling ERROR: Invalid loaded M2 length.\n");
    exit(1);
  } 

  //**/ ----------------------------------------------------------------
  //**/ if the seed matrices have been loaded, set M1 and M2
  //**/ ----------------------------------------------------------------
  if (VecM2_.length()/nInputs_ == nReps)
  {
    if (psConfig_.DiagnosticsIsOn())
      printf("SobolSampling init2: Using M1/M2 from createM1M2\n");
    if (psConfig_.InteractiveIsOn() && printLevel_ > 0)
      printf("SobolSampling: Using M1/M2 from createM1M2\n");
    for (iD = 0; iD < nReps; iD++)
    {
      for (iD2 = 0; iD2 < nInputs_; iD2++) 
      {
        M1Mat[iD][iD2] = VecM1_[iD*nInputs_+iD2];
        M2Mat[iD][iD2] = VecM2_[iD*nInputs_+iD2];
      }
    }
  }
  //**/ if not loaded already, generate M1 and M2
  //**/ assume uniform
  else
  {
    if (psConfig_.DiagnosticsIsOn())
      printf("SobolSampling init2: Create M1 and M2 locally\n");
    psIVector vecPTypes;
    pdfData sobj;
    sobj.nSamples_ = nReps;
    sobj.nInputs_  = nInputs_;
    sobj.VecLBs_.load(nInputs_, vecLBs_.getDVector());
    sobj.VecUBs_.load(nInputs_, vecUBs_.getDVector());
    vecPTypes.setLength(nInputs_);
    sobj.VecPTypes_.load(nInputs_, vecPTypes.getIVector());
    createM1M2(sobj);
    for (iD = 0; iD < nReps; iD++)
    {
      for (iD2 = 0; iD2 < nInputs_; iD2++) 
      {
        ddata = VecM1_[iD*nInputs_+iD2];
        M1Mat[iD][iD2] = ddata;
        ddata = VecM2_[iD*nInputs_+iD2];
        M2Mat[iD][iD2] = ddata;
      }
    }
  }

  //**/ now generate 2nd order Sobol' samples from M1 and M2
  int sampleCount = 0, iR, ii;
  for (iR = 0; iR < nReps; iR++)
  {
    //**/ first point in the block: totally from M2
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata = M2Mat[iR][iD2];
      vecSamInps_[sampleCount*nInputs_+iD2] = ddata;
    }
    sampleCount++;
    //**/ start from M2, modify iD, iD2 entries
    for (iD = 0; iD < nInputs_; iD++)
    {
      for (iD2 = iD+1; iD2 < nInputs_; iD2++)
      {
        //**/ copy from M2 
        for (ii = 0; ii < nInputs_; ii++)
        {
          ddata = M2Mat[iR][ii];
          vecSamInps_[sampleCount*nInputs_+ii] = ddata;
        }
        //**/ modify the iD-th input from M1
        ddata = M1Mat[iR][iD];
        vecSamInps_[sampleCount*nInputs_+iD] = ddata;
        //**/ modify the iD2-th input from M1
        ddata = M1Mat[iR][iD2];
        vecSamInps_[sampleCount*nInputs_+iD2] = ddata;
        sampleCount++;
      }
    }
    //**/ last point in the block: totally from M1
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata = M1Mat[iR][iD2];
      vecSamInps_[sampleCount*nInputs_+iD2] = ddata;
    }
    sampleCount++;
  }
  if (sampleCount != nSamples_)
  {
    printf("SobolSampling initialize2 INTERNAL ERROR.\n");
    printf("     Inconsistent Sample sizes %d != %d\n",
           sampleCount, nSamples_);
    exit(1);
  }
  return 0;
}

// ************************************************************************
// initialize the sampling data for group SA
// ------------------------------------------------------------------------
int SobolSampling::initialize3(int initLevel, psIMatrix matGrpInfo)
{
  //**/ ----------------------------------------------------------------
  //**/ revise sample size
  //**/ ----------------------------------------------------------------
  int blkSize = matGrpInfo.nrows() + 2;
  int nReps = nSamples_ / blkSize;
  if (nReps * blkSize != nSamples_ || nReps < 10000)
  {
    printf("SobolSampling: nSamples must be multiples of %d.\n",blkSize);
    printf("               nSamples must be >= %d.\n",blkSize*10000);
    if (nReps < 10000) nReps = 10000;
    nSamples_ = nReps * blkSize;
    printf("SobolSampling : nSamples has been changed to be %d\n",
            nSamples_);
  }
  if (initLevel != 0) return 0;

  //**/ ----------------------------------------------------------------
  //**/ allocate space for matrix
  //**/ ----------------------------------------------------------------
  allocSampleData();

  //**/ ----------------------------------------------------------------
  //**/ allocate space for the matrices
  //**/ ----------------------------------------------------------------
  psMatrix matM1, matM2;
  matM1.setFormat(2);
  matM1.setDim(nReps, nInputs_);
  matM2.setFormat(2);
  matM2.setDim(nReps, nInputs_);
  double **M1Mat = matM1.getMatrix2D();
  double **M2Mat = matM2.getMatrix2D();
  int    iD, iD2;

  //**/ ----------------------------------------------------------------
  //**/ check to see if the seed matrices have been loaded
  //**/ ----------------------------------------------------------------
  if (VecM1_.length()/nInputs_ != nReps && VecM1_.length() > 0)
  {
    printf("SobolSampling ERROR: Invalid loaded M1 length.\n");
    printf("                     Loaded M1 length   = %d.\n",
           VecM1_.length()/nInputs_);
    printf("                     Expected M1 length = %d.\n",
           nReps);
    exit(1);
  } 
  if (VecM2_.length()/nInputs_ != nReps && VecM2_.length() > 0)
  {
    printf("SobolSampling ERROR: Invalid loaded M2 length.\n");
    exit(1);
  } 

  //**/ ----------------------------------------------------------------
  //**/ if the seed matrices have been loaded, set M1 and M2
  //**/ ----------------------------------------------------------------
  if (VecM2_.length()/nInputs_ == nReps)
  {
    if (psConfig_.DiagnosticsIsOn())
      printf("SobolSampling init3: Loading M1 and M2\n");
    if (psConfig_.InteractiveIsOn() && printLevel_ > 0)
      printf("SobolSampling: Uses loaded M1 and M2.\n"); 
    for (iD = 0; iD < nReps; iD++)
    {
      for (iD2 = 0; iD2 < nInputs_; iD2++) 
      {
        M1Mat[iD][iD2] = VecM1_[iD*nInputs_+iD2];
        M2Mat[iD][iD2] = VecM2_[iD*nInputs_+iD2];
      }
    }
  }
  //**/ if not loaded already, generate M1 and M2
  //**/ assume uniform
  else
  {
    if (psConfig_.DiagnosticsIsOn())
      printf("SobolSampling init3: Create M1 and M2 locally\n");
    psIVector vecPTypes;
    vecPTypes.setLength(nInputs_);
    pdfData pdfObj;
    pdfObj.nInputs_  = nInputs_;
    pdfObj.nSamples_ = nReps;
    pdfObj.VecPTypes_.load(nInputs_, vecPTypes.getIVector());
    pdfObj.VecLBs_.load(nInputs_, vecLBs_.getDVector());
    pdfObj.VecUBs_.load(nInputs_, vecUBs_.getDVector());
    createM1M2(pdfObj);
    for (iD = 0; iD < nReps; iD++)
    {
      for (iD2 = 0; iD2 < nInputs_; iD2++)
      {
        M1Mat[iD][iD2] = VecM1_[iD*nInputs_+iD2];
        M2Mat[iD][iD2] = VecM2_[iD*nInputs_+iD2];
      }
    }
  }

  //**/ now generate group order Sobol' samples from M1 and M2
  int    iR, inputID, sampleCount = 0, kk;
  double ddata;
  for (iR = 0; iR < nReps; iR++)
  {
    //**/ first point in the block: totally from M2
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata = M2Mat[iR][iD2];
      vecSamInps_[sampleCount*nInputs_+iD2] = ddata;
    }
    sampleCount++;
    //**/ start from M2, modify non-zero entries in group matrix
    for (iD = 0; iD < blkSize-2; iD++)
    {
      //**/ copy from M2 
      for (inputID = 0; inputID < nInputs_; inputID++)
      {
        ddata = M2Mat[iR][inputID];
        vecSamInps_[sampleCount*nInputs_+inputID] = ddata;
      }
      //**/ check from group information
      for (iD2 = 0; iD2 < nInputs_; iD2++)
      {
        kk = matGrpInfo.getEntry(iD, iD2);
        //**/ modify the kk-th input from M1 if it is non-zero
        if (kk != 0)
        {
          ddata = M1Mat[iR][iD2];
          vecSamInps_[sampleCount*nInputs_+iD2] = ddata;
        }
      }
      sampleCount++;
    }
    //**/ last point in the block: totally from M1
    for (iD2 = 0; iD2 < nInputs_; iD2++)
    {
      ddata = M1Mat[iR][iD2];
      vecSamInps_[sampleCount*nInputs_+iD2] = ddata;
    }
    sampleCount++;
  }
  if (sampleCount != nSamples_)
  {
    printf("SobolSampling initialize3 INTERNAL ERROR.\n");
    printf("     Inconsistent Sample sizes %d != %d\n",
           sampleCount, nSamples_);
    exit(1);
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

  if (order_ != 1)
  {
    printf("SobolSampling ERROR: refine does not support for order != 1\n");
    exit(1);
  }
  if (VecM1_.length() != 0)
  {
    printf("SobolSampling ERROR: refine does not support non-uniform PDF\n");
    exit(1);
  }

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
#ifdef PSUADE_SAL_GRID
  //**/ M1 and M2 are generated on a fixed grid (nReps/2 grid points)
  generate(M2Mat, nReps);
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
  generate(M2Mat, nReps);
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
  sampleCount = nSamples_;
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
// set the M1 and M2 matrices
// ------------------------------------------------------------------------
int SobolSampling::setM1M2(psVector vecM)
{
  int leng = vecM.length();
  if (leng % 2 != 0)
  {
    printf("SobolSampling::setM1M2 ERROR: Invalid length.\n");
    exit(1);
  }
  leng = leng / 2;
  double *dataPtr = vecM.getDVector();
  VecM1_.load(leng, dataPtr);
  VecM2_.load(leng, &(dataPtr[leng]));
#if 0
  //**/ diagnostics
  int ii, jj;
  double dm=0, ds=0;
  if (psConfig_.DiagnosticsIsOn())
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      dm = 0;
      for (jj = 0; jj < leng/nInputs_; jj++) 
        dm += VecM1_[jj*nInputs_+ii];
      dm /= (double) (1.0*leng/nInputs_);
      ds = 0;
      for (jj = 0; jj < leng/nInputs_; jj++) 
        ds += pow(VecM1_[jj*nInputs_+ii]-dm,2.0);
      ds /= (double) (1.0*leng/nInputs_);
      printf("SobolSampling M1 input %d mean, std = %e %e\n",
             ii+1,dm,sqrt(ds));
      dm = 0;
      for (jj = 0; jj < leng/nInputs_; jj++) 
        dm += VecM2_[jj*nInputs_+ii];
      dm /= (double) (1.0*leng/nInputs_);
      ds = 0;
      for (jj = 0; jj < leng/nInputs_; jj++) 
        ds += pow(VecM2_[jj*nInputs_+ii]-dm,2.0);
      ds /= (double) (1.0*leng/nInputs_);
      printf("SobolSampling M2 input %d mean, std = %e %e\n",
             ii+1,dm,sqrt(ds));
    }
  }
#endif
  return 0;
}

// ************************************************************************
// set the M1 and M2 matrices
// ------------------------------------------------------------------------
int SobolSampling::createM1M2(pdfData pobj)
{
  int      hasPDF=0, ii, iOne=1, ss;
  psVector vecLBs, vecUBs;

  //**/ check to see if all input types are uniform
  for (ii = 0; ii < nInputs_; ii++)
    if (pobj.VecPTypes_[ii] != 0) hasPDF = 1;	

  //**/ if all uniform, Generate M1,M2 using 1 set of LPtau samples
  //**/ Note: This is the correct way (use 2*nInputs)
  if (hasPDF == 0)
  {
    vecLBs.setLength(2*nInputs_);
    vecUBs.setLength(2*nInputs_);
    for (ii = 0; ii < nInputs_; ii++)
    {
      vecLBs[ii] = vecLBs[ii+nInputs_] = pobj.VecLBs_[ii];
      vecUBs[ii] = vecUBs[ii+nInputs_] = pobj.VecUBs_[ii];
    }
    Sampling *sampler = NULL;
    if (2*nInputs_ < 51) sampler = (Sampling *) new LPtauSampling();
    else                 sampler = (Sampling *) new LHSampling();
    sampler->setInputBounds(2*nInputs_,vecLBs.getDVector(),
                            vecUBs.getDVector());
    sampler->setOutputParams(iOne);
    sampler->setSamplingParams(pobj.nSamples_, iOne, iOne);
    sampler->initialize(0);
    psVector  vecX, vecY;
    psIVector vecS;
    vecX.setLength(pobj.nSamples_*2*nInputs_);
    vecY.setLength(pobj.nSamples_);
    vecS.setLength(pobj.nSamples_);
    sampler->getSamples(pobj.nSamples_,2*nInputs_,iOne,vecX.getDVector(),
                        vecY.getDVector(), vecS.getIVector());
    VecM1_.setLength(pobj.nSamples_*nInputs_);
    VecM2_.setLength(pobj.nSamples_*nInputs_);
    for (ss = 0; ss < pobj.nSamples_; ss++)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        VecM1_[ss*nInputs_+ii] = vecX[ss*2*nInputs_+ii];
        VecM2_[ss*nInputs_+ii] = vecX[ss*2*nInputs_+nInputs_+ii];
      }
    }
    delete sampler;
  }
  else
  {
    PDFManager *pdfman = new PDFManager();
    psIVector vecTypes;
    vecTypes.setLength(2*nInputs_);
    psVector vecParam1, vecParam2;
    vecParam1.setLength(2*nInputs_);
    vecParam2.setLength(2*nInputs_);
    psMatrix matLocalCor;
    matLocalCor.setDim(2*nInputs_,2*nInputs_);
    for (ii = 0; ii < nInputs_; ii++)
    {
      vecTypes[ii]  = vecTypes[nInputs_+ii]  = pobj.VecPTypes_[ii];
      vecParam1[ii] = vecParam1[nInputs_+ii] = pobj.VecParam1_[ii];
      vecParam2[ii] = vecParam2[nInputs_+ii] = pobj.VecParam2_[ii];
      for (ss = 0; ss < nInputs_; ss++)
      {
        matLocalCor.setEntry(ii,ss,pobj.MatCor_.getEntry(ii,ss));
        matLocalCor.setEntry(ii+nInputs_,ss+nInputs_,
                             pobj.MatCor_.getEntry(ii,ss));
      }
    }
    pdfman->initialize(2*nInputs_,vecTypes.getIVector(),
                       vecParam1.getDVector(),vecParam2.getDVector(),
                       matLocalCor,NULL,NULL);
    vecLBs.setLength(2*nInputs_);
    vecUBs.setLength(2*nInputs_);
    for (ii = 0; ii < nInputs_; ii++)
    {
      vecLBs[ii] = vecLBs[ii+nInputs_] = pobj.VecLBs_[ii];
      vecUBs[ii] = vecUBs[ii+nInputs_] = pobj.VecUBs_[ii];
    }
    psVector vecX;
    vecX.setLength(pobj.nSamples_*2*nInputs_);
    pdfman->genSample(pobj.nSamples_, vecX, vecLBs, vecUBs);
    VecM1_.setLength(pobj.nSamples_*nInputs_);
    VecM2_.setLength(pobj.nSamples_*nInputs_);
    for (ss = 0; ss < pobj.nSamples_; ss++)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        VecM1_[ss*nInputs_+ii] = vecX[ss*2*nInputs_+ii];
        VecM2_[ss*nInputs_+ii] = vecX[ss*2*nInputs_+nInputs_+ii];
      }
    }
    delete pdfman;
  }
  return 0;
}

// ************************************************************************
// set the order
// ------------------------------------------------------------------------
int SobolSampling::setOrder(int order)
{
  order_ = order;
  if (order != 1 && order != 2 && order != 3)
  {
    printf("SobolSampling ERROR: Wrong order. Default to 1.\n");
    order_ = 1;
  }
  return 0;
}

// ************************************************************************
// set internal scheme
// ------------------------------------------------------------------------
int SobolSampling::setParam(char *sparam)
{
  char winput[1001];
  sscanf(sparam, "%s", winput);
  if (!strcmp(winput, "setOrder"))
  {
    sscanf(sparam, "%s %d", winput, &order_);
    if (order_ < 1 || order_ > 3)
      printf("SobolSampling ERROR: Invalid order %d.\n",order_);
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

