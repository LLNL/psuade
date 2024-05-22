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
// Functions for the factorial sampling class 
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************
#include <stdio.h>
//**/using namespace std;

#include "sysdef.h"
#include "PsuadeUtil.h"
#include "FactorialSampling.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
FactorialSampling::FactorialSampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_FACT;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
FactorialSampling::~FactorialSampling()
{
}

// ************************************************************************
// initialize the sampler data
// ------------------------------------------------------------------------
int FactorialSampling::initialize(int initLevel)
{
  int    inputID, sampleID, curSymbol, nsym;
  double scale, ddata, dpower;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  if (nSamples_ == 0)
  {
    printf("FactorialSampling initialize WARNING - nSamples = 0.\n");
    printf("NOTE: If no symbols are detected, an error will be flagged.\n");
  }
  if (nInputs_ == 0)
  {
    printf("FactorialSampling initialize ERROR - input not set up.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ copy or generate symbol table
  //**/ ----------------------------------------------------------------
  if (vecInpSettings_ != NULL)
  {
    if (printLevel_ > 4)
       printf("FactorialSampling initialize: inputSettings used.\n");
    if (printLevel_ > 4 && vecSymTable_.length() > 0)
       printf("FactorialSampling initialize: symbol table overwritten.\n");
    maxNumSettings_ = 0;
    for (inputID = 0; inputID < nInputs_; inputID++) 
      if (vecInpNumSettings_[inputID] > maxNumSettings_)
        maxNumSettings_ = vecInpNumSettings_[inputID];
    
    vecSymTable_.setLength(nInputs_);
    for (inputID = 0; inputID < nInputs_; inputID++) 
      vecSymTable_[inputID] = vecInpNumSettings_[inputID];
    nSamples_ = 1;
    for (inputID = 0; inputID < nInputs_; inputID++) 
      nSamples_ *= vecSymTable_[inputID];
    if (nSamples_ == 0)
    {
      printf("FactorialSampling::initialize ERROR - ");
      printf("incomplete inputSettings.\n");
      exit(1);
    }
  } 
  else if (vecSymTable_.length() != nInputs_)
  {
    if (nSamples_ <= 0)
    {
      printf("FactorialSampling initialize ERROR: nSamples = 0.\n");
      exit(1);
    }
    ddata  = (double) nSamples_;
    dpower = 1.0 / (double) nInputs_;
    ddata  = pow(ddata, dpower+1.0E-12);
    nsym   = (int) ddata;
    vecSymTable_.setLength(nInputs_);
    for (inputID = 0; inputID < nInputs_; inputID++) 
      vecSymTable_[inputID] = nsym;
    nSamples_ = 1;
    for (inputID = 0; inputID < nInputs_; inputID++) 
      nSamples_ *= vecSymTable_[inputID];
  }
  if (initLevel == 1) return 0;

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 4)
  {
    printf("FactorialSampling initialize: nSamples = %d\n", nSamples_);
    printf("FactorialSampling initialize: nInputs  = %d\n", nInputs_);
    printf("FactorialSampling initialize: nOutputs = %d\n", nOutputs_);
    if (randomize_ != 0)
         printf("FactorialSampling initialize: randomize on\n");
    else printf("FactorialSampling initialize: randomize off\n");
    for (inputID = 0; inputID < nInputs_; inputID++)
      printf("    FactorialSampling input %3d = [%e %e]\n", inputID+1,
             vecLBs_[inputID], vecUBs_[inputID]);
    if (vecSymTable_.length() == nInputs_)
      for (inputID = 0; inputID < nInputs_; inputID++) 
            printf("    FactorialSampling symbol table %2d = %d\n",
                   inputID+1, vecSymTable_[inputID]);
  }

  //**/ ----------------------------------------------------------------
  //**/ fill the permutation matrix
  //**/ ----------------------------------------------------------------
  allocSampleData();
  if (nSamples_ == 1)
  {
    for (inputID = 0; inputID < nInputs_; inputID++) 
    {
      vecSamInps_[0*nInputs_+inputID] = 0.5 * (vecLBs_[inputID] +
                                               vecUBs_[inputID]);
    }
    return 0;
  }
  psVector  vecRanges;
  psIVector vecCntMax, vecCnts, vecSyms;
  vecRanges.setLength(nInputs_);
  for (inputID = 0; inputID < nInputs_; inputID++)
    vecRanges[inputID] = vecUBs_[inputID] - vecLBs_[inputID];
  vecCntMax.setLength(nInputs_);
  vecCnts.setLength(nInputs_);
  vecSyms.setLength(nInputs_);
  vecCntMax[0] = 1;
  for (inputID = 1; inputID < nInputs_; inputID++)
    vecCntMax[inputID] = vecCntMax[inputID-1] * vecSymTable_[inputID-1];
  for (inputID = 0; inputID < nInputs_; inputID++)
    vecCnts[inputID] = vecSyms[inputID] = 0;

  if (randomize_ != 0)
  {
    for (sampleID = 0; sampleID < nSamples_; sampleID++)
    {
      for (inputID = 0; inputID < nInputs_; inputID++) 
      {
        curSymbol = vecSyms[inputID];
        vecCnts[inputID]++;
        if (vecCnts[inputID] == vecCntMax[inputID])
        {
          vecCnts[inputID] = 0;
          vecSyms[inputID]++;
          if (vecSyms[inputID] >= vecSymTable_[inputID]) 
            vecSyms[inputID] = 0;
        }
        scale = vecRanges[inputID] / ((double) (vecSymTable_[inputID]));
        vecSamInps_[sampleID*nInputs_+inputID] = 
                        curSymbol * scale + vecLBs_[inputID];
        vecSamInps_[sampleID*nInputs_+inputID] += (PSUADE_drand() * scale);
      }
    }
  }
  else
  {
    for (sampleID = 0; sampleID < nSamples_; sampleID++)
    {
      for (inputID = 0; inputID < nInputs_; inputID++) 
      {
        curSymbol = vecSyms[inputID];
        vecCnts[inputID]++;
        if (vecCnts[inputID] == vecCntMax[inputID])
        {
          vecCnts[inputID] = 0;
          vecSyms[inputID]++;
          if (vecSyms[inputID] >= vecSymTable_[inputID]) 
            vecSyms[inputID] = 0;
        }
        scale = vecRanges[inputID] / ((double) (vecSymTable_[inputID] - 1));
        if (vecInpNumSettings_.length() > 0)
           vecSamInps_[sampleID*nInputs_+inputID] =
                         vecInpSettings_[inputID][curSymbol]; 
        else
           vecSamInps_[sampleID*nInputs_+inputID] = 
                         curSymbol * scale + vecLBs_[inputID];
      }
    }
  }
  return 0;
}

// ************************************************************************
// refine the sample space
// ------------------------------------------------------------------------
int FactorialSampling::refine(int refineRatio, int randomize, double thresh,
                              int nSamples, double *sampleErrors)
{
  int    newNSamples, inputID, *useArray, *iCounts, *iCntMax, *iSymbol;
  int    sampleID, index, curSymbol, *newSampleStates, outputID, nLevels;
  double **newSampleMatrix, stepSize, ddata;
  double *newSampleOutput;

  //**/ ----------------------------------------------------------------
  //**/ unused parameters
  //**/ ----------------------------------------------------------------
  (void) randomize;
  (void) thresh;
  (void) nSamples;
  (void) sampleErrors;

  //**/ ----------------------------------------------------------------
  //**/ error checking (only support nLevels = 2)
  //**/ ----------------------------------------------------------------
  nLevels = refineRatio;
  if (nLevels != 2)
  {
    printf("FactorialSampling refine WARNING - nLevels set to 2.\n");
    nLevels = 2;
  }

  //**/ ----------------------------------------------------------------
  //**/ update internal data : compute the new nSamples
  //**/ ----------------------------------------------------------------
  newNSamples = 1;
  if (randomize_ != 0)
  {
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      vecSymTable_[inputID] *= 2;
      newNSamples *= vecSymTable_[inputID];
    }
  }
  else
  {
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      vecSymTable_[inputID] = vecSymTable_[inputID] * 2 - 1;
      newNSamples *= vecSymTable_[inputID];
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for registering old data points and others
  //**/ ----------------------------------------------------------------
  psVector vecRanges;
  vecRanges.setLength(nInputs_);
  for (inputID = 0; inputID < nInputs_; inputID++)
    vecRanges[inputID] = vecUBs_[inputID] - vecLBs_[inputID];
  psIVector vecUseArray;
  vecUseArray.setLength(newNSamples);
  for (sampleID = 0; sampleID < newNSamples; sampleID++)
    vecUseArray[sampleID] = 0;
  psIVector vecCntMax, vecCnts, vecSyms;
  vecCntMax.setLength(nInputs_);
  vecCnts.setLength(nInputs_);
  vecSyms.setLength(nInputs_);
  vecCntMax[0] = 1;
  for (inputID = 1; inputID < nInputs_; inputID++)
    vecCntMax[inputID] = vecCntMax[inputID-1] * vecSymTable_[inputID-1];
  for (inputID = 0; inputID < nInputs_; inputID++)
    vecCnts[inputID] = vecSyms[inputID] = 0;

  //**/ ----------------------------------------------------------------
  //**/ register old data points (useArray)
  //**/ ----------------------------------------------------------------
  for (sampleID = 0; sampleID < nSamples_; sampleID++)
  {
    index = 0;
    for (inputID = 0; inputID < nInputs_; inputID++) 
    {
      if (randomize_ != 0)
         stepSize = ((double) (vecSymTable_[inputID]))/vecRanges[inputID];
      else
         stepSize = ((double) (vecSymTable_[inputID]-1))/vecRanges[inputID];
      ddata = vecSamInps_[sampleID*nInputs_+inputID] - vecLBs_[inputID];
      curSymbol = (int) (ddata * stepSize * 1.00000000001);
      index += curSymbol * vecCntMax[inputID];
    } 
    vecUseArray[index] = sampleID + 1;
    //printf("Fact: use %d = %d\n", index, useArray[index]);
  }

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for new samples
  //**/ ----------------------------------------------------------------
  psVector  vecSamInps2, vecSamOuts2;
  psIVector vecSamStas2;
  
  vecSamInps2 = vecSamInps_;
  vecSamOuts2 = vecSamOuts2;
  vecSamStas2 = vecSamStas2;
  vecSamInps_.setLength(newNSamples*nInputs_);
  vecSamOuts_.setLength(newNSamples*nOutputs_);
  vecSamStas_.setLength(newNSamples);

  for (sampleID = 0; sampleID < newNSamples*nOutputs_; sampleID++)
    vecSamOuts_[sampleID] = PSUADE_UNDEFINED;
  for (sampleID = 0; sampleID < newNSamples; sampleID++)
    vecSamStas_[sampleID] = 0;
   
  //**/ ----------------------------------------------------------------
  //**/ generate new samples 
  //**/ ----------------------------------------------------------------
  for (sampleID = 0; sampleID < newNSamples; sampleID++)
  {
    index = -1;
    if (vecUseArray[sampleID] > 0) index = vecUseArray[sampleID] - 1;

    if (index >= 0)
    {
      for (inputID = 0; inputID < nInputs_; inputID++) 
        vecSamInps_[sampleID*nInputs_+inputID] = 
                        vecSamInps2[index*nInputs_+inputID]; 
      for (outputID = 0; outputID < nOutputs_; outputID++) 
        vecSamOuts_[sampleID*nOutputs_+outputID] = 
                        vecSamOuts2[index*nOutputs_+outputID]; 
      vecSamStas_[sampleID] = vecSamStas2[index];
    }
    for (inputID = 0; inputID < nInputs_; inputID++) 
    {
      curSymbol = vecSyms[inputID];
      vecCnts[inputID]++;
      if (vecCnts[inputID] == vecCntMax[inputID])
      {
        vecCnts[inputID] = 0;
        vecSyms[inputID]++;
        if (vecSyms[inputID] >= vecSymTable_[inputID]) 
          vecSyms[inputID] = 0;
      }
      if (index < 0)
      {
        if (randomize_ != 0)
        {
          stepSize = vecRanges[inputID]/((double) (vecSymTable_[inputID]));
          vecSamInps_[sampleID*nInputs_+inputID] = vecLBs_[inputID] +
                                (curSymbol + PSUADE_drand()) * stepSize;
        }
        else
        {
          stepSize = vecRanges[inputID]/((double) (vecSymTable_[inputID]-1));
          vecSamInps_[sampleID*nInputs_+inputID] = 
                           curSymbol * stepSize + vecLBs_[inputID];
        }
      }
    }
  }
  nSamples_ = newNSamples;

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 4)
  {
    printf("FactorialSampling refine: nSamples = %d\n", nSamples_);
    printf("FactorialSampling refine: nInputs  = %d\n", nInputs_);
    printf("FactorialSampling refine: nOutputs = %d\n", nOutputs_);
    if (randomize_ != 0)
         printf("FactorialSampling refine: randomize on\n");
    else printf("FactorialSampling refine: randomize off\n");
    for (inputID = 0; inputID < nInputs_; inputID++)
      printf("    FactorialSampling input %3d = [%e %e]\n", inputID+1,
             vecLBs_[inputID], vecUBs_[inputID]);
    if (vecSymTable_.length() == nInputs_)
      for (inputID = 0; inputID < nInputs_; inputID++) 
        printf("    FactorialSampling symbol table %2d = %d\n",
               inputID+1, vecSymTable_[inputID]);
  }
  return 0;
}

// ************************************************************************
// set input settings
// ------------------------------------------------------------------------
int FactorialSampling::setInputParams(int nInputs, int *counts, 
                                      double **settings, int *symtable)
{
  int inputID, sID;

  if (nInputs_ != 0 && nInputs != nInputs_)
  {
    printf("FactorialSampling setInputParams ERROR: nInputs mismatch.\n");
    exit(1);
  }
  nInputs_ = nInputs;
  if (symtable != NULL)
  {
    vecSymTable_.setLength(nInputs);
    for (inputID = 0; inputID < nInputs_; inputID++)
      vecSymTable_[inputID] = symtable[inputID];
  }
  if (counts != NULL)
  {
    maxNumSettings_ = 0;
    for (inputID = 0; inputID < nInputs_; inputID++) 
      if (counts[inputID] > maxNumSettings_)
        maxNumSettings_ = counts[inputID];
    vecInpNumSettings_.setLength(nInputs_);
    vecInpSettings_ = new psVector[nInputs_];
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      vecInpSettings_[inputID].setLength(counts[inputID]);
      vecInpNumSettings_[inputID] = counts[inputID];
      if (settings != NULL)
      {
        for (sID = 0; sID < counts[inputID]; sID++)
          vecInpSettings_[inputID][sID] = settings[inputID][sID];
      }
      else
      {
        for (sID = 0; sID < counts[inputID]; sID++)
          vecInpSettings_[inputID][sID] = 
            (vecUBs_[inputID] - vecLBs_[inputID]) /
            (counts[inputID] - 1.0) * sID + vecLBs_[inputID];
      }
    }
  }
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
FactorialSampling& FactorialSampling::operator=(const FactorialSampling &)
{
  printf("FactorialSampling operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// check whether the incoming sample is factorial or not
// ------------------------------------------------------------------------
int FactorialSampling::checkFactorial(int nSamps, int nInps, 
                                  psVector vecInputs, psIVector &vecDims)
{
  if (nInps != 3)
  {
    printf("ERROR: This checkFactorial function currently supports ");
    printf("only nInputs=3.\n");
    return 1;
  }
  int ii;
  psVector vecD1, vecD2, vecD3;
  vecD1.setLength(nSamps);
  vecD2.setLength(nSamps);
  vecD3.setLength(nSamps);
  for (ii = 0; ii < nSamps; ii++)
  {
    vecD1[ii] = vecInputs[nInps*ii];
    vecD2[ii] = vecInputs[nInps*ii+1];
    vecD3[ii] = vecInputs[nInps*ii+2];
  }
  //**/ find n1, n2, and n3
  int n1, n2, n3;
  for (ii = 1; ii < nSamps; ii++)
    if (vecD1[ii] == vecD1[0]) break;
  n1 = ii;
  for (ii = n1; ii < nSamps; ii+=n1)
    if (vecD2[ii] == vecD2[0]) break;
  n2 = ii / n1;
  n3 = nSamps / n1 / n2;
  if (n1 * n2 * n3 != nSamps) return 1;

  //**/ check that the first n1 elements of input 1 are increasing
  for (ii = 1; ii < n1; ii++)
    if (vecD1[ii] <= vecD1[ii-1]) break;
  if (ii != n1) return 1; 

  //**/ check that the first n1 elements of input 1 are repeating
  int kk;
  for (kk = 1; kk < n2*n3; kk++)
  {
    for (ii = 0; ii < n1; ii++)
      if (vecD1[kk*n1+ii] != vecD1[ii]) break;
    if (ii != n1) break;
  }
  if (kk != n2*n3) return 1; 

  //**/ check that the first n2 elements of input 2 are increasing
  for (ii = 1; ii < n2; ii++)
    if (vecD2[ii*n1] <= vecD2[(ii-1)*n1]) break;
  if (ii != n2) return 1;

  //**/ check that each group of n1 elements of input 2 are the same
  for (kk = 0; kk < n2*n3; kk++)
  {
    for (ii = 1; ii < n1; ii++)
      if (vecD2[kk*n1+ii] != vecD2[kk*n1]) break;
    if (ii != n1) break;
  }
  if (kk != n2*n3) return 1;

  //**/ check that the first n2 elements of input 2 are repeating
  for (kk = 1; kk < n3; kk++)
  {
    for (ii = 0; ii < n1*n2; ii++)
      if (vecD2[kk*n1*n2+ii] != vecD2[ii]) break;
    if (ii != n1*n2) break;
  }
  if (kk != n3) return 1;

  //**/ check that the first n3 elements of input 3 are increasing
  for (ii = 1; ii < n3; ii++)
    if (vecD3[ii*n1*n2] <= vecD3[(ii-1)*n1*n2]) break;
  if (ii != n3) return 1;

  //**/ check that the n3 elements of input 3 are repeating
  for (kk = 0; kk < n3; kk++)
  {
    for (ii = 1; ii < n1*n2; ii++)
      if (vecD3[kk*n1*n2+ii] != vecD3[kk*n1*n2]) break;
    if (ii != n1*n2) break;
  }
  if (kk != n3) return 1; 
  vecDims.setLength(nInps);
  vecDims[0] = n1;
  vecDims[1] = n2;
  vecDims[2] = n3;
  return 0;
}

