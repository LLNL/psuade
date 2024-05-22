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
// Functions for the OA-based Latin Hypercube sampling class
// AUTHOR : CHARLES TONG
// DATE   : 2004
// ************************************************************************
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "OALHSampling.h"
#include "psVector.h"
#include "psMatrix.h"

// ************************************************************************
// link to external OA generator
// ************************************************************************
#ifdef HAVE_BOSE
extern "C" 
{
  int  bose_link(int n, int ninputs, int str, int ***AA);
  void OA_strength(int q,int nrow,int ncol,int** A,int *str,int verbose);
}
#endif

// ************************************************************************
// Constructor
// ------------------------------------------------------------------------
OALHSampling::OALHSampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_OALH;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
OALHSampling::~OALHSampling()
{
}

// ************************************************************************
// initialize sampling data
// ------------------------------------------------------------------------
int OALHSampling::initialize(int initLevel)
{
#ifndef HAVE_BOSE
  printf("OALHSampling ERROR: Bose library not installed.\n");
  exit(1);
#else
  int    inputID, sampleID, strength, offset, nReps;
  int    repID, status, **itempMatrix, nSamples1;
  int    symbolID, symbolID2, index, nSubSamples, nSamples2;
  double scale, ddata;
  psVector  vecRanges;
  psIVector vecPerm, vecI1, vecI2;
  psMatrix  matPerturb;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  if (nInputs_ == 0)
  {
    printf("OALHSampling ERROR: input not set up.\n");
    exit(1);
  }
  if (nSamples_ == 0)
  {
    printf("OALHSampling ERROR: nSamples = 0.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ clean up, set parameters, and allocate storage
  //**/ ----------------------------------------------------------------
  nReps = nReplications_;
  if (nSamples_ / nReps * nReps != nSamples_)
  {
    printf("OALHSampling ERROR: nSamples not multiples of replications.\n");
    exit(1);
  }
  ddata     = (double) (nSamples_ / nReps);
  ddata     = pow(ddata, 0.5000001);
  nSymbols_ = (int) ddata;
  nSamples1  = nSymbols_ * nSymbols_ * nReps;
  if (nSamples1 < nSamples_)
  {
    nSamples2 = (nSymbols_ + 1) * (nSymbols_ + 1) * nReps;
    if ((nSamples_ - nSamples1) < (nSamples2 - nSamples_))
       nSamples_ = nSamples1;
    else
    {
      nSamples_ = nSamples2;
      nSymbols_++;
    }
  }
  if (initLevel != 0) return 0;
  allocSampleData();

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 4)
  {
    printf("OALHSampling initialize: nSamples = %d\n", nSamples_);
    printf("OALHSampling initialize: nInputs  = %d\n", nInputs_);
    printf("OALHSampling initialize: nOutputs = %d\n", nOutputs_);
    if (randomize_ != 0)
         printf("OALHSampling initialize: randomize on\n");
    else printf("OALHSampling initialize: randomize off\n");
    for (inputID = 0; inputID < nInputs_; inputID++)
       printf("    OALHSampling input %3d = [%e %e]\n", inputID+1,
              vecLBs_[inputID], vecUBs_[inputID]);
    if (vecInpNumSettings_.length() > 0)
    {
      for (inputID = 0; inputID < nInputs_; inputID++)
      {
        if (vecInpNumSettings_[inputID] != nSymbols_)
        {
          printf("OALHSampling ERROR: inputSetting not compabible.\n");
          exit(1);
        }
      }
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ generate the matrix
  //**/ ----------------------------------------------------------------
  psIMatrix matOASamples;
  matOASamples.setFormat(2);
  matOASamples.setDim(nSamples_, nInputs_);
  int **OAMatrix = matOASamples.getIMatrix2D();

  nReps       = nSamples_ / (nSymbols_ * nSymbols_);
  nSubSamples = nSamples_ / nReps;
  vecI1.setLength(nSymbols_);
  strength = 2;
  offset   = 0;

  for (repID = 0; repID < nReps; repID++)
  {
    //**/ use the Bose tool to generate OA
    status = bose_link(nSubSamples, nInputs_, strength, &itempMatrix);
    if ((status >= 0 && status != nSubSamples) || status < 0)
    {
      printf("OALHSampling ERROR: Bose failure.\n");
      printf("    ==> Consult PSUADE developers for help.\n");
      exit(1);
    }
    for (sampleID = 0; sampleID < nSubSamples; sampleID++) 
    {
      for (inputID = 0; inputID < nInputs_; inputID++) 
        OAMatrix[offset+sampleID][inputID] = 
                            itempMatrix[sampleID][inputID];
      free(itempMatrix[sampleID]);
    }
    free(itempMatrix);

    //**/ randomize the permutation
    for (inputID = 0; inputID < nInputs_; inputID++) 
    {
      generateRandomIvector(nSymbols_, vecI1.getIVector());
      for (sampleID = 0; sampleID < nSubSamples; sampleID++) 
        OAMatrix[offset+sampleID][inputID] = 
                     vecI1[OAMatrix[offset+sampleID][inputID]];
    }

    //**/ check for strength of 2
    OA_strength(nSymbols_,nSubSamples, nInputs_, &(OAMatrix[offset]),
                &strength, 0);
    if (strength != 2)
    {
      printf("OALHSampling ERROR: failed strength 2 test.\n");
      printf("   ==> Please consult PSUADE developers.\n");
      exit(1);
    }
    offset += nSubSamples;
  }

  //**/ ----------------------------------------------------------------
  //**/ permute to form Latin Hypercube
  //**/ ----------------------------------------------------------------
  psIMatrix matI;
  matI.setFormat(2);
  matI.setDim(nSymbols_, nSymbols_);
  int **iMatrix = matI.getIMatrix2D();
  for (symbolID = 0; symbolID < nSymbols_; symbolID++) 
    iMatrix[symbolID] = new int[nSymbols_];
  vecI1.setLength(nSymbols_);
  vecI2.setLength(nSymbols_);
  vecPerm.setLength(nSubSamples);
  offset  = 0;

  for (repID = 0; repID < nReps; repID++)
  {
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      for (symbolID = 0; symbolID < nSymbols_; symbolID++)
      {
        for (symbolID2 = 0; symbolID2 < nSymbols_; symbolID2++)
          vecI1[symbolID2] = symbolID * nSymbols_ + symbolID2;
        generateRandomIvector(nSymbols_, vecI2.getIVector());
        for (symbolID2 = 0; symbolID2 < nSymbols_; symbolID2++ )
          iMatrix[symbolID][symbolID2] = vecI1[vecI2[symbolID2]];
      }
      for (symbolID = 0; symbolID < nSymbols_; symbolID++) 
        vecI1[symbolID] = 0;
      for (sampleID = 0; sampleID < nSubSamples; sampleID++)
      {
        index = OAMatrix[offset+sampleID][inputID];
        vecPerm[sampleID] = iMatrix[index][vecI1[index]++];
      }
      for (sampleID = 0; sampleID < nSubSamples; sampleID++) 
        OAMatrix[offset+sampleID][inputID] = vecPerm[sampleID];
    }
    offset += nSubSamples;
  }

  //**/ ----------------------------------------------------------------
  //**/ checking for consistency and clean up
  //**/ ----------------------------------------------------------------

  //**/ strength = 1 test
  if (nSamples_ < 1000)
  {
    offset = 0;
    for (repID = 0; repID < nReps; repID++)
    {
      OA_strength(nSubSamples,nSubSamples,nInputs_,
                  &(OAMatrix[nSubSamples*repID]),&strength,0);
      if (strength != 1)
      {
        printf("OALHSampling ERROR: failed strength 1 test.\n");
        printf("   ==> Please consult PSUADE developers.\n");
        exit(1);
      }
      offset += nSubSamples;
    }

    //**/ strength = 2 test
    matI.setDim(nSamples_, nInputs_);
    iMatrix = matI.getIMatrix2D();
    offset = 0;

    for (repID = 0; repID < nReps; repID++)
    {
      for (sampleID = 0; sampleID < nSubSamples; sampleID++)
        for (inputID = 0; inputID < nInputs_; inputID++)
          iMatrix[sampleID][inputID] = 
                   OAMatrix[offset+sampleID][inputID] / nSymbols_;
      OA_strength(nSymbols_, nSubSamples, nInputs_, iMatrix, &strength, 0);
      if (strength != 2)
      {
        printf("OALHSampling ERROR: failed strength 2 test.\n");
        printf("   ==> Please consult PSUADE developers.\n");
        exit(1);
      }
      offset += nSubSamples;
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ generate actual input data from permMatrix
  //**/ ----------------------------------------------------------------
  double **perturbMatrix;
  vecRanges.setLength(nInputs_);
  for (inputID = 0; inputID < nInputs_; inputID++) 
    vecRanges[inputID] = vecUBs_[inputID] - vecLBs_[inputID];
  if (randomize_ != 0)
  {
    matPerturb.setFormat(2);
    matPerturb.setDim(nInputs_, nSymbols_*nSymbols_);
    perturbMatrix = matPerturb.getMatrix2D();
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      for (symbolID = 0; symbolID < nSymbols_*nSymbols_; symbolID++)
        perturbMatrix[inputID][symbolID] = PSUADE_drand();
    }
    scale = 1.0 / (double) nSubSamples;
    for (sampleID = 0; sampleID < nSamples_; sampleID++)
    {
      for (inputID = 0; inputID < nInputs_; inputID++)
      {
        index = OAMatrix[sampleID][inputID];
        ddata = (perturbMatrix[inputID][index] + index) * scale;
        vecSamInps_[sampleID*nInputs_+inputID] = 
                   ddata * vecRanges[inputID] + vecLBs_[inputID];
      }
    }
  }
  else
  {
    scale = 1.0 / (double) (nSubSamples - 1);
    for (sampleID = 0; sampleID < nSamples_; sampleID++)
    {
      for (inputID = 0; inputID < nInputs_; inputID++)
      {
        if (vecInpSettings_ != NULL && vecInpSettings_[inputID].length()>0)
        {
          index = OAMatrix[sampleID][inputID];
          vecSamInps_[sampleID*nInputs_+inputID] = 
                   vecInpSettings_[inputID][index];
        }
        else
        {
          ddata = (double) OAMatrix[sampleID][inputID];
          ddata = ddata * scale;
          vecSamInps_[sampleID*nInputs_+inputID] = 
                   ddata * vecRanges[inputID] + vecLBs_[inputID];
        }
      }
    }
  }
#endif
  return 0;
}

// ************************************************************************
// perform refinements
// ------------------------------------------------------------------------
int OALHSampling::refine(int refineRatio, int randomize, double thresh,
                         int nSamples, double *sampleErrors)
{
#ifdef HAVE_BOSE
  int    nReps, symMult, *factors, ncount, ind1, samMult, strength;
  int    status, **itempMatrix, newNSymbols, newNSamples;
  int    sampleID, repID, inputID, symbolID, nSubSamples;
  int    nFactors, sampleOffset, newSampleOffset, ind2, sampleID2;
  int    newNSubSamples, *symbolArray;
  int    symbolID2, outputID, nLevels;
  double scale, dtemp;
  psVector  vecRanges, vecNewSamInps, vecNewSamOuts;
  psIVector vecI1, vecI2, vecI3, vecNewSamStas;
  psMatrix  matPerturb, matBounds;
  psIMatrix matI, matNewOA, matOASamples;

  //**/ ----------------------------------------------------------------
  //**/ unused parameters
  //**/ ----------------------------------------------------------------
  (void) randomize;
  (void) thresh;
  (void) nSamples;
  (void) sampleErrors;

  //**/ ----------------------------------------------------------------
  //**/ check 1 : prime power
  //**/ ----------------------------------------------------------------
  printf("OALHSampling refine - not implemented yet.\n");
  return -1;
  nLevels = refineRatio;
  nReps = nSamples_ / (nSymbols_ * nSymbols_);
  nSubSamples = nSamples_ / nReps;
  symMult = nLevels;
  factors  = factorize(nSamples_/nReps);
  nFactors = factors[0];
  ncount = 0;
  for (ind1 = 2; ind1 < nFactors; ind1++)
    if (factors[ind1] != factors[ind1-1]) ncount++;
  if (ncount > 1)
  {
    printf("OALHSampling refine ERROR: %d not prime power.\n",
           nSamples_/nReps);
    exit(1);
  }
  nLevels = nInputs_ - 1;
  printf("OALHSampling refine INFO: nLevels set to %d.\n",nLevels);
  symMult = factors[1] * nLevels;
  delete [] factors;

  //**/ ----------------------------------------------------------------
  //**/ check 2 : nLevels enough for nInputs?
  //**/ ----------------------------------------------------------------
  samMult = symMult * symMult;
  strength = 2;
  status = bose_link(samMult, nInputs_, strength, &itempMatrix);
  if (status < 0)
  {
    printf("OASampling refine ERROR: cannot refine.\n");
    printf("         ==> Consult PSUADE developers for help.\n");
    exit(1);
  }
  for (sampleID = 0; sampleID < samMult; sampleID++)
    if (itempMatrix[sampleID] != NULL) free(itempMatrix[sampleID]);
  free(itempMatrix);

  //**/ ----------------------------------------------------------------
  //**/ initialization
  //**/ ----------------------------------------------------------------
  randomize_ = 1;
  newNSymbols    = nSymbols_ * symMult;
  newNSamples    = newNSymbols * newNSymbols * nReps;
  newNSubSamples = newNSamples / nReps;

  //**/ ----------------------------------------------------------------
  //**/ create new perturbation matrix (actually no binning is needed)
  //**/ ==> newPerturb
  //**/ ----------------------------------------------------------------
  vecRanges.setLength(nInputs_);
  for (inputID = 0; inputID < nInputs_; inputID++)
    vecRanges[inputID] = vecUBs_[inputID] - vecLBs_[inputID];
  matBounds.setFormat(2);
  matBounds.setDim(nInputs_, newNSubSamples+1);
  double **bounds = matBounds.getMatrix2D();
 
  for (inputID = 0; inputID < nInputs_; inputID++)
  {
    for (symbolID = 0; symbolID <= newNSubSamples; symbolID++)
      bounds[inputID][symbolID] = vecRanges[inputID] / newNSubSamples *
                                  symbolID + vecLBs_[inputID];
  }
  matPerturb.setFormat(2);
  matPerturb.setDim(nInputs_, newNSubSamples);
  double **perturbMatrix = matPerturb.getMatrix2D();
  for (inputID = 0; inputID < nInputs_; inputID++)
  {
    for (symbolID = 0; symbolID < newNSubSamples; symbolID++)
      perturbMatrix[inputID][symbolID] = PSUADE_drand();
  }
  for (sampleID = 0; sampleID < nSubSamples; sampleID++)
  {
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      dtemp = vecSamInps_[sampleID*nInputs_+inputID];
      for (symbolID = 1; symbolID <= newNSubSamples; symbolID++)
         if (dtemp < bounds[inputID][symbolID]) break;
      symbolID--;
      if (symbolID >= newNSubSamples)
      {
         printf("OALHSampling refine ERROR (3)\n");
         exit(1);
      }
      dtemp -= vecLBs_[inputID];
      dtemp  = dtemp / vecRanges[inputID] * newNSubSamples;
      dtemp -= (double) symbolID;
      perturbMatrix[inputID][symbolID] = dtemp;
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ recover the OA patterns ==> OAMatrix
  //**/ ----------------------------------------------------------------
  for (inputID = 0; inputID < nInputs_; inputID++)
  {
    for (symbolID = 0; symbolID <= newNSymbols; symbolID++)
      bounds[inputID][symbolID] = vecRanges[inputID] / newNSymbols *
                                     symbolID + vecLBs_[inputID];
  }
  matOASamples.setFormat(2);
  matOASamples.setDim(nSamples_, nInputs_);
  int **OAMatrix = matOASamples.getIMatrix2D();
  for (sampleID = 0; sampleID < nSamples_; sampleID++)
  {
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      dtemp = vecSamInps_[sampleID*nInputs_+inputID];
      for (symbolID = 1; symbolID <= newNSymbols; symbolID++)
        if (dtemp < bounds[inputID][symbolID]) break;
      OAMatrix[sampleID][inputID] = symbolID - 1;
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ create new patterns ==> newOAMatrix
  //**/ ----------------------------------------------------------------
  matNewOA.setFormat(2);
  matNewOA.setDim(newNSamples, nInputs_);
  int **newOAMatrix = matNewOA.getIMatrix2D();

  sampleOffset = 0;
  newSampleOffset = 0;

  for (repID = 0; repID < nReps; repID++)
  {
    for (sampleID = 0; sampleID < nSubSamples; sampleID++)
      for (inputID = 0; inputID < nInputs_; inputID++)
        newOAMatrix[newSampleOffset+sampleID][inputID] =
             OAMatrix[sampleOffset+sampleID][inputID];
    newSampleOffset += nSubSamples;

    for (sampleID = 0; sampleID < nSubSamples; sampleID++)
    {
      status = bose_link(samMult, nInputs_, strength, &itempMatrix);

      //**/ permute to match the first pattern
      for (inputID = 0; inputID < nInputs_; inputID++)
      {
        ind1 = itempMatrix[0][inputID];
        ind2 = OAMatrix[sampleOffset+sampleID][inputID] % symMult;
        if (ind1 != ind2)
        {
          for (sampleID2 = 1; sampleID2 < samMult; sampleID2++)
          {
            ind1 = itempMatrix[sampleID2][inputID];
            if (ind1 == ind2) break;
          }
        }
        if (ind1 == ind2)
        {
          ind1 = itempMatrix[0][inputID];
          for (sampleID2 = 0; sampleID2 < samMult; sampleID2++)
          {
            if (itempMatrix[sampleID2][inputID] == ind1)
              itempMatrix[sampleID2][inputID] = ind2;
            else if (itempMatrix[sampleID2][inputID] == ind2)
              itempMatrix[sampleID2][inputID] = ind1;
          }
        }
      }

      //**/ record the new set of pattern
      for (sampleID2 = 1; sampleID2 < samMult; sampleID2++)
      {
        for (inputID = 0; inputID < nInputs_; inputID++)
        {
          ind1 = OAMatrix[sampleOffset+sampleID][inputID] / symMult; 
          newOAMatrix[newSampleOffset+sampleID2-1][inputID] = 
                  itempMatrix[sampleID2][inputID] + ind1 * symMult;
        }
      }
      newSampleOffset += (samMult - 1);

      //**/ clean itempMatrix
      for (sampleID2 = 0; sampleID2 < samMult; sampleID2++)
        if (itempMatrix[sampleID2] != NULL) free(itempMatrix[sampleID2]);
      free(itempMatrix);
    }
    sampleOffset += nSubSamples;
  }

  //**/ ----------------------------------------------------------------
  //**/ permute the new patterns (permute newOAMatrix)
  //**/ ----------------------------------------------------------------
  newSampleOffset = 0;

  for (repID = 0; repID < nReps; repID++)
  {
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      for (symbolID = 0; symbolID < newNSymbols; symbolID++)
        symbolArray[symbolID] = symbolID;
      for (sampleID = 0; sampleID < nSubSamples; sampleID++)
      {
        ind1 = newOAMatrix[newSampleOffset+sampleID][inputID];
        symbolArray[ind1] = -1;
      }
      ncount = 0;
      for (symbolID = 0; symbolID < newNSymbols; symbolID++)
        if (symbolArray[symbolID] >= 0) ncount++;
      if (ncount > 1)
      {
        ncount = 0;
        for (symbolID = 0; symbolID < newNSymbols; symbolID++)
          if (symbolArray[symbolID] >= 0)
            vecI3[ncount++] = symbolArray[symbolID];
        generateRandomIvector(ncount, vecI1.getIVector());
        for (symbolID = 0; symbolID < ncount; symbolID++)
          vecI2[symbolID] = vecI3[vecI1[symbolID]];
        ncount = 0;
        for (symbolID = 0; symbolID < newNSymbols; symbolID++)
        {
          if (symbolArray[symbolID] >= 0)
            symbolArray[symbolID] = vecI2[ncount++];
          else
            symbolArray[symbolID] = symbolID;
        }
        for (sampleID = 0; sampleID < nSubSamples; sampleID++)
        {
          ind1 = newOAMatrix[newSampleOffset+sampleID][inputID];
          newOAMatrix[newSampleOffset+sampleID][inputID] = 
                                               symbolArray[ind1];
        }
      }
    }

    //**/ check strength
    OA_strength(newNSymbols, newNSubSamples, nInputs_,
                &(newOAMatrix[newSampleOffset]), &strength, 0);
#if 0
    for (sampleID = 0; sampleID < newNSubSamples; sampleID++)
    {
      printf("sample %3d (%3d) : ",sampleID, currOffset);
      for (inputID = 0; inputID < nInputs_; inputID++)
        printf(" %3d ", OASamples[newSampleOffset+sampleID][inputID]);
      printf("\n");
    }
#endif
    if (strength != 2)
    {
      printf("OALHSampling refine ERROR: strength %d != 2\n",strength);
      exit(1);
    }
    newSampleOffset += newNSubSamples;
  }

  //**/ ----------------------------------------------------------------
  //**/ permute to form Latin Hypercube ==> newOAMatrix
  //**/ ----------------------------------------------------------------
  matI.setFormat(2);
  matI.setDim(newNSymbols, newNSymbols);
  int **iMatrix = matI.getIMatrix2D();
  vecI1.setLength(newNSymbols);
  vecI2.setLength(newNSymbols);
  vecRanges.setLength(nInputs_);
  for (inputID = 0; inputID < nInputs_; inputID++)
    vecRanges[inputID] = vecUBs_[inputID] - vecLBs_[inputID];
  matBounds.setFormat(2);
  matBounds.setDim(nInputs_, newNSubSamples+1);
  bounds = matBounds.getMatrix2D();
  for (inputID = 0; inputID < nInputs_; inputID++)
  {
    for (symbolID = 0; symbolID <= newNSubSamples; symbolID++)
      bounds[inputID][symbolID] = vecRanges[inputID] / newNSubSamples *
                                  symbolID + vecLBs_[inputID];
  }

  sampleOffset = 0;
  newSampleOffset = 0;
  for (repID = 0; repID < nReps; repID++)
  {
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      for (symbolID = 0; symbolID < newNSymbols; symbolID++)
      {
        for (symbolID2 = 0; symbolID2 < newNSymbols; symbolID2++ )
           iMatrix[symbolID][symbolID2] = symbolID * newNSymbols +
                                          symbolID2;
      }
      for (sampleID = 0; sampleID < nSubSamples; sampleID++)
      {
        dtemp = vecSamInps_[(sampleOffset+sampleID)*nInputs_+inputID];
        for (symbolID = 1; symbolID <= newNSubSamples; symbolID++)
          if (dtemp < bounds[inputID][symbolID]) break;
        symbolID--;
        iMatrix[symbolID/newNSubSamples][symbolID%newNSubSamples] =
           - iMatrix[symbolID/newNSubSamples][symbolID%newNSubSamples] - 1;
      }
      for (symbolID = 0; symbolID < newNSymbols; symbolID++)
      {
        ncount = 0;
        for (symbolID2 = 0; symbolID2 < newNSymbols; symbolID2++)
          if (iMatrix[symbolID][symbolID2] >= 0) 
             iMatrix[symbolID][ncount++] = iMatrix[symbolID][symbolID2];
        if (ncount > 1)
        {
          for (symbolID2 = 0; symbolID2 < ncount; symbolID2++)
            vecI1[symbolID2] = iMatrix[symbolID][symbolID2];
          generateRandomIvector(ncount, vecI2.getIVector());
          for (symbolID2 = 0; symbolID2 < ncount; symbolID2++ )
            iMatrix[symbolID][symbolID2] = vecI1[vecI2[symbolID2]];
        }
      }
      for (symbolID = 0; symbolID < newNSymbols; symbolID++) 
        vecI1[symbolID] = 0;
      for (sampleID = nSubSamples; sampleID < newNSubSamples; sampleID++) 
      {
        ind1 = newOAMatrix[newSampleOffset+sampleID][inputID];
        ind2  = iMatrix[ind1][vecI1[ind1]++];
        if (ind2 < 0 || ind2 >= newNSubSamples)
        {
          printf("OALHSampling::refine ERROR : ind2 problem.\n");
          exit(1);
        }
        newOAMatrix[newSampleOffset+sampleID][inputID] = ind2;
      }
    }
    newSampleOffset += newNSubSamples;
    sampleOffset += nSubSamples;
  }

  //**/ ----------------------------------------------------------------
  //**/ create new sample matrices and initialize state vector
  //**/ ----------------------------------------------------------------
  vecNewSamInps.setLength(newNSamples*nInputs_);
  vecNewSamOuts.setLength(newNSamples*nOutputs_);
  vecNewSamStas.setLength(newNSamples);
  for (sampleID = 0; sampleID < newNSamples; sampleID++)
  {
    for (outputID = 0; outputID < nOutputs_; outputID++)
      vecNewSamOuts[sampleID*nOutputs_+outputID] = PSUADE_UNDEFINED;
  }

  //**/ ----------------------------------------------------------------
  //**/ generate actual input data from permMatrix
  //**/ ----------------------------------------------------------------
  vecRanges.setLength(nInputs_);
  for (inputID = 0; inputID < nInputs_; inputID++) 
    vecRanges[inputID] = vecUBs_[inputID] - vecLBs_[inputID];
  scale = 1.0 / (double) newNSubSamples;
  newSampleOffset = 0;
  sampleOffset = 0;
  for (repID = 0; repID < nReps; repID++)
  {
    for (sampleID = 0; sampleID < nSubSamples; sampleID++)
    {
      for (inputID = 0; inputID < nInputs_; inputID++)
        vecNewSamInps[(newSampleOffset+sampleID)*nInputs_+inputID] =
             vecSamInps_[(sampleOffset+sampleID)*nInputs_+inputID];
      for (outputID = 0; outputID < nOutputs_; outputID++)
        vecNewSamOuts[(newSampleOffset+sampleID)*nOutputs_+outputID] =
             vecSamOuts_[(sampleOffset+sampleID)*nOutputs_+outputID];
      vecNewSamStas[newSampleOffset+sampleID] =
            vecSamStas_[sampleOffset+sampleID];
    }
    for (sampleID = nSubSamples; sampleID < newNSamples; sampleID++)
    {
      for (inputID = 0; inputID < nInputs_; inputID++)
      {
        ind1  = newOAMatrix[newSampleOffset+sampleID][inputID];
        dtemp = (perturbMatrix[inputID][ind1 ] + ind1 ) * scale;
        vecNewSamInps[(newSampleOffset+sampleID)*nInputs_+inputID] = 
                   dtemp * vecRanges[inputID] + vecLBs_[inputID];
      }
    }
    newSampleOffset += newNSubSamples; 
    sampleOffset += nSubSamples; 
  }

  //**/ ----------------------------------------------------------------
  //**/ update internal variables
  //**/ ----------------------------------------------------------------
  nSamples_ = newNSamples;
  nSymbols_ = newNSymbols;
  vecSamInps_ = vecNewSamInps;
  vecSamOuts_ = vecNewSamOuts;
  vecSamStas_ = vecNewSamStas;

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 4)
  {
    printf("OALHSampling refine: nSamples = %d\n", nSamples_);
    printf("OALHSampling refine: nInputs  = %d\n", nInputs_);
    printf("OALHSampling refine: nOutputs = %d\n", nOutputs_);
    if (randomize_ != 0)
         printf("OALHSampling refine: randomize on\n");
    else printf("OALHSampling refine: randomize off\n");
    for (inputID = 0; inputID < nInputs_; inputID++)
      printf("    OALHSampling input %3d = [%e %e]\n", inputID+1,
             vecLBs_[inputID], vecUBs_[inputID]);
    if (vecInpNumSettings_.length() > 0 || vecSymTable_.length() > 0)
      printf("OALHSampling refine: diable input settings, symbol table.\n");
  }
#endif
  return 0;
}

// ************************************************************************
// set input settings
// ------------------------------------------------------------------------
int OALHSampling::setInputParams(int nInputs, int *counts,
                                 double **settings, int *symtable)
{
  int inputID, inputCnt, sID;

  if (nInputs_ != 0 && nInputs != nInputs_)
  {
    printf("OALHSampling setInputParams ERROR: nInputs mismatch.\n");
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
    inputCnt = 0;
    for (inputID = 0; inputID < nInputs_; inputID++)
    {
      if (counts[inputID] != 0 && counts[inputID] != nSymbols_*nSymbols_)
      {
        printf("OALHSampling setInputParams ERROR: counts mismatch %d %d.\n",
               counts[inputID], nSymbols_*nSymbols_);
        exit(1);
      }
      else if (counts[inputID] == nSymbols_*nSymbols_) inputCnt++;
    }
    if (inputCnt > 0)
    {
      vecInpSettings_ = new psVector[nInputs_];
      vecInpNumSettings_.setLength(nInputs_);
      for (inputID = 0; inputID < nInputs_; inputID++)
      {
        if (counts[inputID] == nSymbols_*nSymbols_)
        {
          vecInpSettings_[inputID].setLength(counts[inputID]);
          vecInpNumSettings_[inputID] = counts[inputID];
          for (sID = 0; sID < counts[inputID]; sID++)
            vecInpSettings_[inputID][sID] = settings[inputID][sID];
        }
        else
        {
          vecInpNumSettings_[inputID] = 0;
        }
      }
    }
  }
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
OALHSampling& OALHSampling::operator=(const OALHSampling &)
{
   printf("OALHSampling operator= ERROR: operation not allowed.\n");
   exit(1);
   return (*this);
}

