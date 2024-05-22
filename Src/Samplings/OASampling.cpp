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
// Functions for the orthogonal array sampling class
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "OASampling.h"
#include "psVector.h"

#define PABS(x) ((x) > 0 ? (x) : -(x))

// ************************************************************************
// external functions
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
OASampling::OASampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_OA;
  trueRandom_ = 0;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
OASampling::~OASampling()
{
}

// ************************************************************************
// initialize sampling data
// ------------------------------------------------------------------------
int OASampling::initialize(int initLevel)
{
#ifndef HAVE_BOSE
  printf("OASampling ERROR: Bose library not installed.\n");
  exit(1);
#else
  int    ii, jj, kk, ll, ss, ss2, index, status, strength, maxMinDist;
  int    dist, dist2, ntimes=1, nReps, repID, offset, nSamples1, nSamples2;
  int    minDist;
  double scale, ddata;
  psVector  vecRanges;
  psIVector vecIT;
  psIMatrix matPerm, matStore, matPerturb;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  if (nInputs_ == 0)
  {
    printf("OASampling::initialize ERROR - input not set up.\n");
    exit(1);
  }
  if (nSamples_ == 0)
  {
    printf("OASampling::initialize ERROR - nSamples = 0.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ error checking, and set or generate parameters
  //**/ ----------------------------------------------------------------
  trueRandom_ = 0;
  if (randomize_ & 2) trueRandom_ = 1;
  if (randomize_ & 1) randomize_ = 1;
  nReps = nReplications_;
  if (nSamples_ / nReps * nReps != nSamples_)
  {
    printf("OASampling : nSamples must be multiples of replications.\n");
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
    printf("OASampling: initialize: nSamples = %d\n", nSamples_);
    printf("OASampling: initialize: nInputs  = %d\n", nInputs_);
    printf("OASampling: initialize: nOutputs = %d\n", nOutputs_);
    if (randomize_ != 0)
         printf("OASampling: initialize: randomize on\n");
    else printf("OASampling: initialize: randomize off\n");
    if (trueRandom_ != 0)
         printf("OASampling: initialize: more randomize on\n");
    else printf("OASampling: initialize: more randomize off\n");
    for (jj = 0; jj < nInputs_; jj++)
      printf("    OASampling input %3d = [%e %e]\n", jj+1,
             vecLBs_[jj], vecUBs_[jj]);
    if (vecInpNumSettings_.length() > 0)
    {
      for (jj = 0; jj < nInputs_; jj++)
      {
        if (vecInpNumSettings_[jj] != nSymbols_)
        {
          printf("OASampling ERROR: inputSetting not compabible.\n");
          exit(1);
        }
      }
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ generate permutation matrix
  //**/ ----------------------------------------------------------------
  nReps = nSamples_ / (nSymbols_ * nSymbols_);
  matPerm.setFormat(2);
  matPerm.setDim(nSamples_+nSamples_/nReps, nInputs_);
  matStore.setFormat(2);
  matStore.setDim(nSamples_/nReps, nInputs_);
  int **permMatrix  = matPerm.getIMatrix2D();
  int **storeMatrix = matStore.getIMatrix2D();
  int **itempMatrix=NULL;
  vecIT.setLength(nSymbols_);

  strength = 2;
  offset = 0;
  maxMinDist = 0;
  for (repID = 0; repID < nReps; repID++)
  {
    if (printLevel_ > 4)
      printf("OASampling: creating the %d-th (of %d) replication.\n",
             repID+1,nReps);

    //**/ use the Bose tool to generate OA
    status = bose_link(nSamples_/nReps,nInputs_,strength,&itempMatrix);
    if ((status >= 0 && status != (nSamples_/nReps)) || status < 0)
    {
      printf("OASampling ERROR: Bose failure.\n");
      printf("    OA requires that the sample size be the square ");
      printf("of a prime integer.\n");
      printf("    ==> Consult PSUADE developers for help.\n");
      exit(1);
    }
    for (ii = 0; ii < nSamples_/nReps; ii++) 
    {
      for (jj = 0; jj < nInputs_; jj++) 
        permMatrix[nSamples_+ii][jj] = itempMatrix[ii][jj];
      free(itempMatrix[ii]);
    }
    free(itempMatrix);
     
    for (ll = 0; ll < ntimes; ll++)
    {
      //**/ randomize the permutation
      for (jj = 0; jj < nInputs_; jj++) 
      {
        generateRandomIvector(nSymbols_, vecIT.getIVector());
        for (ii = 0; ii < nSamples_/nReps; ii++) 
          permMatrix[offset+ii][jj] = 
                        vecIT[permMatrix[nSamples_+ii][jj]];
      }
      if (ntimes > 1)
      {
        minDist = 2 * nSymbols_ * nInputs_;
        for (ss = 0; ss < nSamples_/nReps; ss++)
        {
          for (ss2 = ss+1; ss2 < nSamples_/nReps; ss2++)
          {
            dist = 0;
            for (ii = 0; ii < nInputs_; ii++)
            {
              dist2 = permMatrix[offset+ss][ii]-permMatrix[offset+ss2][ii];
              dist += PABS(dist2);
            }
            if (dist > 0 && dist < minDist) minDist = dist;
          }
        }
      }
      else minDist = maxMinDist + 1;
      if (minDist > maxMinDist)
      {
        for (ss = 0; ss < nSamples_/nReps; ss++)
          for (ii = 0; ii < nInputs_; ii++)
            storeMatrix[ss][ii] = permMatrix[offset+ss][ii];
        maxMinDist = minDist;
      }
    }
    for (ss = 0; ss < nSamples_/nReps; ss++)
      for (ii = 0; ii < nInputs_; ii++)
        permMatrix[offset+ss][ii] = storeMatrix[ss][ii];

    //**/ check for strength of 2
    if (nSamples_/nReps < 1000 && printLevel_ > 5)
    {
      OA_strength(nSymbols_,nSamples_/nReps, nInputs_, 
                  &(permMatrix[offset]), &strength, 0);
      if (strength != 2)
      {
        printf("OASampling ERROR: fail strength 2 test.\n");
        printf("   ==> Please consult PSUADE developers.\n");
        exit(1);
      }
    }
    offset += nSamples_/nReps;
  }

#if 0
  for (ii = 0; ii < nSamples_; ii++) 
  {
    printf("OA sample %3d = ", ii);
    for (jj = 0; jj < nInputs_; jj++) printf("%d ",permMatrix[ii][jj]);
    printf("\n");
  }
#endif

  // ----------------------------------------------------------------
  // generate sample data
  // ----------------------------------------------------------------
  vecRanges.setLength(nInputs_);
  for (jj = 0; jj < nInputs_; jj++) 
    vecRanges[jj] = vecUBs_[jj] - vecLBs_[jj];

  //**/ case 1 : total random perturbation (even across replicates)
  if (trueRandom_ != 0)
  {
    scale = 1.0 / (double) nSymbols_;
    for (ii = 0; ii < nSamples_; ii++)
    {
      for (jj = 0; jj < nInputs_; jj++)
      {
        index = permMatrix[ii][jj];
        ddata = (PSUADE_drand() + index) * scale;
        vecSamInps_[ii*nInputs_+jj] = ddata * vecRanges[jj] + vecLBs_[jj];
      }
    }
  }

  //**/ case 2 : semi-random perturbation (same across replicates)
  else if (randomize_ != 0)
  {
    matPerturb.setFormat(2);
    matPerturb.setDim(nInputs_, nSymbols_);
    for (jj = 0; jj < nInputs_; jj++)
    {
      for (kk = 0; kk < nSymbols_; kk++)
        matPerturb.setEntry(jj,kk, PSUADE_drand());
    }
    scale = 1.0 / (double) nSymbols_;
    for (ii = 0; ii < nSamples_; ii++)
    {
      for (jj = 0; jj < nInputs_; jj++)
      {
        index = permMatrix[ii][jj];
        ddata = (matPerturb.getEntry(jj,index) + index) * scale;
        vecSamInps_[ii*nInputs_+jj] = ddata * vecRanges[jj]+vecLBs_[jj];
      }
    }
  }

  //**/ case 3 : no perturbation
  else
  {
    scale = 1.0 / (double) (nSymbols_ - 1);
    for (ii = 0; ii < nSamples_; ii++)
    {
      for (jj = 0; jj < nInputs_; jj++)
      {
        index = permMatrix[ii][jj];
        if (vecInpNumSettings_.length() > 0 && 
            index < vecInpNumSettings_[jj])
        {
          vecSamInps_[ii*nInputs_+jj] = vecInpSettings_[jj][index];
        }
        else
        {
          ddata = scale * index;
          vecSamInps_[ii*nInputs_+jj] = ddata*vecRanges[jj]+vecLBs_[jj];
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
int OASampling::refine(int refineRatio, int randomize, double threshold,
                       int nSamples, double *sampleErrors)
{
#ifdef HAVE_BOSE
  int    ii2, nReps, symMult, nFactors, ncount, ind1, samMult, nLevels;
  int    strength, status, newNSamples, newNSymbols, ind2, currOffset;
  int    ii, kk, jj, repID, outputID, sampleOffset, newSampleOffset;
  int    **itempMatrix;
  double scale, ddata;
  psIVector vecNewSamStas, vecI1, vecI2, vecI3, vecSymbols;
  psVector  vecRanges, vecNewSamInps, vecNewSamOuts;
  psMatrix  matPerturb, matBounds;
  psIMatrix matOASamples;

  //**/ ----------------------------------------------------------------
  //**/ unused parameters
  //**/ ----------------------------------------------------------------
  (void) refineRatio;
  (void) randomize;
  (void) threshold;
  (void) nSamples;
  (void) sampleErrors;

  //**/ ----------------------------------------------------------------
  //**/ check 1 : prime power
  //**/ ----------------------------------------------------------------
  nReps = nSamples_ / (nSymbols_ * nSymbols_);
  int *factors  = factorize(nSamples_/nReps);
  nFactors = factors[0];
  ncount = 0;
  for (ind1 = 2; ind1 < nFactors; ind1++)
    if (factors[ind1] != factors[ind1-1]) ncount++;
  if (ncount > 1)
  {
    printf("OASampling refine ERROR: %d not prime power.\n",
           nSamples_/nReps);
    exit(1);
  }
  if (nInputs_ > 2) nLevels = nInputs_ - 1;
  else              nLevels = 2;
  printf("OASampling refine INFO: nLevels set to %d.\n",nInputs_-1);
  symMult = nLevels;
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
  for (ii = 0; ii < samMult; ii++)
    if (itempMatrix[ii] != NULL) free(itempMatrix[ii]);
  free(itempMatrix);

  //**/ ----------------------------------------------------------------
  //**/ initialization
  //**/ ----------------------------------------------------------------
  vecRanges.setLength(nInputs_);
  for (jj = 0; jj < nInputs_; jj++)
    vecRanges[jj] = vecUBs_[jj] - vecLBs_[jj];
  randomize_ = 1;

  //**/ ----------------------------------------------------------------
  //**/ steps in creating new sample points
  //**/ ----------------------------------------------------------------

  //**/ create new sample matrices and initialize state vector
  newNSymbols = nSymbols_ * symMult;
  newNSamples = newNSymbols * newNSymbols * nReps;

  vecNewSamInps.setLength(newNSamples*nInputs_);
  vecNewSamOuts.setLength(newNSamples*nOutputs_);
  vecNewSamStas.setLength(newNSamples);
  for (ii = 0; ii < newNSamples; ii++)
  {
    for (outputID = 0; outputID < nOutputs_; outputID++)
      vecNewSamOuts[ii*nOutputs_+outputID] = PSUADE_UNDEFINED;
  }

  //**/ generate bounds array for binning
  matBounds.setFormat(2);
  matBounds.setDim(nInputs_, newNSymbols+1);
  double **bounds = matBounds.getMatrix2D();
  for (jj = 0; jj < nInputs_; jj++)
  {
    for (kk = 0; kk <= newNSymbols; kk++)
      bounds[jj][kk] = vecRanges[jj] / newNSymbols * kk + vecLBs_[jj];
  }

  //**/ construct a perturbation matrix
  matPerturb.setFormat(2);
  matPerturb.setDim(nInputs_, newNSymbols);
  for (jj = 0; jj < nInputs_; jj++)
  {
    for (kk = 0; kk < newNSymbols; kk++)
      matPerturb.setEntry(jj,kk, PSUADE_drand());
  }

  //**/ fix part of the perturbation matrix from previous points
  for (ii = 0; ii < nSamples_/nReps; ii++)
  {
    for (jj = 0; jj < nInputs_; jj++)
    {
      ddata = vecSamInps_[ii*nInputs_+jj];
      if (ddata == bounds[jj][newNSymbols]) kk = newNSymbols;
      else
      {
        for (kk = 1; kk < (newNSymbols+1); kk++)
          if (ddata < bounds[jj][kk]) break;
      }
      kk--;
      if (kk >= newNSymbols)
      {
        printf("OASampling refine ERROR (3): %d %d %e\n",kk,
               newNSymbols,ddata);
        exit(1);
      }
      ddata -= vecLBs_[jj];
      ddata  = ddata / vecRanges[jj] * (double) newNSymbols;
      ddata -= (double) kk;
      matPerturb.setEntry(jj, kk, ddata);
    }
  }

  //**/ allocate auxiliary arrays for binning
  vecSymbols.setLength(newNSymbols);
  vecI1.setLength(newNSymbols);
  vecI2.setLength(newNSymbols);
  vecI3.setLength(newNSymbols);
  matOASamples.setFormat(PS_MAT2D);
  matOASamples.setDim(newNSamples, nInputs_);;

  //**/ for each replication, do the following
  sampleOffset = 0;
  newSampleOffset = 0;
  for (repID = 0; repID < nReps; repID++)
  {
    //**/ fill in old sample data
    for (ii = 0; ii < nSamples_/nReps; ii++)
    {
      for (jj = 0; jj < nInputs_; jj++)
        vecNewSamInps[(newSampleOffset+ii)*nInputs_+jj] =
                        vecSamInps_[(sampleOffset+ii)*nInputs_+jj];
      for (outputID = 0; outputID < nOutputs_; outputID++)
        vecNewSamOuts[(newSampleOffset+ii)*nOutputs_+outputID] =
                vecSamOuts_[(sampleOffset+ii)*nOutputs_+outputID];
      vecNewSamStas[newSampleOffset+ii] = vecSamStas_[sampleOffset+ii];
    }

    //**/ recover the OA pattern
    for (ii = 0; ii < nSamples_/nReps; ii++)
    {
      for (jj = 0; jj < nInputs_; jj++)
      {
        ddata = vecSamInps_[(sampleOffset+ii)*nInputs_+jj];
        if (ddata == bounds[jj][newNSymbols]) kk = newNSymbols;
        else
        {
          for (kk = 1; kk < (newNSymbols+1); kk++)
            if (ddata < bounds[jj][kk]) break;
        }
        matOASamples.setEntry(newSampleOffset+ii,jj,kk - 1);
      }
    }
    currOffset = newSampleOffset;
    newSampleOffset += (nSamples_ / nReps);

    //**/ create additional patterns
    for (ii = 0; ii < nSamples_/nReps; ii++)
    {
      status = bose_link(samMult, nInputs_, strength, &itempMatrix);

      //**/ permute to match the first pattern
      for (jj = 0; jj < nInputs_; jj++)
      {
        ind1 = itempMatrix[0][jj];
        kk = matOASamples.getEntry(currOffset+ii, jj);
        ind2 = kk % symMult;
        if (ind1 != ind2)
        {
          for (ii2 = 1; ii2 < samMult; ii2++)
          {
            ind1 = itempMatrix[ii2][jj];
            if (ind1 == ind2) break;
          }
        }
        if (ind1 == ind2)
        {
          ind1 = itempMatrix[0][jj];
          for (ii2 = 0; ii2 < samMult; ii2++)
          {
            if (itempMatrix[ii2][jj] == ind1)
              itempMatrix[ii2][jj] = ind2;
            else if (itempMatrix[ii2][jj] == ind2)
              itempMatrix[ii2][jj] = ind1;
           }
        }
      }

      //**/ record the new set of pattern
      for (ii2 = 1; ii2 < samMult; ii2++)
      {
        for (jj = 0; jj < nInputs_; jj++)
        {
          ind1 = matOASamples.getEntry(currOffset+ii,jj) / symMult; 
          kk = itempMatrix[ii2][jj] + ind1 * symMult;
          matOASamples.setEntry(newSampleOffset+ii2-1,jj,kk); 
        }
      }
      newSampleOffset += (samMult - 1);

      //**/ clean itempMatrix
      for (ii2 = 0; ii2 < samMult; ii2++)
        if (itempMatrix[ii2] != NULL) free(itempMatrix[ii2]);
      free(itempMatrix);
    }

    //**/ permute the entire new OA replication
    for (jj = 0; jj < nInputs_; jj++)
    {
      for (kk = 0; kk < newNSymbols; kk++) vecSymbols[kk] = kk;
      for (ii = 0; ii < nSamples_/nReps; ii++)
      {
        ind1 = matOASamples.getEntry(currOffset+ii, jj);
        vecSymbols[ind1] = -1;
      }
      ncount = 0;
      for (kk = 0; kk < newNSymbols; kk++)
      if (vecSymbols[kk] >= 0) ncount++;
      if (ncount > 1)
      {
        ncount = 0;
        for (kk = 0; kk < newNSymbols; kk++)
          if (vecSymbols[kk] >= 0) vecI3[ncount++] = vecSymbols[kk];
        generateRandomIvector(ncount, vecI1.getIVector());
        for (kk = 0; kk < ncount; kk++)
          vecI2[kk] = vecI3[vecI1[kk]];
        ncount = 0;
        for (kk = 0; kk < newNSymbols; kk++)
        {
          if (vecSymbols[kk] >= 0) vecSymbols[kk] = vecI2[ncount++];
          else                     vecSymbols[kk] = kk;
        }
        for (ii = 0; ii < newNSamples/nReps; ii++)
        {
          ind1 = matOASamples.getEntry(currOffset+ii, jj);
          matOASamples.setEntry(currOffset+ii,jj,vecSymbols[ind1]);
        }
      }
    }

    //**/ check strength
    if (newNSamples/nReps < 1000)
    {
      int **OASamples = matOASamples.getIMatrix2D();
      OA_strength(newNSymbols, newNSamples/nReps, nInputs_,
                  &(OASamples[currOffset]), &strength, 0);
      if (strength != 2)
      {
        printf("OASampling::refine ERROR : strength %d != 2\n",
               strength);
        exit(1);
      }
    }
#if 0
    for (ii = 0; ii < newNSamples/nReps; ii++)
    {
      printf("sample %3d (%3d) : ",ii, currOffset);
      for (jj = 0; jj < nInputs_; jj++)
         printf(" %3d ", matOASamples.getEntry(currOffset+ii,jj));
      printf("\n");
    }
#endif

    //**/ create new sample points
    currOffset += nSamples_ / nReps; 
    for (ii = currOffset; ii < newSampleOffset; ii++)
    {
      scale = 1.0 / ((double) newNSymbols);
      for (jj = 0; jj < nInputs_; jj++)
      {
        ind1 = matOASamples.getEntry(ii,jj);
        ddata = (matPerturb.getEntry(jj,ind1) + ind1) * scale;
        vecNewSamInps[ii*nInputs_+jj] = ddata * vecRanges[jj] + 
                                        vecLBs_[jj];
      }
    }
    sampleOffset += (nSymbols_ * nSymbols_);
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
    printf("OASampling refine: nSamples = %d\n", nSamples_);
    printf("OASampling refine: nInputs  = %d\n", nInputs_);
    printf("OASampling refine: nOutputs = %d\n", nOutputs_);
    if (randomize_ != 0)
         printf("OASampling refine: randomize on\n");
    else printf("OASampling refine: randomize off\n");
    if (trueRandom_ != 0)
         printf("OASampling refine: more randomize on\n");
    else printf("OASampling refine: more randomize off\n");
    for (jj = 0; jj < nInputs_; jj++)
      printf("    OASampling input %3d = [%e %e]\n", jj+1,
             vecLBs_[jj], vecUBs_[jj]);
  }
#endif
  return 0;
}

// ************************************************************************
// set input settings
// ------------------------------------------------------------------------
int OASampling::setInputParams(int nInputs, int *counts, 
                               double **settings, int *symtable)
{
  int jj, inputCnt, sID;

  if (nInputs_ != 0 && nInputs != nInputs_)
  {
    printf("OASampling setInputParams ERROR: nInputs mismatch.\n");
    exit(1);
  }
  nInputs_ = nInputs;
  if (symtable != NULL)
  {
    vecSymTable_.setLength(nInputs);
    for (jj = 0; jj < nInputs_; jj++) vecSymTable_[jj] = symtable[jj];
  }
  if (counts != NULL)
  {
    inputCnt = 0;
    for (jj = 0; jj < nInputs_; jj++)
    {
      if (counts[jj] != 0 && counts[jj] != nSymbols_)
      {
         printf("OASampling setInputParams ERROR: counts mismatch.\n");
         exit(1);
      }
      else if (counts[jj] == nSymbols_) inputCnt++;
    }
    if (inputCnt > 0)
    {
      vecInpNumSettings_.setLength(nInputs_);
      vecInpSettings_ = new psVector[nInputs_];
      for (jj = 0; jj < nInputs_; jj++)
      {
        if (counts[jj] == nSymbols_)
        {
          vecInpSettings_[jj].setLength(counts[jj]);
          vecInpNumSettings_[jj] = counts[jj];
          for (sID = 0; sID < counts[jj]; sID++)
            vecInpSettings_[jj][sID] = settings[jj][sID];
        }
        else
        {
          vecInpSettings_[jj].clean();
          vecInpNumSettings_[jj] = 0;
        }
      }
    }
  }
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
OASampling& OASampling::operator=(const OASampling &)
{
  printf("OASampling operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

