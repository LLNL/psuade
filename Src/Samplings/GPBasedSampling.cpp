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
// Functions for the GPBasedSampling class 
// AUTHOR : CHARLES TONG
// DATE   : 2019
// ************************************************************************

// ------------------------------------------------------------------------
// system and local includes
// ------------------------------------------------------------------------
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sysdef.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "GPBasedSampling.h"
#include "GP3.h"

// ************************************************************************
// Constructor
// ------------------------------------------------------------------------
GPBasedSampling::GPBasedSampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_GP;
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printf("                  GPBasedSampling:\n");
    printDashes(PL_INFO, 0);
    printf("This module performs batch sequential experimental design.\n");
    printf("To track progress and get matlab plots, set print ");
    printf("level > 1 before\n");
    printf("calling this.\n");
    printEquals(PL_INFO, 0);
  }
  MMVorMAV_ = 0; /* default: mmv */
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
GPBasedSampling::~GPBasedSampling()
{
}

// ************************************************************************
// initialize the sampling data
//**/ As of March 2022, this function is being called from odoe_mmv/mav
//**/ (since this object can perform several types of optimization without
//**/ other optimizers), and also from FunctionInterface psLocalFunction 
//**/ (which is called by SCE) to evaluate candidate set using GP.
// ------------------------------------------------------------------------
int GPBasedSampling::initialize(int initLevel)
{
  int    ii, jj, kk, status, nCand, nDesigns, *SS, option1, option2;
  double ddata, *SY, quality[3];
  char   pString[1000], fname[10000];
  FILE   *fp;
  GP3    *rsPtr=NULL;
  psMatrix candMatrix, finalDesign;
  Sampling *sampler;

  //**/ ----------------------------------------------------------------
  //**/ see if a candidate set has been passed in, read it into 
  //**/ candMatrix
  //**/ ----------------------------------------------------------------
  if (psConfig_.MatCommonUse_.nrows() > 0)
  {
    printf("GPBasedSampling INFO: candidate set available from ");
    printf("external source.\n");
    nCand = psConfig_.MatCommonUse_.nrows();
    kk = psConfig_.MatCommonUse_.ncols();
    if (nInputs_ != 0 && kk != nInputs_)
    {
      printf("GPBasedSampling ERROR: candidate set has been ");
      printf("passed in but its\n");
      printf("       nInputs is not equal to nInputs for the ");
      printf("training sample.\n");
      exit(1);
    }
    nInputs_ = kk;
    candMatrix = psConfig_.MatCommonUse_;
  }

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  if (nInputs_ == 0)
  {
    printf("GPBasedSampling::initialize ERROR - input bounds ");
    printf("has not been set up.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ build GP response surface, if sample data is available
  //**/ ----------------------------------------------------------------
  printf("GPBasedSampling requires hyperparameters for building ");
  printf("the covariance\n");
  printf("  matrix needed for estimating prediction variance. GP ");
  printf("hyperparameters\n");
  printf("  can either be\n");
  printf("(1) computed by PSUADE when a sample has been ");
  printf("provided (LOADED), or\n");
  printf("(2) supplied directly by users\n");
  int outputValid=1, secondCall=1; /* assume this has been called before */
  if (vecSamOuts_.length() == 0) outputValid = 0;
  for (ii = 0; ii < vecSamOuts_.length(); ii++)
    if (vecSamOuts_[ii] >= 1e34) outputValid = 0;
  if (vecSamInps_.length() > 0 && outputValid == 1)
  {
    if (nSamples_ < 10)
    {
      printf("GPBasedSampling ERROR: A training sample has ");
      printf("been provided but\n");
      printf("                nSamples must be >= 10 since GP ");
      printf("surrogate may not be\n");
      printf("                good for small samples.\n");
      return -1;
    }
    printf("GPBasedSampling: A training sample has been provided ");
    printf("==> PSUADE will\n");
    printf("                 compute GP hyperparameters.\n");
    rsPtr = new GP3(nInputs_, nSamples_);
    rsPtr->setBounds(vecLBs_.getDVector(), vecUBs_.getDVector());
    rsPtr->setOutputLevel(printLevel_);
    status = rsPtr->initialize(vecSamInps_.getDVector(),
                               vecSamOuts_.getDVector());
    rsPtr->getHyperparameters(VecGPParams_);

    //**/ adjust based on the GP3 formula (times 0.5 to adjust for 0.5)
    for (ii = 0; ii < nInputs_; ii++)
      VecGPParams_[ii] = 0.5 / exp(2.0 * VecGPParams_[ii]);
    //**/ this value (multiplier to exponential) is not important
    VecGPParams_[nInputs_] = 1.0;
    //**/ constant added to all entries
    VecGPParams_[nInputs_+1] = exp(VecGPParams_[nInputs_+1]);
    VecGPParams_[nInputs_+1] *= sqrt(nSamples_);
    //**/ estimation of Ymean (not important)
    VecGPParams_[nInputs_+2] = 0;
    //**/ fudge added to diagonal
    VecGPParams_[nInputs_+3] = exp(VecGPParams_[nInputs_+3]); 
    delete rsPtr;
    for (ii = 0; ii < nInputs_; ii++)
      printf("GPSampling INFO: hyperparameter %d = %e\n",ii+1,
             VecGPParams_[ii]);
    printf("GPSampling INFO: nugget = %e\n",
             VecGPParams_[nInputs_+3]);
  }
  //**/ if GP hyperparameters have not been set before, ask for them
  //**/ That is, if this function is called for the first time
  //**/ (Sometimes it may be called again as in odoe_mmv so in that
  //**/ case questions will not be asked again)
  else if (VecGPParams_.length() == 0)
  {
    secondCall = 0; /* if this is the first time initialize is called */
    printf("GPBasedSampling has not obtained a training sample, ");
    printf("please enter GP\n");
    printf("  hyperparameters (scale_i) by hand in the following ");
    printf("correlation\n");
    printf("  functional form: exp(-sum_i scale_i * dist_i ^ 2).\n");
    VecGPParams_.setLength(nInputs_+4);
    for (ii = 0; ii < nInputs_; ii++)
    {
      snprintf(pString,100,"Enter hyperparameter for input %d (e.g. 0.1): ",
               ii+1);
      VecGPParams_[ii] = 0.0;
      while (VecGPParams_[ii] <= 0.0)
      {
        VecGPParams_[ii] = getDouble(pString);
        if (VecGPParams_[ii] <= 0) 
          printf("ERROR: value needs to be positive.\n");
      }
    }
    //**/ this value (multiplier to exponential) is not important
    VecGPParams_[nInputs_] = 1.0;
    //**/ constant added to all entries
    VecGPParams_[nInputs_+1] = 0;
    //**/ estimation of Ymean (not important)
    VecGPParams_[nInputs_+2] = 0;
    //**/ fudge added to diagonal
    VecGPParams_[nInputs_+3] = 1e-12;
    //**/ hardwired
    //snprintf(pString,100,
    //        "Enter nugget (added to Cdiag) value (> 0, e.g. 1e-6) : ");
    //while (VecGPParams_[nInputs_+3] < 0)
    //{
    //  VecGPParams_[nInputs_+3] = getDouble(pString);
    //  if (VecGPParams_[nInputs_+3] < 0)
    //    printf("ERROR: Nugget should be >= 0.\n");
    //}
    for (ii = 0; ii < nInputs_; ii++)
      printf("GPSampling INFO: hyperparameter %d = %e\n",ii+1,
             VecGPParams_[ii]);
  }

  //**/ ----------------------------------------------------------------
  //**/ if initLevel != 0, it is the odoe_mav/mmv command that is 
  //**/ calling psLocalFunction which in turn calls this function.
  //**/ ----------------------------------------------------------------
  if (initLevel != 0) return 0;

  //**/ ----------------------------------------------------------------
  //**/ ask user for a candidate set
  //**/ Note: if psConfig_MatCommonUse_ is not empty, this means that
  //**/       the candidate set has been passed in, so no need to ask
  //**/ ----------------------------------------------------------------
  if (candMatrix.nrows() == 0)
  {
    printf("NEXT, a candidate design set (a set of sample points ");
    printf("from which the\n");
    printf("final design point will be selected) needs to be ");
    printf("provided by users or\n");
    printf("created by PSUADE.\n");
    printf("Please select whether \n");
    printf("1. PSUADE is to create a candidate design set, or\n");
    printf("2. You will provide a candidate design set\n");
    snprintf(pString,100,"Enter you choice : (1 or 2) ");
    kk = getInt(1,2,pString);
    if (kk == 1)
    {
      printf("PSUADE is to create a candidate design set.\n");
      printf("Select candidate design method:\n");
      printf("1. Full Factorial\n");
      printf("2. LPtau (LHS if nInputs > 51)\n");
      snprintf(pString,100,"Enter candidate design method (1 or 2) : ");
      kk = getInt(1,2,pString);
      if (kk == 1) sampler = SamplingCreateFromID(PSUADE_SAMP_FACT);
      else
      {
        if (nInputs_ > 51) 
             sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
        else sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
      }
      snprintf(pString,100, 
              "Enter candidate design sample size (>= 25): ");
      nCand = getInt(25,1000000,pString);
      sampler->setInputBounds(nInputs_, vecLBs_.getDVector(), 
                              vecUBs_.getDVector());
      sampler->setOutputParams(1);
      sampler->setSamplingParams(nCand, 1, 0);
      sampler->initialize(0);
      psVector  vecSX, vecSY;
      psIVector vecSS;
      vecSX.setLength(nCand*nInputs_);
      vecSY.setLength(nCand);
      vecSS.setLength(nCand);
      sampler->getSamples(nCand, nInputs_, 1, vecSX.getDVector(), 
                          vecSY.getDVector(), vecSS.getIVector());
      delete sampler;
      candMatrix.setDim(nCand, nInputs_);
      for (ii = 0; ii < nCand; ii++)
        for (jj = 0; jj < nInputs_; jj++)
          candMatrix.setEntry(ii,jj,vecSX[ii*nInputs_+jj]);
    }
    else
    {
      printf("The candidate design set should be stored in a text\n");
      printf("file having the following format: \n");
      printf("<number of points> <number of inputs>\n");
      printf("1 input values \n");
      printf("2 input values \n");
      printf(".... \n");
      snprintf(pString,100,
              "Enter the file name of your candidate design set: ");
      getString(pString, fname);
      kk = strlen(fname);
      fname[kk-1] = '\0';
      status = readIReadDataFile(fname, candMatrix);
      if (status != 0)
      {
        printf("GPBasedSampling::initialize: ERROR in reading data\n");
        exit(1);
      }
    }
  }
  printf("NOTE: The evaluation sample is assumed to be the ");
  printf("same as the candidate\n");
  printf("      sample.\n");

  //**/ the number of selected point may be passed in externally
  nCand = candMatrix.nrows();
  if (psConfig_.intCommonUse_ != 0) nDesigns = psConfig_.intCommonUse_;
  else
  {
    nDesigns = nCand - 1;
    snprintf(pString,100,
      "How many design points to be selected from the candidate set? (1-%d) ", 
      nDesigns);
    nDesigns = getInt(1, nDesigns, pString);
  }
  
  //**/ ----------------------------------------------------------------
  //**/ select designs
  //**/ ----------------------------------------------------------------
  secondCall = 1; /* force call to exhaustive search */
  if (secondCall == 0)
  {
    printEquals(PL_INFO, 0);
    printf("** The procedure goes like this:\n");
    printf("I.  PSUADE initially selects %d points from the candidate set.\n",
           nDesigns);
    printf("II. PSUADE uses an insert-and-delete algorithm to ");
    printf("finetine the set.\n");
    printEquals(PL_INFO, 0);
    printf("For Step (I), you can choose between 5 options:\n");
    printf("1. Pick the first %d sample points from the candidate set\n",
           nDesigns);
    printf("2. Randomly select %d sample points from the candidate set\n",
           nDesigns);
    printf("3. Use one-at-a-time sequential sampling to select ");
    printf("sample points from\n");
    printf("   the candidate set based on GP prediction uncertainties\n");
    printf("4. Use pairwise (two-at-a-time) sequential sampling ");
    printf("to select sample\n");
    printf("   points from the candidate set\n");
    printf("5. Use brute-force exhaustive search (may be time-consuming)\n");
    printf("   (Step II is not needed for option 5).\n");
    snprintf(pString,100,"Please make your selection (1, 2, 3, 4, or 5) : ");
    option1 = getInt(1, 5, pString);
  }
  else 
  {
    printf("INFO: Performing exhaustive search\n");
    option1 = 5;
  }
  if (nDesigns == 1 && option1 != 5)
  {
    printf("Since the desired number of design = 1, it makes sense just\n");
    printf("to find the optimal design (not very time-consuming)\n"); 
    option1 = 5;
  }
  if (option1 != 5)
  {
    printf("For Step (II), you can choose between 2 options:\n");
    printf("1. Use add-delete algorithm\n");
    printf("2. Use swap algorithm (more expensive, better performance)\n");
    snprintf(pString,100,"Please make your selection (1 or 2) : ");
    option2 = getInt(1, 2, pString);
  }
  else option2 = 2;

  //**/ initialize the markers for tracking selected test points
  //**/ At the end, markers[ii] contains the location of candMatrix(ii,:)
  //**/ in finalDesign (if it is not -1)
  psVector markers;
  markers.setLength(nCand); 
  for (ii = 0; ii < nCand; ii++) markers[ii] = -1;
  finalDesign.setDim(nDesigns, nInputs_);

  //**/ ---------------------------------------------------
  //**/ Phase 1 
  //**/ ---------------------------------------------------
  printEquals(PL_INFO, 0);
  printf("Phase 1 analysis\n");
  printDashes(PL_INFO, 0);
  if (option1 == 1 || option1 == 2)
  {
    printf("GPBasedSampling: Creating first guess\n");
    for (jj = 0; jj < nDesigns; jj++)
    {
      if (option1 == 2)
      {
        kk = PSUADE_rand() % nCand;
        while (markers[kk] != -1) kk = PSUADE_rand() % nCand;
      }
      else kk = jj;
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = candMatrix.getEntry(kk,ii);
        finalDesign.setEntry(jj,ii, ddata);
      }
      markers[kk] = jj;
    }
  }
  else if (option1 == 3)
  {
    printf("GPBasedSampling: Calling one-at-a-time search\n");
    genInitialDesigns(VecGPParams_,candMatrix,nDesigns,finalDesign,markers);
  }
  else if (option1 == 4)
  {
    printf("GPBasedSampling: Calling two-at-a-time search\n");
    genInitialDesigns2(VecGPParams_,candMatrix,nDesigns,finalDesign,markers);
  }
  else if (option1 == 5)
  {
    printf("GPBasedSampling: Calling exhaustive search\n");
    genDesignsUltimate(VecGPParams_,candMatrix,nDesigns,finalDesign);
  }

  psVector vecSamInps;
  if (nDesigns > 1)
  {
    vecSamInps.setLength(nDesigns*nInputs_);
    for (jj = 0; jj < nDesigns; jj++)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = finalDesign.getEntry(jj,ii);
        vecSamInps[jj*nInputs_+ii] = ddata;
      }
    }
#if 0
    //**/ ---------------------------------------------------
    //**/ checking sample quality
    //**/ disabled in 3/2022 (no need to do it for MMV
    //**/ ---------------------------------------------------
    printDashes(PL_INFO, 0);
    printf("Sample quality after initial design\n");
    SamplingQuality(nDesigns,nInputs_,vecSamInps.getDVector(),
               vecLBs_.getDVector(),vecUBs_.getDVector(),quality);
    printf("Min-min distance (sample-sample ) = %10.4e (large is good)\n",
           quality[0]);
    printf("Avg-avg distance (sample pairs  ) = %10.4e (large is good)\n",
           quality[2]);
    printf("Avg-min distance (corner-samples) = %10.4e (small is good)\n",
           quality[1]);
#endif
  }

  //**/ ---------------------------------------------------
  //**/ Phase 2 (option 5 does not need phase 2)
  //**/ ---------------------------------------------------
  if (option1 != 5) 
  {
    printEquals(PL_INFO, 0);
    printf("Phase 2 analysis\n");
    printDashes(PL_INFO, 0);
    if (option2 == 1)
      genDesigns(VecGPParams_,candMatrix,nDesigns,finalDesign,markers);
    else if (option2 == 2)
      genDesigns2(VecGPParams_,candMatrix,nDesigns,finalDesign,markers);

    //**/ -------------------------------------------------
    //**/ checking sample quality
    //**/ -------------------------------------------------
    vecSamInps.setLength(nDesigns*nInputs_);
    for (jj = 0; jj < nDesigns; jj++)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = finalDesign.getEntry(jj,ii);
        vecSamInps[jj*nInputs_+ii] = ddata;
      }
    }
    printDashes(PL_INFO, 0);
    printf("Sample quality after final design\n");
    SamplingQuality(nDesigns,nInputs_,vecSamInps.getDVector(),
            vecLBs_.getDVector(),vecUBs_.getDVector(), quality, 0);
    printf("Min-min distance (sample-sample ) = %e (large is good)\n",
           quality[0]);
    printf("Avg-avg distance (sample pairs  ) = %e (large is good)\n",
           quality[2]);
    printf("Avg-min distance (corner-samples) = %e (small is good)\n",
           quality[1]);
  }

  printEquals(PL_INFO, 0);
  printf("Final Design : ");
  for (ii = 0; ii < nCand; ii++)
  {
    for (jj = 0; jj < nDesigns; jj++)
    {
      for (kk = 0; kk < nInputs_; kk++)
        if (candMatrix.getEntry(ii,kk) != finalDesign.getEntry(jj,kk))
          break;
      if (kk == nInputs_)
      {
        printf("%d ", ii+1);
        break;
      }
    }
  }
  printf("\n");
  printEquals(PL_INFO, 0);
  if (MMVorMAV_ == 0) fp = fopen("mmv_result", "w");
  else                fp = fopen("mav_result", "w");
  fprintf(fp,"# <numSamples> <numInputs> <numOutputs>\n");
  fprintf(fp,"# <Candidate number> <input values> ...\n");
  fprintf(fp,"%d %d 0\n", nDesigns, nInputs_);
  for (ii = 0; ii < nCand; ii++)
  {
    for (jj = 0; jj < nDesigns; jj++)
    {
      for (kk = 0; kk < nInputs_; kk++)
        if (candMatrix.getEntry(ii,kk) != 
            finalDesign.getEntry(jj,kk))
          break;
      if (kk == nInputs_)
      {
        fprintf(fp,"%d ",ii+1);
        for (kk = 0; kk < nInputs_; kk++)
        {
          ddata = candMatrix.getEntry(ii,kk);
          fprintf(fp,"%12.4e ",ddata);
        }
        fprintf(fp,"\n");
        break;
      }
    }
  }
  fclose(fp);
  if (MMVorMAV_ == 0)
    printf("INFO: best designs are in mmv_result\n");
  else
    printf("INFO: best designs are in mav_result\n");
  printf("NOTE: You can use 'read_std' to read in the final designs.\n");

  //**/ ----------------------------------------------------------------
  //**/ un-scale 
  //**/ ----------------------------------------------------------------
  if (nSamples_ > 0)
  {
    kk = finalDesign.nrows();
    for (jj = 0; jj < kk; jj++)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = finalDesign.getEntry(jj,ii);
        finalDesign.setEntry(jj,ii, ddata);
      }
    }
  }
  return 0;
}

// ************************************************************************
// construct covariance matrix given hyperparameters
//**/ somehow all matrices have to be passed by reference?
// ------------------------------------------------------------------------
void GPBasedSampling::constructCMatrix(psMatrix &sampleMatrix,
                                       psMatrix &CMatrix,psVector params)
{
  int    ii, jj, kk, nrows, ncols;
  double dist, dtmp1, dtmp2;

  nrows = sampleMatrix.nrows();
  ncols = sampleMatrix.ncols();

  CMatrix.setDim(nrows, nrows);
  for (jj = 0; jj < nrows; jj++)
  {
    dist = params[nInputs_] + params[nInputs_+1] + params[nInputs_+3];
    CMatrix.setEntry(jj,jj,dist);
    for (kk = jj+1; kk < nrows; kk++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs_; ii++)
      {
        dtmp1 = sampleMatrix.getEntry(jj,ii);
        dtmp2 = sampleMatrix.getEntry(kk,ii);
        dtmp1 = dtmp1 - dtmp2;
        dist += pow(dtmp1, 2.0) * params[ii];
      }
      dist = params[nInputs_] * exp(-dist);
      if (dist < 1.0e-50) dist = 0;
      CMatrix.setEntry(jj,kk,dist);
      CMatrix.setEntry(kk,jj,dist);
    }
  }
  return;
}

// ************************************************************************
// generate design method I 
//**/ use add-delete method
// ------------------------------------------------------------------------
int GPBasedSampling::genDesigns(psVector &params, psMatrix &candMatrix, 
                                int nDesigns, psMatrix &finalDesign,
                                psVector &markers)
{
  int    ii, jj, kk, mm, nn, status, nCand, minInd, markerSave, avgCnt;
  long   iter;
  double ddata, dist, meas, minVal, maxVal, avgVal;
  psMatrix CMatrix, samMatrix, baseDesign;
  psVector correlations, tmpVec;

  //**/ copy existing design to baseDesign
  nCand = candMatrix.nrows();
  baseDesign.setDim(nDesigns+1, nInputs_);
  for (jj = 0; jj < nDesigns; jj++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      ddata = finalDesign.getEntry(jj,ii);
      baseDesign.setEntry(jj, ii, ddata);
    }
  }
  printf("genDesigns: current final designs\n");
  finalDesign.print();

  //**/ loop a number of times until convergence
  iter = 0;
  while (iter < 100)
  {
    iter++;
    //**/ copy from baseDesign to samMatrix
    //**/ samMatrix has one more row to accommodate addition
    //**/ reason for needing samMatrix: constructCMatrix needs
    //**/ the exact current design set
    samMatrix.setDim(nDesigns+1, nInputs_);
    for (jj = 0; jj < nDesigns; jj++)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = baseDesign.getEntry(jj,ii);
        samMatrix.setEntry(jj, ii, ddata);
      }
    }

    //**/ now add one point at a time from the test set, and, for
    //**/ each addition, compute min(variance).
    //**/ The point that yields min(min(variance)) will be added to
    //**/ the set
    minVal = PSUADE_UNDEFINED;
    minInd = -1;
    for (mm = 0; mm < nCand; mm++)
    {
      //**/ if sample mm is not in the design set, see if adding it
      //**/ to the set will give smallest max(variance)
      if (markers[mm] == -1)
      {
        //**/ append the sample point to samMatrix
        for (ii = 0; ii < nInputs_; ii++)
        {
          ddata = candMatrix.getEntry(mm,ii);
          samMatrix.setEntry(nDesigns, ii, ddata);
        }
        markers[mm] = nDesigns;

        //**/ construct inverse of covariance matrix
        constructCMatrix(samMatrix, CMatrix, params);
        status = CMatrix.computeInverse(CMatrix);
        if (status != 0)
        {
          printf("GPBasedSampling ERROR in covariance inversion\n");
          exit(1);
        }

        //**/ for the rest of the points, search for max variance
        correlations.setLength(nDesigns+1);
        maxVal = - PSUADE_UNDEFINED;
        avgVal = 0;
        avgCnt = 0;
        for (nn = 0; nn < nCand; nn++)
        {
          if (markers[nn] == -1)
          {
            //**/ compute correlation vector
            for (jj = 0; jj < nDesigns+1; jj++)
            {
              dist = 0.0;
              for (ii = 0; ii < nInputs_; ii++)
              {
                ddata = samMatrix.getEntry(jj, ii);
                ddata -= candMatrix.getEntry(nn,ii);
                dist += pow(ddata, 2.0) * params[ii];
              }
              dist = exp(-dist);
              if (dist < 1.0e-50) dist = 0;
              correlations[jj] = dist;
            }
            //**/ compute correlations * inv(C) * correlation
            CMatrix.matvec(correlations, tmpVec, 0);
            meas = 0.0;
            for (jj = 0; jj <= nDesigns; jj++)
              meas += correlations[jj]*tmpVec[jj];
            //**/ negative to mimic variance
            meas = -meas;
            //**/ search for max variance
            if (meas > maxVal) maxVal = meas;
            avgVal += meas;
            avgCnt++;
          }
        }
        avgVal /= (double) avgCnt;

        //**/ if, after adding point mm, max/avg prediction variance for
        //**/ all other points is smallest so far, register mm as a 
        //**/ possible candidate for addition
        if (MMVorMAV_ == 0)
        {
          if (maxVal < minVal)
          {
            minVal = maxVal;
            minInd = mm;
          }
          else if (maxVal == minVal)
          {
            //**/ if same min-max value, toss a coin to decide
            nn = PSUADE_rand() % 2;
            if (nn == 1)
            {
              minVal = maxVal;
              minInd = mm;
            }
          }
          markers[mm] = -1;
        }
        else
        {
          if (avgVal < minVal)
          {
            minVal = avgVal;
            minInd = mm;
          }
          else if (avgVal == minVal)
          {
            //**/ if same min-max value, toss a coin to decide
            nn = PSUADE_rand() % 2;
            if (nn == 1)
            {
              minVal = avgVal;
              minInd = mm;
            }
          }
          markers[mm] = -1;
        }
      }
    }
    //**/ now a sample minInd has been found
    //**/ insert it into the last row of the baseDesign matrix
    for (ii = 0; ii < nInputs_; ii++)
    {
      ddata = candMatrix.getEntry(minInd,ii);
      baseDesign.setEntry(nDesigns, ii, ddata);
    } 
    markers[minInd] = nDesigns;

    //**/ next examine all samples in the baseDesign and see which
    //**/ can be taken out
    minVal = PSUADE_UNDEFINED;
    minInd = -1;
    samMatrix.setDim(nDesigns, nDesigns);
    for (mm = 0; mm <= nDesigns; mm++)
    {
      //**/ copy baseDesign to samMatrix except design mm
      nn = 0;
      for (kk = 0; kk <= nDesigns; kk++)
      {
        if (kk != mm)
        {
          for (ii = 0; ii < nInputs_; ii++)
          {
            ddata = baseDesign.getEntry(kk,ii);
            samMatrix.setEntry(nn, ii, ddata);
          }
          nn++;
        }
      }
      //**/ release sample in baseDesign(mm) for testing
      //**/ (set markers[kk] = -1 where kk is the kk-th row
      //**/ in candMatrix that has bee put into the mm-th
      //**/ row in baseDesign
      for (kk = 0; kk < nCand; kk++)
        if (markers[kk] == mm) break;
      markerSave = kk;
      markers[kk] = -1;

      //**/ construct inverse of covariance matrix
      constructCMatrix(samMatrix, CMatrix, params);
      status = CMatrix.computeInverse(CMatrix);
      if (status != 0)
      {
        printf("GPBasedSampling ERROR in covariance inversion\n");
        exit(1);
      }

      //**/ for the rest of the points, compute max variance
      correlations.setLength(nDesigns);
      maxVal = - PSUADE_UNDEFINED;
      avgVal = 0;
      avgCnt = 0;
      for (nn = 0; nn < nCand; nn++)
      {
        if (markers[nn] == -1)
        {
          //**/ compute correlation vector
          for (jj = 0; jj < nDesigns; jj++)
          {
            dist = 0.0;
            for (ii = 0; ii < nInputs_; ii++)
            {
              ddata = samMatrix.getEntry(jj, ii);
              ddata -= candMatrix.getEntry(nn,ii);
              dist += pow(ddata, 2.0) * params[ii];
            }
            dist = exp(-dist);
            if (dist < 1.0e-50) dist = 0;
            correlations[jj] = dist;
          }
          //**/ compute correlations * inv(C) * correlation
          CMatrix.matvec(correlations, tmpVec, 0);
          meas = 0.0;
          for (jj = 0; jj < nDesigns; jj++)
            meas += correlations[jj]*tmpVec[jj];
          //**/ take negative to reflect prediction variance
          meas = -meas;
          //**/ record the max prediction variance in the space
          if (meas > maxVal) maxVal = meas;
          avgVal += meas;
          avgCnt++;
        }
      }
      avgVal /= (double) avgCnt;
      //**/ design point mm is a candidate for removal if its 
      //**/ removal results in smallest max prediction variance
      //**/ among all current design points
      if (MMVorMAV_ == 0)
      {
        if (maxVal < minVal)
        {
          minVal = maxVal;
          minInd = mm;
        }
      }
      else
      {
        if (avgVal < minVal)
        {
          minVal = avgVal;
          minInd = mm;
        }
      }
      //**/ restore the marker for the mm-th row in baseDesign
      //**/ markers[kk] = mm where kk is the row index in candMatrix
      markers[markerSave] = mm;
    }
    if (printLevel_ > 1)
      printf("genDesigns iteration %ld: best value = %e\n",iter,-minVal);

    //**/ now a sample minInd has been found
    //**/ first see if minInd==nDesigns (that is, the one to remove
    //**/ is the same as the one to be added). If so, nothing is to
    //**/ be done, else (1) find out which row in candMatrix has
    //**/ row minInd in baseDesign, (2) invalidate it, (3) find
    //**/ out which row in candMatrix has row nDesigns, (4) change 
    //**/ markers to minInd, and (5) copy the row over
    if (minInd != nDesigns)
    {
      for (kk = 0; kk < nCand; kk++)
        if (markers[kk] == minInd) break;
      markers[kk] = -1;
      if (printLevel_ > 1)
      {
        printf("genDesigns iteration %ld: sample point deleted = %d\n",
               iter,kk);
      }
      for (kk = 0; kk < nCand; kk++)
        if (markers[kk] == nDesigns) break;
      markers[kk] = minInd;
      if (printLevel_ > 1)
      {
        printf("                           sample point added   = %d\n",
               kk);
      }
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = baseDesign.getEntry(nDesigns,ii);
        baseDesign.setEntry(minInd, ii, ddata);
      } 
    }
    else
    {
      //**/ invalidate the nDesigns-th row in baseDesign since it
      //**/ is not to be added. Also terminate
      for (kk = 0; kk < nCand; kk++)
        if (markers[kk] == nDesigns) break;
      markers[kk] = -1;
      if (printLevel_ > 1)
        printf("genDesigns iteration %ld: no more point exchanged\n",iter);
      break;
    }

    printf("Iter %ld: Selected test sample points so far: (1-based)\n",
           iter);
    jj = 0;
    printf("         "); 
    for (ii = 0; ii < nCand; ii++)
    {
      if (markers[ii] != -1)
      {
        printf("%4d ", ii+1); 
        jj++;
        if (jj >= 10)
        {
          jj = 0;
          printf("\n");
        }
      }
    }
    printf("\n");
  }
  //**/ now copy the first nDesigns rows of baseDesign to finalDesign
  for (kk = 0; kk < nDesigns; kk++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      ddata = baseDesign.getEntry(kk, ii);
      finalDesign.setEntry(kk, ii, ddata);
    }
  }
  printf("genDesigns: total number of iterations = %ld\n", iter);
  return 0;
}

// ************************************************************************
// improve design 
//**/ use swap points
// ------------------------------------------------------------------------
int GPBasedSampling::genDesigns2(psVector &params, psMatrix &candMatrix, 
                                 int nDesigns, psMatrix &finalDesign,
                                 psVector &markers)
{
  int    ii, jj, kk, mm, nn, status, nCand, markerSave, cnt;
  int    m1, m2, minInd, maxInd, converged, avgCnt;
  long   iter;
  double ddata, dist, meas, minVal, maxVal, avgVal;
  psMatrix CMatrix, samMatrix, baseDesign, CIMatrix;
  psVector correlations, tmpVec;

  //**/ copy existing design (from calling function) to baseDesign
  //**/ markers should be storing information about which test samples
  //**/ have been selected
  nCand = candMatrix.nrows();
  baseDesign.setDim(nDesigns, nInputs_);
  for (jj = 0; jj < nDesigns; jj++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      ddata = finalDesign.getEntry(jj,ii);
      baseDesign.setEntry(jj, ii, ddata);
    }
  }

  //**/--------------------------------------------------------
  //**/ loop a number of times until convergence
  //**/--------------------------------------------------------
  iter = 0;
  while (1)
  {
    iter++;
    samMatrix.setDim(nDesigns, nInputs_);

    //**/ scan through all samples in candMatrix and baseDesign
    converged = 1;
    //**/ minVal initialized here to control oscillations
    minVal = PSUADE_UNDEFINED;
    for (m1 = 0; m1 < nDesigns; m1++)
    {
      //**/ copy current designs to samMatrix except m1
      cnt = 0;
      for (kk = 0; kk < nDesigns; kk++)
      {
        if (kk != m1)
        { 
          for (ii = 0; ii < nInputs_; ii++)
          {
            ddata = baseDesign.getEntry(kk,ii);
            samMatrix.setEntry(cnt, ii, ddata);
          }
          cnt++;
        }
      }

      //**/ free up candMatrix(markers[m1],:)=baseDesign(m1,:)
      //**/ (change markers so that switching a row with itself 
      //**/  is allowed for comparison with other choices)
      //**/ markerSave stores which row in candMatrix is in m1-th
      //**/ row in baseDesign
      kk = 0;
      for (kk = 0; kk < nCand; kk++) if (markers[kk] == m1) break;
      markerSave = kk;
      markers[kk] = -1;

      //**/ now add one point from candMatrix
      minInd = -1;
      for (m2 = 0; m2 < nCand; m2++)
      {
        //**/ ======> select not-yet-selected point m2 
        if (markers[m2] == -1)
        {
          //**/ append the sample point to the end of samMatrix
          for (ii = 0; ii < nInputs_; ii++)
          {
            ddata = candMatrix.getEntry(m2,ii);
            samMatrix.setEntry(nDesigns-1, ii, ddata);
          }
          //**/ just set markers[m2] to something except -1
          markers[m2] = nDesigns - 1;

          //**/ construct inverse of covariance matrix
          constructCMatrix(samMatrix, CMatrix, params);

          status = CMatrix.computeInverse(CIMatrix);
          if (status != 0)
          {
            printf("GPBasedSampling ERROR in covariance inversion\n");
            exit(1);
          }

          //**/=========> find max variance given a swap
          //**/ for the rest of the points, search for max variance
          correlations.setLength(nDesigns);
          maxVal = - PSUADE_UNDEFINED;
          avgVal = 0;
          avgCnt = 0;
          for (nn = 0; nn < nCand; nn++)
          {
            if (markers[nn] == -1)
            {
              //**/ compute correlation vector
              for (jj = 0; jj < nDesigns; jj++)
              {
                dist = 0.0;
                for (ii = 0; ii < nInputs_; ii++)
                {
                  ddata = samMatrix.getEntry(jj, ii);
                  ddata -= candMatrix.getEntry(nn,ii);
                  dist += pow(ddata, 2.0) * params[ii];
                }
                dist = exp(-dist);
                if (dist < 1.0e-50) dist = 0;
                correlations[jj] = dist;
              }
              //**/ compute correlations * inv(C) * correlation
              CIMatrix.matvec(correlations, tmpVec, 0);
              meas = 0.0;
              for (jj = 0; jj < nDesigns; jj++)
                meas += correlations[jj]*tmpVec[jj];
              //**/ negative to mimic variance
              meas = -meas;
              //**/ search for max variance
              if (meas > maxVal) 
              {
                maxVal = meas;
                maxInd = nn;
              }
              avgVal += meas;
              avgCnt++;
            } // if markers[nn] == -1 
          } // nn loop
          //**/ <================== find max variance given a swap 
          avgVal /= (double) avgCnt;

          //**/ keep track of min (maxVal)
          //**/ That is, swapping rows m2 and markers[m1] gives
          //**/ prediction with smallest variance
          if (MMVorMAV_ == 0)
          {
            if (maxVal == minVal)
            {
              //**/ in case the same, select randomly from candidates
              kk = PSUADE_rand() % 2;
              if (kk == 0)
              {
                minVal = maxVal;
                minInd = m2;
              }
            }
            else if (maxVal < minVal)
            {
              minVal = maxVal;
              minInd = m2;
            }
          }
          else
          {
            if (avgVal == minVal)
            {
              //**/ in case the same, select randomly from candidates
              kk = PSUADE_rand() % 2;
              if (kk == 0)
              {
                minVal = avgVal;
                minInd = m2;
              }
            }
            else if (avgVal < minVal)
            {
              minVal = avgVal;
              minInd = m2;
            }
          }
          //**/ now candMatrix(maxInd,:) gives the max variance
          //**/ when baseDesign(m1,:) is switched with candMatrix(m2,:)
          //**/ restore the markers for candMatrix(m2,:) 
          markers[m2] = -1;
        } // markers[m2] != -1 loop
      } 
      //**/ <================== m2 loop

      //**/ now swap baseDesign(m1,:) with candMatrix(minInd,:)
      //**/ first see which row in candMatrix is in baseDesign(m1,:)

      //**/ if not swapping the same row, swap
      if (markerSave != minInd && minInd != -1)
      {
        for (ii = 0; ii < nInputs_; ii++)
        {
          ddata = candMatrix.getEntry(minInd,ii);
          baseDesign.setEntry(m1, ii, ddata);
        } 
        //**/ update markers
        markers[minInd] = m1;
        markerSave  = -1;
        if (printLevel_ > 1)
          printf("genDesigns2 iter %ld: swap rows %d (in) and %d (out)\n",
                 iter, minInd, kk);
        converged = 0;
        printf("Best selected set so far: (best value = %e)\n",-minVal);
        jj = 0;
        for (ii = 0; ii < nCand; ii++)
        {
          if (markers[ii] != -1)
          {
            printf("%4d ", ii+1); 
            jj++;
            if (jj >= 10)
            {
              jj = 0;
              printf("\n");
            }
          }
        }
        printf("\n");
      } // if loop
      if (markerSave != -1) markers[markerSave] = m1;
    } // m1 loop
    if (converged == 1) break;
  }
  //**/ finally copy the first nDesigns rows of baseDesign to finalDesign
  for (kk = 0; kk < nDesigns; kk++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      ddata = baseDesign.getEntry(kk, ii);
      finalDesign.setEntry(kk, ii, ddata);
    }
  }
  printf("genDesigns2: total number of iterations = %ld\n", iter);
  return 0;
}

// ************************************************************************
// generate optimal design 
// ------------------------------------------------------------------------
int GPBasedSampling::genDesignsUltimate(psVector &params, 
                               psMatrix &candMatrix, 
                               int nDesigns, psMatrix &finalDesign)
{
  int    ii, jj, kk, nn, nCand, option=2;
  long   iter;
  double ddata, minVal;
  psMatrix  CMatrix, samMatrix, CIMatrix;
  psIVector vecInds, vecBestInds;

  //**/ initialization
  nCand = candMatrix.nrows();
  vecInds.setLength(nCand);
  vecBestInds.setLength(nCand);
  minVal = PSUADE_UNDEFINED;
  iter = 0;

  //**/ option = 2 is more efficient
  if (option == 1)
    evaluateUltimate(nCand,vecInds,0,nDesigns,nDesigns,params,
                     candMatrix,vecBestInds,minVal,iter);
  else
    evaluateUltimateOpt(nCand,nDesigns,params,candMatrix,
                        vecBestInds,minVal);
  printAsterisks(PL_INFO, 0);
  printAsterisks(PL_INFO, 0);
  if (MMVorMAV_ == 0) 
    printf("MMV: FINAL SET OF SELECTED POINTS: \n");
  else
    printf("MAV: FINAL SET OF SELECTED POINTS: \n");
  jj = 0;
  for (ii = 0; ii < nCand; ii++)
  {
    if (vecBestInds[ii] == 1)
    {
      printf("%4d ", ii+1); 
      jj++;
      if (jj >= 10)
      {
        jj = 0;
        printf("\n");
      }
    }
  }
  if (MMVorMAV_ == 0)
    printf("\n ===> MMV best maximum variance reduction = %10.4e\n",
           -minVal);
  else
    printf("\n ===> MAV best average variance reduction = %10.4e\n",
           -minVal);
  printAsterisks(PL_INFO, 0);
  printAsterisks(PL_INFO, 0);
  finalDesign.setDim(nDesigns, nInputs_);
  nn = 0;
  for (kk = 0; kk < nCand; kk++)
  {
    if (vecBestInds[kk] == 1)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = candMatrix.getEntry(kk, ii);
        finalDesign.setEntry(nn, ii, ddata);
      }
      nn++;
    }
  }
  return 0;
} 
  
// ************************************************************************
// cycle all combination 
// ------------------------------------------------------------------------
int GPBasedSampling::evaluateUltimate(int nCand, psIVector &vecInds,
              int position, int nDesigns, int K, psVector &params, 
              psMatrix &candMatrix,psIVector &vecBestInds,double &minVal,
              long &iter)
{
  int    doEval=0, kk, ii, jj, nn, count, status, *localIndices, avgCnt;
  double ddata, dist, meas, maxVal, avgVal;
  FILE   *fp;
  psVector  correlations, tmpVec;
  psIVector vecLocalInds;
  psMatrix  samMatrix, CMatrix, CIMatrix;

  //**/ stop processing
  fp = fopen("psuade_stop", "r");
  if (fp != NULL)
  {
    printf("PSUADE found the psuade_stop file ==> terminate.\n");
    fclose(fp);
    return 0;
  }

  //**/ find out whether to evaluate or not
  vecLocalInds = vecInds;
  doEval = 0;
  if (K == 0)
  {
    doEval = 1;
  }
  //**/ fill out the rest of the array to make up for M 1's
  if (K != 0 && (nCand-position) <= K)
  {
    for (ii = 0; ii < K; ii++) vecLocalInds[position+ii] = 1;
    doEval = 1;
  }

  //**/ if evaluate, go ahead
  if (doEval == 1)
  {
    samMatrix.setDim(nDesigns, nInputs_);
    count = 0;
    printf("Iteration %ld: ", iter+1);
    for (kk = 0; kk < nCand; kk++)
    {
      if (vecLocalInds[kk] != 0)
      { 
        printf("%d ", kk+1);
        for (ii = 0; ii < nInputs_; ii++)
        {
          ddata = candMatrix.getEntry(kk,ii);
          samMatrix.setEntry(count, ii, ddata);
        }
        count++;
      }
    }
    printf("\n");
    if (count != nDesigns)
    {
      printf("Catastrophic ERROR\n");
      exit(1);
    }
    //**/ construct inverse of covariance matrix
    constructCMatrix(samMatrix, CMatrix, params);
    status = CMatrix.computeInverse(CIMatrix);
    if (status != 0)
    {
      printf("GPBasedSampling ERROR in covariance inversion\n");
      exit(1);
    }

    //**/ compute correlation vector c between selected set and
    //**/ each of the points in the evaluation sample, and
    //**/ find the max variance among the evaluation set
    correlations.setLength(nDesigns);
    maxVal = - PSUADE_UNDEFINED;
    avgVal = 0;
    avgCnt = 0;
    for (nn = 0; nn < nCand; nn++)
    {
      if (vecLocalInds[nn] == 0)
      {
        //**/ compute correlation vector
        for (jj = 0; jj < nDesigns; jj++)
        {
          dist = 0.0;
          for (ii = 0; ii < nInputs_; ii++)
          {
            ddata = samMatrix.getEntry(jj, ii);
            ddata -= candMatrix.getEntry(nn,ii);
            dist += pow(ddata, 2.0) * params[ii];
          }
          dist = exp(-dist);
          if (dist < 1.0e-50) dist = 0;
          correlations[jj] = dist;
        }
        //**/ compute correlations * inv(C) * correlation
        CIMatrix.matvec(correlations, tmpVec, 0);
        meas = 0.0;
        for (jj = 0; jj < nDesigns; jj++)
          meas += correlations[jj]*tmpVec[jj];
        //**/ negative to mimic variance (total variance - meas)
        //**/ where total variance is a constant, and so negative
        meas = -meas;
        //**/ search for max variance
        if (meas > maxVal) maxVal = meas;
        avgVal += meas;
      } // if markers[nn] == -1 
    } // nn loop
    avgVal /= (double) avgCnt;

    fp = fopen("psuade_print", "r");
    if (fp != NULL)
    {
      fclose(fp);
      printf("Iteration %ld: selected set = \n",iter+1);
      jj = 0;
      for (ii = 0; ii < nCand; ii++)
      {
        if (vecLocalInds[ii] == 1)
        {
          printf("%4d ", ii+1); 
          jj++;
          if (jj >= 10)
          {
            jj = 0;
            printf("\n");
          }
        }
      }
      printf("\n");
    }
    if (MMVorMAV_ == 0)
      printf(" ==> minimum variance reduction for this set = %10.4e\n",-maxVal);
    else
      printf(" ==> average variance reduction for this set = %10.4e\n",-avgVal);

    //**/ If the max variance among the evaluation sample is less than
    //**/ the min-max value so far, update the min-max value
    if (MMVorMAV_ == 0)
    {
      if (maxVal < minVal)
      {
        minVal = maxVal;
        vecBestInds = vecLocalInds;
        printf("Iteration %ld: selected points\n", iter+1);
        jj = 0;
        printf("              "); 
        for (ii = 0; ii < nCand; ii++)
        {
          if (vecBestInds[ii] == 1)
          {
            printf("%4d ", ii); 
            jj++;
            if (jj >= 10)
            {
              jj = 0;
              printf("\n");
            }
          }
        }
        printf("\n");
        printf(" ==> MMV best maximum variance reduction so far = %10.4e\n",
               -minVal);
      }  
      iter++;
    }
    else
    {
      if (avgVal < minVal)
      {
        minVal = avgVal;
        vecBestInds = vecLocalInds;
        printf("Iteration %ld: selected points\n", iter+1);
        jj = 0;
        printf("              "); 
        for (ii = 0; ii < nCand; ii++)
        {
          if (vecBestInds[ii] == 1)
          {
            printf("%4d ", ii); 
            jj++;
            if (jj >= 10)
            {
              jj = 0;
              printf("\n");
            }
          }
        }
        printf("\n");
        printf(" ==> MAV best average variance reduction so far = %10.4e\n",
               -minVal);
      }  
      iter++;
    }
    return 0;
  }

  //**/ recursion
  vecLocalInds[position] = 0;
  evaluateUltimate(nCand, vecLocalInds, position+1, nDesigns, K, params, 
                   candMatrix, vecBestInds, minVal, iter);
  vecLocalInds[position] = 1;
  evaluateUltimate(nCand, vecLocalInds, position+1, nDesigns, K-1, params, 
                   candMatrix, vecBestInds, minVal, iter);
  return 0;
}

// ************************************************************************
// more efficient version 
// ------------------------------------------------------------------------
int GPBasedSampling::evaluateUltimateOpt(int nCand, int nDesigns, 
             psVector &params, psMatrix &candMatrix,psIVector &vecBestInds,
             double &minVal)
{
  int    ii, jj, kk, nn, count, status, avgCnt;
  double ddata, dist, meas, maxVal, avgVal;
  FILE   *fp=NULL;
  psVector  correlations, tmpVec;
  psIVector vecLocalInds, vecPermute;
  psMatrix  samMatrix, CMatrix, CIMatrix;

  //**/ permute sample for potential better efficiency
  vecPermute.setLength(nCand);
  generateRandomIvector(nCand, vecPermute.getIVector());
  //for (ii = 0; ii < nCand; ii++) 
  //  vecPermute[ii] = nCand - ii - 1;

  //**/ next set up for exhaustive search
  //**/ first set up initial vecLocalInds (vecLocalInds[ii] has
  //**/ the selected sample number minus 1)
  vecLocalInds.setLength(nDesigns);
  for (ii = 0; ii < nDesigns; ii++) vecLocalInds[ii] = ii;

  //**/ iterate 
  long iter=0;
  while (1)
  {
    //**/ extract samples corresponding to the selected set
    //**/ and put them into samMatrix
    samMatrix.setDim(nDesigns, nInputs_);
    count = 0;
    for (kk = 0; kk < nDesigns; kk++)
    {
      jj = vecPermute[vecLocalInds[kk]];
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = candMatrix.getEntry(jj,ii);
        samMatrix.setEntry(count, ii, ddata);
      }
      count++;
    }

    //**/ construct inverse of covariance matrix from samMatrix
    constructCMatrix(samMatrix, CMatrix, params);
    status = CMatrix.computeInverse(CIMatrix);
    if (status != 0)
    {
      printf("GPBasedSampling ERROR in covariance inversion\n");
      exit(1);
    }

    //**/ compute correlation between unselected samples and
    //**/ selected samples
    correlations.setLength(nDesigns);
    maxVal = - PSUADE_UNDEFINED;
    avgVal = 0;
    avgCnt = 0;
    for (nn = 0; nn < nCand; nn++)
    {
      //**/ see if sample in selected set
      for (jj = 0; jj < nDesigns; jj++)
        if (vecLocalInds[jj] == nn) break;

      //**/ if sample is not in selected set, compute its
      //**/ correlation with the selected set
      if (jj == nDesigns)
      {
        //**/ (1) compute correlations
        kk = vecPermute[nn];
        for (jj = 0; jj < nDesigns; jj++)
        {
          dist = 0.0;
          for (ii = 0; ii < nInputs_; ii++)
          {
            ddata = samMatrix.getEntry(jj, ii);
            ddata -= candMatrix.getEntry(kk,ii);
            dist += pow(ddata, 2.0) * params[ii];
          }
          dist = exp(-dist);
          if (dist < 1.0e-50) dist = 0;
          correlations[jj] = dist;
        }

        //**/ (2) compute correlations * inv(C) * correlation
        CIMatrix.matvec(correlations, tmpVec, 0);

        //**/ (3) compute measure
        //**/ negative to mimic variance (total variance - meas)
        //**/ where total variance is a constant, and so negative
        meas = 0.0;
        for (jj = 0; jj < nDesigns; jj++)
          meas += correlations[jj]*tmpVec[jj];
        meas = -meas;

        //**/ (4) update max variance
        if (meas > maxVal) maxVal = meas;
        avgVal += meas;
        avgCnt++;
      } 
    } // nn loop
    avgVal /= (double) avgCnt;

    //**/ update global minimum and display
    if (printLevel_ > 3)
    {
      printf("Iteration %ld: selected set = \n",iter+1);
      for (ii = 0; ii < nDesigns; ii++)
      {
        printf("%5d ", vecPermute[vecLocalInds[ii]]+1); 
        if ((ii != 0) && (ii % 10 == 0)) printf("\n");
      }
      printf("\n");
      if (MMVorMAV_ == 0)
        printf(" ==> minimum variance reduction for this set = %10.4e\n",
               -maxVal);
      else
        printf(" ==> average variance reduction for this set = %10.4e\n",
               -avgVal);
    }
    if (MMVorMAV_ == 0)
    {
      if (maxVal < minVal)
      {
        minVal = maxVal;
        vecBestInds.setLength(nCand);
        for (ii = 0; ii < nDesigns; ii++)
          vecBestInds[vecPermute[vecLocalInds[ii]]] = 1;
        printf("Iteration %ld: selected points = \n",iter+1);
        jj = 0;
        printf("              "); 
        for (ii = 0; ii < nCand; ii++)
        {
          if (vecBestInds[ii] == 1)
          {
            printf("%4d ", ii+1); 
            jj++;
            if (jj >= 10)
            {
              jj = 0;
              printf("\n");
            }
          }
        }
        printf("\n");
        printf(" ==> MMV best maximum variance reduction so far = %10.4e\n",
               -minVal);
      }
    }
    else
    {
      if (avgVal < minVal)
      {
        minVal = avgVal;
        vecBestInds.setLength(nCand);
        for (ii = 0; ii < nDesigns; ii++)
          vecBestInds[vecPermute[vecLocalInds[ii]]] = 1;
        printf("Iteration %ld: selected points = \n", iter+1);
        jj = 0;
        printf("              "); 
        for (ii = 0; ii < nCand; ii++)
        {
          if (vecBestInds[ii] == 1)
          {
            printf("%4d ", ii+1); 
            jj++;
            if (jj >= 10)
            {
              jj = 0;
              printf("\n");
            }
          }
        }
        printf("\n");
        printf(" ==> MAV best average variance reduction so far = %10.4e\n",
               -minVal);
      }
    }

    //**/ update subset
    vecLocalInds[nDesigns-1]++;
    for (ii = nDesigns-1; ii > 0; ii--)
    {
      if (vecLocalInds[ii] <= nCand-nDesigns+ii) break;
      else
      {
        vecLocalInds[ii-1] = vecLocalInds[ii-1] + 1;
      }
    }
    for (ii = 1; ii < nDesigns; ii++)
      if (vecLocalInds[ii] > nCand-nDesigns+ii)
        vecLocalInds[ii] = vecLocalInds[ii-1] + 1;
    if (vecLocalInds[0] > nCand-nDesigns) break;

    iter++;

    if (iter % 10000 == 0)
    {
      fp = fopen("psuade_stop", "r");
      if (fp != NULL)
      {
        printf("PSUADE found the psuade_stop file ==> terminate.\n");
        fclose(fp);
        break;
      }
      fp = fopen("psuade_print", "r");
      if (fp != NULL)
      {
        fclose(fp);
        printLevel_ = 3;
      }
      //else printLevel_ = 0;
    }
  }
  return 0;
}

// ************************************************************************
// generate an initial design 
// ------------------------------------------------------------------------
int GPBasedSampling::genInitialDesigns(psVector params, 
                         psMatrix &candMatrix, int nDesigns, 
                         psMatrix &finalDesign, psVector &markers)
{
  int    ii, jj, mm, nn, ss, status, nCand, minInd, avgCnt;
  double ddata, dist, meas, minVal, maxVal, avgVal;
  psMatrix CMatrix, samMatrix;
  psVector correlations, tmpVec;

  //**/ incrementally add design points
  nCand = candMatrix.nrows();
  for (ss = 0; ss < nDesigns; ss++)
  {
    samMatrix.setDim(ss+1, nInputs_);
    //**/ copy existing design to samMatrix
    for (jj = 0; jj < ss; jj++)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = finalDesign.getEntry(jj,ii);
        samMatrix.setEntry(jj, ii, ddata);
      }
    }

    //**/ now add one point at a time from the test set, and, for
    //**/ each addition, compute min(variance).
    minVal = PSUADE_UNDEFINED;
    minInd = -1;
    for (mm = 0; mm < nCand; mm++)
    {
      //**/ if sample mm is not in the design set, see if adding it
      //**/ to the set will give smallest max(variance)
      if (markers[mm] == -1)
      {
        //**/ append the sample point to samMatrix
        for (ii = 0; ii < nInputs_; ii++)
        {
          ddata = candMatrix.getEntry(mm,ii);
          samMatrix.setEntry(ss, ii, ddata);
        }
        //**/ mark test ss as under consideration
        markers[mm] = ss;

        //**/ construct inverse of covariance matrix
        constructCMatrix(samMatrix, CMatrix, params);
        status = CMatrix.computeInverse(CMatrix);
        if (status != 0)
        {
          printf("GPBasedSampling ERROR in covariance inversion\n");
          exit(1);
        }

        //**/ for the rest of the points, search for max variance
        correlations.setLength(ss+1);
        maxVal = - PSUADE_UNDEFINED;
        avgVal = 0;
        avgCnt = 0;
        for (nn = 0; nn < nCand; nn++)
        {
          if (markers[nn] == -1)
          {
            //**/ compute correlation vector
            for (jj = 0; jj < ss+1; jj++)
            {
              dist = 0.0;
              for (ii = 0; ii < nInputs_; ii++)
              {
                ddata = samMatrix.getEntry(jj, ii);
                ddata -= candMatrix.getEntry(nn,ii);
                dist += pow(ddata, 2.0) * params[ii];
              }
              dist = exp(-dist);
              if (dist < 1.0e-50) dist = 0;
              correlations[jj] = dist;
            }
            //**/ compute correlations * inv(C) * correlation
            CMatrix.matvec(correlations, tmpVec, 0);
            meas = 0.0;
            for (jj = 0; jj <= ss; jj++)
              meas += correlations[jj]*tmpVec[jj];
            //**/ negative to mimic variance
            //**/ original form: total variance - meas
            meas = -meas;
            //**/ search for max variance
            if (meas > maxVal) maxVal = meas;
            avgVal += meas;
            avgCnt++;
          }
        }
        avgVal /= (double) avgCnt;

        //**/ if, after adding point mm, max prediction variance for
        //**/ all other points is smallest so far, register mm as a 
        //**/ possible candidate for addition
        if (MMVorMAV_ == 0)
        {
          if (maxVal < minVal)
          {
            minVal = maxVal;
            minInd = mm;
          }
          else if (maxVal == minVal)
          {
            //**/ if same min-max value, toss a coin to decide
            nn = PSUADE_rand() % 2;
            if (nn == 1)
            {
              minVal = maxVal;
              minInd = mm;
            }
          }
          markers[mm] = -1;
        }
        else
        {
          if (avgVal < minVal)
          {
            minVal = avgVal;
            minInd = mm;
          }
          else if (avgVal == minVal)
          {
            //**/ if same min-max value, toss a coin to decide
            nn = PSUADE_rand() % 2;
            if (nn == 1)
            {
              minVal = avgVal;
              minInd = mm;
            }
          }
          markers[mm] = -1;
        }
      }
    }
    //**/ now a sample minInd has been found
    //**/ insert it into the last row of the baseDesign matrix
    for (ii = 0; ii < nInputs_; ii++)
    {
      ddata = candMatrix.getEntry(minInd,ii);
      finalDesign.setEntry(ss, ii, ddata);
    } 
    markers[minInd] = ss;
    if (printLevel_ > 1)
      printf("genInitDesigns iteration %d: sample point added = %d\n",
             ss+1,minInd);
  }
  return 0;
}

// ************************************************************************
// generate an initial design 
// ------------------------------------------------------------------------
int GPBasedSampling::genInitialDesigns2(psVector params, 
                         psMatrix &candMatrix, int nDesigns, 
                         psMatrix &finalDesign, psVector &markers)
{
  int    ii, jj, ll, mm, nn, ss, status, nCand, minInd, minInd2;
  double ddata, dist, meas, minVal, maxVal;
  psMatrix CMatrix, samMatrix;
  psVector correlations, tmpVec;

  //**/ incrementally add design points
  nCand = candMatrix.nrows();
  for (ss = 0; ss < nDesigns; ss+=2)
  {
    samMatrix.setDim(ss+2, nInputs_);
    //**/ copy existing design to samMatrix
    for (jj = 0; jj < ss; jj++)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = finalDesign.getEntry(jj,ii);
        samMatrix.setEntry(jj, ii, ddata);
      }
    }

    //**/ now one point at a time from the test set, and, for
    //**/ each addition, compute min(variance).
    minVal = PSUADE_UNDEFINED;
    minInd = -1;
    for (mm = 0; mm < nCand; mm++)
    {
      for (ll = 0; ll < nCand; ll++)
      {
        if (markers[ll] == -1 && markers[mm] == -1 && (ll != mm))
        {
          //**/ append the sample point to samMatrix
          for (ii = 0; ii < nInputs_; ii++)
          {
            ddata = candMatrix.getEntry(ll,ii);
            samMatrix.setEntry(ss, ii, ddata);
          }
          for (ii = 0; ii < nInputs_; ii++)
          {
            ddata = candMatrix.getEntry(mm,ii);
            samMatrix.setEntry(ss+1, ii, ddata);
          }
          markers[ll] = ss;
          markers[mm] = ss + 1;

          //**/ construct inverse of covariance matrix
          constructCMatrix(samMatrix, CMatrix, params);
          status = CMatrix.computeInverse(CMatrix);
          if (status != 0)
          {
            printf("GPBasedSampling ERROR in covariance inversion\n");
            exit(1);
          }

          //**/ for the rest of the points, search for max variance
          correlations.setLength(ss+2);
          maxVal = - PSUADE_UNDEFINED;
          for (nn = 0; nn < nCand; nn++)
          {
            if (markers[nn] == -1)
            {
              //**/ compute correlation vector
              for (jj = 0; jj < ss+2; jj++)
              {
                dist = 0.0;
                for (ii = 0; ii < nInputs_; ii++)
                {
                  ddata = samMatrix.getEntry(jj, ii);
                  ddata -= candMatrix.getEntry(nn,ii);
                  dist += pow(ddata, 2.0) * params[ii];
                }
                dist = exp(-dist);
                if (dist < 1.0e-50) dist = 0;
                correlations[jj] = dist;
              }
              //**/ compute correlations * inv(C) * correlation
              CMatrix.matvec(correlations, tmpVec, 0);
              meas = 0.0;
              for (jj = 0; jj < ss+2; jj++)
                meas += correlations[jj]*tmpVec[jj];
              //**/ negative to mimic variance
              meas = -meas;
              //**/ search for max variance
              if (meas > maxVal) maxVal = meas;
            }
          }
          if (maxVal < minVal)
          {
            minVal = maxVal;
            minInd = ll;
            minInd2 = mm;
          }
          else if (maxVal == minVal)
          {
            //**/ if same min-max value, toss a coin to decide
            nn = PSUADE_rand() % 2;
            if (nn == 1)
            {
              minVal = maxVal;
              minInd = ll;
              minInd2 = mm;
            }
          }
          markers[ll] = -1;
          markers[mm] = -1;
        }
      }
    }
    //**/ now a sample minInd has been found
    //**/ insert it into the last row of the baseDesign matrix
    for (ii = 0; ii < nInputs_; ii++)
    {
      ddata = candMatrix.getEntry(minInd,ii);
      finalDesign.setEntry(ss, ii, ddata);
    } 
    markers[minInd] = ss;
    if (ss+1 < nDesigns)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = candMatrix.getEntry(minInd2,ii);
        finalDesign.setEntry(ss+1, ii, ddata);
      }
      markers[minInd2] = ss + 1;
    }
    if (printLevel_ > 1)
      printf("genInitDesigns2 iteration %d: sample points added   = %d %d\n",
             ss+1,minInd,minInd2);
  }
  return 0;
}

// ************************************************************************
// evaluate a candidate set
//**/ candMatrix - a matrix containing all candidates
//**/ designMatrix - a matrix containing current selection to be evaluated
//**/ What this function does: given a design matrix (which is a subset of
//**/    the candidate matrix), construct a covariance matrix and
//**/    evaluate prediction uncertainties at the unselected candidates
// ------------------------------------------------------------------------
double GPBasedSampling::evaluateCandidateSet(psMatrix &candMatrix,
                                             psMatrix &designMatrix)
{
  int    mm, nn, ii, jj, ind, nCand, nDesigns, status, avgCnt;
  double retVal=0, dist, ddata, meas;
  double minVal = PSUADE_UNDEFINED;
  double maxVal, avgVal;
  psVector vecCorrelations, vecTmp;
  psMatrix CMatrix, CIMatrix, gpMatrix;

  //**/ construct covariance matrix using current design selection
  constructCMatrix(designMatrix, CMatrix, VecGPParams_);
  status = CMatrix.computeInverse(CIMatrix);
  if (status != 0)
  {
    printf("GPBasedSampling ERROR in covariance inversion\n");
    printf("Possible reason: the covariance matrix is singular.\n");
    printf("Covariance matrix = \n");
    CMatrix.print();
    exit(1);
  }

  //**/ now, for each of the unselected candidate, estimate prediction
  //**/ uncertainty - negative of inner product of correlation in 
  //**/ covariance norm
  nCand = candMatrix.nrows();
  nDesigns = designMatrix.nrows();
  vecCorrelations.setLength(nDesigns);
  maxVal = - PSUADE_UNDEFINED;
  avgVal = 0;
  avgCnt = 0;
  for (mm = 0; mm < nCand; mm++)
  {
    //**/ see if this candidate is in the selected set
    //**/ compute correlation vector
    for (nn = 0; nn < nDesigns; nn++)
    {
      for (ii = 0; ii < nInputs_; ii++)
        if (candMatrix.getEntry(mm,ii) != designMatrix.getEntry(nn,ii))
          break;
      //**/ if ii = nInputs_ ==> found: cand row mm == design row nn
      if (ii == nInputs_) break;
    }
    //**/ if candidate mm not found in design matrix, then create
    //**/ correlation with selected designs
    if (nn == nDesigns)
    {
      for (jj = 0; jj < nDesigns; jj++)
      {
        dist = 0.0;
        for (ii = 0; ii < nInputs_; ii++)
        {
          ddata = designMatrix.getEntry(jj, ii);
          ddata -= candMatrix.getEntry(mm,ii);
          dist += pow(ddata, 2.0) * VecGPParams_[ii];
        }
        dist = exp(-dist);
        if (dist < 1.0e-50) dist = 0;
        vecCorrelations[jj] = dist;
      }
      CIMatrix.matvec(vecCorrelations, vecTmp, 0);
      meas = 0.0;
      for (jj = 0; jj < nDesigns; jj++)
        meas += vecCorrelations[jj]*vecTmp[jj];
      //**/ negative to mimic variance
      //**/ original form: total variance - meas
      meas = -meas;
      //**/ search for max variance
      if (meas > maxVal) maxVal = meas;
      avgVal += meas;
      avgCnt++;
    }
  }
  avgVal /= (double) avgCnt;
  if (MMVorMAV_ ==  0) retVal = maxVal;
  else                 retVal = avgVal;
  return retVal;
}

// ************************************************************************
// print designs
// ------------------------------------------------------------------------
int GPBasedSampling::printDesigns(psMatrix &designs, FILE *fp)
{
  int ii, kk, nrows, ncols;
  if (fp != NULL)
  {
    nrows = designs.nrows();
    ncols = designs.ncols();
    for (kk = 0; kk < nrows; kk++)
    {
      fprintf(fp, "%4d ", kk+1);
      for (ii = 0; ii < ncols; ii++)
        fprintf(fp, "%12.4e ", designs.getEntry(kk,ii));
      fprintf(fp, "\n");
    }
    return 0;
  }
  else return -1;
}

// ************************************************************************
// set MMV designs
// ------------------------------------------------------------------------
void GPBasedSampling::setMMV()
{
  MMVorMAV_ = 0;
}

// ************************************************************************
// set MAV designs
// ------------------------------------------------------------------------
void GPBasedSampling::setMAV()
{
  MMVorMAV_ = 1;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
GPBasedSampling& GPBasedSampling::operator=(const GPBasedSampling &)
{
  printf("GPBasedSampling operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

