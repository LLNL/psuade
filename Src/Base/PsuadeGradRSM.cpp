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
// Functions for the class PsuadeBase
// AUTHOR : CHARLES TONG
// DATE   : 2008
// ************************************************************************
extern "C" {
  void dsyev_(char *, char *, int *, double *, int *, double *,
              double *, int *, int *);
}

// ------------------------------------------------------------------------
// system includes
// ------------------------------------------------------------------------
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ------------------------------------------------------------------------
// local includes : class definition and utilities
// ------------------------------------------------------------------------
#include "Psuade.h"
#include "PsuadeBase.h"
#include "dtype.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PDFManager.h"
#include "JobCntl.h"
#include "PrintingTS.h"
#include "psVector.h"
#include "psStrings.h"

#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// run the special adaptive mode with gradient analyzer
// ------------------------------------------------------------------------
int PsuadeBase::runAdaptiveGradBased()
{
  int nInputs, nOutputs, nSamples, ii, iR, iS, kk, iRov, nRefineCurr;
  int nROVMax, nRefines, *refineSeps, maxParallelJobs, lwork, info, flag;
  int Ysize, count, status, rovID, rovCnt, initNSamples, sampleIndex;
  int refineSize, nRefineLocal;
  double error, anaThreshold, threshold, Ynew, YVal, errMax, dtemp, eig2;
  double diff, eigMax, dsum2, stdev, mean, tolerance, tol2, threshDec;
  double minEigen=0.0;
  char   jobz, uplo, lineIn[500], fileName[500], pString[1000];
  FILE   *fileIn;
  pData  pPtr, pLowerB, pUpperB, pInputs, pOutputs, pInputs2, pOutputs2;
  psVector   vecSamInps, vecSamOuts, vecSamROVVals, vecSamROVErrs;
  psVector   vecROVStoreVals, vecROVStoreGrads, vecROVStorePts;
  psVector   vecROVStoreEigens, vecYstore, vecRanges, vecWT, vecEigs;
  psVector   vecCurrX, vecGrad, vecHess;
  psIVector  vecStas, vecSamROVInds;
  PsuadeData *psIO;
  FunctionInterface *funcIO;
  psStrings  XNames, YNames;

  //**/ ----------------------------------------------------------------
  //**/ header
  //**/ ----------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO, 
       "PSUADE adaptiveGradBased: adaptive sampling based on using\n");
  printOutTS(PL_INFO,"       gradients to form regions of validity.\n");
  printOutTS(PL_INFO,"       (METIS sampling recommendated)\n");
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO,
     "Note: Turn on rs_expert mode to set RS parameters.\n");
  printOutTS(PL_INFO,
     "      Turn on outputLevel (>0) to get error histogram.\n");
  printEquals(PL_INFO, 0);

  //**/ ----------------------------------------------------------------
  //**/ error checking and correcting
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("input_ninputs", pPtr);
  nInputs  = pPtr.intData_;
  psuadeIO_->getParameter("output_noutputs", pPtr);
  nOutputs  = pPtr.intData_;
  if (nInputs > 12)
  {
    printOutTS(PL_ERROR, 
         "PSUADE adaptiveGradBased ERROR: nInputs should be <= 12.\n");
    exit(1);
  }
  if (nOutputs != 1)
  {
    printOutTS(PL_ERROR, 
         "PSUADE adaptiveGradBased ERROR: nOutputs should be 1.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ get information on maximum number of evaluation
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("method_nsamples", pPtr);
  nSamples = pPtr.intData_;
  psuadeIO_->getParameter("method_nrefinements", pPtr);
  nRefines = pPtr.intData_;
  psuadeIO_->getParameter("method_refine_size", pPtr);
  refineSize = pPtr.intData_;
  nROVMax = nSamples + refineSize * nRefines;
  printOutTS(PL_INFO, 
       "PSUADE adaptiveGradBased: max no. of sample points = %d\n",
         nROVMax);
  printOutTS(PL_INFO, 
       "PSUADE adaptiveGradBased: no. of refinements (set at 4)   = %d\n",
       nRefines);

  //**/ ----------------------------------------------------------------
  //**/ get threshold, refinement, and run information from psuadeIO
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("app_maxparalleljobs", pPtr);
  maxParallelJobs = pPtr.intData_;
  psuadeIO_->getParameter("ana_threshold", pPtr);
  anaThreshold = pPtr.dbleData_;
  printOutTS(PL_INFO,"PSUADE adaptiveGradBased: analysis threshold = %e\n", 
             anaThreshold);
  if (anaThreshold > 0.5)
  {
    printOutTS(PL_INFO, 
         "PSUADE adaptiveGradBased INFO: threshold must be <= 0.5.\n");
    printOutTS(PL_INFO, 
         "                               threshold reset to 0.5.\n");
    anaThreshold = 0.5;
  }
  psuadeIO_->getParameter("input_lbounds", pLowerB);
  double *iLowerB = pLowerB.dbleArray_;
  psuadeIO_->getParameter("input_ubounds", pUpperB);
  double *iUpperB = pUpperB.dbleArray_;
  Ysize = (nInputs + 1) * nInputs / 2 + nInputs + 2;
  vecCurrX.setLength(nInputs);

  //**/ ----------------------------------------------------------------
  //**/ if there is already an existing file, read it and reconstruct
  //**/ ----------------------------------------------------------------
  nSamples = 0;
  threshold = anaThreshold;
  if (psConfig_.AnaExpertModeIsOn())
  {
    printOutTS(PL_INFO, "The current cutoff threshold (2nd order) is %e.\n",
               anaThreshold);
    //**/ progressive thresholding diabled
    //threshold = 0.0;
    //while (threshold <= 0.0 || threshold < anaThreshold)
    //{
    //   printOutTS(PL_INFO, "For progressive thresholding, enter initial ");
    //   printOutTS(PL_INFO, "threshold (%e to 0.5): ", anaThreshold);
    //   scanf("%lg", &threshold);
    //}
    threshold = anaThreshold;
    if (threshold != anaThreshold)
    {
      threshDec = 0.0;
      while (threshDec <= 0.0 || threshDec >= 1.0)
      {
        printf("Enter a threshold decrement factor (> 0, < 1): ");
        scanf("%lg", &threshDec);
      }
    }
    printOutTS(PL_INFO, 
         "Max eigenvalue of Hessian controls extent of ROV.\n");
    printOutTS(PL_INFO, 
         "You can set the minimum value to control the extent.\n");
    minEigen = -1;
    while (minEigen < 0.0)
    {
      printf("Enter the minimum eigenvalue (0 if no minimum) : ");
      scanf("%lg", &minEigen);
    }
    printOutTS(PL_INFO, 
         "Do you have a partial PsuadeGradRSM.rov file? (y or n) ");
    scanf("%s", lineIn);
    if (lineIn[0] == 'y')
    {
      //**/ read ROV data file 
      printAsterisks(PL_INFO, 0);
      printf("Enter ROV file name : ");
      scanf("%s", fileName);
      fileIn = fopen(fileName, "r");
      if (fileIn == NULL)
      {
         printOutTS(PL_INFO, "File name %s found.\n", fileName);
         return -1;
      }
      fclose(fileIn);
      psIO = new PsuadeData();
      status = psIO->readPsuadeFile(fileName);
      if (status != 0)
      {
        printOutTS(PL_ERROR, "ERROR: Problem reading file %s.\n", 
                   fileName);
        return -1;
      }
      psIO->getParameter("input_ninputs", pPtr);
      if (pPtr.intData_ != nInputs)
      {
        printOutTS(PL_INFO, 
            "nInputs in file %s does not match with current file.\n",
            fileName);
        return -1;
      }
      psIO->getParameter("output_noutputs", pPtr);
      if (pPtr.intData_ != Ysize)
      {
        printOutTS(PL_INFO, 
            "nOutputs in file %s (%d) is not valid (should be %d).\n",
            fileName, pPtr.intData_, Ysize);
        return -1;
      }
      psIO->getParameter("method_nsamples", pPtr);
      iRov = pPtr.intData_;
      if (iRov > nROVMax || iRov <= 0)
      {
        printOutTS(PL_ERROR, 
            "ERROR: nROVs in file %s too large (should be <= %d)\n",
            fileName, nROVMax);
        return -1;
      }
      psIO->getParameter("input_sample", pInputs);
      vecROVStorePts.setLength(nROVMax*nInputs);
      for (iR = 0; iR < iRov*nInputs; iR++)
        vecROVStorePts[iR] = pInputs.dbleArray_[iR];
      psIO->getParameter("output_sample", pOutputs);
      vecROVStoreVals.setLength(nROVMax);
      vecROVStoreGrads.setLength(nROVMax*nInputs);
      for (iR = 0; iR < iRov; iR++)
      {
        vecROVStoreVals[iR] = pOutputs.dbleArray_[iR*Ysize];
        for (ii = 0; ii < nInputs; ii++)
          vecROVStoreGrads[iR*nInputs+ii] =
                              pOutputs.dbleArray_[iR*Ysize+ii+1];
      }
      vecROVStoreEigens.setLength(nROVMax);
      for (iR = 0; iR < iRov; iR++)
        vecROVStoreEigens[iR] = pOutputs.dbleArray_[(iR+1)*Ysize-1];
      vecYstore.setLength(nROVMax*Ysize);
      for (iR = 0; iR < iRov*Ysize; iR++)
        vecYstore[iR] = pOutputs.dbleArray_[iR];
      delete psIO;

      //**/ read sample file
      printAsterisks(PL_INFO, 0);
      printf("Enter the corresponding PsuadeGradRSM.sample file name : ");
      scanf("%s", fileName);
      fileIn = fopen(fileName, "r");
      if (fileIn == NULL)
      {
        printOutTS(PL_INFO, "Sample file name %s found.\n", fileName);
        return -1;
      }
      fclose(fileIn);
      psIO = new PsuadeData();
      status = psIO->readPsuadeFile(fileName);
      if (status != 0)
      {
        printOutTS(PL_ERROR, 
            "ERROR: Problem reading sample file %s.\n", fileName);
        return -1;
      }
      psIO->getParameter("input_ninputs", pPtr);
      if (pPtr.intData_ != nInputs)
      {
        printOutTS(PL_INFO, 
            "nInputs in file %s does not match with current file.\n",
            fileName);
        return -1;
      }
      psIO->getParameter("output_noutputs", pPtr);
      if (pPtr.intData_ != nOutputs)
      {
        printOutTS(PL_INFO, 
            "nOutputs in file %s does not match with current file.\n",
            fileName);
        return -1;
      }
      psIO->getParameter("method_nsamples", pPtr);
      nSamples = pPtr.intData_;
      if (nSamples < iRov)
      {
        printOutTS(PL_ERROR, 
             "ERROR: sample in file %s too small (should be > %d)\n",
                fileName, iRov);
        return -1;
      }
      psIO->getParameter("input_sample", pInputs2);
      vecSamInps.setLength(nSamples*nInputs);
      for (iR = 0; iR < nSamples*nInputs; iR++)
        vecSamInps[iR] = pInputs2.dbleArray_[iR];
      psIO->getParameter("output_sample", pOutputs2);
      vecSamROVVals.setLength(nSamples);
      for (iR = 0; iR < nSamples; iR++)
        vecSamROVVals[iR] = pOutputs2.dbleArray_[iR];
      vecSamROVErrs.setLength(nSamples);
      for (iR = 0; iR < nSamples; iR++)
        vecSamROVErrs[iR] = PSUADE_UNDEFINED;
      vecSamROVInds.setLength(nSamples);
      for (iR = 0; iR < nSamples; iR++) vecSamROVInds[iR] = -1;
      delete psIO;

      //**/ put sample data into the ROVs
      for (iR = 0; iR < iRov; iR++)
      {
        YVal = vecROVStoreVals[iR];
        eigMax = vecROVStoreEigens[iR];
        for (iS = 0; iS < nSamples; iS++)
        {
          Ynew = YVal;
          dsum2 = 0.0;
          for (ii = 0; ii < nInputs; ii++)
          {
            diff = vecSamInps[iS*nInputs+ii] - 
                   vecROVStorePts[iR*nInputs+ii]; 
            Ynew += vecROVStoreGrads[iR*nInputs+ii] * diff;
            dsum2 += diff * diff;
          }
          error = 0.5 * PABS(eigMax) * dsum2;
          //**/ track the ROV that gives the smallest error
          if (error < vecSamROVErrs[iS])
          {
            vecSamROVInds[iS] = iR;
            vecSamROVVals[iS] = Ynew;
            vecSamROVErrs[iS] = error;
          }
            //**/ but error > threshold, mark it as not done
            if (error > anaThreshold) vecSamROVInds[iS] = -1;
        }
      }

      //**/ search for next evaluation point
      error = vecSamROVErrs[0];
      kk = 0;
      for (iS = 1; iS < nSamples; iS++)
      {
        if (vecSamROVErrs[iS] > error)
        {
          kk = iS;
          error = vecSamROVErrs[iS];
        }
      }
      for (ii = 0; ii < nInputs; ii++)
        vecCurrX[ii] = vecSamInps[kk*nInputs+ii];
      sampleIndex = -1;
      if (vecSamROVInds[kk] < 0) sampleIndex = kk;
      if (sampleIndex == -1)
      {
        printOutTS(PL_INFO, 
          "PSUADE adaptiveGradBased: the ROV sample is good to go.\n");
        return 0;
      }

      //**/ no refinement information
      refineSeps = new int[2];
      refineSeps[0] = 0;
      refineSeps[1] = nSamples;
      nRefineCurr = 0;
      nRefines = 0;
      printAsterisks(PL_INFO, 0);
      iRov--;
    }
  }
 
  //**/ ----------------------------------------------------------------
  //**/ create and refine a sample and then store it in a file 
  //**/ (vecSamInps, nSamples) will propagate
  //**/ ----------------------------------------------------------------
  if (nSamples == 0)
  {
    psuadeIO_->getParameter("method_nsamples", pPtr);
    initNSamples = nSamples = pPtr.intData_; 
    nRefineLocal = 0;
    while (nSamples < 128000)
    {
      nRefineLocal++;
      nSamples *= 2;
    }
    nSamples = initNSamples; 
    psuadeIO_->getParameter("method_nrefinements", pPtr);
    nRefines = pPtr.intData_;
    refineSeps = new int[nRefineLocal+2];
    refineSeps[0] = 0;
    refineSeps[1] = nSamples;
    nRefineCurr = 1;
    while (nRefineCurr <= nRefineLocal)
    {
      sampler_->refine(2, 0, 0.0, nSamples, NULL);
      nSamples = sampler_->getNumSamples();
      refineSeps[nRefineCurr+1] = nSamples;
      printOutTS(PL_INFO, "PSUADE adaptiveGradBased: refinement %d (%d)\n",
                 nRefineCurr, nSamples);
      nRefineCurr++;
    }
    nRefineCurr = 0;
    //cout << "PsuadeGradRSM: refinement done = " << nRefines
    //     << " (" << nSamples << ")\n";
    vecSamInps.setLength(nSamples*nInputs);
    vecSamROVVals.setLength(nSamples);
    vecStas.setLength(nSamples);
    sampler_->getSamples(nSamples,nInputs,nOutputs,
                 vecSamInps.getDVector(),vecSamROVVals.getDVector(),
                 vecStas.getIVector());
    for (ii = 0; ii < nROVMax; ii++) vecStas[ii] = 0;

    YNames.setNumStrings(nOutputs);
    for (ii = 0; ii < nOutputs; ii++)
    {
      snprintf(pString,100,"Y%d", ii+1);
      YNames.loadOneString(ii, pString);
    }
    XNames.setNumStrings(nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      snprintf(pString,100,"X%d", ii+1);
      XNames.loadOneString(ii, pString);
    }
    psIO = new PsuadeData();
    psIO->updateMethodSection(PSUADE_SAMP_MC,nSamples,1,0,0);
    psIO->updateInputSection(nSamples,nInputs,NULL,iLowerB,iUpperB,
            vecSamInps.getDVector(),XNames.getStrings(),NULL,
            NULL,NULL,NULL);
    psIO->updateOutputSection(nSamples,nOutputs,
                vecSamROVVals.getDVector(),
                vecStas.getIVector(),YNames.getStrings());
    psIO->writePsuadeFile("PsuadeGradBased.sample",0);
    delete psIO;

    //**/ -------------------------------------------------------------
    //**/ Initialize variables that will contain ROV information of sample
    //**/ vecSamROVInds[i] - store index of covering ROV for sample i
    //**/ vecSamROVVals[i] - approximation from covering ROV 
    //**/ vecSamROVErrs[i] - estimated error
    //**/ -------------------------------------------------------------
    vecSamROVInds.setLength(nSamples);
    for (iS = 0; iS < nSamples; iS++) vecSamROVInds[iS] = -1;
    vecSamROVErrs.setLength(nSamples);
    for (iS = 0; iS < nSamples; iS++) vecSamROVErrs[iS] = PSUADE_UNDEFINED;

    //**/ ----------------------------------------------------------------
    //**/ initialize ROV structure and counters
    //**/ ----------------------------------------------------------------
    vecROVStorePts.setLength(nROVMax*nInputs);
    vecROVStoreVals.setLength(nROVMax);
    vecROVStoreGrads.setLength(nROVMax*nInputs);
    vecROVStoreEigens.setLength(nROVMax);
    vecYstore.setLength(nROVMax*Ysize);
    iRov = 0;
    for (ii = 0; ii < nInputs; ii++) vecCurrX[ii] = vecSamInps[ii];
    sampleIndex = 0;
  }

  //**/ ----------------------------------------------------------------
  //**/ initialize first point and allocate memory for others
  //**/ ----------------------------------------------------------------
  vecStas.setLength(nROVMax);
  for (ii = 0; ii < nROVMax; ii++) vecStas[ii] = 1;

  YNames.setNumStrings(Ysize);
  for (ii = 0; ii < Ysize; ii++)
  {
    snprintf(pString,100,"Y%d", ii+1);
    YNames.loadOneString(ii, pString);
  }
  vecGrad.setLength(nInputs);
  vecHess.setLength(nInputs*nInputs);
  jobz = 'N';
  uplo = 'U';
  lwork = 3 * nInputs;
  vecWT.setLength(lwork);
  vecEigs.setLength(nInputs);

  funcIO = createFunctionInterface(psuadeIO_);
  vecRanges.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) 
    vecRanges[ii] = (iUpperB[ii] - iLowerB[ii]);
  
  //**/ ----------------------------------------------------------------
  //**/ loop until done
  //**/ ----------------------------------------------------------------
  error = PSUADE_UNDEFINED;
  if (nRefines > 0)
  {
    printAsterisks(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO, 
        "adaptiveGradBased: the first %d points will be evaluated.\n",
        initNSamples);
    printAsterisks(PL_INFO, 0);
    printAsterisks(PL_INFO, 0);
  }
  while (iRov < nROVMax && threshold >= anaThreshold)
  {
    iRov++;
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO, "ROV %4d = Sample %d\n", iRov, sampleIndex+1);

    //**/ -------------------------------------------------------------
    //**/ compute function value, gradients 
    //**/ -------------------------------------------------------------
    evaluateFull(nInputs, vecCurrX.getDVector(), funcIO, 
                 vecRanges.getDVector(), maxParallelJobs, &YVal, 
                 vecGrad.getDVector(), vecHess.getDVector());
    if (outputLevel_ > 2) 
    {
      for (ii = 0; ii < nInputs; ii++)
        printOutTS(PL_INFO, "ROV %4d: input %2d value    = %e\n",
                   iRov,ii+1,vecCurrX[ii]);
      printOutTS(PL_INFO, "ROV %4d: function value    = %e\n",iRov,YVal);
    }
    if (outputLevel_ > 3) 
    {
      for (ii = 0; ii < nInputs; ii++)
        printOutTS(PL_INFO, "ROV %4d: Gradient %3d      = %e\n",iRov,ii+1,
                   vecGrad[ii]);
      for (ii = 0; ii < nInputs; ii++)
        for (kk = 0; kk < nInputs; kk++)
          printOutTS(PL_INFO, "ROV %4d: Hessian (%3d,%3d) = %e\n",iRov,
                     ii+1,kk+1, vecHess[ii*nInputs+kk]);
    }

    //**/ -------------------------------------------------------------
    //**/ if below initNSamples,  function value, vecGrad 
    //**/ -------------------------------------------------------------
    vecSamROVInds[sampleIndex] = iRov;
    vecSamROVErrs[sampleIndex] = 0.0;
    vecSamROVVals[sampleIndex] = YVal;

    //**/ -------------------------------------------------------------
    //**/ compute absolute max eigenvalue from vecHess
    //**/ -------------------------------------------------------------
    dsyev_(&jobz,&uplo,&nInputs,vecHess.getDVector(),&nInputs,
           vecEigs.getDVector(),vecWT.getDVector(),&lwork,&info);
    if (info != 0)
    {
      printOutTS(PL_INFO, 
          "adaptiveGradBased INFO: dsyev returns a nonzero (%d).\n",
          info);
      exit(1);
    }
    eigMax = PABS(vecEigs[0]);
    for (ii = 1; ii < nInputs; ii++)
      if (PABS(vecEigs[ii]) > eigMax) eigMax = PABS(vecEigs[ii]);
    printOutTS(PL_INFO, "ROV %4d: max eigenvalue    = %e\n", iRov, eigMax);
    if (eigMax < minEigen)
    {
      eigMax = minEigen;
      printOutTS(PL_INFO, "ROV %4d: max value set to  = %e\n", iRov, eigMax);
    }

    //**/ -------------------------------------------------------------
    //**/ store ROV information
    //**/ -------------------------------------------------------------
    for (ii = 0; ii < nInputs; ii++)
    {
      vecROVStorePts[(iRov-1)*nInputs+ii] = vecCurrX[ii];
      vecROVStoreGrads[(iRov-1)*nInputs+ii] = vecGrad[ii];
    }
    vecROVStoreEigens[iRov-1] = eigMax;
    vecROVStoreVals[iRov-1] = YVal;

    //**/ -------------------------------------------------------------
    //**/ interpolation 
    //**/ -------------------------------------------------------------
    rovCnt = count = 0;
    for (iS = 0; iS < nSamples; iS++)
    {
      if (iS >= initNSamples || nRefines == 0) 
      {
        Ynew = YVal;
        dsum2 = 0.0;
        for (ii = 0; ii < nInputs; ii++)
        {
          diff = vecSamInps[iS*nInputs+ii] - vecCurrX[ii]; 
          Ynew += vecGrad[ii] * diff;
          dsum2 += diff * diff;
        }
        tolerance = 0.5 * dsum2 * eigMax;
        //**/ if tolerance is smaller, use the current ROV
        flag = 0;
        if (tolerance < vecSamROVErrs[iS])
        {
          vecSamROVInds[iS] = iRov - 1;
          vecSamROVVals[iS] = Ynew;
          vecSamROVErrs[iS] = tolerance;
          count++;
          flag = 1;
        }
        //**/ if tolerance is not sufficient, invalidate the ROV
        if (vecSamROVErrs[iS] >= threshold)
        {
          vecSamROVInds[iS] = - 1;
          if (flag == 1) count--;
        }
      }
      if (vecSamROVInds[iS] >= 0) rovCnt++;
    }
    printOutTS(PL_INFO, 
       "PSUADE adaptiveGradBased: region of validity limit not imposed.\n");
    printOutTS(PL_INFO, 
       "PSUADE adaptiveGradBased: current threshold      = %e\n",
       threshold);
    printOutTS(PL_INFO, 
       "PSUADE adaptiveGradBased: total number covered   = %d (%d)\n",
       rovCnt, nSamples);
    printOutTS(PL_INFO, 
       "PSUADE adaptiveGradBased: number this ROV covers = %d\n",
       count+1);

    //**/ -------------------------------------------------------------
    //**/ store ROV results
    //**/ -------------------------------------------------------------
    count = (iRov - 1) * Ysize;
    vecYstore[count++] = YVal;
    for (ii = 0; ii < nInputs; ii++) vecYstore[count++] = vecGrad[ii];
    for (ii = 0; ii < nInputs; ii++)
      for (kk = 0; kk <= ii; kk++)
        vecYstore[count++] = vecHess[ii*nInputs+kk];
    vecYstore[count++] = eigMax;
    psuadeIO_->updateMethodSection(-1,iRov,1,0,0);
    psuadeIO_->updateInputSection(iRov,nInputs,NULL,NULL,NULL,
               vecROVStorePts.getDVector(),NULL,NULL,NULL,NULL,NULL);
    psuadeIO_->updateOutputSection(iRov,Ysize,vecYstore.getDVector(),
                 vecStas.getIVector(),YNames.getStrings());
    psuadeIO_->writePsuadeFile("PsuadeGradBased.rov",0);

    //**/ -------------------------------------------------------------
    //**/ pick next ROV: The next ROV will be selected in order
    //**/ -------------------------------------------------------------
    for (iR = nRefineCurr; iR <= nRefineLocal; iR++)
    {
      //**/ see if all current refinement has been evaluated
      //**/ if so, advance to next refinement
      for (kk = refineSeps[iR]; kk < refineSeps[iR+1]; kk++)
      {
        rovID = vecSamROVInds[kk];
        if (rovID < 0) break;
      }
      sampleIndex = -1;
      if (kk < refineSeps[iR+1])
      {
        errMax = 0.0;
        for (kk = refineSeps[iR]; kk < refineSeps[iR+1]; kk++)
        {
          dtemp = vecSamROVErrs[kk];
          if (dtemp > errMax)
          {
            errMax = dtemp;
            sampleIndex = kk;
          }
        }
      }
      //**/ if found, move on
      if (sampleIndex >= 0) break;
      //**/ otherwise, look for candidate in the next level
      //**/ printOutTS(PL_INFO, 
      //**/    "PsuadeGradRSM: advance to refinement level %d (%d)\n",
      //       nRefineCurr, nRefines);
      nRefineCurr++;
    }

    //**/ -------------------------------------------------------------
    //**/ If every sample is covered, reduce threshold and re-do
    //**/ -------------------------------------------------------------
    if (sampleIndex < 0) 
    {
      error = 0.0;
      for (iS = 1; iS < nSamples; iS++)
        error += pow(vecSamROVErrs[iS], 2.0);
      error = sqrt(error / (double) nSamples);
      printOutTS(PL_INFO, 
           "PSUADE adaptiveGradBased: current mean error = %e\n", error);

      //**/ ----------------------------------------------------------
      //**/ compute moments 
      //**/ ----------------------------------------------------------
      mean = 0.0;
      for (iS = 0; iS < nSamples; iS++) mean += vecSamROVVals[iS];
      mean /= (double) nSamples;
      stdev = 0.0;
      for (iS = 0; iS < nSamples; iS++)
        stdev += pow(vecSamROVVals[iS]-mean, 2.0);
      stdev /= (double) nSamples;
      stdev = sqrt(stdev);
      printOutTS(PL_INFO, 
          "PSUADE adaptiveGradBased: Current sample mean   = %16.8e\n",
          mean);
      printOutTS(PL_INFO, 
          "PSUADE adaptiveGradBased: Current std deviation = %16.8e\n",
          stdev);
      
      while (threshold > anaThreshold)
      {
        threshold *= threshDec;
        printOutTS(PL_INFO, 
             "PSUADE adaptiveGradBased: threshold down to  = %e\n",
             threshold);
        if (threshold <= anaThreshold) threshold = anaThreshold;
        //**/ reset the ones that would not have passed the new threshold
        for (iS = 0; iS < nSamples; iS++)
        {
          kk = vecSamROVInds[iS];
          if (kk >= 0)
          {
            Ynew = vecROVStoreVals[kk];
            eigMax = vecROVStoreEigens[kk]; 
            dsum2 = 0.0;
            for (ii = 0; ii < nInputs; ii++)
            {
              diff = vecSamInps[iS*nInputs+ii] - 
                     vecROVStorePts[kk*nInputs+ii]; 
              Ynew += vecROVStoreGrads[kk*nInputs+ii] * diff;
              dsum2 += diff * diff;
            }
            tolerance = 0.5 * dsum2 * eigMax;
            //**/ if current ROV not good enough, search to see if
            //**/ other ROV is good enough. If not, invalidate it
            if (tolerance > threshold)
            {
              vecSamROVErrs[iS] = PSUADE_UNDEFINED;
              for (iR = 0; iR < iRov; iR++)
              {
                eig2 = vecROVStoreEigens[iR]; 
                dsum2 = 0.0;
                Ynew = vecROVStoreVals[iR];
                for (ii = 0; ii < nInputs; ii++)
                {
                  diff = vecSamInps[iS*nInputs+ii] - 
                         vecROVStorePts[iR*nInputs+ii]; 
                  Ynew += vecROVStoreGrads[iR*nInputs+ii] * diff;
                  dsum2 += diff * diff;
                }
                tol2 = 0.5 * dsum2 * eig2;
                if (tol2 < vecSamROVErrs[iS])
                {
                  vecSamROVErrs[iS] = tol2;
                  vecSamROVInds[iS] = iR;
                  vecSamROVVals[iS] = Ynew;
                }
              }
              if (vecSamROVErrs[iS] > threshold) vecSamROVInds[iS] = -1;
            }
          }
        }
        //**/ start from beginning and look for next point
        nRefineCurr = 0;
        for (iR = nRefineCurr; iR <= nRefineLocal; iR++)
        {
          //**/ see if all current refinement has been evaluated
          //**/ if so, advance to next refinement
          for (kk = refineSeps[iR]; kk < refineSeps[iR+1]; kk++)
          {
            rovID = vecSamROVInds[kk];
            if (rovID < 0) break;
          }
          sampleIndex = -1;
          if (kk < refineSeps[iR+1])
          {
            errMax = 0.0;
            for (kk = refineSeps[iR]; kk < refineSeps[iR+1]; kk++)
            {
              dtemp = vecSamROVErrs[kk];
              if (dtemp > errMax)
              {
                errMax = dtemp;
                sampleIndex = kk;
              }
            }
          }
          if (sampleIndex >= 0) break;
          nRefineCurr++;
        }
        //**/ if next point is found, hold off reducing threshold and 
        //**/ hold off moving to the next refinement level
        if (sampleIndex >= 0) break;
      }
    }
    if (sampleIndex < 0)
    {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO, 
          "Cannot find next ROV to evaluate (%d).\n",sampleIndex);
      break;
    }
    //**/ if less than initNSamples, next point is clear
    if (iRov < initNSamples && nRefines > 0) sampleIndex = iRov;
    printOutTS(PL_INFO, 
         "PSUADE adaptiveGradBased: next candidate ROV = %d\n",
         sampleIndex+1);
    for (ii = 0; ii < nInputs; ii++)
      vecCurrX[ii] = vecSamInps[sampleIndex*nInputs+ii];
    printAsterisks(PL_INFO, 0);
  }

  //**/ -------------------------------------------------------------------
  //**/ final evaluation
  //**/ -------------------------------------------------------------------
  for (iS = 0; iS < nSamples; iS++)
  {
    vecSamROVErrs[iS] = PSUADE_UNDEFINED;
    vecSamROVInds[iS] = -1;
    for (iR = 0; iR < iRov; iR++)
    {
      eig2 = vecROVStoreEigens[iR]; 
      dsum2 = 0.0;
      Ynew = vecROVStoreVals[iR];
      for (ii = 0; ii < nInputs; ii++)
      {
        diff = vecSamInps[iS*nInputs+ii] - 
               vecROVStorePts[iR*nInputs+ii]; 
        Ynew += vecROVStoreGrads[iR*nInputs+ii] * diff;
        dsum2 += diff * diff;
      }
      tol2 = 0.5 * dsum2 * eig2;
      if (tol2 < vecSamROVErrs[iS])
      {
        vecSamROVErrs[iS] = tol2;
        vecSamROVInds[iS] = iR;
        vecSamROVVals[iS] = Ynew;
      }
      if (vecSamROVErrs[iS] >= anaThreshold) vecSamROVInds[iS] = -1;
    }
    if (outputLevel_ > 4)
      printOutTS(PL_INFO, "Sample %d's (%d)index = %d, %e (%e)\n", iS+1,
                vecSamROVInds[iS],nSamples,vecSamROVErrs[iS],anaThreshold);
    if (vecSamROVInds[iS] == -1)
    {
      printOutTS(PL_ERROR, 
          "PSUADE adaptiveGradBased: fatal ERROR. Consult developers.\n");
      exit(1);
    }
  }

  //**/ -------------------------------------------------------------------
  //**/ compute moments 
  //**/ -------------------------------------------------------------------
  mean = 0.0;
  for (iS = 0; iS < nSamples; iS++) mean += vecSamROVVals[iS];
  mean /= (double) nSamples;
  stdev = 0.0;
  for (iS = 0; iS < nSamples; iS++)
    stdev += pow(vecSamROVVals[iS]-mean, 2.0);
  stdev /= (double) nSamples;
  stdev = sqrt(stdev);
  printOutTS(PL_INFO, 
       "PSUADE adaptiveGradBased: Final sample mean     = %16.8e\n", mean);
  printOutTS(PL_INFO, 
       "PSUADE adaptiveGradBased: Final std deviation   = %16.8e\n", stdev);
  printOutTS(PL_INFO, 
       "PSUADE adaptiveGradBased: Total number of ROVs  = %d\n", iRov);
  printOutTS(PL_INFO, 
       "PSUADE adaptiveGradBased: Total number of runs  = %d\n",
       (Ysize-1)*iRov);
  if (iRov > 0)
  {
    printOutTS(PL_INFO, 
         "PSUADE adaptiveGradBased: ROV info is in PsuadeGradBased.rov.\n");
  }

  //**/ -------------------------------------------------------------------
  //**/ clean up 
  //**/ -------------------------------------------------------------------
  if (sampler_ != NULL)
  {
    delete sampler_;
    sampler_ = NULL;
  }
  delete funcIO;
  delete [] refineSeps;
  return 0;
}

// ************************************************************************
// evaluate the function, its gradients, and Hessian
// ------------------------------------------------------------------------
int PsuadeBase::evaluateFull(int nInputs, double *currentPt, 
                             FunctionInterface *funcIO,
                             double *ranges, int maxJobs, double *YVal, 
                             double *gradients, double *hessian)
{
  int     launchInterval=1, ii, nJobs, iS, iS2, nOutputs=1, *sampleStates;
  int     jobCnt;
  double  *sampleInputs, *sampleOutputs, hstep, hstep2;
  JobCntl jcntl;

  //**/ -------------------------------------------------------------------
  //**/ set up function interface
  //**/ -------------------------------------------------------------------
  if (maxJobs > 1) funcIO->setAsynchronousMode();
  funcIO->setLaunchInterval(launchInterval);

  //**/ -------------------------------------------------------------------
  //**/ set up the sample 
  //**/ -------------------------------------------------------------------
  nJobs = nInputs+1+(nInputs+1)*nInputs/2;
  sampleStates = new int[nJobs];
  sampleInputs = new double[nJobs*nInputs];
  sampleOutputs = new double[nJobs];
  for (iS = 0; iS < nJobs; iS++) sampleStates[iS] = 0;
  for (ii = 0; ii < nInputs; ii++) sampleInputs[ii] = currentPt[ii];
  jobCnt = 1;
  if (gradients != NULL)
  {
    for (iS = 0; iS < nInputs; iS++)
    {
      hstep = 1.0e-5 * ranges[iS];
      for (ii = 0; ii < nInputs; ii++)
        sampleInputs[jobCnt*nInputs+ii] = currentPt[ii];
      sampleInputs[jobCnt*nInputs+iS] = currentPt[iS] + hstep;
      jobCnt++;
    }
  }
  if (gradients != NULL && hessian != NULL)
  {
    for (iS = 0; iS < nInputs; iS++)
    {
      hstep = 1.0e-5 * ranges[iS];
      for (iS2 = 0; iS2 <= iS; iS2++)
      {
        hstep2 = 1.0e-5 * ranges[iS2];
        for (ii = 0; ii < nInputs; ii++)
          sampleInputs[jobCnt*nInputs+ii] = 
                                sampleInputs[(1+iS)*nInputs+ii];
        sampleInputs[jobCnt*nInputs+iS2] += hstep2; 
        jobCnt++;
      }
    }
  }

  //**/ -------------------------------------------------------------------
  //**/ run the sample
  //**/ -------------------------------------------------------------------
  jcntl.setMaxParallelJobs(maxJobs);
  //jcntl.setOutputLevel(outputLevel_);
  jcntl.loadFuncIO(funcIO);
  jcntl.loadSample(jobCnt,nInputs,nOutputs,sampleInputs,sampleOutputs,
                   sampleStates);
  jcntl.execute();
  jcntl.getSampleOutputs(jobCnt, nOutputs, sampleOutputs); 

  //**/ -------------------------------------------------------------------
  //**/ fetch outputs
  //**/ -------------------------------------------------------------------
  (*YVal) = sampleOutputs[0]; 
  if (gradients != NULL)
  {
    for (iS = 0; iS < nInputs; iS++)
    {
      hstep = 1.0e-5 * ranges[iS];
      gradients[iS] = (sampleOutputs[1+iS] - sampleOutputs[0]) / hstep;
    }
    jobCnt = nInputs + 1;
  }
  if (gradients != NULL && hessian != NULL)
  {
    for (iS = 0; iS < nInputs; iS++)
    {
      hstep = 1.0e-5 * ranges[iS];
      for (iS2 = 0; iS2 <= iS; iS2++)
      {
        hstep2 = 1.0e-5 * ranges[iS2];
        hessian[iS*nInputs+iS2] = 
            ((sampleOutputs[jobCnt]-sampleOutputs[iS2+1]) - 
             (sampleOutputs[iS+1]-sampleOutputs[0]))/(hstep*hstep2);
        hessian[iS2*nInputs+iS] = hessian[iS*nInputs+iS2]; 
        jobCnt++;
      }
    }
  }
  delete [] sampleInputs;
  delete [] sampleOutputs;
  delete [] sampleStates;
  return 0;
}

