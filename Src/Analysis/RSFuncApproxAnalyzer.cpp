// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team. 
// All rights reserved.
//
// Please see the COPYRIGHT and LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public License.
//
// PSUADE is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free 
// Software Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU Lesser
// General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Functions for the class RSFuncApproxAnalyzer  
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"
#include "FuncApprox.h"
#include "Regression.h"
#include "UserRegression.h"
#include "RSFuncApproxAnalyzer.h"
#include "PrintingTS.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
RSFuncApproxAnalyzer::RSFuncApproxAnalyzer() : Analyzer()
{
  setName("RSFA");
  //**/ default is MARS
  rsType_ = PSUADE_RS_MARS;
  useCV_ = 0;
  numCVGroups_ = 10;
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
RSFuncApproxAnalyzer::~RSFuncApproxAnalyzer() 
{ 
} 

// ************************************************************************ 
// perform analysis 
// ------------------------------------------------------------------------
double RSFuncApproxAnalyzer::analyze(aData &adata)
{
  int    ig, ss, ss2, iI, iL, nLast, nPtsPerDim=64, status;
  int    count, nSubSamples, testFlag, iOne=1;
  double ddata, ymax, ymin, retdata=0, sdata, ymean, yvar, ydiff, ssum;
  double cvErr1, cvErr1s, cvErr2, cvErr2s,cvMax, cvMaxs, cvMaxBase, cvMaxs2;
  double cvMaxBases, CVMaxBases, maxBase, maxBases, sumErr11, sumErr11s;
  double CVErr2, CVErr2s, CVErr1, CVErr1s, CVMax, CVMaxs, CVMaxBase;
  double sumErr1, sumErr1s, sumErr2, sumErr2s, maxErr, maxErrs;
  double *arrayXX, maxErrs2, CVMaxs2;
  char   winput1[500], winput2[500], dataFile[500], errFile[500];
  char   pString[500], *targv[3], *cString;
  FILE   *fpData, *fpErr;
  psVector   VecYLocal,VecYT,VecSigmas,VecS,VecE,VecW;
  psVector   VecXX, VecYY, VecD, VecWgts;
  psIVector  IVec1, IVec2;
  FuncApprox *faPtr;

  //**/ ---------------------------------------------------------------
  //**/ extract sample information
  //**/ ---------------------------------------------------------------
  int printLevel = adata.printLevel_;
  int nInputs   = adata.nInputs_;
  int nOutputs  = adata.nOutputs_;
  int nSamples  = adata.nSamples_;
  double *lower = adata.iLowerB_;
  double *upper = adata.iUpperB_;
  double *arrayX = adata.sampleInputs_;
  double *arrayY = adata.sampleOutputs_;
  int    *arrayS = adata.sampleStates_;
  int outputID  = adata.outputID_;
  int nLevels   = adata.currRefineLevel_;
  int *levelSeps = adata.refineSeparators_;
  int wgtID     = adata.regWgtID_;

  //**/ ---------------------------------------------------------------
  //**/ checking for non-uniform PDF and give warning
  //**/ ---------------------------------------------------------------
  if (adata.inputPDFs_ != NULL)
  {
    count = 0;
    for (iI = 0; iI < nInputs; iI++) count += adata.inputPDFs_[iI];
    if (count > 0)
    {
      printOutTS(PL_INFO,
           "RSAnalysis INFO: some inputs have non-uniform PDFs,\n");
      printOutTS(PL_INFO,
           "    but they are not to be used in this analysis.\n");
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ error checking and diagnostics
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nOutputs <= 0)
  {
     printOutTS(PL_ERROR, "RSAnalyzer ERROR: invalid arguments.\n");
     printOutTS(PL_ERROR, "   nInputs  = %d\n", nInputs);
     printOutTS(PL_ERROR, "   nOutputs = %d\n", nOutputs);
     printOutTS(PL_ERROR, "   nSamples = %d\n", nSamples);
     return PSUADE_UNDEFINED;
  } 
  if (nSamples <= 1)
  {
     printOutTS(PL_ERROR, "RSAnalyzer ERROR: nSamples too small.\n");
     printOutTS(PL_ERROR, "   nSamples = %d\n", nSamples);
     return PSUADE_UNDEFINED;
  } 
  status = 0;
  for (ss = 0; ss < nSamples; ss++)
     if (arrayY[nOutputs*ss+outputID] > 0.9*PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR,"RSAnalysis ERROR: Some outputs are undefined.\n");
    printOutTS(PL_ERROR,"    Prune the undefined sample points first.\n");
    return PSUADE_UNDEFINED;
  }
  if (arrayS != NULL)
  {
    status = 0;
    for (ss = 0; ss < nSamples; ss++) if (arrayS[ss] != 1) status++;
    if (status > 0)
    {
      printOutTS(PL_ERROR,
         "RSAnalysis WARNING: %d sample points are invalid (status!=1).\n",
         status);
      return PSUADE_UNDEFINED;
    }
  }
  //**/ print response surface type to be used for analysis
  if (printLevel > 0)
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO, "Response surface method to be used: ");
    printThisFA(rsType_);
    printEquals(PL_INFO, 0);
  }
  if (rsType_ == PSUADE_RS_REGRGL) nLevels = 1;
  if (rsType_ == PSUADE_RS_REGRGL && nOutputs != nInputs+1) 
  {
    printOutTS(PL_ERROR,
       "RSAnalysis ERROR: for gradient-based RS, nOutputs=nInputs+1\n");
    printOutTS(PL_ERROR,
       "  Your loaded sample has nInputs/nOutputs = %d/%d\n",nInputs,
       nOutputs);
    return PSUADE_UNDEFINED;
  }
  if (rsType_ == PSUADE_RS_REGRGL && outputID != 0) 
  {
    printOutTS(PL_WARN,
       "RSAnalysis ERROR: for gradient-based RS, the output ID to be\n");
    printOutTS(PL_WARN,
       "  analyzed must be 1. (You set it to %d)\n",outputID+1);
    printOutTS(PL_WARN,"  It is now reset to be 1.\n");
    outputID = 0;
  }
  
  //**/ ---------------------------------------------------------------
  //**/ copy the selected output data to a new array and find max
  //**/ ---------------------------------------------------------------
  VecYLocal.setLength(nSamples);
  for (ss = 0; ss < nSamples; ss++) 
    VecYLocal[ss] = arrayY[ss*nOutputs+outputID];
  ymax = VecYLocal.max();
  ymin = VecYLocal.min();
  ydiff = ymax - ymin;
  printOutTS(PL_INFO,"RSA: Output ID = %d\n", outputID+1);
  printOutTS(PL_INFO,
     "RSA: Output Maximum/Minimum = %14.6e %14.6e\n",ymax,ymin);
  printOutTS(PL_INFO,
     "INFO: Set printlevel higher (1-4) to display more information.\n");
  printOutTS(PL_INFO,
     "INFO: Turn on ana_expert mode for interpolation error graphics.\n");
  if (ymax == PSUADE_UNDEFINED)
  {
    printOutTS(PL_ERROR,"RSAnalyzer ERROR: some outputs are undefined.\n");
    printOutTS(PL_ERROR,"           Prune them first before analysis.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ---------------------------------------------------------------
  //**/ generate response surfaces for every levels
  //**/ (This is done so that users can check RS with different 
  //**/  sampling resolution. This feature is not exercised in the
  //**   current implement - never called with levelSeps != NULL)
  //**/ ---------------------------------------------------------------
  VecYT.setLength(nSamples);
  for (iL = 0; iL < nLevels; iL++)
  {
    nLast = nSamples;
    if (levelSeps != NULL) nLast = levelSeps[iL];

    //**/ ------------------------------------------------------------
    //**/ create response surface
    //**/ ------------------------------------------------------------
    faPtr = genFA(rsType_, nInputs, iOne, nLast);
    if (faPtr == NULL)
    {
      printOutTS(PL_ERROR,
         "RSFAnalyzer ERROR: failed to create response surface.\n");
      return 1.0;
    }
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(lower, upper);
    if (iL == nLevels-1) faPtr->setOutputLevel(printLevel);

    //**/ users can select one of the outputs to be regression weights
    if (wgtID >= 0 && wgtID < nOutputs)
    {
      VecWgts.setLength(nLast);
      for (ss = 0; ss < nLast; ss++) 
        VecWgts[ss] = arrayY[ss*nOutputs+wgtID];
      faPtr->loadWeights(nLast, VecWgts.getDVector());
    }
    if (rsType_ == PSUADE_RS_REGRGL)
    {
      VecD.setLength(nSamples*nInputs);
      for (ss = 0; ss < nSamples; ss++)
        for (iI = 0; iI < nInputs; iI++)
          VecD[ss*nInputs+iI] = arrayY[ss*nOutputs+iI+1];
      strcpy(pString, "deriv_sample");
      targv[0] = (char *) pString;
      int total = nSamples * nInputs;
      targv[1] = (char *) &total;
      targv[2] = (char *) VecD.getDVector();
      count = 2;
      faPtr->setParams(count, targv);
    }

    //**/ temporarily turn off CodeGen
    if (iL != nLevels-1) psConfig_.RSCodeGenSaveAndReset(); 
    status = faPtr->initialize(arrayX, VecYLocal.getDVector());
    if (iL != nLevels-1) psConfig_.RSCodeGenRestore();
    if (status != 0)
    {
      printOutTS(PL_ERROR,
           "RSAnalysis ERROR: something wrong in FA initialize.\n");
      delete faPtr;
      return PSUADE_UNDEFINED;
    }

    //**/ ------------------------------------------------------------
    //**/ check RS data against the training data .
    //**/ ------------------------------------------------------------
    sumErr1  = sumErr2 = maxErr = sumErr1s = sumErr2s = maxErrs = 0.0;
    sumErr11 = sumErr11s = maxErrs2 = 0.0;
    maxBase = maxBases = ymean = yvar = 0.0;
    faPtr->evaluatePoint(nLast, arrayX, VecYT.getDVector());
    if (printLevel > 2) printf("RSFA resubstitution test: \n");
    for (ss = 0; ss < nLast; ss++)
    {
      ddata = PABS(VecYT[ss] - VecYLocal[ss]);
      if (printLevel > 2) 
        printf("Sample %7d : true, predicted = %14.6e %14.6e\n",ss+1,
               VecYLocal[ss], VecYT[ss]);
      if (VecYLocal[ss] != 0.0) sdata = ddata / PABS(VecYLocal[ss]);
      else                      sdata = ddata;
      sumErr1   += ddata;
      sumErr1s  += sdata;
      sumErr11  += (VecYT[ss] - VecYLocal[ss]);
      if (VecYLocal[ss] != 0.0)
           sumErr11s += (VecYT[ss] - VecYLocal[ss])/PABS(VecYLocal[ss]);
      else sumErr11s += (VecYT[ss] - VecYLocal[ss]);
      if (ddata > maxErr) 
      {
        maxErr = ddata;
        maxBase = PABS(VecYLocal[ss]);
      }
      if (sdata > maxErrs)
      {
        maxErrs = sdata; 
        maxBases = PABS(VecYLocal[ss]);
      }
      if (ydiff > 0 && (ddata/ydiff > maxErrs2)) 
      {
        maxErrs2 = ddata / ydiff;
      }
      sumErr2  += (ddata * ddata);
      sumErr2s += (sdata * sdata);
      ymean += VecYLocal[ss]; 
    }
    ymean /= (double) nLast;
    for (ss = 0; ss < nLast; ss++)
      yvar += (VecYLocal[ss] - ymean) * (VecYLocal[ss] - ymean);
    sumErr1   = sumErr1 / (double) nLast;
    sumErr1s  = sumErr1s / (double) nLast;
    sumErr11  = sumErr11 / (double) nLast;
    sumErr11s = sumErr11s / (double) nLast;
    sumErr2  = sqrt(sumErr2 / (double) nLast);
    sumErr2s = sqrt(sumErr2s / (double) nLast);
    retdata = sumErr1;
    if (printLevel > 2 || iL == nLevels-1)
    {
      printOutTS(PL_INFO,
         "RSAnalysis: L %2d: interpolation error on training set \n",iL+1);
      printOutTS(PL_INFO,
         "             avg error far from 0 ==> systematic bias.\n");
      printOutTS(PL_INFO,
         "             rms error large      ==> average   error large.\n");
      printOutTS(PL_INFO,
         "             max error large      ==> pointwise error large.\n");
      printOutTS(PL_INFO,
         "             max error scaled by (ymax-ymin) large\n");
      printOutTS(PL_INFO,
         "                 ==> fitting probably not good.\n");
      printOutTS(PL_INFO,
         "             R-square may not always be a reliable measure.\n");
      //**/printOutTS(PL_INFO, 
      //**/     "  L1n error = %11.3e (unscaled), %9.3e(scaled)\n",
      //**/     sumErr1,sumErr1s);
      printOutTS(PL_INFO,"  avg error   = %11.3e (unscaled)\n", sumErr11);
      printOutTS(PL_INFO,"  avg error   = %11.3e (scaled)\n", sumErr11s);
      printOutTS(PL_INFO,"  rms error   = %11.3e (unscaled)\n", sumErr2);
      printOutTS(PL_INFO,"  rms error   = %11.3e (scaled)\n", sumErr2s);
      if (maxBase > 0)
        printOutTS(PL_INFO,
             "  max error   = %11.3e (unscaled, BASE=%9.3e)\n",
             maxErr, maxBase);
      else
        printOutTS(PL_INFO,"  max error   = %11.3e (unscaled)\n", maxErr);
      if (maxBases > 0)
        printOutTS(PL_INFO,
             "  max error   = %11.3e (  scaled, BASE=%9.3e)\n",
             maxErrs, maxBases);
      else
        printOutTS(PL_INFO, "  max error   = %11.3e (  scaled)\n",maxErrs);
      if (ydiff > 0)
        printOutTS(PL_INFO,
           "  max error   = %11.3e (  scaled by (ymax-ymin))\n",
           maxErrs2);
      printOutTS(PL_INFO,
           "  R-square    = %16.8e\n",1.0-sumErr2*sumErr2*nLast/yvar);
      printOutTS(PL_INFO,
           "Based on %d training points (total=%d).\n",nLast,nSamples);
    }

    //**/ ---------------------------------------------------------------
    //**/ generate matlab/scilab file (at the last level)
    //**/ ---------------------------------------------------------------
    if (psConfig_.AnaExpertModeIsOn() && iL == nLevels-1) 
    {
      psVector VecX;
      VecX.load(nSamples*nInputs, arrayX);
      retdata = genTrainingErrPlot(faPtr, VecX, VecYLocal, VecYT);
    }
    if (faPtr != NULL) delete faPtr;
  }

  //**/ ---------------------------------------------------------------
  //**/ setting up for cross validation
  //**/ ---------------------------------------------------------------
  nSubSamples = nSamples / numCVGroups_;
  if (rsType_ == PSUADE_RS_SPLINES || rsType_ == PSUADE_RS_REGSG)
  {
    printOutTS(PL_INFO,"Cross validation (CV) cannot be performed on\n");
    printOutTS(PL_INFO,"    splines and sparse grid response surface.\n");
  }
  else if (rsType_ == PSUADE_RS_REGRGL)
  {
    printOutTS(PL_INFO,"Cross validation (CV) not supported on\n");
    printOutTS(PL_INFO,"    gradient-based regression method.\n");
  }
  else
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"Next, you will be asked whether to do ");
    printOutTS(PL_INFO,"cross validation or not. Since\n");
    printOutTS(PL_INFO,"cross validation iterates as many ");
    printOutTS(PL_INFO,"times as the number of groups. The\n");
    printOutTS(PL_INFO,"rs_expert mode will be turned off. ");
    printOutTS(PL_INFO,"To change the default parameters\n");
    printOutTS(PL_INFO,"for different response surface, you ");
    printOutTS(PL_INFO,"will need to exit, create a config\n");
    printOutTS(PL_INFO,"file (use genconfigfile in command ");
    printOutTS(PL_INFO,"line mode), and set config option\n");
    printOutTS(PL_INFO, " in your data file.\n");
    printDashes(PL_INFO,0);
    snprintf(pString,100,"Perform cross validation ? (y or n) ");
    getString(pString, winput1);
    if (winput1[0] == 'y')
    {
      useCV_ = 1;
      snprintf(pString,100,
          "Enter the number of groups to validate : (2 - %d) ",
          nSamples);
      ss = getInt(1, nSamples, pString);
      printOutTS(PL_INFO, "RSFA: number of CV groups = %d\n",ss);
      numCVGroups_ = ss;
      nSubSamples = nSamples / numCVGroups_;
      if (1.5 * nSubSamples * numCVGroups_ < nSamples)
      {
        nSubSamples++;
        printOutTS(PL_INFO,
             "      Each CV group has <= %d sample points\n",
             nSubSamples);
        ss = nSamples / nSubSamples;
        if (ss * nSubSamples < nSamples) ss++;
        if (ss != numCVGroups_)
        {
          numCVGroups_ = ss;
          printOutTS(PL_INFO,
             "      number of CV group revised to %d\n",numCVGroups_);
        }
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ cross validation
  //**/ ---------------------------------------------------------------
  if (useCV_ == 1)
  {
    //**/ ---------------------------------------------------------
    //**/ setting up response surfaces
    //**/ (the reason for setting up multiple faPtr is because of
    //**/ open-mp support)
    //**/ ---------------------------------------------------------
    printOutTS(PL_INFO,
         "RSAnalysis: L %2d:cross validation (CV) begins...\n",nLevels);
    FuncApprox **faPtrs = new FuncApprox*[numCVGroups_];
    psConfig_.RSExpertModeSaveAndReset();
    psConfig_.InteractiveSaveAndReset();
    for (ig = 0; ig < numCVGroups_; ig++)
    {
      faPtrs[ig] = genFA(rsType_, nInputs, iOne, nSamples-nSubSamples);
      if (faPtrs[ig] == NULL)
      {
        printOutTS(PL_INFO,
             "RSAnalysis: genFA returned NULL in file %s line %d\n",
             __FILE__, __LINE__ );
        exit(1);
      }
      faPtrs[ig]->setNPtsPerDim(nPtsPerDim);
      faPtrs[ig]->setBounds(lower, upper);
      faPtrs[ig]->setOutputLevel(0);
    }
    psConfig_.RSExpertModeRestore();
    psConfig_.InteractiveRestore();

    //**/ ---------------------------------------------------------
    //**/ allocate space to store errors
    //**/ ---------------------------------------------------------
    adata.sampleErrors_ = new double[nSamples];
    for (ss = 0; ss < nSamples; ss++) adata.sampleErrors_[ss] = 0.0;

    //**/ ---------------------------------------------------------
    //**/ permute design matrix
    //**/ ---------------------------------------------------------
    IVec1.setLength(nSamples);
    VecXX.setLength(nSamples*nInputs);
    arrayXX = VecXX.getDVector();
    VecYY.setLength(nSamples);
    VecW.setLength(nSamples);
    IVec2.setLength(nSamples);

    snprintf(pString,100,"Random selection of leave-out groups ? (y or n) ");
    getString(pString, winput1);
    if (winput1[0] == 'y')
      generateRandomIvector(nSamples, IVec1.getIVector());
    else
      for (ss = 0; ss < nSamples; ss++) IVec1[ss] = ss;

    for (iI = 0; iI < nInputs; iI++)
    {
      for (ss = 0; ss < nSamples; ss++)
        arrayXX[IVec1[ss]*nInputs+iI] = arrayX[ss*nInputs+iI];
    }
    for (ss = 0; ss < nSamples; ss++) 
      VecYY[IVec1[ss]] = arrayY[ss*nOutputs+outputID];

    if (wgtID >= 0 && wgtID < nOutputs)
    {
      for (ss = 0; ss < nSamples; ss++)
        VecW[IVec1[ss]] = arrayY[ss*nOutputs+wgtID];
    }
    for (ss = 0; ss < nSamples; ss++) IVec2[IVec1[ss]] = ss;

    //**/ ---------------------------------------------------------
    //**/ iterate on groups
    //**/ ---------------------------------------------------------
    VecSigmas.setLength(nSamples);
    VecS.setLength(nSamples);
    VecE.setLength(nSamples);

    psVector VecX2, VecY2, VecT2, VecS2, VecW2;
    int    ig2, count2, iI2, status2, ss2, haveOMP=0;
    double ddata2;
    //**/ these response surfaces use external libraries that do not
    //**/ work well with OpenMP
#ifdef PSUADE_OMP
    haveOMP = 1;
#endif
    if (haveOMP == 1)
    {
      printOutTS(PL_INFO, "RSAnalysis INFO: OpenMP mode on\n");
#pragma omp parallel shared(VecSigmas,VecE,VecS) \
    private(ddata2,ig2,VecX2,VecY2,VecT2,VecS2,VecW2,count2,iI2,status2,ss2)
{
#pragma omp for
      for (ig2 = 0; ig2 < numCVGroups_; ig2++)
      {
        printOutTS(PL_INFO,
             "RSAnalysis:: L %2d: processing CV group %d (out of %d)\n",
             nLevels,ig2+1, numCVGroups_);

        //**/ gather the non-excluded data points
        VecX2.setLength(nSamples*nInputs);
        VecY2.setLength(nSamples);
        VecT2.setLength(nSamples);
        VecS2.setLength(nSamples);
        VecW2.setLength(nSamples);
        count2 = 0;
        for (ss2 = 0; ss2 < ig2*nSubSamples; ss2++)
        {
          for (iI2 = 0; iI2 < nInputs; iI2++)
            VecX2[count2*nInputs+iI2] = VecXX[ss2*nInputs+iI2];
          if (wgtID >= 0 && wgtID < nOutputs)
            VecW2[count2] = VecW[ss2*nOutputs+wgtID];
          VecY2[count2] = VecYY[ss2];
          count2++;
        }
        for (ss2 = (ig2+1)*nSubSamples; ss2 < nSamples; ss2++)
        {
          for (iI2 = 0; iI2 < nInputs; iI2++)
            VecX2[count2*nInputs+iI2] = VecXX[ss2*nInputs+iI2];
          if (wgtID >= 0 && wgtID < nOutputs)
            VecW2[count2] = VecW[ss2*nOutputs+wgtID];
          VecY2[count2] = VecYY[ss2];
          count2++;
        }
        if (wgtID >= 0 && wgtID < nOutputs)
          faPtrs[ig2]->loadWeights(nSamples-nSubSamples,
                                   VecW2.getDVector());

        //**/ generate response surface
        //**/ Note: the following response surface requires access to
        //**/       3rd party software that may not be run in parallel
        if (rsType_ != PSUADE_RS_GP1   && rsType_ != PSUADE_RS_SVM &&
            rsType_ != PSUADE_RS_MARS  && rsType_ != PSUADE_RS_MMARS &&
            rsType_ != PSUADE_RS_MARSB && rsType_ != PSUADE_RS_ACOSSO &&
            rsType_ != PSUADE_RS_KR    && rsType_ != PSUADE_RS_BSSANOVA &&
            rsType_ != PSUADE_RS_HKR   && rsType_ != PSUADE_RS_GP3)
          status2 = faPtrs[ig2]->initialize(VecX2.getDVector(), 
                                            VecY2.getDVector());
        else
        {
#pragma omp critical
          status2 = faPtrs[ig2]->initialize(VecX2.getDVector(), 
                                            VecY2.getDVector());
        }

        //if (status2 == -1) break;
        count2 = nSubSamples;
        if (((ig2+1)*nSubSamples) > nSamples) 
          count2 = nSamples - ig2*nSubSamples;
        if ((ig2+1 == numCVGroups_) && 
            numCVGroups_*nSubSamples < nSamples)
          count2 = nSamples - ig2*nSubSamples;
        faPtrs[ig2]->evaluatePointFuzzy(count2, 
                     &(arrayXX[ig2*nSubSamples*nInputs]), 
                     VecT2.getDVector(), VecS2.getDVector());
        //**/ re-shuffle
        for (ss2 = 0; ss2 < count2; ss2++)
        {
          ddata2 = VecT2[ss2] - VecYY[ig2*nSubSamples+ss2];
          VecE[IVec2[ig2*nSubSamples+ss2]] = ddata2;
          VecS[IVec2[ig2*nSubSamples+ss2]] = VecT2[ss2];
          VecSigmas[IVec2[ig2*nSubSamples+ss2]] = VecS2[ss2];
        }
        //**/ compute errors
        cvErr1 = cvErr2 = cvMax = 0.0;
        for (ss2 = 0; ss2 < count2; ss2++)
        {
          ddata = VecE[IVec2[ig2*nSubSamples+ss2]];
          cvErr1 += ddata;
          cvErr2 += (ddata * ddata);
          if (PABS(ddata) > cvMax) cvMax  = PABS(ddata);
        }
        cvErr1  = cvErr1 / (double) count2;
        cvErr2  = sqrt(cvErr2 / count2);
        printOutTS(PL_INFO,
             "RSA: CV error for sample group %5d = %11.3e (avg unscaled)\n",
             ig2+1, cvErr1);
        printOutTS(PL_INFO,
             "RSA: CV error for sample group %5d = %11.3e (rms unscaled)\n",
             ig2+1, cvErr2);
        printOutTS(PL_INFO,
             "RSA: CV error for sample group %5d = %11.3e (max unscaled)\n",
             ig2+1, cvMax);
        printOutTS(PL_INFO,
             "RSAnalysis:: L %2d: CV group %d processed\n",
             nLevels,ig2+1);
        delete faPtrs[ig2];
        faPtrs[ig2] = NULL;
      }
} /* omp */
    }
    else
    {
      //**/ this section is for non-OpenMP processing
      for (ig2 = 0; ig2 < numCVGroups_; ig2++)
      {
        printOutTS(PL_INFO,
           "RSAnalysis:: L %2d: processing CV group %d (out of %d)\n",
           nLevels,ig2+1, numCVGroups_);

        //**/ gather the non-excluded data points
        VecX2.setLength(nSamples*nInputs);
        VecY2.setLength(nSamples);
        VecT2.setLength(nSamples);
        VecS2.setLength(nSamples);
        VecW2.setLength(nSamples);
        count2 = 0;
        for (ss2 = 0; ss2 < ig2*nSubSamples; ss2++)
        {
          for (iI2 = 0; iI2 < nInputs; iI2++)
            VecX2[count2*nInputs+iI2] = VecXX[ss2*nInputs+iI2];
          if (wgtID >= 0 && wgtID < nOutputs)
            VecW2[count2] = VecW[ss2*nOutputs+wgtID];
          VecY2[count2] = VecYY[ss2];
          count2++;
        }
        for (ss2 = (ig2+1)*nSubSamples; ss2 < nSamples; ss2++)
        {
          for (iI2 = 0; iI2 < nInputs; iI2++)
            VecX2[count2*nInputs+iI2] = VecXX[ss2*nInputs+iI2];
          if (wgtID >= 0 && wgtID < nOutputs)
            VecW2[count2] = VecW[ss2*nOutputs+wgtID];
          VecY2[count2] = VecYY[ss2];
          count2++;
        }
        if (wgtID >= 0 && wgtID < nOutputs)
          faPtrs[ig2]->loadWeights(nSamples-nSubSamples,
                                   VecW2.getDVector());
  
        //**/ generate response surface
        status2 = faPtrs[ig2]->initialize(VecX2.getDVector(), 
                                          VecY2.getDVector());
        //if (status2 == -1) break;
        count2 = nSubSamples;
        if (((ig2+1)*nSubSamples) > nSamples) 
          count2 = nSamples - ig2*nSubSamples;
        if ((ig2+1 == numCVGroups_) && 
            numCVGroups_*nSubSamples < nSamples)
          count2 = nSamples - ig2*nSubSamples;
        faPtrs[ig2]->evaluatePointFuzzy(count2, 
                     &(arrayXX[ig2*nSubSamples*nInputs]), 
                     VecT2.getDVector(), VecS2.getDVector());
        //**/ re-shuffle
        for (ss2 = 0; ss2 < count2; ss2++)
        {
          ddata2 = VecT2[ss2] - VecYY[ig2*nSubSamples+ss2];
          VecE[IVec2[ig2*nSubSamples+ss2]] = ddata2;
          VecS[IVec2[ig2*nSubSamples+ss2]] = VecT2[ss2];
          VecSigmas[IVec2[ig2*nSubSamples+ss2]] = VecS2[ss2];
        }
        //**/ compute errors
        cvErr1 = cvErr2 = cvMax = cvMaxs2 = 0.0;
        for (ss2 = 0; ss2 < count2; ss2++)
        {
          ddata = VecE[IVec2[ig2*nSubSamples+ss2]];
          cvErr1 += ddata;
          cvErr2 += (ddata * ddata);
          if (PABS(ddata) > cvMax) cvMax  = PABS(ddata);
          if (ydiff > 0 && (PABS(ddata/ydiff) > cvMaxs2))
            cvMaxs2 = PABS(ddata/ydiff);
        }
        cvErr1  = cvErr1 / (double) count2;
        cvErr2  = sqrt(cvErr2 / count2);
        printOutTS(PL_INFO,
             "RSA: CV error for sample group %5d = %11.3e (avg unscaled)\n",
             ig2+1, cvErr1);
        printOutTS(PL_INFO,
             "RSA: CV error for sample group %5d = %11.3e (rms unscaled)\n",
             ig2+1, cvErr2);
        printOutTS(PL_INFO,
             "RSA: CV error for sample group %5d = %11.3e (max unscaled)\n",
             ig2+1, cvMax);
        if (ydiff > 0)
        {
          printOutTS(PL_INFO,
             "RSA: CV error for sample group %5d = %11.3e (max",
             ig2+1, cvMaxs2);
          printOutTS(PL_INFO,"   scaled by (ymax-ymin))\n");
        }
        printOutTS(PL_INFO,
             "RSAnalysis:: L %2d: CV group %d processed (size=%d)\n",
             nLevels,ig2+1,count2);
        delete faPtrs[ig2];
        faPtrs[ig2] = NULL;
      }
    }
    if (faPtrs != NULL) delete [] faPtrs;

    //**/ compute errors
    CVErr1 = CVErr1s = CVErr2 = CVErr2s = CVMax = CVMaxs = CVMaxs2 = 0.0;
    for (ig = 0; ig < numCVGroups_; ig++)
    {
      cvErr1 = cvErr2 = cvErr1s = cvErr2s = cvMax = cvMaxs = cvMaxs2 = 0.0;
      cvMaxBase = cvMaxBases = 0.0;
      count = nSubSamples;
      if (((ig+1)*nSubSamples) > nSamples) 
        count = nSamples - ig*nSubSamples;
      for (ss = 0; ss < count; ss++)
      {
        ddata = VecE[IVec2[ig*nSubSamples+ss]];
        cvErr1  += ddata;
        cvErr2  += (ddata * ddata);
        if (PABS(ddata) > cvMax)
        {
          cvMax  = PABS(ddata);
          cvMaxBase = PABS(VecYY[ig*nSubSamples+ss]);
        }
        if (VecYY[ig*nSubSamples+ss] != 0.0) 
             sdata = ddata / PABS(VecYY[ig*nSubSamples+ss]);
        else sdata = ddata;
        cvErr1s += sdata;
        cvErr2s += (sdata * sdata);
        if (PABS(sdata) > cvMaxs)
        {
          cvMaxs = PABS(sdata);
          cvMaxBases = PABS(VecYY[ig*nSubSamples+ss]);
        }
        if (ydiff > 0 && (PABS(ddata/ydiff) > cvMaxs2))
        {
          cvMaxs2 = PABS(ddata/ydiff);
        }
        if (printLevel > 4) 
          printOutTS(PL_INFO, 
             "Sample %6d: predicted =  %e, actual =  %e\n",
             IVec2[IVec1[ig*nSubSamples+ss]],
             VecS[IVec2[ig*nSubSamples+ss]],VecYY[ig*nSubSamples+ss]);
      }
      adata.sampleErrors_[ig] = cvErr2s;
      CVErr1  += cvErr1;
      CVErr1s += cvErr1s;
      CVErr2  += cvErr2;
      CVErr2s += cvErr2s;
      cvErr1  = cvErr1 / (double) count;
      cvErr1s = cvErr1s / (double) count;
      cvErr2  = sqrt(cvErr2 / count);
      cvErr2s = sqrt(cvErr2s / count);
      if (cvMax > CVMax )   {CVMax = cvMax; CVMaxBase = cvMaxBase;}
      if (cvMaxs > CVMaxs ) {CVMaxs = cvMaxs; CVMaxBases = cvMaxBases;}
      if (cvMaxs2 > CVMaxs2) CVMaxs2 = cvMaxs2;

      //**/ print error information
      printOutTS(PL_INFO,"RSA: first member of sample group %5d = %d\n",
           ig+1, IVec1[ig]+1);
      printOutTS(PL_INFO,
           "RSA: CV error for sample group %5d = %11.3e (avg unscaled)\n",
           ig+1, cvErr1);
      printOutTS(PL_INFO,
           "RSA: CV error for sample group %5d = %11.3e (avg scaled)\n",
           ig+1, cvErr1s);
      printOutTS(PL_INFO,
           "RSA: CV error for sample group %5d = %11.3e (rms unscaled)\n",
           ig+1, cvErr2);
      printOutTS(PL_INFO,
           "RSA: CV error for sample group %5d = %11.3e (rms scaled)\n",
           ig+1, cvErr2s);
      printOutTS(PL_INFO,
           "RSA: CV error for sample group %5d = %11.3e (max",
           ig+1, cvMax);
      printOutTS(PL_INFO," unscaled,BASE=%9.3e)\n", cvMaxBase);
      printOutTS(PL_INFO,
           "RSA: CV error for sample group %5d = %11.3e (max",
           ig+1, cvMaxs);
      printOutTS(PL_INFO,"   scaled,BASE=%9.3e)\n", cvMaxBases);
      if (ydiff > 0)
      {
        printOutTS(PL_INFO,
           "RSA: CV error for sample group %5d = %11.3e (max",
           ig+1, cvMaxs2);
        printOutTS(PL_INFO,"   scaled by (ymax-ymin))\n");
      }
    }
    status = 0;
    if (status >= 0)
    {
      CVErr1  = CVErr1 / (double) nSamples;
      CVErr1s = CVErr1s / (double) nSamples;
      CVErr2  = sqrt(CVErr2 / nSamples);
      CVErr2s = sqrt(CVErr2s / nSamples);
      printOutTS(PL_INFO,"RSA: final CV error  = %11.3e (avg unscaled)\n",
                 CVErr1);
      printOutTS(PL_INFO,"RSA: final CV error  = %11.3e (avg   scaled)\n",
                 CVErr1s);
      printOutTS(PL_INFO,"RSA: final CV error  = %11.3e (rms unscaled)\n",
                 CVErr2);
      printOutTS(PL_INFO,"RSA: final CV error  = %11.3e (rms   scaled)\n",
                 CVErr2s);
      printOutTS(PL_INFO,
           "RSA: final CV error  = %11.3e (max unscaled, BASE=%9.3e)\n",
           CVMax, CVMaxBase);
      printOutTS(PL_INFO,
           "RSA: final CV error  = %11.3e (max   scaled, BASE=%9.3e)\n",
           CVMaxs, CVMaxBases);
      if (ydiff > 0)
        printOutTS(PL_INFO,
           "RSA: final CV error  = %11.3e (max   scaled by (ymax-ymin))\n",
           CVMaxs2);
      printOutTS(PL_INFO,
           "RSA: L %2d:cross validation (CV) completed.\n",nLevels);
      if (plotScilab())
      {
        fpData = fopen("RSFA_CV_err.sci", "w");
        if (fpData != NULL)
          printOutTS(PL_INFO,"INFO: cannot open file RSFA_CV_err.sci.\n");
      }
      else
      {
        fpData = fopen("RSFA_CV_err.m", "w");
        if (fpData == NULL)
          printOutTS(PL_INFO,"INFO: cannot open file RSFA_CV_err.m.\n");
      }
      if (fpData != NULL)
      {
        strcpy(pString,"This file stores CV error for each point");
        fwriteComment(fpData, pString);
        strcpy(pString,"Column 1: error (true - predicted)");
        fwriteComment(fpData, pString);
        strcpy(pString,"Column 2: true");
        fwriteComment(fpData, pString);
        strcpy(pString,"Column 3: predicted");
        fwriteComment(fpData, pString);
        strcpy(pString,"Column 4: standard deviations"); 
        fwriteComment(fpData, pString);
        strcpy(pString,"Set morePlots=1 for normalized residual error plot");
        fwriteComment(fpData, pString);
        strcpy(pString,"Set errBoundOn=1 for error bars in CV plot");
        fwriteComment(fpData, pString);
        ssum = 0.0;
        fprintf(fpData, "errBoundOn = 0;\n");
        fprintf(fpData, "morePlots  = 0;\n");
        fprintf(fpData, "A = [\n");
        for (ss = 0; ss < nSamples; ss++)
        {
          fprintf(fpData, "  %e %e %e %e\n", VecE[ss], VecYLocal[ss],
                  VecS[ss], VecSigmas[ss]);
          ssum += PABS(VecSigmas[ss]);
        }
        fprintf(fpData, "];\n");
        fwriteHold(fpData, 0);
        fwritePlotFigure(fpData, 1);
        fprintf(fpData, "subplot(1,2,1)\n");
        if (plotScilab())
        {
          fprintf(fpData, "ymin = min(A(:,1));\n");
          fprintf(fpData, "ymax = max(A(:,1));\n");
          fprintf(fpData, "ywid = 0.1 * (ymax - ymin);\n");
          fprintf(fpData, "if (ywid < 1.0e-12)\n");
          fprintf(fpData, "   disp('range too small.')\n");
          fprintf(fpData, "   halt\n");
          fprintf(fpData, "end;\n");
          fprintf(fpData, "histplot(10, A(:,1), style=2);\n");
          fprintf(fpData, "a = gce();\n");
          fprintf(fpData, "a.children.fill_mode = \"on\";\n");
          fprintf(fpData, "a.children.thickness = 2;\n");
          fprintf(fpData, "a.children.foreground = 0;\n");
          fprintf(fpData, "a.children.background = 2;\n");
        }
        else
        {
          fprintf(fpData, "[nk, xk] = hist(A(:,1), 10);\n");
          fprintf(fpData, "bar(xk,nk/sum(nk))\n");
        }
        fwritePlotAxes(fpData);
        fwritePlotTitle(fpData, "CV Error");
        fwritePlotXLabel(fpData, "Error (unnormalized)");
        fwritePlotYLabel(fpData, "Probabilities");
        if (plotMatlab())
        {
          fprintf(fpData,
              "disp(['Error Mean  = ' num2str(mean(A(:,1)))])\n");
          fprintf(fpData,
              "disp(['Error stdev = ' num2str(std(A(:,1)))])\n");
        }
        fprintf(fpData,"subplot(1,2,2)\n");
        fprintf(fpData,"xmax = max(A(:,2));\n");
        fprintf(fpData,"xmin = min(A(:,2));\n");
        fprintf(fpData,"if (xmax == xmin) \n");
        fprintf(fpData,"   xmin = 0.9 * xmin; \n");
        fprintf(fpData,"   xmax = 1.1 * xmax; \n");
        fprintf(fpData,"end;\n");
        fprintf(fpData,"if (xmax == xmin) \n");
        fprintf(fpData,"   xmin = -0.1; \n");
        fprintf(fpData,"   xmax = 0.1; \n");
        fprintf(fpData,"end;\n");
        fprintf(fpData,"if errBoundOn == 1\n");
        fprintf(fpData,"ymax = max(A(:,3)+A(:,4));\n");
        fprintf(fpData,"ymin = min(A(:,3)-A(:,4));\n");
        fprintf(fpData,"else\n");
        fprintf(fpData,"ymax = max(A(:,3));\n");
        fprintf(fpData,"ymin = min(A(:,3));\n");
        fprintf(fpData,"end;\n");
        fprintf(fpData,"if (ymax == ymin) \n");
        fprintf(fpData, "   ymin = 0.9 * ymin; \n");
        fprintf(fpData, "   ymax = 1.1 * ymax; \n");
        fprintf(fpData, "end;\n");
        fprintf(fpData, "if (ymax == ymin) \n");
        fprintf(fpData, "   ymin = -0.1; \n");
        fprintf(fpData, "   ymax = 0.1; \n");
        fprintf(fpData, "end;\n");
        fprintf(fpData, "xmin = min(xmin, ymin);\n");
        fprintf(fpData, "xmax = max(xmax, ymax);\n");
        fprintf(fpData, "XX = xmin : xmax-xmin : xmax;\n");
        fprintf(fpData, "plot(A(:,2), A(:,3),'*','MarkerSize',12)\n");
        fwriteHold(fpData, 1);
        fprintf(fpData, "plot(XX, XX)\n");
        if (plotScilab())
        {
          fprintf(fpData, "a = get(\"current_axes\");\n");
          fprintf(fpData, "a.data_bounds=[xmin,xmin;xmax,xmax];\n");
        }
        else fprintf(fpData, "axis([xmin xmax xmin xmax])\n");
        if (ssum > 0.0)
        {
          if (plotScilab()) fprintf(fpData,"drawlater\n");
          fprintf(fpData,"cnt1 = 0;\n");
          fprintf(fpData,"cnt2 = 0;\n");
          fprintf(fpData,"cnt3 = 0;\n");
          fprintf(fpData,"cnt4 = 0;\n");
          fprintf(fpData,"if errBoundOn == 1\n");
          fprintf(fpData,"for ii = 1 : %d\n", nSamples);
          fprintf(fpData,"  xx = [A(ii,2) A(ii,2)];\n");
          fprintf(fpData,"  d1 = A(ii,3)-A(ii,4);\n");
          fprintf(fpData,"  d2 = A(ii,3)+A(ii,4);\n");
          fprintf(fpData,"  yy = [d1 d2];\n");
          fprintf(fpData,"  if (xx(1) < d1 | xx(1) > d2)\n");
          fprintf(fpData,"    plot(xx, yy, 'r-', 'lineWidth', 1)\n");
          fprintf(fpData,"    d3 = A(ii,3)-2*A(ii,4);\n");
          fprintf(fpData,"    d4 = A(ii,3)+2*A(ii,4);\n");
          fprintf(fpData,"    if (xx(1) < d3 | xx(1) > d4)\n");
          fprintf(fpData,"      d5 = A(ii,3)-3*A(ii,4);\n");
          fprintf(fpData,"      d6 = A(ii,3)+3*A(ii,4);\n");
          fprintf(fpData,"      if (xx(1) > d5 & xx(1) < d6)\n");
          fprintf(fpData,"        cnt3 = cnt3 + 1;\n");
          fprintf(fpData,"      else\n");
          fprintf(fpData,"        d7 = A(ii,3)-4*A(ii,4);\n");
          fprintf(fpData,"        d8 = A(ii,3)+4*A(ii,4);\n");
          fprintf(fpData,"        if (xx(1) > d7 & xx(1) < d8)\n");
          fprintf(fpData,"          cnt4 = cnt4 + 1;\n");
          fprintf(fpData,"        else\n");
          if (plotScilab())
            fprintf(fpData,
             "          disp('Point outside 4-sigma = ',ii)\n");
          else
            fprintf(fpData,
             "          disp(['Point outside 4-sigma = ',int2str(ii)])\n");
          fprintf(fpData,"        end;\n");
          fprintf(fpData,"      end;\n");
          fprintf(fpData,"    else\n");
          fprintf(fpData,"      cnt2 = cnt2 + 1;\n");
          fprintf(fpData,"    end;\n");
          fprintf(fpData,"  else\n");
          fprintf(fpData,"    plot(xx, yy, 'g-', 'lineWidth', 1);\n");
          fprintf(fpData,"    cnt1 = cnt1 + 1;\n");
          fprintf(fpData,"  end;\n");
          if (plotScilab())
          {
            fprintf(fpData,"//plot(xx(1), yy(1),'rv','markerSize',10);\n");
            fprintf(fpData,"//plot(xx(2), yy(2),'r^','markerSize',10);\n");
          }
          else
          {
            fprintf(fpData,"%%plot(xx(1), yy(1),'rv','markerSize',10);\n");
            fprintf(fpData,"%%plot(xx(2), yy(2),'r^','markerSize',10);\n");
          }
          fprintf(fpData,"end;\n");
          fprintf(fpData,"end;\n");
          if (plotScilab()) fprintf(fpData,"drawnow\n");
          else
          {
            fprintf(fpData,"cnt4 = cnt1 + cnt2 + cnt3 + cnt4;\n");
            fprintf(fpData,"cnt3 = cnt1 + cnt2 + cnt3;\n");
            fprintf(fpData,"cnt2 = cnt1 + cnt2;\n");
            fprintf(fpData,"disp('Total sample size = %d')\n",nSamples);
            fprintf(fpData,
               "disp(['Num points inside 1 sigma = ',int2str(cnt1)])\n");
            fprintf(fpData,
               "disp(['Num points inside 2 sigma = ',int2str(cnt2)])\n");
            fprintf(fpData,
               "disp(['Num points inside 3 sigma = ',int2str(cnt3)])\n");
            fprintf(fpData,
               "disp(['Num points inside 4 sigma = ',int2str(cnt4)])\n");
            fprintf(fpData, "if errBoundOn > 0\n");
            fprintf(fpData,
                "   text(0.1,0.9,'RED: prediction outside +/- 1 std ");
            fprintf(fpData,
                "dev','sc','fontSize',11,'fontweight','bold')\n");
            fprintf(fpData, "end;\n");
          }
        }
        fwriteHold(fpData, 0);
        fwritePlotAxes(fpData);
        if (yvar > 0 && ((1-CVErr2*CVErr2*nSamples/yvar) > 0))
          snprintf(pString,100,
             "Parity Plot (scaled rmse=%11.2e, R2=%11.2e)",
             CVErr2s,1.0-CVErr2*CVErr2*nSamples/yvar);
        else
          snprintf(pString,100,"Parity Plot (scaled rmse=%11.2e)",CVErr2s);
        fwritePlotTitle(fpData, pString);
        fwritePlotXLabel(fpData, "Sample Output");
        fwritePlotYLabel(fpData, "Predicted Output");
        fprintf(fpData, "rsme_unscaled = %e;\n", CVErr2);
        fprintf(fpData, "rsme_scaled   = %e;\n", CVErr2s);
        if (yvar > 0 && ((1-CVErr2*CVErr2*nSamples/yvar) > 0))
          fprintf(fpData, "R2 = %e;\n",1.0-CVErr2*CVErr2*nSamples/yvar);
        fprintf(fpData,"if morePlots == 1\n");
        strcpy(pString," For the following B matrix");
        fwriteComment(fpData, pString);
        strcpy(pString,"Column 1: true values");
        fwriteComment(fpData, pString);
        strcpy(pString,"Column 2: normalized residual");
        fwriteComment(fpData, pString);
        strcpy(pString,"Column 3: predicted values");
        fwriteComment(fpData, pString);
        strcpy(pString,"Column 4-(m+3): inputs");
        fwriteComment(fpData, pString);
        fwritePlotFigure(fpData, 2);
        fprintf(fpData, "B = [\n");
        for (ss = 0; ss < nSamples; ss++)
        {
          if (VecYLocal[ss] == 0) 
            fprintf(fpData, " %e 0 ",VecYLocal[ss]);
          else 
            fprintf(fpData," %e %e ",VecYLocal[ss],VecE[ss]/VecYLocal[ss]);
          fprintf(fpData," %e ", VecS[ss]);
          for (ss2 = 0; ss2 < nInputs; ss2++)
            fprintf(fpData," %e ",arrayXX[ss*nInputs+ss2]);
          fprintf(fpData,"\n");
        }
        fprintf(fpData,"];\n");
        fprintf(fpData,"AA = B(:,1);\n");
        fprintf(fpData,"BB = B(:,2);\n");
        fprintf(fpData,"CC = B(:,3);\n");
        fprintf(fpData,
                "subplot(1,2,1),plot(AA, BB, '*','markerSize',12);\n");
        fwritePlotAxes(fpData);
        fwritePlotTitle(fpData, "Normalized Error Analysis");
        fwritePlotXLabel(fpData, "Actual Data");
        fwritePlotYLabel(fpData, "Normalized Error");
        fprintf(fpData,
                "subplot(1,2,2),plot(AA,BB.*AA, '*','markerSize',12);\n");
        fwritePlotAxes(fpData);
        fwritePlotTitle(fpData, "Unnormalized Error Analysis");
        fwritePlotXLabel(fpData, "Actual Data");
        fwritePlotYLabel(fpData, "Error");
        fprintf(fpData,"figure(3)\n");
        fprintf(fpData,"for ii = 1 : %d\n", nInputs);
        fprintf(fpData,"  XX = B(:,3+ii);\n");
        fprintf(fpData,"  plot(XX, BB.*AA, '*','markerSize',12);\n");
        fprintf(fpData,"  disp(['Error vs Input ' int2str(ii)])\n");
        fprintf(fpData,"  disp('Press enter to continue')\n");
        fwritePlotAxes(fpData);
        fprintf(fpData,"  xlabel(['Input ' int2str(ii)])\n");
        fwritePlotYLabel(fpData,"Error");
        fprintf(fpData,"  pause\n");
        fprintf(fpData,"end;\n");
        fprintf(fpData,"end;\n");
        fclose(fpData);
        if (plotScilab())
             printOutTS(PL_INFO, "CV error file is RSFA_CV_err.sci\n");
        else printOutTS(PL_INFO, "CV error file is RSFA_CV_err.m\n");
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/  validation: fetch data file name from configure file
  //**/ ---------------------------------------------------------------
  testFlag = 0;
  cString = psConfig_.getParameter("RSFA_test_datafile");
  if (cString != NULL)
  {
    sscanf(cString, "%s %s %s",winput1,winput2,dataFile);
    fpData = fopen(dataFile, "r");
    if (fpData != NULL)
    {
      testFlag = 1;
      printOutTS(PL_INFO, "RSFA:: test data file = %s\n", dataFile);
      fclose(fpData);
      fpData = NULL;
    }
  }
  if (testFlag == 1)
  {
    cString = psConfig_.getParameter("RSFA_test_errorfile");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %s",winput1,winput2,errFile);
      fpErr = fopen(errFile, "w");
      if (fpErr != NULL)
      {
        printOutTS(PL_INFO, "RSFA:: error file = %s\n", errFile);
        testFlag += 2;
        fclose(fpErr);
      }
    }
  }
  if (testFlag == 1) retdata = validate(adata, dataFile, NULL);
  if (testFlag == 3) retdata = validate(adata, dataFile, errFile);
  printAsterisks(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ return
  //**/ ---------------------------------------------------------------
  return retdata;
}

// ************************************************************************ 
// set parameters
// ------------------------------------------------------------------------
int RSFuncApproxAnalyzer::setParams(int argc, char **argv)
{
  int   idata;
  char  *request, *dataFile, *errFile;
  aData *adata;

  //**/Analyzer::setParams(argc, argv);
  request = (char *) argv[0]; 
  if (!strcmp(request, "validate"))
  {
    if (argc != 4) printOutTS(PL_WARN, "RSAnalysis WARNING: setParams.\n");
    adata    = (aData *) argv[1];
    dataFile = (char *) argv[2];
    errFile  = (char *) argv[3];
    validate(*adata, dataFile, errFile);
  }
  else if (!strcmp(request, "rstype"))
  {
    if (argc != 2) printOutTS(PL_WARN, "RSAnalysis WARNING: setParams.\n");
    rsType_ = *(int *) argv[1];
    if (rsType_ < 0 || rsType_ > PSUADE_NUM_RS)
    {
      printOutTS(PL_ERROR, 
           "RSAnalysis ERROR: INVALID rstype, set to REGR2.\n");
      rsType_ = PSUADE_RS_REGR2;
    }
  }
  else if (!strcmp(request, "usecv"))
  {
    useCV_ = 1;
  }
  else if (!strcmp(request, "numcvgroups"))
  {
    idata = *(int *) argv[1];
    if (idata > 1) numCVGroups_ = idata;
  }
  else
  {
    printOutTS(PL_ERROR, "RSAnalysis ERROR: setParams - not valid.\n");
    exit(1);
  }
  return 0;
}

// ************************************************************************ 
// validate
// ------------------------------------------------------------------------
double RSFuncApproxAnalyzer::validate(aData &adata, char *dataFile, 
                                      char *errFile)
{
  int    ii, iOne=1, nPtsPerDim=64;
  double sumErr1, sumErr2, maxErr, sumErr1s, sumErr2s, maxErrs, maxErrs2;
  double maxBase, maxBases, sdata, ddata, ssum;
  char   pString[501];
  pData  pPtr, pInputs, pOutputs;
  psVector VecYLocal, VecWgts, VecYT, VecStd;
  FILE     *fpErr;

  //**/ ---------------------------------------------------------------
  //**/ initial message
  //**/ ---------------------------------------------------------------
  printOutTS(PL_INFO, "RSAnalysis: validating against a test set ...\n");

  //**/ fetch training set data
  int nInputs    = adata.nInputs_;
  int nOutputs   = adata.nOutputs_;
  int nSamples   = adata.nSamples_;
  double *lower  = adata.iLowerB_;
  double *upper  = adata.iUpperB_;
  double *arrayX = adata.sampleInputs_;
  double *arrayY = adata.sampleOutputs_;
  int outputID   = adata.outputID_;
  int wgtID      = adata.regWgtID_;
  int fatype     = adata.faType_;
  VecYLocal.setLength(nSamples);
  for (ii = 0; ii < nSamples; ii++) 
    VecYLocal[ii] = arrayY[ii*nOutputs+outputID];
  double ymax = VecYLocal.max();
  double ymin = VecYLocal.min();
  double ydiff = ymax - ymin;

  //**/ ---------------------------------------------------------------
  //**/ fetch info from  data file
  //**/ ---------------------------------------------------------------
  PsuadeData *ioPtr = new PsuadeData();
  int status = ioPtr->readPsuadeFile(dataFile);
  if (status != 0)
  {
    printf("ERROR: cannot read file %s in PSUADE format.\n",dataFile);
    exit(1);
  } 
  ioPtr->getParameter("output_noutputs", pPtr);
  int nTestOut = pPtr.intData_;
  ioPtr->getParameter("method_nsamples", pPtr);
  int nTestSam = pPtr.intData_;
  ioPtr->getParameter("input_ninputs", pPtr);
  int nTestIn = pPtr.intData_;
  if (nTestSam <= 0)
  {
    printOutTS(PL_WARN, "RSAnalysis: file %s has no data.\n", dataFile);
    delete ioPtr;
    return 1.0e12;
  }
  if (nTestIn != nInputs)
  {
    printOutTS(PL_WARN, 
         "RSAnalysis: test data has different number of inputs (%d)\n",
         nTestIn);
    printOutTS(PL_WARN,
         "            than that of the sample (%d)\n", nInputs);
    delete ioPtr;
    return 1.0e12;
  }
  ioPtr->getParameter("input_sample",pInputs);
  double *arrayXX = pInputs.dbleArray_;
  ioPtr->getParameter("output_sample",pOutputs);
  double *arrayYY = pOutputs.dbleArray_;

  //**/ ---------------------------------------------------------------
  //**/ construct response surface using training set
  //**/ Old: fa = genFA(rsType_, nInputs, iOne, nSamples);
  //**/ ---------------------------------------------------------------
  FuncApprox *fa = genFA(fatype, nInputs, iOne, nSamples);
  if (fa == NULL)
  {
    printOutTS(PL_INFO, 
         "RSAnalysis INFO: cannot create function approximator.\n");
    delete ioPtr;
    return 1.0e12;
  }
  fa->setNPtsPerDim(nPtsPerDim);
  fa->setBounds(adata.iLowerB_, adata.iUpperB_);
  fa->setOutputLevel(adata.printLevel_);
  if (wgtID >= 0 && wgtID < nOutputs)
  {
    VecWgts.setLength(nSamples);
    for (ii = 0; ii < nSamples; ii++) 
      VecWgts[ii] = arrayY[ii*nOutputs+wgtID];
    fa->loadWeights(nSamples, VecWgts.getDVector());
  }
  status = fa->initialize(arrayX, VecYLocal.getDVector());

  //**/ ---------------------------------------------------------------
  //**/ evaluate the test set
  //**/ ---------------------------------------------------------------
  if (errFile != NULL) 
  {
    fpErr = fopen(errFile, "w");
    if (fpErr != NULL)
    {
      strcpy(pString, "Surface Fitting Error Histogram");
      fwriteComment(fpErr, pString);
      strcpy(pString, "Interpolation errors on all points");
      fwriteComment(fpErr, pString);
      strcpy(pString, "Column 1: predicted data");
      fwriteComment(fpErr, pString);
      strcpy(pString, "Column 2: actual data");
      fwriteComment(fpErr, pString);
      strcpy(pString, "Column 3: error = predicted - actual data");
      fwriteComment(fpErr, pString);
      strcpy(pString, "Column 4: normalized error");
      fwriteComment(fpErr, pString);
      strcpy(pString, "Column 5: predicted standard deviation");
      fwriteComment(fpErr, pString);
      fprintf(fpErr, "E = [\n");
    }
  }
  else fpErr = NULL;

  double dataMax = -PSUADE_UNDEFINED;
  double dataMin = PSUADE_UNDEFINED;
  for (ii = 0; ii < nTestSam; ii++)
  {
    ddata = PABS(arrayYY[ii*nTestOut+outputID]);
    if (ddata > dataMax) dataMax = ddata;
    if (ddata < dataMin) dataMin = ddata;
  }
  //**/ evaluate many points instead of one at a time
  //**/ for (ii = 0; ii < nTestSam; ii++)
  //**/ {
  //**/    ddata = fa->evaluatePoint(&arrayXX[ii*nInputs]);
  //**/    if (fpErr != NULL) 
  //**/       fprintf(fpErr, "%24.16e %24.16e %24.16e\n", ddata, 
  //**/         arrayYY[ii*nTestOut+outputID],
  //**/         ddata-arrayYY[ii*nTestOut+outputID]);
  //**/    ddata = PABS(ddata - arrayYY[ii*nTestOut+outputID]);
  //**/    if (arrayYY[ii*nTestOut+outputID] != 0.0)
  //**/         sdata = ddata / PABS(arrayYY[ii*nTestOut+outputID]);
  //**/    else sdata = ddata;
  //**/    if (ddata > maxErr1) maxErr1 = ddata;
  //**/    if (sdata > maxErrs)
  //**/    {
  //**/       maxErrs = sdata;
  //**/       maxBase = PABS(arrayYY[ii*nTestOut+outputID]);
  //**/    }
  //**/    sumErr1 += (ddata * ddata);
  //**/    sumErrs += (sdata * sdata);
  //**/ }
  VecYT.setLength(nTestSam);
  VecStd.setLength(nTestSam);
  fa->evaluatePointFuzzy(nTestSam,arrayXX,VecYT.getDVector(),
                         VecStd.getDVector());
  sumErr1 = sumErr1s = sumErr2 = sumErr2s = maxErr = maxErrs = 0.0;
  maxErrs2 = 0;
  for (ii = 0; ii < nTestSam; ii++)
  {
    ddata = VecYT[ii];
    sdata = arrayYY[ii*nTestOut+outputID];
    if (fpErr != NULL) 
    {
      if (sdata != 0)
        fprintf(fpErr, "%16.8e %16.8e %16.8e %16.8e %16.8e\n", ddata, 
           sdata, ddata-sdata, PABS((ddata-sdata)/sdata), VecStd[ii]);
      else
        fprintf(fpErr, "%16.8e %16.8e %16.8e %16.8e %16.8e\n", ddata, 
           sdata, ddata-sdata, PABS(ddata-sdata), VecStd[ii]);
    }
    ddata = ddata - sdata;
    if (sdata != 0.0) sdata = ddata / PABS(sdata);
    else              sdata = ddata;
    sumErr1  += ddata;
    sumErr1s += sdata;
    sumErr2  += (ddata * ddata);
    sumErr2s += (sdata * sdata);
    ddata = PABS(ddata);
    sdata = PABS(sdata);
    if (ddata > maxErr ) 
    {
      maxErr = ddata;
      maxBase = PABS(arrayYY[ii*nTestOut+outputID]);
    }
    if (sdata > maxErrs)
    {
      maxErrs  = sdata;
      maxBases = PABS(arrayYY[ii*nTestOut+outputID]);
    }
    if (ydiff > 0 && (ddata/ydiff > maxErrs2)) 
    {
      maxErrs2 = ddata / ydiff;
    }
  }

  sumErr1  = sumErr1 / (double) nTestSam;
  sumErr1s = sumErr1s / (double) nTestSam;
  sumErr2  = sqrt(sumErr2 / (double) nTestSam);
  sumErr2s = sqrt(sumErr2s / (double) nTestSam);
  printOutTS(PL_INFO,"RSA: Test data maximum/minimum      = %9.3e %9.3e\n",
         dataMax, dataMin);
  printOutTS(PL_INFO, 
       "RSA: Prediction errors = %11.3e (avg unscaled)\n",sumErr1);
  printOutTS(PL_INFO, 
       "RSA: Prediction errors = %11.3e (avg   scaled)\n",sumErr1s);
  printOutTS(PL_INFO, 
       "RSA: Prediction errors = %11.3e (rms unscaled)\n",sumErr2);
  printOutTS(PL_INFO, 
       "RSA: Prediction errors = %11.3e (rms   scaled)\n",sumErr2s);
  printOutTS(PL_INFO, 
       "RSA: Prediction errors = %11.3e (max unscaled, BASE=%9.3e)\n",
        maxErr, maxBase);
  printOutTS(PL_INFO, 
       "RSA: Prediction errors = %11.3e (max   scaled, BASE=%9.3e)\n",
        maxErrs, maxBases);
  if (ydiff > 0)
    printOutTS(PL_INFO, 
       "RSA: Prediction errors = %11.3e (max   scaled by (ymax-ymin))\n",
        maxErrs2);
  ssum = 0.0;
  for (ii = 0; ii < nTestSam; ii++) ssum += VecStd[ii];
  if (ssum != 0)
    printOutTS(PL_INFO, 
         "RSA: Average std. dev. = %11.3e (sum of all points)\n",
         ssum/nSamples);

  if (fpErr != NULL) 
  {
    fprintf(fpErr, "];\n");
    fwritePlotCLF(fpErr);
    fwritePlotFigure(fpErr, 1);
    fprintf(fpErr, "subplot(1,2,1)\n");
    fprintf(fpErr, "plot(E(:,3),'x')\n");
    fwritePlotAxes(fpErr);
    fwritePlotTitle(fpErr, "Error Plot");
    fwritePlotXLabel(fpErr, "Sample Number");
    fwritePlotYLabel(fpErr, "Error");
    fprintf(fpErr, "subplot(1,2,2)\n");
    if (plotScilab())
    {
      fprintf(fpErr, "ymin = min(E(:,3));\n");
      fprintf(fpErr, "ymax = max(E(:,3));\n");
      fprintf(fpErr, "ywid = 0.1 * (ymax - ymin);\n");
      fprintf(fpErr, "if (ywid < 1.0e-12)\n");
      fprintf(fpErr, "   disp('range too small.')\n");
      fprintf(fpErr, "   halt\n");
      fprintf(fpErr, "end;\n");
      fprintf(fpErr, "histplot(10, E(:,3), style=2);\n");
      fprintf(fpErr, "a = gce();\n");
      fprintf(fpErr, "a.children.fill_mode = \"on\";\n");
      fprintf(fpErr, "a.children.thickness = 2;\n");
      fprintf(fpErr, "a.children.foreground = 0;\n");
      fprintf(fpErr, "a.children.background = 2;\n");
    }
    else
    {
      fprintf(fpErr, "[nk,xk]=hist(E(:,3),10);\n");
      fprintf(fpErr, "bar(xk,nk/%d,1.0)\n",nTestSam);
    }
    fwritePlotAxes(fpErr);
    fwritePlotTitle(fpErr, "Histogram of Errors");
    fwritePlotXLabel(fpErr, "Error");
    fwritePlotYLabel(fpErr, "Probabilities");
    fwritePlotFigure(fpErr, 2);
    fprintf(fpErr, "xmax = max(E(:,2));\n");
    fprintf(fpErr, "xmin = min(E(:,2));\n");
    fprintf(fpErr, "if (xmax == xmin) \n");
    fprintf(fpErr, "   xmin = 0.9 * xmin; \n");
    fprintf(fpErr, "   xmax = 1.1 * xmax; \n");
    fprintf(fpErr, "end;\n");
    fprintf(fpErr, "if (xmax == xmin) \n");
    fprintf(fpErr, "   xmin = -0.1; \n");
    fprintf(fpErr, "   xmax = 0.1; \n");
    fprintf(fpErr, "end;\n");
    fprintf(fpErr, "ymax = max(E(:,1)+E(:,5));\n");
    fprintf(fpErr, "ymin = min(E(:,1)-E(:,5));\n");
    fprintf(fpErr, "if (ymax == ymin) \n");
    fprintf(fpErr, "   ymin = 0.9 * ymin; \n");
    fprintf(fpErr, "   ymax = 1.1 * ymax; \n");
    fprintf(fpErr, "end;\n");
    fprintf(fpErr, "if (ymax == ymin) \n");
    fprintf(fpErr, "   ymin = -0.1; \n");
    fprintf(fpErr, "   ymax = 0.1; \n");
    fprintf(fpErr, "end;\n");
    fprintf(fpErr, "xmin = min(xmin, ymin);\n");
    fprintf(fpErr, "xmax = max(xmax, ymax);\n");
    fprintf(fpErr, "plot(E(:,2), E(:,1),'*','MarkerSize',12)\n");
    fwriteHold(fpErr, 1);
    fprintf(fpErr, "XX = xmin : xmax-xmin : xmax;\n");
    fprintf(fpErr, "plot(XX, XX)\n");
    if (plotScilab())
    {
      fprintf(fpErr, "a = get(\"current_axes\");\n");
      fprintf(fpErr, "a.data_bounds=[xmin,xmin;xmax,xmax];\n");
    }
    else
    {
      fprintf(fpErr, "axis([xmin xmax xmin xmax])\n");
    }
    if (ssum > 0.0)
    {
      fprintf(fpErr,"for ii = 1 : %d\n", nTestSam);
      fprintf(fpErr,"  xx = [E(ii,2) E(ii,2)];\n");
      fprintf(fpErr,"  d1 = E(ii,1)-E(ii,5);\n");
      fprintf(fpErr,"  d2 = E(ii,1)+E(ii,5);\n");
      fprintf(fpErr,"  yy = [d1 d2];\n");
      fprintf(fpErr,"  if (xx(1) < d1 | xx(1) > d2);\n");
      fprintf(fpErr,"    plot(xx, yy, 'r-', 'lineWidth', 1);\n");
      fprintf(fpErr,"   else\n");
      fprintf(fpErr,"     plot(xx, yy, 'g-', 'lineWidth', 1);\n");
      fprintf(fpErr,"  end;\n");
      fprintf(fpErr,"%% plot(xx(1), yy(1), 'rv', 'markerSize', 10);\n");
      fprintf(fpErr,"%% plot(xx(2), yy(2), '^', 'markerSize', 10);\n");
      fprintf(fpErr,"end;\n");
    }
    fwriteHold(fpErr, 0);
    if (plotMatlab())
    {
      if (ssum > 0.0)
      {
        fprintf(fpErr,"text(0.1,0.9,'RED: prediction outside +/- 1 std");
        fprintf(fpErr," dev','sc','fontSize',11,'fontweight','bold')\n");
      }
    }
    fwritePlotAxes(fpErr);
    fwritePlotTitle(fpErr, "Interpolated versus actual data");
    fwritePlotXLabel(fpErr, "Actual data");
    fwritePlotYLabel(fpErr, "Interpolated data");
    fclose(fpErr);
    fpErr = NULL;
    printOutTS(PL_INFO, 
         "RSAnalysis: individual prediction errors can be found in %s.\n",
         errFile);
  }

  //**/ clean up
  delete fa;
  delete ioPtr;
  return sumErr1;
}

// ************************************************************************ 
// genTrainingErrPlot
// ------------------------------------------------------------------------
double RSFuncApproxAnalyzer::genTrainingErrPlot(FuncApprox *faPtr, 
             psVector &VecX, psVector &VecYTrue, psVector &VecYPredict)
{
  int    ss, ii, iL, nSamples, nInputs;
  double sumErr1, maxErr, maxBase, ddata, retdata;
  char   pString[1000], winput[1000];
  FILE *fpErr=NULL;

  if (plotScilab())
  {
    fpErr = fopen("RSFA_training_err.sci", "w");
    if (fpErr == NULL)
      printOutTS(PL_INFO,
           "INFO: cannot open file RSFA_training_err.sci.\n");
  }
  else
  {
    fpErr = fopen("RSFA_training_err.m", "w");
    if (fpErr == NULL)
      printOutTS(PL_INFO,
           "INFO: cannot open file RSFA_training_err.m.\n");
  }
  if (fpErr != NULL)
  {
    strcpy(pString, "Surface Fitting Error Histogram");
    fwriteComment(fpErr, pString);
    strcpy(pString, "Interpolation errors on all points");
    fwriteComment(fpErr, pString);
    strcpy(pString, "col 1: interpolated data");
    fwriteComment(fpErr, pString);
    strcpy(pString, "col 2: training data");
    fwriteComment(fpErr, pString);
    strcpy(pString, "col 3: col 1 - col2");
    fwriteComment(fpErr, pString);
    strcpy(pString, "col 4 on: input data");
    fwriteComment(fpErr, pString);
    fprintf(fpErr, "E = [\n");
  }
  sumErr1 = maxBase = maxErr = 0;
  nSamples = VecYTrue.length();
  nInputs  = VecX.length() / nSamples;
  faPtr->evaluatePoint(nSamples,VecX.getDVector(),
                       VecYPredict.getDVector());
  for (ss = 0; ss < nSamples; ss++)
  {
    ddata = VecYPredict[ss];
    if (fpErr != NULL)
    {
      fprintf(fpErr, "%24.16e %24.16e %24.16e ", ddata, 
              VecYTrue[ss], ddata-VecYTrue[ss]);
      for (ii = 0; ii < nInputs; ii++)
        fprintf(fpErr, "%24.16e ", VecX[ss*nInputs+ii]); 
      fprintf(fpErr, "\n");
    }
    ddata = ddata - VecYTrue[ss];
    ddata = PABS(ddata);
    sumErr1  += ddata;
    if (ddata > maxErr )
    {
      maxErr  = ddata; 
      maxBase = PABS(VecYTrue[ss]);
    }
  }
  if (fpErr != NULL)
  {
    fprintf(fpErr, "];\n");
    fwritePlotCLF(fpErr);
    fwritePlotFigure(fpErr, 1);
    fprintf(fpErr, "subplot(1,2,1)\n");
    fprintf(fpErr, "plot(E(:,3),'*', 'markersize',10)\n");
    fwritePlotAxes(fpErr);
    fwritePlotTitle(fpErr, "Interpolation Error Plot");
    fwritePlotXLabel(fpErr, "Sample Number");
    fwritePlotYLabel(fpErr, "Interpolation Error");
    fprintf(fpErr, "subplot(1,2,2)\n");
    fprintf(fpErr, "xmax = max(E(:,2));\n");
    fprintf(fpErr, "xmin = min(E(:,2));\n");
    fprintf(fpErr, "ymax = max(E(:,1));\n");
    fprintf(fpErr, "ymin = min(E(:,1));\n");
    fprintf(fpErr, "xmin = min(xmin, ymin);\n");
    fprintf(fpErr, "xmax = max(xmax, ymax);\n");
    fprintf(fpErr, "XX   = xmin : xmax-xmin : xmax;\n");
    fprintf(fpErr, "plot(E(:,2), E(:,1),'*', 'markersize', 10)\n");
    fwriteHold(fpErr, 1);
    fprintf(fpErr, "plot(XX, XX, 'linewidth', 2)\n");
    if (plotScilab())
    {
      fprintf(fpErr, "a = get(\"current_axes\");\n");
      fprintf(fpErr, "a.data_bounds=[xmin,xmin;xmax,xmax];\n");
    }
    else
    {
      fprintf(fpErr, "axis([xmin xmax xmin xmax])\n");
    }
    fwritePlotAxes(fpErr);
    fwritePlotTitle(fpErr, "Interpolated vs actual data");
    fwritePlotXLabel(fpErr, "Actual data");
    fwritePlotYLabel(fpErr, "Interpolated data");

    fwritePlotFigure(fpErr, 2);
    fwritePlotCLF(fpErr);
    fprintf(fpErr, "subplot(1,2,1)\n");
    if (plotScilab())
    {
      fprintf(fpErr, "ymin = min(E(:,3));\n");
      fprintf(fpErr, "ymax = max(E(:,3));\n");
      fprintf(fpErr, "ywid = 0.1 * (ymax - ymin);\n");
      fprintf(fpErr, "if (ywid < 1.0e-12)\n");
      fprintf(fpErr, "   disp('range too small.')\n");
      fprintf(fpErr, "   halt\n");
      fprintf(fpErr, "end;\n");
      fprintf(fpErr, "histplot(10, E(:,3), style=2);\n");
      fprintf(fpErr, "a = gce();\n");
      fprintf(fpErr, "a.children.fill_mode = \"on\";\n");
      fprintf(fpErr, "a.children.thickness = 2;\n");
      fprintf(fpErr, "a.children.foreground = 0;\n");
      fprintf(fpErr, "a.children.background = 2;\n");
    }
    else
    {
      fprintf(fpErr, "[nk,xk]=hist(E(:,3),10);\n");
      fprintf(fpErr, "bar(xk,nk/%d,1.0)\n",nSamples);
    }
    fwritePlotAxes(fpErr);
    fwritePlotTitle(fpErr, "Interpolation Errors Histogram");
    fwritePlotXLabel(fpErr, "Error");
    fwritePlotYLabel(fpErr, "Probabilities");
    fprintf(fpErr, "subplot(1,2,2)\n");
    if (plotScilab())
    {
      fprintf(fpErr, "ymin = min(E(:,3)/E(:,2));\n");
      fprintf(fpErr, "ymax = max(E(:,3)/E(:,2));\n");
      fprintf(fpErr, "ywid = 0.1 * (ymax - ymin);\n");
      fprintf(fpErr, "if (ywid < 1.0e-12)\n");
      fprintf(fpErr, "   disp('range too small.')\n");
      fprintf(fpErr, "   halt\n");
      fprintf(fpErr, "end;\n");
      fprintf(fpErr, "histplot(10, E(:,3)/E(:,2), style=2);\n");
      fprintf(fpErr, "a = gce();\n");
      fprintf(fpErr, "a.children.fill_mode = \"on\";\n");
      fprintf(fpErr, "a.children.thickness = 2;\n");
      fprintf(fpErr, "a.children.foreground = 0;\n");
      fprintf(fpErr, "a.children.background = 2;\n");
    }
    else
    {
      fprintf(fpErr, "[nk,xk]=hist(E(:,3)./E(:,2),10);\n");
      fprintf(fpErr, "bar(xk,nk/%d,1.0)\n",nSamples);
    }
    fwritePlotAxes(fpErr);
    fwritePlotTitle(fpErr, 
          "Interpolation Errors Histogram (normalized)");
    fwritePlotXLabel(fpErr, "Error");
    fwritePlotYLabel(fpErr, "Probabilities");

    fwritePlotFigure(fpErr, 3);
    fprintf(fpErr, "nn = %d;\n", nInputs);
    iL = (int) pow(1.0*nInputs, 0.5);
    if (iL*iL < nInputs) iL++;
    if (plotScilab()) fprintf(fpErr, "drawlater\n");
    fprintf(fpErr, "for ii = 1 : nn\n");
    fprintf(fpErr, "   subplot(%d,%d,ii)\n", iL, iL);
    fprintf(fpErr, "   plot(E(:,ii+3),E(:,3),'x')\n");
    fwritePlotAxes(fpErr);
    if (plotScilab())
    {
      fwritePlotTitle(fpErr,"Error Plot for Input");
      snprintf(winput,100,
           "a.title.text = \"Error Plot for Input\" + string(ii);\n");
      fprintf(fpErr, "%s", winput);
      fwritePlotXLabel(fpErr, "Input Values");
    }
    else
    {
      fprintf(fpErr, "title(['Error Plot for Input ',int2str(ii)])\n");
      fwritePlotXLabel(fpErr, "Input Values");
    }
    fwritePlotYLabel(fpErr, "Interpolation Error");
    if (plotScilab()) fprintf(fpErr, "drawnow\n");
    fprintf(fpErr, "end\n");

    //**/ provided for users to do scatter plot of errors
    if (nInputs > 2)
    {
      fwritePlotFigure(fpErr, 4);
      if (plotScilab())
      {
        fprintf(fpErr, "f = gcf();\n");
        fprintf(fpErr, "f.color_map = jetcolormap(3);\n");
        fprintf(fpErr, "drawlater\n");
        fprintf(fpErr, "param3d1([E(:,4)' ; E(:,4)'],");
        fprintf(fpErr, "[E(:,5)' ; E(:,5)'],[E(:,3)' ; E(:,3)'])\n");
        fprintf(fpErr, "e = gce();\n");
        fprintf(fpErr, "e.children.mark_mode = \"on\";\n");
        fprintf(fpErr, "e.children.mark_size_unit = \"point\";\n");
        fprintf(fpErr, "e.children.mark_style = 10;\n");
        fprintf(fpErr, "e.children.mark_size = 6;\n");
        fprintf(fpErr, "for i = 1:length(e.children)\n");
        fprintf(fpErr, "   e.children(i).mark_foreground = 1;\n");
        fprintf(fpErr, "end\n");
        fprintf(fpErr, "set(gca(),\"auto_clear\",\"off\")\n");
        fprintf(fpErr, "drawnow\n");
      }
      else
      {
        fprintf(fpErr, "   plot3(E(:,4),E(:,5),E(:,3),'bp')\n");
      }
      fwritePlotXLabel(fpErr, "Input 1");
      fwritePlotYLabel(fpErr, "Input 2");
      fwritePlotTitle(fpErr, "Output Error Scatter Plot");
      fwritePlotAxes(fpErr);
    }
    fclose(fpErr);
    if (plotScilab())
      printOutTS(PL_INFO,
           "Interpolation error info are in RSFA_training_err.sci\n");
    else
      printOutTS(PL_INFO,
           "Interpolation error info are in RSFA_training_err.m\n");
  }
  sumErr1   = sumErr1 / (double) nSamples;
  retdata = sumErr1 / VecYTrue.max();
  return retdata;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
RSFuncApproxAnalyzer& 
RSFuncApproxAnalyzer::operator=(const RSFuncApproxAnalyzer &)
{
  printOutTS(PL_ERROR, 
       "RSAnalysis operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

