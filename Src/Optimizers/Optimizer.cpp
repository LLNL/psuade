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
// Functions for the class Optimizer
// AUTHOR : CHARLES TONG
// DATE   : 2005
// ************************************************************************
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "Globals.h"
#include "PsuadeUtil.h"
#include "FuncApprox.h"
#include "pData.h"
#include "Optimizer.h"
#include "APPSPACKOptimizer.h"
#include "MMOptimizer.h"
#include "CobylaOptimizer.h"
#include "BobyqaOptimizer.h"
#include "SMOptimizer.h"
#include "MinpackOptimizer.h"
#include "TxMathOptimizer.h"
#include "SCEOptimizer.h"
#include "MultiObjectiveOptimizer.h"
#include "OUU1Optimizer.h"
#include "OUU2Optimizer.h"
#include "OUUOptimizer.h"
#include "LincoaOptimizer.h"
#include "NewuoaOptimizer.h"
#include "LBFGSOptimizer.h"
#include "NomadOptimizer.h"
#include "OUUMinlpOptimizer.h"
#include "PSOptimizer.h"

#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
Optimizer::Optimizer()
{
  optimalY_ = 1e35;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
Optimizer::~Optimizer()
{
}

// ************************************************************************
// set objective function
// ------------------------------------------------------------------------
void Optimizer::setObjectiveFunction(void (*func)(int,double*,int,double*))
{
  objFunction_ = func;
}

// ************************************************************************
// optimize
// ------------------------------------------------------------------------
void Optimizer::optimize(oData *odata)
{
  (void) odata;
  return;
}

// ************************************************************************
// set parameter
// ------------------------------------------------------------------------
void Optimizer::setParam(char *)
{
  return;
}

// ************************************************************************
// get optimal X 
// ------------------------------------------------------------------------
double *Optimizer::getOptimalX()
{
  return VecOptX_.getDVector();
}

// ************************************************************************
// get optimal Y 
// ------------------------------------------------------------------------
double Optimizer::getOptimalY()
{
  return optimalY_;
}

// ************************************************************************
// perform optimization 
// ------------------------------------------------------------------------
int OptimizerSearch(PsuadeData *psuadeIO, FunctionInterface *funcIO,
                    psMatrix &matOptData, int nInitialX)
{
  int    optimizeFlag, optimizeNumPts, optimizeUseRS, optMethod;
  int    minIndex, nPtsPerDim, faLeng;
  int    minCheck, returnFlag=0, ss, numYmin, istart, optimalCount;
  int    ii, ind1, ind2, totalEval;
  int    optID, optPrintLevel, optNewCnt, optNumFmin, M1=-1;
  double optCutOff, optTol, Ymin, optFmin;
  char   specsFile[200], sparam[100];
  pData  pPtr, pLowerB, pUpperB, pInpData, pOutData;
  FuncApprox *faPtr;
#ifdef HAVE_TXMATH
  TxMathOptimizer   *TxMathPtr;
#endif
#ifdef HAVE_APPSPACK
  APPSPACKOptimizer *APPSPACKPtr;
#endif
#ifdef HAVE_MINPACK
  MinpackOptimizer  *MinpackPtr;
#endif
  CobylaOptimizer   *CobylaPtr;
  SMOptimizer       *SMPtr;
  MMOptimizer       *MMPtr;
  BobyqaOptimizer   *BobyqaPtr;
  SCEOptimizer      *SCEPtr;
  MultiObjectiveOptimizer *MooPtr;
  OUU1Optimizer     *OUU1Ptr;
  OUU2Optimizer     *OUU2Ptr;
  OUUOptimizer      *OUUPtr;
  LincoaOptimizer   *LincoaPtr;
  NewuoaOptimizer   *NewuoaPtr;
  LBFGSOptimizer    *lbfgsPtr;
  NomadOptimizer    *nomadPtr;
  OUUMinlpOptimizer *minlpPtr;
  PSOptimizer       *psoPtr;
  oData             odata;

  //**/ ----------------------------------------------------------------
  //**/ fetch parameters from PsuadeData
  //**/ ----------------------------------------------------------------
  psuadeIO->getParameter("input_ninputs", pPtr);
  int nInputs = pPtr.intData_;
  psuadeIO->getParameter("output_noutputs", pPtr);
  int nOutputs = pPtr.intData_;
  psuadeIO->getParameter("method_nsamples", pPtr);
  int nSamples = pPtr.intData_;
  psuadeIO->getParameter("input_lbounds", pLowerB);
  double *iLowerB = pLowerB.dbleArray_;
  psuadeIO->getParameter("input_ubounds", pUpperB);
  double *iUpperB = pUpperB.dbleArray_;
  psuadeIO->getParameter("input_sample", pInpData);
  double *sampleInputs = pInpData.dbleArray_;
  psuadeIO->getParameter("output_sample", pOutData);
  double *sampleOutputs = pOutData.dbleArray_;

  //**/ ----------------------------------------------------------------
  //**/ get optimization parameters
  //**/ ----------------------------------------------------------------
  psuadeIO->getParameter("ana_opt_switch", pPtr);
  optimizeFlag = pPtr.intData_;
  psuadeIO->getParameter("ana_opt_nlocalmin", pPtr);
  optimizeNumPts = pPtr.intData_;
  psuadeIO->getParameter("ana_opt_rstype", pPtr);
  optimizeUseRS = pPtr.intData_;
  psuadeIO->getParameter("ana_opt_outputid", pPtr);
  optID = pPtr.intData_;
  psuadeIO->getParameter("ana_opt_printlevel", pPtr);
  optPrintLevel = pPtr.intData_;
  psuadeIO->getParameter("ana_opt_method", pPtr);
  optMethod = pPtr.intData_;
  psuadeIO->getParameter("ana_opt_cutoff", pPtr);
  optCutOff = pPtr.dbleData_;
  psuadeIO->getParameter("ana_opt_tolerance", pPtr);
  optTol = pPtr.dbleData_;
  psuadeIO->getParameter("ana_opt_fmin", pPtr);
  optFmin = pPtr.dbleData_;
  psuadeIO->getParameter("ana_opt_numfmin", pPtr);
  optNumFmin = pPtr.intData_;
  psuadeIO->getParameter("ana_opt_maxfeval", pPtr);
  odata.maxFEval_ = pPtr.intData_;
  psuadeIO->getParameter("ana_opt_deltax", pPtr);
  odata.deltaX_ = pPtr.dbleData_;
  //**/ MINE: SPECS
  psuadeIO->getParameter("ana_opt_smtargetfile", pPtr);
  strcpy(specsFile, pPtr.strArray_[0]);
  optimalCount = optimizeNumPts;
  if (matOptData.ncols() != optimalCount*nInputs)
  {
    printf("PSUADE Optimizer ERROR: matOptData size mismatch\n");
    return 0;
  }
  if (matOptData.nrows() != 4)
  {
    printf("PSUADE Optimizer ERROR: matOptData mus have 4 rows\n");
    return 0;
  }
  psuadeIO->getParameter("app_maxparalleljobs", pPtr);
  odata.maxParallelJobs_ = pPtr.intData_;
  if (optimizeNumPts <= 0)
  {
    printf("PSUADE Optimizer ERROR: no. of desired minimum <= 0.\n");
    return 0;
  }
  double **optData = matOptData.getMatrix2D();

  //**/ ----------------------------------------------------------------
  //**/ search for local minima in the sample space if no response 
  //**/ surface model is used
  //**/ ----------------------------------------------------------------
  if ((optimizeFlag != 0) && (optimizeUseRS == 0) && optMethod != 10)
  {
    if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE : Search for local minima in sample space.\n");
      printf("         number of samples = %d\n", nSamples);
    }

    //**/ sort the sample based on the selected output
    psVector  vecDSort, vecISort;
    vecDSort.setLength(nSamples);
    vecISort.setLength(nSamples);
    for (ss = 0; ss < nSamples; ss++)
    {
      vecDSort[ss] = sampleOutputs[ss*nOutputs+optID];
      vecISort[ss] = (double) ss;
    }
    if (nSamples > optimizeNumPts)
      sortDbleList2(nSamples, vecDSort.getDVector(), 
                    vecISort.getDVector());

    //**/ add points to storage
    optNewCnt = nInitialX;
    ind1 = 0;
    while (optNewCnt < optimalCount && ind1 < nSamples)
    {
      minIndex = (int) vecISort[ind1];
      for (ind2 = 0; ind2 < optNewCnt; ind2++)
      {
        for (ii = 0; ii < nInputs; ii++)
          if (sampleInputs[minIndex*nInputs+ii] != 
              optData[0][ind2*nInputs+ii]) break;
        if (ii == nInputs) break;
      }
      if (ind2 == optNewCnt)
      {
        for (ii = 0; ii < nInputs; ii++)
          optData[0][optNewCnt*nInputs+ii] =
              sampleInputs[minIndex*nInputs+ii]; 
        optData[1][optNewCnt] = vecDSort[ind1]; 
        optNewCnt++;
      }
      ind1++;
    }
    optimalCount = optNewCnt;
    if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
    {
      for (ind1 = 0; ind1 < optimalCount; ind1++)
      {
        printf("\t optimization starting point %d : \n", ind1+1);
        for (ii = 0; ii < nInputs; ii++)
          printf("\t\t %16.8e\n", optData[0][ind1*nInputs+ii]);
        printf("\t\t\t\t Y = %16.8e\n", optData[1][ind1]);
      }
      printAsterisks(PL_INFO, 0);
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ search for local minima in the sample space if response surface
  //**/ model is used
  //**/ ----------------------------------------------------------------
  else if ((optimizeFlag != 0) && (optimizeUseRS == 1))
  {
    if      (nInputs <=  3) nPtsPerDim = 32;
    else if (nInputs ==  4) nPtsPerDim = 24;
    else if (nInputs ==  5) nPtsPerDim = 16;
    else if (nInputs ==  6) nPtsPerDim = 10;
    else if (nInputs ==  7) nPtsPerDim =  8;
    else if (nInputs ==  8) nPtsPerDim =  6;
    else if (nInputs ==  9) nPtsPerDim =  5;
    else if (nInputs == 10) nPtsPerDim =  4;
    else if (nInputs == 11) nPtsPerDim =  3;
    else if (nInputs == 12) nPtsPerDim =  3;
    else if (nInputs == 13) nPtsPerDim =  3;
    else if (nInputs >= 14) nPtsPerDim =  2;
    faPtr = genFAInteractive(psuadeIO, 0);
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);

    //**/ extract the one sample output ==> vecFaXOut, vecFaYOut
    psVector vecYT;
    vecYT.setLength(nSamples);
    for (ss = 0; ss < nSamples; ss++) 
      vecYT[ss] = sampleOutputs[ss*nOutputs+optID];
    double *faXOut, *faYOut;
    faPtr->genNDGridData(sampleInputs,vecYT.getDVector(),
                         &faLeng,&faXOut,&faYOut);
    psVector vecFaXOut, vecFaYOut;
    vecFaYOut.load(faLeng,faYOut);
    vecFaXOut.load(faLeng*nInputs, faXOut);
    delete [] faXOut;
    delete [] faYOut;

    //**/ search for local minima
    printAsterisks(PL_INFO, 0);
    if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
    {
      printf("PSUADE : Search for local minima in RS space.\n");
      printf("         number of samples = %d\n", nSamples);
    }
    optNewCnt = nInitialX;
    psIVector vecIT;
    vecYT.setLength(nInputs+1); 
    vecIT.setLength(nInputs+1); 
    for (ind1 = 0; ind1 < faLeng; ind1++)
    {
      ind2 = ind1;
      vecYT[0] = 1;
      for (ii = 0; ii < nInputs; ii++)
      {
        vecIT[ii] = ind2 % nPtsPerDim;
        vecYT[ii+1] = vecYT[ii] * nPtsPerDim;
        ind2 = ind2 / nPtsPerDim;
      }
      minCheck = 1;
      for (ii = 0; ii < nInputs; ii++)
      {
        if (((ind1-vecIT[ii])>0) &&  
            (vecFaYOut[ind1]>=vecFaYOut[ind1-vecIT[ii]])) minCheck = 0;
        if (((ind1+vecIT[ii])<faLeng) && 
            (vecFaYOut[ind1]>=vecFaYOut[ind1+vecIT[ii]])) minCheck = 0;
      }
      if (minCheck == 1)
      {
        for (ii = 0; ii < nInputs; ii++)
          optData[0][optNewCnt*nInputs+ii] = vecFaXOut[ind1*nInputs+ii];  
        optData[1][optNewCnt] = vecFaYOut[ind1];  
        if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
        {
          printf("PSUADE: Local minimum at X = \n");
          for (ii = 0; ii < nInputs; ii++)
            printf("\t\t %16.8e\n", vecFaXOut[ind1*nInputs+ii]);  
          printf("\tY = %16.8e\n", vecFaYOut[ind1]);  
        }
        optNewCnt++;
      }
      if (optNewCnt >= optimalCount) break;
    }
    if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
    {
      printAsterisks(PL_INFO, 0);
      printf("Total number of local minima = %d\n", optNewCnt);
      for (ind1 = 0; ind1 < optNewCnt; ind1++)
      {
        printf("Local minimum %d: \n", ind1+1);
        for (ii = 0; ii < nInputs; ii++)
          printf(" Input %4d = %16.8e\n",ii+1,optData[0][ind1*nInputs+ii]);
      }
    }
    delete faPtr;
  }

  //**/ ----------------------------------------------------------------
  //**/ optimize (find minima) 
  //**/ ----------------------------------------------------------------

  //**/MINE: SPECS
  strcpy(odata.targetFile_, specsFile);
  odata.psIO_ = psuadeIO;

  //**/ ======= use txmath =======
  if ((optimizeFlag == 1) && (optMethod == 1))
  {
#ifdef HAVE_TXMATH
    numYmin = istart = 0;
    TxMathPtr = new TxMathOptimizer();
    odata.optimalY_ = 1e35;
    totalEval = 0;
    odata.numFuncEvals_ = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if (optData[1][ind1] > optCutOff && ind1 > 0)
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[3][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);
        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.optimalX_ = new double[nInputs];
        odata.tolerance_ = optTol;
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        TxMathPtr->optimize(&odata);
        printf("\t TxMath number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        for (ii = 0; ii < nInputs; ii++)
          printf("\t optimum  X(%6d) = %16.8e\n",ii+1,
                 odata.optimalX_[ii]);
        printf("\t\t\t optimum Y = %16.8e\n",odata.optimalY_);
        for (ii = 0; ii < nInputs; ii++)
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
        optData[3][ind1] = odata.optimalY_;
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete TxMathPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t TxMath total number of function evaluations = %d\n",
           odata.numFuncEvals_);
#else
    printf("TXMATH not linked.\n");
    exit(1);
#endif
  }

  //**/ ======= use appspack =======
  else if ((optimizeFlag == 1) && (optMethod == 2))
  {
#ifdef HAVE_APPSPACK
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    APPSPACKPtr = new APPSPACKOptimizer();
    odata.optimalY_ = 1e35;
    odata.numFuncEvals_ = 0;
    totalEval = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);
        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.nSamples_ = nSamples;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        APPSPACKPtr->optimize(&odata);
        printf("\t APPSPACK number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        for (ii = 0; ii < nInputs; ii++)
          printf("\t optimum  X(%6d) = %16.8e\n",ii+1,
                 odata.optimalX_[ii]);
        printf("\t\t\t optimum Y = %16.8e\n",odata.optimalY_);
        for (ii = 0; ii < nInputs; ii++)
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
        optData[3][ind1] = odata.optimalY_;
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete APPSPACKPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t APPSPACK total number of function evaluations = %d\n",
           odata.numFuncEvals_);
#else
    printf("APPSPACK not linked.\n");
    exit(1);
#endif
  }

  //**/ ======= use minpack =======
  else if ((optimizeFlag == 1) && (optMethod == 3))
  {
#ifdef HAVE_MINPACK
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    MinpackPtr = new MinpackOptimizer();
    odata.optimalY_ = 1e35;
    odata.numFuncEvals_ = 0;
    totalEval = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);
        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        MinpackPtr->optimize(&odata);
        printf("\t Minpack number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        for (ii = 0; ii < nInputs; ii++)
          printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                 odata.optimalX_[ii]);
        printf("\t\t\t optimum Y = %16.8e\n",odata.optimalY_);
        for (ii = 0; ii < nInputs; ii++)
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
        optData[3][ind1] = odata.optimalY_;
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete MinpackPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t Minpack total number of function evaluations = %d\n",
           odata.numFuncEvals_);
#else
    printf("Minpack not linked.\n");
    exit(1);
#endif
  }

  //**/ ======= use cobyla =======
  else if ((optimizeFlag == 1) && (optMethod == 4))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    CobylaPtr = new CobylaOptimizer();
    odata.optimalY_ = 1e35;
    totalEval = 0;
    odata.numFuncEvals_ = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);

        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        CobylaPtr->optimize(&odata);
        printf("\t Cobyla number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        for (ii = 0; ii < nInputs; ii++)
        {
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
          printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                 odata.optimalX_[ii]);
        }
        optData[3][ind1] = odata.optimalY_;
        printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete CobylaPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t Cobyla total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ======= use sm =======
  else if ((optimizeFlag == 1) && (optMethod == 5))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    SMPtr = new SMOptimizer();
    odata.optimalY_ = 1e35;
    totalEval = 0;
    odata.numFuncEvals_ = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);
                                                                             
      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n",
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);
                                                                                
        odata.initialX_ = &(optData[0][ind1*nInputs]);
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        SMPtr->optimize(&odata);
        printf("\t SM number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        for (ii = 0; ii < nInputs; ii++)
        {
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
          printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                 odata.optimalX_[ii]);
        }
        optData[3][ind1] = odata.optimalY_;
        printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete SMPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0)
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t SM total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ======= use mm =======
  else if ((optimizeFlag == 1) && (optMethod == 6 || optMethod == 7))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    MMPtr = new MMOptimizer();
    odata.optimalY_ = 1e35;
    odata.numFuncEvals_ = 0;
    odata.intData_ = 0;
    totalEval = 0;
    int totalFEval = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);
                                                                             
      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n",
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);
                                                                                
        odata.initialX_ = &(optData[0][ind1*nInputs]);
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        if (optMethod == 7)
        {
          strcpy(sparam,"setAdaptive");
          MMPtr->setParam(sparam);
        } 
        MMPtr->optimize(&odata);
        printf("\t MM number of fine   function evaluations = %d\n",
               odata.intData_-totalFEval);
        printf("\t MM number of coarse function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        totalFEval = odata.intData_;
        for (ii = 0; ii < nInputs; ii++)
        {
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
          printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                 odata.optimalX_[ii]);
        }
        optData[3][ind1] = odata.optimalY_;
        printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    printf("\t MM total number of fine   function evaluations = %d\n",
           odata.intData_);
    printf("\t MM total number of coarse function evaluations = %d\n",
           odata.numFuncEvals_);
    delete MMPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0)
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
  }

  //**/ ======= use bobyqa =======
  else if ((optimizeFlag == 1) && (optMethod == 8))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    BobyqaPtr = new BobyqaOptimizer();
    odata.optimalY_ = 1e35;
    odata.numFuncEvals_ = 0;
    totalEval = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);

        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        BobyqaPtr->optimize(&odata);
        printf("\t Bobyqa number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        for (ii = 0; ii < nInputs; ii++)
        {
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
          printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                 odata.optimalX_[ii]);
        }
        optData[3][ind1] = odata.optimalY_;
        printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete BobyqaPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t Bobyqa total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ======= use SCE =======
  else if ((optimizeFlag == 1) && (optMethod == 9))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    SCEPtr = new SCEOptimizer();
    odata.optimalY_ = 1e35;
    odata.numFuncEvals_ = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
      {
        printAsterisks(PL_INFO, 0);
        printf("PSUADE OPTIMIZATION %d (%d) : \n",ind1+1,optimalCount);
      }
      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
          printf("skip optimization (%16.8e > %16.8e) : \n", 
                 optData[1][ind1], optCutOff);
      }
      else
      {
        if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
        {
          for (ii = 0; ii < nInputs; ii++)
            printf("\t starting X(%6d) = %16.8e\n",ii+1,
                   optData[0][ind1*nInputs+ii]);
          printf("\t starting Y = %16.8e\n",optData[1][ind1]);
        }
        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        SCEPtr->optimize(&odata);
        if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
          printf("\t SCE number of function evaluations = %d\n",
                 odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        for (ii = 0; ii < nInputs; ii++)
        {
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
          if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
            printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                   odata.optimalX_[ii]);
        }
        optData[3][ind1] = odata.optimalY_;
        if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
          printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
        printAsterisks(PL_INFO, 0);
    }
    delete SCEPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t SCE total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ======= use multiobjective optimizer =======
  else if ((optimizeFlag == 1) && (optMethod == 10))
  {
    printAsterisks(PL_INFO, 0);
    printf("PSUADE OPTIMIZATION: \n");
    for (ii = 0; ii < nInputs; ii++)
      printf("\t starting X(%6d) = %16.8e\n",ii+1,
             optData[0][ii]);
    printf("\t starting Y = %16.8e\n",optData[1][0]);

    odata.initialX_ = optData[0]; 
    funcIO->setSynchronousMode();
    odata.funcIO_ = funcIO;
    odata.nInputs_ = nInputs;
    odata.nOutputs_ = nOutputs;
    odata.outputID_ = optID;
    odata.lowerBounds_ = iLowerB;
    odata.upperBounds_ = iUpperB;
    odata.outputLevel_ = optPrintLevel;
    odata.tolerance_   = optTol;
    odata.numFuncEvals_ = 0;
    odata.optimalX_ = new double[nInputs];
    MooPtr = new MultiObjectiveOptimizer();
    MooPtr->optimize(&odata);
    delete [] odata.optimalX_;
    odata.optimalX_ = NULL;
    odata.initialX_ = NULL;
    odata.lowerBounds_ = NULL;
    odata.upperBounds_ = NULL;
    odata.funcIO_ = NULL;
    delete MooPtr;
    printAsterisks(PL_INFO, 0);
    printf("\t MOO total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ======= use OUU1 optimizer =======
  else if ((optimizeFlag == 1) && (optMethod == 12))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    OUU1Ptr = new OUU1Optimizer();
    odata.optimalY_ = 1e35;
    totalEval = 0;
    odata.numFuncEvals_ = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);

        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        odata.intData_ = -1;
        OUU1Ptr->optimize(&odata);
        printf("\t OUU1Optimizer number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        totalEval += odata.numFuncEvals_;
        M1 = odata.intData_;
        if (M1 <= 0) M1 = nInputs;
        for (ii = 0; ii < M1; ii++)
        {
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
          printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                 odata.optimalX_[ii]);
        }
        optData[3][ind1] = odata.optimalY_;
        printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        //**/if (optData[3][ind1] <= optFmin) numYmin++;
        numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete OUU1Ptr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t OUU1 total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ======= use OUU2 optimizer =======
  else if ((optimizeFlag == 1) && (optMethod == 13))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    OUU2Ptr = new OUU2Optimizer();
    odata.optimalY_ = 1e35;
    odata.numFuncEvals_ = 0;
    totalEval = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);

        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        odata.intData_ = -1;
        OUU2Ptr->optimize(&odata);
        printf("\t OUU2Optimizer number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        M1 = odata.intData_;
        if (M1 <= 0) M1 = nInputs;
        for (ii = 0; ii < M1; ii++)
        {
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
          printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                 odata.optimalX_[ii]);
        }
        optData[3][ind1] = odata.optimalY_;
        printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        //**/if (optData[3][ind1] <= optFmin) numYmin++;
        numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete OUU2Ptr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t OUU2 total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }
 
  //**/ ======= use OUU optimizer =======
  else if ((optimizeFlag == 1) && ((optMethod == 11) || 
           (optMethod == 16) || (optMethod == 17) || 
           (optMethod == 18) || (optMethod == 20)))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    OUUPtr = new OUUOptimizer();
    //**/ if unconstrained optimization, set it
    if (optMethod == 11) OUUPtr->setLocalOptimizer(0);
    if (optMethod == 16) OUUPtr->setLocalOptimizer(1);
    if (optMethod == 17) OUUPtr->setLocalOptimizer(0);
    if (optMethod == 18) OUUPtr->setLocalOptimizer(2);
    if (optMethod == 20) OUUPtr->setLocalOptimizer(3);
    odata.optimalY_ = 1e35;
    totalEval = 0;
    odata.numFuncEvals_ = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);

        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        odata.intData_ = -1;
        OUUPtr->optimize(&odata);
        printf("\t OUUOptimizer number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        M1 = odata.intData_;
        if (M1 <= 0) M1 = nInputs;
        for (ii = 0; ii < M1; ii++)
        {
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
          printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                 odata.optimalX_[ii]);
        }
        optData[3][ind1] = odata.optimalY_;
        printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        //**/if (optData[3][ind1] <= optFmin) numYmin++;
        numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete OUUPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t OUU total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ======= use Lincoa =======
  else if ((optimizeFlag == 1) && (optMethod == 14))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    LincoaPtr = new LincoaOptimizer();
    odata.optimalY_ = 1e35;
    odata.numFuncEvals_ = 0;
    totalEval = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);

        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        LincoaPtr->optimize(&odata);
        printf("\t Lincoa number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        for (ii = 0; ii < nInputs; ii++)
        {
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
          printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                 odata.optimalX_[ii]);
        }
        optData[3][ind1] = odata.optimalY_;
        printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete LincoaPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t Lincoa total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ======= use Newuoa =======
  else if ((optimizeFlag == 1) && (optMethod == 15))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    NewuoaPtr = new NewuoaOptimizer();
    odata.optimalY_ = 1e35;
    odata.numFuncEvals_ = 0;
    totalEval = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);

        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        NewuoaPtr->optimize(&odata);
        printf("\t Newuoa number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        for (ii = 0; ii < nInputs; ii++)
        {
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
          printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                 odata.optimalX_[ii]);
        }
        optData[3][ind1] = odata.optimalY_;
        printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete NewuoaPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t Newuoa total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ======= use lbfgs =======
  else if ((optimizeFlag == 1) && (optMethod == 19))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    lbfgsPtr = new LBFGSOptimizer();
    odata.optimalY_ = 1e35;
    totalEval = 0;
    odata.numFuncEvals_ = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          printf("\t starting X(%6d) = %16.8e\n",ii+1,
                 optData[0][ind1*nInputs+ii]);
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);

        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        lbfgsPtr->optimize(&odata);
        printf("\t LBFGS number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        for (ii = 0; ii < nInputs; ii++)
        {
          optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
          printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                 odata.optimalX_[ii]);
        }
        optData[3][ind1] = odata.optimalY_;
        printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete lbfgsPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t LBFGS total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ======= use nomad =======
  else if ((optimizeFlag == 1) && (optMethod == 21))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    nomadPtr = new NomadOptimizer();
    odata.optimalY_ = 1e35;
    odata.numFuncEvals_ = 0;
    totalEval = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
        {
          ss = (int) optData[0][ind1*nInputs+ii];
          if (optData[0][ind1*nInputs+ii] - ss == 0)
            printf("\t starting X(%6d) = %d\n",ii+1,ss);
          else
            printf("\t starting X(%6d) = %16.8e\n",ii+1,
                   optData[0][ind1*nInputs+ii]);
        }
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);

        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        nomadPtr->optimize(&odata);
        printf("\t Nomad number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        if (odata.optimalY_ != 1.0e50)
        {
          for (ii = 0; ii < nInputs; ii++)
          {
            optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
            ss = (int) optData[2][ind1*nInputs+ii];
            if (optData[2][ind1*nInputs+ii] - ss == 0)
              printf("\t optimum  X(%6d) = %d\n", ii+1, ss);
            else
              printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                     odata.optimalX_[ii]);
          }
          optData[3][ind1] = odata.optimalY_;
          printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        }
        else
        {
          printf("** PSUADE NOMAD INFO: no feasible solution found.\n");
        }
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete nomadPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t Nomad total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ======= use ouu/minlp =======
  else if ((optimizeFlag == 1) && (optMethod == 22))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    minlpPtr = new OUUMinlpOptimizer();
    odata.optimalY_ = 1e35;
    odata.numFuncEvals_ = 0;
    totalEval = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
        {
          ss = (int) optData[0][ind1*nInputs+ii];
          if (optData[0][ind1*nInputs+ii] - ss == 0)
            printf("\t starting X(%6d) = %d\n",ii+1,ss);
          else
            printf("\t starting X(%6d) = %16.8e\n",ii+1,
                   optData[0][ind1*nInputs+ii]);
        }
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);

        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        minlpPtr->optimize(&odata);
        printf("\t OUU/MINLP number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        if (odata.optimalY_ != 1.0e50)
        {
          for (ii = 0; ii < nInputs; ii++)
          {
            optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
            ss = (int) optData[2][ind1*nInputs+ii];
            if (optData[2][ind1*nInputs+ii] - ss == 0)
              printf("\t optimum  X(%6d) = %d\n", ii+1, ss);
            else
              printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                     odata.optimalX_[ii]);
          }
          optData[3][ind1] = odata.optimalY_;
          printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        }
        else
        {
          printf("** PSUADE OUU/MINLP INFO: no feasible solution found.\n");
        }
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete minlpPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t OUU/MINLP total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ======= use particle swarm =======
  else if ((optimizeFlag == 1) && (optMethod == 23))
  {
    numYmin = 0;
    istart = optimalCount - optimizeNumPts;
    if (istart < 0) istart = 0;
    psoPtr = new PSOptimizer();
    odata.optimalY_ = 1e35;
    odata.numFuncEvals_ = 0;
    psuadeIO->getParameter("ana_opt_fmin", pPtr);
    odata.targetY_ = pPtr.dbleData_;
    totalEval = 0;
    for (ind1 = istart; ind1 < optimalCount; ind1++)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE OPTIMIZATION %d (%d) : \n", ind1+1, optimalCount);

      if ((optData[1][ind1] > optCutOff) && (ind1 > 0))
      {
        printf("skip optimization (%16.8e > %16.8e) : \n", 
               optData[1][ind1], optCutOff);
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
        {
          ss = (int) optData[0][ind1*nInputs+ii];
          if (optData[0][ind1*nInputs+ii] - ss == 0)
            printf("\t starting X(%6d) = %d\n",ii+1,ss);
          else
            printf("\t starting X(%6d) = %16.8e\n",ii+1,
                   optData[0][ind1*nInputs+ii]);
        }
        printf("\t starting Y = %16.8e\n",optData[1][ind1]);

        odata.initialX_ = &(optData[0][ind1*nInputs]); 
        funcIO->setSynchronousMode();
        odata.funcIO_ = funcIO;
        odata.nInputs_ = nInputs;
        odata.nOutputs_ = nOutputs;
        odata.outputID_ = optID;
        odata.lowerBounds_ = iLowerB;
        odata.upperBounds_ = iUpperB;
        odata.outputLevel_ = optPrintLevel;
        odata.tolerance_   = optTol;
        odata.optimalX_ = new double[nInputs];
        if      (optimalCount-istart == 1) odata.setOptDriver_ = 3;
        else if (ind1 == istart)           odata.setOptDriver_ = 1;
        else if (ind1 == optimalCount-1)   odata.setOptDriver_ = 2;
        else                               odata.setOptDriver_ = 0;
        psoPtr->optimize(&odata);
        printf("\t PSO number of function evaluations = %d\n",
               odata.numFuncEvals_-totalEval);
        totalEval = odata.numFuncEvals_;
        if (odata.optimalY_ != 1.0e50)
        {
          for (ii = 0; ii < nInputs; ii++)
          {
            optData[2][ind1*nInputs+ii] = odata.optimalX_[ii];
            ss = (int) optData[2][ind1*nInputs+ii];
            if (optData[2][ind1*nInputs+ii] - ss == 0)
              printf("\t optimum  X(%6d) = %d\n", ii+1, ss);
            else
              printf("\t optimum  X(%6d) = %16.8e\n", ii+1,
                     odata.optimalX_[ii]);
          }
          optData[3][ind1] = odata.optimalY_;
          printf("\t\t\t optimum Y = %16.8e\n", odata.optimalY_);
        }
        else
        {
          printf("** PSUADE PSO INFO: no feasible solution found.\n");
        }
        delete [] odata.optimalX_;
        odata.optimalX_ = NULL;
        odata.initialX_ = NULL;
        odata.lowerBounds_ = NULL;
        odata.upperBounds_ = NULL;
        odata.funcIO_ = NULL;
        if (optData[3][ind1] <= optFmin) numYmin++;
      }
      printAsterisks(PL_INFO, 0);
    }
    delete psoPtr;
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optFmin != 0.0) 
      {
        if (optData[3][ind1] <= optFmin) returnFlag++;
      }
      else
      {
        if (PABS(optData[3][ind1]) <= 1.0e-6) returnFlag++;
      }
    }
    printf("\t PSO total number of function evaluations = %d\n",
           odata.numFuncEvals_);
  }

  //**/ ----------------------------------------------------------------
  //**/ display optimization results
  //**/ ----------------------------------------------------------------
  int nTrack=0, nn, count;
  double minY, range;
  psIVector vecTrackInds;
  psVector  vecTrackYmin;
  vecTrackInds.setLength(optNumFmin);
  vecTrackYmin.setLength(optNumFmin);
  if ((optimalCount > 0) && (optMethod != 0) && (optMethod != 10))
  {
    ind2 = 0;
    Ymin = 1e36;
    if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
    {
      printAsterisks(PL_INFO, 0);
      printf("PSUADE Optimization Results.\n");
    }
    for (ind1 = 0; ind1 < optimalCount; ind1++)
    {
      if (optData[3][ind1] != PSUADE_UNDEFINED)
      {
        if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
          printf("PSUADE Optimization : local optima %d (%d) - \n",
                 ind1+1, optimalCount);
        ind2 = M1;
        if (ind2 < 0) ind2 = nInputs;
        if (psConfig_.InteractiveIsOn() && optPrintLevel >= 0)
        {
          for (ii = 0; ii < ind2; ii++)
          {
            ss = (int) optData[2][ind1*nInputs+ii];
            if (optData[2][ind1*nInputs+ii] - ss == 0)
              printf("\t\tX %5d = %d\n", ii+1,ss);
            else 
              printf("\t\tX %5d = %16.8e\n", ii+1,
                  optData[2][ind1*nInputs+ii]);
          }
          printf("\t\t\tYmin = %16.8e\n",optData[3][ind1]);
        }
        if (nTrack < optNumFmin) 
        {
          count = 0;
          for (nn = 0; nn < nTrack; nn++)
          {
            ind2 = vecTrackInds[nn];
            count = 0;
            for (ii = 0; ii < nInputs; ii++)
            {
              range = iUpperB[ii] - iLowerB[ii];
              if (range == 0) range = 1;
              if (PABS(optData[2][ind2*nInputs+ii]-
                       optData[2][ind1*nInputs+ii])/range<1e-8) count++;
            }
            if (count == nInputs) break;
          }
          if (count < nInputs)
          {
            vecTrackInds[nTrack] = ind1;
            vecTrackYmin[nTrack++] = optData[3][ind1];
          }
        }
        else
        {
          ind2 = 0;
          minY = vecTrackYmin[0];
          for (nn = 1; nn < nTrack; nn++)
          {
            if (vecTrackYmin[nn] > minY) 
            {
              ind2 = nn;
              minY = vecTrackYmin[nn];
            }
          }
          if (optData[3][ind1] < minY)
          {
            count = 0;
            for (ii = 0; ii < nInputs; ii++)
            {
              range = iUpperB[ii] - iLowerB[ii];
              if (range == 0) range = 1;
              if (PABS(optData[2][ind2*nInputs+ii]-
                       optData[2][ind1*nInputs+ii])/range<1e-8) count++;
            }
            if (count < nInputs)
            {
              vecTrackInds[ind2] = ind1;
              vecTrackYmin[ind2] = optData[3][ind1];
            }
          }
        }
      }
    } 
    if (optimalCount > 0 && optPrintLevel >= 0 && psConfig_.InteractiveIsOn())
    {
      printf("##################################################\n");
      printf("PSUADE OPTIMIZATION : CURRENT GLOBAL MINIMUM - \n");
      for (nn = 0; nn < nTrack; nn++)
      {
        if (M1 < 0) M1 = nInputs;
        ind2 = vecTrackInds[nn];
        for (ii = 0; ii < M1; ii++)
        {
          ind1 = (int) optData[2][ind2*nInputs+ii];
          if (optData[2][ind2*nInputs+ii] - ind1 == 0)
            printf("\t\tX %5d = %d\n",ii+1,ind1);
          else
            printf("\t\tX %5d = %16.8e\n",ii+1,
                   optData[2][ind2*nInputs+ii]);
        }
        printf("\t\t\tYmin = %16.8e\n",optData[3][ind2]);
        if (nn < nTrack-1)
          printf("--------------------------------------------------\n");
      }
      printf("##################################################\n");
    }
    if (optPrintLevel >= 0 && psConfig_.InteractiveIsOn())
      printAsterisks(PL_INFO, 0);
  }
  if (optimizeFlag == 1 && returnFlag >= optNumFmin && optMethod != 10) 
  {
    if (optPrintLevel >= 0) 
      printf("PSUADE Optimization : desired minimum found.\n");
    return returnFlag;
  }
  else returnFlag = 0;
  return returnFlag;
}

// ************************************************************************
// assign operator
// ------------------------------------------------------------------------
Optimizer& Optimizer::operator=(const Optimizer &)
{
  printf("Optimizer operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

