// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team. 
// All rights reserved.
//
// Please see the COPYRIGHT and LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public 
// License.
//
// PSUADE is free software; you can redistribute it and/or modify it under 
// the terms of the GNU Lesser General Public License (as published by the 
// Free Software Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT 
// ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY 
// or FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of 
// the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Functions for the class TwoParamAnalyzer 
// AUTHOR : CHARLES TONG
// DATE   : updated in 2006
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#include "TwoParamAnalyzer.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "sysdef.h"
#include "psMatrix.h"
#include "PsuadeData.h"
#include "pData.h"
#include "RSConstraints.h"
#include "PrintingTS.h"
using namespace std;

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
TwoParamAnalyzer::TwoParamAnalyzer(): Analyzer(),nInputs_(0),outputMean_(0),
                  outputStd_(0)
{
  setName("TwoParam");
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
TwoParamAnalyzer::~TwoParamAnalyzer()
{
}

// ************************************************************************
// perform VCE analysis for two parameters
// ------------------------------------------------------------------------
double TwoParamAnalyzer::analyze(aData &adata)
{
  int    ii, ii2, jj, ir, ss, ss2, repID, subID, ncount, index;
  int    symIndex, whichOutput, nReps, nReps1;
  int    nSubs, iZero=0, iOne=1, totalCnt, validnReps;
  double fixedInput, fixedInput2, ddata;
  char   pString[5000], winput[5000];
  FILE   *fp=NULL;
  pData     pdata;
  psVector  vecYY, vecLocalY;
  psMatrix  matBSVCEs;

  //**/ ---------------------------------------------------------------
  //**/ fetch data
  //**/ ---------------------------------------------------------------
  nInputs_ = adata.nInputs_;
  int nInputs  = nInputs_;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;
  double *xLower  = adata.iLowerB_;
  double *xUpper  = adata.iUpperB_;
  double *XIn     = adata.sampleInputs_;
  double *YIn     = adata.sampleOutputs_;
  int outputID    = adata.outputID_;
  int printLevel_ = adata.printLevel_;
  int nSubSamples = adata.nSubSamples_;
  int *pdfFlags   = adata.inputPDFs_;
  PsuadeData *ioPtr = adata.ioPtr_;

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nOutputs <= 0 || nSamples <= 0)
  {
    printOutTS(PL_ERROR,"TwoParamEffect ERROR: Invalid arguments.\n");
    printOutTS(PL_ERROR,"    nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR,"    nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR,"    nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 
  if (nInputs <= 2)
  {
    printOutTS(PL_ERROR,
         "TwoParamEffect ERROR: Not available for nInputs <= 2.\n");
    return PSUADE_UNDEFINED;
  } 
  if (nSamples/nSubSamples*nSubSamples != nSamples)
  {
    printOutTS(PL_ERROR,
               "TwoParamEffect ERROR: nSamples != k*nSubSamples.\n");
    printOutTS(PL_ERROR,"    nSamples    = %d\n", nSamples);
    printOutTS(PL_ERROR,"    nSubSamples = %d\n", nSubSamples);
    return PSUADE_UNDEFINED;
  } 
  whichOutput = outputID;
  if (whichOutput >= nOutputs || whichOutput < 0)
  {
    printOutTS(PL_ERROR,"TwoParamEffect ERROR: Invalid outputID.\n");
    printOutTS(PL_ERROR,"    nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR,"    outputID = %d\n", whichOutput+1);
    printOutTS(PL_ERROR,"    outputID reset to 1\n");
    whichOutput = 0;
  }
  if (ioPtr == NULL)
  {
    printOutTS(PL_ERROR,"TwoParamEffect ERROR: No data (PsuadeData).\n");
    return PSUADE_UNDEFINED;
  }
  int status = 0;
  for (ss = 0; ss < nSamples; ss++)
    if (YIn[nOutputs*ss+whichOutput] > 0.9*PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR, 
               "TwoParamEffect ERROR: Some outputs are undefined.\n");
    printOutTS(PL_ERROR, 
               "        Prune the undefined sample points and re-do.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ---------------------------------------------------------------
  //**/ check PDF and correlation (this method does not support
  //**/ explicit correlation)
  //**/ ---------------------------------------------------------------
  for (ii = 0; ii < nInputs; ii++)
  {
    if (pdfFlags != NULL && pdfFlags[ii] != PSUADE_PDF_UNIFORM)
    {
      printOutTS(PL_INFO,
        "* TwoParamEffect INFO: Some inputs have non-uniform PDFs.\n");
      printOutTS(PL_INFO,
        "*      However, they will not be relevant in this analysis\n");
      printOutTS(PL_INFO,
        "*      (since the sample should have been created with the\n");
      printOutTS(PL_INFO,
        "*      desired distributions.)\n");
      break;
    }
  }
  pData pCorMat;
  ioPtr->getParameter("input_cor_matrix", pCorMat);
  psMatrix *corMatp = (psMatrix *) pCorMat.psObject_;
  for (ii = 0; ii < nInputs; ii++)
  {
    for (jj = 0; jj < ii; jj++)
    {
      if (corMatp->getEntry(ii,jj) != 0.0)
      {
        printOutTS(PL_ERROR,
          "* TwoParamEffect INFO: This method should not be used if\n");
        printOutTS(PL_ERROR,
          "*      additional input correlations not already embedded\n");
        printOutTS(PL_ERROR,
             "*   in the sample are needed.\n");
        return PSUADE_UNDEFINED;
      }     
    }
  }

  //**/ ---------------------------------------------------------------
  // clean up
  //**/ ---------------------------------------------------------------
  VecMainEffects_.clean();
  Mat2ParamEffects_.setFormat(PS_MAT2D);
  Mat2ParamEffects_.setDim(nInputs, nInputs);

  //**/ ---------------------------------------------------------------
  //**/ filter based on constraints
  //**/ ---------------------------------------------------------------
  RSConstraints *constrPtr=NULL;
  if (ioPtr != NULL)
  {
    constrPtr = new RSConstraints();
    constrPtr->genConstraints(ioPtr);
    int nConstr = constrPtr->getNumConstraints();
    if (nConstr == 0)
    {
      delete constrPtr;
      constrPtr = NULL;
    }
    else
      printf("RSMAnalysis INFO: %d constraints detected.\n",nConstr);
  }
  double *localX = XIn;
  vecLocalY.setLength(nSamples);
  ncount = 0;
  for (ss = 0; ss < nSamples; ss++)
  {
    vecLocalY[ss] = YIn[nOutputs*ss+whichOutput];
    status = 1;
    if (constrPtr != NULL)
      ddata = constrPtr->evaluate(&(localX[ss*nInputs]), vecLocalY[ss], 
                                  status);
    if (status == 0) vecLocalY[ss] = PSUADE_UNDEFINED;
    else             ncount++;
  }
  if (ncount == 0)
  {
    printOutTS(PL_ERROR,"TwoParamEffect ERROR: No valid sample point.\n");
    printOutTS(PL_ERROR,"    nSamples before filtering = %d\n",nSamples);
    printOutTS(PL_ERROR,"    nSamples after  filtering = %d\n",ncount);
    printOutTS(PL_ERROR,
         "    INFO: check your data file for undefined's (1e35)\n");
    return 1.0;
  }
  if (ncount != nSamples)
  {
    printOutTS(PL_INFO,
         "* TwoParamEffect INFO: CONSTRAINTS HAVE BEEN APPLIED.\n");
    printOutTS(PL_INFO,
         "* TwoParamEffect: nSamples before filtering = %d\n", nSamples);
    printOutTS(PL_INFO,
         "* TwoParamEffect: nSamples after filtering  = %d\n", ncount);
  }

  //**/ ---------------------------------------------------------------
  //**/ compute basic statistics
  //**/ ---------------------------------------------------------------
  double   aMean, aVariance;
  psVector vecVCE;
  computeMeanVariance(nInputs,1,nSamples,vecLocalY.getDVector(),&aMean,
                      &aVariance,0);
  outputMean_ = aMean;
  outputStd_  = sqrt(aVariance);

  if (PABS(aVariance) < 1.0e-15)
  {
    printOutTS(PL_INFO, 
         "* =====> TwoParamEffect: variance = %11.4e (sd =%10.4e)\n",
         aVariance, sqrt(aVariance));
    printOutTS(PL_ERROR, 
         "TwoParamEffect INFO: std dev too small ==> terminate.\n");
    return PSUADE_UNDEFINED;
  }
  if (nSamples > 2000000)
  {
    printf("INFO: Sample size is large ==> Crude (2D binning) analysis.\n");
    vecVCE.setLength(nInputs*(nInputs+1));
    status = computeVCECrude(nInputs, nSamples, localX, 
                  vecLocalY.getDVector(), aVariance,
                   xLower, xUpper, vecVCE.getDVector());
    //**/ ---------------------------------------------------------
    //**/ return more detailed data
    //**/ ---------------------------------------------------------
    if (status == 0)
    {
      pData *pPtr = ioPtr->getAuxData();
      pPtr->nDbles_ = nInputs * nInputs;
      pPtr->dbleArray_ = new double[nInputs * (nInputs+1)];
      for (ii = 0; ii < nInputs*(nInputs+1); ii++)
        pPtr->dbleArray_[ii] = vecVCE[ii];
      pPtr->dbleData_ = aVariance;
      //genPlots(adata);
    }
    return 1.0;
  }
  if (psConfig_.AnaExpertModeIsOn())
  {
    printOutTS(PL_INFO,"This method normally operates on replicated");
    printOutTS(PL_INFO," orthogonal array (ROA)\n");
    printOutTS(PL_INFO,"samples. If the sample is not ROA, it ");
    printOutTS(PL_INFO,"uses a crude method to compute\n");
    printOutTS(PL_INFO,"sensitivities. However, even though it is");
    printOutTS(PL_INFO," ROA, you can still force it\n");
    printOutTS(PL_INFO, 
     "to use the crude method (when analysis expert mode is on).\n");
    printOutTS(PL_INFO, "Perform crude analysis? (y or n) ");
    fgets(pString,400,stdin);
    if (pString[0] == 'y')
    {
      vecVCE.setLength(nInputs*(nInputs+1));
      status = computeVCECrude(nInputs, nSamples, localX, 
                   vecLocalY.getDVector(), aVariance,
                   xLower, xUpper, vecVCE.getDVector());
      //**/ ---------------------------------------------------------
      //**/ return more detailed data
      //**/ ---------------------------------------------------------
      if (status == 0)
      {
        pData *pPtr = ioPtr->getAuxData();
        pPtr->nDbles_ = nInputs * nInputs;
        pPtr->dbleArray_ = new double[nInputs * (nInputs+1)];
        for (ii = 0; ii < nInputs*(nInputs+1); ii++)
          pPtr->dbleArray_[ii] = vecVCE[ii];
        pPtr->dbleData_ = aVariance;
        //genPlots(adata);
      }
      return 1.0;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ allocate space to store information
  //**/ ---------------------------------------------------------------
  psIVector vecBins;
  psVector  vecVCEMeans, vecVCEVars;
  psVector  vecVarMeans, vecMeanVars, vecVarVars, vecMEs, vecSTIs;
  vecVCEMeans.setLength(nSubSamples);
  vecVCEVars.setLength(nSubSamples);
  vecBins.setLength(nSubSamples);
  vecVCE.setLength(nInputs*(nInputs+1));
  vecVarMeans.setLength(nInputs*nInputs);
  vecMeanVars.setLength(nInputs*nInputs);
  vecVarVars.setLength(nInputs*nInputs);
  vecMEs.setLength(nInputs);
  vecSTIs.setLength(nInputs);
  int errflag = 0;

  printOutTS(PL_INFO,"\n");
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,
     "*       Two-Parameter Sensitivity Analysis on Sample Data\n");
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO,
     "* Note: This method needs large samples for more ");
  printOutTS(PL_INFO,
     "accurate results\n");
  printOutTS(PL_INFO,
     "*       (e.g. tens of thousands or more depending on ");
  printOutTS(PL_INFO,
     "the functions).\n");
  printOutTS(PL_INFO,
     "*       For small to moderate sample sizes, use ");
  printOutTS(PL_INFO,
     "response surface.\n");
  printOutTS(PL_INFO,
     "* Note: This method works on replicated orthogonal ");
  printOutTS(PL_INFO,
     "arrays. For random\n");
  printOutTS(PL_INFO,
     "*       samples, a crude analysis will be performed.\n");
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO,"* Sample size       = %10d\n",nSamples);
  printOutTS(PL_INFO,"* Number of nInputs = %10d\n",nInputs);
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO,"Output %d\n", whichOutput+1);
  printOutTS(PL_INFO,
    "* ====> TwoParamEffect: mean     = %11.4e\n", aMean);
  printOutTS(PL_INFO,
       "* ====> TwoParamEffect: variance = %11.4e (sd = %11.4e)\n",
       aVariance, sqrt(aVariance));

  psVector vecTX, vecTW, vecTY;
  vecTX.setLength(nSamples);
  vecTW.setLength(nSamples);
  vecTY.setLength(nSamples);

  //**/ ---------------------------------------------------------------
  //**/ now compute VCE
  //**/ ---------------------------------------------------------------
  for (ii = 0; ii < nInputs; ii++)
  {
    for (ii2 = ii+1; ii2 < nInputs; ii2++)
    {
      if (printLevel_ > 4 || nSamples > 100000)
        printOutTS(PL_DUMP,
            "TwoParamEffect:: Input pairs = %d %d\n", ii+1, ii2+1);
      for (ss = 0; ss < nSamples; ss++)
      {
        vecTX[ss] = localX[nInputs*ss+ii];
        vecTW[ss] = localX[nInputs*ss+ii2];
        vecTY[ss] = vecLocalY[ss];
      }

      if (printLevel_ > 4)
        printf("TwoParamEffect:: Input pairs = %d %d Sorting\n",
               ii+1,ii2+1);
      sortDbleList3(nSamples, vecTX.getDVector(), vecTW.getDVector(), 
                    vecTY.getDVector());

      for (ss = 1; ss < nSamples; ss++)
        if (PABS(vecTX[ss]-vecTX[ss-1]) > 1.0E-10)
          break;
      nReps1 = ss;
      if (nReps1 <= 1)
      {
        printOutTS(PL_INFO,
           "TwoParamEffect INFO: nReps < 1 for input %d.\n",ii+1);
        printOutTS(PL_INFO,
           "        ==> not replicated orthogonal array nor Factorial\n");
        printOutTS(PL_INFO,
           "        ==> crude (2D binning) 2-way interaction analysis.\n");
        status = computeVCECrude(nInputs, nSamples, localX, 
                       vecLocalY.getDVector(), aVariance,
                       xLower, xUpper, vecVCE.getDVector());
        //**/ ---------------------------------------------------------
        //**/ return more detailed data
        //**/ ---------------------------------------------------------
        if (status == 0)
        {
          pData *pPtr = ioPtr->getAuxData();
          pPtr->nDbles_ = nInputs * nInputs;
          pPtr->dbleArray_ = new double[nInputs * (nInputs+1)];
          for (ii = 0; ii < nInputs*(nInputs+1); ii++) 
            pPtr->dbleArray_[ii] = vecVCE[ii];
          pPtr->dbleData_ = aVariance;
          //genPlots(adata);
        }
        return 1.0;
      }

      for (ss = 0; ss < nSamples; ss+=nReps1)
      {
        fixedInput = vecTX[ss];
        ncount = 0;
        for (ss2 = 0; ss2 < nReps1; ss2++)
          if (PABS(vecTX[ss+ss2]-fixedInput) < 1.0E-10)
            ncount++;
        if (ncount != nReps1) errflag = 1;
      }
      if (errflag != 0)
      {
        printOutTS(PL_WARN,
            "TwoParamEffect WARNING(2): replications not satisified.\n");
        printOutTS(PL_WARN,
            "    Are you using replicated orthogonal array or Factorial?\n");
        printOutTS(PL_WARN,
            "    If so, you need to use > 1 replications. A crude\n");
        printOutTS(PL_WARN,
            "    2-way interaction analysis will be done instead.\n");
        status = computeVCECrude(nInputs, nSamples, localX, 
                       vecLocalY.getDVector(), aVariance, xLower, 
                       xUpper, vecVCE.getDVector());
        //**/ ---------------------------------------------------------
        //**/ return more detailed data
        //**/ ---------------------------------------------------------
        if (status == 0)
        {
          pData *pPtr = ioPtr->getAuxData();
          pPtr->nDbles_ = nInputs * nInputs;
          pPtr->dbleArray_ = new double[nInputs * (nInputs+1)];
          for (ii = 0; ii < nInputs*(nInputs+1); ii++) 
            pPtr->dbleArray_[ii] = vecVCE[ii];
          pPtr->dbleData_ = aVariance;
          //genPlots(adata);
        }
        return 1.0;
      }

      double *tws = vecTW.getDVector();
      double *tys = vecTY.getDVector();
      if (printLevel_ > 4)
        printf("TwoParamEffect:: Input pairs = %d %d Sorting 2\n",
               ii+1,ii2+1);
      for (ss = 0; ss < nSamples; ss+=nReps1)
        sortDbleList2(nReps1,&tws[ss],&tys[ss]);

      for (ss = 1; ss < nReps1; ss++)
        if (PABS(vecTW[ss]-vecTW[ss-1]) > 1.0E-10)
          break;
      nReps = ss;
      if (nReps <= 1 || (nReps1/nReps*nReps != nReps1))
      {
        printOutTS(PL_WARN,
           "TwoParamEffect WARNING(3): replications not satisified.\n");
        printOutTS(PL_WARN,
            " Are you using replicated orthogonal array or Factorial?\n");
        printOutTS(PL_WARN,
            " If so, you need to use > 1 replications.\n");
        printOutTS(PL_WARN,
            " A 2D-binning interaction analysis will be done instead.\n");
        status = computeVCECrude(nInputs, nSamples, localX, 
                       vecLocalY.getDVector(), aVariance,
                       xLower, xUpper, vecVCE.getDVector());
        //**/ ---------------------------------------------------------
        //**/ return more detailed data
        //**/ ---------------------------------------------------------
        if (status == 0)
        {
          pData *pPtr = ioPtr->getAuxData();
          pPtr->nDbles_ = nInputs * nInputs;
          pPtr->dbleArray_ = new double[nInputs * (nInputs+1)];
          for (ii = 0; ii < nInputs*(nInputs+1); ii++)   
            pPtr->dbleArray_[ii] = vecVCE[ii];
          pPtr->dbleData_ = aVariance;
          //genPlots(adata);
        }
        return 1.0;
      }

      if (printLevel_ > 4)
        printf("TwoParamEffect:: Input pairs = %d %d Processing\n",
               ii+1,ii2+1);
      for (ss = 0; ss < nSamples; ss+=nReps1)
      {
        for (ss2 = 0; ss2 < nReps1; ss2+=nReps)
        {
          ncount = 1;
          fixedInput = vecTW[ss+ss2];
          for (repID = 1; repID < nReps; repID++)
          {
            if (PABS(vecTW[ss+ss2+repID]-fixedInput)<1.0E-10)
              ncount++;
          }
          if (ncount != nReps)
          {
            printOutTS(PL_WARN,
                 "TwoParamEffect WARNING(3): sample not rOA/FACT.\n");
            errflag = 1;
          }
        }
        if (errflag != 0) break;
      }
      if (errflag != 0)
      {
        printOutTS(PL_WARN,
            "TwoParamEffect WARNING(4): replications not satisified.\n");
        printOutTS(PL_WARN,
            "    Are you using replicated orthogonal array or Factorial?\n");
        printOutTS(PL_WARN,
            "    If so, you need to use > 1 replications.\n");
        printOutTS(PL_WARN,
            "A 2D-binning interaction analysis will be done instead.\n");
        status = computeVCECrude(nInputs, nSamples, localX, 
                      vecLocalY.getDVector(), aVariance,
                      xLower, xUpper, vecVCE.getDVector());
        //**/ ---------------------------------------------------------
        //**/ return more detailed data
        //**/ ---------------------------------------------------------
        if (status == 0)
        {
          pData *pPtr = ioPtr->getAuxData();
          pPtr->nDbles_ = nInputs * nInputs;
          pPtr->dbleArray_ = new double[nInputs * (nInputs+1)];
          for (ii = 0; ii < nInputs*(nInputs+1); ii++)
            pPtr->dbleArray_[ii] = vecVCE[ii];
          pPtr->dbleData_ = aVariance;
          //genPlots(adata);
        }
        return 1.0;
      }

      for (ss = 0; ss < nSamples; ss+=nReps)
      {
        symIndex = ss / nReps;
        fixedInput  = vecTX[ss];
        fixedInput2 = vecTW[ss];
        vecVCEMeans[symIndex] = 0.0;
        validnReps = 0;
        for (repID = 0; repID < nReps; repID++)
        {
          if (PABS(vecTX[ss+repID] - fixedInput) < 1.0E-10 &&
              PABS(vecTW[ss+repID] - fixedInput2) < 1.0E-10)
          {
            if (vecTY[ss+repID] < 0.9*PSUADE_UNDEFINED)
            {
              validnReps++;
              vecVCEMeans[symIndex] += vecTY[ss+repID];
            }
          }
          else
          {
            printOutTS(PL_ERROR,
                 "TwoParamEffect ERROR: consult developers.\n");
            exit(1);
          }
        }
        vecBins[symIndex] = validnReps;
        if (validnReps > 0) vecVCEMeans[symIndex] /= (double) validnReps;
        vecVCEVars[symIndex] = 0.0;
        for (repID = 0; repID < nReps; repID++)
        {
          if (PABS(vecTX[ss+repID] - fixedInput) < 1.0E-10 &&
              PABS(vecTW[ss+repID] - fixedInput2) < 1.0E-10)
          {
            if (vecTY[ss+repID] < 0.9*PSUADE_UNDEFINED)
            {
              vecVCEVars[symIndex] += 
                   ((vecTY[ss+repID]-vecVCEMeans[symIndex])*
                    (vecTY[ss+repID]-vecVCEMeans[symIndex]));
            }
          }
        }
        if (validnReps > 0)
             vecVCEVars[symIndex] /= ((double) validnReps);
        else vecVCEVars[symIndex] = PSUADE_UNDEFINED;
      }
      nSubs = nSamples / nReps;

      totalCnt = 0;
      for (subID = 0; subID < nSubs; subID++) totalCnt += vecBins[subID];

      vecVarMeans[ii*nInputs+ii2] = 0.0;
      vecMeanVars[ii*nInputs+ii2] = 0.0;
      aMean = 0.0;
      for (subID = 0; subID < nSubs; subID++)
      {
        if (vecVCEVars[subID] != PSUADE_UNDEFINED)
        {
          aMean += (vecVCEMeans[subID] / totalCnt * vecBins[subID]);
        }
      }
      for (subID = 0; subID < nSubs; subID++)
      {
        if (vecVCEVars[subID] != PSUADE_UNDEFINED)
        {
          vecVarMeans[ii*nInputs+ii2] += ((vecVCEMeans[subID] - aMean)*
                                         (vecVCEMeans[subID] - aMean) *
                                          vecBins[subID] / totalCnt);
          vecMeanVars[ii*nInputs+ii2] += vecVCEVars[subID] *
                                          vecBins[subID] / totalCnt;
        }
      }
      vecVCE[ii*nInputs+ii2] = vecVarMeans[ii*nInputs+ii2];

      vecVarVars[ii*nInputs+ii2] = 0.0; 
      for (subID = 0; subID < nSubs; subID++)
        if (vecVCEVars[subID] != PSUADE_UNDEFINED)
          vecVarVars[ii*nInputs+ii2] += 
                ((vecVCEVars[subID]-vecMeanVars[ii*nInputs+ii2])*
                 (vecVCEVars[subID]-vecMeanVars[ii*nInputs+ii2]) * 
                vecBins[subID] / totalCnt);
        if (printLevel_ > 4 || nSamples > 100000) 
          printOutTS(PL_DUMP, "Interaction %3d %d = %10.3e\n",ii+1,
                     ii2+1, vecVCE[ii*nInputs+ii2]/aVariance);
    }
    if (errflag != 0) break;
  }

  double *dptr = vecSTIs.getDVector();
  for (ii = 0; ii < nInputs; ii++)
  {
    vecMEs[ii] = analyze1D(nInputs,nOutputs,nSamples,localX,YIn,
                       ii,whichOutput,nReps1,aMean,aVariance,&(dptr[ii]));
    printOutTS(PL_INFO,"Main effect (Inputs %3d) = %10.3e\n", ii+1,
               vecMEs[ii]/aVariance);
    vecVCE[nInputs*nInputs+ii] = vecMEs[ii];
  }

  //**/ ---------------------------------------------------------------
  //**/ return more detailed data
  //**/ ---------------------------------------------------------------
  pData *pPtr = ioPtr->getAuxData();
  pPtr->nDbles_ = nInputs * nInputs;
  pPtr->dbleArray_ = new double[nInputs * (nInputs+1)];
  for (ii = 0; ii < nInputs*(nInputs+1); ii++) 
    pPtr->dbleArray_[ii] = vecVCE[ii];
  pPtr->dbleData_ = aVariance;
  //genPlots(adata);

  VecMainEffects_.setLength(nInputs_);
  if (errflag == 0)
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,
               "* First-order Sensitivity Indices\n");
    printDashes(PL_INFO, 0);
    for (ii = 0; ii < nInputs; ii++)
    {
      if (vecMEs[ii] < 1.0e-10 * aVariance)
        vecMEs[ii] = 0.0;
      if (vecSTIs[ii] < 0.0) vecSTIs[ii] = 0.0;
      printOutTS(PL_INFO, 
          "Input %3d: Sobol' index = %10.3e (Unnormalized = %10.3e)\n",
          ii+1,vecMEs[ii]/aVariance,vecMEs[ii]);

      //save sensitivity
      VecMainEffects_[ii] = vecMEs[ii];
    }
    if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
    {
      printEquals(PL_INFO, 0);
      printOutTS(PL_INFO, 
           "* Two-parameter sensitivity indices\n");
      printDashes(PL_INFO, 0);
    }
    for (ii = 0; ii < nInputs; ii++)
    {
      for (ii2 = ii+1; ii2 < nInputs; ii2++)
      {
        if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
          printOutTS(PL_INFO, 
           "* Sobol' index (Inputs %2d %2d) = %10.3e (raw = %10.3e)\n",
           ii+1,ii2+1,vecVCE[ii*nInputs+ii2]/aVariance,
           vecVCE[ii*nInputs+ii2]);
        Mat2ParamEffects_.setEntry(ii,ii2,vecVCE[ii*nInputs+ii2]);
      }
    }
    if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
      printAsterisks(PL_INFO, 0);
  }

#if 0
 //**/ bootstrap to be generated external to this file
  printAsterisks(PL_INFO, 0);
  if (psConfig_.AnaExpertModeIsOn())
  {
    printOutTS(PL_INFO, 
     "Bootstrap analysis draws n samples from the orignal sample and\n");
    printOutTS(PL_INFO, 
     "assess whether the sensitivity indices have converged.  If you\n");
    printOutTS(PL_INFO, 
     "are performing iterative analysis with refinements, you will need\n");
    printOutTS(PL_INFO, 
     "to enter 'no index reuse' below at the first iteration and 'yes'\n");
    printOutTS(PL_INFO, 
     "afterward until the final refinement.\n");
    snprintf(pString,100,"Perform bootstrap interaction analysis? (y or n) ");
    getString(pString, winput);

    //**/ will probably not be needed
    int numBS=0;
    if (winput[0] == 'y')
    {
      snprintf(pString,100,"Enter number of bootstrap samples (>=100): ");
      numBS = getInt(100, 2000, pString);
      psVector vecXX;
      vecXX.setLength(nInputs*nSamples);
      vecYY.setLength(nSamples);
      matBSVCEs.setFormat(PS_MAT2D);
      matBSVCEs.setDim(numBS, nInputs*nInputs);
      nReps = nSamples / nSubSamples;

      FILE *fp1 = fopen(".IE_bootstrap_indset", "r");
      if (fp1 != NULL)
      {
        printOutTS(PL_INFO, ".IE_bootstrap_indset file found.\n");
        snprintf(pString,100,"Re-use file? (y or n) ");
        getString(pString, winput);
        if (winput[0] == 'y')
        {
          fscanf(fp1, "%d", &ii);
          if (ii != nReps*numBS)
          {
            printOutTS(PL_ERROR, 
                 "ERROR: expect the first line to be %d.\n",
                 nReps*numBS);
            printOutTS(PL_ERROR, 
                 "       Instead found the first line to be %d\n",ii);
            fclose(fp1);
            exit(1);
          }
        }
        else
        {
          fclose(fp1);
          fp1 = fopen(".IE_bootstrap_indset", "w");
          if (fp1 == NULL)
          {
            printOutTS(PL_ERROR, 
               "ERROR: cannot open bootstrap_indset file (write).\n");
            exit(1);
          }
          fprintf(fp1, "%d\n", nReps*numBS);
        }
      }
      else
      {
        fp1 = fopen(".IE_bootstrap_indset", "w");
        if (fp1 == NULL)
        {
          printOutTS(PL_ERROR, 
               "ERROR: cannot open bootstrap_indset file (write).\n");
          exit(1);
        }
        fprintf(fp1, "%d\n", nReps*numBS);
        winput[0] = 'n';
      }
      fp = NULL;

      for (ii = 0; ii < numBS; ii++)
      {
        if (ii % (numBS / 10) == 0)
          printOutTS(PL_INFO, "Processing %d(a) of %d\n", ii+1, numBS);
        for (ir = 0; ir < nReps; ir++)
        {
          if (fp1 != NULL && winput[0] == 'y')
               fscanf(fp1, "%d\n", &index);
          else index = PSUADE_rand() % nReps;
          if (fp1 != NULL && winput[0] != 'y')
            fprintf(fp1, "%d\n", index);

          for (ss = 0; ss < nSubSamples; ss++)
          {
            for (ii2 = 0; ii2 < nInputs; ii2++)
              vecXX[(ir*nSubSamples+ss)*nInputs+ii2] =
                    XIn[(index*nSubSamples+ss)*nInputs+ii2];
            vecYY[ir*nSubSamples+ss] = 
              YIn[(index*nSubSamples+ss)*nOutputs+whichOutput];
          }
        }

        if (ii % (numBS / 10) == 0)
          printOutTS(PL_INFO, "Processing %d(b) of %d\n", ii+1, numBS);
        computeMeanVariance(nInputs,iOne,nSamples,vecYY.getDVector(),
                            &aMean,&aVariance, iZero);
        if (PABS(aVariance) < 1.0e-15)
        {
          printOutTS(PL_INFO,"INFO: variance too small ==> terminate.\n");
          continue;
        }
        computeVCE2(nInputs, nSamples, nSubSamples, vecXX.getDVector(), 
                    vecYY.getDVector(), fp, aMean, 
                    vecVarMeans.getDVector(), vecMeanVars.getDVector(), 
                    vecVarVars.getDVector(), vecVCE.getDVector());
        double *dptr2 = vecSTIs.getDVector();
        for (ii2 = 0; ii2 < nInputs; ii2++)
          vecMEs[ii2] = analyze1D(nInputs,iOne,nSamples,vecXX.getDVector(),
                           vecYY.getDVector(),ii2,iZero,nReps1,aMean,
                           aVariance,&(dptr2[ii2]));
        for (ii2 = 0; ii2 < nInputs; ii2++)
          for (ss = ii2+1; ss < nInputs; ss++)
            matBSVCEs.setEntry(ii,ii2*nInputs+ss,
                          (vecVCE[ii2*nInputs+ss])/aVariance); 
                  //**/- vecMEs[ii2] - vecMEs[ss]) / aVariance;
      }
      if (fp1 != NULL) fclose(fp1);

      printf("Enter name of matlab/scilab file to store bootstrap info: ");
      scanf("%500s", winput);
      fgets(pString,500,stdin);
      fp = fopen(winput, "w");
      if (fp != NULL)
      {
        if (plotScilab())
        {
          fprintf(fp, "//bootstrap sample of interaction effects\n");
          fprintf(fp, "//VCE((i-1)*n+j,k) = VCE(i,j), bs sample k\n");
        }
        else
        {
          fprintf(fp, "%%bootstrap sample of interaction effects\n");
          fprintf(fp, "%%VCE((i-1)*n+j,k) = VCE(i,j), bs sample k\n");
        }
        fprintf(fp, "VCE = zeros(%d,%d);\n",nInputs*nInputs,numBS);
        if (ioPtr != NULL) ioPtr->getParameter("input_names",pdata);
        char **inputNames = pdata.strArray_;
        if (inputNames == NULL)
        {
          if (plotScilab()) fprintf(fp, "Str = [");
          else              fprintf(fp, "Str = {");
          for (ii = 0; ii < nInputs-1; ii++) fprintf(fp,"'X%d',",ii+1);
          if (plotScilab()) fprintf(fp,"'X%d'];\n",nInputs);
          else              fprintf(fp,"'X%d'};\n",nInputs);
        }
        else
        {
          if (plotScilab()) fprintf(fp, "Str = [");
          else              fprintf(fp, "Str = {");
          for (ii = 0; ii < nInputs-1; ii++)
          {
            if (inputNames[ii] != NULL)
                 fprintf(fp,"'%s',",inputNames[ii]);
            else fprintf(fp,"'X%d',",ii+1);
          }
          if (plotScilab()) 
          {
            if (inputNames[nInputs-1] != NULL)
                 fprintf(fp,"'%s'];\n",inputNames[nInputs-1]);
            else fprintf(fp,"'X%d'];\n",nInputs);
          }
          else 
          {
            if (inputNames[nInputs-1] != NULL)
                 fprintf(fp,"'%s'};\n",inputNames[nInputs-1]);
            else fprintf(fp,"'X%d'};\n",nInputs);
          }
        }
        for (ss = 0; ss < numBS; ss++)
        {
          for (ii = 0; ii < nInputs; ii++)
          {
            for (ii2 = ii+1; ii2 < nInputs; ii2++)
            {
              if (ss == 0)
              {
                if (plotScilab())
                     fprintf(fp,"//Interaction(%d,%d) \n",ii+1,ii2+1);
                else fprintf(fp,"%%Interaction(%d,%d) \n",ii+1,ii2+1);
              }
              fprintf(fp, "VCE(%d,%d) = %e;\n",ii*nInputs+ii2+1,ss+1,
                      matBSVCEs.getEntry(ss,ii*nInputs+ii2));
            }
          }
        }
        fprintf(fp,"VMM = sum(VCE')/%d;\n", numBS);
        fprintf(fp,"VMA = max(VCE');\n");
        fprintf(fp,"VMI = min(VCE');\n");
        fprintf(fp,"ymin = 0;\n");
        fprintf(fp,"ymax = max(VMA);\n");
        fprintf(fp,"hh = 0.05 * (ymax - ymin);\n");
        fprintf(fp,"ymax = ymax + hh;\n");
        fprintf(fp,"nn   = %d;\n",nInputs);
        if (plotScilab())
        {
          fprintf(fp,"VMM = matrix(VMM, nn, nn);\n");
          fprintf(fp,"VMM = VMM';\n");
          fprintf(fp,"VMA = matrix(VMA, nn, nn);\n");
          fprintf(fp,"VMA = VMA';\n");
          fprintf(fp,"VMI = matrix(VMI, nn, nn);\n");
          fprintf(fp,"VMI = VMI';\n");
          fprintf(fp,"drawlater\n");
          fprintf(fp,"hist3d(VMM);\n");
          fwriteHold(fp, 1);
          fprintf(fp,"a=gca();\n");
          fprintf(fp,"a.data_bounds=[0, 0, 0; nn, nn+1, ymax];\n");
          fprintf(fp,"newtick = a.x_ticks;\n");
          fprintf(fp,"newtick(2) = [1:nn]';\n");
          fprintf(fp,"newtick(3) = Str';\n");
          fprintf(fp,"a.x_ticks = newtick;\n");
          fprintf(fp,"a.x_label.font_size = 3;\n");
          fprintf(fp,"a.x_label.font_style = 4;\n");
          fprintf(fp,"a.y_ticks = newtick;\n");
          fprintf(fp,"a.y_label.font_size = 3;\n");
          fprintf(fp,"a.y_label.font_style = 4;\n");
          fprintf(fp,"a.rotation_angles = [5 -70];\n");
          fprintf(fp,"drawnow\n");
        }
        else
        {
          fprintf(fp,"VMM = reshape(VMM, nn, nn);\n");
          fprintf(fp,"VMM = VMM';\n");
          fprintf(fp,"VMA = reshape(VMA, nn, nn);\n");
          fprintf(fp,"VMA = VMA';\n");
          fprintf(fp,"VMI = reshape(VMI, nn, nn);\n");
          fprintf(fp,"VMI = VMI';\n");
          fprintf(fp,"hh = bar3(VMM,0.8);\n");
          fprintf(fp,"alpha = 0.2;\n");
          fprintf(fp,"set(hh,'FaceColor','b','facea',alpha);\n");
          fprintf(fp,"[X,Y] = meshgrid(1:nn,1:nn);\n");
          fprintf(fp,"for k = 1:nn\n");
          fprintf(fp,"  for l = k:nn\n");
          fprintf(fp,"    mkl = VMM(k,l);\n");
          fprintf(fp,"    ukl = VMA(k,l);\n");
          fprintf(fp,"    lkl = VMI(k,l);\n");
          fprintf(fp,"    if (mkl > .02 & (ukl-lkl)/mkl > .02)\n");
          fprintf(fp,"      xkl = [X(k,l), X(k,l)];\n");
          fprintf(fp,"      ykl = [Y(k,l), Y(k,l)];\n");
          fprintf(fp,"      zkl = [lkl, ukl];\n");
          fprintf(fp,"      plot3(xkl,ykl,zkl,'-mo',...\n");
          fprintf(fp,"        'LineWidth',5,'MarkerEdgeColor','k',...\n");
          fprintf(fp,"        'MarkerFaceColor','k','MarkerSize',10);\n");
          fprintf(fp,"    end\n");
          fprintf(fp,"  end\n");
          fprintf(fp,"end\n");
          fwriteHold(fp, 0);
          fprintf(fp,"axis([0.5 nn+0.5 0.5 nn+0.5 0 ymax])\n");
          fprintf(fp,"set(gca,'XTickLabel',Str);\n");
          fprintf(fp,"set(gca,'YTickLabel',Str);\n");
        }
        fwritePlotAxes(fp);
        fwritePlotTitle(fp,"Sobol 1st+2nd Order Indices (with bootstrap)");
        fwritePlotZLabel(fp, "Sobol Indices (Normalized)");
        fwritePlotXLabel(fp, "Inputs");
        fwritePlotYLabel(fp, "Inputs");
        printOutTS(PL_INFO, "Total variance = %e\n", aVariance);
        fclose(fp);
        printOutTS(PL_INFO, 
           "Bootstrapped interaction effect plot in now in %s\n",winput);
      }
      else
      {
        printOutTS(PL_ERROR, "ERROR: cannot open file %s\n", winput);
      }
    }
  }
#endif
  if (constrPtr != NULL) delete constrPtr;
  // return 1.0 to facilitate continuous refinement
  return 1.0;
}

// *************************************************************************
// Compute agglomerated mean and variance
// -------------------------------------------------------------------------
int TwoParamAnalyzer::computeMeanVariance(int nInputs, int nOutputs, 
                                     int nSamples, double *y, double *aMean, 
                                     double *aVariance, int outputID)
{
  int    ss, count;
  double mean, variance;

  mean = 0.0;
  count = 0;
  for (ss = 0; ss < nSamples; ss++)
  {
    if (y[nOutputs*ss+outputID] < 0.9*PSUADE_UNDEFINED)
    {
      mean += y[nOutputs*ss+outputID];
      count++;
    }
  }
  if (count <= 0)
  {
    printOutTS(PL_ERROR, "TwoParamEffect ERROR: no valid data.\n");
    exit(1);
  }
  mean /= (double) count;
  variance = 0.0;
  for (ss = 0; ss < nSamples; ss++) 
  {
    if (y[nOutputs*ss+outputID] < 0.9*PSUADE_UNDEFINED)
    {
      variance += ((y[nOutputs*ss+outputID] - mean) *
                   (y[nOutputs*ss+outputID] - mean));
    }
  }
  variance /= (double) count;
  (*aMean) = mean;
  (*aVariance) = variance;
  return 0;
}

// ************************************************************************
// perform VCE (McKay) analysis
// ------------------------------------------------------------------------
double TwoParamAnalyzer::analyze1D(int nInputs, int nOutputs, int nSamples,
                      double *xIn, double *yIn, int inputID, int outputID,
                      int nReps, double aMean, double aVariance,
                      double *retData)
{
  int    ss, repID, subID, ncount, symIndex, nSubs;
  int    whichOutput, validnReps, totalCnt;
  double meanVCEVar, vce, varVCEMean, fixedInput, tmean;
  psIVector vecBins;
  psVector  vecTX, vecTY, vecVceMean, vecVceVar;

  whichOutput = outputID;
  if (whichOutput >= nOutputs || whichOutput < 0) whichOutput = 0;

  if (nReps <= 1) return 1.0;
  nSubs         = nSamples / nReps;
  vecVceMean.setLength(nSubs);
  vecVceVar.setLength(nSubs);
  vecTX.setLength(nSamples);
  vecTY.setLength(nSamples);
  vecBins.setLength(nSubs);

  for (ss = 0; ss < nSamples; ss++)
  {
    vecTX[ss] = xIn[nInputs*ss+inputID];
    vecTY[ss] = yIn[nOutputs*ss+whichOutput];
  }
  sortDbleList2(nSamples, vecTX.getDVector(), vecTY.getDVector());
  ncount = 0;
  for (ss = 0; ss < nSamples; ss+=nReps)
  {
    symIndex = ss / nReps;
    fixedInput = vecTX[ss];
    vecVceMean[symIndex] = 0.0;
    validnReps = 0;
    for (repID = 0; repID < nReps; repID++)
    {
      if (PABS(vecTX[ss+repID] - fixedInput) < 1.0E-10)
      {
        ncount++;
        if (vecTY[ss+repID] < 0.9*PSUADE_UNDEFINED)
        {
          vecVceMean[symIndex] += vecTY[ss+repID];
          validnReps++;
        }
      }
    }
    if (validnReps > 0) vecVceMean[symIndex] /= (double) validnReps;
    vecVceVar[symIndex] = 0.0;
    tmean = vecVceMean[symIndex];
    for (repID = 0; repID < nReps; repID++)
    {
      if (PABS(vecTX[ss+repID] - fixedInput) < 1.0E-10)
      {
        if (vecTY[ss+repID] < 0.9*PSUADE_UNDEFINED)
          vecVceVar[symIndex] += ((vecTY[ss+repID]-tmean)*
                                  (vecTY[ss+repID]-tmean));
      }
    }
    if (validnReps > 0) vecVceVar[symIndex] /= validnReps;
    else                vecVceVar[symIndex] = PSUADE_UNDEFINED;
    vecBins[symIndex] = validnReps;
  }
  if (ncount != nSamples)
  {
    printOutTS(PL_ERROR,
      "TwoParamEffect ERROR: sample not valid, input %d\n",inputID);
    printOutTS(PL_ERROR, 
      "       error data = %d (%d)\n",ncount, nSamples);
    exit(1);
  }

  varVCEMean = 0.0;
  meanVCEVar = 0.0;
  totalCnt = 0;
  for (subID = 0; subID < nSubs; subID++) totalCnt += vecBins[subID];
  aMean = 0.0;
  for (subID = 0; subID < nSubs; subID++)
  {
    if (vecVceVar[subID] != PSUADE_UNDEFINED)
      aMean += (vecVceMean[subID] / totalCnt * vecBins[subID]);
  }
  for (subID = 0; subID < nSubs; subID++)
  {
    if (vecVceVar[subID] != PSUADE_UNDEFINED)
    {
      varVCEMean += ((vecVceMean[subID]-aMean)*(vecVceMean[subID]-aMean) *
                    vecBins[subID] / totalCnt);
      meanVCEVar += vecVceVar[subID] * vecBins[subID] / totalCnt;
    }
  }
  vce = varVCEMean;

  (*retData) = 1.0 - meanVCEVar / aVariance;
  return vce;
}

// *************************************************************************
// Compute two-way VCE (called when bootstrapped is used)
// -------------------------------------------------------------------------
int TwoParamAnalyzer::computeVCE2(int nInputs,int nSamples, int nSubSamples,
                                 double *X, double *Y, FILE *fp, 
                                 double aMean, double *varVCEMean,
                                 double *meanVCEVar, double *varVCEVar,
                                 double *vce)
{
  int    ii, ii2, ss, nReps1, nReps, nSubs, symIndex, repID, subID;
  int    totalCnt, validnReps;
  double *tws, *tys, fixedInput, fixedInput2;
  psIVector vecBins;
  psVector  vecTX, vecTW, vecTY, vecVceMean, vecVceVar;

  vecTX.setLength(nSamples);
  vecTW.setLength(nSamples);
  vecTY.setLength(nSamples);
  vecVceMean.setLength(nSubSamples);
  vecVceVar.setLength(nSubSamples);
  vecBins.setLength(nSubSamples);
  tws = vecTW.getDVector();
  tys = vecTY.getDVector();

  for (ii = 0; ii < nInputs; ii++)
  {
    for (ii2 = ii+1; ii2 < nInputs; ii2++)
    {
      for (ss = 0; ss < nSamples; ss++)
      {
        vecTX[ss] = X[nInputs*ss+ii];
        vecTW[ss] = X[nInputs*ss+ii2];
        vecTY[ss] = Y[ss];
      }
      sortDbleList3(nSamples, vecTX.getDVector(), vecTW.getDVector(), 
                    vecTY.getDVector());

      for (ss = 1; ss < nSamples; ss++)
        if (PABS(vecTX[ss]-vecTX[ss-1]) > 1.0E-10)
          break;
      nReps1 = ss;

      for (ss = 0; ss < nSamples; ss+=nReps1)
        sortDbleList2(nReps1,&tws[ss],&tys[ss]);

      for (ss = 1; ss < nReps1; ss++)
        if (PABS(vecTW[ss]-vecTW[ss-1]) > 1.0E-10)
          break;
      nReps = ss;

      for (ss = 0; ss < nSamples; ss+=nReps)
      {
        symIndex = ss / nReps;
        fixedInput  = vecTX[ss];
        fixedInput2 = vecTW[ss];
        vecVceMean[symIndex] = 0.0;
        validnReps = 0;
        for (repID = 0; repID < nReps; repID++)
        {
          if (PABS(vecTX[ss+repID] - fixedInput) < 1.0E-10 &&
              PABS(vecTW[ss+repID] - fixedInput2) < 1.0E-10)
          {
            if (vecTY[ss+repID] < 0.9*PSUADE_UNDEFINED)
            {
              vecVceMean[symIndex] += vecTY[ss+repID];
              validnReps++;
            }
          }
        }
        if (validnReps > 0) vecVceMean[symIndex] /= (double) validnReps;
        vecVceVar[symIndex] = 0.0;
        for (repID = 0; repID < nReps; repID++)
        {
          if (vecTY[ss+repID] < 0.9*PSUADE_UNDEFINED)
            vecVceVar[symIndex] += 
                       ((vecTY[ss+repID]-vecVceMean[symIndex])*
                        (vecTY[ss+repID]-vecVceMean[symIndex]));
        }
        if (validnReps > 0) 
             vecVceVar[symIndex] /= ((double) validnReps);
        else vecVceVar[symIndex] = PSUADE_UNDEFINED;
      }
      nSubs = nSamples / nReps;

      varVCEMean[ii*nInputs+ii2] = 0.0;
      meanVCEVar[ii*nInputs+ii2] = 0.0;
      totalCnt = 0;
      for (subID = 0; subID < nSubs; subID++) totalCnt += vecBins[subID];
      for (subID = 0; subID < nSubs; subID++)
      {
        if (vecVceVar[subID] != PSUADE_UNDEFINED)
        {
          varVCEMean[ii*nInputs+ii2] += ((vecVceMean[subID] - aMean)*
                                         (vecVceMean[subID] - aMean) *
                                          vecBins[subID] / totalCnt);
          meanVCEVar[ii*nInputs+ii2] += vecVceVar[subID]*vecBins[subID]/
                                        totalCnt;
        }
      }
      vce[ii*nInputs+ii2] = varVCEMean[ii*nInputs+ii2];

      varVCEVar[ii*nInputs+ii2] = 0.0; 
      for (subID = 0; subID < nSubs; subID++)
        if (vecVceVar[subID] != PSUADE_UNDEFINED)
          varVCEVar[ii*nInputs+ii2] += 
             ((vecVceVar[subID]-meanVCEVar[ii*nInputs+ii2])*
              (vecVceVar[subID]-meanVCEVar[ii*nInputs+ii2]) *
              vecBins[subID] / totalCnt);
    }
  }
  return 0;
}

// *************************************************************************
// Compute VCE 
// -------------------------------------------------------------------------
int TwoParamAnalyzer::computeVCECrude(int nInputs, int nSamples, 
                             double *X, double *Y,double aVariance,
                             double *iLowerB, double *iUpperB, double *vce)
{
  int    ii, ii2, ss, nSize, nIntervals, ind1, ind2, ind;
  int    totalCnt, nIntervals1, nSize1, nFilled;
  double ddata, hstep1, hstep2, aMean;
  char   pString[500];
  psIVector vecBins, vecTags;
  psVector  vecVce1, vecVceMean, vecVceVar;

  nSize = 100;
  ddata = 1.0 * nSamples / nSize;
  ddata = pow(ddata, 0.5);
  nIntervals = (int) ddata;
  nIntervals1 = (int) sqrt(1.0 * nSamples);
  //**/ changed to be consistent with main effect
  //**/nIntervals1 = nIntervals * nIntervals;
  if (nIntervals < 4)
  {
    printOutTS(PL_ERROR,"ERROR: Sample size too small.\n");
    printOutTS(PL_ERROR,
       "       Need larger sample to give meaningful results.\n");
    return -1;
  }
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
     "        Crude (2D-binning) Interaction Effect\n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,
     "TwoParamEffect: default number of levels in each input  = %d\n",
     nIntervals);
    printOutTS(PL_INFO,
     "TwoParamEffect: default number of level for main effect = %d\n",
     nIntervals1);
    printOutTS(PL_INFO,
     "TwoParamEffect: sample size for each 2-dimensional box  = %d\n",
     nSize);
    printDashes(PL_INFO, 0);
    printOutTS(PL_INFO,
     "* For small to moderate sample sizes, this method ");
    printOutTS(PL_INFO,
     "computes rough\n");
    printOutTS(PL_INFO,
     "* estimates of interaction effect (two-parameter ");
    printOutTS(PL_INFO,
     "sensitivity). These\n");
    printOutTS(PL_INFO,
     "* estimates can vary with different choices of ");
    printOutTS(PL_INFO,
     "internal settings. For\n");
    printOutTS(PL_INFO,
     "* example, try different number of levels to ");
    printOutTS(PL_INFO,
     "assess sensitivity of the\n");
    printOutTS(PL_INFO,
     "* computed measures with respect to it.\n");
    printOutTS(PL_INFO,
     "* Turn on analysis expert mode to change internal settings.\n");
    if (psConfig_.AnaExpertModeIsOn())
    {
      printEquals(PL_INFO, 0);
      snprintf(pString,100,
        "Number of levels for 2-way analysis (>= 4, default = %d): ",
        nIntervals);
      nIntervals = getInt(4, nSamples, pString);
      nSize = nSamples / nIntervals / nIntervals;
      if (nSize < 10)
      {
        printOutTS(PL_INFO,
         "* Using number of levels = %d for 2-way effect will give\n",
         nIntervals);
        printOutTS(PL_INFO,
         "  each 2D box less than 10 sample points. That is too small.\n");
        nSize = 10;
        ddata = 1.0 * nSamples / nSize;
        ddata = pow(ddata, 0.5);
        nIntervals = (int) ddata;
        printOutTS(PL_INFO, 
         "* Number of levels for 2-way analysis defaulted to %d\n", 
         nIntervals);
      } 
      snprintf(pString,100,
            "Number of levels for main effect (>= %d): ",nIntervals);
      nIntervals1 = getInt(nIntervals, nSamples, pString);
    } 
    nSize1 = nSamples / nIntervals1;
    if (nSize1 < 10)
    {
      printOutTS(PL_INFO, 
       "Sample size per level for main effect d too small (%d < 10).\n",
       nSize1);
      nSize1 = 10;
      nIntervals1 = nSamples / nSize1;
      printOutTS(PL_INFO, 
       "Default number of levels for main effect to %d\n",nIntervals1);
    } 
  }
  int nsize = nIntervals * nIntervals;
  if (nIntervals1 > nsize) nsize = nIntervals1; 
  vecVceMean.setLength(nsize);
  vecVceVar.setLength(nsize);
  vecBins.setLength(nsize);
  vecTags.setLength(nSamples);
  vecVce1.setLength(nInputs);

  if (psConfig_.InteractiveIsOn() && printLevel_ > 1)
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"* Main Effect Sensitivity Indices\n");
    printDashes(PL_INFO, 0);
  }

  VecMainEffects_.setLength(nInputs_);
  for (ii = 0; ii < nInputs; ii++)
  {
    hstep1 = (iUpperB[ii] - iLowerB[ii]) / nIntervals1;
    for (ss = 0; ss < nIntervals1; ss++)
    {
      vecVceMean[ss] = 0.0;
      vecVceVar[ss] = 0.0;
      vecBins[ss] = 0;
    }
    for (ss = 0; ss < nSamples; ss++)
    {
      ddata = (X[nInputs*ss+ii] - iLowerB[ii]) / hstep1;
      ind1 = (int) ddata;
      if (ind1 <  0) ind1 = 0;
      if (ind1 >= nIntervals1) ind1 = nIntervals1 - 1;
      vecTags[ss] = -1;
      if (Y[ss] < 0.9*PSUADE_UNDEFINED)
      {
        vecVceMean[ind1] += Y[ss];
        vecBins[ind1]++;
        vecTags[ss] = ind1;
      }
    }
    for (ss = 0; ss < nIntervals1; ss++)
      if (vecBins[ss] > 0) vecVceMean[ss] /= (double) vecBins[ss];

    for (ss = 0; ss < nSamples; ss++)
    {
      ind = vecTags[ss];
      if (ind >= 0 && Y[ss] < 0.9*PSUADE_UNDEFINED)
      {
        vecVceVar[ind] += ((Y[ss]-vecVceMean[ind])*
                           (Y[ss]-vecVceMean[ind]));
      }
    }
    for (ss = 0; ss < nIntervals1; ss++)
    {
      if (vecBins[ss] > 0)
           vecVceVar[ss] /= (double) vecBins[ss];
      else vecVceVar[ss] = PSUADE_UNDEFINED;
    }

    totalCnt = 0;
    for (ss = 0; ss < nIntervals1; ss++) totalCnt += vecBins[ss];
    aMean = 0.0;
    for (ss = 0; ss < nIntervals1; ss++)
    {
      if (vecVceVar[ss] != PSUADE_UNDEFINED)
        aMean += (vecVceMean[ss] / totalCnt * vecBins[ss]);
    }
    vecVce1[ii] = 0.0;
    for (ss = 0; ss < nIntervals1; ss++)
    {
      if (vecVceVar[ss] != PSUADE_UNDEFINED)
      {
        vecVce1[ii] += ((vecVceMean[ss] - aMean) *
                        (vecVceMean[ss] - aMean) * vecBins[ss]/totalCnt);
      }
    }
    if (psConfig_.InteractiveIsOn() && printLevel_ > 1)
      printOutTS(PL_INFO,
         "Input %3d: Sobol' index = %10.3e (unnormalized = %10.3e)\n",
         ii+1,vecVce1[ii]/aVariance,vecVce1[ii]);

    // save main effect
    VecMainEffects_[ii] = vecVce1[ii];
  }
  if (psConfig_.InteractiveIsOn()) 
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO, 
       "* Two-parameter sensitivity indices (normalized)\n");
    printDashes(PL_INFO, 0);
  }
  printf("TwoParamEffect: nBins2D, nReps = %d %d\n",
         nIntervals*nIntervals,nSamples/(nIntervals*nIntervals));
  for (ii = 0; ii < nInputs; ii++)
  {
    for (ii2 = 0; ii2 < ii+1; ii2++) vce[ii*nInputs+ii2] = 0;
    for (ii2 = ii+1; ii2 < nInputs; ii2++)
    {
      hstep1 = (iUpperB[ii] - iLowerB[ii]) / nIntervals;
      hstep2 = (iUpperB[ii2] - iLowerB[ii2]) / nIntervals;
      for (ss = 0; ss < nIntervals*nIntervals; ss++)
      {
        vecVceMean[ss] = 0.0;
        vecVceVar[ss] = 0.0;
        vecBins[ss] = 0;
      }
      for (ss = 0; ss < nSamples; ss++)
      {
        ddata = (X[nInputs*ss+ii] - iLowerB[ii]) / hstep1;
        ind1 = (int) ddata;
        if (ind1 < 0) ind1 = 0;
        if (ind1 >= nIntervals) ind1 = nIntervals - 1;
        ddata = (X[nInputs*ss+ii2] - iLowerB[ii2]) / hstep2;
        ind2 = (int) ddata;
        if (ind2 < 0) ind2 = 0;
        if (ind2 >= nIntervals) ind2 = nIntervals - 1;
        ind = ind1 * nIntervals + ind2;
        vecTags[ss] = -1;
        if (Y[ss] < 0.9*PSUADE_UNDEFINED)
        {
          vecVceMean[ind] += Y[ss];
          vecBins[ind]++;
          vecTags[ss] = ind;
        }
      }

      nFilled = 0;
      for (ss = 0; ss < nIntervals*nIntervals; ss++)
      {
        if (vecBins[ss] > 0) 
        {
          vecVceMean[ss] /= (double) vecBins[ss];
          nFilled++;
        }
      }

      for (ss = 0; ss < nSamples; ss++)
      {
        ind = vecTags[ss];
        if (ind >= 0 && Y[ss] < 0.9*PSUADE_UNDEFINED)
        {
          vecVceVar[ind] += ((Y[ss]-vecVceMean[ind])*
                             (Y[ss]-vecVceMean[ind]));
        }
      }
      for (ss = 0; ss < nIntervals*nIntervals; ss++)
      {
        if (vecBins[ss] > 0)
             vecVceVar[ss] /= (double) vecBins[ss];
        else vecVceVar[ss] = PSUADE_UNDEFINED;
      }

      totalCnt = 0;
      for (ss = 0; ss < nIntervals*nIntervals; ss++)
        totalCnt += vecBins[ss];
      aMean = 0.0;
      for (ss = 0; ss < nIntervals*nIntervals; ss++)
      {
        if (vecVceVar[ss] != PSUADE_UNDEFINED)
          aMean += (vecVceMean[ss] / totalCnt * vecBins[ss]);
      }
      vce[ii*nInputs+ii2] = 0.0;
      for (ss = 0; ss < nIntervals*nIntervals; ss++)
      {
        if (vecVceVar[ss] != PSUADE_UNDEFINED)
        {
          vce[ii*nInputs+ii2] += ((vecVceMean[ss] - aMean) *
             (vecVceMean[ss] - aMean) * vecBins[ss]/totalCnt);
        }
      }
      //**/ Should not subtract: pure 2-way effect != 2-way effect - ME 
      //**/vce[ii*nInputs+ii2] -= vecVce1[ii];
      //**/vce[ii*nInputs+ii2] -= vecVce1[ii2];
      if ( vce[ii*nInputs+ii2] < 0.0) vce[ii*nInputs+ii2] = 0.0;
      //printOutTS(PL_INFO, 
      //   "* Sensitivity index ( Inputs %2d %2d ) = %10.2e (raw = %10.2e)\n",
      //   ii+1,ii2+1,vce[ii*nInputs+ii2]/aVariance,vce[ii*nInputs+ii2]);
      //**/fprintf(fp, "V2(%d,%d) = %e;\n",ii+1,ii2+1,
      //           vce[ii*nInputs+ii2]/aVariance);
      Mat2ParamEffects_.setEntry(ii,ii2,vce[ii*nInputs+ii2]);
    }
    vce[nInputs*nInputs+ii] = vecVce1[ii];
  }
  if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn()) 
  {
    printAsterisks(PL_INFO, 0);
    for (ii = 0; ii < nInputs; ii++)
    {
      for (ii2 = ii+1; ii2 < nInputs; ii2++)
        printOutTS(PL_INFO, 
         "Inputs %2d %2d: Sobol' index = %10.3e (unnormalized = %9.2e)\n",
         ii+1,ii2+1,vce[ii*nInputs+ii2]/aVariance,vce[ii*nInputs+ii2]);
    }
    printAsterisks(PL_INFO, 0);
  }
  return 0;
}

// ************************************************************************
// generate Matlab or Scilab plots
// ------------------------------------------------------------------------
int TwoParamAnalyzer::genPlots(aData &adata)
{
  int  ii, jj;
  char pString[500];
  FILE *fp=NULL;
  PsuadeData *ioPtr = adata.ioPtr_;
  pData *pdata = ioPtr->getAuxData();
  pData pINames;

  ioPtr->getParameter("input_names", pINames);
  //printEquals(PL_INFO, 0);
  //printf("Two-Parameter Effect Statistics: \n");
  if (pdata->dbleData_ > 0)
  {
    //for (ii = 0; ii < nInputs_; ii++)
    //{
      //for (jj = ii+1; jj < nInputs_; jj++)
      //{
      //  printf("Inputs %4d %4d: sensitivity index = %11.4e ",
      //         ii+1,jj+1,pdata->dbleArray_[ii*nInputs_+jj]);
      //  printf("(normalized=%11.4e)\n",
      //     pdata->dbleArray_[ii*nInputs_+jj]/pdata->dbleData_);
      //}
    //}
    if (plotScilab()) fp = fopen("scilabie.sci", "w");
    else              fp = fopen("matlabie.m", "w");
    if (fp != NULL)
    {
      snprintf(pString,100,
          " This file contains Sobol' two-parameter indices");
      fwriteComment(fp, pString);
      snprintf(pString,100,
          " set sortFlag = 1 and set nn to be the number");
      fwriteComment(fp, pString);
      snprintf(pString,100," of inputs to display.");
      fwriteComment(fp, pString);
      fprintf(fp, "sortFlag = 0;\n");
      fprintf(fp, "nn = %d;\n", nInputs_);
      fprintf(fp, "Mids = [\n");
      for (ii = 0; ii < nInputs_*nInputs_; ii++)
        fprintf(fp,"%24.16e\n", pdata->dbleArray_[ii]);
      fprintf(fp, "];\n");

      if (plotScilab()) fprintf(fp, "  Str = [");
      else              fprintf(fp, "  Str = {");

      for (ii = 0; ii < nInputs_-1; ii++)
      {
        if (pINames.strArray_ != NULL && pINames.strArray_[ii] != NULL)
             fprintf(fp,"'%s',",pINames.strArray_[ii]);
        else fprintf(fp,"'X%d',",ii+1);
      }
      if (plotScilab())
      {
        if (pINames.strArray_ != NULL && 
            pINames.strArray_[nInputs_-1] != NULL)
             fprintf(fp,"'%s'];\n",pINames.strArray_[nInputs_-1]);
        else fprintf(fp,"'X%d'];\n",nInputs_);
      }
      else
      {
        if (pINames.strArray_ != NULL && 
            pINames.strArray_[nInputs_-1] != NULL)
             fprintf(fp,"'%s'};\n",pINames.strArray_[nInputs_-1]);
        else fprintf(fp,"'X%d'};\n",nInputs_);
      }

      fwritePlotCLF(fp);
      fprintf(fp, "ymin = min(Mids);\n");
      fprintf(fp, "ymax = max(Mids);\n");
      fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
      if (plotScilab())
           fprintf(fp, "Mids = matrix(Mids, %d, %d);\n",
                   nInputs_,nInputs_);
      else fprintf(fp, "Mids = reshape(Mids, %d, %d);\n",
                   nInputs_,nInputs_);
      fprintf(fp, "Mids = Mids';\n");
      if (plotScilab())
      {
        fprintf(fp, "drawlater\n");
        fprintf(fp, "hist3d(Mids);\n");
        fprintf(fp, "a=gca();\n");
        fprintf(fp, "a.data_bounds=[0, 0, 0; %d+1, %d+1, ymax];\n",
                nInputs_, nInputs_);
        fprintf(fp, "newtick = a.x_ticks;\n");
        fprintf(fp, "newtick(2) = [1:nn]';\n");
        fprintf(fp, "newtick(3) = Str';\n");
        fprintf(fp, "a.x_ticks = newtick;\n");
        fprintf(fp, "a.x_label.font_size = 3;\n");
        fprintf(fp, "a.x_label.font_style = 4;\n");
        fprintf(fp, "a.y_ticks = newtick;\n");
        fprintf(fp, "a.y_label.font_size = 3;\n");
        fprintf(fp, "a.y_label.font_style = 4;\n");
        fprintf(fp, "drawnow\n");
      }
      else
      {
        fprintf(fp, "bar3(Mids,0.8);\n");
        fprintf(fp, "axis([0 %d+1 0 %d+1 0 ymax])\n",
                nInputs_, nInputs_);
        fprintf(fp, "set(gca,'XTickLabel',Str);\n");
        fprintf(fp, "set(gca,'YTickLabel',Str);\n");
        fprintf(fp, "set(gca, 'fontsize', 12)\n");
        fprintf(fp, "set(gca, 'fontweight', 'bold')\n");
      }
      fwritePlotAxes(fp);
      fwritePlotTitle(fp,"Sobol 2nd Order Indices (+ 1st order)");
      fwritePlotZLabel(fp, "Sobol Indices");
      fwritePlotXLabel(fp, "Inputs");
      fwritePlotYLabel(fp, "Inputs");
      fclose(fp);
      if (plotScilab())
           printf("Two-parameter sensitivities plot file = scilabie.sci\n");
      else printf("Two-parameter sensitivities plot file = matlabie.m\n");
    }
  }
  else
  {
    printf("Total variance = 0 ==> no interaction effect plot.\n");
  }
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
TwoParamAnalyzer& TwoParamAnalyzer::operator=(const TwoParamAnalyzer &)
{
  printOutTS(PL_ERROR, 
             "TwoParamEffect operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int TwoParamAnalyzer::getNumInputs()
{
  return nInputs_;
}
double TwoParamAnalyzer::getOutputMean()
{
  return outputMean_;
}
double TwoParamAnalyzer::getOutputStd()
{
  return outputStd_;
}
double *TwoParamAnalyzer::getMainEffects()
{
  psVector vecVals = VecMainEffects_;
  double* retVal = vecVals.takeDVector();
  return retVal;
}
double **TwoParamAnalyzer::get2ParamEffects()
{
  psMatrix matVals = Mat2ParamEffects_;
  return matVals.takeMatrix2D();
}

