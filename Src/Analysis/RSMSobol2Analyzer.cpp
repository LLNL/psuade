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
// Functions for the class RSMSobol2Analyzer  
// (Sobol' second order sensitivity analysis - with response surface)
// AUTHOR : CHARLES TONG
// DATE   : 2006
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#include "PsuadeUtil.h"
#include "sysdef.h"
#include "psMatrix.h"
#include "Psuade.h"
#include "FuncApprox.h"
#include "Sampling.h"
#include "RSConstraints.h"
#include "PDFManager.h"
#include "pData.h"
#include "PsuadeData.h"
#include "PsuadeConfig.h"
#include "RSMSobol2Analyzer.h"
#include "SobolSampling.h"
#include "SobolAnalyzer.h"
#include "PrintingTS.h"
#include "pdfData.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
RSMSobol2Analyzer::RSMSobol2Analyzer() : Analyzer(),nInputs_(0),
                   outputMean_(0),outputStd_(0)
{
  setName("RSMSOBOL2");
  method_ = 1; /* 0 = numerical integration, 1 = Sobol' */
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
RSMSobol2Analyzer::~RSMSobol2Analyzer()
{
}

// ************************************************************************
// perform analysis (this is intended for library calls)
// ------------------------------------------------------------------------
void RSMSobol2Analyzer::analyze(int nInps, int nSamp, double *lbs,
                                double *ubs, double *XIn, double *YIn,
                                int rstype)
{
  psVector  vecIMeans, vecIStdvs;
  psIVector vecIPDFs;

  aData adata;
  adata.nInputs_ = nInps;
  adata.nOutputs_ = 1;
  adata.nSamples_ = nSamp;
  adata.iLowerB_ = lbs;
  adata.iUpperB_ = ubs;
  adata.sampleInputs_ = XIn;
  adata.sampleOutputs_ = YIn;
  adata.outputID_ = 0;
  adata.printLevel_ = 0;
  vecIPDFs.setLength(nInps);
  vecIMeans.setLength(nInps);
  vecIStdvs.setLength(nInps);
  printf("RSMSobol2 analyze: No distribution information will be used.\n");
  adata.inputPDFs_   = vecIPDFs.getIVector();
  adata.inputMeans_  = vecIMeans.getDVector();
  adata.inputStdevs_ = vecIStdvs.getDVector();
  analyze(adata);
  adata.inputPDFs_   = NULL;
  adata.inputMeans_  = NULL;
  adata.inputStdevs_ = NULL;
  adata.iLowerB_ = NULL;
  adata.iUpperB_ = NULL;
  adata.sampleInputs_ = NULL;
  adata.sampleOutputs_ = NULL;
  adata.faType_ = rstype;
}

// ************************************************************************
// perform analysis
// ------------------------------------------------------------------------
double RSMSobol2Analyzer::analyze(aData &adata)
{
  int    ii, ii2, iL, iR, jj, kk, currNSam1, totalCnt;
  double vce, ddata, ecv;
  char   pString[500], *cString, winput[500], winput2[500];
  pData  pCorMat, pPDF;
  psMatrix  *corMatp;

  //**/ ===============================================================
  //**/ display header 
  //**/ ===============================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
         "*          RS-based Second Order Sobol' Indices \n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"* TO GAIN ACCESS TO DIFFERENT OPTIONS: SET\n");
    printOutTS(PL_INFO,"*\n");
    printOutTS(PL_INFO,
      "* - ana_expert mode to finetune Analysis parameters \n");
    printOutTS(PL_INFO,
      "*   (e.g. sample size for integration can be adjusted).\n");
    printOutTS(PL_INFO,
      "* - rs_expert mode to finetune response surface\n");
    printOutTS(PL_INFO,
      "* - printlevel to 1 or higher to display more information\n");
    printEquals(PL_INFO, 0);
  }
  
  //**/ ---------------------------------------------------------------
  //**/ extract sample data
  //**/ ---------------------------------------------------------------
  nInputs_       = adata.nInputs_;
  int nInputs    = adata.nInputs_;
  int nOutputs   = adata.nOutputs_;
  int nSamples   = adata.nSamples_;
  double *XIn    = adata.sampleInputs_;
  double *YIn    = adata.sampleOutputs_;
  double *xLower = adata.iLowerB_;
  double *xUpper = adata.iUpperB_;
  int outputID   = adata.outputID_;
  int printLevel = adata.printLevel_;
  PsuadeData *ioPtr = adata.ioPtr_;

  //**/ ---------------------------------------------------------------
  //**/ extract sample statistical information
  //**/ ---------------------------------------------------------------
  int    *pdfFlags   = adata.inputPDFs_;
  double *inputMeans = adata.inputMeans_;
  double *inputStdvs = adata.inputStdevs_;
  psIVector vecPdfFlags;
  psVector  vecInpMeans, vecInpStds;
  if (inputMeans == NULL || pdfFlags == NULL || inputStdvs == NULL)
  {
    vecPdfFlags.setLength(nInputs);
    pdfFlags = vecPdfFlags.getIVector();
    vecInpMeans.setLength(nInputs);
    inputMeans = vecInpMeans.getDVector();
    vecInpStds.setLength(nInputs);
    inputStdvs = vecInpStds.getDVector();
  }
  int hasPDF = 0;
  if (pdfFlags != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (pdfFlags[ii] != 0) hasPDF = 1;
  }
  if (psConfig_.InteractiveIsOn())
  {
    if (hasPDF == 0) 
      printOutTS(PL_INFO,
         "Analysis INFO: All inputs have uniform distributions.\n");
    else
    {
      printOutTS(PL_INFO,"Analysis INFO: Non-uniform ");
      printOutTS(PL_INFO,"distributions detected - will be used in\n");
      printOutTS(PL_INFO,"               this analysis.\n");
    }
  }
  if (ioPtr == NULL)
  {
    printOutTS(PL_INFO,
         "Analysis INFO: No data object (PsuadeData) found.\n");
    printOutTS(PL_INFO,
         "          Several features will be turned off.\n");
    corMatp = new psMatrix();
    corMatp->setDim(nInputs, nInputs);
    for (ii = 0; ii < nInputs; ii++) corMatp->setEntry(ii,ii,1.0e0);
  } 
  else
  {
    //**/ if there is any correlation or if the PDF is of type S,
    ioPtr->getParameter("input_cor_matrix", pCorMat);
    corMatp = (psMatrix *) pCorMat.psObject_;
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = 0; jj < ii; jj++)
      {
        if (corMatp->getEntry(ii,jj) != 0.0)
        {
          printOutTS(PL_INFO,"Analysis INFO: This method cannot ");
          printOutTS(PL_INFO,"handle correlated inputs using joint\n");
          printOutTS(PL_INFO,"               PDFs.\n");
          return PSUADE_UNDEFINED;
        }
      }
      if (pdfFlags[ii] == PSUADE_PDF_SAMPLE)
      {
        printOutTS(PL_ERROR,
          "Analysis INFO: This method cannot handle S PDF type.\n");
        return PSUADE_UNDEFINED;
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 1 || nSamples <= 0 || nOutputs <= 0)
  {
    printOutTS(PL_ERROR,"Analysis ERROR: Invalid arguments.\n");
    printOutTS(PL_ERROR,"   nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR,"   nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR,"   nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  }
  if (nInputs <= 2)
  {
    printOutTS(PL_ERROR,
         "Analysis ERROR: no need for this analysis (nInputs<=2).\n");
    return PSUADE_UNDEFINED;
  }
  if (outputID < 0 || outputID >= nOutputs)
  {
    printOutTS(PL_ERROR,
         "Analysis ERROR: invalid outputID (%d).\n",outputID);
    return PSUADE_UNDEFINED;
  }
  int status = 0;
  for (ii = 0; ii < nSamples; ii++)
     if (YIn[nOutputs*ii+outputID] > 0.9*PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR,"Analysis ERROR: Some outputs are ");
    printOutTS(PL_ERROR,"undefined. Prune the undefined sample\n");
    printOutTS(PL_ERROR,
         "                               points and re-run.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ---------------------------------------------------------------
  //**/ set up constraint filters
  //**/ ---------------------------------------------------------------
  RSConstraints *constrPtr = NULL;
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
  }

  //**/ ===============================================================
  //**/  get internal parameters from users
  //**/ (Dec 2023) Decided to remove sample size choices to make the
  //**/ interface less messy
  //**/ ===============================================================
  int nSam1=50000, nSam2=200;
  if (method_ == 0)
  {
    if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn()) 
    {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO,"This analysis uses 2 sets of samples ");
      printOutTS(PL_INFO,"for each pair of inputs - one\n");
      printOutTS(PL_INFO,"(size = N1) for each input pair, one ");
      printOutTS(PL_INFO,"(size = N2) for the other inputs.\n");
      printOutTS(PL_INFO,"The total sample size is:\n");
      printOutTS(PL_INFO,
           "       N = N1 * N2 * nInputs * (nInputs - 1) / 2.\n");
      printOutTS(PL_INFO,"NOW, nInputs = %d\n", nInputs);
      printOutTS(PL_INFO,"Please select your desired N1 and N2.\n");
      printOutTS(PL_INFO,"Recommendation: N1 >> N2.\n");
      printOutTS(PL_INFO,"Default N1 = %d.\n", nSam1);
      printOutTS(PL_INFO,"Default N2 = %d.\n", nSam2);
      printOutTS(PL_INFO,"NOTE: Large N1 and N2 will take a long time.\n");
      printEquals(PL_INFO, 0);
      snprintf(pString,100,"Enter N1 (suggestion: 10000 - 1000000) : ");
      nSam1 = getInt(10000, 1000000, pString);
      snprintf(pString,100,"Enter N2 (suggestion: 100 - 10000) : ");
      nSam2 = getInt(100, 10000, pString);
      printAsterisks(PL_INFO, 0);
    }
    if (psConfig_.InteractiveIsOn()) 
    {
      printOutTS(PL_INFO,"RSMSobol2 INFO: \n");
      printOutTS(PL_INFO,"Default N1 = %d.\n", nSam1);
      printOutTS(PL_INFO,"Default N2 = %d.\n", nSam2);
    }
  }
  //**/ smaller sample size needed for Sobol'
  else if (method_ == 1)
  {
    //**/ for Sobol method, block size is the following.
    nSam2 = 100000;
  }

  //**/ ---------------------------------------------------------------
  //**/ build response surface
  //**/ ---------------------------------------------------------------
  int rstype = adata.faType_;
  FuncApprox *faPtr=NULL;
  if (rstype < 0)
  {
    printf("Select response surface. Options are: \n");
    writeFAInfo(0);
    strcpy(pString, "Choose response surface: ");
    rstype = getInt(0, PSUADE_NUM_RS, pString);
  }
  faPtr = genFA(rstype, nInputs, 0, nSamples);
  faPtr->setBounds(xLower, xUpper);
  psVector vecYT;
  vecYT.setLength(nSamples);
  for (ii = 0; ii < nSamples; ii++) 
    vecYT[ii] = YIn[ii*nOutputs+outputID];
  if (psConfig_.InteractiveIsOn())
    printOutTS(PL_INFO,"Analysis: Creating response surface...\n");
  psConfig_.InteractiveSaveAndReset();
  status = faPtr->initialize(XIn, vecYT.getDVector());
  psConfig_.InteractiveRestore();
  if (psConfig_.InteractiveIsOn()) printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/  use response surface to compute total variance
  //**/ ---------------------------------------------------------------
  //**/ ----------------------------------------------------
  //**/ use a large sample size
  //**/ ----------------------------------------------------
  int nSamp = nSam1 * nSam2;
  if (nSamp > 2000000) nSamp = 2000000;
  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,
      "Analysis INFO: Creating a sample for basic statistics.\n");
    printOutTS(PL_INFO,"               Sample size = %d\n", nSamp);
  }

  //**/ ----------------------------------------------------
  //**/ allocate space
  //**/ ----------------------------------------------------
  psIVector vecST;
  psVector  vecXX, vecYY;
  vecXX.setLength(nSamp*nInputs);
  vecYY.setLength(nSamp);

  //**/ ----------------------------------------------------
  //**/ create a sample
  //**/ ----------------------------------------------------
  psVector vecLB, vecUB, vecOut;
  Sampling   *sampler=NULL;
  if (hasPDF == 1)
  {
    PDFManager *pdfman = new PDFManager();
    pdfman->initialize(nInputs,pdfFlags,inputMeans,
                       inputStdvs,*corMatp,NULL,NULL);
    vecLB.load(nInputs, xLower);
    vecUB.load(nInputs, xUpper);
    vecOut.setLength(nSamp*nInputs);
    pdfman->genSample(nSamp, vecOut, vecLB, vecUB);
    for (ii = 0; ii < nSamp*nInputs; ii++) vecXX[ii] = vecOut[ii];
    delete pdfman;
  }
  else
  {
    if (nInputs < 51) 
    {
      sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
      sampler->setSamplingParams(nSamp, 1, 0);
    }
    else
    {
      sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
      sampler->setSamplingParams(nSamp, 1, 1);
    }
    sampler->setInputBounds(nInputs, xLower, xUpper);
    sampler->setOutputParams(1);
#ifndef PSUADE_OMP
    psConfig_.SamExpertModeSaveAndReset();
#endif
    sampler->initialize(0);
#ifndef PSUADE_OMP
    psConfig_.SamExpertModeRestore();
#endif
    vecST.setLength(nSamp);
    sampler->getSamples(nSamp, nInputs, 1, vecXX.getDVector(), 
                        vecYY.getDVector(), vecST.getIVector());
    delete sampler;
  }

  //**/ ----------------------------------------------------
  //**/ evaluate 
  //**/ ----------------------------------------------------
  if (psConfig_.InteractiveIsOn())
    printOutTS(PL_INFO,
       "Analysis: Response surface evaluation begins ...\n");

  faPtr->evaluatePoint(nSamp,vecXX.getDVector(),vecYY.getDVector());

  if (psConfig_.InteractiveIsOn())
    printOutTS(PL_INFO,
       "Analysis: Response surface evaluation completed.\n");
  
  //**/ ----------------------------------------------------
  //**/ apply filters
  //**/ ----------------------------------------------------
  double *oneSamplePt=NULL;
  for (ii = 0; ii < nSamp; ii++)
  {
     oneSamplePt = &(vecXX[ii*nInputs]);
     status = 1;
     if (constrPtr != NULL)
        ddata = constrPtr->evaluate(oneSamplePt,vecYY[ii],status);
     if (status == 0) vecYY[ii] = PSUADE_UNDEFINED;
  }
  
  //**/ ----------------------------------------------------
  //**/ compute statistics
  //**/ ----------------------------------------------------
  double dmean = 0.0;
  int    sCnt = 0;
  for (ii = 0; ii < nSamp; ii++)
  {
    if (vecYY[ii] != PSUADE_UNDEFINED)
    {
      dmean += vecYY[ii];
      sCnt++;
    }
  }
  if (sCnt > 1) dmean /= (double) sCnt;
  else
  {
    printOutTS(PL_ERROR, 
         "Analysis ERROR: Too few samples that satisify\n");
    printOutTS(PL_ERROR, "constraints (%d out of %d).\n",sCnt,nSamp);
    delete faPtr;
    if (ioPtr == NULL) delete corMatp;
    if (constrPtr != NULL) delete constrPtr;
    return PSUADE_UNDEFINED;
  }
  double variance = 0.0;
  for (ii = 0; ii < nSamp; ii++)
  {
    if (vecYY[ii] != PSUADE_UNDEFINED)
      variance += (vecYY[ii] - dmean) * (vecYY[ii] - dmean) ;
  }
  variance /= (double) sCnt;
  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,
       "Analysis: Sample mean    (based on N = %d) = %10.3e\n",
       sCnt, dmean);
    printOutTS(PL_INFO,
       "Analysis: Sample std dev (based on N = %d) = %10.3e\n",
       sCnt, sqrt(variance));
  }
  if (variance == 0.0) variance = 1.0;

  //**/ ----------------------------------------------------
  //**/ save mean & std
  //**/ ----------------------------------------------------
  outputMean_ = dmean;
  outputStd_ = sqrt(variance);

  //**/ ===============================================================
  //**/ use Sobol' method if requested
  //**/ ===============================================================
  if (method_ == 1)
  {
    if (psConfig_.DiagnosticsIsOn())
      printf("Entering RSMSobol2Analzer::analyze::Sobol\n"); 
    int blkSize = nInputs * (nInputs - 1) / 2 + 2;
    int nPerBlk = 100000;
    printf("RSMSobol2 Sobol: nSams = %d\n", nPerBlk);
    int nSobolSams = nPerBlk * blkSize, iOne=1;

#if 0
    //**/ create M1 and M2 assuming input distributions
    PDFManager *pdfman = new PDFManager();
    pdfman->initialize(nInputs,pdfFlags,inputMeans,inputStdvs,
                       *corMatp,NULL,NULL);
    vecLB.load(nInputs, xLower);
    vecUB.load(nInputs, xUpper);
    vecOut.setLength(nPerBlk*2*nInputs);
    pdfman->genSample(nPerBlk*2, vecOut, vecLB, vecUB);
    delete pdfman;
#else
    //**/ prepare PDF object for createM1M2
    pdfData pdfObj;
    pdfObj.nInputs_  = nInputs_;
    pdfObj.nSamples_ = nPerBlk;
    pdfObj.VecPTypes_.load(nInputs_, pdfFlags);
    pdfObj.VecParam1_.load(nInputs_, inputMeans);
    pdfObj.VecParam2_.load(nInputs_, inputStdvs);
    pdfObj.VecLBs_.load(nInputs_, xLower);
    pdfObj.VecUBs_.load(nInputs_, xUpper);
    pdfObj.MatCor_ = (*corMatp);
#endif
    
    //**/ Step 1: Create a Sobol' sample
    SobolSampling *sobolSampler = new SobolSampling();
    sobolSampler->setOrder(2);
    sobolSampler->setPrintLevel(-2);
    sobolSampler->setInputBounds(nInputs,xLower,xUpper);
    sobolSampler->setOutputParams(iOne);
    sobolSampler->setSamplingParams(nSobolSams, 1, 0);
#if 0
    sobolSampler->setM1M2(vecOut);
#else
    sobolSampler->createM1M2(pdfObj);
#endif
    sobolSampler->initialize(0);
    psVector  vecSobolX, vecSobolY;
    psIVector vecSobolS;
    vecSobolX.setLength(nInputs*nSobolSams);
    vecSobolY.setLength(nSobolSams);
    vecSobolS.setLength(nSobolSams);
    sobolSampler->getSamples(nSobolSams, nInputs, iOne, 
                     vecSobolX.getDVector(), vecSobolY.getDVector(), 
                     vecSobolS.getIVector());

    //**/ Step 2: evaluate the sample
    faPtr->evaluatePoint(nSobolSams,vecSobolX.getDVector(),
                         vecSobolY.getDVector());

    //**/ Step 3: filter infeasible points
    double *tempV;
    if (constrPtr != NULL)
    { 
      totalCnt = 0;
      tempV = vecSobolX.getDVector();
      for (kk = 0; kk < nSobolSams; kk++)
      {
        ddata = constrPtr->evaluate(&tempV[kk*nInputs],vecSobolY[kk],
                                    status);
        if (status == 0)
        {
          vecSobolY[kk] = PSUADE_UNDEFINED;
          totalCnt++;
        }
      }
      if (totalCnt <= 1)
      { 
        printf("RSMAnalysis ERROR: Too few samples left after filtering\n");
        if (faPtr != NULL) delete faPtr;
        return -1;
      }
      if (totalCnt > nSobolSams/2)
        printf("RSMAnalysis INFO: Constraints filter out > 1/2 points.\n");
    }
    
    //**/ Step 4: analyze 
    SobolAnalyzer *sobolAnalyzer = new SobolAnalyzer();
    sobolAnalyzer->setOrder(2);
    aData sobolAPtr;
    sobolAPtr.printLevel_ = -1;
    sobolAPtr.nSamples_   = nSobolSams;
    sobolAPtr.nInputs_    = nInputs;
    sobolAPtr.nOutputs_   = iOne;
    sobolAPtr.sampleInputs_  = vecSobolX.getDVector();
    sobolAPtr.sampleOutputs_ = vecSobolY.getDVector();
    for (ii = 0; ii < nSobolSams; ii++) vecSobolS[ii] = 1;
    sobolAPtr.sampleStates_  = vecSobolS.getIVector();
    sobolAPtr.iLowerB_  = xLower;
    sobolAPtr.iUpperB_  = xUpper;
    sobolAPtr.outputID_ = 0;
    sobolAPtr.ioPtr_    = NULL;
    sobolAnalyzer->analyze(sobolAPtr);
    delete faPtr;

    //**/ Step 5: print results 
    if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
    {
      printAsterisks(PL_INFO, 0);
      for (ii = 0; ii < nInputs; ii++)
      {
        for (jj = ii+1; jj < nInputs; jj++)
        {
          printOutTS(PL_INFO,
            "RSMSobol2: Sobol' 2nd-order index for input %3d,%3d = %10.3e\n",
            ii+1, jj+1, sobolAnalyzer->get_S2(ii*nInputs+jj));
        }
      }
    }
    pData *pObj = NULL;
    if (ioPtr != NULL)
    {
      pObj = ioPtr->getAuxData();
      pObj->nDbles_ = nInputs*nInputs;
      pObj->dbleArray_ = new double[nInputs*nInputs];
      //**/ returns unnormalized results
      for (ii = 0; ii < nInputs*nInputs; ii++)
        pObj->dbleArray_[ii] = sobolAnalyzer->get_S2(ii) * variance;
      pObj->dbleData_ = variance;
    }
    delete sobolAnalyzer;
    return 0;
  }

  //**/ ---------------------------------------------------------------
  //**/  use response surface to perform Sobol two parameter test
  //**/ ---------------------------------------------------------------
  //**/ set up the sampling method
  psVector vecCLBs, vecCUBs;
  vecCLBs.setLength(nInputs);
  vecCUBs.setLength(nInputs);
  psVector vecXT, vecMeans, vecVars;
  vecXT.setLength(nSam2*nInputs);
  vecYT.setLength(nSam2*nInputs);
  vecMeans.setLength(nSam1);
  vecVars.setLength(nSam1);
  psIVector vecBins;
  vecBins.setLength(nSam1);
  psIVector vecPdfFlags1, vecPdfFlags2;
  psVector  vecInpMeans1, vecInpStds1, vecInpMeans2, vecInpStds2;
  vecPdfFlags1.setLength(2);
  vecInpMeans1.setLength(2);
  vecInpStds1.setLength(2);
  vecPdfFlags2.setLength(nInputs-2);
  vecInpMeans2.setLength(nInputs-2);
  vecInpStds2.setLength(nInputs-2);

  psVector vecSamPts2D, vecSubSamPts;
  vecSamPts2D.setLength(nSam1*2);
  vecSubSamPts.setLength(nInputs*nSam2);

  //**/ ---------------------------------------------------------------
  //**/ set up to return more detailed data
  //**/ ---------------------------------------------------------------
  pData *pPtr = NULL;
  if (ioPtr != NULL)
  {
    pPtr = ioPtr->getAuxData();
    pPtr->nDbles_ = nInputs * nInputs;
    pPtr->dbleArray_ = new double[nInputs * nInputs];
    for (ii = 0; ii < nInputs*nInputs; ii++) pPtr->dbleArray_[ii] = 0.0;
    pPtr->dbleData_ = variance;
  }

  //**/ ---------------------------------------------------------------
  //**/ loop through each pair of inputs
  //**/ ---------------------------------------------------------------
  printf("RSMSobol2 INFO: nSam1, nSam2 = %d %d\n",nSam1,nSam2);
  if (psConfig_.InteractiveIsOn()) printAsterisks(PL_INFO, 0);
  vecVces_.setLength(nInputs*nInputs);
  vecEcvs_.setLength(nInputs*nInputs);
  for (ii = 0; ii < nInputs; ii++)
  {
    for (ii2 = ii+1; ii2 < nInputs; ii2++)
    {
      vce = 0.0;
      if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
        printOutTS(PL_DETAIL,
          "Analysis: Processing input pair %d, %d\n",ii+1,ii2+1);

      //**/ use 2 levels of refinements to compute confidence interval
      currNSam1 = nSam1 / 2;
      for (iR = 0; iR < 2; iR++)
      {
        if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
          printOutTS(PL_DETAIL,
             "Analysis: Processing refinement %d (4)\n",iR+1);

        //**/ create sample with pdf for 2 inputs
        vecCLBs[0] = xLower[ii];
        vecCUBs[0] = xUpper[ii];
        vecCLBs[1] = xLower[ii2];
        vecCUBs[1] = xUpper[ii2];
        if (hasPDF == 1)
        {
          psMatrix corMat1;
          corMat1.setDim(2,2);
          corMat1.setEntry(0, 0, corMatp->getEntry(ii,ii));
          corMat1.setEntry(1, 1, corMatp->getEntry(ii2,ii2));
          vecPdfFlags1[0] = pdfFlags[ii];
          vecPdfFlags1[1] = pdfFlags[ii2];
          vecInpMeans1[0] = inputMeans[ii];
          vecInpMeans1[1] = inputMeans[ii2];
          vecInpStds1[0]  = inputStdvs[ii];
          vecInpStds1[1]  = inputStdvs[ii2];
          PDFManager *pdfman1 = new PDFManager();
          pdfman1->initialize(2,vecPdfFlags1.getIVector(),
                    vecInpMeans1.getDVector(),vecInpStds1.getDVector(),
                    corMat1,NULL,NULL);
          vecLB.load(2, vecCLBs.getDVector());
          vecUB.load(2, vecCUBs.getDVector());
          vecOut.setLength(currNSam1*2);
          pdfman1->genSample(currNSam1,vecOut,vecLB,vecUB);
          for (jj = 0; jj < currNSam1*2; jj++) 
            vecSamPts2D[jj] = vecOut[jj];
          delete pdfman1;
        }
        else
        {
          sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
          sampler->setInputBounds(2,vecCLBs.getDVector(),
                                  vecCUBs.getDVector());
          sampler->setOutputParams(1);
          sampler->setSamplingParams(currNSam1, 1, 0);
#ifndef PSUADE_OMP
          psConfig_.SamExpertModeSaveAndReset();
#endif
          sampler->initialize(0);
#ifndef PSUADE_OMP
          psConfig_.SamExpertModeRestore();
#endif
          vecST.setLength(currNSam1);
          psVector vecZT;
          vecZT.setLength(currNSam1);
          sampler->getSamples(currNSam1,2,1,
                       vecSamPts2D.getDVector(),vecZT.getDVector(),
                       vecST.getIVector());
          delete sampler;
        }

        //**/ create sample with pdf for the other inputs
        if (hasPDF == 1)
        {
          psMatrix corMat2;
          corMat2.setDim(nInputs-2, nInputs-2);
          for (jj = 0; jj < ii; jj++)
          {
            vecCLBs[jj] = xLower[jj];
            vecCUBs[jj] = xUpper[jj];
            corMat2.setEntry(jj, jj, corMatp->getEntry(jj,jj));
            vecPdfFlags2[jj] = pdfFlags[jj];
            vecInpMeans2[jj] = inputMeans[jj];
            vecInpStds2[jj]  = inputStdvs[jj];
          }
          for (jj = ii+1; jj < ii2; jj++)
          {
            vecCLBs[jj-1] = xLower[jj];
            vecCUBs[jj-1] = xUpper[jj];
            corMat2.setEntry(jj-1, jj-1, corMatp->getEntry(jj,jj));
            vecPdfFlags2[jj-1] = pdfFlags[jj];
            vecInpMeans2[jj-1] = inputMeans[jj];
            vecInpStds2[jj-1]  = inputStdvs[jj];
          }
          for (jj = ii2+1; jj < nInputs; jj++)
          {
            vecCLBs[jj-2] = xLower[jj];
            vecCUBs[jj-2] = xUpper[jj];
            corMat2.setEntry(jj-2, jj-2, corMatp->getEntry(jj,jj));
            vecPdfFlags2[jj-2] = pdfFlags[jj];
            vecInpMeans2[jj-2] = inputMeans[jj];
            vecInpStds2[jj-2]  = inputStdvs[jj];
          }
          PDFManager *pdfman2 = new PDFManager();
          pdfman2->initialize(nInputs-2,vecPdfFlags2.getIVector(),
                    vecInpMeans2.getDVector(),vecInpStds2.getDVector(),
                    corMat2,NULL,NULL);
          vecLB.load(nInputs-2, vecCLBs.getDVector());
          vecUB.load(nInputs-2, vecCUBs.getDVector());
          vecOut.setLength(nSam2*(nInputs-2));
          pdfman2->genSample(nSam2, vecOut, vecLB, vecUB);
          for (jj = 0; jj < nSam2*(nInputs-2); jj++)
            vecXT[jj] = vecOut[jj];
          delete pdfman2;
        }
        else
        {
          for (jj = 0; jj < ii; jj++)
          {
            vecCLBs[jj] = xLower[jj];
            vecCUBs[jj] = xUpper[jj];
          }
          for (jj = ii+1; jj < ii2; jj++)
          {
            vecCLBs[jj-1] = xLower[jj];
            vecCUBs[jj-1] = xUpper[jj];
          }
          for (jj = ii2+1; jj < nInputs; jj++)
          {
            vecCLBs[jj-2] = xLower[jj];
            vecCUBs[jj-2] = xUpper[jj];
          }
          if (nInputs-1 > 51)
               sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
          else sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
          sampler->setInputBounds(nInputs-2,vecCLBs.getDVector(), 
                                  vecCUBs.getDVector());
          sampler->setOutputParams(1);
          sampler->setSamplingParams(nSam2, 1, 1);
#ifndef PSUADE_OMP
          psConfig_.SamExpertModeSaveAndReset();
#endif
          sampler->initialize(0);
#ifndef PSUADE_OMP
          psConfig_.SamExpertModeRestore();
#endif
          vecST.setLength(nSam2);
          psVector vecZT;
          vecZT.setLength(nSam2);
          sampler->getSamples(nSam2,nInputs-2,1,vecXT.getDVector(),
                              vecZT.getDVector(),vecST.getIVector());
          delete sampler;
        }

        //**/ use currNLevels levels per input
        double *XPtr = vecXT.getDVector();
        for (iL = 0; iL < currNSam1; iL++)
        {
          //**/ extract the sample point 
          for (jj = 0; jj < nSam2; jj++)
          {
            //**/ extract the sample point and evaluate
            oneSamplePt = &(XPtr[jj*(nInputs-2)]);
            for (kk = 0; kk < ii; kk++)
              vecSubSamPts[jj*nInputs+kk] = oneSamplePt[kk];
            for (kk = ii+1; kk < ii2; kk++)
              vecSubSamPts[jj*nInputs+kk] = oneSamplePt[kk-1];
            for (kk = ii2+1; kk < nInputs; kk++)
              vecSubSamPts[jj*nInputs+kk] = oneSamplePt[kk-2];
            vecSubSamPts[jj*nInputs+ii]  = vecSamPts2D[iL*2];
            vecSubSamPts[jj*nInputs+ii2] = vecSamPts2D[iL*2+1];
          }

          //**/ evaluate
          faPtr->evaluatePoint(nSam2,vecSubSamPts.getDVector(),
                               vecYT.getDVector());

          //**/ go through all filters the sample point and evaluate
          double *dPtr = vecSubSamPts.getDVector();
          for (jj = 0; jj < nSam2; jj++)
          {
            oneSamplePt = &(dPtr[jj*nInputs]);
            status = 1;
            if (constrPtr != NULL)
              ddata = constrPtr->evaluate(oneSamplePt,vecYT[jj],status);
            if (status == 0) vecYT[jj] = PSUADE_UNDEFINED;
          }

          //**/ compute the mean at each input pair levels
          vecMeans[iL] = 0.0;
          sCnt = 0;
          for (jj = 0; jj < nSam2; jj++)
          {
            if (vecYT[jj] != PSUADE_UNDEFINED)
            {
              vecMeans[iL] += vecYT[jj];
              sCnt++;
            }
          }
          vecBins[iL] = sCnt;
          if (sCnt < 1 && printLevel >= 5)
            printOutTS(PL_INFO, 
                 "RSMAnalysis WARNING: Subsample size = 0.\n");
          if (sCnt < 1) vecMeans[iL] = PSUADE_UNDEFINED;
          else          vecMeans[iL] /= (double) sCnt;

          //**/ compute the variance  at each input pair levels
          vecVars[iL] = 0.0;
          ddata = vecMeans[iL];
          for (jj = 0; jj < nSam2; jj++)
          {
            if (vecYT[jj] != PSUADE_UNDEFINED)
              vecVars[iL] += (vecYT[jj]-ddata)*(vecYT[jj]-ddata);
          }
          if (sCnt < 1) vecVars[iL] = PSUADE_UNDEFINED;
          else          vecVars[iL] /= (double) sCnt;

          //**/printOutTS(PL_DUMP,"RSMAnalysis: inputs (%d,%d)\n",
          //**/           ii+1,ii2+1);
          //**/printOutTS(PL_DUMP,
          //**/     "  refinement %d, gridpoint (%d), size %d (%d)\n",
          //**/          iR, iL+1, sCnt, nSam2);
          //**/printOutTS(PL_DUMP,"     mean = %e\n", vecMeans[iL]);
          //**/printOutTS(PL_DUMP,"     var  = %e\n", vecVars[iL]);
        }

        //**/ compute the variance of the means for each input pair
        dmean = 0.0;
        totalCnt = 0;
        for (iL = 0; iL < currNSam1; iL++) 
          totalCnt += vecBins[iL];
        if (totalCnt == 0)
        {
          printOutTS(PL_ERROR,
               "RSMAnalysis ERROR: Empty constrained space.\n");
          printOutTS(PL_ERROR,
               "            Either try larger sample size or\n");
          printOutTS(PL_ERROR,"          use looser constraints.\n");
          exit(1);
        }
        for (iL = 0; iL < currNSam1; iL++)
        {
          if (vecMeans[iL] != PSUADE_UNDEFINED)
            dmean += vecMeans[iL] * vecBins[iL] / totalCnt;
        }
        vce = 0.0;
        for (iL = 0; iL < currNSam1; iL++)
        {
          if (vecMeans[iL] != PSUADE_UNDEFINED)
            vce += (vecMeans[iL]-dmean) * (vecMeans[iL]-dmean) * 
                    vecBins[iL] / totalCnt;
        }

        //**/ compute the mean of the variances for each input pair
        ecv = 0.0;
        for (iL = 0; iL < currNSam1; iL++)
          if (vecVars[iL] != PSUADE_UNDEFINED) 
            ecv += vecVars[iL] * vecBins[iL] / totalCnt;

        if (psConfig_.InteractiveIsOn()) 
        {
          if (printLevel > 1 || iR == 1)
            printOutTS(PL_INFO, 
               "RSMSobol2: VCE(%3d,%3d) = %10.3e, (normalized) = %10.3e\n",
               ii+1, ii2+1, vce, vce/variance);
          if (printLevel > 2)
            printOutTS(PL_INFO, 
              "RSMSobol2: ECV(%3d,%3d) = %10.3e, (normalized) = %10.3e\n",
              ii+1, ii2+1, ecv, ecv/variance);
        }
        currNSam1 *= 2;
      }

      //save vecVces & vecEcvs
      vecVces_[ii*nInputs+ii2] = vce;
      vecEcvs_[ii*nInputs+ii2] = ecv;
      vecVces_[ii2*nInputs+ii] = vce;
      vecEcvs_[ii2*nInputs+ii] = ecv;
      if (pPtr != NULL)
      {
        pPtr->dbleArray_[ii*nInputs+ii2] = vce;
        pPtr->dbleArray_[ii2*nInputs+ii] = vce;
      }
    }
  }
  if (psConfig_.InteractiveIsOn()) printAsterisks(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ clean up
  //**/ ---------------------------------------------------------------
  delete faPtr;
  if (ioPtr == NULL) delete corMatp;
  if (constrPtr != NULL) delete constrPtr;
  return 0.0;
}

// ************************************************************************
// set internal parameters 
// ------------------------------------------------------------------------
int RSMSobol2Analyzer::setParam(int argc, char **argv)
{   
  char  *request = (char *) argv[0];
  if      (!strcmp(request,"ana_rssobol2_ni"))    method_ = 0;
  else if (!strcmp(request,"ana_rssobol2_sobol")) method_ = 1;
  return 0;
}   
    
// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
RSMSobol2Analyzer& RSMSobol2Analyzer::operator=(const RSMSobol2Analyzer &)
{
  printOutTS(PL_ERROR,"RSMSobol2 operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int RSMSobol2Analyzer::get_nInputs()
{
  return nInputs_;
}

double RSMSobol2Analyzer::get_outputMean()
{
  return outputMean_;
}

double RSMSobol2Analyzer::get_outputStd()
{
  return outputStd_;
}

double RSMSobol2Analyzer::get_vce(int ind1, int ind2)
{
  if (ind1 < 0 || ind1 >= nInputs_)
  {
    printf("RSMSobol2 ERROR: get_vce index 1 error.\n");
    printf("          Incoming = %d (range: [0,%d])\n",
           ind1,nInputs_-1);
    return 0.0;
  }
  if (ind2 < 0 || ind2 >= nInputs_)
  {
    printf("RSMSobol2 ERROR: get_vce index 2 error.\n");
    printf("          Incoming = %d (range: [0,%d])\n",
           ind2,nInputs_-1);
    return 0.0;
  }
  if (vecVces_.length() == 0)
  {
    printf("RSMSobol2 ERROR: get_vce has no returned value.\n");
    return 0;
  }
  return vecVces_[ind1*nInputs_+ind2];
}

double RSMSobol2Analyzer::get_ecv(int ind1, int ind2)
{
  if (ind1 < 0 || ind1 >= nInputs_)
  {
    printf("RSMSobol2 ERROR: get_ecv index 1 error.\n");
    return 0.0;
  }
  if (ind2 < 0 || ind2 >= nInputs_)
  {
    printf("RSMSobol2 ERROR: get_ecv index 2 error.\n");
    return 0.0;
  }
  if (vecEcvs_.length() == 0)
  {
    printf("RSMSobol2 ERROR: get_ecv has no value.\n");
    return 0;
  }
  return vecEcvs_[ind1*nInputs_+ind2];
}

