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
// Functions for the class RSMSobol1Analyzer  
// AUTHOR : CHARLES TONG
// DATE   : 2006
//**/ ---------------------------------------------------------------------
//**/ constrained Sobol main effect analysis (recommended for response 
//**/ surface models only since it takes many function evaluations)
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "PsuadeUtil.h"
#include "sysdef.h"
#include "psMatrix.h"
#include "pData.h"
#include "pdfData.h"
#include "RSMSobol1Analyzer.h"
#include "SobolSampling.h"
#include "SobolAnalyzer.h"
#include "Sampling.h"
#include "PDFManager.h"
#include "PDFNormal.h"
#include "RSConstraints.h"
#include "Psuade.h"
#include "PsuadeData.h"
#include "PsuadeConfig.h"
#include "PrintingTS.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
RSMSobol1Analyzer::RSMSobol1Analyzer() : Analyzer(),nInputs_(0),
                     outputMean_(0), outputStd_(0)
{
  setName("RSMSOBOL1");
  method_ = 1;  /* 0 - numerical integration, 1 - Sobol' method */
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
RSMSobol1Analyzer::~RSMSobol1Analyzer()
{
}

// ************************************************************************
// perform analysis (intended for library call)
// ------------------------------------------------------------------------
void RSMSobol1Analyzer::analyze(int nInps, int nSamp, double *lbs, 
                                double *ubs, double *X, double *Y,
                                int rstype)
{
  aData adata;
  adata.nInputs_  = nInps;
  adata.nOutputs_ = 1;
  adata.nSamples_ = nSamp;
  adata.iLowerB_  = lbs;
  adata.iUpperB_  = ubs;
  adata.sampleInputs_  = X;
  adata.sampleOutputs_ = Y;
  adata.outputID_   = 0;
  adata.printLevel_ = 0;
  analyze(adata);
  adata.iLowerB_ = NULL;
  adata.iUpperB_ = NULL;
  adata.sampleInputs_  = NULL;
  adata.sampleOutputs_ = NULL;
  adata.faType_ = rstype;
}

// ************************************************************************
// perform analysis (different parameter list)
// ------------------------------------------------------------------------
double RSMSobol1Analyzer::analyze(aData &adata)
{
  int    ii, jj, kk, ss, count, status, nSamp, rstype;
  double dmean, dstds, ddata, ddata2, *tempV, *oneSamplePt;
  char   pString[500], *cString, winput1[500], winput2[500];
  FuncApprox    *faPtr=NULL;
  psVector      vecIn, vecOut, vecUB, vecLB, vecY;
  pData         pCorMat, pPtr;
  psMatrix      *corMatp=NULL;
  Sampling      *sampler;

  //**/ ===============================================================
  //**/ display header 
  //**/ ===============================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
         "*          RS-based First Order Sobol' Analysis \n");
    printOutTS(PL_INFO,
         "*          (based on numeridal integration)\n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"* TO GAIN ACCESS TO DIFFERENT OPTIONS: SET\n");
    printOutTS(PL_INFO,
         "* - ana_expert to finetune RSMSobol1 parameters\n");
    printOutTS(PL_INFO,
         "*   (e.g. to adjust integration sample size).\n");
    printOutTS(PL_INFO,
         "* - rs_expert mode to finetune response surface\n");
    printOutTS(PL_INFO,"* - printlevel to display more information\n");
    printOutTS(PL_INFO,
         "* Or, use configure file to finetune parameters\n");
    printEquals(PL_INFO, 0);
  }
 
  //**/ ===============================================================
  //**/ extract sample data and information
  //**/ ===============================================================
  int nInputs    = adata.nInputs_;
  nInputs_       = nInputs;
  int nOutputs   = adata.nOutputs_;
  int nSamples   = adata.nSamples_;
  double *xLower = adata.iLowerB_;
  double *xUpper = adata.iUpperB_;
  double *XIn    = adata.sampleInputs_;
  double *YIn    = adata.sampleOutputs_;
  int outputID   = adata.outputID_;
  int printLevel = adata.printLevel_;
  PsuadeData *ioPtr  = adata.ioPtr_;
  int    *pdfFlags   = adata.inputPDFs_;
  double *inputMeans = adata.inputMeans_;
  double *inputStdvs = adata.inputStdevs_;

  //**/ ===============================================================
  //**/ extract input PDF information (if none, set all to none - 0)
  //**/ ===============================================================
  psIVector vecPdfFlags;
  psVector  vecInpMeans, vecInpStds;
  if (inputMeans == NULL || pdfFlags == NULL || inputStdvs == NULL)
  {
    //**/ note: setLength actually sets the vector to all 0's
    vecPdfFlags.setLength(nInputs);
    pdfFlags = vecPdfFlags.getIVector();
    vecInpMeans.setLength(nInputs);
    inputMeans = vecInpMeans.getDVector();
    vecInpStds.setLength(nInputs);
    inputStdvs = vecInpStds.getDVector();
  }
  //**/ if other than uniform PDF, set hasPDF=1 
  //**/ Also, check for S PDFs and flag error
  int hasPDF=0, corFlag=0;
  if (pdfFlags != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (pdfFlags[ii] != 0) hasPDF = 1;
    for (ii = 0; ii < nInputs; ii++)
      if (pdfFlags[ii] == PSUADE_PDF_SAMPLE) corFlag = 1;
  }
  if (psConfig_.InteractiveIsOn())
  {
    if (hasPDF == 0) 
      printOutTS(PL_INFO,
       "* RSMAnalysis INFO: All uniform distributions.\n");
    else
      printOutTS(PL_INFO,
       "* RSMAnalysis INFO: Some inputs have non-uniform distribution.\n");
  }

  //**/ ===============================================================
  //**/ error checking
  //**/ ===============================================================
  if (nInputs <= 0 || nSamples <= 0 || nOutputs <= 0) 
  {
    printOutTS(PL_ERROR, "RSMAnalysis ERROR: Invalid arguments.\n");
    printOutTS(PL_ERROR, "   nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR, "   nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR, "   nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 
  if (outputID >= nOutputs || outputID < 0)
  {
    printOutTS(PL_ERROR,"RSMAnalysis ERROR: Invalid output ID (%d).\n", 
               outputID);
    return PSUADE_UNDEFINED;
  }
  if (nInputs <= 1)
  {
    printOutTS(PL_ERROR,
         "RSMAnalysis INFO: Analysis not needed since nInputs = 1\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ===============================================================
  //**/ get or create correlation matrix
  //**/ ===============================================================
  if (ioPtr == NULL)
  {
    if (psConfig_.InteractiveIsOn())
    {
      printOutTS(PL_INFO,
           "RSMAnalysis INFO: No data object (PsuadeData) found.\n");
      printOutTS(PL_INFO,"   Several features will be turned off.\n");
      printOutTS(PL_INFO,
           "   E.g. No input correlation, if any, will be used.\n");
    }
    //**/ set default correlation to be identity matrix
    corMatp = new psMatrix();
    corMatp->setDim(nInputs, nInputs);
    for (ii = 0; ii < nInputs; ii++) corMatp->setEntry(ii,ii,1.0e0);
  } 
  else
  {
    //**/ detect if correlation matrix is not diagonal
    ioPtr->getParameter("input_cor_matrix", pCorMat);
    corMatp = (psMatrix *) pCorMat.psObject_;
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = 0; jj < ii; jj++)
      {
        if (corMatp->getEntry(ii,jj) != 0.0)
        {
          if (psConfig_.InteractiveIsOn())
          {
            printOutTS(PL_INFO, 
              "RSMAnalysis INFO: Correlated inputs detected.\n");
            printOutTS(PL_INFO, 
              "            Alternative analyis is to be performed.\n");
          }
          corFlag = 1;
        }
      }
    }
  }

  //**/ ===============================================================
  //**/ check the sample outputs to see if any is undefined
  //**/ ===============================================================
  status = 0;
  for (ii = 0; ii < nSamples; ii++)
    if (YIn[nOutputs*ii+outputID] > 0.9*PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR,
        "RSMAnalysis ERROR: Some outputs are undefined.\n");
    printOutTS(PL_ERROR,
        "            Prune the undefined sample points and re-run.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ===============================================================
  //**/ if there are input correlation, use a different analyzer
  //**/ (Note: this includes the presence of S-type PDFs
  //**/ ===============================================================
  if (corFlag != 0) 
  {
    //**/ first clean up
    if (ioPtr == NULL) delete corMatp;
    printf("RSMSobol1 ERROR: Cannot handle input correlations.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ===============================================================
  //**/ set up constraint filters
  //**/ ===============================================================
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
    else
      printf("RSMAnalysis INFO: %d constraints detected.\n",nConstr);
  }

  //**/ ===============================================================
  //**/ get response surface selection from user, if needed 
  //**/ ===============================================================
  if (adata.faType_ < 0)
  {
    printf("Select response surface. Options are: \n");
    writeFAInfo(0);
    strcpy(pString, "Choose response surface: ");
    rstype = getInt(0, PSUADE_NUM_RS, pString);
  }
  else rstype = adata.faType_;

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
      printOutTS(PL_INFO,"for each inputs - one (size = N1)\n");
      printOutTS(PL_INFO,"for each input pair, one ");
      printOutTS(PL_INFO,"(size = N2) for the other inputs.\n");
      printOutTS(PL_INFO,"The total sample size is: ");
      printOutTS(PL_INFO," N = N1 * N2 * nInputs\n");
      printOutTS(PL_INFO,"No, nInputs = %d\n", nInputs);
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
      printOutTS(PL_INFO,"RSMSobol1 INFO: \n");
      printOutTS(PL_INFO,"Default N1 = %d.\n", nSam1);
      printOutTS(PL_INFO,"Default N2 = %d.\n", nSam2);
    }
  }
  else if (method_ == 1)
  {
    //**/ for Sobol method, block size is the following.
    nSam2 = 100000;
  }

  //**/ ===============================================================
  //**/ use response surface to compute mean and variance
  //**/ do it 1000 times using fuzzy evaluations
  //**/ ===============================================================
  int nSamp2 = 100000;
  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,
      "RSMAnalysis INFO: Creating a sample for basic statistics.\n");
    printOutTS(PL_INFO,"            Sample size = %d\n", nSamp2);
  }

  //**/ ---------------------------------------------------------
  //**/ generate a large sample for computing basic statistics
  //**/ ==> vecXX and vecYY (with sample size nSamp2) 
  //**/ hasPDF==1 means some inputs have non-uniform PDFS, but
  //**/ there is no S-type or any correlation allowed here
  //**/ ---------------------------------------------------------
  psIVector vecSS;
  psVector  vecXX, vecYY;
  vecXX.setLength(nSamp2*nInputs);
  vecYY.setLength(nSamp2);
  if (hasPDF == 1)
  {
    //**/ create sample with pdf for all inputs
    PDFManager *pdfman = new PDFManager();
    pdfman->initialize(nInputs,pdfFlags,inputMeans,inputStdvs,
                       *corMatp,NULL,NULL);
    vecLB.load(nInputs, xLower);
    vecUB.load(nInputs, xUpper);
    vecOut.setLength(nSamp2*nInputs);
    pdfman->genSample(nSamp2, vecOut, vecLB, vecUB);
    for (ii = 0; ii < nSamp2*nInputs; ii++) vecXX[ii] = vecOut[ii];
    delete pdfman;
  }
  else
  {
    if (nInputs > 51)
    {
      sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
      sampler->setSamplingParams(nSamp2, 1, 1);
    }
    else 
    {
      sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
      sampler->setSamplingParams(nSamp2, 1, 0);
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
    vecSS.setLength(nSamp2);
    sampler->getSamples(nSamp2, nInputs, 1, vecXX.getDVector(), 
                        vecYY.getDVector(), vecSS.getIVector());
    delete sampler;
  }

  //**/ ---------------------------------------------------------
  //**/ compute basic statistics
  //**/ ---------------------------------------------------------
  psVector  vecBX, vecBY;
  vecBX.setLength(nSamples*nInputs);
  vecBY.setLength(nSamples);
  faPtr = genFA(rstype, nInputs, 0, nSamples);
  faPtr->setBounds(xLower, xUpper);
  faPtr->setOutputLevel(0);
  for (ss = 0; ss < nSamples*nInputs; ss++) vecBX[ss] = XIn[ss];
  for (ss = 0; ss < nSamples; ss++) 
    vecBY[ss] = YIn[ss*nOutputs+outputID];

  //**/ create a response surface
  printf("RSMAnalysis INFO: Building response surface (%d)\n",rstype);
  psConfig_.InteractiveSaveAndReset();
  status = faPtr->initialize(vecBX.getDVector(),vecBY.getDVector());
  psConfig_.InteractiveRestore();
  if (status != 0)
  {
    printf("RSMAnalysis ERROR: In initializing response surface.\n");
    if (faPtr != NULL) delete faPtr;
    return -1;
  }

  //**/ evaluate the large sample
  printf("RSMAnalysis INFO: Evaluating RS for basic statistics\n");
  faPtr->evaluatePoint(nSamp2,vecXX.getDVector(),
                       vecYY.getDVector());

  //**/ filter out unwanted samples
  count = 0;
  if (constrPtr != NULL)
  {
    tempV = vecXX.getDVector();
    for (kk = 0; kk < nSamp2; kk++)
    {
      ddata = constrPtr->evaluate(&tempV[kk*nInputs],vecYY[kk],
                                  status);
      if (status == 0) 
      {
        vecYY[kk] = PSUADE_UNDEFINED;
        count++;
      }
    }
  }
  count = nSamp2 - count;
  if (count < nSamp2)
  {
    printOutTS(PL_INFO,
         "RSMAnalysis INFO: %6.2f percent passes the contraints.\n",
         (double) count * 100.0 /((double) nSamp2));
  }
  if (count <= 1)
  {
    printf("RSMAnalysis ERROR: Too few samples left after filtering\n");
    if (faPtr != NULL) delete faPtr;
    return -1;
  }

  //**/ compute statistics
  dmean = 0.0;
  for (kk = 0; kk < nSamp2; kk++)
    if (vecYY[kk] != PSUADE_UNDEFINED) dmean += vecYY[kk];
  dmean /= (double) count;
  dstds = 0.0;
  for (kk = 0; kk < nSamp2; kk++)
    if (vecYY[kk] != PSUADE_UNDEFINED) 
      dstds += pow(vecYY[kk]-dmean,2.0);
  dstds /= (double) (count - 1);
  dstds = sqrt(dstds);

  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,"RSMAnalysis: Sample mean = %10.3e\n",dmean);
    printOutTS(PL_INFO,"RSMAnalysis: Sample s.d. = %10.3e\n",dstds);
  }

  //**/ if sample std dev = 0, set it to 1 to avoid divide by 0
  if (dstds == 0.0) dstds = 1.0;
  //save mean & std
  outputMean_ = dmean;
  outputStd_  = dstds;
  //cout << outputMean_ << ", " << outputStd_ << endl;

  //**/ ===============================================================
  //**/ use Sobol' method if requested
  //**/ ===============================================================
  if (method_ == 1)
  {
    if (psConfig_.DiagnosticsIsOn())
      printf("Entering RSMSobol1 analyze Sobol\n");

    int nPerBlk = 1000000 / (nInputs + 2);
    int nSobolSams = nPerBlk*(nInputs+2), iOne=1;
    printf("RSMSobol1 Sobol: nSams = %d\n",nSobolSams);

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
    
    //**/ Step 1: Create a Sobol' sample
    SobolSampling *sobolSampler = new SobolSampling();
    sobolSampler->setPrintLevel(-2);
    sobolSampler->setInputBounds(nInputs,xLower,xUpper);
    sobolSampler->setOutputParams(iOne);
    sobolSampler->setSamplingParams(nSobolSams, 1, 0);
    sobolSampler->createM1M2(pdfObj);
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
    if (constrPtr != NULL)
    {
      count = 0;
      tempV = vecSobolX.getDVector();
      for (kk = 0; kk < nSobolSams; kk++)
      {
        ddata = constrPtr->evaluate(&tempV[kk*nInputs],vecSobolY[kk],
                                    status);
        if (status == 0) 
        {
          vecSobolY[kk] = PSUADE_UNDEFINED;
          count++;
        }
      }
      if (count <= 1)
      {
        printf("RSMAnalysis ERROR: Too few samples left after filtering\n");
        if (faPtr != NULL) delete faPtr;
        return -1;
      }
      if (count > nSobolSams/2)
        printf("RSMAnalysis INFO: Constraints filter out > 1/2 points.\n");
    }

    //**/ Step 4: analyze 
    SobolAnalyzer *sobolAnalyzer = new SobolAnalyzer();
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
    //**/ these have been normalized
    VecVces_.setLength(nInputs);
    for (ii = 0; ii < nInputs; ii++) 
      VecVces_[ii] = sobolAnalyzer->get_S(ii);
    delete faPtr;
    delete sobolAnalyzer;

    //**/ Step 5: print results 
    if (psConfig_.InteractiveIsOn()) 
    {
      printAsterisks(PL_INFO, 0);
      for (ii = 0; ii < nInputs; ii++)
      {
        printOutTS(PL_INFO,
          "Sobol' first-order index for input %3d = %10.3e\n",
          ii+1, VecVces_[ii]);
      }
    }
    pData *pObj = NULL;
    if (ioPtr != NULL)
    {
      pObj = ioPtr->getAuxData();
      pObj->nDbles_ = nInputs;
      pObj->dbleArray_ = new double[nInputs];
      //**/ returns unnormalized results
      for (ii = 0; ii < nInputs; ii++)
        pObj->dbleArray_[ii] = VecVces_[ii] * dstds * dstds;
      pObj->dbleData_ = dstds * dstds;
    }
    return 0;
  }

  //**/ ===============================================================
  //**/  use response surface to perform Sobol one parameter test
  //**/ ===============================================================
  //**/ ---------------------------------------------------------
  //**/ allocate space
  //**/ ---------------------------------------------------------
  psVector  vecLower2, vecUpper2, vecZZ, samplePtsND;
  psVector  vecInpMeans2, vecInpStdvs2, vecSamPts1D;
  psIVector vecInpFlags2; 

  vecLower2.setLength(nInputs);
  vecUpper2.setLength(nInputs);
  vecSamPts1D.setLength(nSam1*nInputs);
  samplePtsND.setLength(nSam2*nInputs*nInputs);
  vecInpFlags2.setLength(nInputs-1);
  vecInpMeans2.setLength(nInputs-1);
  vecInpStdvs2.setLength(nInputs-1);

  //**/ ---------------------------------------------------------
  //**/ create sample for each input ==> vecSamPts1D, samplePtsND
  //**/ with sample size = nSam1 and nSam2
  //**/ ---------------------------------------------------------
  if (nSam1 > nSam2) 
  {
    vecSS.setLength(nSam1);
    vecZZ.setLength(nSam1);
  }
  else
  {
    vecSS.setLength(nSam2);
    vecZZ.setLength(nSam2);
  }
  for (ii = 0; ii < nInputs; ii++)
  {
    if (hasPDF == 1)
    {
      //**/ create sample with pdf for input ii+1
      psMatrix corMat1;
      corMat1.setDim(1,1);
      corMat1.setEntry(0, 0, corMatp->getEntry(ii,ii));
      PDFManager *pdfman1 = new PDFManager();
      pdfman1->initialize(1,&pdfFlags[ii],&inputMeans[ii],
                          &inputStdvs[ii],corMat1,NULL,NULL);
      vecLB.load(1, &xLower[ii]);
      vecUB.load(1, &xUpper[ii]);
      vecOut.setLength(nSam1);
      pdfman1->genSample(nSam1, vecOut, vecLB, vecUB);
      for (jj = 0; jj < nSam1; jj++)
        vecSamPts1D[ii*nSam1+jj] = vecOut[jj];
      delete pdfman1;
    }
    else
    {
      sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
      sampler->setInputBounds(1, &xLower[ii], &xUpper[ii]);
      sampler->setOutputParams(1);
      sampler->setSamplingParams(nSam1, 1, 0);
      //psConfig_.SamExpertModeSaveAndReset();
      sampler->initialize(0);
      //psConfig_.SamExpertModeRestore();
      tempV = vecSamPts1D.getDVector();
      sampler->getSamples(nSam1, 1, 1, &(tempV[ii*nSam1]), 
                   vecZZ.getDVector(), vecSS.getIVector());
      delete sampler;
    }

    //**/ create sample with pdf for the other inputs
    if (hasPDF == 1)
    {
      psMatrix corMat2;
      corMat2.setDim(nInputs-1, nInputs-1);
      for (jj = 0; jj < ii; jj++)
      {
        vecLower2[jj] = xLower[jj];
        vecUpper2[jj] = xUpper[jj];
        vecInpFlags2[jj] = pdfFlags[jj];
        vecInpMeans2[jj] = inputMeans[jj];
        vecInpStdvs2[jj] = inputStdvs[jj];
        //**/ correlation matrix is expected to be identity
        for (kk = 0; kk < ii; kk++)
          corMat2.setEntry(jj, kk, corMatp->getEntry(jj,kk));
        for (kk = ii+1; kk < nInputs; kk++)
          corMat2.setEntry(jj, kk-1, corMatp->getEntry(jj,kk));
      }
      for (jj = ii+1; jj < nInputs; jj++)
      {
        vecLower2[jj-1] = xLower[jj];
        vecUpper2[jj-1] = xUpper[jj];
        vecInpFlags2[jj-1] = pdfFlags[jj];
        vecInpMeans2[jj-1] = inputMeans[jj];
        vecInpStdvs2[jj-1] = inputStdvs[jj];
        //**/ correlation matrix is expected to be identity
        for (kk = 0; kk < ii; kk++)
          corMat2.setEntry(jj-1, kk, corMatp->getEntry(jj,kk));
        for (kk = ii+1; kk < nInputs; kk++)
          corMat2.setEntry(jj-1, kk-1, corMatp->getEntry(jj,kk));
      }
      PDFManager *pdfman2 = new PDFManager();
      pdfman2->initialize(nInputs-1,vecInpFlags2.getIVector(),
                    vecInpMeans2.getDVector(),vecInpStdvs2.getDVector(),
                    corMat2,NULL,NULL);
      vecLB.load(nInputs-1, vecLower2.getDVector());
      vecUB.load(nInputs-1, vecUpper2.getDVector());
      vecOut.setLength(nSam2*(nInputs-1));
      pdfman2->genSample(nSam2, vecOut, vecLB, vecUB);
      for (jj = 0; jj < nSam2*(nInputs-1); jj++) 
        samplePtsND[ii*nSam2*nInputs+jj] = vecOut[jj];
      delete pdfman2;
    }
    else
    {
      for (jj = 0; jj < ii; jj++)
      {
        vecLower2[jj] = xLower[jj];
        vecUpper2[jj] = xUpper[jj];
      }
      for (jj = ii+1; jj < nInputs; jj++)
      {
        vecLower2[jj-1] = xLower[jj];
        vecUpper2[jj-1] = xUpper[jj];
      }
      if (nInputs-1 > 51)
           sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
      else sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
      sampler->setInputBounds(nInputs-1, vecLower2.getDVector(), 
                              vecUpper2.getDVector());
      sampler->setOutputParams(1);
      sampler->setSamplingParams(nSam2, 1, 0);
      sampler->initialize(0);
      tempV = samplePtsND.getDVector();
      sampler->getSamples(nSam2, nInputs-1, 1, 
                &(tempV[ii*nSam2*nInputs]), vecZZ.getDVector(), 
                vecSS.getIVector());
      delete sampler;
    }
  }
  //**/ some clean up
  if (ioPtr == NULL) delete corMatp;

  //**/ ---------------------------------------------------------
  //**/ compute first order Sobol indices
  //**/ Note: The following algorithm uses brute force estimator 
  //**/    or double loop (ref: "Different numerical estimators
  //**/    for main effect global sensitivity indices by Sergei
  //**/    Kucherenko and Shufang Song).
  //**/    mathematical formula:
  //**/    VCE(X) = int_{H_1} [int_{H_{d-1}} f(x) d ~X]^2 dX - f0^2
  //**/    where d is the number of inputs
  //**/          H_1 is the space of X
  //**/          ~X  is inputs other than X
  //**/          f0 is the mean
  //**/ ---------------------------------------------------------
  //**/ get ready
  int iL, sCnt, totalCnt, offset;
  psVector  vecEcvs, vecVars, vecMeans,vecmSamPts;
  psIVector vecBins;

  VecVces_.setLength(nInputs);
  vecYY.setLength(nSam2*nSam1);
  vecEcvs.setLength(nInputs);
  vecVars.setLength(nSam1);
  vecBins.setLength(nSam1);
  vecMeans.setLength(nSam1);
  vecmSamPts.setLength(nInputs*nSam2*nSam1);
  printf("INFO: RSMAnalysis (NI) sample sizes (nSam1, nSam2) = %d %d\n",
         nSam1,nSam2);

  //**/ loop through each input
  for (ii = 0; ii < nInputs; ii++)
  {
    if (psConfig_.InteractiveIsOn())
      printOutTS(PL_INFO, "RSMAnalysis: Processing input %d\n",ii+1);

    //**/ use nSam1 levels per input to compute variance of mean
    //**/ first, for each level, create nSam2 ==> mSampletPts
    for (iL = 0; iL < nSam1; iL++)
    {
      offset = iL * nSam2 * nInputs;
      //**/ concatenate the two sample ==> samplePts
      tempV = samplePtsND.getDVector();
      for (jj = 0; jj < nSam2; jj++)
      {
        oneSamplePt = &(tempV[ii*nSam2*nInputs+jj*(nInputs-1)]);
        for (kk = 0; kk < ii; kk++)
          vecmSamPts[offset+jj*nInputs+kk] = oneSamplePt[kk];
        for (kk = ii+1; kk < nInputs; kk++)
          vecmSamPts[offset+jj*nInputs+kk] = oneSamplePt[kk-1];
        vecmSamPts[offset+jj*nInputs+ii] = vecSamPts1D[ii*nSam1+iL];
      }
    }

    //**/ evaluate ==> vecmSamPts ==> vecYY
    if (psConfig_.InteractiveIsOn() && printLevel > 3)
      printOutTS(PL_INFO,
                 "RSMAnalysis: Response surface evaluations\n");
    faPtr->evaluatePoint(nSam2*nSam1,
                   vecmSamPts.getDVector(),vecYY.getDVector());
    if (constrPtr != NULL)
    {
      tempV = vecmSamPts.getDVector();
      for (kk = 0; kk < nSam1*nSam2; kk++)
      {
        ddata = constrPtr->evaluate(&tempV[kk*nInputs],
                                    vecYY[kk],status);
        if (status == 0) vecYY[kk] = PSUADE_UNDEFINED;
      }
    }
   
    if (psConfig_.InteractiveIsOn() && printLevel > 3 &&
       (iL % (nSam1/10) == 0))
      printOutTS(PL_INFO,"RSMAnalysis: Computing mean and std dev\n");

    //**/ compute mean for level iL
    for (iL = 0; iL < nSam1; iL++)
    {
      vecMeans[iL] = 0.0;
      sCnt = 0;
      for (jj = 0; jj < nSam2; jj++)
      {
        if (vecYY[iL*nSam2+jj] != PSUADE_UNDEFINED)
        {
          vecMeans[iL] += vecYY[iL*nSam2+jj];
          sCnt++;
        }
      }
      vecBins[iL] = sCnt;
      if (sCnt < nSam2/10 && printLevel >= 5)
        printOutTS(PL_INFO,
             "RSMAnalysis WARNING: Subsample size = %d\n",sCnt);
      if (sCnt >= 1) vecMeans[iL] /= (double) sCnt;
      else           vecMeans[iL] = PSUADE_UNDEFINED;
      vecVars[iL] = 0.0;
      if (sCnt > 1)
      {
        for (jj = 0; jj < nSam2; jj++)
          if (vecYY[iL*nSam2+jj] != PSUADE_UNDEFINED)
              vecVars[iL] += pow(vecYY[iL*nSam2+jj]-
                vecMeans[iL],2.0);
        vecVars[iL] /= (double) sCnt;
      }
      else vecVars[iL] = PSUADE_UNDEFINED;
    }

    //**/ count number of successes and compute vces
    totalCnt = 0;
    for (iL = 0; iL < nSam1; iL++) totalCnt += vecBins[iL];
    if (totalCnt == 0) 
    {
      printOutTS(PL_ERROR,"RSMAnalysis ERROR: no feasible region.\n");
      exit(1);
    }

    //**/ compute variances for each input
    ddata = 0.0;
    for (iL = 0; iL < nSam1; iL++) 
    {
      if (vecMeans[iL] != PSUADE_UNDEFINED)
        ddata += vecMeans[iL]*vecBins[iL]/totalCnt;
    }
    VecVces_[ii] = 0.0;
    for (iL = 0; iL < nSam1; iL++)
      if (vecMeans[iL] != PSUADE_UNDEFINED)
        VecVces_[ii] += pow(vecMeans[iL]-ddata,2.0) * 
                            vecBins[iL] / totalCnt;
    vecEcvs[ii] = 0.0;
    for (iL = 0; iL < nSam1; iL++)
    {
      if (vecVars[iL] != PSUADE_UNDEFINED)
         vecEcvs[ii] += vecVars[iL]*vecBins[iL]/totalCnt;
    }
  }

  //**/ some clean up
  delete faPtr;
  if (constrPtr != NULL) delete constrPtr;

  //**/ compute max, min and others of the vces
  psVector vecVceU;
  vecVceU.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++)
    vecVceU[ii] = (VecVces_[ii]-vecEcvs[ii]/nSam2);

  //**/ ---------------------------------------------------------------
  //**/ print unsorted indices
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn()) printAsterisks(PL_INFO, 0);
  for (ii = 0; ii < nInputs; ii++)
  {
    if (psConfig_.InteractiveIsOn() && printLevel > 0)
    {
      printOutTS(PL_INFO,
        "Analysis: Unnormalized mean VCE for input %3d = %10.3e\n",
        ii+1, VecVces_[ii]);
    }
  }
  if (psConfig_.InteractiveIsOn() && printLevel >= 2)
  {
    ddata = ddata2 = 0.0;
    for (ii = 0; ii < nInputs; ii++)
    {
      printOutTS(PL_INFO,
           "Unnormalized VCE for input %3d = %10.3e\n",ii+1,
           VecVces_[ii]);
      printOutTS(PL_INFO,
           "Unnormalized VCE for input %3d = %10.3e (unbiased)\n",
           ii+1, VecVces_[ii]);
      ddata  += VecVces_[ii];
      ddata2 += vecVceU[ii];
    }
    printOutTS(PL_INFO,"Sum of   biased VCEs = %10.3e\n",ddata);
    printOutTS(PL_INFO,"Sum of unbiased VCEs = %10.3e\n",ddata2);
    printOutTS(PL_INFO,"Total variance       = %10.3e\n",dstds*dstds);
  }

  //**/ ---------------------------------------------------------------
  //**/ return more detailed data
  //**/ ---------------------------------------------------------------
  pData *pObj = NULL;
  if (ioPtr != NULL)
  {
    pObj = ioPtr->getAuxData();
    {
      pObj->nDbles_ = nInputs;
      pObj->dbleArray_ = new double[nInputs];
      //**/ returns unnormalized results
      for (ii = 0; ii < nInputs; ii++)
        pObj->dbleArray_[ii] = VecVces_[ii];
      pObj->dbleData_ = dstds * dstds;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ print sorted indices
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && printLevel >= 2)
  {
    for (ii = 0; ii < nInputs; ii++) vecMeans[ii] = (double) ii;
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"RSM Sobol' first-order VCE (ordered): \n");
    printDashes(PL_INFO, 0);
    vecVceU = VecVces_;
    sortDbleList2(nInputs,vecVceU.getDVector(),
                  vecMeans.getDVector());
    for (ii = nInputs-1; ii >= 0; ii--)
       printOutTS(PL_INFO,
              "    Unnormalized VCE for input %3d = %10.3e\n",
              (int) vecMeans[ii]+1,VecVces_[ii]);
    printAsterisks(PL_INFO, 0);
  }
  return 0.0;
}

// ************************************************************************
// set internal parameters
// ------------------------------------------------------------------------
int RSMSobol1Analyzer::setParam(int argc, char **argv)
{
  char  *request = (char *) argv[0];
  if      (!strcmp(request,"ana_rssobol1_ni"))    method_ = 0;
  else if (!strcmp(request,"ana_rssobol1_sobol")) method_ = 1;
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
RSMSobol1Analyzer& RSMSobol1Analyzer::operator=(const RSMSobol1Analyzer &)
{
  printOutTS(PL_ERROR, 
       "RSMSobol1 operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int RSMSobol1Analyzer::get_nInputs()
{
  return nInputs_;
}
double RSMSobol1Analyzer::get_outputMean()
{
  return outputMean_;
}
double RSMSobol1Analyzer::get_outputStd()
{
  return outputStd_;
}
double RSMSobol1Analyzer::get_vce(int ind)
{
  if (ind < 0 || ind >= nInputs_)
  {
    printf("RSMSobol1 ERROR: get_vce index error.\n");
    return 0.0;
  }
  if (VecVces_.length() <= ind)
  {
    printf("RSMSobol1 ERROR: get_vce has not value.\n");
    return 0.0;
  }
  return VecVces_[ind];
}

