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
// Functions for the class RSMSobolTSIAnalyzer  
// AUTHOR : CHARLES TONG
// DATE   : 2006
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "PsuadeUtil.h"
#include "sysdef.h"
#include "psMatrix.h"
#include "psVector.h"
#include "Psuade.h"
#include "FuncApprox.h"
#include "Sampling.h"
#include "RSConstraints.h"
#include "PDFManager.h"
#include "PDFNormal.h"
#include "PsuadeData.h"
#include "PsuadeConfig.h"
#include "RSMSobolTSIAnalyzer.h"
#include "sysdef.h"
#include "PrintingTS.h"

#ifdef HAVE_METIS
extern "C" 
{
void METIS_PartGraphRecursive(int *, int *, int *, int *, int *,
                              int *, int *, int *, int *, int *, int *);
}
#endif

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
RSMSobolTSIAnalyzer::RSMSobolTSIAnalyzer() : Analyzer(), nInputs_(0), 
                  outputMean_(0), outputStd_(0)
{
  setName("RSMSOBOLTSI");
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
RSMSobolTSIAnalyzer::~RSMSobolTSIAnalyzer()
{
}

// ************************************************************************
// perform analysis (intended for use in library calls)
// ------------------------------------------------------------------------
void RSMSobolTSIAnalyzer::analyze(int nInps, int nSamp, double *lbs,
                                  double *ubs, double *X, double *Y)
{
  aData adata;
  adata.nInputs_ = nInps;
  adata.nOutputs_ = 1;
  adata.nSamples_ = nSamp;
  adata.iLowerB_ = lbs;
  adata.iUpperB_ = ubs;
  adata.sampleInputs_ = X;
  adata.sampleOutputs_ = Y;
  adata.outputID_ = 0;
  adata.printLevel_ = 0;
  analyze(adata);
  adata.iLowerB_ = NULL;
  adata.iUpperB_ = NULL;
  adata.sampleInputs_ = NULL;
  adata.sampleOutputs_ = NULL;
}

// ************************************************************************
// perform analysis 
// ------------------------------------------------------------------------
double RSMSobolTSIAnalyzer::analyze(aData &adata)
{
  int    ii, jj, kk, iL, sCnt, status, nSubSamples=10000, nLevels=50;
  int    corFlag, noPDF, offset, pdfNull=0, hasSPDF, totalCnt, count;
  double ddata, *dPtr;
  char   pString[500], *cString, winput1[500], winput2[500];
  Sampling      *sampler=NULL;
  FuncApprox    *faPtr=NULL;
  RSConstraints *constrPtr=NULL;
  PDFManager    *pdfman=NULL, *pdfman1, *pdfman2;
  psVector  vecIn, vecOut, vecUB, vecLB, vecY, vecInpMeans, vecInpStdvs;
  psVector  vecSamInps, vecSamOuts;
  psIVector vecSamStas, vecInpPDFs;
  pData     pCorMat, *pPtr;
  psMatrix  *corMatp=NULL, corMat, corMat1, corMat2;

  //**/ ===============================================================
  //**/ display header 
  //**/ ===============================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
         "*          RS-based Total Order Sobol' Indices \n");
    printEquals(PL_INFO, 0); 
    printOutTS(PL_INFO,"* TO GAIN ACCESS TO DIFFERENT OPTIONS: SET \n");
    printOutTS(PL_INFO,"*\n");
    printOutTS(PL_INFO,
         "* - ana_expert mode to finetune internal parameters\n");
    printOutTS(PL_INFO,
         "*   (e.g. adjust sample size for integration).\n");
    printOutTS(PL_INFO,
         "* - rs_expert mode to finetune response surface\n");
    printOutTS(PL_INFO,
         "* - printlevel to 1 to display more information.\n");
    printEquals(PL_INFO, 0);
  }

  //**/ ===============================================================
  //**/ extract test problem data
  //**/ ===============================================================
  int printLevel = adata.printLevel_;
  int nInputs    = nInputs_ = adata.nInputs_;
  int nOutputs   = adata.nOutputs_;
  int nSamples   = adata.nSamples_;
  int outputID   = adata.outputID_;
  int *pdfFlags  = adata.inputPDFs_;
  double *xLower = adata.iLowerB_;
  double *xUpper = adata.iUpperB_;
  double *XIn    = adata.sampleInputs_;
  double *YIn    = adata.sampleOutputs_;
  PsuadeData *ioPtr = adata.ioPtr_;

  //**/ ===============================================================
  //**/ error checking
  //**/ ===============================================================
  if (nInputs <= 0 || nSamples <= 0 || nOutputs <= 0) 
  {
    printOutTS(PL_ERROR, "RSMSobolTSI ERROR: invalid arguments.\n");
    printOutTS(PL_ERROR, "   nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR, "   nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR, "   nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 
  if (nInputs == 1)
  {
    printOutTS(PL_ERROR,
         "RSMSobolTSI: analysis not needed for nInputs = 1.\n");
    printOutTS(PL_ERROR,
         "             (normalized total sensitivity = 1).\n");
    return PSUADE_UNDEFINED;
  }
  if (outputID >= nOutputs || outputID < 0)
  {
    printOutTS(PL_ERROR, "RSMSobolTSI ERROR: invalid output ID (%d).\n", 
               outputID);
    return PSUADE_UNDEFINED;
  }
  status = 0;
  for (ii = 0; ii < nSamples; ii++)
    if (YIn[nOutputs*ii+outputID] > 0.9*PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR,
         "RSMSobolTSI ERROR: Some outputs are undefined. Prune\n");
    printOutTS(PL_ERROR,
         "                   the undefined sample points first.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ===============================================================
  //**/ if no pdf information, create local nopdf version giving
  //**/ vecInpMeans, vecInpStdvs, vecInpPDFs
  //**/ ===============================================================
  if (adata.inputMeans_ == NULL || adata.inputPDFs_ == NULL || 
      adata.inputStdevs_ == NULL)
  {
    pdfNull = 1;
    vecInpPDFs.setLength(nInputs);
    vecInpMeans.setLength(nInputs);
    vecInpStdvs.setLength(nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      vecInpPDFs[ii] = 0;
      vecInpMeans[ii] = 0;
      vecInpStdvs[ii] = 0;
    }
  }
  else
  {
    if (adata.inputPDFs_ != NULL) 
      vecInpPDFs.load(nInputs,adata.inputPDFs_);
    if (adata.inputMeans_ != NULL) 
      vecInpMeans.load(nInputs,adata.inputMeans_);
    if (adata.inputStdevs_ != NULL) 
      vecInpStdvs.load(nInputs,adata.inputStdevs_);
  }

  //**/ ===============================================================
  //**/ check user-provided pdf to see if all uniform (noPDF=1)
  //**/ also check to see if there is any Sample PDFs ('S' type)
  //**/ If so, set the hasSPDF flag
  //**/ ===============================================================
  noPDF = 1;
  hasSPDF = 0;
  if (adata.inputPDFs_ != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (adata.inputPDFs_[ii] != 0) noPDF = 0;
    for (ii = 0; ii < nInputs; ii++)
      if (adata.inputPDFs_[ii] == PSUADE_PDF_SAMPLE) hasSPDF = 1;
  }
  if (hasSPDF == 1) 
  {
    printOutTS(PL_INFO,"RSMSobolTSI INFO: Some inputs have S PDFs.\n");
    printOutTS(PL_INFO,"            Switching to a different method.\n");
    return analyze2(adata);
  }
  if (psConfig_.InteractiveIsOn())
  {
    if (noPDF == 1) 
      printOutTS(PL_INFO,
           "RSMSobolTSI INFO: all uniform distributions.\n");
    else
    {
      printOutTS(PL_INFO,
           "RSMSobolTSI INFO: Non-uniform distributions ");
      printOutTS(PL_INFO,"detected.\n");
    }
  }

  //**/ ===============================================================
  //**/ construct correlation matrix
  //**/ ===============================================================
  if (ioPtr == NULL)
  {
    if (psConfig_.InteractiveIsOn())
    {
      printOutTS(PL_INFO,
        "RSMSobolTSI INFO: no data object (PsuadeData) found.\n");
      printOutTS(PL_INFO,
        "      Several features will be turned off.\n");
      printOutTS(PL_INFO,
        "      E.g. assume no parameter correlation.\n");
    }
    corMatp = new psMatrix();
    corMatp->setDim(nInputs, nInputs);
    for (ii = 0; ii < nInputs; ii++) corMatp->setEntry(ii,ii,1.0e0);
  }
  else
  {
    ioPtr->getParameter("input_cor_matrix", pCorMat);
    corMatp = (psMatrix *) pCorMat.psObject_;
    int hasCor = 0;
    for (ii = 0; ii < nInputs; ii++) 
      for (jj = 0; jj < ii; jj++) 
        if (corMatp->getEntry(ii,jj) != 0) hasCor = 1;
    if (hasCor == 1) 
    {
      printOutTS(PL_INFO,
           "RSMSobolTSI INFO: Some inputs have correlations.\n");
      printOutTS(PL_INFO,
           "            Switching to a different method.\n");
      return analyze2(adata);
    }
  }

  //**/ ===============================================================
  //**/ set up constraint filters, if any
  //**/ ===============================================================
  if (ioPtr != NULL) 
  {
    constrPtr = new RSConstraints();
    constrPtr->genConstraints(ioPtr);
    count = constrPtr->getNumConstraints();
    if (count == 0)
    {
      delete constrPtr;
      constrPtr = NULL;
    }
  }
  else
  {
    constrPtr = NULL;
    if (psConfig_.InteractiveIsOn())
      printf("RSMSobolTSI INFO: no PsuadeData ==> no constraints.\n");
  }

  //**/ ===============================================================
  //**/ build response surface
  //**/ ===============================================================
  if (ioPtr == NULL)
  {
    printf("Select response surface. Options are: \n");
    writeFAInfo(0);
    strcpy(pString, "Choose response surface: ");
    kk = getInt(0, PSUADE_NUM_RS, pString);
    faPtr = genFA(kk, nInputs, 0, nSamples);
  }
  else faPtr = genFAInteractive(ioPtr, 0);

  faPtr->setBounds(xLower, xUpper);
  vecY.setLength(nSamples);
  for (ii = 0; ii < nSamples; ii++) vecY[ii] = YIn[ii*nOutputs+outputID];
  if (psConfig_.InteractiveIsOn())
    printOutTS(PL_INFO,"RSMSobolTSI: Creating a response surface ...\n");
  psConfig_.InteractiveSaveAndReset();
  status = faPtr->initialize(XIn, vecY.getDVector());
  psConfig_.InteractiveRestore();
  if (status != 0)
  {
    printf("RSMSobolTSI ERROR: failed to build response surface.\n");
    printf("       Suggestion: re-run this with rs_expert mode on\n");
    printf("                   to examine what went wrong.\n");
    if (ioPtr == NULL && corMatp != NULL) delete corMatp;
    if (faPtr != NULL) delete faPtr;
    return -1;
  }

  //**/ ===============================================================
  //**/  get internal parameters from users
  //**/ ===============================================================
  if (psConfig_.AnaExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"RSMSobolTSI computes the total sensitivities ");
    printOutTS(PL_INFO,"one input at a time. For\n");
    printOutTS(PL_INFO,"        each input, it first creates a sample ");
    printOutTS(PL_INFO,"of size K (that is, K\n");
    printOutTS(PL_INFO,"        levels). For each level a sample of ");
    printOutTS(PL_INFO,"size M is drawn from all\n");
    printOutTS(PL_INFO,"        other inputs.\n");
    printOutTS(PL_INFO,
       "The total sample size is thus: M * K * nInputs.\n");
    nSubSamples = 5000;
    nLevels = 1000;
    printOutTS(PL_INFO,"nInputs   = %d\n", nInputs);
    printOutTS(PL_INFO,"default M = %d\n", nSubSamples);
    printOutTS(PL_INFO,"default K = %d\n", nLevels);
    printOutTS(PL_INFO,"Please enter your desired M and K below.\n");
    printOutTS(PL_INFO,"NOTE: large M and K may take a long time\n");
    printEquals(PL_INFO, 0);
    snprintf(pString,100,"Enter M (1000 - 50000) : ");
    nSubSamples = getInt(1000,50000,pString);
    snprintf(pString,100,"Enter K (100 - 5000) : ");
    nLevels = getInt(100,5000,pString);
    printAsterisks(PL_INFO, 0);
  }
  else
  {  
    nSubSamples = 5000;
    nLevels = 1000;
    cString = psConfig_.getParameter("RSMSoboltsi_nsubsamples");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nSubSamples);
      if (nSubSamples < 1000)
      {
        printOutTS(PL_INFO,
             "RSMSobolTSI INFO: nSubSamples should be >= 1000.\n");
        nSubSamples = 1000;
      }
      else
      {
        printOutTS(PL_INFO, 
             "RSMSobolTSI INFO: nSubSamples = %d (config).\n",
             nSubSamples);
      }
    }
    cString = psConfig_.getParameter("RSMSoboltsi_nlevels");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nLevels);
      if (nLevels < 100)
      {
        printOutTS(PL_INFO,
             "RSMSobolTSI INFO: nLevels should be >= 100.\n");
        nLevels = 100;
      }
      else
      {
        printOutTS(PL_INFO,
             "RSMSobolTSI INFO: nLevels = %d (config).\n",nLevels);
      }
    }
    if (psConfig_.InteractiveIsOn() && printLevel > 0)
    {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO,"RSMSobolTSI computes the total sensitivities ");
      printOutTS(PL_INFO,"one input at a time. For\n");
      printOutTS(PL_INFO,"        each input, it first creates a sample ");
      printOutTS(PL_INFO,"of size K (that is, K\n");
      printOutTS(PL_INFO,"        levels). For each level a sample of ");
      printOutTS(PL_INFO,"size M is drawn from all\n");
      printOutTS(PL_INFO,"        other inputs.\n");
      printOutTS(PL_INFO,
         "The total sample size is thus: M * K * nInputs.\n");
      printOutTS(PL_INFO,"nInputs   = %d\n", nInputs);
      printOutTS(PL_INFO,"default M = %d\n", nSubSamples);
      printOutTS(PL_INFO,"default K = %d\n", nLevels);
      printOutTS(PL_INFO,
        "To change settings, re-run with ana_expert mode on.\n");
      printAsterisks(PL_INFO, 0);
    }
  }

  //**/ ===============================================================
  //**/  use response surface to compute mean and variance
  //**/ ===============================================================
  //**/ ---------------------------------------------------------------
  //**/ generate a large sample size
  //**/ ---------------------------------------------------------------
  int nSamp = 1000000;
  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,
      "RSMSobolTSI INFO: Creating a sample for basic statistics.\n");
    printOutTS(PL_INFO,"                  Sample size = %d\n", nSamp);
  }

  //**/ ---------------------------------------------------------------
  //**/ allocate space
  //**/ ---------------------------------------------------------------
  vecSamInps.setLength(nSamp*nInputs);
  vecSamOuts.setLength(nSamp);

  //**/ ---------------------------------------------------------------
  //**/ create the sample from input distributions
  //**/ Note: at this point, any S-type PDFs or correlation is not 
  //**/       allowed so the following is adequate
  //**/ ---------------------------------------------------------------
  pdfman = new PDFManager();
  pdfman->initialize(nInputs,vecInpPDFs.getIVector(),
              vecInpMeans.getDVector(),vecInpStdvs.getDVector(),
              *corMatp,NULL,NULL);
  vecLB.load(nInputs, xLower);
  vecUB.load(nInputs, xUpper);
  vecOut.setLength(nSamp*nInputs);
  pdfman->genSample(nSamp, vecOut, vecLB, vecUB);
  for (ii = 0; ii < nSamp*nInputs; ii++) vecSamInps[ii] = vecOut[ii];

  //**/ ---------------------------------------------------------------
  //**/ evaluate the sample using the response surface
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && printLevel > 1)
    printOutTS(PL_INFO,
         "RSMSobolTSI: Response surface evaluation begins...\n");

  faPtr->evaluatePoint(nSamp,vecSamInps.getDVector(),
                       vecSamOuts.getDVector());
  if (psConfig_.InteractiveIsOn() && printLevel > 1)
    printOutTS(PL_INFO,
         "RSMSobolTSI: Response surface evaluation ends...\n");

  //**/ ---------------------------------------------------------------
  //**/ compute sample mean and standard deviation
  //**/ ---------------------------------------------------------------
  double dmean=0.0;
  double *samInpPtr = vecSamInps.getDVector();
  count = 0;
  for (ii = 0; ii < nSamp; ii++) 
  {
    status = 1;
    if (constrPtr != NULL)
      ddata = constrPtr->evaluate(&(samInpPtr[ii*nInputs]),
                                   vecSamOuts[ii],status);
    if (status != 0)
    {
      dmean += vecSamOuts[ii];
      count++;
    }
    else vecSamOuts[ii] = PSUADE_UNDEFINED;
  }
  if (count < 1000)
  {
    printOutTS(PL_ERROR,
         "RSMSobolTSI ERROR: too few samples that satisify the ");
    printOutTS(PL_ERROR, 
         "constraints (%d out of %d).\n", count, nSamp);
    delete faPtr;
    delete pdfman;
    return PSUADE_UNDEFINED;
  }
  dmean /= (double) count;
  double dvar = 0.0;
  for (ii = 0; ii < nSamp; ii++)
    if (vecSamOuts[ii] != PSUADE_UNDEFINED)
      dvar += (vecSamOuts[ii] - dmean) * (vecSamOuts[ii] - dmean) ;
  dvar /= (double) count;
  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,"RSMSobolTSI: sample mean (N=%d) = %11.4e.\n",
               count, dmean);
    printOutTS(PL_INFO,"RSMSobolTSI: sample std  (N=%d) = %11.4e.\n",
               count, sqrt(dvar));
    printOutTS(PL_INFO,"RSMSobolTSI: sample var  (N=%d) = %11.4e.\n",
               count, dvar);
    if (constrPtr != NULL)
      printOutTS(PL_INFO,
         "RSMSobolTSI INFO: %6.2f percent passes the contraints\n",
         (double) count * 100.0 /((double) nSamp));
  }

  //**/ ---------------------------------------------------------------
  //**/ save mean & std
  //**/ ---------------------------------------------------------------
  outputMean_ = dmean;
  outputStd_  = sqrt(dvar);

  //**/ ===============================================================
  //**/  use response surface to perform Sobol one parameter test
  //**/ ===============================================================
  //**/ ---------------------------------------------------------------
  //**/ allocate space
  //**/ ---------------------------------------------------------------
  psVector  vecLower, vecUpper, vecSamPt1D, vecSamPtsND;
  psVector  vecInpMeans1, vecInpStdvs1;
  psIVector vecInpPDFs1;
  vecLower.setLength(nInputs);
  vecUpper.setLength(nInputs);
  vecSamOuts.setLength(nSubSamples);
  vecSamStas.setLength(nSubSamples);
  vecSamPt1D.setLength(nLevels*nInputs);
  vecInpMeans1.setLength(nInputs);
  vecInpStdvs1.setLength(nInputs);
  vecInpPDFs1.setLength(nInputs);
  vecSamPtsND.setLength(nSubSamples*nInputs*nInputs);

  //**/ ---------------------------------------------------------------
  //**/ create sample points ==> samplePts1D, samplePtsND (nSubSamples)
  //**/ ---------------------------------------------------------------
  int allUPDFs=0;
  for (ii = 0; ii < nInputs; ii++)
  {
    if (psConfig_.InteractiveIsOn())
      printOutTS(PL_INFO,
           "RSMSobolTSI: Processing input %d (phase 1)\n",ii+1);

    if (adata.inputPDFs_ != NULL && 
        adata.inputPDFs_[ii] == PSUADE_PDF_SAMPLE)
    {
      printOutTS(PL_INFO,
           "RSMSobolTSI INFO: skip input %d (S PDF)\n",ii+1);
      continue;
    }

    //**/ see if there is any correlation between this input and others
    corFlag = 0;
    for (jj = 0; jj < nInputs; jj++)
      if (ii != jj && corMatp->getEntry(ii,jj) != 0.0) corFlag = 1;

    //**/ if there is no correlation between ii and the other inputs,
    //**/ but not all are uniform PDFs (noPDF == 0), create 2 
    //**/ samples based on pdfs, and, there is no need to transform 
    //**/ later since uncorrelation does not affect sample combination 
    if (corFlag == 0 && noPDF == 0)
    {
      if (psConfig_.InteractiveIsOn())
      {
        printOutTS(PL_DETAIL,
           "RSMSobolTSI: Creating sample (PDFs not uniform but ");
        printOutTS(PL_DETAIL, "no cross correlation)\n");
      }
      corMat1.setDim(nInputs-1, nInputs-1);
      //**/ copy pdf information except the i-th input
      for (jj = 0; jj < ii; jj++)
      {
        vecLower[jj] = xLower[jj];
        vecUpper[jj] = xUpper[jj];
        vecInpPDFs1[jj] = vecInpPDFs[jj];
        vecInpMeans1[jj] = vecInpMeans[jj];
        vecInpStdvs1[jj] = vecInpStdvs[jj];
        for (kk = 0; kk < ii; kk++)
          corMat1.setEntry(jj, kk, corMatp->getEntry(jj,kk));
        for (kk = ii+1; kk < nInputs; kk++)
          corMat1.setEntry(jj, kk-1, corMatp->getEntry(jj,kk));
      }
      for (jj = ii+1; jj < nInputs; jj++)
      {
        vecLower[jj-1] = xLower[jj];
        vecUpper[jj-1] = xUpper[jj];
        vecInpPDFs1[jj-1] = vecInpPDFs[jj];
        vecInpMeans1[jj-1] = vecInpMeans[jj];
        vecInpStdvs1[jj-1] = vecInpStdvs[jj];
        for (kk = 0; kk < ii; kk++)
          corMat1.setEntry(jj-1, kk, corMatp->getEntry(jj,kk));
        for (kk = ii+1; kk < nInputs; kk++)
          corMat1.setEntry(jj-1, kk-1, corMatp->getEntry(jj,kk));
      }
      //**/ create two samples: vecSamPtsND and vecSamPt1D
      pdfman1 = new PDFManager();
      pdfman1->initialize(nInputs-1,vecInpPDFs1.getIVector(),
                   vecInpMeans1.getDVector(),vecInpStdvs1.getDVector(),
                   corMat1,NULL,NULL);
      vecLB.load(nInputs-1, vecLower.getDVector());
      vecUB.load(nInputs-1, vecUpper.getDVector());
      vecOut.setLength(nSubSamples*(nInputs-1));
      pdfman1->genSample(nSubSamples, vecOut, vecLB, vecUB);
      for (jj = 0; jj < nSubSamples*(nInputs-1); jj++)
        vecSamPtsND[nSubSamples*ii*nInputs+jj] = vecOut[jj];
      delete pdfman1;
      corMat2.setDim(1, 1);
      corMat2.setEntry(0, 0, corMatp->getEntry(0,0));
      pdfman2 = new PDFManager();
      pdfman2->initialize(1, &(adata.inputPDFs_[ii]), 
                  &(adata.inputMeans_[ii]),&(adata.inputStdevs_[ii]),
                  corMat2,NULL,NULL);
      vecLB.load(1, &xLower[ii]);
      vecUB.load(1, &xUpper[ii]);
      vecOut.setLength(nLevels);
      pdfman2->genSample(nLevels, vecOut, vecLB, vecUB);
      for (iL = 0; iL < nLevels; iL++)
        vecSamPt1D[ii*nLevels+iL] = vecOut[iL];
      delete pdfman2;
    }
    //**/ if correlated (corFlag == 1), create 2 separate space 
    //**/ filling sample (based on uniform distribution), and it
    //**/ needs to transform later with different combinations
    //**/ If, however, AllUPDFs = 1 (which implies corFlag must
    //**/ be 0, since uniform distributions do not have correlation
    else
    {
      if (psConfig_.InteractiveIsOn() && corFlag == 1)
        printOutTS(PL_DETAIL,
          "RSMSobolTSI: Creating (correlation: %d and the rest)\n",
          ii+1);
      if (psConfig_.InteractiveIsOn() && allUPDFs == 1)
        printOutTS(PL_DETAIL,
          "RSMSobolTSI: Creating sample (All uniform PDFs)\n");
      if (nInputs > 51)
           sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
      else sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
      for (jj = 0; jj < ii; jj++)
      {
        vecLower[jj] = xLower[jj];
        vecUpper[jj] = xUpper[jj];
      }
      for (jj = ii+1; jj < nInputs; jj++)
      {
        vecLower[jj-1] = xLower[jj];
        vecUpper[jj-1] = xUpper[jj];
      }
      sampler->setInputBounds(nInputs-1, vecLower.getDVector(), 
                              vecUpper.getDVector());
      sampler->setOutputParams(1);
      sampler->setSamplingParams(nSubSamples, 1, 0);
      sampler->initialize(0);
      dPtr = vecSamPtsND.getDVector();
      sampler->getSamples(nSubSamples, nInputs-1, 1, 
                     &(dPtr[ii*nSubSamples*nInputs]),
                     vecSamOuts.getDVector(), vecSamStas.getIVector());
      delete sampler;
      for (iL = 0; iL < nLevels; iL++)
        vecSamPt1D[iL+ii*nLevels] = (xUpper[ii]-xLower[ii])/(nLevels-1) * 
                                    iL + xLower[ii];
    }
  }

  //**/ ===============================================================
  //**/ now the samples are in vecSamPtsND (and vecSamPt1D for some)
  //**/ use response surface to perform Sobol test
  //**/ ===============================================================
  //**/ ---------------------------------------------------------------
  //**/ allocate space
  //**/ ---------------------------------------------------------------
  psVector  vecMeans, vecVars, vecTSI, vecTSIRes;
  psVector  vecmSamPts, vecYZ, vecYFuzzy;
  psIVector vecBins;
  PDFNormal *rsPDF;
  vecYZ.setLength(nLevels);
  vecYFuzzy.setLength(nLevels*nSubSamples);
  vecmSamPts.setLength(nSubSamples*nLevels*nInputs);
  vecMeans.setLength(nSubSamples);
  vecBins.setLength(nSubSamples);
  vecVars.setLength(nSubSamples);
  vecTSI.setLength(nInputs);
  vecTSIRes.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++)
  {
    if (psConfig_.InteractiveIsOn())
      printOutTS(PL_INFO,"RSMSobolTSI: Processing input %d (phase 2)\n", 
                 ii+1);
    //**/ for each sample point (nInputs-1), sample nLevels of input ii
    if (psConfig_.InteractiveIsOn())
      printOutTS(PL_DETAIL,"             Preparing sample\n"); 

    //**/ see if there is any correlation between this input and others
    corFlag = 0;
    for (jj = 0; jj < nInputs; jj++)
      if (ii != jj && corMatp->getEntry(ii,jj) != 0.0) corFlag = 1;

    //**/ process the sample 
    for (jj = 0; jj < nSubSamples; jj++)
    {
      offset = jj * nLevels * nInputs;
      //**/ extract sample points 
      for (iL = 0; iL < nLevels; iL++)
      {
        for (kk = 0; kk < ii; kk++)
          vecmSamPts[offset+iL*nInputs+kk] = 
              vecSamPtsND[ii*nSubSamples*nInputs+jj*(nInputs-1)+kk];
        for (kk = ii+1; kk < nInputs; kk++)
          vecmSamPts[offset+iL*nInputs+kk] = 
              vecSamPtsND[ii*nSubSamples*nInputs+jj*(nInputs-1)+kk-1];
        vecmSamPts[offset+iL*nInputs+ii] = vecSamPt1D[iL+ii*nLevels];
      } 

      //**/ if correlated, transform the sample 
      if (corFlag == 1)
      {
        dPtr = vecmSamPts.getDVector();
        vecIn.load(nLevels*nInputs, &(dPtr[offset]));
        vecOut.setLength(nLevels*nInputs);
        pdfman->invCDF(nLevels, vecIn, vecOut, vecLB, vecUB);
        for (kk = 0; kk < nLevels*nInputs; kk++) 
          vecmSamPts[offset+kk] = vecOut[kk];
      }
    }

    //**/ evaluate
    if (psConfig_.InteractiveIsOn())
      printOutTS(PL_DETAIL,"             Evaluating sample\n"); 
    faPtr->evaluatePoint(nLevels*nSubSamples,vecmSamPts.getDVector(),
                         vecYFuzzy.getDVector());

    if (psConfig_.InteractiveIsOn())
      printOutTS(PL_DETAIL,"             Analyzing sample\n"); 
    for (jj = 0; jj < nSubSamples; jj++)
    {
      offset = jj * nLevels;
      dPtr   = vecYZ.getDVector();
      for (iL = 0; iL < nLevels; iL++)
        vecYZ[iL] = vecYFuzzy[offset+iL];

      //**/ filter the sample points 
      dPtr = vecmSamPts.getDVector();
      for (iL = 0; iL < nLevels; iL++)
      {
        dPtr = &(dPtr[jj*nLevels*nInputs+iL*nInputs]);
        status = 1;
        if (constrPtr != NULL)
          ddata = constrPtr->evaluate(dPtr,vecYZ[iL],status);
        //**/ if it fails any filter, set the output to be undefined
        if (status == 0) vecYZ[iL] = PSUADE_UNDEFINED;
      }

      //**/ compute mean for level iL
      vecMeans[jj] = 0.0;
      vecVars[jj] = 0.0;
      sCnt = 0;
      for (iL = 0; iL < nLevels; iL++)
      {
        if (vecYZ[iL] != PSUADE_UNDEFINED)
        {
          vecMeans[jj] += vecYZ[iL];
          sCnt++;
        }
      }
      vecBins[jj] = sCnt;
      if (sCnt >= 1) vecMeans[jj] /= (double) sCnt;
      else           vecMeans[jj] = PSUADE_UNDEFINED;
      if (vecMeans[jj] == PSUADE_UNDEFINED)
        vecVars[jj] = PSUADE_UNDEFINED;
      else
      {
        for (iL = 0; iL < nLevels; iL++)
        {
          if (vecYZ[iL] != PSUADE_UNDEFINED)
            vecVars[jj] += 
               pow(vecYZ[iL]-vecMeans[jj],2.0);
        }
        if (sCnt == 1) vecVars[jj] = 0.0;
        else           vecVars[jj] /= (double) sCnt;
      }
    }

    //**/ now means and vars are ready for all nSubSamples
    //**/ compute mean of the variance of the LHS samples
    totalCnt = 0;
    for (jj = 0; jj < nSubSamples; jj++) 
      if (vecVars[jj] != PSUADE_UNDEFINED) totalCnt++;
    if (totalCnt == 0)
    {
      printf("RSMSobolTSI ERROR: unable to compute TSI for input %d\n",
             ii+1);
      printf("       Possible reason: constraints too tight.\n"); 
      printf("       Suggestion: check/loosen the constraints.\n"); 
      exit(1);
    }
    //**/ compute mean of variance at levels in ~inputs
    dvar = 0.0;
    for (jj = 0; jj < nSubSamples; jj++)
    {
      if (vecVars[jj] != PSUADE_UNDEFINED) dvar += vecVars[jj];
    }
    vecTSI[ii] = dvar / (double) totalCnt;
    if (psConfig_.InteractiveIsOn() && printLevel > 3)
      printf("Conditional expectation of variance of ~inputs %d = %10.3e\n",
             ii+1, vecTSI[ii]);

    //**/ compute variance of the means at levels in ~inputs
    dmean = 0.0;
    for (jj = 0; jj < nSubSamples; jj++)
    {
      if (vecMeans[jj] != PSUADE_UNDEFINED)
        dmean += vecMeans[jj];
    }
    dmean /= (double) totalCnt;
    dvar = 0.0;
    for (jj = 0; jj < nSubSamples; jj++)
      if (vecMeans[jj] != PSUADE_UNDEFINED)
        dvar += pow(vecMeans[jj]-dmean,2.0);
    vecTSIRes[ii] = dvar / (double) totalCnt;
    if (psConfig_.InteractiveIsOn() && printLevel > 3)
      printf("Variance of conditional expectation of ~inputs %d = %10.3e\n",
             ii+1, vecTSIRes[ii]);
  }

  //**/ ---------------------------------------------------------------
  //**/ compute statistics (E_{~X}(V(X|~X)))/variance
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn()) 
  {
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < nInputs; ii++)
      printOutTS(PL_INFO,
        "Total sensitivity (normalized) for input %3d = %10.3e\n",
        ii+1, 1.0-vecTSIRes[ii]/(outputStd_*outputStd_));
  }

  //**/ ---------------------------------------------------------------
  //**/ return more detailed data
  //**/ ---------------------------------------------------------------
  if (ioPtr != NULL)
  {
    pPtr = ioPtr->getAuxData();
    pPtr->nDbles_ = nInputs;
    pPtr->dbleArray_ = new double[nInputs];
    for (ii = 0; ii < nInputs; ii++) 
      pPtr->dbleArray_[ii] = outputStd_*outputStd_ - vecTSIRes[ii];
    pPtr->dbleData_ = outputStd_ * outputStd_;
  }
  if (psConfig_.InteractiveIsOn()) printAsterisks(PL_INFO, 0);
    
  //**/ ===============================================================
  //**/ return more detailed data
  //**/ ===============================================================
  vecTSIs_.setLength(nInputs_);
  for (ii = 0; ii < nInputs; ii++) 
     vecTSIs_[ii] = outputStd_*outputStd_ - vecTSIRes[ii];

  if (psConfig_.InteractiveIsOn() && printLevel > 1)
  {
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < nInputs; ii++)
      printOutTS(PL_INFO, 
        "Total sensitivity (unnormalized) for input %3d = %10.3e\n",
        ii+1, outputStd_*outputStd_-vecTSIRes[ii]);
    //for (ii = 0; ii < nInputs; ii++) vecMeans[ii] = (double) ii;
    //sortDbleList2(nInputs,vecTSI.getDVector(),vecMeans.getDVector());
    //for (ii = nInputs-1; ii >= 0; ii--)
    //  printOutTS(PL_INFO, 
    //    "Total sensitivity (unnormalized,ordered) for input %3d = %12.4e\n",
    //       (int) vecMeans[ii]+1,vecTSI[ii]);
    printAsterisks(PL_INFO, 0);
  }

  //**/ ----------------------------------------------------------------
  //**/ print results
  //**/ ----------------------------------------------------------------
  printResults(nInputs,outputStd_*outputStd_,vecTSIs_.getDVector(),
               ioPtr);

  //**/ ===============================================================
  //**/ clean up
  //**/ ===============================================================
  if (constrPtr != NULL) delete constrPtr;
  if (pdfman != NULL) delete pdfman;
  if (faPtr != NULL) delete faPtr;
  if (ioPtr == NULL && corMatp != NULL) delete corMatp;
  return 0.0;
}

// ************************************************************************
// perform analysis for correlated inputs and S-type PDFs
// ------------------------------------------------------------------------
double RSMSobolTSIAnalyzer::analyze2(aData &adata)
{
  int    ss, ii, jj, inputID, nnz, itmp, jtmp, n1d, nSubdomains, nFilled;
  int    count, graphN, index, totalCnt;
  double dvar, dtmp;
  char   pString[500];
  psVector  vecYT, vecInpMeans, vecInpStdvs;
  psIVector vecInpPDFs;

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (adata.ioPtr_ == NULL)
  {
    printOutTS(PL_ERROR, "RSMSobolTSI ERROR: no data.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ---------------------------------------------------------------
  //**/ extract data
  //**/ ---------------------------------------------------------------
  int printLevel = adata.printLevel_;
  int nInputs  = adata.nInputs_;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;
  double *XIn  = adata.sampleInputs_;
  double *YIn  = adata.sampleOutputs_;
  int outputID = adata.outputID_;
  double *lbounds = adata.iLowerB_;
  double *ubounds = adata.iUpperB_;
  PsuadeData *ioPtr = adata.ioPtr_;

  //**/ ===============================================================
  //**/ construct statistical data
  //**/ ===============================================================
  if (adata.inputPDFs_ != NULL) 
    vecInpPDFs.load(nInputs,adata.inputPDFs_);
  if (adata.inputMeans_ != NULL) 
    vecInpMeans.load(nInputs,adata.inputMeans_);
  if (adata.inputStdevs_ != NULL) 
    vecInpStdvs.load(nInputs,adata.inputStdevs_);
  pData pCorMat;
  ioPtr->getParameter("input_cor_matrix", pCorMat);
  psMatrix *corMatp = (psMatrix *) pCorMat.psObject_;

  //**/ ===============================================================
  //**/ set up constraint filters, if any
  //**/ ===============================================================
  RSConstraints *constrPtr=NULL;
  constrPtr = new RSConstraints();
  constrPtr->genConstraints(ioPtr);
  count = constrPtr->getNumConstraints();
  if (count == 0)
  {
    delete constrPtr;
    constrPtr = NULL;
  }

  //**/ ===============================================================
  //**/ build response surface
  //**/ ===============================================================
  FuncApprox *faPtr = genFAInteractive(ioPtr, 0);
  faPtr->setBounds(lbounds, ubounds);
  vecYT.setLength(nSamples);
  for (ss = 0; ss < nSamples; ss++) 
    vecYT[ss] = YIn[ss*nOutputs+outputID];
  if (psConfig_.InteractiveIsOn())
    printOutTS(PL_INFO,"RSMSobolTSI: Creating a response surface ...\n");
  psConfig_.InteractiveSaveAndReset();
  int status = faPtr->initialize(XIn, vecYT.getDVector());
  psConfig_.InteractiveRestore();
  if (status != 0)
  {
    printf("RSMSobolTSI ERROR: failed to build response surface.\n");
    if (faPtr != NULL) delete faPtr;
    return -1;
  }

  //**/ ---------------------------------------------------------------
  //**/ create a large sample
  //**/ ---------------------------------------------------------------
  int nSamp = 5000000;
  psVector vecSamInps, vecSamOuts, vecLB, vecUB;
  vecSamInps.setLength(nSamp*nInputs);
  vecSamOuts.setLength(nSamp);
  PDFManager *pdfman;
  pdfman = new PDFManager();
  pdfman->initialize(ioPtr);
  vecLB.load(nInputs, lbounds);
  vecUB.load(nInputs, ubounds);
  pdfman->genSample(nSamp, vecSamInps, vecLB, vecUB);
  delete pdfman;

  //**/ ---------------------------------------------------------------
  //**/ evaluate large sample with response surface
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && printLevel > 1)
    printOutTS(PL_INFO,
         "RSMSobolTSI: Response surface evaluation begins...\n");
  faPtr->evaluatePoint(nSamp,vecSamInps.getDVector(),
                       vecSamOuts.getDVector());
  if (psConfig_.InteractiveIsOn() && printLevel > 1)
    printOutTS(PL_INFO,
         "RSMSobolTSI: Response surface evaluation ends...\n");
  delete faPtr;

  //**/ ---------------------------------------------------------------
  //**/ filter and compress
  //**/ ---------------------------------------------------------------
  double ddata, *samInpPtr = vecSamInps.getDVector();
  count = 0;
  for (ii = 0; ii < nSamp; ii++)
  {
    status = 1;
    if (constrPtr != NULL)
      ddata = constrPtr->evaluate(&(samInpPtr[ii*nInputs]),
                                   vecSamOuts[ii],status);
    if (status == 1)
    {
      for (jj = 0; jj < nInputs; jj++)
        vecSamInps[count*nInputs+jj] = vecSamInps[ii*nInputs+jj];
      vecSamOuts[count] = vecSamOuts[ii];
      count++;
    }
  }
  if (count < 10000)
  {
    printOutTS(PL_ERROR,
         "RSMSobolTSI ERROR: Too few samples that satisify the ");
    printOutTS(PL_ERROR,
         "constraints\n");
    printOutTS(PL_ERROR,
         "                   (%d out of %d).\n", count, nSamp);
    delete constrPtr;
    return PSUADE_UNDEFINED;
  }
  nSamp = count;

  //**/ ---------------------------------------------------------------
  //**/ more checking
  //**/ ---------------------------------------------------------------
  psVector vecRanges;
  vecRanges.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++)
  {
    vecRanges[ii] = ubounds[ii] - lbounds[ii];
    if (vecRanges[ii] <= 0.0)
    {
      printOutTS(PL_ERROR,
            "Total Effect ERROR: lbound/ubound mismatch.\n");
      exit(1);
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ compute mean and variance
  //**/ ---------------------------------------------------------------
  double dmean = 0.0;
  for (ss = 0; ss < nSamp; ss++) dmean += vecSamOuts[ss];
  dmean /= (double) nSamp;
  double variance = 0.0;
  for (ss = 0; ss < nSamp; ss++)
    variance += ((vecSamOuts[ss] - dmean) * (vecSamOuts[ss] - dmean));
  variance /= (double) (nSamp - 1);
  printOutTS(PL_INFO,"RSMSobolTSI: output mean     = %10.3e\n",dmean);
  printOutTS(PL_INFO,"RSMSobolTSI: output variance = %10.3e\n",
             variance);
  outputMean_ = dmean;
  outputStd_  = sqrt(variance);

  //**/ ---------------------------------------------------------------
  //**/ set up mesh
  //**/ ---------------------------------------------------------------
  if (nInputs > 21)
  {
    printOutTS(PL_ERROR,
       "RSMSobolTSI ERROR: nInputs > 21 currently not supported.\n");
    exit(1);
  }
  if (nInputs == 21) n1d =  2;
  if (nInputs == 20) n1d =  2;
  if (nInputs == 19) n1d =  2;
  if (nInputs == 18) n1d =  2;
  if (nInputs == 17) n1d =  2;
  if (nInputs == 16) n1d =  2;
  if (nInputs == 15) n1d =  2;
  if (nInputs == 14) n1d =  3;
  if (nInputs == 13) n1d =  3;
  if (nInputs == 12) n1d =  4;
  if (nInputs == 11) n1d =  5;
  if (nInputs == 10) n1d =  6;
  if (nInputs ==  9) n1d =  7;
  if (nInputs ==  8) n1d = 10;
  if (nInputs ==  7) n1d = 14;
  if (nInputs ==  6) n1d = 24;
  if (nInputs ==  5) n1d = 48;
  if (nInputs ==  4) n1d = 200;
  if (nInputs ==  3) n1d = 3000;
  if (nInputs ==  2) n1d = 100000;

  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"*          Crude Total Sensitivity Indices\n");
  printEquals(PL_INFO,0);
  nSubdomains = (int) (nSamp / 100.0);
  if (nSubdomains > 10000) nSubdomains = 10000;
  printOutTS(PL_INFO,
     "* RSMSobolTSI: number of subdomains          = %d\n",nSubdomains);
  printOutTS(PL_INFO,
     "* RSMSobolTSI: avg sample size per subdomain = %d\n",
     (int) (nSamp / nSubdomains));
  printDashes(PL_INFO,0);
  printOutTS(PL_INFO,
     "* NOTE: for small to moderate sample size, this method in general\n");
  printOutTS(PL_INFO,
     "*       gives rough estimates of total sensitivity.\n");
  printOutTS(PL_INFO,
     "* Recommendation: Try different numbers of subdomains to assess\n");
  printOutTS(PL_INFO,
     "*   the goodness of the measures. A rule of thumb for sample size\n");
  printOutTS(PL_INFO,
     "*   per subdomain is > 50.\n");
  printOutTS(PL_INFO,
     "* Turn on analysis expert mode to modify default settings.\n");
  printEquals(PL_INFO,0);
  if (psConfig_.AnaExpertModeIsOn())
  {
    strcpy(pString,"Enter the number of subdomains (> 5): ");
    nSubdomains = getInt(5,nSamp, pString);
  }

  //**/ ---------------------------------------------------------------
  //**/ generate a graph from a (nInputs-1)-dimensional mesh 
  //**/ ---------------------------------------------------------------
  psIVector vecIncrs;
  vecIncrs.setLength(nInputs);
  graphN = 1;
  vecIncrs[0] = graphN;
  for (jj = 1; jj < nInputs; jj++)
  {
    graphN *= n1d;
    vecIncrs[jj] = graphN;
  }

  psIVector vecIA, vecJA;
  vecIA.setLength(graphN+1);
  vecJA.setLength(graphN*(nInputs-1)*2+1);
  nnz = 0;
  vecIA[0] = nnz;
  for (ii = 0; ii < graphN; ii++)
  {
    itmp = ii;
    for (jj = 0; jj < nInputs-1; jj++)
    {
      jtmp = itmp % n1d;
      itmp = itmp / n1d;
      if (jtmp > 0    ) vecJA[nnz++] = ii - vecIncrs[jj];
      if (jtmp < n1d-1) vecJA[nnz++] = ii + vecIncrs[jj];
    }
    vecIA[ii+1] = nnz;
  }

  //**/ ----------------------------------------------------------------
  //**/ call metis to perform partitioning
  //**/ ----------------------------------------------------------------
  int wgtflag=0, numflag=0, edgeCut=0, options[10];
  psIVector vecCellsOccupied;
  vecCellsOccupied.setLength(graphN);
  options[0] = 0;
#ifdef HAVE_METIS
  if (psConfig_.InteractiveIsOn() && printLevel > 1)
    printf("RSMSobolTSI: partitioning begins ...\n");
  METIS_PartGraphRecursive(&graphN,vecIA.getIVector(),vecJA.getIVector(), 
        NULL,NULL,&wgtflag,&numflag,&nSubdomains,options,&edgeCut,
        vecCellsOccupied.getIVector());
  if (psConfig_.InteractiveIsOn() && printLevel > 1)
    printf("RSMSobolTSI: partitioning ends.\n");
#else
  printOutTS(PL_ERROR, "RSMSobolTSI ERROR : METIS not installed.\n");
  nInputs = 0;
#endif

  //**/ ----------------------------------------------------------------
  //**/ allocate temporary storage
  //**/ ----------------------------------------------------------------
  psIVector vecSam2Aggr, vecAggrCnts;
  psVector  vecAggrMeans, vecTSI;
  vecSam2Aggr.setLength(nSamp);
  vecAggrCnts.setLength(nSubdomains);
  vecAggrMeans.setLength(nSubdomains);
  vecTSI.setLength(nInputs);

  //**/ ----------------------------------------------------------------
  //**/ For each input i, compute 1 - V(E(X_i|X_{~i}))
  //**/ ----------------------------------------------------------------
  for (inputID = 0; inputID < nInputs; inputID++)
  {
    //**/ ----------------------------------------------------------------
    //**/ locate which aggregate each sample point belongs to
    //**/ ----------------------------------------------------------------
    for (ss = 0; ss < nSamp; ss++)
    {
      itmp = 0;
      for (ii = 0; ii < nInputs; ii++)
      {
        if (ii != inputID)
        {
          itmp = itmp * n1d;
          dtmp = vecSamInps[ss*nInputs+ii];
          dtmp = (dtmp - lbounds[ii]) / vecRanges[ii];
          jtmp = (int) (dtmp * n1d);
          if (jtmp < 0) jtmp = 0;
          if (jtmp >= n1d) jtmp = n1d - 1;
          itmp += jtmp;
        }
      }
      if (itmp >= graphN)
      {
        printf("FATAL ERROR: mesh cell %d >= %d\n",itmp,graphN);
        printf("      offending sample = %d\n", ss+1);
        printf("==> Consult PSUADE developers.\n");
        exit(1);
      }
      if (vecCellsOccupied[itmp] < 0 || vecCellsOccupied[itmp] >= nSubdomains)
      {
        printf("FATAL ERROR: mesh cell %d - assigned aggregate %d\n",
               itmp,vecCellsOccupied[itmp]);
        printf("      number number of mesh cells = %d\n", graphN);
        printf("      offending sample = %d\n", ss+1);
        printf("==> Consult PSUADE developers.\n");
        exit(1);
      }
      vecSam2Aggr[ss] = vecCellsOccupied[itmp];
    }

    //**/ ----------------------------------------------------------------
    //**/ processing
    //**/ ----------------------------------------------------------------
    for (ii = 0; ii < nSubdomains; ii++)
    {
      vecAggrMeans[ii] = 0.0;
      vecAggrCnts[ii] = 0;
    }
    for (ss = 0; ss < nSamp; ss++)
    {
      index = vecSam2Aggr[ss];
      if (index < 0 || index >= nSubdomains)
      {
        printf("FATAL ERROR: index = %d ([0,%d])\n",index,nSubdomains);
        exit(1);
      }
      vecAggrMeans[index] += vecSamOuts[ss];
      vecAggrCnts[index]++;
    }
    totalCnt = 0;
    int emptyCnt = 0;
    for (ii = 0; ii < nSubdomains; ii++)
    {
      totalCnt += vecAggrCnts[ii];
      if (vecAggrCnts[ii] == 0) emptyCnt++;
    }
    if (totalCnt != nSamp)
    {
      printf("RSMSobolTSI ERROR: totalCnt %d != nSamples %d\n",
             totalCnt,nSamp);
      exit(1);
    }
    if (emptyCnt > 0 && psConfig_.InteractiveIsOn() && printLevel > 1)
    {
      printOutTS(PL_WARN,
        "RSMSobolTSI INFO: when computing TSI for input %d, %d bins out of\n",
        inputID+1, emptyCnt);
      printOutTS(PL_WARN, 
        "            %d are empty. This may be due to unconventional input\n",
        nSubdomains);
      printOutTS(PL_WARN,
        "            distributions so that sample points are not distributed\n");
      printOutTS(PL_WARN,
        "            evenly over the parameter space.\n");
    }
    nFilled = 0;
    for (ii = 0; ii < nSubdomains; ii++)
    {
      if (vecAggrCnts[ii] > 0) 
      {
        vecAggrMeans[ii] /= (double) vecAggrCnts[ii];
        nFilled++;
      }
    }
    if (psConfig_.InteractiveIsOn() && printLevel > 1)
      printf("(RSTSI INFO) Input %4d : %d out of %d subdomains populated.\n",
             inputID+1, nFilled, nSubdomains);
    dmean = 0.0;
    for (ii = 0; ii < nSubdomains; ii++) 
      dmean += vecAggrMeans[ii] * vecAggrCnts[ii];
    dmean /= (double) nSamp;
    dvar = 0.0;
    for (ii = 0; ii < nSubdomains; ii++)
      if (vecAggrCnts[ii] > 0) 
        dvar += pow(vecAggrMeans[ii]-dmean,2.0)*vecAggrCnts[ii];
    dvar /= (double) (nSamp- 1.0);

    if (dvar > variance)
    {
      printOutTS(PL_INFO,
        "Input %4d: Approximate total sensitivity index %e > variance %e?\n",
        inputID+1, dvar, variance);
      printOutTS(PL_INFO,"            Is your sample evenly distributed?\n");
      printOutTS(PL_INFO,
        "            Do you have too many subdomains (too few in each)?\n");
      for (ii = 0; ii < nSubdomains; ii++)
        printf("Aggregate mean %d = %e (dmean=%e, count=%d)\n",ii+1,
               vecAggrMeans[ii],dmean,vecAggrCnts[ii]);
    }
    vecTSI[inputID] = variance - dvar;
  }

  //**/ ----------------------------------------------------------------
  //**/ print results
  //**/ ----------------------------------------------------------------
  printResults(nInputs, variance, vecTSI.getDVector(), ioPtr);
  return 0;
}

// ************************************************************************
// create matlab file
// ------------------------------------------------------------------------
int RSMSobolTSIAnalyzer::printResults(int nInputs, double variance,
                                      double *tsi, PsuadeData *ioPtr)
{
  int   ii;
  char  pString[1000];
  FILE  *fp;
  pData qData;

  printEquals(PL_INFO, 0);
  if (variance == 0.0)
  {
    printOutTS(PL_INFO,
       "Total variance = 0. Hence, no total effect plot.\n");
    return 0;
  }
  printOutTS(PL_INFO, "Approximate Total Effect Statistics: \n");
  for (ii = 0; ii < nInputs; ii++)
    printOutTS(PL_INFO,
      "Input %2d: Sobol' total sensitivity = %8.2e (normalized = %8.2e)\n",
      ii+1,tsi[ii],tsi[ii]/variance);
  if (plotScilab()) fp = fopen("scilabrssoboltsi.sci", "w");
  else              fp = fopen("matlabrssoboltsi.m", "w");

  if (ioPtr != NULL) ioPtr->getParameter("input_names", qData);
  if (fp != NULL)
  {
    strcpy(pString, "This file contains Sobol' total indices");
    fwriteComment(fp, pString);
    strcpy(pString, "set sortFlag = 1 and set nn to be the number");
    fwriteComment(fp, pString);
    strcpy(pString, "of inputs to display.");
    fwriteComment(fp, pString);
    fprintf(fp, "sortFlag = 0;\n");
    fprintf(fp, "nn = %d;\n", nInputs);
    fprintf(fp, "Mids = [\n");
    for (ii = 0; ii < nInputs; ii++) 
      fprintf(fp,"%24.16e\n",tsi[ii]/variance);
    fprintf(fp, "];\n");
    if (qData.strArray_ == NULL) 
    {
      if (plotScilab()) fprintf(fp, "  Str = [");
      else              fprintf(fp, "  Str = {");
      for (ii = 0; ii < nInputs-1; ii++) fprintf(fp,"'X%d',",ii+1);
      if (plotScilab()) fprintf(fp,"'X%d'];\n",nInputs);
      else              fprintf(fp,"'X%d'};\n",nInputs);
    }
    else
    {
      if (plotScilab()) fprintf(fp, "  Str = [");
      else              fprintf(fp, "  Str = {");
      for (ii = 0; ii < nInputs-1; ii++)
      {
        if (qData.strArray_[ii] != NULL) 
             fprintf(fp,"'%s',",qData.strArray_[ii]);
        else fprintf(fp,"'X%d',",ii+1);
      }
      if (plotScilab()) 
      {
        if (qData.strArray_[nInputs-1] != NULL) 
             fprintf(fp,"'%s']",qData.strArray_[nInputs-1]);
        else fprintf(fp,"'X%d'];\n",nInputs);
      }
      else
      {
        if (qData.strArray_[nInputs-1] != NULL) 
             fprintf(fp,"'%s'}",qData.strArray_[nInputs-1]);
        else fprintf(fp,"'X%d'};\n",nInputs);
      }
    }
    fwritePlotCLF(fp);
    fprintf(fp, "if (sortFlag == 1)\n");
    if (plotScilab())
         fprintf(fp, "  [Mids, I2] = gsort(Mids);\n");
    else fprintf(fp, "  [Mids, I2] = sort(Mids,'descend');\n");
    fprintf(fp, "  Str  = Str(I2);\n");
    fprintf(fp, "  I2 = I2(1:nn);\n");
    fprintf(fp, "  Mids = Mids(1:nn);\n");
    fprintf(fp, "  Str  = Str(1:nn);\n");
    fprintf(fp, "end\n");
    fprintf(fp, "ymin = min(Mids);\n");
    fprintf(fp, "ymin = 0.0;\n");
    fprintf(fp, "ymax = max(Mids);\n");
    fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
    fprintf(fp, "bar(Mids,0.8);\n");
    fwritePlotAxes(fp);
    if (plotScilab())
    {
      fprintf(fp, "a=gca();\n");
      fprintf(fp, "a.data_bounds=[0, ymin; nn+1, ymax];\n");
      fprintf(fp, "newtick = a.x_ticks;\n");
      fprintf(fp, "newtick(2) = [1:nn]';\n");
      fprintf(fp, "newtick(3) = Str';\n");
      fprintf(fp, "a.x_ticks = newtick;\n");
      fprintf(fp, "a.x_label.font_size = 3;\n");
      fprintf(fp, "a.x_label.font_style = 4;\n");
    }
    else
    {
      fprintf(fp, "axis([0 nn+1 ymin ymax])\n");
      fprintf(fp, "set(gca,'XTickLabel',[]);\n");
      fprintf(fp, "th=text(1:nn, repmat(ymin-0.07*(ymax-ymin),nn,1),Str,");
      fprintf(fp, "'HorizontalAlignment','left','rotation',90);\n");
      fprintf(fp, "set(th, 'fontsize', 12)\n");
      fprintf(fp, "set(th, 'fontweight', 'bold')\n");
    }
    fwritePlotTitle(fp,"Sobol Total Order Indices");
    fwritePlotYLabel(fp, "Sobol Indices");
    fclose(fp);
    if (plotScilab())
         printOutTS(PL_INFO, "Plot file = scilabrssoboltsi.sci\n");
    else printOutTS(PL_INFO, "Plot file = matlabrssoboltsi.m\n");
  }
  else
  {
    printOutTS(PL_ERROR,
        "RSMSobolTSI ERROR: cannot create tsi plot file.\n");
  }
  printAsterisks(PL_INFO, 0);
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
RSMSobolTSIAnalyzer& RSMSobolTSIAnalyzer::operator=(const RSMSobolTSIAnalyzer&)
{
  printOutTS(PL_ERROR,
             "RSMSobolTSI operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int RSMSobolTSIAnalyzer::get_nInputs()
{
  return nInputs_;
}
double RSMSobolTSIAnalyzer::get_outputMean()
{
  return outputMean_;
}
double RSMSobolTSIAnalyzer::get_outputStd()
{
  return outputStd_;
}
double RSMSobolTSIAnalyzer::get_tsi(int ind)
{
  if (ind < 0 || ind >= nInputs_)
  {
    printf("RSMSobolTSI ERROR: get_tsi index error.\n");
    return 0.0;
  }
  if (vecTSIs_.length() == 0)
  {
    printf("RSMSobolTSI ERROR: get_tsi has not value.\n");
    return 0.0;
  }
  return vecTSIs_[ind];
}

