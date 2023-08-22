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
#include "PrintingTS.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
RSMSobol2Analyzer::RSMSobol2Analyzer() : Analyzer(),nInputs_(0),
                   outputMean_(0),outputStd_(0)
{
  setName("RSMSOBOL2");
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
                                double *ubs, double *XIn, double *YIn)
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
  adata.inputPDFs_   = vecIPDFs.getIVector();
  adata.inputMeans_  = vecIMeans.getDVector();
  adata.inputStdevs_ = vecIStdvs.getDVector();
  analyze2(adata);
  adata.inputPDFs_   = NULL;
  adata.inputMeans_  = NULL;
  adata.inputStdevs_ = NULL;
  adata.iLowerB_ = NULL;
  adata.iUpperB_ = NULL;
  adata.sampleInputs_ = NULL;
  adata.sampleOutputs_ = NULL;
}

// ************************************************************************
// perform analysis
// ------------------------------------------------------------------------
double RSMSobol2Analyzer::analyze(aData &adata)
{
  int    ii, ii2, iL, iR, jj, kk, currNLevels, totalCnt, selectedInput=-1;
  double vce, ddata, ecv;
  char   pString[500], *cString, winput[500], winput2[500];
  pData  pCorMat, pPDF;
  psMatrix  *corMatp, corMat;

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
      "* - ana_expert mode to finetune RSMSobol2 parameters \n");
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
  nInputs_ = adata.nInputs_;
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
  int    *pdfFlags    = adata.inputPDFs_;
  double *inputMeans  = adata.inputMeans_;
  double *inputStdevs = adata.inputStdevs_;
  psIVector vecPdfFlags;
  psVector  vecInpMeans, vecInpStds;
  if (inputMeans == NULL || pdfFlags == NULL || inputStdevs == NULL)
  {
    vecPdfFlags.setLength(nInputs);
    pdfFlags = vecPdfFlags.getIVector();
    vecInpMeans.setLength(nInputs);
    inputMeans = vecInpMeans.getDVector();
    vecInpStds.setLength(nInputs);
    inputStdevs = vecInpStds.getDVector();
  }
  int noPDF = 1;
  if (pdfFlags != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (pdfFlags[ii] != 0) noPDF = 0;
  }
  if (psConfig_.InteractiveIsOn())
  {
    if (noPDF == 1) 
      printOutTS(PL_INFO,
         "RSMSobol2 INFO: all uniform distributions.\n");
    else
    {
      printOutTS(PL_INFO,"RSMSobol2 INFO: non-uniform ");
      printOutTS(PL_INFO,"distributions detected - will be used in\n");
      printOutTS(PL_INFO,"                this analysis.\n");
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 1 || nSamples <= 0 || nOutputs <= 0)
  {
    printOutTS(PL_ERROR,"RSMSobol2 ERROR: invalid arguments.\n");
    printOutTS(PL_ERROR,"   nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR,"   nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR,"   nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  }
  if (nInputs <= 2)
  {
    printOutTS(PL_ERROR,
         "RSMSobol2 ERROR: no need for this analysis (nInputs<=2).\n");
    return PSUADE_UNDEFINED;
  }
  if (outputID < 0 || outputID >= nOutputs)
  {
    printOutTS(PL_ERROR,
         "RSMSobol2 ERROR: invalid outputID (%d).\n",outputID);
    return PSUADE_UNDEFINED;
  }
  if (ioPtr == NULL)
  {
    printOutTS(PL_INFO,
         "RSMSobol2 INFO: no data object (PsuadeData) found.\n");
    printOutTS(PL_INFO,
         "          Several features will be turned off.\n");
    corMatp = new psMatrix();
    corMatp->setDim(nInputs, nInputs);
    for (ii = 0; ii < nInputs; ii++) corMatp->setEntry(ii,ii,1.0e0);
  } 
  else
  {
    //**/ if there is any correlation or if the PDF is of type S,
    //**/ call analyze2
    ioPtr->getParameter("input_cor_matrix", pCorMat);
    corMatp = (psMatrix *) pCorMat.psObject_;
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = 0; jj < ii; jj++)
      {
        if (corMatp->getEntry(ii,jj) != 0.0)
        {
          printOutTS(PL_INFO,"RSMSobol2 INFO: this method cannot ");
          printOutTS(PL_INFO,"handle correlated inputs using\n");
          printOutTS(PL_INFO,"          joint PDFs. PSUADE will ");
          printOutTS(PL_INFO,"try a variant of this method, or\n");
          printOutTS(PL_INFO,"          you can also run using the ");
          printOutTS(PL_INFO,"group variance-based method.\n");
          return analyze2(adata);
        }
      }
      if (pdfFlags[ii] == PSUADE_PDF_SAMPLE)
      {
        printOutTS(PL_ERROR,
          "RSMSobol2 INFO: this method cannot handle S PDF type.\n");
        printOutTS(PL_INFO,
          "          PSUADE will try a variant of this method.\n");
        return analyze2(adata);
      }
    }
  }
  int status = 0;
  for (ii = 0; ii < nSamples; ii++)
     if (YIn[nOutputs*ii+outputID] > 0.9*PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR,
     "RSMSobol2 ERROR: Some outputs are undefined. Prune the undefined\n");
    printOutTS(PL_ERROR,
     "                 sample points and re-run.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ---------------------------------------------------------------
  //**/ set up constraint filters
  //**/ ---------------------------------------------------------------
  RSConstraints *constrPtr;
  if (ioPtr != NULL)
  {
    constrPtr = new RSConstraints();
    constrPtr->genConstraints(ioPtr);
  }
  else 
  {
    constrPtr = NULL;
    printf("RSMSobolTSI INFO: no PsuadeData ==> no constraints.\n");
  }

  //**/ ---------------------------------------------------------------
  //**/ build response surface
  //**/ ---------------------------------------------------------------
  FuncApprox *faPtr=NULL;
  if (ioPtr == NULL)
  {
    printf("Select response surface. Options are: \n");
    writeFAInfo(0);
    strcpy(pString, "Choose response surface: ");
    int rstype = getInt(0, PSUADE_NUM_RS, pString);
    faPtr = genFA(rstype, nInputs, 0, nSamples);
  }
  else faPtr = genFAInteractive(ioPtr, 0);
  faPtr->setBounds(xLower, xUpper);
  psVector vecYT;
  vecYT.setLength(nSamples);
  for (ii = 0; ii < nSamples; ii++) 
    vecYT[ii] = YIn[ii*nOutputs+outputID];
  if (psConfig_.InteractiveIsOn())
    printOutTS(PL_INFO,"RSMSobol2: creating response surface...\n");
  psConfig_.InteractiveSaveAndReset();
  status = faPtr->initialize(XIn, vecYT.getDVector());
  psConfig_.InteractiveRestore();

  //**/ ---------------------------------------------------------------
  //**/  get internal parameters from users
  //**/ ---------------------------------------------------------------
  int nSubSamples=100, nLevels=100;
  if (psConfig_.InteractiveIsOn()) printAsterisks(PL_INFO, 0);
  if (psConfig_.AnaExpertModeIsOn())
  {
     printOutTS(PL_INFO,"RSMSobol2 generates a mesh of ");
     printOutTS(PL_INFO,"size K x K for every pair of inputs\n");
     printOutTS(PL_INFO,"          and then creates a sample ");
     printOutTS(PL_INFO,"of size M for each mesh point.\n");
     printOutTS(PL_INFO,"The total sample size is:\n");
     printOutTS(PL_INFO,
          "      N = M * K * K * nInputs * (nInputs - 1) / 2.\n");
     printOutTS(PL_INFO,"NOW, nInputs = %d\n", nInputs);
     printOutTS(PL_INFO,"Please select M and K below.\n");
     printOutTS(PL_INFO,"Recommendation: K x K >> M.\n");
     printOutTS(PL_INFO,"NOTE: large M and K can take a long time.\n");
     printOutTS(PL_INFO,"Default M = %d\n", nSubSamples);
     printOutTS(PL_INFO,"Default K = %d\n", nLevels);
     printEquals(PL_INFO, 0);
     snprintf(pString,100,"Enter M (suggestion: 100 - 1000) : ");
     nSubSamples = getInt(100, 1000, pString);
     snprintf(pString,100,"Enter K (suggestion:  50 - 500) : ");
     nLevels = getInt(50, 500, pString);

     //**/snprintf(pString,100,
     //**/  "If analyze one input only, enter input number (0 if all): ");
     //**/ don't remember what it does, disable for now (2/2014)
     //**/selectedInput = getInt(0, nInputs, pString);
     //**/selectedInput--;
     //**/printAsterisks(PL_INFO, 0);
  }
  else
  {
    nSubSamples = 100;
    nLevels = 100;
    cString = psConfig_.getParameter("RSMSobol2_nsubsamples");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput, winput2, &nSubSamples);
      if (nSubSamples < 100)
      {
        printOutTS(PL_INFO, 
             "RSMSobol2 INFO: nSubSamples should be >= 100.\n");
        nSubSamples = 100;
      }
      else
      {
        printOutTS(PL_INFO, 
           "RSMSobol2 INFO: nSubSamples = %d (config).\n",
           nSubSamples);
      }
    }
    cString = psConfig_.getParameter("RSMSobol2_nlevels");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput, winput2, &nLevels);
      if (nLevels < 100)
      {
        printOutTS(PL_INFO, 
             "RSMSobol2 INFO: nLevels should be >= 100.\n");
        nLevels = 100;
      }
      else
      {
        printOutTS(PL_INFO, 
             "RSMSobol2 INFO: nLevels = %d (config).\n",nLevels);
      }
    }
    if (psConfig_.InteractiveIsOn() && printLevel > 0)
    {
      printOutTS(PL_INFO,"RSMSobol2: default M = %d.\n", nSubSamples);
      printOutTS(PL_INFO,"RSMSobol2: default K = %d.\n", nLevels);
      printOutTS(PL_INFO,
           "To change these settings, re-run with ana_expert mode on.\n");
    }
  }
  if (psConfig_.InteractiveIsOn()) printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/  use response surface to compute total variance
  //**/ ---------------------------------------------------------------
  //**/ ----------------------------------------------------
  //**/ use a large sample size
  //**/ ----------------------------------------------------
  int nSamp = nLevels * nLevels * nSubSamples;
  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,
       "RSMSobol2 INFO: Creating a sample for basic statistics.\n");
    printOutTS(PL_INFO,"                Sample size = %d\n", nSamp);
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
  PDFManager *pdfman=NULL;
  Sampling   *sampler=NULL;
  if (noPDF == 0)
  {
    pdfman = new PDFManager();
    pdfman->initialize(nInputs,pdfFlags,inputMeans,
                       inputStdevs,*corMatp,NULL,NULL);
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
       "RSMSobol2: Response surface evaluation begins ...\n");

  faPtr->evaluatePoint(nSamp,vecXX.getDVector(),vecYY.getDVector());

  if (psConfig_.InteractiveIsOn())
    printOutTS(PL_INFO,
       "RSMSobol2: Response surface evaluation completed.\n");
  
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
         "RSMSobol2 ERROR: too few samples that satisify\n");
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
       "RSMSobol2: sample mean    (based on N = %d) = %10.3e\n",
       sCnt, dmean);
    printOutTS(PL_INFO,
       "RSMSobol2: sample std dev (based on N = %d) = %10.3e\n",
       sCnt, sqrt(variance));
  }
  if (variance == 0.0) variance = 1.0;

  //**/ ----------------------------------------------------
  //**/ save mean & std
  //**/ ----------------------------------------------------
  outputMean_ = dmean;
  outputStd_ = sqrt(variance);

  //**/ ---------------------------------------------------------------
  //**/  use response surface to perform Sobol two parameter test
  //**/ ---------------------------------------------------------------
  //**/ set up the sampling method
  psVector vecCLBs, vecCUBs;
  vecCLBs.setLength(nInputs);
  vecCUBs.setLength(nInputs);
  nSamp  = nSubSamples;
  psVector vecXT, vecMeans, vecVars;
  vecXT.setLength(nSubSamples*nInputs);
  vecYT.setLength(nSubSamples*nInputs);
  vecMeans.setLength(nLevels*nLevels);
  vecVars.setLength(nLevels*nLevels);
  psIVector vecBins;
  vecBins.setLength(nLevels*nLevels);
  psIVector vecPdfFlags1, vecPdfFlags2;
  psVector  vecInpMeans1, vecInpStds1, vecInpMeans2, vecInpStds2;
  vecPdfFlags1.setLength(2);
  vecInpMeans1.setLength(2);
  vecInpStds1.setLength(2);
  vecPdfFlags2.setLength(nInputs-2);
  vecInpMeans2.setLength(nInputs-2);
  vecInpStds2.setLength(nInputs-2);

  psVector vecSamPts2D, vecSubSamPts;
  vecSamPts2D.setLength(nLevels*nLevels*2);
  vecSubSamPts.setLength(nInputs*nSubSamples);

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
  if (psConfig_.InteractiveIsOn()) printAsterisks(PL_INFO, 0);
  vecVces_.setLength(nInputs*nInputs);
  vecEcvs_.setLength(nInputs*nInputs);
  for (ii = 0; ii < nInputs; ii++)
  {
    for (ii2 = 0; ii2 < nInputs; ii2++)
    {
      vce = 0.0;
      if (ii2 <= ii) continue;
      if (selectedInput != -1 && ii != selectedInput && 
          ii2 != selectedInput)
        continue;
      printOutTS(PL_DETAIL, "RSMSobol2: processing input pair %d, %d\n",
                 ii+1, ii2+1);

      //**/ use 4 levels of refinements to compute confidence interval
      currNLevels = nLevels / 2;
      for (iR = 0; iR < 2; iR++)
      {
        printOutTS(PL_DETAIL,
             "RSMSobol2: Processing refinement %d (4)\n",iR+1);

        //**/ create sample with pdf for 2 inputs
        vecCLBs[0] = xLower[ii];
        vecCUBs[0] = xUpper[ii];
        vecCLBs[1] = xLower[ii2];
        vecCUBs[1] = xUpper[ii2];
        if (noPDF == 0)
        {
          corMat.setDim(2,2);
          corMat.setEntry(0, 0, corMatp->getEntry(ii,ii));
          corMat.setEntry(1, 1, corMatp->getEntry(ii2,ii2));
          vecPdfFlags1[0] = pdfFlags[ii];
          vecPdfFlags1[1] = pdfFlags[ii2];
          vecInpMeans1[0] = inputMeans[ii];
          vecInpMeans1[1] = inputMeans[ii2];
          vecInpStds1[0]  = inputStdevs[ii];
          vecInpStds1[1]  = inputStdevs[ii2];
          PDFManager *pdfman1 = new PDFManager();
          pdfman1->initialize(2,vecPdfFlags1.getIVector(),
                    vecInpMeans1.getDVector(),vecInpStds1.getDVector(),
                    corMat,NULL,NULL);
          vecLB.load(2, vecCLBs.getDVector());
          vecUB.load(2, vecCUBs.getDVector());
          vecOut.setLength(currNLevels*currNLevels*2);
          pdfman1->genSample(currNLevels*currNLevels,vecOut,vecLB,vecUB);
          for (jj = 0; jj < currNLevels*currNLevels*2; jj++) 
            vecSamPts2D[jj] = vecOut[jj];
          delete pdfman1;
        }
        else
        {
          sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
          sampler->setInputBounds(2,vecCLBs.getDVector(),
                                  vecCUBs.getDVector());
          sampler->setOutputParams(1);
          sampler->setSamplingParams(currNLevels*currNLevels, 1, 0);
#ifndef PSUADE_OMP
          psConfig_.SamExpertModeSaveAndReset();
#endif
          sampler->initialize(0);
#ifndef PSUADE_OMP
          psConfig_.SamExpertModeRestore();
#endif
          vecST.setLength(currNLevels*currNLevels);
          psVector vecZT;
          vecZT.setLength(currNLevels*currNLevels);
          sampler->getSamples(currNLevels*currNLevels,2,1,
                       vecSamPts2D.getDVector(),vecZT.getDVector(),
                       vecST.getIVector());
          delete sampler;
        }

        //**/ create sample with pdf for the other inputs
        if (noPDF == 0)
        {
          corMat.setDim(nInputs-2, nInputs-2);
          for (jj = 0; jj < ii; jj++)
          {
            vecCLBs[jj] = xLower[jj];
            vecCUBs[jj] = xUpper[jj];
            corMat.setEntry(jj, jj, corMatp->getEntry(jj,jj));
            vecPdfFlags2[jj] = pdfFlags[jj];
            vecInpMeans2[jj] = inputMeans[jj];
            vecInpStds2[jj]  = inputStdevs[jj];
          }
          for (jj = ii+1; jj < ii2; jj++)
          {
            vecCLBs[jj-1] = xLower[jj];
            vecCUBs[jj-1] = xUpper[jj];
            corMat.setEntry(jj-1, jj-1, corMatp->getEntry(jj,jj));
            vecPdfFlags2[jj-1] = pdfFlags[jj];
            vecInpMeans2[jj-1] = inputMeans[jj];
            vecInpStds2[jj-1]  = inputStdevs[jj];
          }
          for (jj = ii2+1; jj < nInputs; jj++)
          {
            vecCLBs[jj-2] = xLower[jj];
            vecCUBs[jj-2] = xUpper[jj];
            corMat.setEntry(jj-2, jj-2, corMatp->getEntry(jj,jj));
            vecPdfFlags2[jj-2] = pdfFlags[jj];
            vecInpMeans2[jj-2] = inputMeans[jj];
            vecInpStds2[jj-2]  = inputStdevs[jj];
          }
          PDFManager *pdfman2 = new PDFManager();
          pdfman2->initialize(nInputs-2,vecPdfFlags2.getIVector(),
                    vecInpMeans2.getDVector(),vecInpStds2.getDVector(),
                    corMat,NULL,NULL);
          vecLB.load(nInputs-2, vecCLBs.getDVector());
          vecUB.load(nInputs-2, vecCUBs.getDVector());
          vecOut.setLength(nSubSamples*(nInputs-2));
          pdfman2->genSample(nSubSamples, vecOut, vecLB, vecUB);
          for (jj = 0; jj < nSubSamples*(nInputs-2); jj++)
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
          //**/ May, 2014
          //**/if (nInputs-1 > 51)
          //**/     sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
          //**/else sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
          sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
          sampler->setInputBounds(nInputs-2,vecCLBs.getDVector(), 
                                  vecCUBs.getDVector());
          sampler->setOutputParams(1);
          sampler->setSamplingParams(nSubSamples, 1, 1);
#ifndef PSUADE_OMP
          psConfig_.SamExpertModeSaveAndReset();
#endif
          sampler->initialize(0);
#ifndef PSUADE_OMP
          psConfig_.SamExpertModeRestore();
#endif
          vecST.setLength(nSubSamples);
          psVector vecZT;
          vecZT.setLength(nSubSamples);
          sampler->getSamples(nSubSamples,nInputs-2,1,vecXT.getDVector(),
                              vecZT.getDVector(),vecST.getIVector());
          delete sampler;
        }

        //**/ use currNLevels levels per input
        double *XPtr = vecXT.getDVector();
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
        {
          //**/ extract the sample point 
          for (jj = 0; jj < nSubSamples; jj++)
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
          faPtr->evaluatePoint(nSubSamples,vecSubSamPts.getDVector(),
                               vecYT.getDVector());

          //**/ go through all filters the sample point and evaluate
          double *dPtr = vecSubSamPts.getDVector();
          for (jj = 0; jj < nSubSamples; jj++)
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
          for (jj = 0; jj < nSubSamples; jj++)
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
                 "RSMSobol2 WARNING: subsample size = 0.\n");
          if (sCnt < 1) vecMeans[iL] = PSUADE_UNDEFINED;
          else          vecMeans[iL] /= (double) sCnt;

          //**/ compute the variance  at each input pair levels
          vecVars[iL] = 0.0;
          ddata = vecMeans[iL];
          for (jj = 0; jj < nSubSamples; jj++)
          {
            if (vecYT[jj] != PSUADE_UNDEFINED)
              vecVars[iL] += (vecYT[jj]-ddata)*(vecYT[jj]-ddata);
          }
          if (sCnt < 1) vecVars[iL] = PSUADE_UNDEFINED;
          else          vecVars[iL] /= (double) sCnt;

          //**/printOutTS(PL_DUMP,"RSMSobol2: inputs (%d,%d)\n",
          //**/           ii+1,ii2+1);
          //**/printOutTS(PL_DUMP,
          //**/     "  refinement %d, gridpoint (%d), size %d (%d)\n",
          //**/          iR, iL+1, sCnt, nSubSamples);
          //**/printOutTS(PL_DUMP,"     mean = %e\n", vecMeans[iL]);
          //**/printOutTS(PL_DUMP,"     var  = %e\n", vecVars[iL]);
        }

        //**/ compute the variance of the means for each input pair
        dmean = 0.0;
        totalCnt = 0;
        for (iL = 0; iL < currNLevels*currNLevels; iL++) 
          totalCnt += vecBins[iL];
        if (totalCnt == 0)
        {
          printOutTS(PL_ERROR,
               "RSMSobol2 ERROR: empty constrained space.\n");
          printOutTS(PL_ERROR,
               "          Either try larger sample size or\n");
          printOutTS(PL_ERROR,"          use looser constraints.\n");
          exit(1);
        }
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
        {
          if (vecMeans[iL] != PSUADE_UNDEFINED)
            dmean += vecMeans[iL] * vecBins[iL] / totalCnt;
        }
        vce = 0.0;
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
        {
          if (vecMeans[iL] != PSUADE_UNDEFINED)
            vce += (vecMeans[iL]-dmean) * (vecMeans[iL]-dmean) * 
                    vecBins[iL] / totalCnt;
        }

        //**/ compute the mean of the variances for each input pair
        ecv = 0.0;
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
          if (vecVars[iL] != PSUADE_UNDEFINED) 
            ecv += vecVars[iL] * vecBins[iL] / totalCnt;

        if (psConfig_.InteractiveIsOn()) 
        {
          if (printLevel > 1 || iR == 1)
            printOutTS(PL_INFO, 
               "VCE(%3d,%3d) = %10.3e, (normalized) = %10.3e\n",
               ii+1, ii2+1, vce, vce/variance);
          if (printLevel > 2)
            printOutTS(PL_INFO, 
                 "ECV(%3d,%3d) = %10.3e, (normalized) = %10.3e\n",
                 ii+1, ii2+1, ecv, ecv/variance);
        }
        currNLevels *= 2;
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
// perform analysis (for problems with joint PDFs)
// ------------------------------------------------------------------------
double RSMSobol2Analyzer::analyze2(aData &adata)
{
  int    ii, ii2, jj, iL, iR, currNLevels, nSamp, totalCnt, bin1, bin2;
  double ecv, ddata, vce, width1, width2;
  char   pString[500];
  pData  pPDF;

  //**/ ---------------------------------------------------------------
  //**/ display message if it is not a library call
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"RSMSobol2: Since joint PDFs have ");
    printOutTS(PL_INFO,"been specified, a different two-way\n");
    printOutTS(PL_INFO,"           analysis will be performed.\n");
  }
  //**/ ---------------------------------------------------------------
  //**/ extract test data
  //**/ ---------------------------------------------------------------
  nInputs_       = adata.nInputs_;
  int nInputs    = adata.nInputs_;
  int nOutputs   = adata.nOutputs_;
  int nSamples   = adata.nSamples_;
  double *xLower = adata.iLowerB_;
  double *xUpper = adata.iUpperB_;
  double *XIn    = adata.sampleInputs_;
  double *YIn    = adata.sampleOutputs_;
  int outputID   = adata.outputID_;
  int printLevel  = adata.printLevel_;
  PsuadeData *ioPtr = adata.ioPtr_;
  psVector vecYT;
  vecYT.setLength(nSamples);
  for (ii = 0; ii < nSamples; ii++) 
    vecYT[ii] = YIn[ii*nOutputs+outputID];

  //**/ ---------------------------------------------------------------
  //**/ set up constraint filters
  //**/ ---------------------------------------------------------------
  RSConstraints *constrPtr=NULL;
  if (ioPtr != NULL)
  {
    constrPtr = new RSConstraints();
    constrPtr->genConstraints(ioPtr);
  }
  else constrPtr = NULL;

  //**/ ---------------------------------------------------------------
  //**/ build response surface
  //**/ ---------------------------------------------------------------
  FuncApprox *faPtr=NULL;
  if (ioPtr == NULL)
  {
    if (rstype_ < 0) jj = 0;
    else             jj  = rstype_; 
    faPtr = genFA(jj, nInputs, 0, nSamples);
    faPtr->setBounds(xLower, xUpper);
    faPtr->setOutputLevel(0);
  }
  else faPtr = genFAInteractive(ioPtr, 0);
  if (faPtr == NULL)
  {
    printOutTS(PL_ERROR,
         "RSMSobol2 ERROR: cannot create response surface.\n");
    delete constrPtr;
    return 1.0e12;
  }
  if (psConfig_.InteractiveIsOn())
    printOutTS(PL_INFO,"RSMSobol2: creating a response surface ...\n");
  psConfig_.InteractiveSaveAndReset();
  int status = faPtr->initialize(XIn, vecYT.getDVector());
  psConfig_.InteractiveRestore();

  //**/ ---------------------------------------------------------------
  //**/  get internal parameters from users
  //**/ ---------------------------------------------------------------
  int nLevels=100, nSubSamples=100;
  if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"RSMSobol2 generates a mesh of size ");
    printOutTS(PL_INFO,"K x K for every pair of inputs\n");
    printOutTS(PL_INFO,"   and then creates a sample of size ");
    printOutTS(PL_INFO,"M for each mesh point. The total\n");
    printOutTS(PL_INFO,"   sample size is:\n");
    printOutTS(PL_INFO,
         "       N = M * K * K * nInputs * (nInputs - 1) / 2.\n");
    printOutTS(PL_INFO,"NOW, nInputs = %d\n", nInputs);
    printOutTS(PL_INFO,"Please select your desired M and K.\n");
    printOutTS(PL_INFO,"Recommendation: K x K >> M.\n");
    printOutTS(PL_INFO,"default M = %d.\n", nSubSamples);
    printOutTS(PL_INFO,"default K = %d.\n", nLevels);
    printOutTS(PL_INFO,"NOTE: large M and K can take a long time.\n");
    printEquals(PL_INFO, 0);
    snprintf(pString,100,"Enter M (suggestion: 100 - 1000) : ");
    nSubSamples = getInt(100, 1000, pString);
    snprintf(pString,100,"Enter nLevels (suggestion: 50 - 500) : ");
    nLevels = getInt(50, 500, pString);
    printAsterisks(PL_INFO, 0);
  }
  else
  {
    nSubSamples = 100;
    nLevels = 100;
    if (psConfig_.InteractiveIsOn())
    {
      printOutTS(PL_INFO,"RSMSobol2: default M = %d.\n", nSubSamples);
      printOutTS(PL_INFO,"RSMSobol2: default K = %d.\n", nLevels);
      printOutTS(PL_INFO,
        "* To change these settings, re-run with ana_expert mode on.\n");
      printAsterisks(PL_INFO, 0);
    }
  }

  //**/ ---------------------------------------------------------------
  //**/  use response surface to compute total variance
  //**/ ---------------------------------------------------------------
  //**/ ----------------------------------------------------
  //**/ use a large sample size
  //**/ ----------------------------------------------------
  nSamp = nLevels * nLevels * nSubSamples;
  if (psConfig_.InteractiveIsOn() && printLevel > 1)
  {
    printOutTS(PL_INFO,
         "RSMSobol2 INFO: Creating a sample for basic statistics.\n");
    printOutTS(PL_INFO,"*                 Sample size = %d\n", nSamp);
  }

  //**/ ----------------------------------------------------
  //**/ allocate space
  //**/ ----------------------------------------------------
  psVector vecXX, vecYY;
  vecXX.setLength(nSamp*nInputs);
  vecYY.setLength(nSamp);

  //**/ ----------------------------------------------------
  //**/ create a sample from pdf
  //**/ ----------------------------------------------------
  PDFManager *pdfman = NULL;
  if (ioPtr != NULL)
  {
    pdfman = new PDFManager();
    pdfman->initialize(ioPtr);
  }
  else
  {
    //**/ this path is for library call only
    pdfman = new PDFManager();
    int *inputPDFs = adata.inputPDFs_;
    double *inputMeans = adata.inputMeans_;
    double *inputStdevs = adata.inputStdevs_;
    if (inputPDFs == NULL || inputMeans == NULL || inputStdevs == NULL)
    {
      printf("RSMSobol2 ERROR: PDF information not provided.\n");
      exit(1);
    }
    psMatrix cMat;
    cMat.setDim(nInputs, nInputs);
    for (ii = 0; ii < nInputs; ii++) cMat.setEntry(ii,ii,1);
    psStrings strNames;
    strNames.setNumStrings(nInputs);
    for (ii = 0; ii < nInputs; ii++) 
    {
      snprintf(pString,100,"X%d", ii+1);
      strNames.loadOneString(ii, pString);
    }
    pdfman->initialize(nInputs, inputPDFs, inputMeans, inputStdevs,
                       cMat, strNames.getStrings(), NULL);
  }
  psVector vecLB, vecUB, vecOut;
  vecLB.load(nInputs, xLower);
  vecUB.load(nInputs, xUpper);
  vecOut.setLength(nSamp*nInputs);
  pdfman->genSample(nSamp, vecOut, vecLB, vecUB);
  for (ii = 0; ii < nSamp*nInputs; ii++) vecXX[ii] = vecOut[ii];

  //**/ ----------------------------------------------------
  //**/ evaluate 
  //**/ ----------------------------------------------------
  if (psConfig_.InteractiveIsOn())
    printOutTS(PL_INFO, 
       "RSMSobol2: Response surface evaluation begins ...\n");

  faPtr->evaluatePoint(nSamp, vecXX.getDVector(), vecYY.getDVector());

  if (psConfig_.InteractiveIsOn())
    printOutTS(PL_INFO, 
       "RSMSobol2: Response surface evaluation completed.\n");
  
  //**/ ------------------------
  //**/ apply filters
  //**/ ------------------------
  double *oneSamplePt=NULL;
  if (constrPtr != NULL)
  {
    for (ii = 0; ii < nSamp; ii++)
    {
      oneSamplePt = &(vecXX[ii*nInputs]);
      ddata = constrPtr->evaluate(oneSamplePt,vecYY[ii],status);
      if (status == 0) vecYY[ii] = PSUADE_UNDEFINED;
    }
  }
  
  //**/ ----------------------------------------------------
  //**/ compute statistics: mean, stdev
  //**/ ----------------------------------------------------
  int    sCnt = 0;
  double dmean = 0.0;
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
               "RSMSobol2 ERROR: too few samples satisify constraints.\n");
    printOutTS(PL_ERROR,"                 (%d out of %d).\n",sCnt,nSamp);
    delete faPtr;
    delete pdfman;
    delete constrPtr;
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
      "RSMSobol2: sample mean    (based on %d points) = %e\n",
      sCnt, dmean);
    printOutTS(PL_INFO,
      "RSMSobol2: total std dev  (based on %d points) = %e\n",
      sCnt, sqrt(variance));
    printAsterisks(PL_INFO, 0);
  }
  if (variance == 0.0) variance = 1.0;
  delete pdfman;

  //**/ ----------------------------------------------------
  //**/ save mean & std
  //**/ ----------------------------------------------------
  outputMean_ = dmean;
  outputStd_ = sqrt(variance);

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
  //**/  use response surface to perform Sobol two parameter test
  //**/ ---------------------------------------------------------------
  vecVces_.setLength(nInputs*nInputs);
  vecEcvs_.setLength(nInputs*nInputs);
  psVector  vecMeans, vecVars;
  psIVector vecBins;
  vecMeans.setLength(nLevels*nLevels);
  vecVars.setLength(nLevels*nLevels);
  vecBins.setLength(nLevels*nLevels);

  //**/ loop through each pair of inputs
  for (ii = 0; ii < nInputs; ii++)
  {
    for (ii2 = ii+1; ii2 < nInputs; ii2++)
    {
      if (psConfig_.InteractiveIsOn())
        printOutTS(PL_DETAIL, "RSMSobol2: Processing input pair %d, %d\n",
               ii+1, ii2+1);

      //**/ use 2 levels of refinements to compute confidence interval
      currNLevels = nLevels / 2;
      for (iR = 0; iR < 2; iR++)
      {
        if (psConfig_.InteractiveIsOn())
          printOutTS(PL_DETAIL, "RSMSobol2: Processing refinement %d\n",iR);

        //**/ initialize parameters
        width1 = (xUpper[ii] - xLower[ii]) / currNLevels;
        width2 = (xUpper[ii2] - xLower[ii2]) / currNLevels;
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
        {
          vecMeans[iL] = 0.0;
          vecVars[iL] = 0.0;
          vecBins[iL] = 0;
        }
        //**/ compute the conditional means
        for (jj = 0; jj < nSamp; jj++)
        {
          if (vecYY[jj] != PSUADE_UNDEFINED)
          {
            ddata = vecXX[jj*nInputs+ii];
            bin1 = (int) ((ddata - xLower[ii]) / width1);
            ddata = vecXX[jj*nInputs+ii2];
            bin2 = (int) ((ddata - xLower[ii2]) / width2);
            if (bin1 == currNLevels) bin1 = currNLevels - 1;
            if (bin2 == currNLevels) bin2 = currNLevels - 1;
            vecMeans[bin1*currNLevels+bin2] += vecYY[jj];
            vecBins[bin1*currNLevels+bin2]++;
          }
        }
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
        {
          sCnt = vecBins[iL];
          if (sCnt < 1 && printLevel >= 5)
             printOutTS(PL_DUMP,
                  "RSMSobol2 WARNING: subsample size = 0.\n");
          if (sCnt < 1) vecMeans[iL] = PSUADE_UNDEFINED;
          else          vecMeans[iL] /= (double) sCnt;
        }
        //**/ compute the conditional variances
        for (jj = 0; jj < nSamp; jj++)
        {
          if (vecYY[jj] != PSUADE_UNDEFINED)
          {
            ddata = vecXX[jj*nInputs+ii];
            bin1 = (int) ((ddata - xLower[ii]) / width1);
            ddata = vecXX[jj*nInputs+ii2];
            bin2 = (int) ((ddata - xLower[ii2]) / width2);
            if (bin1 == currNLevels) bin1 = currNLevels - 1;
            if (bin2 == currNLevels) bin2 = currNLevels - 1;
            ddata = vecMeans[bin1*currNLevels+bin2];
            vecVars[bin1*currNLevels+bin2] += 
                   (vecYY[jj]-ddata)*(vecYY[jj]-ddata);
          }
        }
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
        {
          sCnt = vecBins[iL];
          if (sCnt < 1) vecVars[iL] = PSUADE_UNDEFINED;
          else          vecVars[iL] /= (double) sCnt;
        }

        //**/ compute number of successes
        totalCnt = 0;
        for (iL = 0; iL < currNLevels*currNLevels; iL++) 
          totalCnt += vecBins[iL];
        if (totalCnt == 0)
        {
          printOutTS(PL_ERROR, 
               "RSMSobol2 ERROR: empty constrained space.\n");
          printOutTS(PL_ERROR, 
               "          Either try larger sample size or\n");
          printOutTS(PL_ERROR, "          use looser constraints.\n");
          exit(1);
        }

#if 1
        //**/ April 2019: do not need weighted means using bins
        //**/ compute the variance of the means for each input pair
        //**/ Nov 2021: We do need this for correlated inputs
        dmean = 0.0;
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
        {
          if (vecMeans[iL] != PSUADE_UNDEFINED)
            dmean += vecMeans[iL] * vecBins[iL] / totalCnt;
        }
        vce = 0.0;
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
          if (vecMeans[iL] != PSUADE_UNDEFINED)
            vce += (vecMeans[iL]-dmean) * (vecMeans[iL]-dmean) * 
                   vecBins[iL] / totalCnt;

        //**/ compute the mean of the variances for each input pair
        ecv = 0.0;
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
        {
          if (vecVars[iL] != PSUADE_UNDEFINED)
            ecv += vecVars[iL] * vecBins[iL] / totalCnt;
        }

#else
        //**/ Nov 2021: this is not correct. I don't remember why
        //**/ this is done
        totalCnt = 0;
        for (iL = 0; iL < currNLevels*currNLevels; iL++) 
          if (vecMeans[iL] != PSUADE_UNDEFINED) totalCnt++;
        dmean = 0.0;
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
          if (vecMeans[iL] != PSUADE_UNDEFINED) dmean += vecMeans[iL];
        dmean /= (double) totalCnt;
        vce = 0.0;
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
          if (vecMeans[iL] != PSUADE_UNDEFINED)
            vce += (vecMeans[iL]-dmean) * (vecMeans[iL]-dmean); 
        vce /= (double) totalCnt;

        //**/ compute the mean of the variances for each input pair
        ecv = 0.0;
        for (iL = 0; iL < currNLevels*currNLevels; iL++)
          if (vecVars[iL] != PSUADE_UNDEFINED) ecv += vecVars[iL];
        ecv /= (double) totalCnt;
#endif
        if (psConfig_.InteractiveIsOn() && (printLevel > 2 || iR == 1))
        {
          printOutTS(PL_INFO, 
               "VCE(%3d,%3d) = %10.3e, (normalized) = %10.3e\n",
               ii+1, ii2+1, vce, vce/variance);
        }
        if (psConfig_.InteractiveIsOn() && (printLevel > 3 || iR == 1))
        {
          printOutTS(PL_DETAIL, 
               "ECV(%3d,%3d) = %10.3e, (normalized) = %10.3e\n",
               ii+1, ii2+1, ecv, ecv/variance);
        }
        currNLevels *= 2;
      }
      //save vces & ecvs
      vecVces_[ii*nInputs+ii2] = vecVces_[ii2*nInputs+ii] = vce;
      vecEcvs_[ii*nInputs+ii2] = vecEcvs_[ii2*nInputs+ii] = ecv;
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
  delete constrPtr;
  delete faPtr;
  return 0.0;
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
  if (outputStd_ > 0)
    return vecVces_[ind1*nInputs_+ind2]/outputStd_/outputStd_;
  else
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
  if (outputStd_ > 0)
    return vecEcvs_[ind1*nInputs_+ind2]/outputStd_/outputStd_;
  else
    return vecEcvs_[ind1*nInputs_+ind2];
}

