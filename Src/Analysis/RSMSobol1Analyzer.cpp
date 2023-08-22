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
#include "RSMSobol1Analyzer.h"
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
                                double *ubs, double *X, double *Y)
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
}

// ************************************************************************
// perform analysis (different parameter list)
// ------------------------------------------------------------------------
double RSMSobol1Analyzer::analyze(aData &adata)
{
  int    ii, jj, kk, nn, ss, count, status, nSamp, rstype;
  int    nSubSamples=1000, nLevels=200, ntimes=1, nConstr;
  double dmean, dstds, ddata, ddata2, frac=0.8, *tempV, *oneSamplePt;
  char   pString[500], *cString, winput1[500], winput2[500];
  FuncApprox    *faPtr=NULL;
  RSConstraints *constrPtr;
  psVector      vecIn, vecOut, vecUB, vecLB, vecY;
  pData         pCorMat, pPtr;
  psMatrix      *corMatp=NULL, corMat;
  PDFManager    *pdfman;
  Sampling      *sampler;

  //**/ ===============================================================
  //**/ display header 
  //**/ ===============================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
         "*          RS-based First Order Sobol' Analysis \n");
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
  PsuadeData *ioPtr   = adata.ioPtr_;
  int    *pdfFlags    = adata.inputPDFs_;
  double *inputMeans  = adata.inputMeans_;
  double *inputStdevs = adata.inputStdevs_;

  //**/ extract input PDF information (if none, set all to none - 0)
  psIVector vecPdfFlags;
  psVector  vecInpMeans, vecInpStds;
  if (inputMeans == NULL || pdfFlags == NULL || inputStdevs == NULL)
  {
    //**/ note: setLength actually sets the vector to all 0's
    vecPdfFlags.setLength(nInputs);
    pdfFlags = vecPdfFlags.getIVector();
    vecInpMeans.setLength(nInputs);
    inputMeans = vecInpMeans.getDVector();
    vecInpStds.setLength(nInputs);
    inputStdevs = vecInpStds.getDVector();
  }
  //**/ if other than uniform PDF, set noPDF=0 
  //**/ Also, check for S PDFs and flag error
  int noPDF=1, corFlag=0;
  if (pdfFlags != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (pdfFlags[ii] != 0) noPDF = 0;
    for (ii = 0; ii < nInputs; ii++)
      if (pdfFlags[ii] == PSUADE_PDF_SAMPLE) corFlag = 1;
  }
  if (psConfig_.InteractiveIsOn())
  {
    if (noPDF == 1) 
      printOutTS(PL_INFO,
         "* RSMSobol1 INFO: all uniform distributions.\n");
    else
      printOutTS(PL_INFO,
         "RSMSobol1 INFO: non-uniform distributions detected.\n");
  }

  //**/ ===============================================================
  //**/ error checking
  //**/ ===============================================================
  if (nInputs <= 0 || nSamples <= 0 || nOutputs <= 0) 
  {
    printOutTS(PL_ERROR, "RSMSobol1 ERROR: invalid arguments.\n");
    printOutTS(PL_ERROR, "   nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR, "   nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR, "   nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 
  if (outputID >= nOutputs || outputID < 0)
  {
    printOutTS(PL_ERROR,"RSMSobol1 ERROR: invalid output ID (%d).\n", 
               outputID);
    return PSUADE_UNDEFINED;
  }
  if (nInputs <= 1)
  {
    printOutTS(PL_ERROR,
         "RSMSobol1 INFO: analysis not needed since nInputs = 1\n");
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
           "RSMSobol1 INFO: no data object (PsuadeData) found.\n");
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
              "RSMSobol1 INFO: Correlated inputs detected.\n");
            printOutTS(PL_INFO, 
              "   Alternative analyis is to be performed.\n");
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
        "RSMSobol1 ERROR: Some outputs are undefined.\n");
    printOutTS(PL_ERROR,
        "     Please prune the undefined sample points and re-run.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ===============================================================
  //**/ if there are input correlation, use a different analyzer
  //**/ ===============================================================
  if (corFlag != 0) 
  {
    //**/ first clean up
    if (ioPtr == NULL) delete corMatp;
    return analyze2(adata);
  }

  //**/ ===============================================================
  //**/ set up constraint filters
  //**/ ===============================================================
  if (ioPtr != NULL)
  {
    constrPtr = new RSConstraints();
    constrPtr->genConstraints(ioPtr);
    nConstr = constrPtr->getNumConstraints();
    if (nConstr == 0)
    {
      delete constrPtr;
      constrPtr = NULL;
    }
    else
      printf("RSMSobol1 INFO: %d constraints detected.\n",nConstr);
  }
  else
  {
    nConstr = 0;
    constrPtr = NULL;
    if (psConfig_.InteractiveIsOn())
      printf("RSMSobol1 INFO: no PsuadeData ==> no constraints.\n");
  }

  //**/ ===============================================================
  //**/ get response surface selection from user, if needed 
  //**/ ===============================================================
  if (ioPtr == NULL)
  {
    if (rstype_ < 0)
    {
      printf("Select response surface. Options are: \n");
      writeFAInfo(0);
      strcpy(pString, "Choose response surface: ");
      rstype = getInt(0, PSUADE_NUM_RS, pString);
    }
    else rstype = rstype_;
  }
  else
  {
    ioPtr->getParameter("ana_rstype", pPtr);
    rstype = pPtr.intData_;
  }

  //**/ ===============================================================
  //**/  get internal parameters from users
  //**/ ===============================================================
  if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"* RSMSobol1 creates one sample ");
    printOutTS(PL_INFO,"of size M for each of the K levels of\n");
    printOutTS(PL_INFO,"* each input. Therefore, the total ");
    printOutTS(PL_INFO,"sample size is N = M * K * nInputs\n");
    nSubSamples = 1000;
    nLevels = 1000;
    printOutTS(PL_INFO,"* nInputs   = %d\n", nInputs);
    printOutTS(PL_INFO,"* default M = %d\n", nSubSamples);
    printOutTS(PL_INFO,"* default K = %d\n", nLevels);
    printOutTS(PL_INFO,"* Recommendation: K >> M\n");
    printOutTS(PL_INFO,"* Please select different M and K.\n");
    printOutTS(PL_INFO,
         "* NOTE: large M and K may take very long time.\n");
    printEquals(PL_INFO, 0);
    snprintf(pString,100,"Enter M (suggestion: 100 - 10000) : ");
    nSubSamples = getInt(100, 50000, pString);
    snprintf(pString,100,"Enter K (suggestion: 1000 - 10000) : ");
    nLevels = getInt(1000, 50000, pString);

    //**/ ntimes should be 1 from now on
    //printOutTS(PL_INFO,"* To include response surface ");
    //printOutTS(PL_INFO,"uncertainties in this analysis, the\n");
    //printOutTS(PL_INFO,"* sensitivity calculation is to be ");
    //printOutTS(PL_INFO,"repeated a number of times using\n");
    //printOutTS(PL_INFO,"* different bootstrapped samples. ");
    //printOutTS(PL_INFO,"Please specify the number of\n");
    //printOutTS(PL_INFO,"* bootstrapped samples below.\n");
    //printOutTS(PL_INFO,
    //   "* If you do not need error bars, set it to 1.\n");
    //snprintf(pString,100,
    //   "Enter the number of bootstrapped samples (1 - 500) : ");
    //ntimes = getInt(1, 500, pString);
    ntimes = 1;
    printAsterisks(PL_INFO, 0);
  }
  else
  {
    nSubSamples = 1000;
    nLevels = 1000;
    cString = psConfig_.getParameter("RSMSobol1_nsubsamples");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nSubSamples);
      if (nSubSamples < 100)
      {
        printOutTS(PL_INFO,
             "RSMSobol1 INFO: nSubSamples should be >= 100.\n");
        nSubSamples = 1000;
      }
      else
      {
        printOutTS(PL_INFO,
             "RSMSobol1 INFO: nSubSamples = %d (config).\n",
             nSubSamples);
      }
    }
    cString = psConfig_.getParameter("RSMSobol1_nlevels");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nLevels);
      if (nLevels < 1000)
      {
        printOutTS(PL_INFO,
             "RSMSobol1 INFO: nLevels should be >= 1000.\n");
        nLevels = 1000;
      }
      else
      {
        printOutTS(PL_INFO,
             "RSMSobol1 INFO: nLevels = %d (config).\n",nLevels);
      }
    }
    if (psConfig_.InteractiveIsOn() && printLevel > 1)
    {
      printOutTS(PL_INFO,"* RSMSobol1 creates one sample ");
      printOutTS(PL_INFO,"of size M for each of the K levels of\n");
      printOutTS(PL_INFO,"* each input. Therefore, the total ");
      printOutTS(PL_INFO,"sample size is N = M * K * nInputs\n");
      printOutTS(PL_INFO,"* nInputs m = %d\n", nInputs);
      printOutTS(PL_INFO,"* default M = %d\n", nSubSamples);
      printOutTS(PL_INFO,"* default K = %d\n", nLevels);
      printOutTS(PL_INFO,
           "* To make changes, re-run with ana_expert mode on.\n");
      printEquals(PL_INFO, 0);
    }
  }
  if (ntimes > 1)
  {
    if (psConfig_.InteractiveIsOn())
    {
      printOutTS(PL_INFO,"RSMSobol1 INFO: number of bootstrapped ");
      printOutTS(PL_INFO,"samples larger than 1 is not\n");
      printOutTS(PL_INFO,"          recommended. Use the ");
      printOutTS(PL_INFO,"rssobol1b command instead.\n");
    }
    ntimes = 1;
  }
  if (ntimes == 1) frac = 1.0;

  //**/ ===============================================================
  //**/ use response surface to compute mean and variance
  //**/ do it 1000 times using fuzzy evaluations
  //**/ ===============================================================
  int nSamp2 = 1000000;
  if (psConfig_.InteractiveIsOn())
  {
    printOutTS(PL_INFO,
      "RSMSobol1 INFO: Creating a sample for basic statistics.\n");
    printOutTS(PL_INFO,"                Sample size = %d\n", nSamp2);
  }

  //**/ ---------------------------------------------------------
  //**/ generate a large sample for computing basic statistics
  //**/ ==> vecXX and vecYY (with sample size nSamp2) 
  //**/ noPDF!=0 means some inputs have non-uniform PDFS, but
  //**/ there is no S-type or any correlation allowed here
  //**/ ---------------------------------------------------------
  psIVector vecSS;
  psVector  vecXX, vecYY;
  vecXX.setLength(nSamp2*nInputs);
  vecYY.setLength(nSamp2);
  if (noPDF == 0)
  {
    //**/ create sample with pdf for all inputs
    pdfman = new PDFManager();
    pdfman->initialize(nInputs,pdfFlags,inputMeans,inputStdevs,
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
  //**/ use bootstrapping to compute basic statistics
  //**/ (Nov 2021: bootstrapping is turned off ==> ntimes=1)
  //**/ ---------------------------------------------------------
  psVector  vecBsX, vecBsY, vecBsMeans, vecBsStds;
  psIVector vecBsFlags;

  vecBsFlags.setLength(nSamples);
  vecBsX.setLength(nSamples*nInputs);
  vecBsY.setLength(nSamples);
  vecBsMeans.setLength(ntimes);
  vecBsStds.setLength(ntimes);
  count = (int) (frac * nSamples);
  faPtr = genFA(rstype, nInputs, 0, count);
  faPtr->setBounds(xLower, xUpper);
  faPtr->setOutputLevel(0);
  for (nn = 0; nn < ntimes; nn++)
  {
    //**/ generate a bootstrap
    if (ntimes == 1)
    {
      for (ss = 0; ss < nSamples*nInputs; ss++) vecBsX[ss] = XIn[ss];
      for (ss = 0; ss < nSamples; ss++) 
        vecBsY[ss] = YIn[ss*nOutputs+outputID];
    }
    else
    {
      for (ss = 0; ss < nSamples; ss++) vecBsFlags[ss] = 0;
      count = 0;
      while (count < frac * nSamples)
      {
        jj = PSUADE_rand() % nSamples;
        if (vecBsFlags[jj] == 0)
        {
          for (ii = 0; ii < nInputs; ii++)
            vecBsX[count*nInputs+ii] = XIn[jj*nInputs+ii];
          vecBsY[count] = YIn[jj*nOutputs+outputID];
          vecBsFlags[jj] = 1;
          count++;
        }
      }
    }
    //**/ create a response surface
    psConfig_.InteractiveSaveAndReset();
    status = faPtr->initialize(vecBsX.getDVector(),
                               vecBsY.getDVector());
    psConfig_.InteractiveRestore();
    if (status != 0)
    {
      printf("RSMSobol1 ERROR: in initializing response surface.\n");
      return -1;
    }
    //**/ evaluate the large sample
    faPtr->evaluatePoint(nSamp2,vecXX.getDVector(),
                         vecYY.getDVector());
    //**/ filter out unwanted samples
    count = 0;
    if (nConstr > 0)
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
    if (nConstr > 0)
    {
      printOutTS(PL_INFO,
           "RSMSobol1 INFO: %6.2f percent passes the contraints.\n",
           (double) count * 100.0 /((double) nSamp2));
    }
    if (count <= 1)
    {
      printf("RSMSobol1 ERROR: too few samples left after filtering\n");
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
    vecBsMeans[nn] = dmean;
    vecBsStds[nn]  = sqrt(dstds);
  }   

  //**/ now process all means and variances
  dmean = 0.0; 
  for (nn = 0; nn < ntimes; nn++) dmean += vecBsMeans[nn];
  dmean /= (double) ntimes;
  dstds = 0.0;
  if (ntimes > 1)  
  {
    for (nn = 0; nn < ntimes; nn++) 
      dstds += pow(vecBsMeans[nn]-dmean,2.0);
    dstds = sqrt(dstds/(double) (ntimes - 1));
  } 
  double smean=0, sstd=0;
  smean = 0.0; 
  for (nn = 0; nn < ntimes; nn++) smean += vecBsStds[nn];
  smean /= (double) ntimes;
  sstd = 0.0;
  if (ntimes > 1)  
  {
    for (nn = 0; nn < ntimes; nn++) 
      sstd += pow(vecBsStds[nn]-smean,2.0);
    sstd = sqrt(sstd/(double) (ntimes - 1));
  } 
  if (psConfig_.InteractiveIsOn())
  {
    if (ntimes > 1)
    {
      printOutTS(PL_INFO,
        "RSMSobol1: sample mean (s.d. of mean) = %10.3e (%10.3e)\n",
        dmean, dstds);
      printOutTS(PL_INFO,
        "RSMSobol1: std dev (s.d. of std dev)  = %10.3e (%10.3e)\n",
        smean, sstd);
    }
    else
    {
      printOutTS(PL_INFO,"RSMSobol1: sample mean = %10.3e\n",dmean);
      printOutTS(PL_INFO,"RSMSobol1: sample s.d. = %10.3e\n",smean);
    }
  }
  //**/ if sample std dev = 0, set it to 1 to avoid divide by 0
  if (smean == 0.0) smean = 1.0;
  //save mean & std
  outputMean_ = dmean;
  outputStd_  = smean;
  //cout << outputMean_ << ", " << outputStd_ << endl;

  //**/ ===============================================================
  //**/  use response surface to perform Sobol one parameter test
  //**/ ===============================================================
  //**/ ---------------------------------------------------------
  //**/ allocate space
  //**/ ---------------------------------------------------------
  psVector  vecLower2, vecUpper2, vecZZ, samplePtsND;
  psVector  vecInpMeans2, vecInpStdvs2, vecSamPts1D;
  psIVector vecInpFlags2; 
  PDFManager *pdfman1, *pdfman2;

  vecLower2.setLength(nInputs);
  vecUpper2.setLength(nInputs);
  vecSamPts1D.setLength(nLevels*nInputs);
  samplePtsND.setLength(nSubSamples*nInputs*nInputs);
  vecInpFlags2.setLength(nInputs-1);
  vecInpMeans2.setLength(nInputs-1);
  vecInpStdvs2.setLength(nInputs-1);

  //**/ ---------------------------------------------------------
  //**/ create sample for each input ==> vecSamPts1D, samplePtsND
  //**/ with sample size = nLevels and nSubSamples
  //**/ ---------------------------------------------------------
  if (nLevels > nSubSamples) 
  {
    vecSS.setLength(nLevels);
    vecZZ.setLength(nLevels);
  }
  else
  {
    vecSS.setLength(nSubSamples);
    vecZZ.setLength(nSubSamples);
  }
  for (ii = 0; ii < nInputs; ii++)
  {
    if (noPDF == 0)
    {
      //**/ create sample with pdf for input ii+1
      corMat.setDim(1,1);
      corMat.setEntry(0, 0, corMatp->getEntry(ii,ii));
      pdfman1 = new PDFManager();
      pdfman1->initialize(1,&pdfFlags[ii],&inputMeans[ii],
                           &inputStdevs[ii],corMat,NULL,NULL);
      vecLB.load(1, &xLower[ii]);
      vecUB.load(1, &xUpper[ii]);
      vecOut.setLength(nLevels);
      pdfman1->genSample(nLevels, vecOut, vecLB, vecUB);
      for (jj = 0; jj < nLevels; jj++)
        vecSamPts1D[ii*nLevels+jj] = vecOut[jj];
      delete pdfman1;
    }
    else
    {
      sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
      sampler->setInputBounds(1, &xLower[ii], &xUpper[ii]);
      sampler->setOutputParams(1);
      sampler->setSamplingParams(nLevels, 1, 0);
      //psConfig_.SamExpertModeSaveAndReset();
      sampler->initialize(0);
      //psConfig_.SamExpertModeRestore();
      tempV = vecSamPts1D.getDVector();
      sampler->getSamples(nLevels, 1, 1, &(tempV[ii*nLevels]), 
                   vecZZ.getDVector(), vecSS.getIVector());
      delete sampler;
    }

    //**/ create sample with pdf for the other inputs
    if (noPDF == 0)
    {
      corMat.setDim(nInputs-1, nInputs-1);
      for (jj = 0; jj < ii; jj++)
      {
        vecLower2[jj] = xLower[jj];
        vecUpper2[jj] = xUpper[jj];
        vecInpFlags2[jj] = pdfFlags[jj];
        vecInpMeans2[jj] = inputMeans[jj];
        vecInpStdvs2[jj] = inputStdevs[jj];
        //**/ correlation matrix is expected to be identity
        for (kk = 0; kk < ii; kk++)
          corMat.setEntry(jj, kk, corMatp->getEntry(jj,kk));
        for (kk = ii+1; kk < nInputs; kk++)
          corMat.setEntry(jj, kk-1, corMatp->getEntry(jj,kk));
      }
      for (jj = ii+1; jj < nInputs; jj++)
      {
        vecLower2[jj-1] = xLower[jj];
        vecUpper2[jj-1] = xUpper[jj];
        vecInpFlags2[jj-1] = pdfFlags[jj];
        vecInpMeans2[jj-1] = inputMeans[jj];
        vecInpStdvs2[jj-1] = inputStdevs[jj];
        //**/ correlation matrix is expected to be identity
        for (kk = 0; kk < ii; kk++)
          corMat.setEntry(jj-1, kk, corMatp->getEntry(jj,kk));
        for (kk = ii+1; kk < nInputs; kk++)
          corMat.setEntry(jj-1, kk-1, corMatp->getEntry(jj,kk));
      }
      pdfman2 = new PDFManager();
      pdfman2->initialize(nInputs-1,vecInpFlags2.getIVector(),
                    vecInpMeans2.getDVector(),vecInpStdvs2.getDVector(),
                    corMat,NULL,NULL);
      vecLB.load(nInputs-1, vecLower2.getDVector());
      vecUB.load(nInputs-1, vecUpper2.getDVector());
      vecOut.setLength(nSubSamples*(nInputs-1));
      pdfman2->genSample(nSubSamples, vecOut, vecLB, vecUB);
      for (jj = 0; jj < nSubSamples*(nInputs-1); jj++) 
        samplePtsND[ii*nSubSamples*nInputs+jj] = vecOut[jj];
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
      sampler->setSamplingParams(nSubSamples, 1, 0);
      sampler->initialize(0);
      tempV = samplePtsND.getDVector();
      sampler->getSamples(nSubSamples, nInputs-1, 1, 
                &(tempV[ii*nSubSamples*nInputs]), vecZZ.getDVector(), 
                vecSS.getIVector());
      delete sampler;
    }
  }
  //**/ some clean up
  if (ioPtr == NULL) delete corMatp;

  //**/ ---------------------------------------------------------
  //**/ compute first order Sobol indices
  //**/ ---------------------------------------------------------
  //**/ get ready
  int iL, sCnt, totalCnt, offset;
  psVector  vecVceMaxs,vecVceMins,vecVceMeds,vecVces,vecEcvs,vecVars;
  psVector  vecMeans,vecmSamPts;
  psIVector vecBins;

  vecYY.setLength(nSubSamples*nLevels);
  vecBsX.setLength(nSamples*nInputs);
  vecBsY.setLength(nSamples);
  vecBsFlags.setLength(nSamples);
  vecVceMaxs.setLength(nInputs);
  vecVceMins.setLength(nInputs);
  vecVceMeds.setLength(nInputs);
  vecVars.setLength(nLevels*ntimes);
  vecVces.setLength(nInputs*ntimes);
  vecEcvs.setLength(nInputs*ntimes);
  vecBins.setLength(nLevels*ntimes);
  vecMeans.setLength(nLevels*ntimes);
  vecmSamPts.setLength(nInputs*nSubSamples*nLevels);

  //**/ loop through each input
  for (ii = 0; ii < nInputs; ii++)
  {
    if (psConfig_.InteractiveIsOn())
      printOutTS(PL_INFO, "RSMSobol1: processing input %d\n",ii+1);

    //**/ use nLevels levels per input to compute variance of mean
    //**/ first, for each level, create nSubSamples ==> mSampletPts
    for (iL = 0; iL < nLevels; iL++)
    {
      offset = iL * nSubSamples * nInputs;
      //**/ concatenate the two sample ==> samplePts
      tempV = samplePtsND.getDVector();
      for (jj = 0; jj < nSubSamples; jj++)
      {
        oneSamplePt = &(tempV[ii*nSubSamples*nInputs+jj*(nInputs-1)]);
        for (kk = 0; kk < ii; kk++)
          vecmSamPts[offset+jj*nInputs+kk] = oneSamplePt[kk];
        for (kk = ii+1; kk < nInputs; kk++)
          vecmSamPts[offset+jj*nInputs+kk] = oneSamplePt[kk-1];
        vecmSamPts[offset+jj*nInputs+ii] = vecSamPts1D[ii*nLevels+iL];
      }
    }

    //**/ evaluate ==> vecmSamPts ==> vecYY
    for (nn = 0; nn < ntimes; nn++)
    {
      if (ntimes == 1)
      {
        for (ss = 0; ss < nSamples*nInputs; ss++) 
          vecBsX[ss] = XIn[ss];
        for (ss = 0; ss < nSamples; ss++) 
          vecBsY[ss] = YIn[ss*nOutputs+outputID];
      }
      else
      {
        for (ss = 0; ss < nSamples; ss++) vecBsFlags[ss] = 0;
        count = 0;
        while (count < frac * nSamples)
        {
          jj = PSUADE_rand() % nSamples;
          if (vecBsFlags[jj] == 0)
          {
            for (kk = 0; kk < nInputs; kk++)
              vecBsX[count*nInputs+kk] = XIn[jj*nInputs+kk];
            vecBsY[count] = YIn[jj*nOutputs+outputID];
            vecBsFlags[jj] = 1;
            count++;
          }
        }
      }
      //**/ initialize is needed only for bootstrapping
      if (ntimes > 1) 
        status = faPtr->initialize(vecBsX.getDVector(), 
                                   vecBsY.getDVector());

      if (psConfig_.InteractiveIsOn() && printLevel > 3)
        printOutTS(PL_INFO,
                   "RSMSobol1: Response surface evaluations\n");
      faPtr->evaluatePoint(nSubSamples*nLevels,
                   vecmSamPts.getDVector(),vecYY.getDVector());
      if (nConstr > 0)
      {
        tempV = vecmSamPts.getDVector();
        for (kk = 0; kk < nLevels*nSubSamples; kk++)
        {
          ddata = constrPtr->evaluate(&tempV[kk*nInputs],
                                      vecYY[kk],status);
          if (status == 0) vecYY[kk] = PSUADE_UNDEFINED;
        }
      }
   
      if (psConfig_.InteractiveIsOn() && printLevel > 3 &&
         (iL % (nLevels/10) == 0))
        printOutTS(PL_INFO,"RSMSobol1: Computing mean and std dev\n");
      //**/ compute mean for level iL
      for (iL = 0; iL < nLevels; iL++)
      {
        vecMeans[iL*ntimes+nn] = 0.0;
        sCnt = 0;
        for (jj = 0; jj < nSubSamples; jj++)
        {
          if (vecYY[iL*nSubSamples+jj] != PSUADE_UNDEFINED)
          {
            vecMeans[iL*ntimes+nn] += vecYY[iL*nSubSamples+jj];
            sCnt++;
          }
        }
        vecBins[iL*ntimes+nn] = sCnt;
        if (sCnt < nSubSamples/10 && printLevel >= 5)
          printOutTS(PL_INFO,
               "RSMSobol1 WARNING: subsample size = %d\n",sCnt);
        if (sCnt >= 1) vecMeans[iL*ntimes+nn] /= (double) sCnt;
        else           vecMeans[iL*ntimes+nn] = PSUADE_UNDEFINED;
        vecVars[iL*ntimes+nn] = 0.0;
        if (sCnt > 1)
        {
          for (jj = 0; jj < nSubSamples; jj++)
            if (vecYY[iL*nSubSamples+jj] != PSUADE_UNDEFINED)
              vecVars[iL*ntimes+nn] += pow(vecYY[iL*nSubSamples+jj]-
                                        vecMeans[iL*ntimes+nn],2.0);
          vecVars[iL*ntimes+nn] /= (double) sCnt;
        }
        else vecVars[iL*ntimes+nn] = PSUADE_UNDEFINED;
      }
    }

    //**/ count number of successes and compute vces
    for (nn = 0; nn < ntimes; nn++)
    {
      totalCnt = 0;
      for (iL = 0; iL < nLevels; iL++) 
        totalCnt += vecBins[iL*ntimes+nn];
      if (totalCnt == 0) 
      {
        printOutTS(PL_ERROR,"RSMSobol1 ERROR: no feasible region.\n");
        exit(1);
      }

      //**/ compute variances for each input
      ddata = 0.0;
      for (iL = 0; iL < nLevels; iL++) 
      {
        if (vecMeans[iL*ntimes+nn] != PSUADE_UNDEFINED)
          ddata += vecMeans[iL*ntimes+nn]*vecBins[iL*ntimes+nn]/
                   totalCnt;
      }
      vecVces[ii*ntimes+nn] = 0.0;
      for (iL = 0; iL < nLevels; iL++)
        if (vecMeans[iL*ntimes+nn] != PSUADE_UNDEFINED)
          vecVces[ii*ntimes+nn] += 
                 pow(vecMeans[iL*ntimes+nn]-ddata,2.0) * 
                 vecBins[iL*ntimes+nn] / totalCnt;
      vecEcvs[ii*ntimes+nn] = 0.0;
      for (iL = 0; iL < nLevels; iL++)
      {
        if (vecVars[iL*ntimes+nn] != PSUADE_UNDEFINED)
          vecEcvs[ii*ntimes+nn] += vecVars[iL*ntimes+nn]*
                     vecBins[iL*ntimes+nn]/totalCnt;
      }
    }
  }
  //**/ some clean up
  delete faPtr;
  if (constrPtr != NULL) delete constrPtr;

  //**/ compute max, min and others of the vces
  psVector vecVceMedu;
  vecVceMedu.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++)
  {
    vecVceMaxs[ii] = -PSUADE_UNDEFINED;
    vecVceMins[ii] =  PSUADE_UNDEFINED;
    vecVceMeds[ii] =  0.0;
    vecVceMedu[ii] =  0.0;
    for (nn = 0; nn < ntimes; nn++)
    {
      if (vecVces[ii*ntimes+nn] > vecVceMaxs[ii])
        vecVceMaxs[ii] = vecVces[ii*ntimes+nn];
      if (vecVces[ii*ntimes+nn] < vecVceMins[ii]) 
        vecVceMins[ii] = vecVces[ii*ntimes+nn];
      vecVceMeds[ii] += vecVces[ii*ntimes+nn];
      vecVceMedu[ii] += 
        (vecVces[ii*ntimes+nn]-vecEcvs[ii*ntimes+nn]/nSubSamples);
    }
    vecVceMeds[ii] /= (double) ntimes;
    vecVceMedu[ii] /= (double) ntimes;
    if (smean != 0)
    {
      vecVceMeds[ii] /= (smean * smean);
      vecVceMedu[ii] /= (smean * smean);
      vecVceMaxs[ii] /= (smean * smean);
      vecVceMins[ii] /= (smean * smean);
    }
    else 
      vecVceMeds[ii]=vecVceMedu[ii]=vecVceMaxs[ii]=vecVceMins[ii]=0;
  }

  //**/ ---------------------------------------------------------------
  //**/ print unsorted indices
  //**/ ---------------------------------------------------------------
  vecVces_.setLength(nInputs_);
  if (psConfig_.InteractiveIsOn()) printAsterisks(PL_INFO, 0);
  for (ii = 0; ii < nInputs; ii++)
  {
    if (psConfig_.InteractiveIsOn())
    {
      printOutTS(PL_INFO,
        "RSMSobol1: Normalized mean VCE for input %3d = %10.3e",
        ii+1, vecVceMeds[ii]);
      if (ntimes > 1)
        printOutTS(PL_INFO,",bounds = [%12.4e, %12.4e]\n",
                   vecVceMins[ii], vecVceMaxs[ii]);
      else printOutTS(PL_INFO, "\n");
    }
    //save vce
    vecVces_[ii] = vecVceMeds[ii];
  }
  if (psConfig_.InteractiveIsOn() && printLevel >= 2)
  {
    ddata = ddata2 = 0.0;
    for (ii = 0; ii < nInputs; ii++)
    {
      printOutTS(PL_INFO,
           "Unnormalized VCE for input %3d = %10.3e\n",ii+1,
           vecVceMeds[ii]*smean*smean);
      printOutTS(PL_INFO,
           "Unnormalized VCE for input %3d = %10.3e (unbiased)\n",
           ii+1, vecVceMedu[ii]*smean*smean);
      ddata  += vecVceMeds[ii]*smean*smean;
      ddata2 += vecVceMedu[ii]*smean*smean;
    }
    printOutTS(PL_INFO,"Sum of   biased VCEs = %10.3e\n",ddata);
    printOutTS(PL_INFO,"Sum of unbiased VCEs = %10.3e\n",ddata2);
    printOutTS(PL_INFO,"Total variance       = %10.3e\n",smean*smean);
  }

  //**/ ---------------------------------------------------------------
  //**/ return more detailed data
  //**/ ---------------------------------------------------------------
  pData *pObj = NULL;
  if (ioPtr != NULL)
  {
    pObj = ioPtr->getAuxData();
    if (ntimes == 1)
    {
      pObj->nDbles_ = nInputs;
      pObj->dbleArray_ = new double[nInputs];
      for (ii = 0; ii < nInputs; ii++)
        pObj->dbleArray_[ii] = vecVceMeds[ii] * smean * smean;
      pObj->dbleData_ = smean * smean;
    }
    else
    {
      pObj->nDbles_ = 3*nInputs;
      pObj->dbleArray_ = new double[nInputs*3];
      for (ii = 0; ii < nInputs; ii++)
        pObj->dbleArray_[ii] = vecVceMeds[ii] * smean * smean;
      for (ii = 0; ii < nInputs; ii++)
        pObj->dbleArray_[nInputs+ii] = vecVceMins[ii]*smean*smean;
      for (ii = 0; ii < nInputs; ii++)
        pObj->dbleArray_[2*nInputs+ii] = vecVceMaxs[ii]*smean*smean;
      pObj->dbleData_ = smean * smean;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ print sorted indices
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && printLevel >= 1)
  {
    for (ii = 0; ii < nInputs; ii++) vecMeans[ii] = (double) ii;
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"RSMSobol1: ordered normalized VCE : \n");
    printDashes(PL_INFO, 0);
    sortDbleList2(nInputs,vecVceMeds.getDVector(),
                  vecMeans.getDVector());
    for (ii = nInputs-1; ii >= 0; ii--)
       printOutTS(PL_INFO,
              "RSMSobol1: Normalized VCE for input %3d = %10.3e\n",
              (int) vecMeans[ii]+1,vecVceMeds[ii]);
    printAsterisks(PL_INFO, 0);
  }
  return 0.0;
}

// ************************************************************************
// perform analysis (for problems with general joint PDFs)
// (so it creates a large sample and performs ME)
// ------------------------------------------------------------------------
double RSMSobol1Analyzer::analyze2(aData &adata)
{
  int    ii, jj, iL, iR, status, currNLevels, nSteps, bin, totalCnt;
  double ddata, width;
  char   pdfFile[500], scatterFile[500], pString[500], winput1[500];
  FILE   *fp;
  FuncApprox    *faPtr;
  RSConstraints *constrPtr;
  psVector      vecIn, vecOut, vecUB, vecLB, vecXX, vecYY, vecY;
  PDFManager    *pdfman;

  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"RSMSobol1: Since joint PDFs have ");
  printOutTS(PL_INFO,"been specified, a different main\n");
  printOutTS(PL_INFO,"           effect analysis is to be performed.\n");
  //**/ ---------------------------------------------------------------
  //**/ extract test data
  //**/ ---------------------------------------------------------------
  printEquals(PL_INFO, 0);
  int nInputs    = adata.nInputs_;
  int nOutputs   = adata.nOutputs_;
  int nSamples   = adata.nSamples_;
  int outputID   = adata.outputID_;
  int printLevel = adata.printLevel_;
  double *xLower = adata.iLowerB_;
  double *xUpper = adata.iUpperB_;
  double *XIn    = adata.sampleInputs_;
  double *YIn    = adata.sampleOutputs_;
  PsuadeData *ioPtr = adata.ioPtr_;
  vecY.setLength(nSamples);
  for (ii = 0; ii < nSamples; ii++) vecY[ii] = YIn[ii*nOutputs+outputID];

  //**/ ---------------------------------------------------------------
  //**/ set up constraint filters
  //**/ ---------------------------------------------------------------
  constrPtr = new RSConstraints();
  constrPtr->genConstraints(ioPtr);

  //**/ ---------------------------------------------------------------
  //**/ build response surface 
  //**/ ---------------------------------------------------------------
  faPtr = genFAInteractive(ioPtr, 0);
  if(faPtr == NULL)
  {
    printOutTS(PL_INFO, "faPtr is NULL in file %s line %d aborting. \n", 
               __FILE__, __LINE__);
    abort();
  }
  else status = faPtr->initialize(XIn, vecY.getDVector());

  //**/ ---------------------------------------------------------------
  //**/  get internal parameters from users (nSubSamples, nLevels)
  //**/ ---------------------------------------------------------------
  int nSubSamples=2000, nLevels=1000;
  printAsterisks(PL_INFO, 0);
  if (psConfig_.AnaExpertModeIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"* RSMSobol1 creates one sample ");
    printOutTS(PL_INFO,"of size M for each of the K levels of\n");
    printOutTS(PL_INFO,"* each input. Therefore, the total ");
    printOutTS(PL_INFO,"sample size is N = M * K * nInputs\n");
    printf("* NOW, nInputs = %d\n", nInputs);
    printf("* As a user, please decide on M and K.\n\n");
    printEquals(PL_INFO, 0);

    snprintf(pString,100,"Enter M (suggestion: 100-10000) : ");
    nSubSamples = getInt(100, 100000, pString);
    if (nSubSamples > 100000)
      printOutTS(PL_INFO, "An M of %d may take very long time.\n", 
                 nSubSamples);

    snprintf(pString,100,"Enter K (suggestion: 1000 - 10000) : ");
    nLevels = getInt(1000, 10000, pString);
    if (nLevels > 5000)
      printOutTS(PL_INFO, "* A K of %d may take very long time.\n", 
                 nLevels);
    printAsterisks(PL_INFO, 0);
  }
  else
  {
    nSubSamples = 2000;
    nLevels     = 1000;
    printOutTS(PL_INFO,"* RSMSobol1: default M = %d.\n", nSubSamples);
    printOutTS(PL_INFO,"* RSMSobol1: default K = %d.\n", nLevels);
    printOutTS(PL_INFO,
               "* To change settings, rerun with ana_expert on.\n");
  }
  printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ graphical output option
  //**/ ---------------------------------------------------------------
  int pdfFileFlag=0;
  if (psConfig_.MasterModeIsOn())
  {
    printOutTS(PL_INFO,
      "* RSMSobol1 will create a sample for basic statistics.\n");
    printOutTS(PL_INFO,"* You have the option to plot the PDF.\n");
    snprintf(pString,100,"Create a pdf (probability) bar graph? (y or n) ");
    getString(pString, winput1);
    if (winput1[0] == 'y')
    {
      snprintf(pString,100,"Enter the file name for the bar graph : ");
      getString(pString, pdfFile);
      pdfFile[strlen(pdfFile)-1] = '\0';
      fp = fopen(pdfFile, "w");
      if (fp != NULL)
      {
        fclose(fp);
        if (plotMatlab()) pdfFileFlag = 1; 
      }
      else 
      {
        printOutTS(PL_INFO,"RSMSobol1 WARNING: cannot open file %s\n", 
                   pdfFile);
        pdfFileFlag = 0; 
      }
    }
    printEquals(PL_INFO, 0);
  }
  else pdfFileFlag = 0;

  //**/ ---------------------------------------------------------------
  //**/ graphical output option
  //**/ ---------------------------------------------------------------
  int scatterFileFlag=0;;
  if (psConfig_.MasterModeIsOn())
  {
    printOutTS(PL_INFO,
        "RSMSobol1 creates many samples for Sobol1 analysis.\n");
    printOutTS(PL_INFO,
        "You have the option to plot the sample data.\n");
    snprintf(pString,100,"Create a scatter plot for RSMSobol1? (y or n) ");
    getString(pString, winput1);
    if (winput1[0] == 'y')
    {
      snprintf(pString,100,"Enter the file name for the scatter plot : ");
      getString(pString, scatterFile);
      scatterFile[strlen(scatterFile)-1] = '\0';
      fp = fopen(scatterFile, "w");
      if (fp != NULL)
      {
        fclose(fp);
        scatterFileFlag = 1;
      }
      else 
      {
        printOutTS(PL_ERROR, "RSMSobol1 ERROR: cannot open file %s\n",
               scatterFile); 
        scatterFileFlag = 0;
      }
    }
    printEquals(PL_INFO, 0);
  }
  else scatterFileFlag = 0;

  //**/ ---------------------------------------------------------------
  //**/  use response surface to compute mean and variance
  //**/ ---------------------------------------------------------------
  //**/ ----------------------------------
  //**/ use a large sample size
  //**/ ----------------------------------
  int nSamp = nSubSamples * nLevels;
  if (nSamp < 100000) nSamp = 100000;
  if (printLevel > 1)
  {
    printOutTS(PL_INFO,
      "* RSMSobol1 INFO: Creating a sample for basic statistics.\n");
    printOutTS(PL_INFO,
      "*                 Sample size = %d\n",nSamp);
  }
  //**/ ----------------------------------
  //**/ allocate space
  //**/ ----------------------------------
  vecXX.setLength(nSamp*nInputs);
  vecYY.setLength(nSamp);
  //**/ ----------------------------------
  //**/ create a sample
  //**/ ----------------------------------
  pdfman = new PDFManager();
  pdfman->initialize(ioPtr);
  vecLB.load(nInputs, xLower);
  vecUB.load(nInputs, xUpper);
  vecOut.setLength(nSamp*nInputs);
  pdfman->genSample(nSamp, vecOut, vecLB, vecUB);
  for (ii = 0; ii < nSamp*nInputs; ii++) vecXX[ii] = vecOut[ii];
  //**/ ----------------------------------
  //**/ evaluate 
  //**/ ----------------------------------
  printOutTS(PL_INFO,
     "RSMSobol1: Response surface evaluation begins...\n");
  faPtr->evaluatePoint(nSamp,vecXX.getDVector(),vecYY.getDVector());
  printOutTS(PL_INFO,
     "RSMSobol1: Response surface evaluation ends...\n");
  //**/ ----------------------------------
  //**/ apply filters
  //**/ ----------------------------------
  double *oneSamplePt;
  int count = 0;
  for (ii = 0; ii < nSamp; ii++)
  {
    oneSamplePt = &(vecXX[ii*nInputs]);
    status = 1;
    if (constrPtr != NULL)
      ddata = constrPtr->evaluate(oneSamplePt,vecYY[ii],status);
    if (status == 0) vecYY[ii] = PSUADE_UNDEFINED;
  }
  //**/ ----------------------------------
  //**/ compute statistics
  //**/ ----------------------------------
  double dmean=0;
  int sCnt = 0;
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
       "RSMSobol1 ERROR: too few samples that satisify");
    printOutTS(PL_ERROR,
       "the constraints (%d out of %d).\n", sCnt, nSamp);
    delete faPtr;
    delete pdfman;
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
  printOutTS(PL_INFO,
       "* RSMSobol1: sample mean     (based on %d points) = %10.3e\n",
       sCnt,dmean);
  printOutTS(PL_INFO,
       "* RSMSobol1: sample std dev  (based on %d points) = %10.3e\n",
       sCnt, sqrt(variance));
  printOutTS(PL_INFO,
       "* RSMSobol1: sample variance (based on %d points) = %10.3e\n",
       sCnt, variance);
  if (variance == 0.0) variance = 1.0;
  delete pdfman;
  printAsterisks(PL_INFO, 0);
  //**/ ----------------------------------
  //**/ create matlab file for PDF
  //**/ ----------------------------------
  if (pdfFileFlag == 1 && (fp = fopen(pdfFile, "w")) != NULL)
  {
    if (plotScilab())
         fprintf(fp,"// sample PDF based on the response surface.\n");
    else fprintf(fp,"%% sample PDF based on the response surface.\n");
    fprintf(fp, "A = [\n");
    nSteps = 1;
    if (nSamp > 2000) nSteps = nSamp / 1000;
    for (ii = 0; ii < nSamp; ii++)
    {
      if (vecYY[ii] != PSUADE_UNDEFINED)
      {
        if ((pdfFileFlag == 1) && (ii % nSteps) == 0) 
          fprintf(fp,"   %e\n", vecYY[ii]);
      }
    }
    fprintf(fp, "];\n");
    fprintf(fp, "amax = max(A);\n");
    fprintf(fp, "amin = min(A);\n");
    fprintf(fp, "step = (amax - amin) / 10;\n");
    fprintf(fp, "numa = size(A,1);\n");
    fprintf(fp, "B = zeros(numa,1);\n");
    fprintf(fp, "for ii = 1 : numa\n");
    fprintf(fp, "   B(ii) = ceil(((A(ii) - amin))/step);\n");
    fprintf(fp, "end;\n");
    fprintf(fp, "X = zeros(10,1);\n");
    fprintf(fp, "Y = zeros(10,1);\n");
    fprintf(fp, "for ii = 1 : 10\n");
    fprintf(fp, "   X(ii) = amin + (ii - 1 + 0.5) * step;\n");
    fprintf(fp, "   [ia,ja,aa] = find(B==ii);\n");
    fprintf(fp, "   Y(ii) = length(ia);\n");
    fprintf(fp, "end;\n");
    fprintf(fp, "Y = Y / numa;\n");
    fprintf(fp, "bar(X,Y,1.0)\n");
    fwritePlotXLabel(fp, "Output Values");
    fwritePlotYLabel(fp, "Probabilities");
    fwritePlotAxes(fp);
    fclose(fp);
  }
  //**/ ------------------------
  //**/ create matlab scatter plot 
  //**/ ------------------------
  fp = NULL;
  if (scatterFileFlag == 1 && 
      (fp = fopen(scatterFile, "w")) != NULL)
  {
    //fp = fopen(scatterFile, "w");
    fprintf(fp, "%% scatter plot for RSMSobol1 sample\n");
    fprintf(fp, "A = [ \n");
    nSteps = 1;
    if (nSamp > 2000) nSteps = nSamp / 1000;
    for (ii = 0; ii < nSamp; ii++)
    {
      if (vecYY[ii] != PSUADE_UNDEFINED)
      {
        if ((pdfFileFlag == 1) && (ii % nSteps) == 0) 
          fprintf(fp,"   %e\n", vecYY[ii]);
      }
    }
    fprintf(fp, "];\n");
    fprintf(fp, "for ii = 1 : %d\n", nInputs);
    fprintf(fp, "   plot(A(:,ii),A(:,%d),'x')\n", nInputs+1);
    fwritePlotXLabel(fp, "['input ' int2str(ii)]");
    fwritePlotYLabel(fp, "Y");
    fwritePlotAxes(fp);
    fprintf(fp, "   disp('Press enter to continue')\n");
    fprintf(fp, "   pause\n");
    fprintf(fp, "end\n");
    fclose(fp);
  }

  //**/ ---------------------------------------------------------------
  //**/  use the same sample to perform Sobol one parameter test
  //**/ ---------------------------------------------------------------
  psVector  vecVces, vecVars, vecEcvs, vecMeans;
  psIVector vecBins;
  vecMeans.setLength(nLevels);
  vecVars.setLength(nLevels);
  vecVces.setLength(nInputs);
  vecEcvs.setLength(nInputs);
  vecBins.setLength(nLevels);

  //**/ diagnostics
#if 0
  FILE *fp2 = fopen("100k.std","w");
  for (jj = 0; jj < 100000; jj++)
  {
    for (ii = 0; ii < nInputs; ii++)
    fprintf(fp2,"%e ",vecXX[jj*nInputs+ii]);
    fprintf(fp2,"%e\n",vecYY[jj]);
  }
  fclose(fp2);
#endif

  //**/ ------------------------
  //**/ loop through each input
  //**/ ------------------------
  for (ii = 0; ii < nInputs; ii++)
  {
    printOutTS(PL_INFO, "RSMSobol1: Processing input %d\n", ii+1);

    //**/ analyze using 4 levels to inspect convergence
    currNLevels = nLevels / 8;
    for (iR = 0; iR < 4; iR++)
    {
      //**/ initialize registers
      width = (xUpper[ii] - xLower[ii]) / currNLevels;
      for (iL = 0; iL < currNLevels; iL++)
      {
        vecBins[iL] = 0;
        vecMeans[iL] = 0.0;
        vecVars[iL] = 0.0;
      }
      //**/ compute the conditional means (find which bin)
      for (jj = 0; jj < nSamp; jj++)
      {
        if (vecYY[jj] != PSUADE_UNDEFINED)
        {
          ddata = vecXX[jj*nInputs+ii];
          bin = (int) ((ddata - xLower[ii]) / width); 
          if (bin == currNLevels) bin = currNLevels - 1;
          vecMeans[bin] += vecYY[jj];
          vecBins[bin]++;
        }
      }
      for (iL = 0; iL < currNLevels; iL++)
      {
        if (vecBins[iL] < nSubSamples/10 && printLevel >= 5)
          printOutTS(PL_INFO,
               "RSMSobol1 WARNING: subsample size = %d\n",vecBins[iL]);
        if (vecBins[iL] >= 1) vecMeans[iL] /= (double) vecBins[iL];
        else                  vecMeans[iL] = PSUADE_UNDEFINED;
        if (printLevel > 3)
        {
          printOutTS(PL_INFO,"RSMSobol1: input %d :\n", ii+1);
          printOutTS(PL_INFO,
               "  refinement %2d, level %3d, size %d), mean = %e\n",
               iR, iL, nSubSamples, vecMeans[iL]);
        }
      }
      //**/ compute the conditional variances
      for (jj = 0; jj < nSamp; jj++)
      {
        ddata = vecXX[jj*nInputs+ii];
        bin = (int) ((ddata - xLower[ii]) / width); 
        if (bin == currNLevels) bin = currNLevels - 1;
        if (vecYY[jj] != PSUADE_UNDEFINED)
          vecVars[bin] += pow(vecYY[jj]-vecMeans[bin], 2.0); 
      }
      for (iL = 0; iL < currNLevels; iL++)
      {
        if (vecBins[iL] >= 1) vecVars[iL] /= (double) vecBins[iL];
        else                  vecVars[iL] = PSUADE_UNDEFINED;
      }

#if 1
      //**/ Nov 2021: this is needed
      totalCnt = 0;
      for (iL = 0; iL < currNLevels; iL++) totalCnt += vecBins[iL];
      if (totalCnt == 0) 
      {
        printOutTS(PL_ERROR, "RSMSobol1 ERROR: no feasible region.\n");
        exit(1);
      }
      //**/ compute variances for each input
      ddata = 0.0;
      for (iL = 0; iL < currNLevels; iL++) 
      {
        if (vecMeans[iL] != PSUADE_UNDEFINED)
          ddata += vecMeans[iL] * vecBins[iL] / totalCnt;
      }
      //**/ compute variances for each input
      vecVces[ii] = 0.0;
      for (iL = 0; iL < currNLevels; iL++)
        if (vecMeans[iL] != PSUADE_UNDEFINED)
          vecVces[ii] += (vecMeans[iL]-ddata) * (vecMeans[iL]-ddata) * 
                        vecBins[iL] / totalCnt;
      vecEcvs[ii] = 0.0;
      for (iL = 0; iL < currNLevels; iL++)
      {
        if (vecVars[iL] != PSUADE_UNDEFINED)
          vecEcvs[ii] += vecVars[iL] * vecBins[iL] / totalCnt;
      }
#else
      //**/ Nov 2021: this section is incorrect
      //**/ I don't remember why this was used
      totalCnt = 0;
      for (iL = 0; iL < currNLevels; iL++) 
        if (vecBins[iL] > 0) totalCnt++;
      if (totalCnt == 0) 
      {
        printOutTS(PL_ERROR, 
            "RSMSobol1 ERROR: no feasible region.\n");
        exit(1);
      }

      //**/ compute variances for each input
      ddata = 0.0;
      for (iL = 0; iL < currNLevels; iL++) 
        if (vecMeans[iL] != PSUADE_UNDEFINED) ddata += vecMeans[iL];
      ddata /= (double) totalCnt;

      vecVces[ii] = 0.0;
      for (iL = 0; iL < currNLevels; iL++)
        if (vecMeans[iL] != PSUADE_UNDEFINED)
          vecVces[ii] += (vecMeans[iL]-ddata) * (vecMeans[iL]-ddata); 
      vecVces[ii] /= (double) totalCnt;

      vecEcvs[ii] = 0.0;
      for (iL = 0; iL < currNLevels; iL++)
        if (vecVars[iL] != PSUADE_UNDEFINED) 
          vecEcvs[ii] += vecVars[iL];
      vecEcvs[ii] /= (double) totalCnt;
#endif

      if (printLevel > 2 || iR == 3)
      {
        printOutTS(PL_INFO,
             "VCE(%3d) (refinement=%3d) = %10.3e, (normalized) = %10.3e\n",
             ii+1, iR, vecVces[ii], vecVces[ii]/variance);
      }
      if (iR == 3) vecVces[ii] = vecVces[ii] / variance;

      if (printLevel > 3)
        printOutTS(PL_INFO, 
           "ECV(%3d) (refinement=%3d) = %10.3e, (normalized) = %10.3e\n",
           ii+1, iR, vecEcvs[ii], vecEcvs[ii]/variance);

      currNLevels *= 2;
    }
  }
  printAsterisks(PL_INFO, 0);
  for (ii = 0; ii < nInputs; ii++)
    printOutTS(PL_INFO,"RSMSobol1: Normalized VCE for input %3d = %10.3e\n",
               ii+1,vecVces[ii]);
  printAsterisks(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ clean up
  //**/ ---------------------------------------------------------------
  delete faPtr;
  if (constrPtr != NULL) delete constrPtr;
  return 0.0;
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
  if (vecVces_.length() <= ind)
  {
    printf("RSMSobol1 ERROR: get_vce has not value.\n");
    return 0.0;
  }
  return vecVces_[ind];
}

