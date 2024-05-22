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
// Functions for the class RSMEntropy2  
// AUTHOR : CHARLES TONG
// DATE   : 2023
//**/ ---------------------------------------------------------------------
//**/ Entropy-based pairwise effect analysis 
//**/ For theory, refer to RSMEntropy1Analyzer.cpp
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "PsuadeUtil.h"
#include "sysdef.h"
#include "psMatrix.h"
#include "pData.h"
#include "RSMEntropy2Analyzer.h"
#include "ProbMatrix.h"
#include "Sampling.h"
#include "PDFManager.h"
#include "PDFNormal.h"
#include "Psuade.h"
#include "PsuadeData.h"
#include "FunctionInterface.h"
#include "PsuadeConfig.h"
#include "PrintingTS.h"
#include "FuncApprox.h"
#include "psMatrix3D.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))
#define DO_ENTROPY 0
#define DO_DELTA   1

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
RSMEntropy2Analyzer::RSMEntropy2Analyzer() : Analyzer(),nInputs_(0),
                          outputMean_(0), outputStd_(0), outputEntropy_(0),
                          useSimulator_(0), entropyDelta_(0), M1_(4000),
                          M2_(4000), useRS_(1)
{
  setName("RSMEntropy2");
  adaptive_ = 1;       /* default: adaptive histogramming is on */
  nLevels_  = 50;      /* number of levels used for binning */
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
RSMEntropy2Analyzer::~RSMEntropy2Analyzer()
{
}

// ************************************************************************
// perform analysis (intended for library call, that users use this 
// function and this function aggregates information and call analyze with
// aData)
// ------------------------------------------------------------------------
void RSMEntropy2Analyzer::analyze(int nInps, int nSamp, double *lbs, 
                                  double *ubs, double *X, double *Y)
{
  //**/ check data validity
  char pString[1000];
  snprintf(pString, 25, " RSMEntropy2 analyze (X)");
  checkDbleArray(pString, nSamp*nInps, X);
  snprintf(pString, 25, " RSMEntropy2 analyze (Y)");
  checkDbleArray(pString, nSamp, Y);
  snprintf(pString, 25, " RSMEntropy2 analyze (LBs)");
  checkDbleArray(pString, nInps, lbs);
  snprintf(pString, 25, " RSMEntropy2 analyze (UBs)");
  checkDbleArray(pString, nInps, ubs);
  //**/ create aData structure before calling analyze function
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
  //**/ call analyze function
  analyze(adata);
  //**/ clean up
  adata.iLowerB_ = NULL;
  adata.iUpperB_ = NULL;
  adata.sampleInputs_  = NULL;
  adata.sampleOutputs_ = NULL;
}

// ************************************************************************
// perform analysis 
// ------------------------------------------------------------------------
double RSMEntropy2Analyzer::analyze(aData &adata)
{
  //**/ ===============================================================
  //**/ display header 
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    if (entropyDelta_ == DO_ENTROPY)
      printOutTS(PL_INFO,
         "*          Two-Parameter Entropy (M.I.) Analysis \n");
    else if (entropyDelta_ == DO_DELTA)
      printOutTS(PL_INFO,
         "*          Two-Parameter Delta Analysis\n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,
       "* TO GAIN ACCESS TO DIFFERENT OPTIONS: SET\n");
    printOutTS(PL_INFO,
       "* - ana_expert to finetune internal parameters\n");
    printOutTS(PL_INFO,
       "*   (to adjust binning resolution or select adaptive ");
    printOutTS(PL_INFO, "algorithm)\n");
    printOutTS(PL_INFO,
       "* - rs_expert mode to finetune response surfaces\n");
    printOutTS(PL_INFO,
       "* - printlevel to display more information\n");
    printOutTS(PL_INFO,
       "* Or, use configure file to finetune parameters\n");
    printEquals(PL_INFO, 0);
  }
 
  //**/ ===============================================================
  //**/ extract sample data and information, check errors 
  //**/ ---------------------------------------------------------------
  int nInputs    = adata.nInputs_;
  nInputs_       = nInputs;
  int nOutputs   = adata.nOutputs_;
  int nSamples   = adata.nSamples_;
  double *YIn    = adata.sampleOutputs_;
  int outputID   = adata.outputID_;

  if (nInputs <= 0 || nOutputs <= 0)
  {
    printOutTS(PL_ERROR, "RSEntropy2 ERROR: Invalid arguments.\n");
    printOutTS(PL_ERROR, "   nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR, "   nOutputs = %d\n", nOutputs);
    return PSUADE_UNDEFINED;
  }
  if (useSimulator_ == 0 && nSamples <= 0)
  {
    printOutTS(PL_ERROR, "RSEntropy2 ERROR: Invalid arguments.\n");
    printOutTS(PL_ERROR, "   nSamples = %d <= 0\n", nSamples);
    return PSUADE_UNDEFINED;
  }
  if (nInputs < 2)
  {
    printOutTS(PL_INFO,
      "RSEntropy2 INFO: This method is for nInputs >= 2.\n");
    return PSUADE_UNDEFINED;
  }
  if (outputID >= nOutputs || outputID < 0)
  {
    printOutTS(PL_ERROR,
      "RSEntropy2 ERROR: Invalid output ID (%d).\n",outputID);
    return PSUADE_UNDEFINED;
  }

  int ii, status = 0;
  for (ii = 0; ii < nSamples; ii++)
    if (YIn[nOutputs*ii+outputID] > 0.9*PSUADE_UNDEFINED) 
      status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR,"RSEntropy2 ERROR: Some outputs are ");
    printOutTS(PL_ERROR,"undefined. Please prune all the\n");
    printOutTS(PL_ERROR,"                  undefined sample ");
    printOutTS(PL_ERROR,"points and re-run.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ===============================================================
  //**/ ask users to choose which algorithm
  //**/ ===============================================================
  if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn() &&
      entropyDelta_ != DO_DELTA)
  {
    char pString[101], winput[101];
    printf("Two algorithms are provided for computing entropy: \n");
    printf("(1) Uniform partitioning in histogramming, and\n");
    printf("(2) Adaptive partitioning in histogramming.\n");
    printf("Adaptive partitioning may give more accurate ");
    printf("result for small samples,\n");
    printf("but if the samples generated from response surface ");
    printf("interpolations, the\n");
    printf("uniform partitioning may give comparable result. ");
    printf("Both algorithms have\n");
    printf("been implemented here. The default is adaptive ");
    printf("partitioning.\n");
    snprintf(pString,100,
             "Use (1) uniform or (2) adaptive algorithm ? (1 or 2) ");
    ii = getInt(1, 2, pString);
    if (ii  == 1) adaptive_ = 0;
    else          adaptive_ = 1;
  }

  //**/ ===============================================================
  //**/ get internal parameters from users
  //**/ This method uses histogramming by dividing the X/Y space into K 
  //**/ bins (K=nLevels).
  //**/ ---------------------------------------------------------------
  char pString[500], *cString, winput1[500], winput2[500];
  if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn())
  {
    printAsterisks(PL_INFO, 0);
    if (entropyDelta_ != DO_DELTA)
    {
      printOutTS(PL_INFO,"* RSEntropy2 uses histogram ");
      printOutTS(PL_INFO,"bins to represent probabilities.\n");
      printOutTS(PL_INFO,
        "* The default number of bins K = %d\n",nLevels_);
      printOutTS(PL_INFO,
        "* To change K, re-run with ana_expert mode on.\n");
      printOutTS(PL_INFO,"* Users may select a different K.\n");
      printOutTS(PL_INFO,
        "* NOTE: Large K may be more accurate but take longer time.\n");
      printEquals(PL_INFO, 0);
      snprintf(pString,100,"Enter K (suggestion: 10 - 100) : ");
      nLevels_ = getInt(10, 500, pString);
      printEquals(PL_INFO, 0);
    }
    printOutTS(PL_INFO,
      "* RSEntropy2/Delta2 use 2 samples: 1 for input pair, ");
    printOutTS(PL_INFO,"1 for the rest.\n");
    printOutTS(PL_INFO,
      "Default sample size for the first  sample (M1) = %d\n",M1_);
    printOutTS(PL_INFO,
      "Default sample size for the second sample (M2) = %d\n",M2_);
    snprintf(pString,100,
      "Select a new M1 (suggestion: 1000 - 10000) : ");
    M1_ = getInt(1000, 50000, pString);
    snprintf(pString,100,
      "Select a new M2 (suggestion: 1000 - 10000) : ");
    M2_ = getInt(1000, 50000, pString);
    printAsterisks(PL_INFO, 0);
  }
  else if (entropyDelta_ != DO_DELTA)
  {
    //**/ alternatively, binning information can be obtained in the
    //**/ configuration object
    cString = psConfig_.getParameter("ana_entropy2_nlevels");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nLevels_);
      printOutTS(PL_INFO,
         "RSEntropy2/Delta2 INFO: K from configure object = %d\n",
         nLevels_);
      if (nLevels_ < 10)
      {
        printOutTS(PL_INFO,"             K should be >= 10.\n");
        printOutTS(PL_INFO,"             Set K = 10.\n");
        nLevels_ = 10;
      }
    }
    cString = psConfig_.getParameter("ana_entropy2_M1");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &M1_);
      printOutTS(PL_INFO,
        "RSEntropy2/Delta2 INFO: M1 from configure object = %d\n",M1_);
      if (M1_ < 1000) M1_ = 1000;
    }
    cString = psConfig_.getParameter("ana_entropy2_M2");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &M2_);
      printOutTS(PL_INFO,
        "RSEntropy2/Delta2 INFO: M2 from configure object = %d\n",M2_);
      if (M2_ < 1000) M2_ = 1000;
    }
  }

  //**/ ===============================================================
  //**/ Call different analyzers based on entropyDelta_
  //**/ ---------------------------------------------------------------
  analyzeEntDelta(adata);
  return 0.0;
}

// ************************************************************************
// perform entropy or delta analysis 
// ------------------------------------------------------------------------
double RSMEntropy2Analyzer::analyzeEntDelta(aData &adata)
{
  //**/ ===============================================================
  //**/ extract sample data and information
  //**/ ---------------------------------------------------------------
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

  //**/ ===============================================================
  //**/ extract input PDF information (if none, set all to none - 0)
  //**/ (these are needed for creating samples later) 
  //**/ ---------------------------------------------------------------
  int    *pdfFlags    = adata.inputPDFs_;
  double *inputMeans  = adata.inputMeans_;
  double *inputStdevs = adata.inputStdevs_;
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
  //**/ if other than uniform PDF, set hasPDF=1 
  //**/ Also, check for S PDFs and flag error
  int ii, jj, hasPDF=0, samPDF=0;
  if (pdfFlags != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (pdfFlags[ii] != 0) hasPDF = 1;
    for (ii = 0; ii < nInputs; ii++)
      if (pdfFlags[ii] == PSUADE_PDF_SAMPLE) samPDF = 1;
  }
  if (psConfig_.InteractiveIsOn())
  {
    if (samPDF == 1)
    {
      printOutTS(PL_INFO,
        "RSEntropy2/Delta2 INFO: Inputs have S-type (sample) PDF.\n");
      printOutTS(PL_INFO,
        "           ===> Cannot perform entropy or delta analysis.\n");
      return PSUADE_UNDEFINED;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ get or create correlation matrix
  //**/ (needed for creating samples later) 
  //**/ ---------------------------------------------------------------
  PsuadeData *ioPtr = adata.ioPtr_;
  psMatrix *corMatp=NULL;
  int corFlag=0;
  if (ioPtr == NULL)
  {
    //**/ set default correlation to be identity matrix
    corMatp = new psMatrix();
    corMatp->setDim(nInputs, nInputs);
    for (ii = 0; ii < nInputs; ii++) corMatp->setEntry(ii,ii,1.0e0);
  } 
  else
  {
    //**/ detect if correlation matrix is not diagonal
    pData pCorMat;
    ioPtr->getParameter("input_cor_matrix", pCorMat);
    corMatp = (psMatrix *) pCorMat.psObject_;
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = 0; jj < ii; jj++)
      {
        if (corMatp->getEntry(ii,jj) != 0.0)
        {
          //if (psConfig_.InteractiveIsOn())
          //{
          //  printOutTS(PL_INFO, 
          //    "RSMEntropy2/Delta2 WARNING: Correlated inputs detected.\n");
          //  printOutTS(PL_INFO, 
          //    "INFO: Correlation will be ignored in analyis.\n");
          //}
          corFlag = 1;
          break;
        }
      }
      if (corFlag == 1) break;
    }
  }
  char pString[1000], winput[1000];
  if (corFlag > 0)
  {
    printOutTS(PL_INFO,
         "RSEntropy2/Delta1 INFO: Correlated inputs detected.\n");
    snprintf(pString, 100,
         "Correlation will be ignored in analyis. Continue? (y or n) ");
    getString(pString, winput);
    if (winput[0] != 'y')
    {
      VecEnt2_.setLength(nInputs*nInputs);
      VecDelta2_.setLength(nInputs*nInputs);
      printf("RSEntropy2 INFO: Terminate\n");
      return 0.0;
    }
  }

  //**/ ---------------------------------------------------------
  //**/ generate a large sample for computing basic statistics
  //**/ ==> vecXX (with sample size nSamp2) 
  //**/ ---------------------------------------------------------
  int nSam1 = M1_, nSam2 = M2_;
  int nSamp2 = nSam1 * nSam2;
  if (nSamp2 > 2000000) nSamp2 = 2000000;
  psVector vecXX, vecLB, vecUB;
  PDFManager *pdfman = new PDFManager();
  pdfman->initialize(nInputs,pdfFlags,inputMeans,inputStdevs,
                     *corMatp,NULL,NULL);
  vecLB.load(nInputs, xLower);
  vecUB.load(nInputs, xUpper);
  vecXX.setLength(nSamp2*nInputs);
  pdfman->genSample(nSamp2, vecXX, vecLB, vecUB);
  delete pdfman;

  //**/ ---------------------------------------------------------
  //**/ create a function evaluator or a response surface 
  //**/ ---------------------------------------------------------
  int  ss, status, rstype=-1;
  psVector vecY;
  FunctionInterface *funcIO = NULL;
  FuncApprox *faPtr = NULL;
  if (useSimulator_ == 1)
  {
    funcIO = createFunctionInterface(adata.ioPtr_);
    if (funcIO == NULL)
    {
      printOutTS(PL_INFO,
           "RSEntropy2 ERROR: No dataIO object found.\n");
      printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
      exit(1);
    }
  }
  else
  {
    while (rstype < 0 || rstype >= PSUADE_NUM_RS)
    {
      printf("Select response surface. Options are: \n");
      writeFAInfo(0);
      strcpy(pString, "Choose response surface: ");
      rstype = getInt(0, PSUADE_NUM_RS, pString);
    }
    faPtr = genFA(rstype, nInputs, 0, nSamples);
    faPtr->setBounds(xLower, xUpper);
    faPtr->setOutputLevel(0);
    vecY.setLength(nSamples);
    for (ss = 0; ss < nSamples; ss++) 
      vecY[ss] = YIn[ss*nOutputs+outputID];
    psConfig_.InteractiveSaveAndReset();
    status = faPtr->initialize(XIn,vecY.getDVector());
    psConfig_.InteractiveRestore();
    if (status != 0)
    {
      printf("RSEntropy2/Delta2 ERROR: In building RS.\n");
      printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
      exit(1);
    }
  }

  //**/ ---------------------------------------------------------
  //**/ evaluate the large sample: vecXX ==> vecYY
  //**/ ---------------------------------------------------------
  int iOne=1;
  psVector vecYY;
  vecYY.setLength(nSamp2);
  if (useSimulator_ == 1)
  {
    status = funcIO->ensembleEvaluate(nSamp2,nInputs,
                         vecXX.getDVector(),iOne,
                         vecYY.getDVector(),iOne);
    if (status != 0)
    {
      printOutTS(PL_INFO,
           "Entropy2 ERROR: Function evaluator returns nonzero.\n");
      printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
      exit(1);
    }
  }
  else
  {
    status = faPtr->evaluatePoint(nSamp2,vecXX.getDVector(),
                                  vecYY.getDVector());
    if (status != 0)
    {
      printOutTS(PL_INFO,
           "RSEntropy2 ERROR: RS returns nonzero.\n");
      printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
      exit(1);
    }
  }

  //**/ ---------------------------------------------------------
  //**/ compute statistics (mean, std dev)
  //**/ ---------------------------------------------------------
  computeBasicStat(vecYY);

  //**/ ---------------------------------------------------------
  //**/ calculate total entropy and total delta
  //**/ ---------------------------------------------------------
  computeTotalEntDelta(vecYY);

  //**/ ===============================================================
  //**/ now ready for performing analysis on individual inputs
  //**/ need to generate a sample for input ii and one for the rest
  //**/ ==> vecSamPts2D, vecSamPtsND
  //**/ ===============================================================
  //**/ ---------------------------------------------------------
  //**/ first create storage for sampling and others
  //**/ ---------------------------------------------------------
  //**/ for sampling
  psVector  vecLower1, vecUpper1, vecSamPts2D;
  psVector  vecLower2, vecUpper2, vecSamPtsND;
  psVector  vecInpMeans1, vecInpStdvs1, vecInpMeans2, vecInpStdvs2;
  psIVector vecInpFlags1, vecInpFlags2, vecActive; 

  vecLower1.setLength(2);
  vecUpper1.setLength(2);
  vecInpFlags1.setLength(2);
  vecInpMeans1.setLength(2);
  vecInpStdvs1.setLength(2);
  vecSamPts2D.setLength(nSam1*2);

  vecActive.setLength(nInputs);
  if (nInputs > 2)
  {
    vecLower2.setLength(nInputs-2);
    vecUpper2.setLength(nInputs-2);
    vecInpFlags2.setLength(nInputs-2);
    vecInpMeans2.setLength(nInputs-2);
    vecInpStdvs2.setLength(nInputs-2);
  }
  vecSamPtsND.setLength(nSam2*nInputs);

  //**/ for analysis
  psVector  vecL1, vecU1;
  vecL1.setLength(iOne);
  vecU1.setLength(iOne);
  psVector vecmSamPts, vecOut, vecCMeans;
  vecmSamPts.setLength(nInputs*nSam1*nSam2);
  vecYY.setLength(nSam1*nSam2);
  VecEnt2_.setLength(nInputs*nInputs);
  VecDelta2_.setLength(nInputs*nInputs);

  //**/ ===============================================================
  //**/ computing expectation of conditional entropies
  //**/ ===============================================================
  int    kk, ss1, ss2, offset, indx, indy, iL, ii2;
  double ddata, Ymax, Ymin, *tempV, *tempY;
  double ECEnt, ECDel, DCMean, ECMean, ECEnt2, ECDel2, DCMean2,ECMean2;
  ProbMatrix matProbY;

  if (psConfig_.InteractiveIsOn() && adata.printLevel_ >= 0)
  {
    printAsterisks(PL_INFO, 0);
    if (entropyDelta_ == DO_ENTROPY)
    {
      printf("Possible sensitivity metrics for entropy (Let H(Y) ");
      printf("= total entropy):\n");
      printf("* X - selected input pair, ~X - all other inputs\n");
      printf("* Entropy of Conditional Expectation : ");
      printf("H(E_{~X}[Y|X])\n");
      printf("  - Where E_{~X}[*] is expectation w.r.t. all ");
      printf("inputs given X\n");
      printf("  - Meaning: Entropy induced by X when effect ");
      printf("of ~X are averaged\n");
      printf("  - i.e. for each possible X, compute E_{~X}[Y] ");
      printf("and take H\n");
      printf("  - H(E_{~X}[Y|X]) small if Y is weakly dependent ");
      printf("on X.\n");
      printf("  - H(E_{~X}[Y|X]) can be negative (differential entropy)\n");
      printf("  - H(E_{~X}[Y|X]) is analogous to VCE(X)\n");
      printf("* Expectation of Conditional Entropy : ");
      printf("E_{X}[H(Y|X)]\n");
      printf("  - Expected (average wrt X) entropy induced by ~X\n");
      printf("  - i.e. compute H(Y|X=x) for all x and take average\n");
      printf("  - usually denoted as H(Y|X)\n");
      printf("  - E_{X}[H(Y|X)] small relative to H(Y) ==> X sensitive\n");
      printf("* Here we use the following sensitivity metric:\n");
      printf("  - S(X) = H(Y) - E_{X}[H(Y|X)]\n");
      printf("  - S(X) is analogous to TSI(X)\n");
      printf("  - S(X) is the same as mutual information:\n");
      printf("    I(X,Y) = H(Y) - H(Y|X) = H(Y) - E_{X}[H(Y|X)]\n");
    }
    if (entropyDelta_ == DO_DELTA)
    {
      printf("Possible sensitivity metrics for delta (Let TD ");
      printf("= total delta):\n");
      printf("* X - selected input pair, ~X - all other inputs\n");
      printf("* Delta of Conditional Expectation : ");
      printf("D(E_{~X}[Y|X])\n");
      printf("  - Where E_{~X}[*] is expectation w.r.t. all ");
      printf("inputs given X\n");
      printf("  - Meaning: Delta induced by X when effect ");
      printf("of ~X are averaged\n");
      printf("  - i.e. for each possible X, compute E_{~X}[Y] ");
      printf("and take D\n");
      printf("  - D(E_{~X}[Y|X]) = 0 if Y does not depend ");
      printf("on X.\n");
      printf("  - D(E_{~X}[Y|X]) is analogous to VCE(X)\n");
      printf("* Expectation of Conditional Delta : ");
      printf("E_{X}[D(Y|X)]\n");
      printf("  - Expected (average wrt X) delta induced by ~X\n");
      printf("  - i.e. compute D(Y|X=x) for all x and take average\n");
      printf("  - usually denoted as D(Y|X)\n");
      printf("  - TD - E_{X}[D(Y|X)] is analogous to TSI(X)\n");
      printf("* Here we use D(E_{~X}[Y|X]) as sensitivity metric\n");
    }
    printDashes(PL_INFO, 0);
  }
  if (nInputs == 2)
  {
    nSam1 = 100000;
    nSam2 = 1;
  }
  if (entropyDelta_ == DO_ENTROPY)
    printf("INFO: RSEntropy2 sample sizes (nSam1, nSam2) = %d %d\n",
           nSam1,nSam2);
  else if (entropyDelta_ == DO_DELTA)
    printf("INFO: RSDelta2 sample sizes (nSam1, nSam2) = %d %d\n",
           nSam1,nSam2);

  for (ii = 0; ii < nInputs; ii++)
  {
    for (ii2 = ii+1; ii2 < nInputs; ii2++)
    {
      if (psConfig_.InteractiveIsOn())
      {
        if (entropyDelta_ == DO_ENTROPY)
          printOutTS(PL_INFO, 
          "Calculating H(Y) induced by inputs (%d,%d)\n",ii+1,ii2+1);
        if (entropyDelta_ == DO_DELTA)
          printOutTS(PL_INFO, 
          "Calculating D(Y) induced by inputs (%d,%d)\n",ii+1,ii2+1);
      }
      
      //**/ ------------------------------------------------------
      //**/ create sample for the ii-th input ==> vecSamPts2D
      //**/ ------------------------------------------------------
      for (jj = 0; jj < nInputs; jj++)
      {
        if ((jj != ii && corMatp->getEntry(ii,jj) != 0) ||
            (jj != ii2 && corMatp->getEntry(ii2,jj) != 0))
        {
          printf("WARNING: Correlation between input ");
          printf("(%d,%d) and other inputs != 0.\n",ii+1,ii2+1);
          printf("         This correlation will be ignored.\n");
        }
      }
      vecLower1[0] = xLower[ii]; vecLower1[1] = xLower[ii2];
      vecUpper1[0] = xUpper[ii]; vecUpper1[1] = xUpper[ii2];

      //**/ if both input ii and ii2 have uniform distributions,
      //**/ just use LPTAU, otherwise call pdf manager
      if (pdfFlags[ii] == 0 && pdfFlags[ii2] == 0)
      {
        Sampling *sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
        sampler->setSamplingParams(nSam1, 1, 0);
        sampler->setInputBounds(2, vecLower1.getDVector(), 
                                vecUpper1.getDVector());
        sampler->setOutputParams(1);
        sampler->initialize(0);
        psVector vecY1;
        psIVector vecS1;
        vecS1.setLength(nSam1);
        vecY1.setLength(nSam1);
        vecSamPts2D.setLength(nSam1*2);
        sampler->getSamples(nSam1, 2, 1,vecSamPts2D.getDVector(),
                    vecY1.getDVector(), vecS1.getIVector());
        delete sampler;
      }
      else
      {
        vecInpFlags1[0] = pdfFlags[ii];
        vecInpFlags1[1] = pdfFlags[ii2];
        vecInpMeans1[0] = inputMeans[ii];
        vecInpMeans1[1] = inputMeans[ii2];
        vecInpStdvs1[0] = inputStdevs[ii];
        vecInpStdvs1[1] = inputStdevs[ii2];
        psMatrix corMat1;
        corMat1.setDim(2,2);
        corMat1.setEntry(0, 0, corMatp->getEntry(ii,ii));
        corMat1.setEntry(0, 1, corMatp->getEntry(ii,ii2));
        corMat1.setEntry(1, 0, corMatp->getEntry(ii2,ii));
        corMat1.setEntry(1, 1, corMatp->getEntry(ii2,ii2));
        PDFManager *pdfman1 = new PDFManager();
        pdfman1->initialize(2,vecInpFlags1.getIVector(),
            vecInpMeans1.getDVector(),vecInpStdvs1.getDVector(),
            corMat1,NULL,NULL);

        vecOut.setLength(2*nSam1);
        pdfman1->genSample(nSam1,vecOut,vecLower1,vecUpper1);
        for (jj = 0; jj < nSam1*2; jj++)
          vecSamPts2D[jj] = vecOut[jj];
        delete pdfman1;
      }

      //**/ ------------------------------------------------------
      //**/ create sample for the other inputs ==> vecSamPtsND
      //**/ (If nInputs == 2, don't need to create sample)
      //**/ ------------------------------------------------------
      if (nInputs > 2)
      {
        //**/ select lower and upper bounds
        kk = 0;
        for (jj = 0; jj < nInputs; jj++)
        {
          if (jj != ii && jj != ii2)
          {
            vecLower2[kk] = xLower[jj];
            vecUpper2[kk] = xUpper[jj];
          }
        }

        //**/ if all inputs ~X are uniform, just use LPTAU 
        //**/ otherwise, use pdf manager
        int checkUniform = 0;
        for (jj = 0; jj < nInputs; jj++)
          if (jj != ii && jj != ii2)
            checkUniform += pdfFlags[jj];

        //**/ if all inputs are uniform, just create a LPTAU sample
        if (checkUniform == 0)
        {
          Sampling *sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
          sampler->setSamplingParams(nSam2, 1, 0);
          sampler->setInputBounds(nInputs-2, vecLower2.getDVector(), 
                                  vecUpper2.getDVector());
          sampler->setOutputParams(1);
          sampler->initialize(0);
          psVector vecY2;
          psIVector vecS2;
          vecS2.setLength(nSam2);
          vecY2.setLength(nSam2);
          vecSamPtsND.setLength(nSam2*(nInputs-2));
          sampler->getSamples(nSam2, nInputs-2, 1, 
                      vecSamPtsND.getDVector(),vecY2.getDVector(),
                      vecS2.getIVector());
          delete sampler;
        }
        else
        {
          psMatrix corMat2;
          corMat2.setDim(nInputs-2, nInputs-2);
          kk = 0;
          for (jj = 0; jj < nInputs; jj++)
          {
            if (jj != ii && jj != ii2)
            {
              vecActive[kk] = jj;
              kk++;
            }
          }
          kk = 0;
          for (jj = 0; jj < nInputs; jj++)
          {
            if (jj != ii && jj != ii2)
            {
              vecInpFlags2[kk] = pdfFlags[jj];
              vecInpMeans2[kk] = inputMeans[jj];
              vecInpStdvs2[kk] = inputStdevs[jj];
              kk++;
            }
          }
          corMat2.submatrix(*corMatp,nInputs-2,vecActive.getIVector());
          PDFManager *pdfman2 = new PDFManager();
          pdfman2->initialize(nInputs-2,vecInpFlags2.getIVector(),
                     vecInpMeans2.getDVector(),vecInpStdvs2.getDVector(),
                     corMat2,NULL,NULL);
          vecOut.setLength(nSam2*(nInputs-2));
          pdfman2->genSample(nSam2, vecOut, vecLower2, vecUpper2);
          for (jj = 0; jj < nSam2*(nInputs-2); jj++)
            vecSamPtsND[jj] = vecOut[jj];
          delete pdfman2;
        }
      }

      //**/ ------------------------------------------------------
      //**/ populate vecmSamPts by combining the 2 subsamples
      //**/ So, for each of the nSam2 for z (S\X), create a sample 
      //**/ for X. After the following, there will be nSam2 blocks 
      //**/ of nSam1 points, each of which corresponds to a 
      //**/ different and unique instance of S\X with X varying 
      //**/ ------------------------------------------------------
      //**/ populate vecmSamPts for each level (box)
      for (ss2 = 0; ss2 < nSam2; ss2++)
      {
        offset = ss2 * nSam1 * nInputs;
        //**/ concatenate the two sample ==> vecmSamPts
        for (ss1 = 0; ss1 < nSam1; ss1++)
        {
          kk = 0;
          for (jj = 0; jj < nInputs; jj++)
          {
            if (jj != ii && jj != ii2)
            {
              vecmSamPts[offset+ss1*nInputs+jj] =
                         vecSamPtsND[ss2*(nInputs-2)+kk];
              kk++;
            }
            if (jj == ii)
              vecmSamPts[offset+ss1*nInputs+jj]=vecSamPts2D[ss1*2];
            if (jj == ii2)
              vecmSamPts[offset+ss1*nInputs+jj]=vecSamPts2D[ss1*2+1];
          }
        }
      }
      
      //**/ ------------------------------------------------------
      //**/ evaluate ==> vecmSamPts ==> vecYY
      //**/ ------------------------------------------------------
      nSamp2 = nSam1*nSam2;
      if (useSimulator_ == 1)
      {
        if (psConfig_.InteractiveIsOn() && printLevel > 3)
          printOutTS(PL_INFO,"Entropy2 INFO: Function evaluations\n");
        for (ss1 = 0; ss1 < nSamp2; ss1+=nSam2)
        {
          tempV = vecmSamPts.getDVector();
          tempV = &(tempV[ss1*nInputs]);
          tempY = vecYY.getDVector();
          tempY = &(tempY[ss1]);
          status = funcIO->ensembleEvaluate(nSam2,nInputs,tempV,iOne,
                                            tempY, ss1+1);
          if (status != 0)
          {
            printOutTS(PL_INFO,
                 "Entropy2 ERROR: Function evaluator returns nonzero.\n");
            printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
            exit(1);
          }
        }
      }
      else
      {
        if (psConfig_.InteractiveIsOn() && printLevel > 3)
          printOutTS(PL_INFO,"RSEntropy2 INFO: RS evaluations\n");
        for (ss1 = 0; ss1 < nSamp2; ss1+=nSam2)
        { 
          tempV = vecmSamPts.getDVector();
          tempV = &(tempV[ss1*nInputs]);
          tempY = vecYY.getDVector();
          tempY = &(tempY[ss1]);
          status = faPtr->evaluatePoint(nSam2,tempV,tempY);
          if (status != 0)
          {
            printOutTS(PL_INFO,
              "RSEntropy2 ERROR: RS evaluator returns nonzero.\n");
            printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
            exit(1);
          }
        }
      }

      //**/ ------------------------------------------------------
      //**/ To compute:
      //**/ (1) Expectation of conditional delta (ECDel)
      //**/      int D(Y|X) dX = 1/M1 sum_{X} D(Y|X,~X)
      //**/ (2) Expectation of conditional entropy (ECEnt)
       //**/     int H(Y|X) dX = 1/M1 sum_{X} H(Y|X,~X)
      //**  where
      //**/  D(Y|X1,X2,~X) = Ymax(\sum_{k=1}^M2)-Ymin(\sum_{k=1}^2)
      //**/ ------------------------------------------------------
      ECDel = 0; /* expectation of conditional delta */
      ECEnt = 0; /* expectation of conditional entropy */
      vecCMeans.setLength(nSam1);
      double emax = -PSUADE_UNDEFINED;
      double emin =  PSUADE_UNDEFINED;
      for (ss1 = 0; ss1 < nSam1; ss1++)
      {
        //**/ find Ymax and Ymin and create conditional means 
        Ymax = -PSUADE_UNDEFINED;
        Ymin =  PSUADE_UNDEFINED;
        for (ss2 = 0; ss2 < nSam2; ss2++)
        {
          ddata = vecYY[ss2*nSam1+ss1];
          if (ddata > Ymax) Ymax = ddata;
          if (ddata < Ymin) Ymin = ddata;
          vecCMeans[ss1] += ddata;
        }
        vecCMeans[ss1] /= (double) nSam2;
        vecL1[0] = Ymin;
        vecU1[0] = Ymax;
        ECDel += Ymax - Ymin;

        //**/ -------------------------------------------------------
        //**/ partition output Y for a fixed X (create matProbY)
        //**/ -------------------------------------------------------
        //**/ for entropy calculation
        if (entropyDelta_ == DO_ENTROPY)
        {
          ddata = 2.0 * PABS(Ymax - Ymin) / (PABS(Ymax) + PABS(Ymin));
          if (ddata >= 1e-8) 
          {
            //**/ create and load probability matrix 
            matProbY.setDim(nSam2, iOne);
            for (ss2 = 0; ss2 < nSam2; ss2++)
              matProbY.setEntry(ss2,0,vecYY[ss2*nSam1+ss1]);

            //**/ binning and compute entropy
            status = computeEntropy(matProbY,vecL1,vecU1,ddata,adaptive_);
            if (status == 0)
            {
              ECEnt += ddata;
              if (ddata > emax) emax = ddata;
              if (ddata < emin) emin = ddata;
              if (printLevel > 4)
              {
                printf("  Inputs %d,%d ss1 = %6d, entropy = %10.3e ",
                       ii+1, ii2+1, ss1+1, ddata);
                printf("(Ymin,Ymax = %10.3e, %10.3e)\n",
                       vecL1[0],vecU1[0]);
              }
              if (printLevel > 4 && ddata > outputEntropy_)
                printf("WARNING: k=%d of %d H(Y|X_k)=%10.3e > H(Y)=%10.3e\n",
                       ss1+1,nSam1,ddata,outputEntropy_); 
            }
            else
            {
              printf("RSEntropy1 ERROR: returned from computeEntropy\n");
              printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
              exit(1);
            }
          }
          else if (nSam2 > 1)
          {
            if (printLevel > 3)
            {
              printf("Entropy2 INFO: Small Ymax - Ymin = %e\n",Ymax-Ymin);
              printf("==> Assume conditional entropy = 0 for ss1 = %d\n",
                     ss1+1);
            }
          }
        } /* entropyDelta_ = DO_ENTROPY */
      } /* ss1 of nSam1 */ 
      ECEnt /= (double) nSam1;
      ECDel /= (double) nSam1;

      //**/ ------------------------------------------------------
      //**/ To compute:
      //**/ (1) Delta of conditional expectation (DCMean)
      //**/ (2) Entropy of conditional expectation (ECMean)
      //**/ ------------------------------------------------------
      Ymax = -PSUADE_UNDEFINED;
      Ymin =  PSUADE_UNDEFINED;
      for (ss1 = 0; ss1 < nSam1; ss1++)
      {
        ddata = vecCMeans[ss1];
        if (ddata > Ymax) Ymax = ddata;
        if (ddata < Ymin) Ymin = ddata;
      }
      DCMean = Ymax - Ymin;
      ECMean = 0;
      VecDelta2_[ii*nInputs+ii2] = DCMean;
      VecDelta2_[ii2*nInputs+ii] = DCMean;

      //**/ compute entropy only when it is requested
      if (entropyDelta_ == DO_ENTROPY)
      {
        ddata = 2.0 * PABS(Ymax - Ymin) / (PABS(Ymax) + PABS(Ymin));
        if (ddata >= 1e-8) 
        {
          //**/ create and load probability matrix
          matProbY.setDim(nSam1, iOne);
          for (ss1 = 0; ss1 < nSam1; ss1++)
            matProbY.setEntry(ss1,0,vecCMeans[ss1]);

          //**/ binning
          vecL1[0] = Ymin;
          vecU1[0] = Ymax;
          status = computeEntropy(matProbY,vecL1,vecU1,ECMean,adaptive_);
        }
        else
        {
          if (printLevel > 3)
          {
            printf("Entropy2 INFO: Small Ymax - Ymin = %e\n",Ymax-Ymin);
            printf("==> Assume entropy of conditional mean=0 for ss1 = %d\n",
                   ss1+1);
          }
          ECMean = 0.0;
        }
      } /* entropyDelta_ == DO_ENTROPY */

      if (entropyDelta_ == DO_ENTROPY)
      {
        printf("  Inputs %3d,%3d: \n",ii+1,ii2+1);
        printf("        H(E_{~X}[Y|X])         = %10.3e (VCE-like)\n",
               ECMean);
        printf("        E_{X}[H(Y|X)] = H(Y|X) = %10.3e\n",ECEnt);
        VecEnt2_[ii*nInputs+ii2] = outputEntropy_ - ECEnt;
        VecEnt2_[ii2*nInputs+ii] = outputEntropy_ - ECEnt;
        printf("        H(Y) - E_{X}[H(Y|X)]   = %10.3e (TSI-Like)\n",
               VecEnt2_[ii*nInputs+ii2]);
        if (printLevel > 3)
          printf("        max/min H(Y|X)         = %10.3e %10.3e\n",
                 emax,emin);
      }
      else if (printLevel > 1 && entropyDelta_ == DO_DELTA)
      {
        printf("  Inputs %3d,%3d: \n",ii+1,ii2+1);
        printf("        D(E_{~X}[Y|X])         = %10.3e (VCE-like)\n",
               DCMean);
        printf("        E_{X}[D(Y|X)] = D(Y|X) = %10.3e\n",ECDel);
        printf("        D(Y) - E_{X}[D(Y|X)]   = %10.3e (TSI-like)\n",
               outputDelta_-ECDel);
      }

      //**/ ------------------------------------------------------
      //**/ compute E[H(Y|~X)]
      //**/ int H(Y|X1,X2) d~X = 1/N sum_{~X} H(Y|X1,X2,~X)
      //**  where
      //**/    H(Y|X1,X2,~X) = -\sum_{k=1}^L H(Y_k|X1,X2,~X) dY_k
      //**/ where`
      //**/    H(Y_k|X1,X2,~X)=p(Y_k|X1,X2,~X) log[p(Y_k|X1,X2,~X)
      //**/ ------------------------------------------------------
      ECEnt2 = ECDel2 = 0;
      double accum;
      for (ss2 = 0; ss2 < nSam2; ss2++)
      {
        offset = ss2 * nSam1;
        //**/ ----------------------------------------------------
        //**/ find output max and min in order to partition Y
        //**/ ----------------------------------------------------
        Ymax = -PSUADE_UNDEFINED;
        Ymin =  PSUADE_UNDEFINED;
        for (ss = 0; ss < nSam1; ss++)
        {
          ddata = vecYY[offset+ss];
          if (ddata > Ymax) Ymax = ddata;
          if (ddata < Ymin) Ymin = ddata;
        }
        vecL1[0] = Ymin;
        vecU1[0] = Ymax;
        ECDel2 += (Ymax - Ymin);
        if (Ymax == Ymin) continue;

        //**/ -------------------------------------------------------
        //**/ compute entropy
        //**/ -------------------------------------------------------
        if (entropyDelta_ == DO_ENTROPY)
        {
          //**/ create and load probability matrix
          matProbY.setDim(nSam1, iOne);
          for (ss = 0; ss < nSam1; ss++)
            matProbY.setEntry(ss,0,vecYY[offset+ss]);

          //**/ binning and compute entropy
          status = computeEntropy(matProbY,vecL1,vecU1,accum,adaptive_);
          if (status == 0) ECEnt2 += accum;
        }
      } /* ss2 of nSam2 */
      ECEnt2 /= (double) nSam2;
      ECDel2 /= (double) nSam2;
      if (printLevel > 1 && entropyDelta_ == DO_ENTROPY)
      {
        printf("        E_{~X}[H(Y|~X)]        = %10.3e\n",ECEnt2);
      }
      if (printLevel > 1 && entropyDelta_ == DO_DELTA)
      {
        printf("        E_{~X}[D(Y|~X)]        = %10.3e\n",ECDel2);
      }

      //**/ ------------------------------------------------------
      //**/ Now compute TSI-like quantities
      //**/ total entropy minus H(E[Y|~X]) 
      //**/ total delta   minus D(E[Y|~X]) 
      //**/ ------------------------------------------------------
      vecCMeans.setLength(nSam2);
      for (ss2 = 0; ss2 < nSam2; ss2++)
      {
        offset = ss2 * nSam1;
        for (ss1 = 0; ss1 < nSam1; ss1++)
        {
          ddata = vecYY[offset+ss1];
          vecCMeans[ss2] += ddata;
        }
        vecCMeans[ss2] /= (double) nSam1;
      }
      Ymax = -PSUADE_UNDEFINED;
      Ymin =  PSUADE_UNDEFINED;
      for (ss2 = 0; ss2 < nSam2; ss2++)
      {
        ddata = vecCMeans[ss2];
        if (ddata > Ymax) Ymax = ddata;
        if (ddata < Ymin) Ymin = ddata;
      }
      DCMean2  = Ymax - Ymin;
      vecL1[0] = Ymin;
      vecU1[0] = Ymax;

      //**/ ------------------------------------------------------
      //**/ binning
      //**/ ------------------------------------------------------
      ECMean2 = 0;
      if (entropyDelta_ == DO_ENTROPY)
      {
        ddata = 2.0 * PABS(Ymax - Ymin) / (PABS(Ymax) + PABS(Ymin));
        if (ddata >= 1e-8) 
        {
          //**/ create and load probability matrix
          matProbY.setDim(nSam2, iOne);
          for (ss2 = 0; ss2 < nSam2; ss2++)
            matProbY.setEntry(ss2,0,vecCMeans[ss2]);
  
          //**/ binning and compute entropy
          status = computeEntropy(matProbY,vecL1,vecU1,ECMean2,adaptive_);
          if (status != 0) ECMean2 = 0;
        }
        else if (nSam2 > 1)
        {
          if (printLevel > 3)
          {
            printf("Entropy2 INFO: Small Ymax - Ymin = %e\n",Ymax-Ymin);
            printf("==> Assume entropy of conditional mean = 0 for ~X\n");
          }
        }
      }
      if (printLevel > 1 && entropyDelta_ == DO_ENTROPY)
      {
        printf("        H(E_{X}[Y|~X])         = %10.3e\n",ECMean2);
      }
      if (printLevel > 1 && entropyDelta_ == DO_DELTA)
      {
        printf("        D(E_{X}[Y|~X])         = %10.3e\n",DCMean2);
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ some clean up
  //**/ ---------------------------------------------------------------
  if (ioPtr == NULL) delete corMatp;
  if (faPtr != NULL) delete faPtr;
  if (funcIO != NULL) delete funcIO;

  //**/ ---------------------------------------------------------------
  //**/ print unsorted indices
  //**/ ---------------------------------------------------------------
#if 0
  if (psConfig_.InteractiveIsOn() && adata.printLevel_ >= 0)
  {
    printAsterisks(PL_INFO, 0);
    if (entropyDelta_ == DO_ENTROPY)
    {
      for (ii = 0; ii < nInputs; ii++)
        for (ii2 = ii+1; ii2 < nInputs; ii2++)
          printOutTS(PL_INFO,
            "    H(Y) induced by inputs (%d,%d) = %10.4e\n",ii+1,ii2+1,
            VecEnt2_[ii*nInputs+ii2]);
    }
    else if (entropyDelta_ == DO_DELTA)
    {
      for (ii = 0; ii < nInputs; ii++)
        for (ii2 = ii+1; ii2 < nInputs; ii2++)
          printOutTS(PL_INFO,
            "    D(Y) induced by inputs (%d,%d) = %10.4e\n",ii+1,ii2+1,
            VecDelta2_[ii*nInputs+ii2]);
    }
    printAsterisks(PL_INFO, 0);
  }
#endif
  return 0.0;
}

// ************************************************************************
// perform total entropy and total delta
// ------------------------------------------------------------------------
int RSMEntropy2Analyzer::computeTotalEntDelta(psVector vecY)
{
  //**/ ---------------------------------------------------------------
  //**/ compute min-max measure
  //**/ ---------------------------------------------------------------
  int    ss, iOne=1, nSams = vecY.length();
  double ddata;
  double Ymax = -PSUADE_UNDEFINED;
  double Ymin =  PSUADE_UNDEFINED;
  for (ss = 0; ss < nSams; ss++)
  {
    ddata = vecY[ss];
    if (ddata > Ymax) Ymax = ddata;
    if (ddata < Ymin) Ymin = ddata;
  }
  outputDelta_ = Ymax - Ymin;

  //**/ ---------------------------------------------------------------
  //**/ if compute delta only, don't need to move on 
  //**/ ---------------------------------------------------------------
  if (entropyDelta_ == DO_DELTA) return 0;

  //**/ ---------------------------------------------------------------
  //**/ compute total entropy (make sure to use adaptive because the
  //**/ rest of this function uses adaptive)
  //**/ ---------------------------------------------------------------
  printf("RSEntropy2 INFO: Sample size for computing total entropy = %d\n",
         nSams);
  printf("RSEntropy2 INFO: Binning resolution for total entropy = %d\n",
         nLevels_);
  ProbMatrix matProbY;
  matProbY.setDim(nSams, iOne);
  for (ss = 0; ss < nSams; ss++)
  {
    ddata = vecY[ss];
    matProbY.setEntry(ss,0,ddata);
  }
  psVector vecL1, vecU1;
  vecL1.setLength(iOne);
  vecU1.setLength(iOne);
  vecL1[0] = Ymin;
  vecU1[0] = Ymax;
  if (psConfig_.DiagnosticsIsOn())
    printf("RSEntropy2: Computing total entropy.\n");
  int status = computeEntropy(matProbY, vecL1, vecU1, outputEntropy_, iOne);
  if (psConfig_.DiagnosticsIsOn())
    printf("RSEntropy2: Computing total entropy completed.\n");
  if (status != 0)
  {
    printf("RSEntropy2 INFO: computeEntropy returns nonzero.\n");
    outputEntropy_ = 0;
  }
  //else printf("Output total entropy = %e\n", outputEntropy_);
  return 0;
}

// ************************************************************************
// compute basic statistics
// ------------------------------------------------------------------------
double RSMEntropy2Analyzer::computeBasicStat(psVector vecY)
{
  int ss, nSams;

  nSams = vecY.length();
  outputMean_ = 0;
  for (ss = 0; ss < nSams; ss++) outputMean_ += vecY[ss];
  outputMean_ /= (double) nSams;
  outputStd_  = 0.0;
  for (ss = 0; ss < nSams; ss++)
    outputStd_  += pow(vecY[ss]-outputMean_,2.0);
  outputStd_ /= (double) (nSams - 1);
  outputStd_ = sqrt(outputStd_ );
  //**/ if sample std dev = 0, set it to 1 to avoid divide by 0
  if (outputStd_  == 0) outputStd_  = 1.0;
  return 0;
}

// ************************************************************************
// compute entropy 
//**/ The reason matProbY is passed in (instead of created locally here) is
//**/ that the calling function may need information in matProbY after 
//**/ binning. 
// ------------------------------------------------------------------------
int RSMEntropy2Analyzer::computeEntropy(ProbMatrix &matProbY,psVector vecL,
                         psVector vecU, double &entropy, int adaptOrNot)
{
  int    ss, kk, status=-1, nSams;
  double ddata, dY;

  nSams = matProbY.nrows();
  entropy = 0;
  //**/ compute entropy using adaptive scheme 
  if (adaptOrNot == 1)
  {
    //**/ adaptive version (in the spirit of the following)
    //**/ Ref: Statistical Validation of Mutual Information 
    //**/      Calculations: Comparison of Alternative 
    //**/      Numerical Algorithms by C. Celluci, A. Albano
    //**/      and P. Rapp. Physical Review E 71 (6 Pt 2):066208
    //**/      DOI:10.1103/PhysRevE.71.066208
    //**/ Conclusion: for large samples, such as those created
    //**/             via RS, adaptive and uniform binning give
    //**/             similar results
    status = matProbY.binAdaptive(nLevels_, vecL, vecU);
    if (status != 0)
    {
      if (psConfig_.DiagnosticsIsOn())
      {
        printf("RSEntropy2 ERROR: When computing entropy (a).\n");
        printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
        printf("Note: Switch to uniform binning, but the ");
        printf("result may not be correct.\n");
      }
    }
    else
    {
      //**/ Note: sum of matProbY should be equal to nLevels
      for (kk = 0; kk < matProbY.nrows(); kk++)
      {
        //**/ get probability of bin kk
        ddata = 1.0 * matProbY.getCount(kk) / nSams;
        //**/ to obtain PDF, need to multiply by grid width
        //**/ that is, ddata = P(Y) dY
        //**/ entropy = - sum_{k} P(Y_k) log P(Y_k) dY_k
        dY = matProbY.getEntry(kk, 0);
        if (dY <= 0)
        {
          printf("RSEntropy2 ERROR: bin width(%d) = %e <= 0 (a)\n",
                 kk+1, matProbY.getEntry(kk,0));
          printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
          exit(1);
        }
        if (ddata > 0) entropy -= ddata * log(ddata/dY);
        //**/ October 2023
        //**/ this following option does not work for lognormal
        //**/ This seems non-intuitive, but it is true (has
        //**/ been tested with normal/lognormal distributions
        //if (ddata > 0) entropy -= ddata * log(ddata) * dY;
      }
    }
  }
  //**/ compute entropy using uniform scheme or when adaptive 
  //**/ scheme fails
  if (status != 0)
  {
    status = matProbY.binUniform(nLevels_, vecL, vecU);
    if (status != 0)
    {
      if (psConfig_.DiagnosticsIsOn())
      {
        printf("RSEntropy2 ERROR: When computing entropy (b).\n");
        printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
      }
      return -1;
    }
    else
    {
      //**/ Note: sum of matProbY should be equal to nLevels
      for (kk = 0; kk < matProbY.nrows(); kk++)
      {
        ddata = 1.0 * matProbY.getCount(kk) / nSams;
        //**/ adjust to reflect true P(Y)
        dY = (vecU[0] - vecL[0]) / nLevels_;
        ddata /= dY;
        //**/ to obtain PDF, need to multiply by grid width
        //**/ that is, ddata = P(Y) dY
        //**/ entropy = - sum_{k} P(Y_k) log P(Y_k) dY_k
        if (ddata > 0) entropy -= ddata * log(ddata) * dY;
      }
    }
  }
  return 0;
}

// ************************************************************************
// set internal parameters
// ------------------------------------------------------------------------
int RSMEntropy2Analyzer::setParam(int argc, char **argv)
{
  char  *request = (char *) argv[0];
  if (!strcmp(request,"ana_entropy2_nlevels")) 
    nLevels_ = *(int *) argv[1];
  else if (!strcmp(request, "ana_entropy2_entropy")) 
    entropyDelta_ = DO_ENTROPY;
  else if (!strcmp(request, "ana_entropy2_delta"))   
    entropyDelta_ = DO_DELTA;
  else if (!strcmp(request, "ana_entropy2_use_simulator"))   
    useSimulator_ = 1;
  else if (!strcmp(request, "ana_entropy2_nors"))   
    useRS_ = 0;
  else
  {
    printOutTS(PL_ERROR,
         "RSEntropy2 ERROR: setParam - not valid.\n");
    printOutTS(PL_ERROR,
         "                          param = %s\n",request);
    exit(1);
  }
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
RSMEntropy2Analyzer& RSMEntropy2Analyzer::operator=(const RSMEntropy2Analyzer &)
{
  printOutTS(PL_ERROR, 
       "RSEntropy2 operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int RSMEntropy2Analyzer::get_nInputs()
{
  return nInputs_;
}
double RSMEntropy2Analyzer::get_outputMean()
{
  return outputMean_;
}
double RSMEntropy2Analyzer::get_outputStd()
{
  return outputStd_;
}
double RSMEntropy2Analyzer::get_outputEntropy()
{
  return outputEntropy_;
}
double RSMEntropy2Analyzer::get_outputDelta()
{
  return outputDelta_;
}
double RSMEntropy2Analyzer::get_entropy2(int ind1, int ind2)
{
  if (ind1 < 0 || ind1 >= nInputs_)
  {
    printf("RSEntropy2 ERROR: get_entropy2 index error %d.\n",ind1);
    return 0.0;
  }
  if (ind2 < 0 || ind2 >= nInputs_)
  {
    printf("RSEntropy2 ERROR: get_entropy2 index error %d.\n",ind2);
    return 0.0;
  }
  if (VecEnt2_.length() <= ind1*nInputs_+ind2)
  {
    printf("RSEntropy2 ERROR: get_entropy2 has not value.\n");
    return 0.0;
  }
  return VecEnt2_[ind1*nInputs_+ind2];
}
double RSMEntropy2Analyzer::get_delta2(int ind1, int ind2)
{
  if (ind1 < 0 || ind1 >= nInputs_)
  {
    printf("RSEntropy2 ERROR: get_delta2 index error %d.\n",ind1);
    return 0.0;
  }
  if (ind2 < 0 || ind2 >= nInputs_)
  {
    printf("RSEntropy2 ERROR: get_delta2 index error %d.\n",ind2);
    return 0.0;
  }
  if (VecDelta2_.length() <= ind1*nInputs_+ind2)
  {
    printf("RSEntropy2 ERROR: get_delta2 has not value.\n");
    return 0.0;
  }
  return VecDelta2_[ind1*nInputs_+ind2];
}

