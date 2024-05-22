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
// Functions for the class RSMEntropy1  
// AUTHOR : CHARLES TONG
// DATE   : 2023
//**/ ---------------------------------------------------------------------
//**/ Entropy-based, and max-min main effect analysis 
//**/ Theory:
//**/  H(Y|X) =  sum_{x in X} p(x) H(Y|X=x)
//**/         = -sum_{x in X} p(x) sum_{y in Y} p(y|x) log p(y|x)
//**/         = -sum_{x in X, y in Y} p(x) p(y|x) log p(y|x)
//**/         = -sum_{x in X, y in Y} p(x,y) log p(x,y)/p(x)
//**/         = -sum_{x in X, y in Y} [p(x,y) log p(x,y) - p(x,y) log p(x)]
//**/         = H(X,Y) - H(X) 
//**/  H(X,Y) = joint entropy 
//**/ Properties:
//**/  H(Y|X) = 0 if Y is completely determined by X
//**/  H(Y|X) = H(Y) if X and Y are independent
//**/           (in this case: H(X,Y) = H(X) + H(Y))
//**/  I(X,Y) = H(Y) - H(Y|X) = H(Y) - H(X,Y) + H(X)
//**/ Since small H(Y|X) means Y is sensitive to X, H(Y|X) is not an 
//**/ intuitive sensitivity indicator. Hence, we use I(X,Y) to be the
//**/ entropy-based sensitivity metirc
//**/ Note: This analysis will not handle input correlations
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "PsuadeUtil.h"
#include "sysdef.h"
#include "psMatrix.h"
#include "pData.h"
#include "RSMEntropy1Analyzer.h"
#include "ProbMatrix.h"
#include "Sampling.h"
#include "PDFManager.h"
#include "PDFNormal.h"
#include "Psuade.h"
#include "PsuadeData.h"
#include "PsuadeConfig.h"
#include "PrintingTS.h"
#include "FuncApprox.h"
#include "FunctionInterface.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))
#define DO_ENTROPY 0
#define DO_DELTA   1
#define DO_TOTALE  2

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
RSMEntropy1Analyzer::RSMEntropy1Analyzer() : Analyzer(),nInputs_(0),
                          outputMean_(0), outputStd_(0), outputEntropy_(0),
                          entropyDelta_(DO_ENTROPY), M1_(4000),
                          M2_(4000), useSimulator_(0), useRS_(1)
{
  setName("RSMEntropy1");
  adaptive_ = 1;       /* default: adaptive histogramming is on */
  nLevels_ = 100;      /* number of levels for the histograms */
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
RSMEntropy1Analyzer::~RSMEntropy1Analyzer()
{
}

// ************************************************************************
// perform analysis (intended for library call)
// (package incoming information into a aData object and call analyze)
// ------------------------------------------------------------------------
void RSMEntropy1Analyzer::analyze(int nInps, int nSamp, double *lbs, 
                                  double *ubs, double *X, double *Y)
{
  //**/ check data validity
  char pString[1000];
  snprintf(pString, 25, " RSMEntropy1 analyze (X)");
  checkDbleArray(pString, nSamp*nInps, X);
  snprintf(pString, 25, " RSMEntropy1 analyze (Y)");
  checkDbleArray(pString, nSamp, Y);
  snprintf(pString, 25, " RSMEntropy1 analyze (LBs)");
  checkDbleArray(pString, nInps, lbs);
  snprintf(pString, 25, " RSMEntropy1 analyze (UBs)");
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
double RSMEntropy1Analyzer::analyze(aData &adata)
{
  //**/ ===============================================================
  //**/ display header 
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    if (entropyDelta_ == DO_ENTROPY)
      printOutTS(PL_INFO,
        "*     Single-Parameter Entropy (M.I.) Analysis\n");
    else if (entropyDelta_ == DO_DELTA)
      printOutTS(PL_INFO,
        "*     Single-Parameter Delta Analysis\n");
    else if (entropyDelta_ == DO_TOTALE)
      printOutTS(PL_INFO,
        "*     Output Entropy Analysis\n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,
       "* TO GAIN ACCESS TO DIFFERENT OPTIONS: SET\n");
    printOutTS(PL_INFO,
       "* - ana_expert to finetune internal parameters\n");
    printOutTS(PL_INFO,
       "*   (e.g. to adjust binning resolution or select ");
    printOutTS(PL_INFO,
       "adaptive algorithm)\n");
    printOutTS(PL_INFO,
       "* - rs_expert mode to finetune response surfaces\n");
    printOutTS(PL_INFO,
       "* - printlevel to display more information\n");
    printOutTS(PL_INFO,
       "* Or, use configure file to finetune parameters\n");
    printEquals(PL_INFO, 0);
  }
 
  //**/ ===============================================================
  //**/ extract sample data and information
  //**/ ---------------------------------------------------------------
  int nInputs    = adata.nInputs_;
  nInputs_       = nInputs;
  int nOutputs   = adata.nOutputs_;
  int nSamples   = adata.nSamples_;
  double *YIn    = adata.sampleOutputs_;
  int outputID   = adata.outputID_;

  //**/ ===============================================================
  //**/ check errors 
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nOutputs <= 0) 
  {
    printOutTS(PL_ERROR, "RSEntropy1 ERROR: Invalid arguments.\n");
    printOutTS(PL_ERROR, "   nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR, "   nOutputs = %d\n", nOutputs);
    return PSUADE_UNDEFINED;
  } 
  if (useSimulator_ == 0 && nSamples <= 0)
  {
    printOutTS(PL_ERROR, "RSEntropy1 ERROR: Invalid arguments.\n");
    printOutTS(PL_ERROR, "   nSamples = %d <= 0\n", nSamples);
    return PSUADE_UNDEFINED;
  }
  if (outputID >= nOutputs || outputID < 0)
  {
    printOutTS(PL_ERROR,
        "RSMEntropy1 ERROR: Invalid output ID (%d).\n",outputID);
    return PSUADE_UNDEFINED;
  }
  //**/ make sure no output is undefined
  int ii, status = 0;
  for (ii = 0; ii < nSamples; ii++)
    if (YIn[nOutputs*ii+outputID] > 0.9*PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR,"RSEntropy1 ERROR: Some outputs are ");
    printOutTS(PL_ERROR,"undefined. Please prune all the\n");
    printOutTS(PL_ERROR,"                  undefined sample ");
    printOutTS(PL_ERROR,"points and re-run.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ===============================================================
  //**/ ask users to choose which histogramming algorithm (not needed
  //**/ for delta analysis)
  //**/ ===============================================================
  if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn() &&
      (entropyDelta_ == DO_ENTROPY || entropyDelta_ == DO_TOTALE))
  {
    char pString[101], winput[101];
    printf("Two algorithms are provided for computing entropy:\n");
    printf("(1) Uniform partitioning in histogramming, and\n");
    printf("(2) Adaptive partitioning in histogramming.\n");
    printf("Adaptive partitioning may give more accurate ");
    printf("result for small samples,\n");
    printf("but for samples generated from response surface ");
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
  //**/ ---------------------------------------------------------------
  char pString[500], *cString, winput1[500], winput2[500];
  if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn())
  {
    printAsterisks(PL_INFO, 0);
    //**/ not needed for delta analysis
    if (entropyDelta_ == DO_ENTROPY || entropyDelta_ == DO_TOTALE)
    {
      printOutTS(PL_INFO,
        "* RSEntropy1 uses binning to represent probabilities.\n");
      printOutTS(PL_INFO,
        "* The default number of bins K = %d\n",nLevels_);
      snprintf(pString,100,"Select a new K (suggestion: 10 - 100) : ");
      nLevels_ = getInt(5, 500, pString);
      printAsterisks(PL_INFO, 0);
    }
    //**/ not needed for total entropy calculation
    if (entropyDelta_ <= DO_DELTA)
    {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO,
        "* RSEntropy1/Delta1 uses 2 samples: 1 for 1 input, ");
      printOutTS(PL_INFO, "1 for the rest.\n");
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
  }
  else
  {
    //**/ alternatively, binning information can be obtained in the
    //**/ configuration object
    cString = psConfig_.getParameter("ana_entropy1_nlevels");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nLevels_);
      printOutTS(PL_INFO,
        "RSEntropy1 INFO: K from configure object = %d\n",nLevels_);
      if (nLevels_ < 10)
      {
        printOutTS(PL_INFO,"             K should be >= 10.\n");
        printOutTS(PL_INFO,"             Set K = 10.\n");
        nLevels_ = 10;
      }
    }
    cString = psConfig_.getParameter("ana_entropy1_M1");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &M1_);
      printOutTS(PL_INFO,
        "RSEntropy1/Delta1 INFO: M1 from configure object = %d\n",M1_);
      if (M1_ < 1000) M1_ = 1000;
    }
    cString = psConfig_.getParameter("ana_entropy1_M2");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &M2_);
      printOutTS(PL_INFO,
        "RSEntropy1/Delta1 INFO: M2 from configure object = %d\n",
        M2_);
      if (M2_ < 1000) M2_ = 1000;
    }
  }

  //**/ ---------------------------------------------------------
  //**/ calculate total entropy for raw sample 
  //**/ Note: The following code will be called by first calling
  //**/       setParam to reset useRS_ and then call RSEntropy1
  //**/       analyze
  //**/ ---------------------------------------------------------
  if (useRS_ == 0 && nSamples > 0)
  {
    psVector vecYY;
    vecYY.load(nSamples, YIn);
    computeTotalEntDelta(vecYY); 
    //**/ if only total entropy is requested, just return
    if (entropyDelta_ == DO_TOTALE) return 0;
  }

  //**/ ===============================================================
  //**/ Call entropy or delta analyzer based on entropyDelta_
  //**/ ---------------------------------------------------------------
  analyzeEntDelta(adata); 
  return 0.0;
}

// ************************************************************************
// perform entropy or delta analysis 
// ------------------------------------------------------------------------
double RSMEntropy1Analyzer::analyzeEntDelta(aData &adata)
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
  //**/ (these are needed for creating samples later for binning) 
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
  int ii, hasPDF=0, samPDF=0;
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
        "RSEntropy1/Delta1 INFO: Inputs have S-type (sample) PDF.\n");
      printOutTS(PL_INFO,
        "           ===>  Cannot perform entropy/delta analysis.\n");
      return PSUADE_UNDEFINED;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ get or create correlation matrix
  //**/ (needed for creating samples for binning later) 
  //**/ ---------------------------------------------------------------
  int jj, corFlag=0;
  psMatrix *corMatp=NULL;
  PsuadeData *ioPtr = adata.ioPtr_;
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
          //if (psConfig_.InteractiveIsOn() && printLevel > 3)
          //{
          //  printOutTS(PL_INFO, 
          //    "RSEntropy1/Delta1 INFO: Correlated inputs detected.\n");
          //  printOutTS(PL_INFO, 
          //    "           Correlation will be ignored in analyis.\n");
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
         "RSEntropy1/Delta1 INFO: Correlated inputs detected.\n");
    snprintf(pString, 100,
         "Correlation will be ignored in analyis. Continue? (y or n) ");
    getString(pString, winput);
    if (winput[0] != 'y') 
    {
      VecEnt1_.setLength(nInputs);
      VecDelta1_.setLength(nInputs);
      printf("RSEntropy1 INFO: Terminate\n");
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
           "RSEntropy1 ERROR: No dataIO object found.\n");
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
      printf("RSEntropy1/Delta1 ERROR: In building RS.\n");
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
           "Entropy1 ERROR: Function evaluator returns nonzero.\n");
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
           "RSEntropy1 ERROR: RS returns nonzero.\n");
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
  //**/ if only total entropy is requested, just return
  if (entropyDelta_ == DO_TOTALE) return 0;

  //**/ ===============================================================
  //**/ now ready for performing analysis on individual inputs
  //**/ need to generate a sample for input ii and one for the rest
  //**/ ==> vecSamPts1D, vecSamPtsND
  //**/ ===============================================================
  //**/ ---------------------------------------------------------
  //**/ first create storage for sampling and others
  //**/ ---------------------------------------------------------
  //**/ for sampling
  psVector  vecLower2, vecUpper2, vecSamPtsND;
  psVector  vecInpMeans2, vecInpStdvs2, vecSamPts1D;
  psIVector vecInpFlags2;

  vecLower2.setLength(nInputs);
  vecUpper2.setLength(nInputs);
  vecSamPts1D.setLength(nSam1);
  vecSamPtsND.setLength(nSam2*nInputs);
  if (nInputs > 1)
  {
    vecInpFlags2.setLength(nInputs-1);
    vecInpMeans2.setLength(nInputs-1);
    vecInpStdvs2.setLength(nInputs-1);
  }

  //**/ for analysis
  psVector  vecL1, vecU1;
  vecL1.setLength(iOne);
  vecU1.setLength(iOne);
  psVector vecmSamPts, vecOut, vecCMeans;
  vecmSamPts.setLength(nInputs*nSam1*nSam2);
  vecYY.setLength(nSam1*nSam2);
  VecEnt1_.setLength(nInputs);
  VecDelta1_.setLength(nInputs);

  //**/ ===============================================================
  //**/ computing expectation of conditional entropies
  //**/ ===============================================================
  int    kk, ss1, ss2, offset, indx, indy, iL;
  double DCMean, ddata, *oneSamplePt, Ymax, Ymin, *tempV;
  double ECEnt2, ECMean, ECMean2, DCMean2, ECDel, ECEnt, XEnt=0;
  double ECDel2, emax, emin, *tempY;
  ProbMatrix matProbX, matProbY;

  if (psConfig_.InteractiveIsOn() && adata.printLevel_ >= 0)
  {
    printAsterisks(PL_INFO, 0);
    if (entropyDelta_ == DO_ENTROPY)
    {
      printf("Possible sensitivity metrics for entropy (Let H(Y) ");
      printf("= total entropy):\n");
      printf("* X - selected input, ~X - all other inputs\n");
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
      printf("* Joint entropy H(X,Y) = H(Y|X) + H(X)\n");
      printf("  - if X and Y are independent, H(X,Y) = H(Y) + H(X)\n");
    }
    if (entropyDelta_ == DO_DELTA)
    {
      printf("Possible sensitivity metrics for delta (Let TD ");
      printf("= total delta):\n");
      printf("* X - selected input, ~X - all other inputs\n");
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
  if (nInputs == 1)
  {
    nSam1 = 100000;
    nSam2 = 1;
  }
  if (entropyDelta_ == DO_ENTROPY)
    printf("INFO: RSEntropy1 sample sizes (nSam1, nSam2) = %d %d\n",
           nSam1,nSam2);
  else if (entropyDelta_ == DO_DELTA)
    printf("INFO: RSDelta1 sample sizes (nSam1, nSam2) = %d %d\n",
           nSam1,nSam2);
  for (ii = 0; ii < nInputs; ii++)
  {
    if (psConfig_.InteractiveIsOn())
    {
      if (entropyDelta_ == DO_ENTROPY)
        printOutTS(PL_INFO,
        "Calculating H(Y) induced by input %d\n",ii+1);
      if (entropyDelta_ == DO_DELTA)
        printOutTS(PL_INFO,
        "Calculating D(Y) induced by input %d\n",ii+1);
    }

    //**/ ------------------------------------------------------
    //**/ create sample for the ii-th input ==> vecSamPts1D
    //**/ ------------------------------------------------------
    for (jj = 0; jj < nInputs; jj++)
    {
      if (jj != ii && corMatp->getEntry(ii,jj) != 0)
      {
        printf("WARNING: Correlation between input ");
        printf("(%d,%d) and other inputs != 0.\n",ii+1,jj+1);
        printf("         This correlation will be ignored.\n");
      }
    }

    //**/ if both input ii and ii2 have uniform distributions,
    //**/ just use LPTAU, otherwise call pdf manager
    if (pdfFlags[ii] == 0)
    {
      Sampling *sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
      sampler->setSamplingParams(nSam1, 1, 0);
      sampler->setInputBounds(iOne, &(xLower[ii]), &(xUpper[ii]));
      sampler->setOutputParams(1);
      sampler->initialize(0);
      psVector vecY1;
      psIVector vecS1;
      vecS1.setLength(nSam1);
      vecY1.setLength(nSam1);
      vecSamPts1D.setLength(nSam1);
      sampler->getSamples(nSam1,iOne,1,vecSamPts1D.getDVector(),
                     vecY1.getDVector(), vecS1.getIVector());
      delete sampler;
    }
    else
    {
      psMatrix corMat1;
      corMat1.setDim(1,1);
      corMat1.setEntry(0, 0, corMatp->getEntry(ii,ii));
      PDFManager *pdfman1 = new PDFManager();
      pdfman1->initialize(1,&pdfFlags[ii],&inputMeans[ii],
                          &inputStdevs[ii],corMat1,NULL,NULL);
      vecLB.load(1, &xLower[ii]);
      vecUB.load(1, &xUpper[ii]);
      vecOut.setLength(nSam1);
      pdfman1->genSample(nSam1, vecOut, vecLB, vecUB);
      for (jj = 0; jj < nSam1; jj++)
        vecSamPts1D[jj] = vecOut[jj];
      delete pdfman1;
    }

    //**/ ------------------------------------------------------
    //**/ create sample for the other inputs ==> vecSamPtsND
    //**/ ------------------------------------------------------
    if (nInputs > 1)
    {
      int checkUniform = 0;
      for (jj = 0; jj < nInputs; jj++) checkUniform += pdfFlags[jj];

      //**/ If all input distributions are uniform, just create
      //**/ a quasi-random sample. Otherwise, generate sample
      //**/ via the PDF manager
      if (checkUniform == 0)
      {
        Sampling *sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
        sampler->setSamplingParams(nSam2, 1, 0);
        for (jj = 0; jj < nInputs; jj++)
        {
          if (jj < ii)
          {
            vecLower2[jj] = xLower[jj];
            vecUpper2[jj] = xUpper[jj];
          }
          else if (jj > ii)
          {
            vecLower2[jj-1] = xLower[jj];
            vecUpper2[jj-1] = xUpper[jj];
          }
        }
        sampler->setInputBounds(nInputs-1, vecLower2.getDVector(),
                                vecUpper2.getDVector());
        sampler->setOutputParams(1);
        sampler->initialize(0);
        vecOut.setLength(nSam2*(nInputs-1));
        psVector vecY2;
        psIVector vecS2;
        vecY2.setLength(nSam2);
        vecS2.setLength(nSam2);
        sampler->getSamples(nSam2, nInputs-1, 1, vecOut.getDVector(),
                            vecY2.getDVector(), vecS2.getIVector());
        for (jj = 0; jj < nSam2*(nInputs-1); jj++)
          vecSamPtsND[jj] = vecOut[jj];
        delete sampler;
      }
      else
      {
        psMatrix corMat2;
        corMat2.setDim(nInputs-1, nInputs-1);
        for (jj = 0; jj < nInputs; jj++)
        {
          if (jj < ii)
          {
            vecLower2[jj] = xLower[jj];
            vecUpper2[jj] = xUpper[jj];
            vecInpFlags2[jj] = pdfFlags[jj];
            vecInpMeans2[jj] = inputMeans[jj];
            vecInpStdvs2[jj] = inputStdevs[jj];
            for (kk = 0; kk < ii; kk++)
            {
              if (kk < ii)
                corMat2.setEntry(jj, kk, corMatp->getEntry(jj,kk));
              else if (kk > ii)
                corMat2.setEntry(jj, kk-1, corMatp->getEntry(jj,kk));
            }
          }
          else if (jj > ii)
          {
            vecLower2[jj-1] = xLower[jj];
            vecUpper2[jj-1] = xUpper[jj];
            vecInpFlags2[jj-1] = pdfFlags[jj];
            vecInpMeans2[jj-1] = inputMeans[jj];
            vecInpStdvs2[jj-1] = inputStdevs[jj];
            for (kk = 0; kk < ii; kk++)
            {
              if (kk < ii)
                corMat2.setEntry(jj-1, kk, corMatp->getEntry(jj,kk));
              else if (kk > ii)
                corMat2.setEntry(jj-1, kk-1, corMatp->getEntry(jj,kk));
            }
          }
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
    tempV = vecSamPtsND.getDVector();
    for (ss2 = 0; ss2 < nSam2; ss2++)
    {
      oneSamplePt = &(tempV[ss2*(nInputs-1)]);
      offset = ss2 * nSam1 * nInputs;
      for (ss1 = 0; ss1 < nSam1; ss1++)
      {
        for (kk = 0; kk < ii; kk++)
          vecmSamPts[offset+ss1*nInputs+kk] = oneSamplePt[kk];
        for (kk = ii+1; kk < nInputs; kk++) 
          vecmSamPts[offset+ss1*nInputs+kk] = oneSamplePt[kk-1];
        vecmSamPts[offset+ss1*nInputs+ii] = vecSamPts1D[ss1];
      }
    }

    //**/ ------------------------------------------------------
    //**/ evaluate ==> vecmSamPts ==> vecYY
    //**/ ------------------------------------------------------
    nSamp2 = nSam1*nSam2;
    if (useSimulator_ == 1)
    {
      if (psConfig_.InteractiveIsOn() && printLevel > 3)
        printOutTS(PL_INFO,"Entropy1 INFO: Function evaluations\n");
      int incr = nSam2;
      if (incr < nSam1) incr = nSam1;
      for (ss1 = 0; ss1 < nSamp2; ss1+=incr)
      {
        tempV = vecmSamPts.getDVector();
        tempV = &(tempV[ss1*nInputs]);
        tempY = vecYY.getDVector();
        tempY = &(tempY[ss1]);
        status = funcIO->ensembleEvaluate(incr,nInputs,tempV,iOne,
                                          tempY, ss1+1);
        if (status != 0)
        {
          printOutTS(PL_INFO,
            "RSEntropy1 ERROR: RS evaluator returns nonzero.\n");
          printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
          exit(1);
        }
      }
    }
    else
    {
      if (psConfig_.InteractiveIsOn() && printLevel > 3)
        printOutTS(PL_INFO,"RSEntropy2 INFO: RS evaluations\n");
      int incr = nSam2;
      if (incr < nSam1) incr = nSam1;
      for (ss1 = 0; ss1 < nSamp2; ss1+=incr)
      {
        tempV = vecmSamPts.getDVector();
        tempV = &(tempV[ss1*nInputs]);
        tempY = vecYY.getDVector();
        tempY = &(tempY[ss1]);
        status = faPtr->evaluatePoint(incr,tempV,tempY);
        if (status != 0)
        {
          printOutTS(PL_INFO,
            "RSEntropy1 ERROR: RS evaluator returns nonzero.\n");
          printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
          exit(1);
        }
      }
    }

    //**/ ------------------------------------------------------
    //**/ need to compute:
    //**/ * Expectation of conditional delta (ECDel)
    //**/ * Expectation of conditional entropy (ECEnt)
    //**/   E_{X}[H(Y|X)]
    //**/   for each value of X, compute H(Y|X) induced by ~X   
    //**/ ------------------------------------------------------
    ECDel = 0; /* expectation of conditional delta */
    ECEnt = 0; /* expectation of conditional entropy */
    vecCMeans.setLength(nSam1);
    emax = -PSUADE_UNDEFINED;
    emin = +PSUADE_UNDEFINED;
    for (ss1 = 0; ss1 < nSam1; ss1++)
    {
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
      //**/ Variation of Y is due to variation of ~X
      //**/ -------------------------------------------------------
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
              printf("  Inputs %d ss1 = %6d, entropy = %10.3e ",
                     ii+1, ss1+1, ddata);
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
            printf("Entropy1 INFO: Small Ymax - Ymin = %e\n",Ymax-Ymin);
            printf("==> Assume conditional entropy = 0 for ss1 = %d\n",
                   ss1+1);
          }
        }
      }
    } /* ss1 of nSam1 */
    ECEnt /= (double) nSam1;
    ECDel /= (double) nSam1;

    //**/ ------------------------------------------------------
    //**/ * Entropy of conditional expectation (ECMean)
    //**/   H(E_{~X}[Y|X])
    //**/ * Delta of conditional expectation (DCMean)
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
    VecDelta1_[ii] = DCMean;

    //**/ compute entropy only when it is requested
    ECMean = 0;
    if (entropyDelta_ == DO_ENTROPY)
    {
      ddata = 2.0 * PABS(Ymax - Ymin) / (PABS(Ymax) + PABS(Ymin));
      if (ddata >= 1e-8)
      {
        //**/ create and load probability matrix
        matProbY.setDim(nSam1, iOne);
        for (ss1 = 0; ss1 < nSam1; ss1++)
          matProbY.setEntry(ss1,0,vecCMeans[ss1]);
        vecL1[0] = Ymin;
        vecU1[0] = Ymax;

        //**/ binning and compute entropy 
        status = computeEntropy(matProbY,vecL1,vecU1,ECMean,adaptive_);
        //if (status == 0) VecEnt1_[ii] = ECMean;
      }
      else
      {
        if (printLevel > 3)
        {
          printf("Entropy1 INFO: Small Ymax - Ymin = %e\n",Ymax-Ymin);
          printf("==> Assume entropy of conditional mean=0 for ss1 = %d.\n",
                 ss1+1);
        }
        ECMean = 0.0;
      }

      //**/ compute entropy for X: first create probability 
      //**/ matrix
      matProbX.setDim(nSam1, iOne);
      vecL1[0] =  PSUADE_UNDEFINED;
      vecU1[0] = -PSUADE_UNDEFINED;
      for (ss1 = 0; ss1 < nSam1; ss1++)
      {
        ddata = vecSamPts1D[ss1];
        matProbX.setEntry(ss1,0,ddata);
        if (ddata < vecL1[0]) vecL1[0] = ddata;
        if (ddata > vecU1[0]) vecU1[0] = ddata;
      }
      //**/ then compute entropy for X
      status = computeEntropy(matProbX,vecL1,vecU1,XEnt,adaptive_);
      if (status != 0) XEnt = 0;
    }

    if (entropyDelta_ == DO_ENTROPY)
    {
      //**/ Now ECMean = H(E_{~X}[Y])
      printf("  Input %3d: \n",ii+1);
      printf("        H(E_{~X}[Y|X])         = %10.3e (VCE-like)\n",
             ECMean);
      //**/ Now ECEnt = E_{x}[H(Y|X=x])
      printf("        E_{X}[H(Y|X)] = H(Y|X) = %10.3e\n",ECEnt);
      VecEnt1_[ii] = outputEntropy_ - ECEnt;
      printf("        H(Y) - E_{X}[H(Y|X)]   = %10.3e (TSI-Like)\n",
             VecEnt1_[ii]);
      printf("        H(X)                   = %10.3e\n",XEnt);
      printf("        H(X,Y) = H(Y|X) + H(X) = %10.3e\n",ECEnt+XEnt);
      if (printLevel > 3)
        printf("        max/min H(Y|X)         = %10.3e %10.3e\n",
               emax,emin);
    }
    if (entropyDelta_ == DO_DELTA)
    {
      printf("  Input %3d: \n",ii+1);
      printf("        D(E_{~X}[Y|X])         = %10.3e (VCE-like)\n",
             DCMean);
      printf("        E_{X}[D(Y|X)] = D(Y|X) = %10.3e\n",ECDel);
      printf("        D(Y) - E_{X}[D(Y|X)]   = %10.3e (TSI-like)\n",
             outputDelta_-ECDel);
    }

    //**/ ------------------------------------------------------
    //**/ Somehow ECMean is the same as the following
    //**/  int H(Y|X,~X) dZ = -sum_{y,~x} p(y|x,~x) log[p(y|x,~x)]
    //**/ ECMean = E_{X}[H(Y|X)]
    //**/ DCMean = E_{X}[D(Y|X)]
    //**/ compute E_{~X}[H(Y|~X)] = ECEnt2 and 
    //**/         E_{~X}[D(Y|~X)] = ECDel2
    //**/ ------------------------------------------------------
    ECEnt2 = ECDel2 = 0;
    //**/ for every point ss2 for ~X
    for (ss2 = 0; ss2 < nSam2; ss2++)
    {
      offset = ss2 * nSam1;
      //**/ ---------------------------------------------------
      //**/ need to find output max and min to partition Y
      //**/ ---------------------------------------------------
      Ymax = -PSUADE_UNDEFINED;
      Ymin =  PSUADE_UNDEFINED;
      for (ss1 = 0; ss1 < nSam1; ss1++)
      {
        ddata = vecYY[offset+ss1];
        if (ddata > Ymax) Ymax = ddata;
        if (ddata < Ymin) Ymin = ddata;
      }
      //**/ add to E_{~X}[D(Y|X)] (Ymax-Ymin wrt X for 1 pt in ~X)
      ECDel2 += (Ymax - Ymin);
      //**/ if max = min, skip entropy calculation
      if (Ymax == Ymin) continue;
      //**/ otherwise, get ready for entropy
      vecL1[0] = Ymin;
      vecU1[0] = Ymax;

      //**/ ------------------------------------------------------
      //**/ add to E_{~X}[H(Y|X)] (H(Y) wrt varying X for 1 pt in ~X)
      //**/ ------------------------------------------------------
      if (entropyDelta_ == DO_ENTROPY)
      {
        //**/ create and load probability matrix
        matProbY.setDim(nSam1, iOne);
        for (ss1 = 0; ss1 < nSam1; ss1++)
          matProbY.setEntry(ss1,0,vecYY[ss2*nSam1+ss1]);

        //**/ compute entropy
        computeEntropy(matProbY,vecL1,vecU1,ddata,adaptive_);
        ECEnt2 += ddata;
      }
    } /* ss2 of nSam2 */
    ECEnt2 /= (double) nSam2;
    ECDel2 /= (double) nSam2;
    if (entropyDelta_ == DO_ENTROPY && printLevel > 1)
    {
      printf("        E_{~X}[H(Y|~X)]        = %10.3e\n",ECEnt2);
    }
    if (entropyDelta_ == DO_DELTA && printLevel > 1)
    {
      printf("        E_{~X}[D(Y|~X)]        = %10.3e\n",ECDel2);
    }
    //**/ ------------------------------------------------------
    //**/ Now compute TSI-like quantities
    //**/ total entropy minus H(E[Y|~X]) 
    //**/ total delta   minus D(E[Y|~X]) 
    //**/ ------------------------------------------------------
    //**/ compute conditional mean E_{X}[Y|~X]
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
    //**/ compute entropy metric H(E_{X}[Y|~X])
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

        //**/ compute entropy
        status = computeEntropy(matProbY,vecL1,vecU1,ECMean2,adaptive_);
        if (status != 0) ECMean2 = 0;
      }
      else if (nSam2 > 1)
      {
        if (printLevel > 4)
        {
          printf("Entropy1 INFO: Small Ymax - Ymin = %e\n",Ymax-Ymin);
          printf("==> Assume entropy of conditional mean = 0 for ~X\n");
        }
      }
    } /* if entropyDelta_ == DO_ENTROPY */

    if (printLevel > 1 && entropyDelta_ == DO_ENTROPY)
    {
      printf("        H(E_{X}[Y|~X])         = %10.3e\n",ECMean2);
    }
    if (printLevel > 1 && entropyDelta_ == DO_DELTA)
    {
      printf("        D(E_{X}[Y|~X])         = %10.3e\n",DCMean2);
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
  if (psConfig_.InteractiveIsOn() && adata.printLevel_ >= 0)
  {
    printAsterisks(PL_INFO, 0);
#if 0
    //**/ let the calling function prints the following information
    if (entropyDelta_ == DO_ENTROPY)
    {
      for (ii = 0; ii < nInputs; ii++)
        printOutTS(PL_INFO,
          "     Expected H(Y) induced by input %d = %10.4e\n",ii+1,
          VecEnt1_[ii]);
      printOutTS(PL_INFO,
          "     Total output entropy    = %10.4e\n",outputEntropy_);
    }
    else if (entropyDelta_ == DO_DELTA)
    {
      for (ii = 0; ii < nInputs; ii++)
        printOutTS(PL_INFO,
          "     Expected D(Y) induced by input %d = %10.4e\n",ii+1,
          VecDelta1_[ii]);
      printOutTS(PL_INFO,
          "     Overal D(Y) = %10.4e\n",outputDelta_);
    }
    printAsterisks(PL_INFO, 0);
#endif
  }
  return 0.0;
}

// ************************************************************************
// compute basic statistics
// ------------------------------------------------------------------------
double RSMEntropy1Analyzer::computeBasicStat(psVector vecY)
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
int RSMEntropy1Analyzer::computeEntropy(ProbMatrix &matProbY,psVector vecL,
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
    if (psConfig_.DiagnosticsIsOn())
      printf("RSEntropy1 INFO: computeEntropy calling binAdaptive\n");
    status = matProbY.binAdaptive(nLevels_, vecL, vecU);
    if (status != 0)
    {
      if (psConfig_.DiagnosticsIsOn())
      {
        printf("RSEntropy1 ERROR: When computing entropy (a).\n");
        printf("ERROR in file %s, line %d.\n",__FILE__,__LINE__);
        printf("Note: Switch to uniform binning, but the ");
        printf("result may not be correct.\n");
      }
    }
    else
    {
      if (psConfig_.DiagnosticsIsOn())
        printf("RSEntropy1 INFO: computeEntropy processing bins\n");
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
          printf("RSEntropy1 ERROR: bin width(%d) = %e <= 0 (a)\n",
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
      if (psConfig_.DiagnosticsIsOn())
        printf("RSEntropy1 INFO: computeEntropy ends\n");
    }
  }
  //**/ compute entropy using uniform scheme or when adaptive 
  //**/ scheme fails
  if (status == -1)
  {
    status = matProbY.binUniform(nLevels_, vecL, vecU);
    if (status != 0)
    {
      if (psConfig_.DiagnosticsIsOn())
      {
        printf("RSEntropy1 ERROR: When computing entropy (b).\n");
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
// perform total entropy and total delta
// ------------------------------------------------------------------------
int RSMEntropy1Analyzer::computeTotalEntDelta(psVector vecY)
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
  if (useRS_ == 0)
  {
    printf("Entropy1 INFO: Sample size for computing total entropy = %d\n",
           nSams);
    printf("Entropy1 INFO: Binning resolution for total entropy = %d\n",
           nLevels_);
  }
  else
  {
    printf("RSEntropy1 INFO: Sample size for computing total entropy = %d\n",
           nSams);
    printf("RSEntropy1 INFO: Binning resolution for total entropy = %d\n",
           nLevels_);
  }
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
    printf("RSEntropy1: Computing total entropy.\n");
  int status = computeEntropy(matProbY, vecL1, vecU1, outputEntropy_, iOne);
  if (psConfig_.DiagnosticsIsOn())
    printf("RSEntropy1: Computing total entropy completed.\n");
  if (status != 0)
  {
    printf("RSEntropy1 INFO: computeEntropy returns nonzero.\n");
    outputEntropy_ = 0;
  }
  //else printf("Output total entropy = %e\n", outputEntropy_);
  return 0;
}

// ************************************************************************
// set internal parameters
// ------------------------------------------------------------------------
int RSMEntropy1Analyzer::setParam(int argc, char **argv)
{
  char  *request = (char *) argv[0];
  if (!strcmp(request,"ana_entropy1_nlevels")) 
    nLevels_ = *(int *) argv[1];
  else if (!strcmp(request, "ana_entropy1_entropy")) 
    entropyDelta_ = DO_ENTROPY;
  else if (!strcmp(request, "ana_entropy1_delta"))
    entropyDelta_ = DO_DELTA;
  else if (!strcmp(request, "ana_entropy1_tentropy"))
    entropyDelta_ = DO_TOTALE;
  else if (!strcmp(request, "ana_entropy1_use_simulator"))
    useSimulator_ = 1;
  else if (!strcmp(request, "ana_entropy1_nors"))
    useRS_ = 0;
  else
  {
    printOutTS(PL_ERROR, "RSEntropy1 ERROR: setParam - not valid.\n");
    printOutTS(PL_ERROR,
      "                          param = %s\n",request);    
    exit(1);
  }
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
RSMEntropy1Analyzer& RSMEntropy1Analyzer::operator=
                      (const RSMEntropy1Analyzer &)
{
  printOutTS(PL_ERROR, 
       "RSEntropy1 operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// functions for getting results
// ------------------------------------------------------------------------
int RSMEntropy1Analyzer::get_nInputs()
{
  return nInputs_;
}
double RSMEntropy1Analyzer::get_outputMean()
{
  return outputMean_;
}
double RSMEntropy1Analyzer::get_outputStd()
{
  return outputStd_;
}
double RSMEntropy1Analyzer::get_outputEntropy()
{
  return outputEntropy_;
}
double RSMEntropy1Analyzer::get_outputDelta()
{
  return outputDelta_;
}
double RSMEntropy1Analyzer::get_entropy1(int ind)
{
  if (ind < 0 || ind >= nInputs_)
  {
    printf("RSEntropy1 ERROR: get_entropy1 index error %d.\n",ind);
    return 0.0;
  }
  if (VecEnt1_.length() <= ind)
  {
    printf("RSEntropy1 ERROR: get_entropy1 has not value.\n");
    return 0.0;
  }
  return VecEnt1_[ind];
}
double RSMEntropy1Analyzer::get_delta1(int ind)
{
  //**/ delta is the difference between max and min
  if (ind < 0 || ind >= nInputs_)
  {
    printf("RSEntropy1 ERROR: get_delta1 index error %d.\n",ind);
    return 0.0;
  }
  if (VecDelta1_.length() <= ind)
  {
    printf("RSEntropy1 ERROR: get_delta1 has not value.\n");
    return 0.0;
  }
  return VecDelta1_[ind];
}

