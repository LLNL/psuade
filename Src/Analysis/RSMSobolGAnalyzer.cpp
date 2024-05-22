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
// Functions for the class RSMSobolGAnalyzer  
// Sobol' group main effect analysis (with response surface)
// AUTHOR : CHARLES TONG
// DATE   : 2007
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "PsuadeUtil.h"
#include "sysdef.h"
#include "psVector.h"
#include "psMatrix.h"
#include "Psuade.h"
#include "FuncApprox.h"
#include "RSConstraints.h"
#include "Sampling.h"
#include "PDFManager.h"
#include "PsuadeData.h"
#include "pData.h"
#include "RSMSobolGAnalyzer.h"
#include "SobolSampling.h"
#include "SobolAnalyzer.h"
#include "pdfData.h"
#include "sysdef.h"
#include "PrintingTS.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
RSMSobolGAnalyzer::RSMSobolGAnalyzer() : Analyzer()
{
  setName("RSMSOBOLG");
  method_ = 1;
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
RSMSobolGAnalyzer::~RSMSobolGAnalyzer()
{
}

// ************************************************************************
// perform analysis
// ------------------------------------------------------------------------
double RSMSobolGAnalyzer::analyze(aData &adata)
{
  //**/ ===============================================================
  //**/ display header 
  //**/ ===============================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
        "*          RS-based Group First Order Sobol' Indices \n");
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"* TO GAIN ACCESS TO DIFFERENT OPTIONS: SET\n");
    printOutTS(PL_INFO,"*\n");
    printOutTS(PL_INFO,
        "* - ana_expert mode to finetune RSMSobolG parameters, \n");
    printOutTS(PL_INFO,
        "*   (e.g. sample size for integration can be adjusted).\n");
    printOutTS(PL_INFO,
        "* - rs_expert to finetune response surface for RSMSobolG,\n");
    printOutTS(PL_INFO,
        "* - printlevel to 1 or higher to display more information.\n");
    printEquals(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ extract sample information
  //**/ ---------------------------------------------------------------
  int printLevel = adata.printLevel_;
  int nInputs  = adata.nInputs_;
  int nOutputs = adata.nOutputs_;
  int nSamples = adata.nSamples_;
  double *xLower = adata.iLowerB_;
  double *xUpper = adata.iUpperB_;
  double *XIn  = adata.sampleInputs_;
  double *YIn  = adata.sampleOutputs_;
  int outputID = adata.outputID_;
  PsuadeData *ioPtr = adata.ioPtr_;
  int *pdfFlags     = adata.inputPDFs_;
  double *inputMeans = adata.inputMeans_;
  double *inputStdvs = adata.inputStdevs_;
  int hasPDF = 0, ii;
  if (pdfFlags != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (pdfFlags[ii] != 0) hasPDF = 1;
    for (ii = 0; ii < nInputs; ii++)
    {
      if (pdfFlags[ii] == PSUADE_PDF_SAMPLE)
      {
        printOutTS(PL_ERROR,
          "* RSMSobolG ERROR: S PDF type currently not supported.\n");
        return PSUADE_UNDEFINED;
      }
    }
  }
  if (psConfig_.InteractiveIsOn())
  {
    if (hasPDF == 0) 
      printOutTS(PL_INFO,
                 "RSMSobolG INFO: All uniform distributions.\n");
    else
    {
      printOutTS(PL_INFO,
          "RSMSobolG INFO: Non-uniform distributions detected,\n");
      printOutTS(PL_INFO,
          "                which will be used in this analysis.\n");
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 1 || nSamples <= 0 || nOutputs <= 0)
  {
    printOutTS(PL_ERROR,"RSMSobolG ERROR: Invalid arguments.\n");
    printOutTS(PL_ERROR,"   nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR,"   nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR,"   nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 
  if (nInputs <= 2)
  {
    printOutTS(PL_ERROR,"RSMSobolG INFO: nInputs == 2.\n");
    printOutTS(PL_ERROR, 
         "   You do not need to perform this when nInputs = 2.\n");
    return PSUADE_UNDEFINED;
  }
  if (outputID >= nOutputs || outputID < 0)
  {
    printOutTS(PL_ERROR,
         "RSMSobolG ERROR: Invalid output ID (%d).\n",outputID);
    return PSUADE_UNDEFINED;
  }
  if (ioPtr == NULL)
  {
    printOutTS(PL_ERROR, "RSMSobolG ERROR: No data.\n");
    return PSUADE_UNDEFINED;
  } 
  int status = 0;
  for (ii = 0; ii < nSamples; ii++)
     if (YIn[nOutputs*ii+outputID] > 0.9*PSUADE_UNDEFINED) 
       status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR,
        "RSMSobolG ERROR: Some outputs are undefined. Prune\n");
    printOutTS(PL_ERROR,
        "                 the undefined sample points first.\n");
    return PSUADE_UNDEFINED;
  }

  //**/ ---------------------------------------------------------------
  //**/ get group information
  //**/ ---------------------------------------------------------------
  //**/ first check if the group file has been passed in via adata
  char cfname[1001],pString[1001],*cString,winput1[1001],winput2[1001];
  FILE *fp = fopen(adata.grpFileName_, "r");
  if (fp != NULL) 
  {
    printOutTS(PL_INFO,"RSMSOBOLG: Found group file found (1).\n");
    int length = strlen(adata.grpFileName_);
    strncpy(cfname, adata.grpFileName_, length);
    fclose(fp);
  }
  else if (fp == NULL && psConfig_.InteractiveIsOn())
  {
    //**/ ask user for the group file, if interactive is on
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
      "This group sensitivity analysis requires a file specifying\n");
    printOutTS(PL_INFO,
      "group information in the form of : \n");
    printOutTS(PL_INFO,
      "line 1: PSUADE_BEGIN\n");
    printOutTS(PL_INFO,
      "line 2: <d> specifying the number of groups\n");
    printOutTS(PL_INFO,
      "line 3 to line <d>+2: group number, size, input numbers\n");
    printOutTS(PL_INFO,"last line: PSUADE_END\n");
    while (1)
    {
      printOutTS(PL_INFO,"Enter the group file name : ");
      scanf("%s", cfname);
      fp = fopen(cfname, "r");
      if (fp != NULL) 
      {
        fclose(fp);
        break;
      }
      else 
        printOutTS(PL_ERROR,
         "ERROR: File %s not found (or file name too long).\n",cfname);
    }
  }
  else
  {
    printOutTS(PL_ERROR,
      "ERROR: Group sensitivity analysis group file not provided.\n");
    exit(1);
  }
  //**/ at this point group file should be in cfname
  psIMatrix matGrpMembers;
  matGrpMembers.setFormat(PS_MAT2D);
  //cfname[strlen(cfname)-1] = '\0';
  int errFlag = readGrpInfoFile(cfname, nInputs, matGrpMembers);
  if (errFlag != 0)
  {
    printOutTS(PL_ERROR,
      "ERROR: Failed to read group information file.\n");
    exit(1);
  } 
  int nGroups = matGrpMembers.nrows();
  if (printLevel > 3) 
  {
    printOutTS(PL_INFO, 
        "RSMSobolG INFO: Group information: \n");
    matGrpMembers.print();
  }

  //**/ ---------------------------------------------------------------
  //**/ analyze whether group information consistent with joint PDFs
  //**/ ---------------------------------------------------------------
  pData pCorMat;
  ioPtr->getParameter("input_cor_matrix", pCorMat);
  psMatrix *corMatp = (psMatrix *) pCorMat.psObject_;
  int ii2, jj, ind1, ind2;
  for (ii = 0; ii < nGroups; ii++)
  {
    for (ii2 = 0; ii2 < nInputs; ii2++)
    {
      for (jj = 0; jj < nInputs; jj++)
      {
        ind1 = matGrpMembers.getEntry(ii,ii2);
        ind2 = matGrpMembers.getEntry(ii,jj);
        if (((ind1 != 0 && ind2 == 0) || (ind1 == 0 && ind2 != 0)) &&
            corMatp->getEntry(ii2,jj) != 0.0)
        {
          printOutTS(PL_ERROR,
               "RSMSobolG INFO: Currently cannot handle\n");
          printOutTS(PL_ERROR,
               "          correlated inputs (joint PDF)\n");
          printOutTS(PL_ERROR,
               "          across different groups.\n");
          matGrpMembers.clean();
          return PSUADE_UNDEFINED;
        }
      }
    }
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

  //**/ ---------------------------------------------------------------
  //**/ build response surface
  //**/ ---------------------------------------------------------------
  int rstype=-1;
  if (adata.faType_ < 0)
  {
    printf("Select response surface. Options are: \n");
    writeFAInfo(0); 
    strcpy(pString, "Choose response surface: ");
    rstype = getInt(0, PSUADE_NUM_RS, pString);
  }
  else rstype = adata.faType_;

  FuncApprox *faPtr = genFA(rstype, nInputs, 0, nSamples);
  faPtr->setBounds(xLower, xUpper);
  faPtr->setOutputLevel(0);
  psVector vecYIn2;
  vecYIn2.setLength(nSamples);
  for (ii = 0; ii < nSamples; ii++) 
    vecYIn2[ii] = YIn[ii*nOutputs+outputID];
  status = faPtr->initialize(XIn, vecYIn2.getDVector());

  //**/ ---------------------------------------------------------------
  //**/  get internal parameters 
  //**/ ---------------------------------------------------------------
  int nSubSamplesG=50000, nSubSamplesN=1000;
  if (psConfig_.InteractiveIsOn() && psConfig_.AnaExpertModeIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"This analysis creates a sample of ");
    printOutTS(PL_INFO,"size M1 for each subgroup of inputs\n");
    printOutTS(PL_INFO,"and M2 for the other inputs when ");
    printOutTS(PL_INFO,"computing group sensitivity indices.\n");
    printOutTS(PL_INFO,"The total sample size is thus:\n");
    printOutTS(PL_INFO,"      N = M1 * M2 * nGroups.\n");
    printOutTS(PL_INFO,"  Default M1 = %d.\n", nSubSamplesG);
    printOutTS(PL_INFO,"  Default M2 = %d.\n", nSubSamplesN);
    printOutTS(PL_INFO,"You can change M1 and M2 (make sure M1 >> M2).\n");
    printOutTS(PL_INFO,"NOTE: Large M1 and M2 can take a long time.\n");
    printEquals(PL_INFO, 0);
    snprintf(pString,100,"Enter M1 (suggestion: 20000 - 1000000) : ");
    nSubSamplesG = getInt(1000, 1000000, pString);
    snprintf(pString,100,"Enter M2 (suggestion: 100 - 10000) : ");
    nSubSamplesN = getInt(100, 10000, pString);
    printAsterisks(PL_INFO, 0);
  }
  else
  {
    cString = psConfig_.getParameter("RSMSobolG_nsubsamples_ingroup");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nSubSamplesG);
      if (nSubSamplesG < 10000)
      {
        printOutTS(PL_INFO,
             "RSMSobolG INFO: nSubSamplesG should be >= 20000.\n");
        nSubSamplesG = 20000;
      }
      else
      {
        printOutTS(PL_INFO,
             "RSMSobolG INFO: nSubSamplesG = %d (config).\n",
             nSubSamplesG);
      }
    }
    cString = psConfig_.getParameter("RSMSobolG_nsubsamples_outgroup");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nSubSamplesN);
      if (nSubSamplesN < 100)
      {
        printOutTS(PL_INFO,
             "RSMSobolG INFO: nSubSamplesN should be >= 100.\n");
        nSubSamplesN = 100;
      }
      else
      {
        printOutTS(PL_INFO,
             "RSMSobolG INFO: nSubSamplesN = %d (config).\n",
             nSubSamplesN);
      }
    }
    if (psConfig_.InteractiveIsOn())
    {
      printOutTS(PL_INFO,"RSMSobolG: Default M1 = %d.\n",nSubSamplesG);
      printOutTS(PL_INFO,"RSMSobolG: Default M2 = %d.\n",nSubSamplesN);
      printOutTS(PL_INFO,
        "To change these settings, re-run with ana_expert mode on.\n");
    }
  }
  if (psConfig_.InteractiveIsOn()) printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/  use response surface to compute total variance
  //**/ ---------------------------------------------------------------
  //**/ ------------------------
  //**/ use a large sample size
  //**/ ------------------------
  int nSamp = 1000000;
  if (psConfig_.InteractiveIsOn()) 
  {
    printOutTS(PL_INFO,
         "RSMSobolG INFO: creating a sample for basic statistics.\n");
    printOutTS(PL_INFO,"                sample size = %d\n", nSamp);
  }
  //**/ ------------------------
  //**/ allocate space
  //**/ ------------------------
  psVector  vecXT, vecYT, vecUB, vecLB, vecOut;
  psIVector vecST;
  vecXT.setLength(nSamp*nInputs);
  vecYT.setLength(nSamp);
  vecST.setLength(nSamp);

  if (psConfig_.InteractiveIsOn()) 
    printOutTS(PL_INFO, 
       "RSMSobolG: Compute statistics, sample size = %d\n",nSamp);
       
  //**/ ------------------------
  //**/ create a sample
  //**/ ------------------------
  PDFManager *pdfman=NULL;
  Sampling   *sampler=NULL;
  if (hasPDF == 1)
  {
    printOutTS(PL_INFO,"RSMSobolG INFO: Non-uniform PDFs detected.\n");
    pdfman = new PDFManager();
    pdfman->initialize(nInputs,pdfFlags,inputMeans,
                       inputStdvs,*corMatp,NULL,NULL);
    vecLB.load(nInputs, xLower);
    vecUB.load(nInputs, xUpper);
    vecOut.setLength(nSamp*nInputs);
    pdfman->genSample(nSamp, vecOut, vecLB, vecUB);
    for (ii = 0; ii < nSamp*nInputs; ii++) vecXT[ii] = vecOut[ii];
    delete pdfman;
  }
  else
  {
    if (nInputs > 51)
    {
      sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
      sampler->setSamplingParams(nSamp, 1, 1);
    }
    else
    {
      sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
      sampler->setSamplingParams(nSamp, 1, 0);
    }
    sampler->setInputBounds(nInputs, xLower, xUpper);
    sampler->setOutputParams(1);
    sampler->initialize(0);
    sampler->getSamples(nSamp, nInputs, 1, vecXT.getDVector(), 
                        vecYT.getDVector(), vecST.getIVector());
    delete sampler;
  }

  //**/ ------------------------
  //**/ evaluate 
  //**/ ------------------------
  if (psConfig_.InteractiveIsOn() && printLevel > 1)
    printOutTS(PL_INFO,
      "RSMSobolG: Running the sample with response surface...\n");
  faPtr->evaluatePoint(nSamp, vecXT.getDVector(), vecYT.getDVector());
  if (psConfig_.InteractiveIsOn() && printLevel > 1)
    printOutTS(PL_INFO,
      "RSMSobolG: Running the sample with response surface complete.\n");

  //**/ ------------------------
  //**/ apply filters
  //**/ ------------------------
  double *XT = vecXT.getDVector();
  double *YT = vecYT.getDVector();
  double *oneSamplePt, ddata;
  for (ii = 0; ii < nSamp; ii++)
  {
    oneSamplePt = &(XT[ii*nInputs]);
    status = 1;
    if (constrPtr != NULL)
      ddata = constrPtr->evaluate(oneSamplePt,YT[ii],status);
    if (status == 0) vecYT[ii] = PSUADE_UNDEFINED;
  }

  //**/ ------------------------
  //**/ compute statistics
  //**/ ------------------------
  double dmean = 0.0;
  int sCnt = 0;
  for (ii = 0; ii < nSamp; ii++)
  {
    if (vecYT[ii] != PSUADE_UNDEFINED)
    {
      dmean += vecYT[ii];
      sCnt++;
    }
  }
  if (sCnt > 1) dmean /= (double) sCnt;
  else
  {
    printOutTS(PL_ERROR, 
          "RSMSobolG ERROR: Too few samples that satisify the\n");
    printOutTS(PL_ERROR,"constraints (%d out of %d)\n",sCnt,nSamp);
    delete faPtr;
    return PSUADE_UNDEFINED;
  }
  double variance = 0.0;
  for (ii = 0; ii < nSamp; ii++)
  {
    if (vecYT[ii] != PSUADE_UNDEFINED)
      variance += (vecYT[ii] - dmean) * (vecYT[ii] - dmean) ;
  }
  variance /= (double) sCnt;
  if (psConfig_.InteractiveIsOn() && printLevel > 3)
  {
    printOutTS(PL_INFO,
       "RSMSobolG: Sample mean    (based on N = %d) = %10.3e\n",
       sCnt, dmean);
    printOutTS(PL_INFO,
       "RSMSobolG: Sample std dev (based on N = %d) = %10.3e\n",
       sCnt, sqrt(variance));
  }
  if (variance == 0.0) variance = 1.0;

  //**/ ===============================================================
  //**/ use Sobol' method if requested
  //**/ ===============================================================
  if (method_ == 1)
  {
    if (psConfig_.DiagnosticsIsOn())
      printf("Entering RSMSobolGAnalzer::analyze::Sobol\n");
    int nPerBlk = 1000000 / (nGroups + 2);
    printf("RSMSobol1 Sobol: nSams = 1000000\n");
    int nSobolSams = nPerBlk*(nGroups+2), iOne=1;

    //**/ create M1 and M2 assuming input distributions
#if 0
    pdfman = new PDFManager();
    pdfman->initialize(nInputs,pdfFlags,inputMeans,inputStdvs,
                       *corMatp,NULL,NULL);
    vecLB.load(nInputs, xLower);
    vecUB.load(nInputs, xUpper);
    vecOut.setLength(nPerBlk*2*nInputs);
    pdfman->genSample(nPerBlk*2, vecOut, vecLB, vecUB);
    delete pdfman;
#endif
    pdfData pdfObj;
    pdfObj.nInputs_  = nInputs;
    pdfObj.nSamples_ = nPerBlk;
    pdfObj.VecPTypes_.load(nInputs, pdfFlags);
    pdfObj.VecParam1_.load(nInputs, inputMeans);
    pdfObj.VecParam2_.load(nInputs, inputStdvs);
    pdfObj.VecLBs_.load(nInputs, xLower);
    pdfObj.VecUBs_.load(nInputs, xUpper);
    pdfObj.MatCor_ = (*corMatp);

    //**/ Step 1: Create a Sobol' sample
    SobolSampling *sobolSampler = new SobolSampling();
    sobolSampler->setPrintLevel(-2);
    sobolSampler->setInputBounds(nInputs,xLower,xUpper);
    sobolSampler->setOutputParams(iOne);
    sobolSampler->setSamplingParams(nSobolSams, 1, 0);
    //sobolSampler->setM1M2(vecOut);
    sobolSampler->createM1M2(pdfObj);
    sobolSampler->initialize3(0, matGrpMembers);
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

    //**/ Step 3: filter out infeasible points
    int count, kk;
    double *tempV;
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
    strcpy(sobolAPtr.grpFileName_, cfname);
    for (ii = 0; ii < nSobolSams; ii++) vecSobolS[ii] = 1;
    sobolAPtr.sampleStates_  = vecSobolS.getIVector();
    sobolAPtr.iLowerB_  = xLower;
    sobolAPtr.iUpperB_  = xUpper;
    sobolAPtr.outputID_ = 0;
    sobolAPtr.ioPtr_    = NULL;
    sobolAnalyzer->analyze3(sobolAPtr);
    //**/ these have been normalized
    int nGroups = sobolAnalyzer->get_nGroups();
    VecVCEG_.setLength(nGroups);
    for (ii = 0; ii < nGroups; ii++)
      VecVCEG_[ii] = sobolAnalyzer->get_SG(ii);
    delete sobolAnalyzer;
    delete faPtr;

    //**/ Step 5: print results
    if (psConfig_.InteractiveIsOn())
    {
      printAsterisks(PL_INFO, 0);
      for (ii = 0; ii < nGroups; ii++)
      {
        printOutTS(PL_INFO,
          "Sobol' group-order index for group %3d = %10.3e\n",
          ii+1, VecVCEG_[ii]);
      }
    }
    pData *pObj = NULL;
    if (ioPtr != NULL)
    {
      pObj = ioPtr->getAuxData();
      pObj->nDbles_ = nGroups;
      pObj->dbleArray_ = new double[nGroups];
      //**/ returns unnormalized results
      for (ii = 0; ii < nGroups; ii++)
        pObj->dbleArray_[ii] = VecVCEG_[ii] * variance;
      pObj->dbleData_ = variance;
    }
    return 0;
  }

  //**/ ===============================================================
  //**/  use response surface to perform Sobol group test
  //**/ ===============================================================
  //**/ ---------------------------------------------------------------
  //**/ set up the sampling method for out of group
  //**/ ---------------------------------------------------------------
  int      kk, ir, nInputsG, nInputsN, totalCnt, currNSamples;
  double   vce, ecv;
  psMatrix corMat;
  psVector vecCLs, vecCUs, vecXTG, vecXTN, vecYY, vecMeans, vecVars;
  vecCLs.setLength(nInputs);
  vecCUs.setLength(nInputs);
  vecXTG.setLength(nSubSamplesG*nInputs);
  vecXTN.setLength(nSubSamplesN*nInputs);
  vecYY.setLength(nSubSamplesN+nSubSamplesG);
  vecMeans.setLength(nSubSamplesG);
  vecVars.setLength(nSubSamplesG);
  psIVector vecBins, vecIArray;
  vecBins.setLength(nSubSamplesG);
  vecIArray.setLength(nInputs);
  psVector vecMSamPts;
  vecMSamPts.setLength(nInputs*nSubSamplesN);
  psVector vecInpMeansG, vecInpStdsG, vecInpMeansN, vecInpStdsN;
  vecInpMeansG.setLength(nInputs);
  vecInpMeansN.setLength(nInputs);
  vecInpStdsG.setLength(nInputs);
  vecInpStdsN.setLength(nInputs);
  psIVector vecPdfFlagsG, vecPdfFlagsN;
  vecPdfFlagsG.setLength(nInputs);
  vecPdfFlagsN.setLength(nInputs);
  PDFManager *pdfmanN=NULL, *pdfmanG=NULL;

  //**/ loop through each pair of inputs
  VecVCEG_.setLength(nGroups);
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printf("**                Group Sensitivity Analysis Summary\n");
    printEquals(PL_INFO, 0);
  }
  printf("RSMSobolG(ni): nSam1, nSam2 = %d %d\n",nSubSamplesN,
         nSubSamplesG);
  for (ii = 0; ii < nGroups; ii++)
  {
    if (psConfig_.InteractiveIsOn() && printLevel > 1)
    {
      printOutTS(PL_INFO, "RSMSobolG: processing group %d\n", ii+1);
      printOutTS(PL_INFO, "           group members: ");
      for (jj = 0; jj < nInputs; jj++)
      {
        if (matGrpMembers.getEntry(ii,jj) != 0) 
          printOutTS(PL_INFO,"%d ",jj+1);
      }
      printOutTS(PL_INFO, "\n");
    }
    //**/ create sample for the out of group
    nInputsN = 0;
    for (jj = 0; jj < nInputs; jj++)
    {
      if (matGrpMembers.getEntry(ii,jj) == 0)
      {
        vecCLs[nInputsN] = xLower[jj];
        vecCUs[nInputsN] = xUpper[jj];
        vecIArray[nInputsN] = jj;
        nInputsN++;
      }
    }
    if (hasPDF == 1) 
    {
      nInputsN = 0;
      for (jj = 0; jj < nInputs; jj++)
      {
        if (matGrpMembers.getEntry(ii,jj) == 0)
        {
          vecPdfFlagsN[nInputsN] = pdfFlags[jj];
          vecInpMeansN[nInputsN] = inputMeans[jj];
          vecInpStdsN[nInputsN]  = inputStdvs[jj];
          nInputsN++;
        }
      }
      corMat.setDim(nInputsN, nInputsN);
      for (jj = 0; jj < nInputsN; jj++)
        for (ii2 = 0; ii2 < nInputsN; ii2++)
          corMat.setEntry(jj,ii2,corMatp->getEntry(vecIArray[jj],
                                                   vecIArray[ii2])); 
      pdfmanN = new PDFManager();
      pdfmanN->initialize(nInputsN, vecPdfFlagsN.getIVector(), 
                  vecInpMeansN.getDVector(), vecInpStdsN.getDVector(), 
                  corMat,NULL,NULL);
      vecLB.load(nInputsN, vecCLs.getDVector());
      vecUB.load(nInputsN, vecCUs.getDVector());
      vecOut.setLength(nSubSamplesN*nInputsN);
      pdfmanN->genSample(nSubSamplesN, vecOut, vecLB, vecUB);
      for (jj = 0; jj < nSubSamplesN*nInputsN; jj++)
        vecXTN[jj] = vecOut[jj];
      delete pdfmanN;
    }
    else
    {
      if (nInputsN > 51)
           sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
      else sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
      sampler->setInputBounds(nInputsN, vecCLs.getDVector(), 
                              vecCUs.getDVector());
      sampler->setOutputParams(1);
      sampler->setSamplingParams(nSubSamplesN, 1, 1);
      sampler->initialize(0);
      vecST.setLength(nSubSamplesN);
      sampler->getSamples(nSubSamplesN,nInputsN,1,vecXTN.getDVector(),
                          vecYY.getDVector(),vecST.getIVector());
      delete sampler;
    }

    //**/ use 3 levels of refinements to compute confidence interval
    currNSamples = nSubSamplesG / 4;
    nInputsG = 0;
    for (jj = 0; jj < nInputs; jj++)
    {
      if (matGrpMembers.getEntry(ii,jj) != 0)
      {
        vecCLs[nInputsG] = xLower[jj];
        vecCUs[nInputsG] = xUpper[jj];
        vecIArray[nInputsG] = jj;
        nInputsG++;
      }
    }
    if (hasPDF == 1) 
    {
      nInputsG = 0;
      for (jj = 0; jj < nInputs; jj++)
      {
        if (matGrpMembers.getEntry(ii,jj) != 0)
        {
          vecPdfFlagsG[nInputsG] = pdfFlags[jj];
          vecInpMeansG[nInputsG] = inputMeans[jj];
          vecInpStdsG[nInputsG]  = inputStdvs[jj];
          nInputsG++;
        }
      }
      corMat.setDim(nInputsG, nInputsG);
      for (jj = 0; jj < nInputsG; jj++)
        for (ii2 = 0; ii2 < nInputsG; ii2++)
          corMat.setEntry(jj,ii2,
                 corMatp->getEntry(vecIArray[jj],vecIArray[ii2])); 
      vecLB.load(nInputsG, vecCLs.getDVector());
      vecUB.load(nInputsG, vecCUs.getDVector());
      vecOut.setLength(nSubSamplesG*nInputsG);
    }

    for (ir = 0; ir < 3; ir++)
    {
      printOutTS(PL_DETAIL,
           "   processing refinement %d\n",ir+1);
      printOutTS(PL_DETAIL,"   nSamplesG = %d, nSamplesN = %d\n",
                 currNSamples, nSubSamplesN);
      if (hasPDF == 1) 
      {
        pdfmanG = new PDFManager();
        pdfmanG->initialize(nInputsG, vecPdfFlagsG.getIVector(), 
                   vecInpMeansG.getDVector(),vecInpStdsG.getDVector(), 
                   corMat,NULL,NULL);
        pdfmanG->genSample(nSubSamplesG, vecOut, vecLB, vecUB);
        for (jj = 0; jj < nSubSamplesG*nInputsG; jj++)
          vecXTG[jj] = vecOut[jj];
        delete pdfmanG;
      }
      else
      {
        if (nInputsN > 51)
             sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
        else sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
        sampler->setInputBounds(nInputsG, vecCLs.getDVector(), 
                                vecCUs.getDVector());
        sampler->setOutputParams(1);
        sampler->setSamplingParams(nSubSamplesG, 1, 1);
        sampler->initialize(0);
        vecST.setLength(nSubSamplesG);
        sampler->getSamples(nSubSamplesG,nInputsG,1,vecXTG.getDVector(),
                  vecYY.getDVector(),vecST.getIVector());
        delete sampler;
      }

      //**/ evaluate sample points
      if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
        printOutTS(PL_DETAIL,"   evaluating ... \n");
      for (ii2 = 0; ii2 < nSubSamplesG; ii2++)
      {
        //**/ set the within group data
        for (jj = 0; jj < nSubSamplesN; jj++)
        {
          sCnt = 0;
          for (kk = 0; kk < nInputs; kk++)
          {
            if (matGrpMembers.getEntry(ii,kk) != 0)
            {
              vecMSamPts[jj*nInputs+kk] = vecXTG[ii2*nInputsG+sCnt];
              sCnt++;
            }
          }
        }

        //**/ set the outside group data
        for (jj = 0; jj < nSubSamplesN; jj++)
        {
          sCnt = 0;
          for (kk = 0; kk < nInputs; kk++)
          {
            if (matGrpMembers.getEntry(ii,kk) == 0)
            {
              vecMSamPts[jj*nInputs+kk] = vecXTN[jj*nInputsN+sCnt];
              sCnt++;
            }
          }
        }

        //**/ evaluate
        faPtr->evaluatePoint(nSubSamplesN,vecMSamPts.getDVector(),
                             vecYY.getDVector());

        //**/ go through all filters the sample point and evaluate
        double *mSamplePts = vecMSamPts.getDVector();
        for (jj = 0; jj < nSubSamplesN; jj++)
        {
          status = 1;
          if (constrPtr != NULL)
            ddata = constrPtr->evaluate(&(mSamplePts[jj*nInputs]),
                                        vecYY[jj],status);
          if (status == 0) vecYY[jj] = PSUADE_UNDEFINED;
        }

        //**/ compute the mean at each input group levels
        vecMeans[ii2] = 0.0;
        sCnt = 0;
        for (jj = 0; jj < nSubSamplesN; jj++)
        {
          if (vecYY[jj] != PSUADE_UNDEFINED)
          {
            vecMeans[ii2] += vecYY[jj];
            sCnt++;
          }
        }
        vecBins[ii2] = sCnt;
        if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
          if (sCnt < 1 && printLevel >= 5)
            printOutTS(PL_DUMP, "RSMSobolG WARNING: subsample size = 0.\n");
        if (sCnt < 1) vecMeans[ii2] = PSUADE_UNDEFINED;
        else          vecMeans[ii2] /= (double) sCnt;

        //**/ compute the variance  at each input pair levels
        vecVars[ii2] = 0.0;
        ddata = vecMeans[ii2];
        for (jj = 0; jj < nSubSamplesN; jj++)
        {
          if (vecYY[jj] != PSUADE_UNDEFINED)
            vecVars[ii2] += (vecYY[jj] - ddata) * (vecYY[jj] - ddata);
        }
        if (sCnt < 1) vecVars[ii2] = PSUADE_UNDEFINED;
        else          vecVars[ii2] /= (double) sCnt;

        if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
        {
          printOutTS(PL_DUMP, "RSMSobolG: Group %d\n", ii+1);
          printOutTS(PL_DUMP, "  refinement = %d, size = %d (%d),",ir,
                     sCnt, nSubSamplesN);
          printOutTS(PL_DUMP, 
             " mean = %12.4e, var = %12.4e\n",vecMeans[ii2],vecVars[ii2]);
        }
      }

      //**/ compute the variance of the means for each group
      if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
        printOutTS(PL_DETAIL,"   computing statistics ... \n");
      totalCnt = 0;
      for (ii2 = 0; ii2 < nSubSamplesG; ii2++) totalCnt += vecBins[ii2];
      if (totalCnt == 0)
      {
        printOutTS(PL_ERROR, 
                   "RSMSobolG ERROR: empty constrained space.\n");
        exit(1);
      }
 
      dmean = 0.0;
      for (ii2 = 0; ii2 < nSubSamplesG; ii2++)
      {
        if (vecMeans[ii2] != PSUADE_UNDEFINED)
          dmean += vecMeans[ii2] * vecBins[ii2] / totalCnt;
      }

      vce = 0.0;
      for (ii2 = 0; ii2 < nSubSamplesG; ii2++)
        if (vecMeans[ii2] != PSUADE_UNDEFINED)
          vce += (vecMeans[ii2] - dmean) * (vecMeans[ii2] - dmean) *
                 vecBins[ii2] / totalCnt;

      //**/ compute the mean of the variances for each input group
      ecv = 0.0;
      for (ii2 = 0; ii2 < nSubSamplesG; ii2++)
      {
        if (vecVars[ii2] != PSUADE_UNDEFINED)
          ecv += vecVars[ii2] * vecBins[ii2] / totalCnt;
      }

      if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
      {
        printOutTS(PL_DETAIL,
             "   Unnormalized ECV (refinement =%3d) ",ir+1);
        printOutTS(PL_DETAIL,
             "for input group %3d = %10.3e\n", ii+1, ecv);
        printOutTS(PL_DETAIL,
             "   Unnormalized VCE (refinement =%3d) ",ir+1);
        printOutTS(PL_DETAIL,
             "for input group %3d = %10.3e\n", ii+1, vce);
      }
      currNSamples *= 2;
    }
    VecVCEG_[ii] = vce;
  }
  if (psConfig_.InteractiveIsOn() || psConfig_.DiagnosticsIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printf("*          Group Main Effect Summary\n");
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < nGroups; ii++)
      printOutTS(PL_INFO,
        "** VCE for input group %3d = %10.3e (normalized = %10.3e)\n",
        ii+1, VecVCEG_[ii], VecVCEG_[ii]/variance);
    printAsterisks(PL_INFO, 0);
  }
    
  //**/ ---------------------------------------------------------------
  //**/ return more detailed data
  //**/ ---------------------------------------------------------------
  pData *pObj = NULL;
  if (ioPtr != NULL)
  {
    pObj = ioPtr->getAuxData();
    {
      pObj->nDbles_ = nGroups;
      pObj->dbleArray_ = new double[nGroups];
      //**/ returns unnormalized results
      for (ii = 0; ii < nGroups; ii++)
        pObj->dbleArray_[ii] = VecVCEG_[ii];
      pObj->dbleData_ = variance;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ clean up
  //**/ ---------------------------------------------------------------
  if (constrPtr != NULL) delete constrPtr;
  delete faPtr;
  return 0.0;
}

// ************************************************************************
// set internal parameters
// ------------------------------------------------------------------------
int RSMSobolGAnalyzer::setParam(int argc, char **argv)
{
  char  *request = (char *) argv[0];
  if      (!strcmp(request,"ana_rssobolg_ni"))    method_ = 0;
  else if (!strcmp(request,"ana_rssobolg_sobol")) method_ = 1; 
  return 0; 
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
RSMSobolGAnalyzer& RSMSobolGAnalyzer::operator=(const RSMSobolGAnalyzer &)
{
   printOutTS(PL_ERROR, "RSMSobolG operator= ERROR: operation not allowed.\n");
   exit(1);
   return (*this);
}

