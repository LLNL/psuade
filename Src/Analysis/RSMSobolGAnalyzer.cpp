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
#include "sysdef.h"
#include "PrintingTS.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
RSMSobolGAnalyzer::RSMSobolGAnalyzer() : Analyzer()
{
   setName("RSMSOBOLG");
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
  int ii, ii2, jj, kk, ir, index, length, sCnt, nGroups, groupID;
  int currNSamples, nSubSamplesG, nInputsG, nInputsN;
  int nSubSamplesN, nSamp, totalCnt;
  size_t compFlag;
  double ddata, variance, vce, dmean, ecv;
  FILE   *fp=NULL;
  char   cfname[1001], pString[1001], *cString, winput1[1001];
  char   winput2[1001], lineIn[1001];
  psVector   vecIn, vecOut, vecUB, vecLB;
  pData      pCorMat;
  psMatrix   *corMatp, corMat;
  PDFManager *pdfman, *pdfmanN, *pdfmanG;
  Sampling   *sampler;

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
  //**/ extract test data
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
  double *inputMeans  = adata.inputMeans_;
  double *inputStdevs = adata.inputStdevs_;
  int noPDF = 1;
  if (pdfFlags != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (pdfFlags[ii] != 0) noPDF = 0;
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
  if (noPDF == 1) 
    printOutTS(PL_INFO,"RSMSobolG INFO: all uniform distributions.\n");
  else
  {
    printOutTS(PL_INFO,
        "RSMSobolG INFO: non-uniform distributions detected,\n");
    printOutTS(PL_INFO,
        "                which will be used in this analysis.\n");
  }

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 1 || nSamples <= 0 || nOutputs <= 0)
  {
    printOutTS(PL_ERROR, "RSMSobolG ERROR: invalid arguments.\n");
    printOutTS(PL_ERROR, "   nInputs  = %d\n", nInputs);
    printOutTS(PL_ERROR, "   nOutputs = %d\n", nOutputs);
    printOutTS(PL_ERROR, "   nSamples = %d\n", nSamples);
    return PSUADE_UNDEFINED;
  } 
  if (nInputs <= 2)
  {
    printOutTS(PL_ERROR, "RSMSobolG INFO: nInputs == 2.\n");
    printOutTS(PL_ERROR, 
         "   You do not need to perform this when nInputs = 2.\n");
    return PSUADE_UNDEFINED;
  }
  if (outputID >= nOutputs || outputID < 0)
  {
    printOutTS(PL_ERROR,
         "RSMSobolG ERROR: invalid output ID (%d).\n",outputID);
    return PSUADE_UNDEFINED;
  }
  if (ioPtr == NULL)
  {
    printOutTS(PL_ERROR, "RSMSobolG ERROR: no data.\n");
    return PSUADE_UNDEFINED;
  } 
  int status = 0;
  for (ii = 0; ii < nSamples; ii++)
     if (YIn[nOutputs*ii+outputID] > 0.9*PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    printOutTS(PL_ERROR,
        "RSMSobolG ERROR: Some outputs are undefined. Prune\n");
    printOutTS(PL_ERROR,
        "                 the undefined sample points first.\n");
    return PSUADE_UNDEFINED;
  }
  psVector vecYIn2;
  vecYIn2.setLength(nSamples);
  for (ii = 0; ii < nSamples; ii++) 
    vecYIn2[ii] = YIn[ii*nOutputs+outputID];

  //**/ ---------------------------------------------------------------
  //**/ get group information
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"To use this function, you need to provide a file\n");
  printOutTS(PL_INFO,"specifying group information, in the form of : \n");
  printOutTS(PL_INFO,"line 1: PSUADE_BEGIN\n");
  printOutTS(PL_INFO,"line 2: <d> specifying the number of groups\n");
  printOutTS(PL_INFO,
       "line 3 to line <d>+2: group number, size, input numbers\n");
  printOutTS(PL_INFO,"last line: PSUADE_END\n");
  while (1)
  {
    printOutTS(PL_INFO,"Enter the group file : ");
    scanf("%s", cfname);
    fp = fopen(cfname, "r");
    if (fp != NULL) 
    {
      fclose(fp);
      break;
    }
    else 
      printOutTS(PL_ERROR,
           "ERROR : file not found (or file name too long).\n");
  }
  fgets(lineIn, 1000, stdin);
  fp = fopen(cfname, "r");

  psIMatrix matGrpMembers;
  matGrpMembers.setFormat(PS_MAT2D);
  if (fp != NULL)
  {
    fgets(lineIn, 1000, fp);
    sscanf(lineIn, "%s", pString);
    if (!strcmp(pString, "PSUADE_BEGIN"))
    {
      fscanf(fp, "%d", &nGroups);
      if (nGroups <= 0)
      {
        printOutTS(PL_ERROR, "RSMSobolG ERROR: nGroups <= 0.\n");
        fclose(fp);
        exit(1);
      }
      matGrpMembers.setDim(nGroups, nInputs);
      for (ii = 0; ii < nGroups; ii++)
      {
        fscanf(fp, "%d", &groupID);
        if (groupID != ii+1)
        {
          printOutTS(PL_ERROR,
               "RSMSobolG ERROR: invalid groupID %d",groupID);
          printOutTS(PL_ERROR," should be %d\n", ii+1);
          fclose(fp);
          exit(1);
        }
        fscanf(fp, "%d", &length);
        if (length <= 0 || length >= nInputs)
        {
          printOutTS(PL_ERROR, 
              "RSMSobolG ERROR: invalid group length.\n");
          fclose(fp);
          exit(1);
        }
        sCnt = 1;
        for (jj = 0; jj < length; jj++)
        {
          fscanf(fp, "%d", &index);
          if (index <= 0 || index > nInputs)
          {
            printOutTS(PL_ERROR, 
                 "RSMSobolG ERROR: invalid group member.\n");
            fclose(fp);
            exit(1);
          }
          matGrpMembers.setEntry(ii,index-1, sCnt);
          sCnt++;
        }
      }
      fgets(lineIn, 1000, fp);
      fgets(lineIn, 1000, fp);
      sscanf(lineIn, "%s", pString);
      if (strcmp(pString, "PSUADE_END"))
      {
        printOutTS(PL_ERROR, 
           "RSMSobolG ERROR: PSUADE_END not found.\n");
        fclose(fp);
        exit(1);
      }
    }
    else
    {
      printOutTS(PL_ERROR, "RSMSobolG ERROR: PSUADE_BEGIN not found.\n");
      fclose(fp);
      exit(1);
    }
    fclose(fp);
  }

  //**/ ---------------------------------------------------------------
  //**/ analyze whether group information consistent with joint PDFs
  //**/ ---------------------------------------------------------------
  ioPtr->getParameter("input_cor_matrix", pCorMat);
  corMatp = (psMatrix *) pCorMat.psObject_;
  int ind1, ind2;
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
          printOutTS(PL_ERROR,"RSMSobolG INFO: currently cannot handle\n");
          printOutTS(PL_ERROR,"          correlated inputs (joint PDF)\n");
          printOutTS(PL_ERROR,"          across different groups.\n");
          matGrpMembers.clean();
          return PSUADE_UNDEFINED;
        }
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ set up constraint filters
  //**/ ---------------------------------------------------------------
  RSConstraints *constrPtr = new RSConstraints();
  constrPtr->genConstraints(ioPtr);

  //**/ ---------------------------------------------------------------
  //**/ build response surface
  //**/ ---------------------------------------------------------------
  FuncApprox *faPtr = genFAInteractive(ioPtr, 0);
  status = faPtr->initialize(XIn, vecYIn2.getDVector());

  //**/ ---------------------------------------------------------------
  //**/  get internal parameters 
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn()) printAsterisks(PL_INFO, 0);
  if (psConfig_.AnaExpertModeIsOn())
  {
    printOutTS(PL_INFO,
         "* RSMSobolG creates a sample of size M1 for each subgroup\n");
    printOutTS(PL_INFO,
         "* of inputs and M2 for the other inputs when computing\n");
    printOutTS(PL_INFO,
         "* group sensitivity indices. The total sample size is thus:\n");
    printOutTS(PL_INFO,"*     N = M1 * M2 * nGroups.\n");
    nSubSamplesG = 20000;
    nSubSamplesN = 100;
    printOutTS(PL_INFO,"  default M1 = %d.\n", nSubSamplesG);
    printOutTS(PL_INFO,"  default M2 = %d.\n", nSubSamplesN);
    printOutTS(PL_INFO,"* Please select your desired M1 and M2.\n");
    printOutTS(PL_INFO,"* Recommendation: M1 > M2.\n");
    printOutTS(PL_INFO,"* Note: large M1 and M2 can take a long time.\n");
    printEquals(PL_INFO, 0);
    snprintf(pString,100,
         "Enter M1 (suggestion: 10000 - 100000, default = 20000) : ");
    nSubSamplesG = getInt(10000, 200000, pString);
    snprintf(pString,100,"Enter M2 (suggestion: 100 - 500, default = 100) : ");
    nSubSamplesN = getInt(100, 1000, pString);
    printAsterisks(PL_INFO, 0);
  }
  else
  {
    nSubSamplesG = 20000;
    nSubSamplesN = 100;
    cString = psConfig_.getParameter("RSMSobolG_nsubsamples_ingroup");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nSubSamplesG);
      if (nSubSamplesG < 10000)
      {
        printOutTS(PL_INFO,
             "RSMSobolG INFO: nSubSamplesG should be >= 10000.\n");
        nSubSamplesG = 10000;
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
    printOutTS(PL_INFO,"RSMSobolG: default M1 = %d.\n",nSubSamplesG);
    printOutTS(PL_INFO,"RSMSobolG: default M2 = %d.\n",nSubSamplesN);
    printOutTS(PL_INFO,
        "To change these settings, re-run with ana_expert mode on.\n");
  }
  if (psConfig_.InteractiveIsOn()) printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/  use response surface to compute total variance
  //**/ ---------------------------------------------------------------
  //**/ ------------------------
  //**/ use a large sample size
  //**/ ------------------------
  nSamp = 25000;
  if (psConfig_.InteractiveIsOn()) 
  {
    printOutTS(PL_INFO,
         "RSMSobolG INFO: creating a sample for basic statistics.\n");
    printOutTS(PL_INFO,"                sample size = %d\n", nSamp);
  }
  //**/ ------------------------
  //**/ allocate space
  //**/ ------------------------
  psVector  vecXT, vecYT;
  psIVector vecST;
  vecXT.setLength(nSamp*8*nInputs);
  vecYT.setLength(nSamp*8);
  vecST.setLength(nSamp*8);

  for (ir = 0; ir < 3; ir++)
  {
    if (psConfig_.InteractiveIsOn()) 
      printOutTS(PL_INFO, 
         "RSMSobolG: compute statistics, sample size = %d\n",nSamp);
       
    //**/ ------------------------
    //**/ create a sample
    //**/ ------------------------
    if (noPDF == 0)
    {
      printOutTS(PL_INFO,"RSMSobolG INFO: non-uniform PDFs detected. \n");
      pdfman = new PDFManager();
      pdfman->initialize(nInputs,pdfFlags,inputMeans,
                         inputStdevs,*corMatp,NULL,NULL);
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
    if (ir < 3 && printLevel > 1)
      printOutTS(PL_INFO,
           "RSMSobolG: running the sample with response surface...\n");
    faPtr->evaluatePoint(nSamp, vecXT.getDVector(), vecYT.getDVector());
    if (ir < 3 && printLevel > 1)
      printOutTS(PL_INFO,
           "RSMSobolG: done running the sample with response surface.\n");

    //**/ ------------------------
    //**/ apply filters
    //**/ ------------------------
    double *XT = vecXT.getDVector();
    double *YT = vecYT.getDVector();
    double *oneSamplePt;
    for (ii = 0; ii < nSamp; ii++)
    {
      oneSamplePt = &(XT[ii*nInputs]);
      ddata = constrPtr->evaluate(oneSamplePt,YT[ii],status);
      if (status == 0) vecYT[ii] = PSUADE_UNDEFINED;
    }

    //**/ ------------------------
    //**/ compute statistics
    //**/ ------------------------
    dmean = 0.0;
    sCnt = 0;
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
           "RSMSobolG ERROR: too few samples that satisify the\n");
      printOutTS(PL_ERROR,"constraints (%d out of %d)\n",sCnt,nSamp);
      delete faPtr;
      return PSUADE_UNDEFINED;
    }
    variance = 0.0;
    for (ii = 0; ii < nSamp; ii++)
    {
      if (vecYT[ii] != PSUADE_UNDEFINED)
        variance += (vecYT[ii] - dmean) * (vecYT[ii] - dmean) ;
    }
    variance /= (double) sCnt;
    if (printLevel > 3 || ir == 2)
    {
      printOutTS(PL_INFO,
           "RSMSobolG: sample mean    (based on N = %d) = %10.3e\n",
           sCnt, dmean);
      printOutTS(PL_INFO,
           "RSMSobolG: sample std dev (based on N = %d) = %10.3e\n",
           sCnt, sqrt(variance));
    }
    nSamp *= 2;
  }
  if (variance == 0.0) variance = 1.0;

  //**/ ---------------------------------------------------------------
  //**/  use response surface to perform Sobol group test
  //**/ ---------------------------------------------------------------
  //**/ set up the sampling method for out of group
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

  //**/ loop through each pair of inputs
  psVector vecVCEs;
  vecVCEs.setLength(nGroups);
  printAsterisks(PL_INFO, 0);
  printf("**                Group Sensitivity Analysis Summary\n");
  printEquals(PL_INFO, 0);
  for (ii = 0; ii < nGroups; ii++)
  {
    if (printLevel > 1)
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
    if (noPDF == 0) 
    {
      nInputsN = 0;
      for (jj = 0; jj < nInputs; jj++)
      {
        if (matGrpMembers.getEntry(ii,jj) == 0)
        {
          vecPdfFlagsN[nInputsN] = pdfFlags[jj];
          vecInpMeansN[nInputsN] = inputMeans[jj];
          vecInpStdsN[nInputsN]  = inputStdevs[jj];
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
    if (noPDF == 0) 
    {
      nInputsG = 0;
      for (jj = 0; jj < nInputs; jj++)
      {
        if (matGrpMembers.getEntry(ii,jj) != 0)
        {
          vecPdfFlagsG[nInputsG] = pdfFlags[jj];
          vecInpMeansG[nInputsG] = inputMeans[jj];
          vecInpStdsG[nInputsG]  = inputStdevs[jj];
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
      if (noPDF == 0) 
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

        printOutTS(PL_DUMP, "RSMSobolG: Group %d\n", ii+1);
        printOutTS(PL_DUMP, "  refinement = %d, size = %d (%d),",ir,
                   sCnt, nSubSamplesN);
        printOutTS(PL_DUMP, 
           " mean = %12.4e, var = %12.4e\n",vecMeans[ii2],vecVars[ii2]);
      }

      //**/ compute the variance of the means for each group
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

      printOutTS(PL_DETAIL,"   Unnormalized ECV (refinement =%3d) ",ir+1);
      printOutTS(PL_DETAIL,"for input group %3d = %10.3e\n", ii+1, ecv);
      printOutTS(PL_DETAIL,"   Unnormalized VCE (refinement =%3d) ",ir+1);
      printOutTS(PL_DETAIL,"for input group %3d = %10.3e\n", ii+1, vce);
      currNSamples *= 2;
    }
    vecVCEs[ii] = vce;
  }
  printAsterisks(PL_INFO, 0);
  printf("*          Group RSSobol Summary\n");
  printEquals(PL_INFO, 0);
  for (ii = 0; ii < nGroups; ii++)
    printOutTS(PL_INFO,
        "** VCE for input group %3d = %10.3e (normalized = %10.3e)\n",
        ii+1, vecVCEs[ii], vecVCEs[ii]/variance);
  printAsterisks(PL_INFO, 0);
    
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
RSMSobolGAnalyzer& RSMSobolGAnalyzer::operator=(const RSMSobolGAnalyzer &)
{
   printOutTS(PL_ERROR, "RSMSobolG operator= ERROR: operation not allowed.\n");
   exit(1);
   return (*this);
}

