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
// Functions for AnalysisManager 
// AUTHOR : CHARLES TONG
// DATE   : 2005
// ************************************************************************
#include <stdio.h>
#include <assert.h>
#include "pData.h"
#include "sysdef.h"
#include "Globals.h"
#include "AnalysisManager.h"
#include "AnovaAnalyzer.h"
#include "GradStatAnalyzer.h"
#include "MOATAnalyzer.h"
#include "MainEffectAnalyzer.h"
#include "TwoParamAnalyzer.h"
#include "SobolAnalyzer.h"
#include "RSFuncApproxAnalyzer.h"
#include "MomentAnalyzer.h"
#include "CorrelationAnalyzer.h"
#include "IntegrationAnalyzer.h"
#include "FASTAnalyzer.h"
#include "FFAnalyzer.h"
#include "LSAnalyzer.h"
#include "PCAnalyzer.h"
#include "OneSigmaAnalyzer.h"
#include "FORMAnalyzer.h"
#include "RSMSobol1Analyzer.h"
#include "RSMSobol2Analyzer.h"
#include "RSMSobolTSIAnalyzer.h"
#include "BootstrapAnalyzer.h"
#include "RSMSobolGAnalyzer.h"
#include "OneSampleAnalyzer.h"
#include "TwoSampleAnalyzer.h"
#include "MCMCAnalyzer.h"
#include "DeltaAnalyzer.h"
#include "EtaAnalyzer.h"
#include "GowerAnalyzer.h"
#include "PrintingTS.h"
#include "psStrings.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------ 
AnalysisManager::AnalysisManager()
{
  numAnalyzers_ = 27;
  analyzers_ = new Analyzer*[numAnalyzers_];;
  for (int ii = 0; ii < numAnalyzers_; ii++) analyzers_[ii] = NULL;
  analysisSampleErrors_ = NULL;

#ifdef HAVE_PYTHON
  AnalysisDataList = PyList_New(0);
#endif
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------ 
AnalysisManager::~AnalysisManager()
{ 
  if (analyzers_ != NULL) 
  {
    for (int ii = 0; ii < numAnalyzers_; ii++)
      if (analyzers_[ii] != NULL) delete analyzers_[ii];
    delete [] analyzers_;
  }
  if (analysisSampleErrors_ != NULL) delete [] analysisSampleErrors_;

#ifdef HAVE_PYTHON
  Py_DECREF(AnalysisDataList);
#endif
}

// ************************************************************************
// prepare for a new analysis run 
// ------------------------------------------------------------------------ 
int AnalysisManager::clearAnalyzers()
{ 
  for (int ii = 0; ii < numAnalyzers_; ii++)
  {
    if (analyzers_[ii] != NULL)
    {
      delete analyzers_[ii];
      analyzers_[ii] = NULL;
    }
  }
  return 0;
}

// ************************************************************************
// set up various analysis
// ------------------------------------------------------------------------
int AnalysisManager::setup(PsuadeData *psuadeIO)
{
  int   anaMethod, faType;
  pData pPtr;

  if (psuadeIO == NULL)
  {
    printOutTS(PL_ERROR,"AnalysisManager setup ERROR: missing psuadeIO\n");
    exit(1);
  }
  psuadeIO->getParameter("ana_method", pPtr);
  anaMethod = pPtr.intData_;
  psuadeIO->getParameter("ana_rstype", pPtr);
  faType = pPtr.intData_;
  setup(anaMethod, faType);
  return 0;
}

// ************************************************************************
// set up analysis (different parameter set)
// ------------------------------------------------------------------------
int AnalysisManager::setup(int anaMethod, int faType)
{
  int  targc;
  char *targv[3], request[100];

  if ((anaMethod & PSUADE_ANA_MOMENT) != 0 && (analyzers_[0] == NULL))
    analyzers_[0] = new MomentAnalyzer();
  if ((anaMethod & PSUADE_ANA_GLSA) != 0 && (analyzers_[1] == NULL))
    analyzers_[1] = new GradStatAnalyzer();
  if ((anaMethod & PSUADE_ANA_CORRELATION) != 0 && (analyzers_[2] == NULL))
    analyzers_[2] = new CorrelationAnalyzer();
  if ((anaMethod & PSUADE_ANA_ME) != 0 && (analyzers_[3] == NULL))
    analyzers_[3] = new MainEffectAnalyzer();
  if ((anaMethod & PSUADE_ANA_IE) != 0 && (analyzers_[4] == NULL))
    analyzers_[4] = new TwoParamAnalyzer();
  if ((anaMethod & PSUADE_ANA_MOAT) != 0 && (analyzers_[5] == NULL))
    analyzers_[5] = new MOATAnalyzer();
  if ((anaMethod & PSUADE_ANA_SOBOL) != 0 && (analyzers_[6] == NULL))
    analyzers_[6] = new SobolAnalyzer();
  if ((anaMethod & PSUADE_ANA_ANOVA) != 0 && (analyzers_[7] == NULL))
     analyzers_[7] = new AnovaAnalyzer();
  if ((anaMethod & PSUADE_ANA_RSFA) != 0 && (analyzers_[8] == NULL))
  {
    analyzers_[8] = new RSFuncApproxAnalyzer();
    targc = 2;
    strcpy(request, "rstype");
    targv[0] = (char *) request;
    targv[1] = (char *) &faType;
    analyzers_[8]->setParams(targc, targv);
  }
  if ((anaMethod & PSUADE_ANA_INTEGRATION) != 0 && (analyzers_[9] == NULL))
    analyzers_[9] = new IntegrationAnalyzer();
  if ((anaMethod & PSUADE_ANA_FAST) != 0 && (analyzers_[10] == NULL))
    analyzers_[10] = new FASTAnalyzer();
  if ((anaMethod & PSUADE_ANA_FF) != 0 && (analyzers_[11] == NULL))
    analyzers_[11] = new FFAnalyzer();
  if ((anaMethod & PSUADE_ANA_PCA) != 0 && (analyzers_[12] == NULL))
    analyzers_[12] = new PCAnalyzer();
  if ((anaMethod & PSUADE_ANA_ONESIGMA) != 0 && (analyzers_[13] == NULL))
  {
    analyzers_[13] = new OneSigmaAnalyzer();
    targc = 2;
    strcpy(request, "rstype");
    targv[0] = (char *) request;
    targv[1] = (char *) &faType;
    analyzers_[13]->setParams(targc, targv);
  }
  if ((anaMethod & PSUADE_ANA_FORM) != 0 && (analyzers_[14] == NULL))
  {
    analyzers_[14] = new FORMAnalyzer();
    targc = 2;
    strcpy(request, "rstype");
    targv[0] = (char *) request;
    targv[1] = (char *) &faType;
    analyzers_[14]->setParams(targc, targv);
  }
  if ((anaMethod & PSUADE_ANA_RSSOBOL1) != 0 && (analyzers_[15] == NULL))
    analyzers_[15] = new RSMSobol1Analyzer();
  if ((anaMethod & PSUADE_ANA_RSSOBOL2) != 0 && (analyzers_[16] == NULL))
    analyzers_[16] = new RSMSobol2Analyzer();
  if ((anaMethod & PSUADE_ANA_RSSOBOLTSI) != 0 && (analyzers_[17] == NULL))
    analyzers_[17] = new RSMSobolTSIAnalyzer();
  if ((anaMethod & PSUADE_ANA_BSTRAP) != 0 && (analyzers_[18] == NULL))
    analyzers_[18] = new BootstrapAnalyzer();
  if ((anaMethod & PSUADE_ANA_RSSOBOLG) != 0 && (analyzers_[19] == NULL))
    analyzers_[19] = new RSMSobolGAnalyzer();
  if ((anaMethod & PSUADE_ANA_1SAMPLE) != 0 && (analyzers_[20] == NULL))
    analyzers_[20] = new OneSampleAnalyzer();
  if ((anaMethod & PSUADE_ANA_2SAMPLE) != 0 && (analyzers_[21] == NULL))
    analyzers_[21] = new TwoSampleAnalyzer();
  if ((anaMethod & PSUADE_ANA_MCMC) != 0 && (analyzers_[22] == NULL))
    analyzers_[22] = new MCMCAnalyzer();
  if ((anaMethod & PSUADE_ANA_DTEST) != 0 && (analyzers_[23] == NULL))
    analyzers_[23] = new DeltaAnalyzer();
  if ((anaMethod & PSUADE_ANA_GOWER) != 0 && (analyzers_[24] == NULL))
    analyzers_[24] = new GowerAnalyzer();
  if ((anaMethod & PSUADE_ANA_ETEST) != 0 && (analyzers_[25] == NULL))
    analyzers_[25] = new EtaAnalyzer();
  if ((anaMethod & PSUADE_ANA_LSA) != 0 && (analyzers_[26] == NULL))
    analyzers_[26] = new LSAnalyzer();
  return 0;
}

// ************************************************************************
// set log transform flags (use logarithms of the inputs/outputs)
// ------------------------------------------------------------------------
int AnalysisManager::loadLogXsformFlags(int leng, int *flags)
{
  if (leng <= 0)
  {
    printOutTS(PL_ERROR,
         "AnalysisManager::loadLogXsformFlags ERROR: n <= 0.\n");
    exit(1);
  }
  VecLogXsformFlags_.setLength(leng);
  for (int ii = 0; ii < leng; ii++) VecLogXsformFlags_[ii] = flags[ii]; 
  return 0;
}

// ************************************************************************
// analyze
// ------------------------------------------------------------------------
int AnalysisManager::analyze(PsuadeData *psuadeIO, int nLevels, 
                             int *levelSeps, int analysisOutputID)
{
  int    anaMethod, nInputs, nOutputs, nSamples, samplingMethod, nReps;
  int    refineFlag=1, jj, analysisTransform, wgtID, nActive;
  int    ii, outputLevel, errCnt, pdfFlag, onlyMCMC;
  double dmin, analysisThreshold, analysisData;
  char   inStr[1000], lineIn[10000], **names;
  pData  pPtr, pLowerB, pUpperB, pInpData, pOutData, pPDFs, pMeans, pStds;
  pData  pStates;
  aData  aPtr;
  psStrings Ynames;

  //**/ ----------------------------------------------------------------
  //**/ first check to see if any analyzer has been instantiated.
  //**/ ----------------------------------------------------------------
  nActive = 0;
  for (ii = 0; ii < numAnalyzers_; ii++) 
    if (analyzers_[ii] != NULL) nActive++;
  if (nActive == 0) return 0;
  //**/ MCMC accepts the case when there is no sample outputs
  onlyMCMC = 0;
  if (nActive == 1 && analyzers_[22] != NULL) onlyMCMC = 1;

  //**/ ----------------------------------------------------------------
  //**/ retrieve parameters and data from PsuadeData object
  //**/ ----------------------------------------------------------------
  if (psuadeIO == NULL && onlyMCMC == 0)
  {
    printOutTS(PL_ERROR, "AnalysisManager ERROR: no DataIO.\n");
    return -1;
  }
  psuadeIO->getParameter("ana_method", pPtr);
  anaMethod = pPtr.intData_;
  psuadeIO->getParameter("input_ninputs", pPtr);
  nInputs = pPtr.intData_;
  psuadeIO->getParameter("output_noutputs", pPtr);
  nOutputs = pPtr.intData_;
  psuadeIO->getParameter("method_nsamples", pPtr);
  nSamples = pPtr.intData_;
  psuadeIO->getParameter("method_sampling", pPtr);
  samplingMethod = pPtr.intData_;
  psuadeIO->getParameter("method_nreplications", pPtr);
  nReps = pPtr.intData_;
  psuadeIO->getParameter("input_lbounds", pLowerB);
  double *iLowerB = pLowerB.dbleArray_;
  psuadeIO->getParameter("input_ubounds", pUpperB);
  double *iUpperB = pUpperB.dbleArray_;
  double *XIns=NULL, *YIns=NULL;
  int    *states=NULL;
  if (nSamples > 0)
  {
    psuadeIO->getParameter("input_sample", pInpData);
    XIns = pInpData.dbleArray_;
    psuadeIO->getParameter("output_sample", pOutData);
    YIns = pOutData.dbleArray_;
    psuadeIO->getParameter("output_states", pStates);
    states = pStates.intArray_;
  }
  psuadeIO->getParameter("ana_threshold", pPtr);
  analysisThreshold = pPtr.dbleData_;
  psuadeIO->getParameter("ana_transform", pPtr);
  analysisTransform = pPtr.intData_;
  if (XIns == NULL)
  {
    printOutTS(PL_WARN,
         "AnalysisManager WARNING: no sample input data.\n");
    nSamples = 0;
  }
  if (YIns == NULL)
  {
    printOutTS(PL_WARN,
         "AnalysisManager WARNING: no sample output data.\n");
    nSamples = 0;
  }
  if (analysisOutputID < 0)
  {
    psuadeIO->getParameter("ana_outputid", pPtr);
    analysisOutputID = pPtr.intData_;
    if (analysisOutputID < 0)
    {
      printOutTS(PL_ERROR,"AnalysisManager ERROR: output ID <= 0.\n");
      return -1;
    }
  }
  psuadeIO->getParameter("ana_regressionwgtid", pPtr);
  wgtID = pPtr.intData_;
  psuadeIO->getParameter("ana_diagnostics", pPtr);
  outputLevel = pPtr.intData_;

  //**/ ----------------------------------------------------------------
  //**/ input/output data transformation
  //**/ ----------------------------------------------------------------
  psIVector VecXsforms;
  if (VecLogXsformFlags_.length() > 0)
  {
    VecXsforms.setLength(nInputs);
    //**/ transform here, these should not be used in actual analyzers
    aPtr.inputXsforms_ = VecXsforms.getIVector();
    for (ii = 0; ii < nInputs; ii++)
      VecXsforms[ii] = VecLogXsformFlags_[0] & 1;
    analysisTransform = VecLogXsformFlags_[0] | VecLogXsformFlags_[1];
  }
  else   
  {
    VecXsforms.setLength(nInputs);
    aPtr.inputXsforms_ = VecXsforms.getIVector();
    for (ii = 0; ii < nInputs; ii++) 
      VecXsforms[ii] = analysisTransform & 1;
  }
  psVector VecSamInps, VecSamOuts;
  if (nSamples > 0)
  {
    if (analysisTransform == 0)
    {
      if (psConfig_.InteractiveIsOn())
        printOutTS(PL_INFO,
           "No transformation (e.g. log) on sample inputs or outputs.\n");
      VecSamInps.load(nInputs*nSamples, XIns);
      VecSamOuts.load(nOutputs*nSamples, YIns);
    }
    else
    {
      if ((analysisTransform & 1) && XIns != NULL)
      {
        for (ii = 0; ii < nInputs*nSamples; ii++)
        {
          if (XIns[ii] <= 0.0)
          {
            analysisTransform &= 2;
            for (jj = 0; jj < nInputs; jj++) VecXsforms[jj] = 0;
            printOutTS(PL_INFO,
              "Some inputs are < 0 => TURN OFF INPUT TRANSFORMATION.\n");
            printf("Continue with no input transformation (y or n)? ");
            scanf("%s", inStr);
            fgets(lineIn, 1000, stdin);
            if (inStr[0] != 'y') 
            {
              aPtr.sampleInputs_ = NULL;
              aPtr.sampleOutputs_ = NULL;
              aPtr.sampleStates_ = NULL;
              aPtr.iLowerB_ = NULL;
              aPtr.iUpperB_ = NULL;
              aPtr.inputXsforms_ = NULL;
              aPtr.inputPDFs_ = NULL;
              aPtr.inputMeans_ = NULL;
              aPtr.inputStdevs_ = NULL;
              return -1;
            }
            break;
          }
        }
      }
      if ((analysisTransform & 2) && YIns != NULL)
      {
        for (ii = 0; ii < nSamples; ii++)
        {
          if (YIns[ii*nOutputs+analysisOutputID] <= 0.0)
          {
            analysisTransform &= 1;
            printOutTS(PL_ERROR, 
              "SOME OUTPUT ARE < 0 => TURN OFF OUTPUT TRANSFORMATION.\n");
            printf("E.G. Sample %d output %d is negative.\n",
                   ii+1, analysisOutputID+1);
            printf("Continue with output no transformation (y or n)? ");
            scanf("%s", inStr);
            fgets(lineIn, 1000, stdin);
            if (inStr[0] != 'y') 
            {
              aPtr.sampleInputs_ = NULL;
              aPtr.sampleOutputs_ = NULL;
              aPtr.sampleStates_ = NULL;
              aPtr.iLowerB_ = NULL;
              aPtr.iUpperB_ = NULL;
              aPtr.inputXsforms_ = NULL;
              aPtr.inputPDFs_ = NULL;
              aPtr.inputMeans_ = NULL;
              aPtr.inputStdevs_ = NULL;
              return -1;
            }
            break;
          }
        }
      }
      if (analysisTransform == 0)
      {
        VecSamInps.load(nInputs*nSamples, XIns);
        VecSamOuts.load(nOutputs*nSamples, YIns);
      }
      else
      {
        VecSamInps.setLength(nInputs*nSamples);
        VecSamOuts.setLength(nOutputs*nSamples);
        if (analysisTransform & 1) 
          printOutTS(PL_INFO, "Log transformation on inputs.\n");
        for (ii = 0; ii < nInputs; ii++)
        {
          if (VecXsforms[ii] == 1)
          {
            for (jj = 0; jj < nSamples; jj++)
              VecSamInps[jj*nInputs+ii] = log(XIns[jj*nInputs+ii]);
            if (iLowerB[ii] > 0) iLowerB[ii] = log(iLowerB[ii]);
            else
            {
              dmin = PSUADE_UNDEFINED;
              for (jj = 0; jj < nSamples; jj++)
                if (XIns[jj*nInputs+ii] < dmin) 
                  dmin = XIns[jj*nInputs+ii];
              iLowerB[ii] = log(dmin);
              iUpperB[ii] = log(iUpperB[ii]);
            }
          }
          else
          {
            for (jj = 0; jj < nSamples; jj++)
              VecSamInps[jj*nInputs+ii] = XIns[jj*nInputs+ii];
          }
        }
        if (analysisTransform & 2) 
          printOutTS(PL_INFO, "Log transformation on outputs.\n");
        if ((analysisTransform & 2) && YIns != NULL)
        {
          for (ii = 0; ii < nSamples; ii++)
            VecSamOuts[ii] = log(YIns[nOutputs*ii+analysisOutputID]);
        }
        else if (YIns != NULL)
        {
          for (ii = 0; ii < nSamples*nOutputs; ii++)
            VecSamOuts[ii] = YIns[ii];
        }
      }
    }
  }
  errCnt = 0;
  for (ii = 0; ii < VecSamOuts.length(); ii++)
    if (VecSamOuts[ii] == PSUADE_UNDEFINED) errCnt++;
  if (errCnt > 0)
  {
    printOutTS(PL_WARN, 
         "AnalysisManager ERROR: undefined data found (%d).\n",errCnt);
    aPtr.sampleInputs_ = NULL;
    aPtr.sampleOutputs_ = NULL;
    aPtr.sampleStates_ = NULL;
    aPtr.iLowerB_ = NULL;
    aPtr.iUpperB_ = NULL;
    aPtr.inputXsforms_ = NULL;
    aPtr.inputPDFs_ = NULL;
    aPtr.inputMeans_ = NULL;
    aPtr.inputStdevs_ = NULL;
    return -1;
  }

  //**/ ----------------------------------------------------------------
  //**/ set up analysis data object
  //**/ ----------------------------------------------------------------
  aPtr.ioPtr_ = psuadeIO;
  aPtr.nSamples_ = nSamples;
  aPtr.nInputs_ = nInputs;
  aPtr.nOutputs_ = nOutputs;
  aPtr.outputID_ = analysisOutputID;
  aPtr.iLowerB_ = iLowerB;
  aPtr.iUpperB_ = iUpperB;
  aPtr.sampleInputs_ = VecSamInps.getDVector();
  aPtr.sampleOutputs_ = VecSamOuts.getDVector();
  aPtr.sampleStates_ = states;
  aPtr.printLevel_ = outputLevel;

  psuadeIO->getParameter("input_pdfs", pPDFs);
  pdfFlag = 0;
  for (ii = 0; ii < nInputs; ii++) pdfFlag += pPDFs.intArray_[ii];

  psIVector VecAuxPDFs;
  psVector  VecAuxMeans, VecAuxStds;
  VecAuxPDFs.setLength(nInputs);
  VecAuxMeans.setLength(nInputs);
  VecAuxStds.setLength(nInputs);
  aPtr.inputPDFs_   = VecAuxPDFs.getIVector();
  aPtr.inputMeans_  = VecAuxMeans.getDVector();
  aPtr.inputStdevs_ = VecAuxStds.getDVector();
  if (pdfFlag > 0)
  {
    psuadeIO->getParameter("input_pdfs", pPDFs);
    psuadeIO->getParameter("input_means", pMeans);
    psuadeIO->getParameter("input_stdevs", pStds);
    for (ii = 0; ii < nInputs; ii++)
    {
      VecAuxPDFs[ii]  = pPDFs.intArray_[ii];
      VecAuxMeans[ii] = pMeans.dbleArray_[ii];
      VecAuxStds[ii]  = pStds.dbleArray_[ii];
    }
  }
  else
  {
    for (ii = 0; ii < nInputs; ii++)
    {
      VecAuxMeans[ii] = 0.0;
      VecAuxStds[ii] = 0.0;
      VecAuxPDFs[ii] = 0;
    }
  } 

  //**/ ----------------------------------------------------------------
  //**/ perform selected analysis
  //**/ ----------------------------------------------------------------
  //**/ compute standard error of mean 
  if (analyzers_[0] != NULL)
  {
    aPtr.currRefineLevel_ = 0;
    aPtr.refineSeparators_ = NULL;
    aPtr.nSubSamples_ = nSamples/nReps;
    analysisData = analyzers_[0]->analyze(aPtr);
    if (analysisData < analysisThreshold || analysisData <= 0 || 
        analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED &&
        refineFlag == 0)
       printOutTS(PL_INFO, 
            "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
            analysisData, analysisThreshold);
  }

  //**/ global SA based on derivative - NOT VERIFIED COMPLETELY
  if (analyzers_[1] != NULL)
  {
    aPtr.currRefineLevel_ = nLevels;
    aPtr.refineSeparators_ = levelSeps;
    aPtr.nSubSamples_ = nSamples/nReps;
    aPtr.analysisThreshold_ = analysisThreshold;
    analysisData = analyzers_[1]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
        analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
       printOutTS(PL_INFO, 
            "AnalysisManager : analysis error = %8.2e <? %8.2e\n",
            analysisData, analysisThreshold);
  }

  //**/ Correlation analysis
  if (analyzers_[2] != NULL)
  {
    aPtr.currRefineLevel_ = nLevels;
    aPtr.refineSeparators_ = levelSeps;
    aPtr.nSubSamples_ = nSamples/nReps;
    analysisData = analyzers_[2]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
        analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
       printOutTS(PL_INFO, 
            "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
            analysisData, analysisThreshold);
  }

  //**/ McKay analysis
  if (analyzers_[3] != NULL)
  {
    //**/ perform McKay analysis
    aPtr.currRefineLevel_ = 0;
    aPtr.refineSeparators_ = NULL;
    aPtr.nSubSamples_ = nSamples / nReps;
    analysisData = analyzers_[3]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
        analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
       printOutTS(PL_INFO, 
            "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
            analysisData, analysisThreshold);
  }

  //**/ interaction study
  if (analyzers_[4] != NULL)
  {
    aPtr.currRefineLevel_ = 0;
    aPtr.refineSeparators_ = NULL;
    aPtr.nSubSamples_ = nSamples / nReps;
    analysisData = analyzers_[4]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
        analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ MOAT analysis
  if (analyzers_[5] != NULL)
  {
    aPtr.currRefineLevel_ = nLevels;
    aPtr.refineSeparators_ = levelSeps;
    aPtr.nSubSamples_ = 0;
    analysisData = analyzers_[5]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
        analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
       printOutTS(PL_INFO, 
            "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
            analysisData, analysisThreshold);
  }

  //**/ Sobol analysis
  if (analyzers_[6] != NULL)
  {
    aPtr.currRefineLevel_ = nLevels;
    aPtr.refineSeparators_ = levelSeps;
    aPtr.nSubSamples_ = 0;
    analysisData = analyzers_[6]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
        analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ analysis of variance - NOT VERIFIED COMPLETELY
  if (analyzers_[7] != NULL)
  {
    aPtr.currRefineLevel_ = nLevels;
    aPtr.refineSeparators_ = levelSeps;
    aPtr.nSubSamples_ = 0;
    analysisData = analyzers_[7]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
        analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ response surface fitting
  if (analyzers_[8] != NULL)
  {
    aPtr.currRefineLevel_ = nLevels;
    aPtr.refineSeparators_ = levelSeps;
    aPtr.nSubSamples_ = 0;
    aPtr.regWgtID_ = wgtID;
    aPtr.cvFlag_ = 0;
    //**/ since this is very expensive for large sample size, perform 
    //**/ this only when needed or especially requested.
    if (((refineFlag == 1) && (outputLevel >= 3)) || (outputLevel >= 4))
      aPtr.cvFlag_ = 1;
    //**/ validation against a test set
    if (outputLevel >= 5) aPtr.cvFlag_ = 2;
    analysisData = analyzers_[8]->analyze(aPtr);
    if (analysisSampleErrors_ != NULL) delete [] analysisSampleErrors_;
    analysisSampleErrors_ = aPtr.sampleErrors_;
    aPtr.sampleErrors_ = NULL;
    if (analysisData < analysisThreshold || 
        analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ numerical integration
  if (analyzers_[9] != NULL)
  {
    aPtr.currRefineLevel_ = nLevels;
    aPtr.refineSeparators_ = levelSeps;
    aPtr.nSubSamples_ = 0;
    analysisData = analyzers_[9]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
      analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
         "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
         analysisData, analysisThreshold);
  }

  //**/ FAST analysis
  if (analyzers_[10] != NULL)
  {
    analysisData = analyzers_[10]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
      analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ Fractional factorial main effect and interaction analysis
  if (analyzers_[11] != NULL)
  {
    aPtr.samplingMethod_ = samplingMethod;
    analysisData = analyzers_[11]->analyze(aPtr);
    aPtr.samplingMethod_ = -1;
    if (analysisData < analysisThreshold || 
      analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ Principal Component analysis
  if (analyzers_[12] != NULL)
  {
    analysisData = analyzers_[12]->analyze(aPtr);
    if (analysisData < analysisThreshold) refineFlag = 0;
    if (outputLevel > 2 && analysisData > 0.0)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
    if (aPtr.nOutputs_ < nOutputs)
    {
      Ynames.setNumStrings(nOutputs);
      for (ii = 0; ii < aPtr.nOutputs_; ii++)
      {
        snprintf(lineIn,100,"PC%d", ii+1);
        Ynames.loadOneString(ii, lineIn);
      }
      names = Ynames.getStrings();
    }
    else names = NULL;
    psuadeIO->updateOutputSection(nSamples, aPtr.nOutputs_,
                    aPtr.sampleOutputs_, states, names);
    psuadeIO->writePsuadeFile(NULL,0);
  }

  //**/ One Sigma analysis
  if (analyzers_[13] != NULL)
  {
    analysisData = analyzers_[13]->analyze(aPtr);
    if (analysisData < analysisThreshold) refineFlag = 0;
    if (outputLevel > 2 && analysisData > 0.0)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
    psuadeIO->updateOutputSection(nSamples, aPtr.nOutputs_,
                             aPtr.sampleOutputs_, states, NULL);
    if (analysisSampleErrors_ != NULL) delete [] analysisSampleErrors_;
    analysisSampleErrors_ = aPtr.sampleErrors_;
    aPtr.sampleErrors_ = NULL;
  }

  //**/ First order reliability analysis
  if (analyzers_[14] != NULL)
  {
    aPtr.analysisThreshold_ = analysisThreshold; 
    analysisData = analyzers_[14]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
      analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ Sobol first order variance decomposition
  if (analyzers_[15] != NULL)
  {
    aPtr.analysisThreshold_ = analysisThreshold; 
    analysisData = analyzers_[15]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
      analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ Sobol second order variance decomposition
  if (analyzers_[16] != NULL)
  {
    aPtr.analysisThreshold_ = analysisThreshold; 
    analysisData = analyzers_[16]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
      analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ Sobol total sensitivity indices
  if (analyzers_[17] != NULL)
  {
    aPtr.analysisThreshold_ = analysisThreshold; 
    analysisData = analyzers_[17]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
      analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ bootstrap analysis
  if (analyzers_[18] != NULL)
  {
    aPtr.analysisThreshold_ = analysisThreshold; 
    analysisData = analyzers_[18]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
      analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ Sobol group sensitivity indices
  if (analyzers_[19] != NULL)
  {
    aPtr.analysisThreshold_ = analysisThreshold; 
    analysisData = analyzers_[19]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
      analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ OneSample test
  if (analyzers_[20] != NULL)
  {
    aPtr.analysisThreshold_ = analysisThreshold; 
    analysisData = analyzers_[20]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
      analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ TwoSample test
  if (analyzers_[21] != NULL)
  {
    aPtr.analysisThreshold_ = analysisThreshold; 
    analysisData = analyzers_[21]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
      analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }

  //**/ MCMC 
  if (analyzers_[22] != NULL)
  {
    aPtr.analysisThreshold_ = analysisThreshold; 
    analysisData = analyzers_[22]->analyze(aPtr);
    if (analysisData < analysisThreshold || 
      analysisData == PSUADE_UNDEFINED) refineFlag = 0;
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
           analysisData, analysisThreshold);
  }
  
  //**/ Delta test 
  if (analyzers_[23] != NULL)
  {
    analysisData = analyzers_[23]->analyze(aPtr);
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis metric = %8.2e\n",
           analysisData);
  }

  //**/ Gower test 
  if (analyzers_[24] != NULL)
  {
    analysisData = analyzers_[24]->analyze(aPtr);
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis metric = %8.2e\n",
           analysisData);
  }

  //**/ Eta test 
  if (analyzers_[25] != NULL)
  {
    analysisData = analyzers_[25]->analyze(aPtr);
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis metric = %8.2e\n",
           analysisData);
  }

  //**/ LSA test 
  if (analyzers_[26] != NULL)
  {
    analysisData = analyzers_[26]->analyze(aPtr);
    if (outputLevel > 2 && analysisData != PSUADE_UNDEFINED)
      printOutTS(PL_INFO, 
           "AnalysisManager: analysis metric = %8.2e\n",
           analysisData);
  }
#ifdef HAVE_PYTHON
  //**/ Store each dictionary of output data
  for (ii = 0; ii < numAnalyzers_; ii++)
    if (analyzers_[ii] != NULL)
      PyList_Append(AnalysisDataList, 
                    analyzers_[ii]->AnalysisDataDict);
#endif

  //**/ clean up
  aPtr.sampleInputs_ = NULL;
  aPtr.sampleOutputs_ = NULL;
  aPtr.sampleStates_ = NULL;
  aPtr.iLowerB_ = NULL;
  aPtr.iUpperB_ = NULL;
  aPtr.inputXsforms_ = NULL;
  aPtr.inputPDFs_ = NULL;
  aPtr.inputMeans_ = NULL;
  aPtr.inputStdevs_ = NULL;
  return refineFlag;
}

// ************************************************************************
// analyze (simple, uniform distribution only)
// ------------------------------------------------------------------------
int AnalysisManager::analyze(int anaMethod, int nSamples, psVector &vLower, 
                             psVector &vUpper, psVector &vInputs, 
                             psVector &vOutputs, int outputLevel)
{
  int    ii, nInputs, refineFlag=1;
  double analysisThreshold, analysisData;
  aData  aPtr;
  psVector  VecSamInps, VecSamOuts;
  psIVector VecStates;

  //**/ ----------------------------------------------------------------
  //**/ retrieve parameters 
  //**/ ----------------------------------------------------------------
  nInputs = vLower.length();
  ii = vInputs.length() / nInputs;
  assert(ii == nSamples);
  assert(nInputs == vUpper.length());
  assert(vOutputs.length() == nSamples);
  VecStates.setLength(nSamples);
  for (ii = 0; ii < nSamples; ii++) VecStates[ii] = 1;
  analysisThreshold = 1.0;

  //**/ ----------------------------------------------------------------
  //**/ set up analysis data object
  //**/ ----------------------------------------------------------------
  aPtr.nSamples_ = nSamples;
  aPtr.nInputs_ = nInputs;
  aPtr.nOutputs_ = 1;
  aPtr.outputID_ = 0;
  aPtr.iLowerB_ = vLower.getDVector();
  aPtr.iUpperB_ = vUpper.getDVector();
  aPtr.sampleInputs_ = vInputs.getDVector();
  aPtr.sampleOutputs_ = vOutputs.getDVector();
  aPtr.sampleStates_ = VecStates.getIVector();
  aPtr.printLevel_ = outputLevel;
  aPtr.currRefineLevel_ = 0;
  aPtr.refineSeparators_ = NULL;
  aPtr.nSubSamples_ = nSamples;

  //**/ ----------------------------------------------------------------
  //**/ perform selected analysis
  //**/ ----------------------------------------------------------------
  for (ii = 0; ii < numAnalyzers_; ii++) 
  {
    if (analyzers_[ii] != NULL) 
    {
      analysisData = analyzers_[0]->analyze(aPtr);
      if (analysisData < analysisThreshold || 
        analysisData == PSUADE_UNDEFINED) refineFlag = 0;
      if (outputLevel > 0 && analysisData != PSUADE_UNDEFINED)
        printOutTS(PL_INFO, 
             "AnalysisManager: analysis error = %8.2e <? %8.2e\n",
             analysisData, analysisThreshold);
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ clean up 
  //**/ ----------------------------------------------------------------
  aPtr.sampleInputs_ = NULL;
  aPtr.sampleOutputs_ = NULL;
  aPtr.sampleStates_ = NULL;
  aPtr.iLowerB_ = NULL;
  aPtr.iUpperB_ = NULL;
  return refineFlag;
}

// ************************************************************************
// analyze (one and two sample test only)
// ------------------------------------------------------------------------
int AnalysisManager::analyze(int anaMethod)
{
  aData aPtr;

  if ((anaMethod & PSUADE_ANA_1SAMPLE) != 0 && (analyzers_[20] != NULL))
    analyzers_[20]->analyze(aPtr);
  else if ((anaMethod & PSUADE_ANA_2SAMPLE) != 0 && 
           (analyzers_[21] != NULL))
    analyzers_[21]->analyze(aPtr);
  else
  {
    printOutTS(PL_INFO, 
         "AnalysisManager ERROR: this analyze call is not valid.\n");
    return 1;
  }
  return 0;
}

// ************************************************************************
// get sample errors (errors for individual sample point)
// ------------------------------------------------------------------------ 
double *AnalysisManager::getSampleErrors()
{ 
  return analysisSampleErrors_;
}

// ************************************************************************
// special request (send specialized parameters to individual methods)
// ------------------------------------------------------------------------ 
int AnalysisManager::specialRequest(int anaMethod, int narg, char **argv)
{ 
  if (((anaMethod & PSUADE_ANA_MOMENT) != 0) && (analyzers_[0] != NULL))
    analyzers_[0]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_GLSA) != 0) && (analyzers_[1] != NULL))
    analyzers_[1]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_CORRELATION) != 0) && 
       (analyzers_[2] != NULL))
    analyzers_[2]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_ME) != 0) && (analyzers_[3] != NULL))
    analyzers_[3]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_IE) != 0) && (analyzers_[4] != NULL))
    analyzers_[4]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_MOAT) != 0) && (analyzers_[5] != NULL))
    analyzers_[5]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_SOBOL) != 0) && (analyzers_[6] != NULL))
    analyzers_[6]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_ANOVA) != 0) && (analyzers_[7] != NULL))
    analyzers_[7]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_RSFA) != 0) && (analyzers_[8] != NULL))
    analyzers_[8]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_INTEGRATION) != 0) && 
       (analyzers_[9] != NULL))
    analyzers_[9]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_FAST) != 0) && (analyzers_[10] != NULL))
    analyzers_[10]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_FF) != 0) && (analyzers_[11] != NULL))
    analyzers_[11]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_PCA) != 0) && (analyzers_[12] != NULL))
    analyzers_[12]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_ONESIGMA) != 0) && 
       (analyzers_[13] != NULL))
    analyzers_[13]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_FORM) != 0) && (analyzers_[14] != NULL))
    analyzers_[14]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_RSSOBOL1) != 0) && 
       (analyzers_[15] != NULL))
    analyzers_[15]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_RSSOBOL2) != 0) && 
       (analyzers_[16] != NULL))
    analyzers_[16]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_RSSOBOLTSI) != 0) && 
       (analyzers_[17] != NULL))
    analyzers_[17]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_BSTRAP) != 0) && (analyzers_[18] != NULL))
    analyzers_[18]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_RSSOBOLG) != 0) && 
       (analyzers_[19] != NULL))
    analyzers_[19]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_1SAMPLE) != 0) && (analyzers_[20] != NULL))
    analyzers_[20]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_2SAMPLE) != 0) && (analyzers_[21] != NULL))
    analyzers_[21]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_MCMC) != 0) && (analyzers_[22] != NULL))
    analyzers_[22]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_DTEST) != 0) && (analyzers_[23] != NULL))
    analyzers_[23]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_GOWER) != 0) && (analyzers_[24] != NULL))
    analyzers_[24]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_ETEST) != 0) && (analyzers_[25] != NULL))
    analyzers_[25]->setParams(narg, argv);
  if (((anaMethod & PSUADE_ANA_LSA) != 0) && (analyzers_[26] != NULL))
    analyzers_[26]->setParams(narg, argv);
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
AnalysisManager& AnalysisManager::operator=(const AnalysisManager &)
{
  printOutTS(PL_ERROR, 
       "AnalysisManager operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

// ************************************************************************
// return the MOAT analyzer (this function is for library call to MOAT)
// ------------------------------------------------------------------------
MOATAnalyzer *AnalysisManager::getMOATAnalyzer()
{
  return reinterpret_cast<MOATAnalyzer *>(analyzers_[5]);
}

