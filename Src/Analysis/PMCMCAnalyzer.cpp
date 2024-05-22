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
// Functions for the class PMCMCAnalyzer
// ------------------------------------------------------------------------
// AUTHOR : CHARLES TONG
// DATE   : 2007
// Latest revision: May 2013
// Add model inadequacy function
// ************************************************************************
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include "dtype.h"
#include "CommManager.h"
#include "PMCMCAnalyzer.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "FunctionInterface.h"
#include "pData.h"
#include "PDFBase.h"
#include "PDFNormal.h"
#include "PDFLogNormal.h"
#include "PDFTriangle.h"
#include "PDFBeta.h"
#include "PDFWeibull.h"
#include "PDFGamma.h"
#include "Psuade.h"
#include "Sampling.h"
#include "RSConstraints.h"
#include "PrintingTS.h"
#include "TwoSampleAnalyzer.h"
#include "psVector.h"
#include "psStrings.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
PMCMCAnalyzer::PMCMCAnalyzer(CommManager *comm) : nInputs_(0), nOutputs_(0)
{
  strcpy(name_,"PMCMC");
  mode_ = 0; // RS mode (only one available for now)
  commMgr_ = comm;
  if (comm == NULL)
  {
    printOutTS(PL_ERROR,"MCMC ERROR: no communicator.\n");
    exit(1);
  }
  mypid_  = psCommMgr_->getPID();
  nprocs_ = psCommMgr_->getNumProcs();
  means_  = NULL;
  sigmas_ = NULL;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
PMCMCAnalyzer::~PMCMCAnalyzer()
{
}

// ************************************************************************
// perform MCMC analysis 
// ------------------------------------------------------------------------
double PMCMCAnalyzer::analyze(aData &adata)
{
  int    ii, ii2, jj, kk, status, cnt, iOne=1, iZero=0, strLeng, commFlag;
  int    maxPts=257, nbins, printLevel, faType, nPlots, proc;
  int    maxSamples,burnInSamples,modelFormFlag=0,numChains=3,nChainGood=0;
  int    freq=1, dnSamples=0, dnInputs=0, rsErrFlag=0, genPosteriors=0;
  double psrfThreshold=1.05;
  char   lineIn[1001], charString[1001], cfname[501], *rsFile=NULL;
  char   commBuffer[1001], winput[1001];
  FILE   *fp=NULL;
  pData      pPtr, qData, pOutputs;
  FuncApprox **faPtrs=NULL, **faPtrs1=NULL;
  PDFBase    **inputPDFs;
  psIVector  vecPlotInds, vecRSInds, vecDesParams;
  psVector   vecRSVals, vecDSamInps, vecDMeans, vecDStdvs, vecYT;
  psVector   vecDiscConstMeans, vecDiscConstStdvs;

  //**/ ---------------------------------------------------------------
  //**/ display header 
  //**/ ---------------------------------------------------------------
  printLevel = adata.printLevel_;
  if (mypid_ == 0)
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"*             MCMC Optimizer\n");
    printEquals(PL_INFO, 0);
    if (printLevel > 0)
    {
      printOutTS(PL_INFO,"TO GAIN ACCESS TO DIFFERENT OPTIONS: TURN ON\n\n");
      printOutTS(PL_INFO," * ana_expert to finetune PMCMC parameters, \n");
      printOutTS(PL_INFO,"   (e.g. adjust sample size for burn-in).\n");
      printOutTS(PL_INFO," * printlevel 3 for more diagnostics.\n");
      printOutTS(PL_INFO," * printlevel 4 for even more information.\n");
      printDashes(PL_INFO,0);
      printOutTS(PL_INFO,"FEATURES AVAILABLE IN THE CURRENT VERSION :\n");
      printOutTS(PL_INFO," * Support different priors (default: uniform)\n");
      printOutTS(PL_INFO,"   - ana_expert to use non-uniform priors.\n");
      printOutTS(PL_INFO," * Support likelihood from multiple outputs\n");
      printOutTS(PL_INFO," * Option to include response surface errors\n");
      printOutTS(PL_INFO,"   for polynomial regressions, bootstrapped\n");
      printOutTS(PL_INFO,"   MARS, and Gaussian process (GP1).\n");
      printOutTS(PL_INFO," * Option to include model form errors in the\n");
      printOutTS(PL_INFO,"   form of discrepancy models.\n");
      printOutTS(PL_INFO,"   - select these options in ana_expert mode\n");
      printOutTS(PL_INFO," * Some input parameters may be designated as\n");
      printOutTS(PL_INFO,"   design parameters.\n");
      printOutTS(PL_INFO," * Some input parameters may be disabled (set\n");
      printOutTS(PL_INFO,"   to default values)\n");
      printOutTS(PL_INFO,"   - use rs_index_file in PSUADE data file\n");
      printOutTS(PL_INFO,"   - feature not available with discrepancy\n");
      printOutTS(PL_INFO," * Option to generate a posterior sample\n");
      printOutTS(PL_INFO,"   - select this option in ana_expert mode\n");
      printOutTS(PL_INFO," * This analysis can be terminated gracefully\n");
      printOutTS(PL_INFO,"   by creating a file named 'psuade_stop' in\n");
      printOutTS(PL_INFO,"   the same directory during execution.\n");
      if (psConfig_.AnaExpertModeIsOn())
      {
        printOutTS(PL_INFO," * For multi-modal posteriors, a large number\n");
        printOutTS(PL_INFO,"   of chains may be needed. You can make this\n");
        printOutTS(PL_INFO,"   selection in ana_expert mode.\n");
      }
    }
    printEquals(PL_INFO, 0);
  }

  //**/ ---------------------------------------------------------------
  //**/ extract data from aData object (passed in from outside)
  //**/ ---------------------------------------------------------------
  nInputs_      = adata.nInputs_;
  nOutputs_     = adata.nOutputs_;
  int nInputs   = adata.nInputs_;
  int nOutputs  = adata.nOutputs_;
  int nSamples  = adata.nSamples_;
  double *X     = adata.sampleInputs_;
  double *Y     = adata.sampleOutputs_;
  double *lower = adata.iLowerB_;
  double *upper = adata.iUpperB_;
  //**/ indicate pdf type
  int    *pdfFlags    = adata.inputPDFs_;
  double *inputMeans  = adata.inputMeans_;
  double *inputStdevs = adata.inputStdevs_;
  //**/ get names for input (for use in plotting)
  PsuadeData *ioPtr = adata.ioPtr_;
  if (ioPtr != NULL) ioPtr->getParameter("input_names", qData);

  //**/ ---------------------------------------------------------------
  //**/ non-uniform distribution used for expert mode only
  //**/ ---------------------------------------------------------------
  if (psConfig_.AnaExpertModeIsOn())
  {
    if (pdfFlags != NULL)
    {
      ii2 = 0;
      for (ii = 0; ii < nInputs; ii++) ii2 += pdfFlags[ii];
      if (ii2 > 0)
      {
        if (mypid_ == 0)
        {
          printOutTS(PL_ERROR,"PMCMC WARNING: non-uniform priors\n");
          printOutTS(PL_ERROR,"   only supported in ana_expert mode.\n");
          printOutTS(PL_ERROR,"   CONTINUE with uniform priors.\n");
        }
        for (ii = 0; ii < nInputs; ii++) pdfFlags[ii] = 0;
      }
    }
  }

  //**/ ---------------------------------------------------------------
  // error checking
  //**/ ---------------------------------------------------------------
  if (nInputs <= 0 || nOutputs <= 0 || nSamples <= 0)
  {
    commBuffer[0] = '\0';
    if (mypid_ == 0)
    {
      printOutTS(PL_ERROR,"PMCMC ERROR: invalid nInputs/nOutputs/nSamples\n");
      printOutTS(PL_ERROR,"    nSamples = %d\n", nSamples);
      printOutTS(PL_ERROR,"    nInputs  = %d\n", nInputs);
      printOutTS(PL_ERROR,"    nOutputs = %d\n", nOutputs);
      strcpy(commBuffer,"psuadeError");
      strLeng = 12;
      commBuffer[strLeng-1] = '\0';
    }
    commMgr_->bcast((void *) &strLeng, iOne, INT, 0);
    commMgr_->bcast((void *) commBuffer, strLeng, CHAR, 0);
    if (!strcmp(commBuffer, "psuadeError")) return PSUADE_UNDEFINED;
  }
  status = 0;
  for (ii = 0; ii < nSamples*nOutputs; ii++)
    if (Y[ii] > 0.9*PSUADE_UNDEFINED) status = 1;
  if (status == 1)
  {
    if (mypid_ == 0)
    {
      printOutTS(PL_ERROR,"PMCMC ERROR: Some outputs are undefined.\n");
      printOutTS(PL_ERROR,"  Prune the undefined sample points & re-run\n");
    }
    return PSUADE_UNDEFINED;
  }
  if (mypid_ == 0)
  {
    fp = fopen("psuade_stop", "r");
    if (fp != NULL)
    {
      fclose(fp);
      fp = NULL;
      printOutTS(PL_INFO,"PMCMC INFO: psuade_stop file found\n");
      printOutTS(PL_INFO,"            (will be removed).\n");
      strcpy(charString, "psuade_stop");
      unlink(charString);
    }
  }

  //**/ ---------------------------------------------------------------
  // get experimental data information from the spec file
  //**/ ---------------------------------------------------------------
  if (printLevel > 0 && mypid_ == 0)
  {
    printOutTS(PL_INFO,"*** NEED DATA TO CREATE LIKELIHOOD FUNCTION:\n\n");
    printOutTS(PL_INFO,"PMCMC creates a Gaussian likelihood function.\n");
    printOutTS(PL_INFO,"Please provide a data file containing design\n");
    printOutTS(PL_INFO,"parameter values, mean, and std. dev. of the\n");
    printOutTS(PL_INFO,"observation data for each output.\n");
    printOutTS(PL_INFO,"NOTE: Design parameters should be defined in the\n");
    printOutTS(PL_INFO,"   observation data file if the data used in PMCMC\n");
    printOutTS(PL_INFO,"   are collected at different design values.\n");
    printOutTS(PL_INFO,"IMPORTANT: IF m DESIGN PARAMETERS ARE SPECIFIED,\n");
    printOutTS(PL_INFO,"   YOU NEED TO SPECIFY WHICH INPUTS THEY ARE.\n");
    printOutTS(PL_INFO,"   THESE DESIGN PARAMETERS WILL BE EXCLUDED FROM\n");
    printOutTS(PL_INFO,"   BEING IN THE CALIBRATION PARAMETER SET.\n");
    printDashes(PL_INFO, 0);
    printOutTS(PL_INFO,"*** OBSERVATION DATA FILE FORMAT : \n");
    printOutTS(PL_INFO,"        M   - no. of design parameters, \n");
    printOutTS(PL_INFO,"        K   - no. of model outputs, \n");
    printOutTS(PL_INFO,"        P   - no. of experiments \n");
    printOutTS(PL_INFO,"        O1m - Output 1 mean\n");
    printOutTS(PL_INFO,"        O1s - Output 1 std. dev.\n");
    printOutTS(PL_INFO,"        OKs - Output K std. dev.\n");
    printOutTS(PL_INFO,"PSUADE_BEGIN\n");
    printOutTS(PL_INFO,"<P> <K> <M> <design parameter identifiers>\n");
    printOutTS(PL_INFO,"1 <design values...> <O1m> <O1s> ... <OKs> \n");
    printOutTS(PL_INFO,"2 <design values...> <O1m> <O1s> ... <OKs> \n");
    printOutTS(PL_INFO,"...\n");
    printOutTS(PL_INFO,"P <design values...> <O1m> <O1s> ... <OK> \n");
    printOutTS(PL_INFO,"PSUADE_END\n");
    printDashes(PL_INFO, 0);
    printOutTS(PL_INFO,"The likelihood function is in the form of:\n");
    printOutTS(PL_INFO,"  C exp(-0.5*S) \n");
    printOutTS(PL_INFO,"where C is the normalization constant and\n");
    printOutTS(PL_INFO,"  S=1/P sum_{p=1}^P sum_{k=1)^K (Y_pk-m_pk)^2/sd_pk^2\n");
    printOutTS(PL_INFO,"where K is the number of outputs and m_pk and\n");
    printOutTS(PL_INFO,"  sd_pk are the mean and std. dev. of output k\n");
    printOutTS(PL_INFO,"  of experiment k.\n");
    printDashes(PL_INFO, 0);
    printOutTS(PL_INFO,"NOTE: Alternately, your response surface output\n");
    printOutTS(PL_INFO,"   may be some error measure from comparison of\n");
    printOutTS(PL_INFO,"   model outputs with observation data. In this\n");
    printOutTS(PL_INFO,"   case, set nOutputs=1, mean=0 and std. dev.=1\n");
    printOutTS(PL_INFO,"   in the specification file (that is, your\n");
    printOutTS(PL_INFO,"   simulation output is S above and MCMC will\n");
    printOutTS(PL_INFO,"   compute likelihood as C exp(-0.5 S).\n");
  }

  //**/ check existence of spec file for likelihood function
  if (mypid_ == 0)
  {
    printf("===> Enter the spec file for building the likelihood function : ");
    scanf("%s", cfname);
    fgets(lineIn, 1000, stdin);
    kk = strlen(cfname);
    if (kk <= 1000)
    {
      cfname[kk] = '\0';
      fp = fopen(cfname, "r");
      if (fp == NULL)
      {
        printOutTS(PL_ERROR,"PMCMC ERROR : cannot open spec file %s.\n", 
                   cfname);
        strcpy(commBuffer,"psuadeError");
        strLeng = 12;
        commBuffer[strLeng-1] = '\0';
      }
      else
      {
        fclose(fp);
        strcpy(commBuffer,"psuadeOkay");
        strLeng = 11;
        commBuffer[strLeng-1] = '\0';
      }
    }
    else
    {
      printOutTS(PL_ERROR,"PMCMC ERROR: file name too long.\n");
      strcpy(commBuffer,"psuadeError");
      strLeng = 12;
      commBuffer[strLeng-1] = '\0';
    }
  }
  commMgr_->bcast((void *) &strLeng, iOne, INT, 0);
  commMgr_->bcast((void *) commBuffer, strLeng, CHAR, 0);
  if (!strcmp(commBuffer, "psuadeError")) return PSUADE_UNDEFINED;

  //**/ read spec file for likelihood function
  if (mypid_ == 0)
  {
    strcpy(commBuffer,"psuadeOkay");
    strLeng = 11;
    commBuffer[strLeng-1] = '\0';
    fp = fopen(cfname, "r");
    fgets(lineIn, 1000, fp);
    sscanf(lineIn, "%s", winput);
    if (!strcmp(winput, "PSUADE_BEGIN"))
    {
      fscanf(fp, "%d %d %d", &dnSamples, &kk, &dnInputs);
      if (dnSamples <= 0)
      {
        printOutTS(PL_ERROR,"PMCMC ERROR: no. of experiments <= 0.\n");
        fclose(fp);
        strcpy(commBuffer,"psuadeError");
        strLeng = 12;
        commBuffer[strLeng-1] = '\0';
      }
      else
      {
        printOutTS(PL_INFO,"SPEC FILE: Number of experiments = %d\n",
                   dnSamples);
      }
      if (kk != nOutputs)
      {
        printOutTS(PL_ERROR,"PMCMC ERROR: nOutputs in spec file\n");
        printOutTS(PL_ERROR,"      does not match PSUADE file nOutputs.\n");
        printOutTS(PL_ERROR,"      %d compared to %d\n", kk, nOutputs);
        fclose(fp);
        strcpy(commBuffer,"psuadeError");
        strLeng = 12;
        commBuffer[strLeng-1] = '\0';
      }
      printOutTS(PL_INFO,"SPEC FILE: Number of outputs = %d\n",nOutputs);
      if (dnInputs < 0)
      {
        printOutTS(PL_ERROR,"PMCMC ERROR: no. of design variables < 0.\n");
        fclose(fp);
        strcpy(commBuffer,"psuadeError");
        strLeng = 12;
        commBuffer[strLeng-1] = '\0';
      }
      if (dnInputs > nInputs)
      {
        printOutTS(PL_ERROR,"PMCMC ERROR: no. of design variables %d\n",
                   dnInputs);
        printOutTS(PL_ERROR,"      cannot be larger than the total number\n");
        printOutTS(PL_ERROR,"      of inputs %d.\n", nInputs);
        fclose(fp);
        strcpy(commBuffer,"psuadeError");
        strLeng = 12;
        commBuffer[strLeng-1] = '\0';
      }
      printOutTS(PL_INFO,"SPEC FILE: Number of design parameters = %d\n",
                 dnInputs);
      if (dnInputs > 0)
      {
        vecDesParams.setLength(nInputs);
        for (ii = 0; ii < nInputs; ii++) vecDesParams[ii] = 0;
        cnt = 0;
        for (ii = 0; ii < dnInputs; ii++)
        {
          fscanf(fp, "%d", &kk);
          if (kk <= 0 || kk > nInputs)
          {
            printOutTS(PL_ERROR,
                 "PMCMC ERROR: invalid design parameter %d\n",kk);
            fclose(fp);
            strcpy(commBuffer,"psuadeError");
            strLeng = 12;
            commBuffer[strLeng-1] = '\0';
          }
          if (kk <= cnt)
          {
            printOutTS(PL_ERROR,"PMCMC ERROR: design parameters should \n");
            printOutTS(PL_ERROR,"             be in ascending order.\n");
            fclose(fp);
            strcpy(commBuffer,"psuadeError");
            strLeng = 12;
            commBuffer[strLeng-1] = '\0';
          }
          vecDesParams[kk-1] = 1;
          printOutTS(PL_INFO,"SPEC FILE: Input %d is a design parameter\n",
                     kk);
          cnt = kk;
        }
        vecDSamInps.setLength(dnSamples*dnInputs);
      }
      vecDMeans.setLength(dnSamples*nOutputs);
      vecDStdvs.setLength(dnSamples*nOutputs);
      for (ii = 0; ii < dnSamples; ii++)
      {
        fscanf(fp, "%d", &kk);
        if (kk != ii+1)
        {
          printOutTS(PL_ERROR,"PMCMC ERROR: invalid experiment number %d\n",
                     kk);
          printOutTS(PL_ERROR,"      at line %d in spec the file:\n",ii+2);
          printOutTS(PL_ERROR,"            (Expecting %d).\n", ii+1);
          printOutTS(PL_ERROR,"==> check line %d\n", ii+3);
          fclose(fp);
          strcpy(commBuffer,"psuadeError");
          strLeng = 12;
          commBuffer[strLeng-1] = '\0';
        }
        if (printLevel > 0)
          printOutTS(PL_INFO,"Calibration Data Set %d\n", kk);
        for (jj = 0; jj < dnInputs; jj++)
        {
          fscanf(fp, "%lg", &vecDSamInps[ii*dnInputs+jj]);
          if (printLevel > 0)
            printOutTS(PL_INFO,"   Design parameter %d = %e\n", jj+1, 
                       vecDSamInps[ii*dnInputs+jj]);
        }
        for (jj = 0; jj < nOutputs; jj++)
        {
          fscanf(fp, "%lg %lg", &vecDMeans[ii*nOutputs+jj],
                                &vecDStdvs[ii*nOutputs+jj]);
          if (printLevel > 0)
            printOutTS(PL_INFO,"      Data mean/stdev = %16.8e %16.8e\n",
                 vecDMeans[ii*nOutputs+jj],vecDStdvs[ii*nOutputs+jj]);
          if (vecDStdvs[ii*nOutputs+jj] < 0.0)
          {
            fclose(fp);
            printOutTS(PL_ERROR,"PMCMC ERROR: std dev in spec file <= 0.\n");
            printOutTS(PL_ERROR,"=> check the last entry in line %d\n",ii+3);
            strcpy(commBuffer,"psuadeError");
            strLeng = 12;
            commBuffer[strLeng-1] = '\0';
          }
        }
      }
    }
    else
    {
      printOutTS(PL_ERROR, 
              "PMCMC ERROR: PSUADE_BEGIN missing in the spec file.\n");
      fclose(fp);
      strcpy(commBuffer,"psuadeError");
      strLeng = 12;
      commBuffer[strLeng-1] = '\0';
    }
    fgets(lineIn, 1000, fp);
    fgets(lineIn, 1000, fp);
    sscanf(lineIn, "%s", winput);
    if (strcmp(winput, "PSUADE_END"))
    {
      printOutTS(PL_ERROR,
                "PMCMC ERROR: PSUADE_END missing in the spec file.\n");
      fclose(fp);
      strcpy(commBuffer,"psuadeError");
      strLeng = 12;
      commBuffer[strLeng-1] = '\0';
    }
    fclose(fp);
    printEquals(PL_INFO, 0);
  }
  commMgr_->bcast((void *) &strLeng, iOne, INT, 0);
  commMgr_->bcast((void *) commBuffer, strLeng, CHAR, 0);
  if (!strcmp(commBuffer, "psuadeError")) return PSUADE_UNDEFINED;

  //**/ ------------------------------------------------------------
  //**/ send spec file information to other processors
  //**/ ------------------------------------------------------------
  commMgr_->bcast((void *) &dnSamples, iOne, INT, 0);
  commMgr_->bcast((void *) &dnInputs, iOne, INT, 0);
  if (mypid_ != 0)
  {
    if (dnInputs > 0)
    {
      vecDesParams.setLength(nInputs);
      vecDSamInps.setLength(dnSamples*dnInputs);
    }
    vecDMeans.setLength(dnSamples*nOutputs);
    vecDStdvs.setLength(dnSamples*nOutputs);
  }
  if (dnInputs > 0)
  {
    commMgr_->bcast((void *) &nInputs, iOne, INT, 0);
    commMgr_->bcast((void *) vecDesParams.getIVector(),nInputs,INT,0);
    kk = dnSamples * dnInputs;
    commMgr_->bcast((void *) vecDSamInps.getDVector(), kk, INT, 0);
  }
  kk = dnSamples * nOutputs;
  commMgr_->bcast((void *) vecDMeans.getDVector(), kk, DOUBLE, 0);
  commMgr_->bcast((void *) vecDStdvs.getDVector(), kk, DOUBLE, 0);

  //**/ ------------------------------------------------------------
  //**/ option to add response surface uncertainties to data std dev
  //**/ ------------------------------------------------------------
  if (psConfig_.AnaExpertModeIsOn() && mypid_ == 0)
  {
    printOutTS(PL_INFO,"*** OPTION TO INCLUDE RESPONSE SURFACE UNCERTAINTIES:\n");
    printOutTS(PL_INFO,"\nTo incorporate the response surface errors into\n");
    printOutTS(PL_INFO,"the likelihood function, make sure that either GP,\n");
    printOutTS(PL_INFO,"Kriging, polynomial regression, or bootstrapped\n");
    printOutTS(PL_INFO,"MARS response surface is selected in the simulation\n");
    printOutTS(PL_INFO,"data file. Otherwise, no RS uncertainties will be\n");
    printOutTS(PL_INFO,"included.\n\n");
    printOutTS(PL_INFO,"NOTE: if you don't know what this is, just say no.\n");
    printf( "===> Include response surface uncertainties? (y or n) ");
    scanf("%s", charString);
    fgets(lineIn,1000,stdin);
    if (charString[0] == 'y') rsErrFlag = 1;
    printEquals(PL_INFO, 0);
  }
  commMgr_->bcast((void *) &rsErrFlag, iOne, INT, 0);

  //**/ ---------------------------------------------------------------
  //**/ create response surface for use in computing likelihood
  //**/ This is an option to specify a rs index file in the data file.
  //**/ This option allows one to disable a certain input in the MCMC
  //**/ optimization ==> faPtrs.
  //**/ ---------------------------------------------------------------
  if (ioPtr != NULL && mypid_ == 0)
  {
    ioPtr->getParameter("ana_rstype", pPtr);
    faType = pPtr.intData_;
    ioPtr->getParameter("ana_rsindexfile", pPtr);
    rsFile = pPtr.strArray_[0];
    if (strcmp(rsFile, "NONE"))
    {
      printOutTS(PL_INFO,"A response surface index file has been specified.\n");
      fp = fopen(rsFile, "r");
      if (fp == NULL)
      {
        printOutTS(PL_ERROR,"PMCMC ERROR: rs_index_file %s not found.\n",
                   rsFile);
        strcpy(commBuffer,"psuadeError");
        strLeng = 12;
        commBuffer[strLeng-1] = '\0';
      }
      else
      {
        printOutTS(PL_INFO,"INFO: rs_index_file %s found.\n",rsFile);
        fscanf(fp,"%d", &kk);
        if (kk != nInputs)
        {
          printOutTS(PL_ERROR,
               "PMCMC ERROR: invalid nInputs in rs_index_file.\n");
          printOutTS(PL_ERROR," Data format should be: \n");
          printOutTS(PL_ERROR," line 1: nInputs in rs data (driver) file\n");
          printOutTS(PL_ERROR,
               " line 2: 1 <1 or 0> <default value if first number==0>\n");
          printOutTS(PL_ERROR,
               " line 3: 2 <2 or 0> <0 if first number != 0>\n");
          printOutTS(PL_ERROR,
               " line 4: 3 <3 or 0> <default value if first number==0>\n");
          printOutTS(PL_ERROR,
               " line 5: 4 <4 or 0> <0 if first number != 0>\n");
          printOutTS(PL_ERROR,"  ...\n");
          fclose(fp);
          strcpy(commBuffer,"psuadeError");
          strLeng = 12;
          commBuffer[strLeng-1] = '\0';
        }
        vecRSInds.setLength(nInputs);
        vecRSVals.setLength(nInputs);
        for (ii = 0; ii < nInputs; ii++) vecRSInds[ii] = 0;
        for (ii = 0; ii < nInputs; ii++)
        {
          fscanf(fp, "%d", &kk);
          if (kk != ii+1)
          {
            printOutTS(PL_ERROR,
               "PMCMC ERROR: 1st index in indexFile = %d (must be %d]).\n",
               kk, ii+1);
            printOutTS(PL_ERROR," Data format should be: \n");
            printOutTS(PL_ERROR,
               " line 1: nInputs in rs data (driver) file\n");
            printOutTS(PL_ERROR,
               " line 2: 1 <1 or 0> <default value if first number==0>\n");
            printOutTS(PL_ERROR,
               " line 3: 2 <2 or 0> <0 if first number != 0>\n");
            printOutTS(PL_ERROR,
               " line 4: 3 <3 or 0> <default value if first number==0>\n");
            printOutTS(PL_ERROR,
               " line 5: 4 <4 or 0> <0 if first number != 0>\n");
            printOutTS(PL_ERROR,"  ...\n");
            fclose(fp);
            strcpy(commBuffer,"psuadeError");
            strLeng = 12;
            commBuffer[strLeng-1] = '\0';
          }
          fscanf(fp, "%d", &vecRSInds[ii]);
          if (vecRSInds[ii] == 0)
            printOutTS(PL_INFO,"PMCMC INFO: input %3d inactive\n",ii+1);

          if (vecRSInds[ii] == 0 && vecDesParams.length() > 0 && 
              vecDesParams[ii] == 1)
          {
            printOutTS(PL_ERROR,
                 "PMCMC ERROR: inactive input %d cannot be design parameter\n",
                 ii+1);
            fclose(fp);
            strcpy(commBuffer,"psuadeError");
            strLeng = 12;
            commBuffer[strLeng-1] = '\0';
          }
          if (vecRSInds[ii] < 0 || vecRSInds[ii] > nInputs)
          {
            printOutTS(PL_ERROR, "MCMC INFO: input %3d = %d invalid\n",
                       ii+1,vecRSInds[ii]);
            fclose(fp);
            strcpy(commBuffer,"psuadeError");
            strLeng = 12;
            commBuffer[strLeng-1] = '\0';
          }
          vecRSInds[ii]--;
          fscanf(fp, "%lg", &vecRSVals[ii]);
        }
        fclose(fp);
        printOutTS(PL_INFO, "Response surface index information: \n");
        for (ii = 0; ii < nInputs; ii++)
          printOutTS(PL_INFO, "Input %4d: index = %4d, default = %e\n",
                     ii+1, vecRSInds[ii]+1, vecRSVals[ii]);
      }
    }
  }
  commMgr_->bcast((void *) &strLeng, iOne, INT, 0);
  commMgr_->bcast((void *) commBuffer, strLeng, CHAR, 0);
  if (!strcmp(commBuffer, "psuadeError")) return PSUADE_UNDEFINED;

  //**/ broadcast vecRSInds and vecRSVals, if any
  if (mypid_ == 0)
  {
    commFlag = 1;
    if (vecRSInds.length() == 0) commFlag = 0;
  }
  commMgr_->bcast((void *) &commFlag, iOne, INT, 0);
  if (mypid_ != 0 && commFlag == 1)
  {
    vecRSInds.setLength(nInputs);
    vecRSVals.setLength(nInputs);
  }
  if (commFlag == 1)
  {
    commMgr_->bcast((void *) vecRSInds.getIVector(),nInputs,INT,0);
    commMgr_->bcast((void *) vecRSVals.getDVector(),nInputs,INT,0);
  }
  commMgr_->bcast((void *) &faType, iOne, INT, 0);

  if (mypid_ == 0)
     printOutTS(PL_INFO,
          "MCMC INFO: CREATING RESPONSE SURFACES FOR ALL OUTPUTS.\n");
  //**/ Turn off response surface expert mode not to ask too many
  //**/ questions
  psConfig_.RSExpertModeOff();
  faPtrs = new FuncApprox*[nOutputs];
  vecYT.setLength(nSamples);
  for (ii = 0; ii < nOutputs; ii++)
  {
    faType = -1;
    if (mypid_ == 0)
    {
      printOutTS(PL_INFO,
           "MCMC INFO: CREATING RESPONSE SURFACE FOR OUTPUT %d.\n",ii+1);
      faPtrs[ii] = genFA(faType, nInputs, iZero, nSamples);
      faType = faPtrs[ii]->getID();
    }
    commMgr_->bcast((void *) &faType, iOne, INT, 0);
    if (mypid_ != 0)
      faPtrs[ii] = genFA(faType, nInputs, iZero, nSamples);
    faPtrs[ii]->setNPtsPerDim(16);
    faPtrs[ii]->setBounds(lower, upper);
    faPtrs[ii]->setOutputLevel(0);
    for (kk = 0; kk < nSamples; kk++) vecYT[kk] = Y[kk*nOutputs+ii];

    status = faPtrs[ii]->initialize(X, vecYT.getDVector());
    commMgr_->allReduce((void *) &status, iOne, INT, '+');
    if (status != 0)
    {
      if (mypid_ == 0)
      {
        printOutTS(PL_ERROR,"PMCMC ERROR: Unable to create response surface\n");
        printOutTS(PL_ERROR,"             Consult PSUADE developers.\n");
      }
      for (kk = 0; kk < ii; kk++) delete [] faPtrs[kk];
      delete [] faPtrs;
      return PSUADE_UNDEFINED;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ special set up (sample size) using interactive mode
  //    ==> burnInSamples, maxSamples, nbins, vecPlotInds, nPlots
  //**/ ---------------------------------------------------------------
  maxSamples = 10000;
  burnInSamples = maxSamples / 2;
  nbins = 20;
  if (mypid_ == 0)
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"*** CURRENT SETTINGS OF MCMC PARAMETERS: \n\n");
    printOutTS(PL_INFO,"MCMC Burn-in sample size      (default) = %d\n",
               burnInSamples);
    printOutTS(PL_INFO,"MCMC sample increment         (default) = %d\n", 
               maxSamples);
    printOutTS(PL_INFO,"MCMC no. of bins in histogram (default) = %d\n", 
               nbins);
    printOutTS(PL_INFO,"NOTE: sample increment - sample size to run ");
    printOutTS(PL_INFO,"before convergence check\n");
    printOutTS(PL_INFO,"NOTE: histogram nBins  - define granularity of ");
    printOutTS(PL_INFO,"histogram bar graph\n");
  }
  vecPlotInds.setLength(nInputs);
  nPlots = 0;
  for (ii = 0; ii < nInputs; ii++) 
  {
    if (vecRSInds.length() == 0 || vecRSInds[ii] >= 0)
      if (vecDesParams.length() == 0 || vecDesParams[ii] == 0) 
        vecPlotInds[nPlots++] = ii;
  }

  //**/ ---------------------------------------------------------------
  // option to add discrepancy function and a posterior sample
  // ==> modelFormFlag, genPosteriors
  //**/ ---------------------------------------------------------------
  if (psConfig_.AnaExpertModeIsOn() && mypid_ == 0)
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,"*** OPTION TO ADD A DISCREPANCY FUNCTION:\n\n");
    printOutTS(PL_INFO,
       "To use this feature, first make sure that the observation\n");
    printOutTS(PL_INFO,
       "data file specified earlier has design parameters specified\n");
    printOutTS(PL_INFO,
       "since the discrepancy function is to be a function of these\n");
    printOutTS(PL_INFO,
       "design parameters (if not, a constant discrepancy function\n");
    printOutTS(PL_INFO,"is to be created).\n");
    printOutTS(PL_INFO,"NOTE: if you don't know what this is, just say NO.\n");
    printf("===> Add discrepancy function ? (y or n) ");
    scanf("%s", charString);
    fgets(lineIn,1000,stdin);
    if (charString[0] == 'y') modelFormFlag = 1;

    //**/ add posterior propagation
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,
      "*** OPTION TO CREATE A SAMPLE FROM THE POSTERIOR DISTRIBUTIONS:\n\n");
    printOutTS(PL_INFO,
      "In addition to generating the posterior distributions, you can\n");
    printOutTS(PL_INFO,
      "also draw a sample from these posteriors. The posterior sample\n");
    printOutTS(PL_INFO,
      "can be used as prior sample for another simulator/emulator.\n");
    printOutTS(PL_INFO,
      "NOTE: if you don't know what this is, just say no.\n");
    printf("==> Generate posterior samples for the parameters? (y or n) ");
    scanf("%s", charString);
    fgets(lineIn,1000,stdin);
    if (charString[0] == 'y') genPosteriors = 1;
  } 
  if (mypid_ == 0) printEquals(PL_INFO, 0);
  commMgr_->bcast((void *) &modelFormFlag, iOne, INT, 0);
  commMgr_->bcast((void *) &genPosteriors, iOne, INT, 0);

  //**/ ---------------------------------------------------------------
  // setup input PDF, if there is any
  //**/ ---------------------------------------------------------------
  if (printLevel > 2 && mypid_ == 0) 
    printOutTS(PL_INFO,"*** INFORMATION ON PARAMETER PRIOR DISTRIBUTIONS\n");
  inputPDFs = new PDFBase*[nInputs];
  for (ii = 0; ii < nInputs; ii++)
  {
    if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_NORMAL)
    {
      inputPDFs[ii] = (PDFBase *) new PDFNormal(inputMeans[ii],inputStdevs[ii]);
      if (mypid_ == 0) 
        printOutTS(PL_INFO,
             "Parameter %3d has normal prior distribution (%e,%e).\n",
             ii+1, inputMeans[ii], inputStdevs[ii]);
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_LOGNORMAL)
    {
      inputPDFs[ii] = (PDFBase *) new PDFLogNormal(inputMeans[ii],
                                                   inputStdevs[ii]);
      if (mypid_ == 0) 
        printOutTS(PL_INFO,
             "Parameter %3d has lognormal prior distribution.\n",ii+1);
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_TRIANGLE)
    {
      inputPDFs[ii] = (PDFBase *) new PDFTriangle(inputMeans[ii],
                                                  inputStdevs[ii]);
      if (mypid_ == 0) 
        printOutTS(PL_INFO,
             "Parameter %3d has triangle prior distribution.\n",ii+1);
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_BETA)
    {
      inputPDFs[ii] = (PDFBase *) new PDFBeta(inputMeans[ii],
                                              inputStdevs[ii]);
      if (mypid_ == 0) 
        printOutTS(PL_INFO,
             "Parameter %3d has beta prior distribution.\n",ii+1);
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_WEIBULL)
    {
      inputPDFs[ii] = (PDFBase *) new PDFWeibull(inputMeans[ii],
                                                 inputStdevs[ii]);
      if (mypid_ == 0) 
        printOutTS(PL_INFO,
             "Parameter %3d has Weibull prior distribution.\n",ii+1);
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_GAMMA)
    {
      inputPDFs[ii] = (PDFBase *) new PDFGamma(inputMeans[ii],
                                               inputStdevs[ii]);
      if (mypid_ == 0) 
        printOutTS(PL_INFO,
             "Parameter %3d has gamma prior distribution.\n",ii+1);
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == 1000+PSUADE_PDF_NORMAL)
    {
      inputPDFs[ii] = NULL;
      if (mypid_ == 0) 
      {
        printOutTS(PL_INFO,
             "Parameter %3d: multi-parameter normal distribution.\n",ii+1);
        printOutTS(PL_INFO,"               curently not supported.\n");
      }
      return -1.0;
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == 1000+PSUADE_PDF_LOGNORMAL)
    {
      inputPDFs[ii] = NULL;
      if (mypid_ == 0) 
      {
        printOutTS(PL_INFO,
             "Parameter %3d: multi-parameter lognormal distribution.\n",ii+1);
        printOutTS(PL_INFO,"               curently not supported.\n");
      }
      return -1.0;
    }
    else if (pdfFlags == NULL || pdfFlags[ii] == PSUADE_PDF_UNIFORM)
    {
      inputPDFs[ii] = NULL;
      if (mypid_ == 0) 
        printOutTS(PL_INFO,
             "Parameter %3d has uniform prior distribution.\n",ii+1);
    }
    else if (pdfFlags != NULL && pdfFlags[ii] == PSUADE_PDF_SAMPLE)
    {
      inputPDFs[ii] = NULL;
      if (mypid_ == 0) 
      {
        printOutTS(PL_INFO,
             "Parameter %3d: user-provided distribution currently not\n",
             ii+1);
        printOutTS(PL_INFO,"               supported.\n");
      }
      return -1.0;
    }
  }
  if (printLevel > 2 && mypid_ == 0) printEquals(PL_INFO, 0);

  //**/ ---------------------------------------------------------------
  //**/ set max number of points in proposal distributions
  //**/ ---------------------------------------------------------------
  maxPts = nbins * 5;
  if (psConfig_.AnaExpertModeIsOn() && mypid_ == 0)
  {
    printOutTS(PL_INFO,"*** SETTING PROPOSAL DISTRIBUTION RESOLUTION\n");
    printOutTS(PL_INFO,
         "Since MCMC uses many function evaluations to construct\n");
    printOutTS(PL_INFO,
         "the proposal distributions, you have the option to set\n");
    printOutTS(PL_INFO,
         "how many points are used to construct it in order to\n");
    printOutTS(PL_INFO,
         "keep the inference cost reasonable.\n");
    printf("Sample size to construct proposal distributions.\n");
    printf("Default is %d.\n",maxPts);
    snprintf(charString,100,"Enter new sample size (%d - %d): ",
             nbins*3,nbins*10);
    maxPts = getInt(nbins*3, nbins*10, charString);
    maxPts = maxPts / nbins * nbins;
    printOutTS(PL_INFO,"Proposal distribution sample size = %d.\n",maxPts);
  }
  commMgr_->bcast((void *) &maxPts, iOne, INT, 0);

  //**/ ---------------------------------------------------------------
  //**/ create discrepancy function, if desired ==> faPtrs1
  //**/ ---------------------------------------------------------------
  double *discOutputs=NULL;
  if (modelFormFlag == 1)
  {
    int    *ExpSamStates, ind, ExpNSamples, dfaType, dnPerDim=16;
    double *dOneSample, expdata, simdata;
    double *ExpSamInputs, *tSamInputs, *settings;

    //**/ allocate space for discrepancy data
    ExpNSamples   = dnSamples;
    ExpSamInputs  = new double[ExpNSamples*nInputs];
    discOutputs   = new double[ExpNSamples*nOutputs];
    ExpSamStates  = new int[ExpNSamples];
    vecDiscConstMeans.setLength(nOutputs);
    vecDiscConstStdvs.setLength(nOutputs);
    for (ii2 = 0; ii2 < nOutputs; ii2++)
    {
      vecDiscConstMeans[ii2] = PSUADE_UNDEFINED;
      vecDiscConstStdvs[ii2] = PSUADE_UNDEFINED;
    }
    //**/ create experiment response surfaces
    if (mypid_ == 0)
    {
      printOutTS(PL_INFO,
           "*** SELECT RESPONSE SURFACE TYPE FOR DISCREPANCY FUNCTION:\n");
      dfaType = -1;
      while (dfaType < 0 || dfaType >= PSUADE_NUM_RS)
      {
        writeFAInfo(0);
        snprintf(charString,100,"===> Enter your choice : ");
        dfaType = getInt(0, PSUADE_NUM_RS-1, charString);
      }
    }
    commMgr_->bcast((void *) &dfaType, iOne, INT, 0);

    //**/ set discrepancy function calibration parameter default 
    //**/ and also fixed parameters ==> settings
    settings = new double[nInputs];
    checkAllocate(settings, "settings in PMCMC::analyze");
    for (ii2 = 0; ii2 < nInputs; ii2++)
    {
      //if (inputPDFs == NULL || (inputPDFs != NULL && inputPDFs[ii2] == NULL))
      //   settings[ii2] = 0.5*(lower[ii2] + upper[ii2]);
      //else
      //   settings[ii2] = inputPDFs[ii2]->getMean();
      settings[ii2] = 0.5*(lower[ii2] + upper[ii2]);
      if (vecRSInds.length() != 0 && vecRSInds[ii2] < 0)
        settings[ii2] = vecRSVals[ii2];
    }

    //**/ create discrepancy response surfaces
    faPtrs1 = new FuncApprox*[nOutputs];
    dOneSample = new double[nInputs];
    tSamInputs = new double[ExpNSamples*nInputs];
    int        askFlag = 0, *states=NULL;
    double     *tLowers = new double[nInputs];
    double     *tUppers = new double[nInputs];
    checkAllocate(tUppers, "tUppers in PMCMC::analyze");

    PsuadeData *dataPtr = new PsuadeData(); 
    psStrings  XNames;
    for (ii = 0; ii < nOutputs; ii++)
    {
      for (kk = 0; kk < dnSamples; kk++)
      {
        //**/ inject the design, calibration default and fixed 
        //**/ parameter values
        cnt = 0;
        for (ii2 = 0; ii2 < nInputs; ii2++)
        {
          if (vecDesParams.length() > 0 && vecDesParams[ii2] == 1)
          {
            dOneSample[ii2] = vecDSamInps[kk*dnInputs+cnt];
            cnt++;
          }
          else dOneSample[ii2] = settings[ii2];
        }

        //**/ at this point, dOneSample has been set to calibration default
        //**/ and design default (at sample kk) and at fixed values
        //**/ now evaluate it
        simdata = faPtrs[ii]->evaluatePoint(dOneSample);
        expdata = vecDMeans[kk*nOutputs+ii];

        //**/ generate the difference between experiment and simulation
        discOutputs[ii*dnSamples+kk] = expdata - simdata;
      }

      //**/ store discrepancy file
      if (dnInputs > 0) 
      {
        for (kk = 0; kk < ExpNSamples*dnInputs; kk++)
          tSamInputs[kk] = vecDSamInps[kk];
        XNames.setNumStrings(dnInputs);
        cnt = 0;
        for (ii2 = 0; ii2 < nInputs; ii2++)
        {
          if (vecDesParams[ii2] == 1)
          {
            tLowers[cnt] = lower[ii2];
            tUppers[cnt] = upper[ii2];
            if (qData.strArray_ == NULL)
                 snprintf(winput,100,"X%d", ii2+1);
            else strcpy(winput, qData.strArray_[ii2]);
            XNames.loadOneString(cnt, winput);
            cnt++;
          }
        }
        dataPtr->updateInputSection(ExpNSamples,dnInputs,NULL,tLowers,
                tUppers,vecDSamInps.getDVector(), XNames.getStrings(), 
                NULL,NULL,NULL,NULL);
      }
      else
      {
        XNames.setNumStrings(1);
        snprintf(winput,100,"X0");
        XNames.loadOneString(0, winput);
        //**/ set input = 0.5 (and later output = constant)
        for (ii2 = 0; ii2 < ExpNSamples; ii2++) tSamInputs[ii2] = 0.5;
        tLowers[0] = 0.0;
        tUppers[0] = 1.0;
        dataPtr->updateInputSection(ExpNSamples, iOne, NULL, tLowers, 
                        tUppers,tSamInputs,XNames.getStrings(),NULL,
                        NULL,NULL,NULL);
      }

      states = new int[ExpNSamples];
      for (kk = 0; kk < ExpNSamples; kk++) states[kk] = 1;
      XNames.setNumStrings(1);
      snprintf(winput,100,"Y%d", ii+1);
      XNames.loadOneString(0, winput);
      dataPtr->updateOutputSection(ExpNSamples,iOne,
              &discOutputs[ii*dnSamples],states,XNames.getStrings());
      delete [] states;
      dataPtr->updateMethodSection(PSUADE_SAMP_MC, ExpNSamples, 1, -1, -1);
      snprintf(charString,100,"psDiscrepancyModel%d", ii+1);
      if (mypid_ == 0) dataPtr->writePsuadeFile(charString, 0);
      delete dataPtr;

      //**/ create the discrepancy response surface for the current output
      //**/ nInputs different for 2 different modes
      if (mypid_ == 0)
        printOutTS(PL_INFO,
           "Creating discrepancy response surface for output %d\n",ii+1);
      faPtrs1[ii] = NULL;
      if (dnInputs > 0 && dnSamples > 1)
      {
        faPtrs1[ii] = genFA(dfaType,dnInputs,iOne,ExpNSamples);
        if (faPtrs1[ii] == NULL)
        {
          if (mypid_ == 0)
            printOutTS(PL_ERROR,
              "PMCMC ERROR: cannot create discrepancy func for output %d.\n",
              ii+1);
          return -1.0;
        }
      }
      if (faPtrs1[ii] != NULL)
      {
        faPtrs1[ii]->setNPtsPerDim(dnPerDim);
        faPtrs1[ii]->setBounds(lower, upper);
        faPtrs1[ii]->setOutputLevel(0);
        //**/ Feb 2015: no need to create faPtrs1, need work later
        //**/faPtrs1[ii]->initialize(tSamInputs,&discOutputs[ii*dnSamples]);
      }
      else
      {
        vecDiscConstMeans[ii] = 0.0;
        for (kk = 0; kk < ExpNSamples; kk++)
          vecDiscConstMeans[ii] += discOutputs[ii*dnSamples+kk];
        vecDiscConstMeans[ii] /= (double) ExpNSamples;
        vecDiscConstStdvs[ii] = 0.0;
        for (kk = 0; kk < ExpNSamples; kk++)
          vecDiscConstStdvs[ii] += 
                 pow(discOutputs[ii*dnSamples+kk]-
                     vecDiscConstMeans[ii],2.0);
        vecDiscConstStdvs[ii]=sqrt(vecDiscConstStdvs[ii]/ExpNSamples);
      }
    }
    delete [] ExpSamInputs;
    delete [] dOneSample;
    delete [] settings;
    delete [] tSamInputs;
    delete [] tLowers;
    delete [] tUppers;
    //**/ at this point faPtrs1 and vecDiscConst are ready
  }

  //**/ ---------------------------------------------------------------
  //    set up constraint filters, if any
  //**/ ---------------------------------------------------------------
  if (mypid_ == 0)
  {
    printEquals(PL_INFO, 0);
    printOutTS(PL_INFO,
               "PMCMC INFO: creating constraints, if there is any.\n");
    printOutTS(PL_INFO,
         "  Constraints remove infeasible regions from the priors.\n");
    printOutTS(PL_INFO,
         "  Constraints can be specified by RS constraint files.\n");
  }
  RSConstraints *constrPtr = NULL;
  if (ioPtr != NULL)
  {
    constrPtr = new RSConstraints();
    constrPtr->genConstraints(ioPtr);
  }
  printEquals(PL_INFO, 0);
  if (psConfig_.AnaExpertModeIsOn() && mypid_ == 0)
  {
    snprintf(charString,100,"How many MCMC chains? (2-20, default=3) : ");
    numChains = getInt(2,20,charString);
    snprintf(charString,100,"PSRF threshold? (1.0 - 1.2, default = 1.05) : ");
    psrfThreshold = getDouble(charString);
    if (psrfThreshold < 1.0 || psrfThreshold > 1.2)
    {
      printf("PMCMC : invalid PSRF threshold ==> reset to 1.05.\n");
      psrfThreshold = 1.05;
    }
  }
  if (mypid_ == 0) 
  {
    if (numChains < nprocs_) 
    {
      numChains = nprocs_;
      printf("PMCMC : set number of chains = %d\n", numChains);
    }
  }
  commMgr_->bcast((void *) &numChains, iOne, INT, 0);
  commMgr_->bcast((void *) &psrfThreshold, iOne, DOUBLE, 0);

  //**/ ---------------------------------------------------------------
  //    set up for MCMC iterations
  //**/ ---------------------------------------------------------------
  int    *Ivec, **bins, ****bins2, globalIts, countTrack, dcnt;
  int    mcmcFail=0, sumBins, index2, nFail;
  int    ii3, jj2, kk2, index, length, count, iChain, chainCnt, mcmcIts;
  int    maxGlobalIts=20, chainCntSave, *chainStatus;
  double *XRange=NULL, *XGuess=NULL, *XDist=NULL, *XDesignS, *YDesignS;
  double *YDesignStds=NULL, *XGuessS=NULL, *YGuessS=NULL, *YGuessStds=NULL;
  double Xtemp, Ytemp, Ytemp2, *Xmax, Ymax, *SDist;
  double **XChains=NULL, stdev, stdev2, ddata, ddata2, WStat, BStat;
  double *chainMeans=NULL, *chainStdevs=NULL, *psrfs=NULL;
  //**/ ----- for storing the input ranges
  XRange  = new double[nInputs];
  for (ii = 0; ii < nInputs; ii++) XRange[ii] = upper[ii] - lower[ii]; 
  //**/ ----- XDist is for storing the proposal distribution
  XDist   = new double[maxPts+1];
  SDist   = new double[maxPts+1];
  //**/ ----- for model evaluation
  XGuess  = new double[nInputs];
  XGuessS = new double[dnSamples*nInputs*(maxPts+1)];
  YGuessS = new double[dnSamples*nOutputs*(maxPts+1)];
  YGuessStds = new double[dnSamples*nOutputs*(maxPts+1)];
  //**/ ----- for discrepancy evaluation
  XDesignS = new double[dnSamples*nInputs*(maxPts+1)];
  YDesignS = new double[dnSamples*nOutputs*(maxPts+1)];
  YDesignStds = new double[dnSamples*nOutputs*(maxPts+1)];
  //**/ ----- for keeping the point of maximum likelihood
  Xmax = new double[nInputs];
  means_ = new double[nInputs];
  sigmas_ = new double[nInputs];
  checkAllocate(sigmas_, "sigmas_ in PMCMC::analyze");
  for (ii = 0; ii < nInputs; ii++) Xmax[ii] = means_[ii] = sigmas_[ii] = 0;
  Ymax = -PSUADE_UNDEFINED;
  //**/ ----- for randomization of order of inputs to be processed
  Ivec = new int[nInputs];
  Ivec[nInputs-1] = -1;
  //**/ ----- for saving the input values at each iteration
  XChains = new double*[numChains];
  for (ii = 0; ii < numChains; ii++)
     XChains[ii] = new double[maxGlobalIts*maxSamples*(nInputs+1)];
  chainMeans = new double[numChains];
  chainStdevs = new double[numChains];
  chainStatus  = new int[numChains];
  checkAllocate(chainStatus, "chainStatus in PMCMC::analyze");
  for (ii = 0; ii < numChains; ii++) chainMeans[ii] = chainStdevs[ii] = 0.0;
  for (ii = 0; ii < numChains; ii++) chainStatus[ii] = 0;
  psrfs = new double[nInputs];
  checkAllocate(psrfs, "psrfs in PMCMC::analyze");
  for (ii = 0; ii < nInputs; ii++) psrfs[ii] = 0.0;
  //**/ ----- for tracking the frequency of visits
  bins = new int*[nbins];
  for (ii = 0; ii < nbins; ii++)
  {
     bins[ii] = new int[nInputs];
     for (jj = 0; jj < nInputs; jj++) bins[ii][jj] = 0;
  }
  bins2 = new int***[nbins];
  for (jj = 0; jj < nbins; jj++)
  {
    bins2[jj] = new int**[nbins];
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      bins2[jj][jj2] = new int*[nInputs];
      for (ii = 0; ii < nInputs; ii++)
      {
        bins2[jj][jj2][ii] = new int[nInputs];
        for (ii2 = 0; ii2 < nInputs; ii2++)
          bins2[jj][jj2][ii][ii2] = 0;
      }
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ generate LHS or LPTAU MCMC seed points for different chains
  //**/ ---------------------------------------------------------------
  Sampling *sampler;
  if (nInputs > 50) sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
  else              sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
  sampler->setInputBounds(nInputs, lower, upper);
  sampler->setOutputParams(1);
  sampler->setSamplingParams(numChains, 1, 1);
  sampler->initialize(0);
  double *mcmcSeeds = new double[numChains*nInputs];
  double *tmpOuts = new double[numChains];
  int    *tmpStates = new int[numChains];
  checkAllocate(tmpStates, "tmpStates in PMCMC::analyze");
  sampler->getSamples(numChains,nInputs,1,mcmcSeeds,tmpOuts,tmpStates);
  delete [] tmpOuts;
  delete [] tmpStates;
  delete sampler;
  //**/ normalize the sample
  for (iChain = 0; iChain < numChains; iChain++)
  {
    for (ii = 0; ii < nInputs; ii++)
    {
      ddata = mcmcSeeds[iChain*nInputs+ii];
      ddata = (ddata - lower[ii]) / XRange[ii];
      mcmcSeeds[iChain*nInputs+ii] = ddata;
    }
  }
   
  //**/ ---------------------------------------------------------------
  // run the Gibbs algorithm
  //**/ ---------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  if (mypid_ == 0) printOutTS(PL_INFO, "PMCMC begins ... \n");
  fflush(stdout);
  fp = NULL;
  globalIts = chainCnt = 0;
  //**/ ensure different processors have different random number seed
  PSUADE_randInit(mypid_*1234567+7654321);
  while (globalIts < maxGlobalIts)
  {
    for (iChain = 0; iChain < numChains; iChain++)
    {
      if (iChain % nprocs_ != mypid_) continue;

      //**/ initially, set LPTAU/LHS initial guess for the calibration
      //**/ parameters. Otherwise, retrieve the last input values
      printOutTS(PL_INFO,"\n%d : PMCMC : chain %d, iteration = %d\n",
                 mypid_,iChain+1,globalIts+1);
      if (iChain / nprocs_ == 0) chainCntSave = chainCnt;
      else                       chainCnt  = chainCntSave;
      //**/ insert seed points initially
      if (chainCnt == 0)
      {
        for (ii = 0; ii < nInputs; ii++)
        {
          //**/ if design points, default = mid point
          if (vecDesParams.length() == 0 || vecDesParams[ii] == 0)
               XGuess[ii] = mcmcSeeds[iChain*nInputs+ii];
          else XGuess[ii] = 0.5;
          XChains[iChain][0*(nInputs+1)+ii] = XGuess[ii];
          XGuess[ii] = XGuess[ii] * (upper[ii] - lower[ii]) + lower[ii];
        }
        //**/ if fixed input, insert the fixed values
        if (vecRSInds.length() != 0)
        {
          for (ii = 0; ii < nInputs; ii++)
            if (vecRSInds[ii] < 0) XGuess[ii] = vecRSVals[ii];
        }
        //**/ this is the normalization factor
        XChains[iChain][0*(nInputs+1)+nInputs] = -1;
      }
      //**/ otherwise, fetch from the chain
      else
      {
        for (ii = 0; ii < nInputs; ii++)
        {
          ddata = XChains[iChain][(chainCnt-1)*(nInputs+1)+ii];
          XGuess[ii] = ddata * (upper[ii] - lower[ii]) + lower[ii];
        }
      }
      
      //**/ run maxSamples for the chain
      mcmcIts = countTrack = 0;
      while (mcmcIts < maxSamples)
      {
        //**/ output dots to indicate progress 
        count = (mcmcIts+1) / (maxSamples/10);
        if (count != countTrack)
        { 
          countTrack++;
          if (mypid_ == 0)
          {
            printOutTS(PL_INFO, "%3.0f%% ",10.0*countTrack );
            fflush(stdout);
          }
        }
        //**/ cycle through all selected inputs in random order
        //**/ prevent sampling same input in consecutive steps
        jj = Ivec[nInputs-1];
        generateRandomIvector(nInputs, Ivec);
        if (Ivec[0] == jj && nInputs > 1)
        {
          Ivec[0] = Ivec[nInputs-1];
          Ivec[nInputs-1] = jj;
        }
        for (kk = 0; kk < nInputs; kk++)
        {
          ii = Ivec[kk];
          //**/ only selected inputs will be walking
          if ((vecRSInds.length() == 0 ||
              (vecRSInds.length() > 0 && vecRSInds[ii] >=0)) && 
              (vecDesParams.length() == 0 || vecDesParams[ii] == 0))
          {
            //**/ store away current value of parameter ii
            Xtemp = XGuess[ii];
 
            //**/ creating a CDF for the current input given the others
            //**/ (fix all inputs, scan input ii to build distribution)
            //**/ CDF will be put into XDist[0:maxPts]
            //**/ Case 1 : 1 output
            if (nOutputs == 1)
            {
              cnt = 0;
              for (jj = 0; jj <= maxPts; jj++)
              {
                //**/ marching ii-th input upward from lower to upper
                XGuess[ii] = lower[ii]+jj*XRange[ii]/maxPts;
                    
                //**/ create likelihood function
                cnt++;
                if (cnt >= freq) cnt = 0; 
                index = jj * dnSamples;
                //**/ create XGuessS and XDesignS
                for (kk2 = 0; kk2 < dnSamples; kk2++) 
                {
                  dcnt = 0;
                  for (ii2 = 0; ii2 < nInputs; ii2++) 
                  {
                    XGuessS[(index+kk2)*nInputs+ii2] = XGuess[ii2];
                    if (vecDesParams.length() > 0 && vecDesParams[ii2] == 1)
                    {
                      XGuessS[(index+kk2)*nInputs+ii2] = 
                             vecDSamInps[kk2*dnInputs+dcnt];
                      XDesignS[(index+kk2)*dnInputs+dcnt] = 
                             vecDSamInps[kk2*dnInputs+dcnt];
                      if (XGuessS[(index+kk2)*nInputs+ii2]<lower[ii2] ||
                          XGuessS[(index+kk2)*nInputs+ii2] > upper[ii2])
                      {
                        printOutTS(PL_INFO,
                             "WARNING: design parameter value ");
                        printOutTS(PL_INFO,
                             "out of bound.\n");
                        printOutTS(PL_INFO,"  Input = %d\n", ii2+1);
                        printOutTS(PL_INFO,
                             "  Bounds = [%16.8e, %16.8e]\n", 
                             lower[ii2],upper[ii2]);
                        printOutTS(PL_INFO,"   Design value = %e\n", 
                                   XGuessS[(index+kk2)*nInputs+ii2]);
                        printOutTS(PL_INFO,
                             "   Design sample = %d\n", kk2+1);
                      }
                      dcnt++;
                    }
                  }
                }
                //**/ set design outputs and stds (discrepancy) to zero 
                //**/ since they will be used but may not be set later
                for (kk2 = 0; kk2 < dnSamples; kk2++) 
                {
                  YDesignS[index+kk2] = YDesignStds[index+kk2] = 0.0;
                  YGuessS[index+kk2] = YGuessStds[index+kk2] = 0.0;
                }
              }

              //**/ run XGuessS and XDesignS through response surfaces
              //**/ case 1 : if response surface error information 
              //**/          should be used
              if (rsErrFlag == 1)
              {
                faPtrs[0]->evaluatePointFuzzy((maxPts+1)*dnSamples,
                                          XGuessS,YGuessS,YGuessStds);
                //**/ add discrepancy function, if any
                if (faPtrs1 != NULL && faPtrs1[0] != NULL)
                {
                  //**/ Feb 2015: do not need this
                  //**/faPtrs1[0]->evaluatePointFuzzy((maxPts+1)*dnSamples,
                  //**/                    XDesignS,YDesignS,YDesignStds);
                  for (ii3 = 0; ii3 <= maxPts; ii3++)
                  {
                    for (kk2 = 0; kk2 < dnSamples; kk2++)
                    {
                      index = ii3 * dnSamples + kk2;
                      YDesignS[index] = discOutputs[kk2];
                    }
                  }
                }
                else if (vecDiscConstMeans.length() > 0 &&
                         vecDiscConstMeans[0] != PSUADE_UNDEFINED)
                {
                  for (kk2 = 0; kk2 < (maxPts+1)*dnSamples; kk2++) 
                    YDesignS[kk2] = vecDiscConstMeans[0];
                }
              }
              //**/ case 2 : if response surface error information 
              //**/          not to be used
              else
              {
                faPtrs[0]->evaluatePoint((maxPts+1)*dnSamples,
                                         XGuessS,YGuessS);
                //**/ add discrepancy function, if any
                if (faPtrs1 != NULL && faPtrs1[0] != NULL)
                {
                  //**/ Feb 2015: do not need this
                  //**/faPtrs1[0]->evaluatePoint((maxPts+1)*dnSamples,
                  //**/                    XDesignS,YDesignS);
                  for (ii3 = 0; ii3 <= maxPts; ii3++)
                  {
                    for (kk2 = 0; kk2 < dnSamples; kk2++)
                    {
                      index = ii3 * dnSamples + kk2;
                      YDesignS[index] = discOutputs[kk2];
                    }
                  }
                }
                else if (vecDiscConstMeans.length() > 0 &&
                         vecDiscConstMeans[0] != PSUADE_UNDEFINED)
                {
                  for (kk2 = 0; kk2 < (maxPts+1)*dnSamples; kk2++) 
                    YDesignS[kk2] = vecDiscConstMeans[0];
                }
              }

              //**/ compute distributions XDist
              for (jj = 0; jj <= maxPts; jj++)
              {
                index = jj * dnSamples;
                for (kk2 = 0; kk2 < dnSamples; kk2++) 
                  YGuessS[index+kk2] += YDesignS[index+kk2];

                XDist[jj] = 0.0;
                for (kk2 = 0; kk2 < dnSamples; kk2++)
                {
                  Ytemp = YGuessS[index+kk2];
                  stdev = YGuessStds[index+kk2];
                  stdev2 = YDesignStds[index+kk2];
                  Ytemp2 = pow((Ytemp-vecDMeans[kk2]),2.0) /
                               (pow(vecDStdvs[kk2],2.0)+
                                stdev*stdev+stdev2*stdev2);
                  XDist[jj] += Ytemp2;
                }
                XDist[jj] = XDist[jj] / dnSamples;
              }
            }
            //**/ Case 2 : multiple outputs
            else
            {
              //**/ prepare XGuessS and XDesignS, and initialize 
              //**/ YGuessS and YDesignS
              for (jj = 0; jj <= maxPts; jj++)
              {
                //**/ marching ii-th input upward from lower to upper
                XGuess[ii] = lower[ii]+jj*XRange[ii]/maxPts;
                //**/ create likelihood function
                //**/ implementation 
                //**/ set up XGuessS and XDesignS
                index = jj * dnSamples;
                for (kk2 = 0; kk2 < dnSamples; kk2++) 
                {
                  dcnt = 0;
                  for (ii2 = 0; ii2 < nInputs; ii2++) 
                  {
                    XGuessS[(index+kk2)*nInputs+ii2] = XGuess[ii2];
                    if (vecDesParams.length() > 0 && vecDesParams[ii2] == 1)
                    {
                      XGuessS[(index+kk2)*nInputs+ii2] = 
                                     vecDSamInps[kk2*dnInputs+dcnt];
                      XDesignS[(index+kk2)*dnInputs+dcnt] = 
                                     vecDSamInps[kk2*dnInputs+dcnt];
                      if (XGuessS[(index+kk2)*nInputs+ii2]<lower[ii2] ||
                          XGuessS[(index+kk2)*nInputs+ii2]>upper[ii2])
                      {
                        printOutTS(PL_INFO,
                            "WARNING: design parameter value out ");
                        printOutTS(PL_INFO,"of bound.\n");
                        printOutTS(PL_INFO,"   Input = %d\n", ii2+1);
                        printOutTS(PL_INFO,
                            "   Bounds = [%24.16e, %24.16e].\n", 
                            lower[ii2],upper[ii2]);
                        printOutTS(PL_INFO,"   Design value = %e\n", 
                            XGuessS[(index+kk2)*nInputs+ii2]);
                        printOutTS(PL_INFO,
                            "   Design sample = %d\n", kk2+1);
                      }
                      dcnt++;
                    }
                  }
                }
                //**/ set design outputs and stds (discrepancy) to zero 
                //**/ since they will be used but may not be set later
                for (ii2 = 0; ii2 < dnSamples*nOutputs; ii2++) 
                {
                  YDesignS[index*nOutputs+ii2] = 
                              YDesignStds[index*nOutputs+ii2] = 0.0;
                  YGuessS[index*nOutputs+ii2] = 
                              YGuessStds[index*nOutputs+ii2] = 0.0;
                }
              }
              //**/ run XGuessS and XDesignS through response surfaces
              for (ii2 = 0; ii2 < nOutputs; ii2++) 
              {
                //**/ case 1: if RS error is requested
                if (rsErrFlag == 1)
                {
                  faPtrs[ii2]->evaluatePointFuzzy((maxPts+1)*dnSamples,
                           XGuessS,&YGuessS[ii2*dnSamples*(maxPts+1)],
                           &YGuessStds[ii2*dnSamples*(maxPts+1)]);
                  //**/ add discrepancy function, if available
                  if (faPtrs1 != NULL && faPtrs1[ii2] != NULL)
                  {
                    //**/ Feb 2015: do not need this
                    //**/faPtrs1[ii2]->evaluatePointFuzzy((maxPts+1)*dnSamples,
                    //**/             XDesignS,
                    //**/             &YDesignS[ii2*dnSamples*(maxPts+1)],
                    //**/             &YDesignStds[ii2*dnSamples*(maxPts+1)]);
                    for (ii3 = 0; ii3 <= maxPts; ii3++)
                    {
                      for (kk2 = 0; kk2 < dnSamples; kk2++)
                      {
                        index=ii2*dnSamples*(maxPts+1)+ii3*dnSamples+kk2;
                        YDesignS[index] = discOutputs[ii2*dnSamples+kk2];
                      }
                    }
                  }
                  else if (vecDiscConstMeans.length() > 0 &&
                           vecDiscConstMeans[ii2] != PSUADE_UNDEFINED)
                  {
                    for (kk2 = 0; kk2 < dnSamples*(maxPts+1); kk2++)
                    {
                      YDesignS[ii2*dnSamples*(maxPts+1)+kk2] = 
                                        vecDiscConstMeans[ii2];
                      YDesignStds[ii2*dnSamples*(maxPts+1)+kk2] = 0.0;
                    }
                  }
                }
                //**/ case 2: if RS error is to be turned off
                else
                {
                  faPtrs[ii2]->evaluatePoint((maxPts+1)*dnSamples,XGuessS,
                                     &YGuessS[ii2*dnSamples*(maxPts+1)]);
                  //**/ add discrepancy function, if available
                  if (faPtrs1 != NULL && faPtrs1[ii2] != NULL)
                  {
                    //**/ Feb 2015: do not need this
                    //**/faPtrs1[ii2]->evaluatePoint(dnSamples*(maxPts+1),
                    //**/                XDesignS,
                    //**/                &YDesignS[ii2*dnSamples*(maxPts+1)]);
                    for (ii3 = 0; ii3 <= maxPts; ii3++)
                    {
                      for (kk2 = 0; kk2 < dnSamples; kk2++)
                      {
                        index=ii2*dnSamples*(maxPts+1)+ii3*dnSamples+kk2;
                        YDesignS[index] = discOutputs[ii2*dnSamples+kk2];
                      }
                    }
                  }
                  else if (vecDiscConstMeans.length() > 0 &&
                           vecDiscConstMeans[ii2] != PSUADE_UNDEFINED)
                  {
                    for (kk2 = 0; kk2 < dnSamples*(maxPts+1); kk2++)
                      YDesignS[ii2*dnSamples*(maxPts+1)+kk2] = 
                                       vecDiscConstMeans[ii2];
                  }
                  for (kk2 = 0; kk2 < dnSamples*(maxPts+1); kk2++)
                    YGuessStds[ii2*dnSamples*(maxPts+1)+kk2] = 
                        YDesignStds[ii2*dnSamples*(maxPts+1)+kk2] = 0.0;
                }
              }
              //**/ compute XDist
              for (jj = 0; jj <= maxPts; jj++)
              {
                XDist[jj] = 0.0;
                index = jj * dnSamples;
                for (ii2 = 0; ii2 < nOutputs; ii2++) 
                {
                  for (kk2 = 0; kk2 < dnSamples; kk2++)
                  {
                    Ytemp = YGuessS[ii2*dnSamples*(maxPts+1)+index+kk2]+ 
                            YDesignS[ii2*dnSamples*(maxPts+1)+index+kk2];
                    stdev = YGuessStds[ii2*dnSamples*(maxPts+1)+index+kk2];
                    stdev2 = YDesignStds[ii2*dnSamples*(maxPts+1)+index+kk2];
                    Ytemp2=pow((Ytemp-vecDMeans[kk2*nOutputs+ii2]),2.0)/
                              (pow(vecDStdvs[kk2*nOutputs+ii2],2.0) + 
                               stdev*stdev + stdev2*stdev2);
                    XDist[jj] += Ytemp2;
                  }
                }
                XDist[jj] = XDist[jj] / (dnSamples*nOutputs);
              }
            }

            //**/ at this point XDist[jj] has been computed
            //**/ apply filters to the likelihood function XDist[jj]
            //**/ and then compute prior distribution probability 
            //**/ in SDist[jj]
            nFail = 0;
            for (jj = 0; jj <= maxPts; jj++)
            {
              XGuess[ii] = lower[ii]+jj*XRange[ii]/maxPts;
              status = 1;
              if (constrPtr != NULL)
                Ytemp = constrPtr->evaluate(XGuess,XDist[jj],status);
              if (status == 0)
              {
                XDist[jj] = 0.0;
                nFail++;
              }

              //**/ compute prior SDist
              ddata = 1.0;
              if (inputPDFs != NULL)
              {
                for (ii2 = 0; ii2 < nInputs; ii2++)
                {
                  if ((vecDesParams.length() == 0 || 
                       vecDesParams[ii2] == 0) && 
                       inputPDFs[ii2] != NULL &&
                      (vecRSInds.length() == 0 || vecRSInds[ii2] >= 0))
                  {
                    inputPDFs[ii2]->getPDF(iOne,&XGuess[ii2],&ddata2);
                    ddata *= ddata2;
                  }
                }
              }
              SDist[jj] = ddata; 
            }

            //**/ now XDist and SDist are ready
            //**/ find the max exp(-0.5*XDist*SDist) for normalization
            ddata = XDist[0];
            for (jj = 1; jj <= maxPts; jj++) 
            {
              ddata2 = XDist[jj];
              if (ddata2 < ddata) ddata = ddata2;
            }
            //**/ store away the normalization factor for weighting later
            XChains[iChain][chainCnt*(nInputs+1)+nInputs] = ddata;
            for (jj = 0; jj <= maxPts; jj++)
            {
              //**/ scale distributions for stability
              XDist[jj] = SDist[jj] * exp(-0.5 * (XDist[jj] - ddata));

              //**/ store away the optimal point in any chain
              ddata2 = XDist[jj] * exp(-0.5*ddata);
              if (ddata2 > Ymax)
              {
                Ymax = ddata2;
                for (ii2 = 0; ii2 < nInputs; ii2++) 
                  Xmax[ii2] = XGuess[ii2];
                Xmax[ii] = lower[ii] + (upper[ii]-lower[ii]) * jj / maxPts;
              }
              if (jj > 0) XDist[jj] += XDist[jj-1];
            }

            //**/ if the distribution is nonzero, select a point for X[ii]
            if (XDist[maxPts] - XDist[0] > 0.0e-24)
            {
              //**/ normalize XDist to [0,1] to make it a CDF
              for (jj = 1; jj <= maxPts; jj++)
                XDist[jj] = (XDist[jj]-XDist[0])/(XDist[maxPts]-XDist[0]);
              XDist[0] = 0;
              XGuess[ii] = PSUADE_drand();
              index = binarySearchDble(XGuess[ii], XDist, maxPts+1);
              //if (index == maxPts) index = maxPts - 1;
              if (index >= 0) ddata = (double) index;
              else
              {
                index = - index - 1;
                if (PABS(XDist[index]-XDist[index+1]) > 1.0e-16)
                  ddata=index + (XGuess[ii]-XDist[index])/
                                (XDist[index+1]-XDist[index]);
                else ddata = (double) index;
              }
              //**/ create the next guess values
              XGuess[ii] = lower[ii]+ddata*XRange[ii]/maxPts;
            }
            //**/ if the distribution is zero, do not move
            else 
            {
              XGuess[ii] = Xtemp;
              if (nFail == maxPts+1) 
              {
                printOutTS(PL_ERROR,
                  "ERROR: Constraints have resulted in zero distribution\n");
                exit(1);
              }
            }

            //**/ update statistics and histograms, store all points in XSave
            //**/ for computing overall statistics (leave the first 
            //**/ maxSamples/2 points as burn-in
            if (mcmcIts >= maxSamples/2 || globalIts > 0)
            {
              //**/ update the 1D histogram
              for (ii2 = 0; ii2 < nInputs; ii2++) 
              {
                ddata = (XGuess[ii2] - lower[ii2]) / XRange[ii2];
                XChains[iChain][chainCnt*(nInputs+1)+ii2] = ddata;
                index = (int) (ddata * nbins);
                if (index >= nbins) index = nbins - 1;
                bins[index][ii2]++;
              }

              //**/ update the 2D histogram
              for (ii2 = 0; ii2 < nInputs; ii2++) 
              {
                ddata = (XGuess[ii2] - lower[ii2]) / XRange[ii2];
                index = (int) (ddata * nbins);
                if (index >= nbins) index = nbins - 1;
                for (ii3 = 0; ii3 < nInputs; ii3++) 
                {
                  ddata2 = (XGuess[ii3] - lower[ii3]) / XRange[ii3];
                  index2 = (int) (ddata2 * nbins);
                  if (index2 >= nbins) index2 = nbins - 1;
                  bins2[index][index2][ii2][ii3]++;
                }
              }
              chainCnt++;
            }
            mcmcIts++;
          }
          fp = fopen("psuade_stop", "r");
          if (fp != NULL)
          {
            if (mypid_ == 0)
              printOutTS(PL_INFO,
                 "MCMC INFO: psuade_stop FILE FOUND - TERMINATE MCMC.\n");
            fclose(fp);
            return 0.0;
          }
          if (mcmcIts >= maxSamples) break;
        }
        //**/ scanning through all inputs completed
      }
      if (mypid_ == 0)
      {
        if (countTrack <= 10) printOutTS(PL_INFO,"100%%\n");
        else                  printOutTS(PL_INFO,"\n");
        //**/ scanning through all maxSamples for the current chain 
      }
    }

    //**/ now compute convergence statistics
    //Communicate the chain information
    commFlag = 1;
    commMgr_->bcast((void *) &commFlag, iOne, INT, 0);
    for (iChain = 0; iChain < numChains; iChain++)
    {
      if (mypid_ == 0 && iChain % nprocs_ != mypid_)
      {
        proc = iChain % nprocs_;
        kk = chainCnt * (nInputs + 1);
        commMgr_->recv((void *) XChains[iChain],kk,DOUBLE,iChain,proc);
      }
      if (mypid_ != 0 && iChain % nprocs_ == mypid_)
      {
        kk = chainCnt * (nInputs + 1);
        commMgr_->send((void *) XChains[iChain],kk,DOUBLE,iChain,0);
      }
    }

    //**/ now compute convergence statistics
    globalIts++;
    if (mypid_ == 0)
    {
      printOutTS(PL_INFO, "\nIteration %d summary: \n", globalIts);
      mcmcFail = nInputs - dnInputs;
      if (vecRSInds.length() > 0)
      {
        for (ii = 0; ii < nInputs; ii++)
          if (vecRSInds[ii] < 0) mcmcFail--; 
      }
      for (ii = 0; ii < nInputs; ii++)
      {
        if ((vecRSInds.length() == 0 || (vecRSInds.length() > 0 && 
             vecRSInds[ii] >=0)) && 
            (vecDesParams.length() == 0 || vecDesParams[ii] == 0))
        {
          if (printLevel > 2) printOutTS(PL_INFO, "Input = %d\n", ii+1);
       
          //**/ check that if a chain does not walk at all and turn it off
          for (iChain = 0; iChain < numChains; iChain++)
          {
            ddata = 0.0;
            for (jj = 0; jj < chainCnt; jj++) 
              ddata += XChains[iChain][jj*(nInputs+1)+ii];
            ddata /= chainCnt;
            ddata2 = 0.0;
            for (jj = 0; jj < chainCnt; jj++) 
              ddata2 += pow(XChains[iChain][jj*(nInputs+1)+ii]-ddata,2.0);
            ddata2 /= (double) (chainCnt - 1);
            chainMeans[iChain] = ddata;
            chainStdevs[iChain] = ddata2;
            if (globalIts > 2 && chainStdevs[iChain] < 1.0e-20) 
            {
              printf("PMCMC INFO: chain %d disabled.\n",iChain+1);
              chainStatus[iChain] = 1;
            }
          }
          nChainGood = 0;
          for (iChain = 0; iChain < numChains; iChain++)
          {
            if (chainStatus[iChain] == 0) nChainGood++;
          }
          if (nChainGood <= 1)
          {
            printf("PMCMC ERROR: too few chains <= 1.\n");
            printf("Suggestion: you may want to relax on the experimental\n");
            printf("    data uncertainties (make them larger).\n");
            printf("    To see if this is the problem, turn on printlevel\n");
            printf("    to 3 and run again. If the variance of the chains\n");
            printf("    are small, small data uncertainties is probably \n");
            printf("    the problem.\n");
            exit(1);
          }
          //**/ compute PSRF
          WStat = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
          {
            if (chainStatus[iChain] == 0) WStat += chainStdevs[iChain];
          }
          WStat /= (double) nChainGood;
          if (WStat < 0) WStat = PSUADE_UNDEFINED;
          if (printLevel > 2) 
            printf("  Within  chain variance W = %e\n", WStat);
          ddata = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
          {
            if (chainStatus[iChain] == 0)
              ddata += chainMeans[iChain];
          }
          ddata /= (double) nChainGood;
          BStat = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
          {
            if (chainStatus[iChain] == 0)
              BStat += pow(chainMeans[iChain]-ddata,2.0);
          }
          BStat = BStat / (nChainGood - 1.0) * chainCnt;
          if (printLevel > 2) 
            printf("  Between chain variance B = %e\n", BStat/chainCnt);
          ddata = (1 - 1.0/chainCnt) * WStat + BStat / chainCnt;
          ddata = ddata / WStat * (numChains + 1) / numChains - 
                    (chainCnt - 1.0) / (double) (chainCnt * numChains); 
          if (ddata < 0) ddata2 = PSUADE_UNDEFINED;
          else           ddata2 = sqrt(ddata);
          if (printLevel > 2)
          {
            for (iChain = 0; iChain < numChains; iChain++)
              printOutTS(PL_INFO,"  Chain %4d statistics = %16.8e %16.8e\n",
                   iChain+1, chainMeans[iChain]*XRange[ii]+lower[ii],
                   chainStdevs[iChain]*XRange[ii]*XRange[ii]);
            printf("  Chain length             = %d\n", chainCnt);
            printf("  Weighted average of B, W = %e\n", ddata);
          }
          printf("  Input %d PSRF = %e\n", ii+1, ddata2);
          psrfs[ii] = ddata2;
          if (ddata2 < psrfThreshold)
          {
            printOutTS(PL_INFO,"MCMC INFO : PSRF < %e ==> converged.\n",
                       psrfThreshold);
            mcmcFail--;
          }
          //**/ update 
          ddata = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
          {
            if (chainStatus[iChain] == 0)
              for (jj = 0; jj < chainCnt; jj++)
                ddata += XChains[iChain][jj*(nInputs+1)+ii];
          }
          ddata /= (double) (nChainGood * chainCnt);
          means_[ii] = ddata;
          ddata2 = 0.0;
          for (iChain = 0; iChain < numChains; iChain++)
          {
            if (chainStatus[iChain] == 0)
              for (jj = 0; jj < chainCnt; jj++)
                ddata2 += pow(XChains[iChain][jj*(nInputs+1)+ii]-ddata,2.0);
          }
          ddata2 /= (double) (chainCnt*nChainGood-1);
          sigmas_[ii] = sqrt(ddata2);
        }
      }

      //**/ output statistics
      for (ii = 0; ii < nInputs; ii++) 
      {
        if ((vecRSInds.length() == 0 || (vecRSInds.length() > 0 && 
             vecRSInds[ii] >=0)) && 
            (vecDesParams.length() == 0 || vecDesParams[ii] == 0))
        {
          printOutTS(PL_INFO,
               "PMCMC: input %3d value at peak of likelihood = %e\n",
               ii+1, Xmax[ii]);
          ddata = means_[ii]*(upper[ii]-lower[ii])+lower[ii];
          printOutTS(PL_INFO,"MCMC: input %3d mean    = %e\n", ii+1, ddata);
          ddata = sigmas_[ii]*(upper[ii]-lower[ii]);
          printOutTS(PL_INFO,"MCMC: input %3d std dev = %e\n", ii+1, ddata);
        }
      }
    }   
    commMgr_->bcast((void *) &mcmcFail, iOne, INT, 0);
    if (mcmcFail == 0) break;

    //**/ generate matlabmcmc2.m file at every major iteration
    if (mypid_ == 0)
      genMatlabFile(nInputs,lower,upper,XRange,nPlots,
          vecPlotInds.getIVector(),nbins,bins,bins2,qData,
          numChains,chainCnt,XChains,chainStatus);
 
    //**/ these are features for better run time diagnostics
    fp = fopen("psuade_stop", "r");
    if (fp != NULL)
    {
      if (mypid_ == 0)
        printOutTS(PL_INFO,
             "MCMC INFO: psuade_stop FILE FOUND - TERMINATE MCMC.\n");
      fclose(fp);
      fp = NULL;
      strcpy(charString, "psuade_stop");
      unlink(charString);
      break;
    }
  }
  if (globalIts >= maxGlobalIts)
  {
    commMgr_->bcast((void *) psrfs, nInputs, DOUBLE, 0);
    mcmcFail = 0;
    for (ii = 0; ii < nInputs; ii++) 
    {
      if ((vecRSInds.length() == 0 || (vecRSInds.length() > 0 && 
           vecRSInds[ii] >=0)) && 
          (vecDesParams.length() == 0 || vecDesParams[ii] == 0))
        if (psrfs[ii] > psrfThreshold) mcmcFail = 1;
    }
    if (mcmcFail == 1 && mypid_ == 0) 
      printOutTS(PL_INFO,
           "MCMC maximum iterations exceeded but no convergence.\n");
  }
  else if (mypid_ == 0) printOutTS(PL_INFO, "MCMC iterations completed\n");

  //**/ ---------------------------------------------------------------
  //**/ create binning
  //**/ ---------------------------------------------------------------
  if (mypid_ == 0)
  {
    for (ii = 0; ii < nInputs; ii++) 
      for (jj = 0; jj < nbins; jj++) bins[jj][ii] = 0;
    for (ii = 0; ii < nInputs; ii++) 
      for (ii2 = 0; ii2 < nInputs; ii2++) 
        for (jj = 0; jj < nbins; jj++)
          for (jj2 = 0; jj2 < nbins; jj2++)
            bins2[jj][jj2][ii][ii2] = 0;
    for (iChain = 0; iChain < numChains; iChain++) 
    { 
      if (chainStatus[iChain] == 0)
      {
        for (jj = 0; jj < chainCnt; jj++) 
        {
          for (ii2 = 0; ii2 < nInputs; ii2++) 
          {
            ddata = XChains[iChain][jj*(nInputs+1)+ii2];
            index = (int) (ddata * nbins);
            if (index >= nbins) index = nbins - 1;
            bins[index][ii2]++;
          }
          for (ii2 = 0; ii2 < nInputs; ii2++) 
          {
            ddata = XChains[iChain][jj*(nInputs+1)+ii2];
            index = (int) (ddata * nbins);
            if (index >= nbins) index = nbins - 1;
            for (ii3 = 0; ii3 < nInputs; ii3++) 
            {
              ddata2 = XChains[iChain][jj*(nInputs+1)+ii3];
              index2 = (int) (ddata2 * nbins);
              if (index2 >= nbins) index2 = nbins - 1;
              bins2[index][index2][ii2][ii3]++;
            }
          }
        }
      }
    }
 
    //**/ ------------------------------------------------------------
    //**/ generate matlab posterior files 
    //**/ ------------------------------------------------------------
    genMatlabFile(nInputs,lower,upper,XRange,nPlots,
        vecPlotInds.getIVector(),nbins,bins,bins2,qData,numChains,
        chainCnt,XChains,chainStatus);

    //**/ ---------------------------------------------------------------
    //**/  generate the log likelihood distribution 
    //**/ ---------------------------------------------------------------
    if (genPosteriors == 1)
    {
      cnt = nChainGood * chainCnt;
      if (cnt > 200000) cnt = 200000;
      cnt /= nChainGood;
      genPostLikelihood(nInputs, lower, upper, XRange, numChains, chainCnt,
                 XChains, chainStatus, cnt, vecRSInds.getIVector(), 
                 vecRSVals.getDVector(), vecDesParams.getIVector(), 
                 dnInputs, dnSamples, vecDSamInps.getDVector(), faPtrs,
                 faPtrs1, nOutputs, discOutputs, 
                 vecDiscConstMeans.getDVector(),
                 vecDMeans.getDVector(), vecDStdvs.getDVector());
    }

    //**/ ------------------------------------------------------------
    //**/  generate the posterior sample file
    //**/ ------------------------------------------------------------
    if (genPosteriors == 1)
    {
      fp = fopen("MCMCPostSample", "w");
      if (fp != NULL)
      {
        fprintf(fp, "PSUADE_BEGIN\n");
        cnt = nChainGood * chainCnt;
        if (cnt > 200000) cnt = 200000;
        cnt /= nChainGood;
        fprintf(fp, "%d %d\n", cnt*nChainGood,nInputs);
        if (qData.strArray_ != NULL)
        {
          fprintf(fp, "# ");
          for (jj = 0; jj < nInputs; jj++)
            fprintf(fp,"%s ", qData.strArray_[jj]);
          fprintf(fp, "\n");
        }
        ii2 = 0;
        for (iChain = 0; iChain < numChains; iChain++)
        { 
          if (chainStatus[iChain] == 0)
          {
            for (ii = chainCnt-cnt; ii < chainCnt; ii++)
            {
              fprintf(fp, "%d ", ii2+1);
              for (jj = 0; jj < nInputs; jj++)
              {
                if ((vecRSInds.length() == 0 || vecRSInds[jj] >= 0) &&
                    (vecDesParams.length() == 0 || vecDesParams[jj] == 0)) 
                {
                  ddata = XChains[iChain][ii*(nInputs+1)+jj] * 
                          XRange[jj] + lower[jj];
                  fprintf(fp, "%e ", ddata);
                }
                else if (vecRSInds.length() > 0 && vecRSInds[jj] == 0)
                  fprintf(fp, "%e ", vecRSVals[jj]);
                else if (vecDesParams.length() > 0 && vecDesParams[jj] != 0) 
                  fprintf(fp, "%e ", 0.5 * (upper[jj] + lower[jj]));
              }
              fprintf(fp, "\n");
              ii2++;
            }
          }
        }
        fprintf(fp, "PSUADE_END\n");
        fprintf(fp, "#N=%d;\n",nChainGood*cnt);
        fprintf(fp, "#m=%d;\n",cnt);
        for (iChain = 0; iChain < numChains; iChain++)
        {
          if (chainStatus[iChain] == 0)
            fprintf(fp, "#A%d = A(%d*m+1:%d*m,:);\n",iChain,
                    iChain,iChain+1);
        }
        fprintf(fp, "#for ii = 2 : %d\n", nInputs+1);
        for (iChain = 0; iChain < numChains; iChain++)
        {
          if (chainStatus[iChain] == 0)
          {
            fprintf(fp, "#subplot(*,*,%d)\n",iChain+1);
            fprintf(fp, "#hist(A%d(:,ii))\n",iChain+1);
          }
          fprintf(fp, "#ii-1\n");
          fprintf(fp, "#pause;\n");
        }
        fprintf(fp, "#end;\n");
        fclose(fp);
      }
      printOutTS(PL_INFO,
          "MCMC: 'MCMCPostSample' file has a posterior sample.\n");
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ create discrepancy function, if desired ==> faPtrs1
  //**/ ---------------------------------------------------------------
  if (mypid_ == 0)
  {
    int    nInps, nOuts, nSams, *states;
    double *allOuts;
    psStrings YNames;
    PsuadeData *filePtr1, *filePtr2;
    if (modelFormFlag == 1)
    {
      snprintf(charString,100,"psDiscrepancyModel1");
      filePtr1 = new PsuadeData();
      status = filePtr1->readPsuadeFile(charString);
      if (status != 0)
      {
        printf("PMCMC ERROR: cannot read file %s in PSUADE format.\n",
               charString);
        exit(1);
      } 
    }
    if (modelFormFlag == 1 && status == 0)
    {
      filePtr1->getParameter("input_ninputs", pPtr);
      nInps = pPtr.intData_;
      filePtr1->getParameter("output_noutputs", pPtr);
      nOuts = pPtr.intData_;
      filePtr1->getParameter("method_nsamples", pPtr);
      nSams = pPtr.intData_;
      filePtr1->getParameter("output_sample", pOutputs);
      unlink(charString);
      allOuts = new double[nOutputs * nSams];
      for (jj = 0; jj < nSams; jj++) 
         allOuts[jj*nOutputs] = pOutputs.dbleArray_[jj];
      pOutputs.clean();

      YNames.setNumStrings(nOutputs);
      snprintf(winput,100,"Y1");
      YNames.loadOneString(0, winput);

      for (ii = 1; ii < nOutputs; ii++)
      {
        filePtr2 = new PsuadeData();
        snprintf(charString,100,"psDiscrepancyModel%d", ii+1);
        status = filePtr2->readPsuadeFile(charString);
        if (status != 0) break;
        filePtr2->getParameter("input_ninputs", pPtr);
        if (pPtr.intData_ != nInps) break;
        filePtr2->getParameter("output_noutputs", pPtr);
        if (pPtr.intData_ != nOuts) break;
        filePtr2->getParameter("method_nsamples", pPtr);
        if (pPtr.intData_ != nSams) break;
        filePtr2->getParameter("output_sample", pOutputs);
        delete filePtr2;
        unlink(charString);
        for (jj = 0; jj < nSams; jj++) 
          allOuts[jj*nOutputs+ii] = pOutputs.dbleArray_[jj];
        pOutputs.clean();
        snprintf(winput,100,"Y%d", ii+1);
        YNames.loadOneString(ii, winput);
      }
      if (nOutputs == 1)
      { 
        snprintf(charString,100,"psDiscrepancyModel");
        filePtr1->writePsuadeFile(charString, 0);
      }
      else if (ii == nOutputs)
      {
        states = new int[nSams];
        for (jj = 0; jj < nSams; jj++) states[jj] = 1;
        filePtr1->updateOutputSection(nSams,nOutputs,allOuts,states,
                                      YNames.getStrings());
        snprintf(charString,100,"psDiscrepancyModel");
        filePtr1->writePsuadeFile(charString, 0);
        printOutTS(PL_INFO,
          "MCMC INFO: a sample (inputs/outputs) the discrepancy model\n");
        printOutTS(PL_INFO,"           is now in psDiscrepancyModel.\n");
        delete [] states;
      }
      else
      {
        printOutTS(PL_INFO,
          "MCMC INFO: unsuccessful creation of discrepancy sample file\n");
      }
      delete filePtr1;
      delete [] allOuts;
    }
  }
 
  //**/ ---------------------------------------------------------------
  // clean up
  //**/ ---------------------------------------------------------------
  for (ii = 0; ii < numChains; ii++) delete [] XChains[ii];
  delete [] XChains;
  if (inputPDFs != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      if (inputPDFs[ii] != NULL) delete inputPDFs[ii];
    delete [] inputPDFs;
  }
  delete [] Xmax;
  delete [] XDist;
  delete [] SDist;
  delete [] XGuess;
  delete [] XGuessS;
  delete [] YGuessS;
  delete [] YGuessStds;
  delete [] XDesignS;
  delete [] YDesignS;
  delete [] YDesignStds;
  delete [] XRange;
  delete [] psrfs;
  if (discOutputs != NULL) delete [] discOutputs;
  delete [] mcmcSeeds;
  if (faPtrs != NULL)
  {
    for (ii = 0; ii < nOutputs; ii++) 
      if (faPtrs[ii] != NULL) delete faPtrs[ii];
    delete [] faPtrs;
  }
  if (faPtrs1 != NULL)
  {
    for (ii = 0; ii < nOutputs; ii++) 
      if (faPtrs1[ii] != NULL) delete faPtrs1[ii];
    delete [] faPtrs1;
  }
  for (ii = 0; ii < nbins; ii++) delete [] bins[ii];
  delete [] bins;
  for (jj = 0; jj < nbins; jj++)
  {
    for (jj2 = 0; jj2 < nbins; jj2++)
    {
      for (ii = 0; ii < nInputs; ii++) delete [] bins2[jj][jj2][ii];
      delete [] bins2[jj][jj2];
    }
    delete [] bins2[jj];
  }
  delete [] bins2;
  if (constrPtr != NULL) delete constrPtr;
  delete [] Ivec;
  return 0.0;
}

// ************************************************************************
// write to matlab file 
// ------------------------------------------------------------------------
double PMCMCAnalyzer::genMatlabFile(int nInputs,double *lower,double *upper,
                               double *XRange,int nPlots,int *plotIndices,
                               int nbins, int **bins, int ****bins2, 
                               pData &qData, int nChains, int chainCnt,
                               double **XChains, int *chainStatus)
{
  int    kk, kk2, ii2, jj, jj2, sumBins;
  double ddata;
  char   cfname[1001], charString[1001];;
  FILE   *fp;

  //**/ ---------------------------------------------------------------
  //**/ create posterior plots
  //**/ ---------------------------------------------------------------
  if (plotScilab()) strcpy(cfname, "scilabmcmc2.sci");
  else              strcpy(cfname, "matlabmcmc2.m");
  fp = fopen(cfname, "w");
  if (fp == NULL)
  {
    printOutTS(PL_ERROR, "ERROR: cannot open %s file.\n", cfname);
    return 0;
  }
  snprintf(charString,100,"This file shows posteriors plots");
  fwriteComment(fp, charString);
  snprintf(charString,100,"ns  - set to 1 for 1-step smoothing of 2D contours");
  fwriteComment(fp, charString);
  snprintf(charString,100,"ns1 - set to 1 for 1-step smoothing of 1D histgrams");
  fwriteComment(fp, charString);
  fprintf(fp, "ns  = 0;\n");
  fprintf(fp, "ns1 = 0;\n");
  fwritePlotCLF(fp);
  fprintf(fp, "active = [\n");
  for (kk = 0; kk < nInputs; kk++)
  {
    ii2 = binarySearchInt(kk, plotIndices, nPlots);
    if (ii2 < 0) fprintf(fp, "0\n");
    else         fprintf(fp, "1\n");
  }
  fprintf(fp, "];\n");
  fprintf(fp, "L = [\n");
  for (kk = 0; kk < nInputs; kk++) fprintf(fp, "%e ",lower[kk]);
  fprintf(fp, "];\n");
  fprintf(fp, "U = [\n");
  for (kk = 0; kk < nInputs; kk++) fprintf(fp, "%e ",upper[kk]);
  fprintf(fp, "];\n");
  if (plotScilab()) fprintf(fp, "iStr = [\n");
  else              fprintf(fp, "iStr = {\n");
  for (kk = 0; kk < nInputs-1; kk++)
  {
    if (qData.strArray_ != NULL)
         fprintf(fp, "'%s',", qData.strArray_[kk]);
    else fprintf(fp, "'Input %d',", kk+1);
  }
  if (plotScilab()) 
  {
    if (qData.strArray_ != NULL)
         fprintf(fp, "'%s'];\n", qData.strArray_[nInputs-1]);
    else fprintf(fp, "'Input %d'];\n", nInputs);
  }
  else
  {
    if (qData.strArray_ != NULL)
         fprintf(fp, "'%s'};\n", qData.strArray_[nInputs-1]);
    else fprintf(fp, "'Input %d'};\n", nInputs);
  }
  fprintf(fp, "X = zeros(%d,%d);\n", nInputs, nbins);
  fprintf(fp, "D = zeros(%d,%d);\n", nInputs, nbins);
  fprintf(fp, "NC = zeros(%d,%d,%d,%d);\n",nInputs,nInputs,nbins,nbins);
  for (kk = 0; kk < nInputs; kk++)
  {
    for (kk2 = 0; kk2 < nInputs; kk2++)
    {
      if (kk == kk2)
      {
        fprintf(fp, "X(%d,:) = [\n", kk+1);
        for (jj = 0; jj < nbins; jj++)
          fprintf(fp, "%e ", XRange[kk]/nbins*(jj+0.5)+lower[kk]);
        fprintf(fp, "];\n");
        fprintf(fp, "D(%d,:) = [\n", kk+1);
        sumBins = 0;
        for (jj = 0; jj < nbins; jj++) sumBins += bins[jj][kk];
        if (sumBins == 0) sumBins = 1;
        for (jj = 0; jj < nbins; jj++)
          fprintf(fp, "%e ", (double) bins[jj][kk]/(double) sumBins);
        fprintf(fp, "];\n");
      }
      else
      {
        fprintf(fp, "NC(%d,%d,:,:) = [\n", kk+1, kk2+1);
        for (jj = 0; jj < nbins; jj++)
        {
          for (jj2 = 0; jj2 < nbins; jj2++)
            fprintf(fp, "%d ", bins2[jj][jj2][kk][kk2]);
          fprintf(fp, "\n");
        }
        fprintf(fp, "]';\n");
      }
    }
  }
  fprintf(fp, "nInps  = length(active);\n");
  fprintf(fp, "nPlots = 0;\n");
  fprintf(fp, "for ii = 1 : nInps\n");
  fprintf(fp, "   if (active(ii) == 1)\n");
  fprintf(fp, "      nPlots = nPlots + 1;\n");
  fprintf(fp, "      active(ii) = nPlots;\n");
  fprintf(fp, "   end;\n");
  fprintf(fp, "end;\n");
  fprintf(fp, "for ii = 1 : nInps\n");
  fprintf(fp, "  for jj = ii : nInps\n");
  fprintf(fp, "    if (active(ii) ~= 0 & active(jj) ~= 0)\n");
  fprintf(fp, "      index = (active(ii)-1) * nPlots + active(jj);\n");
  fprintf(fp, "      subplot(nPlots,nPlots,index)\n");
  fprintf(fp, "      if (ii == jj)\n");
  fprintf(fp, "        n = length(D(ii,:));\n");
  fprintf(fp, "        DN = D(ii,:);\n");
  fprintf(fp, "        for kk = 1 : ns1\n");
  fprintf(fp, "          DN1 = DN;\n");
  fprintf(fp, "          for ll = 2 : n-1\n");
  fprintf(fp, "            DN(ll) = DN(ll) + DN1(ll+1);\n");
  fprintf(fp, "            DN(ll) = DN(ll) + DN1(ll-1);\n");
  fprintf(fp, "            DN(ll) = DN(ll) / 3;\n");
  fprintf(fp, "          end;\n");
  fprintf(fp, "        end;\n");
  fprintf(fp, "        bar(X(ii,:), DN, 1.0);\n");
  fprintf(fp, "        xmin = min(X(ii,:));\n");
  fprintf(fp, "        xmax = max(X(ii,:));\n");
  fprintf(fp, "        xwid = xmax - xmin;\n");
  fprintf(fp, "        xmin = xmin - 0.5 * xwid / %d;\n", nbins);
  fprintf(fp, "        xmax = xmax + 0.5 * xwid / %d;\n", nbins);
  fprintf(fp, "        ymax = max(DN);\n");
  if (plotScilab())
  {
    fprintf(fp, "        e = gce();\n");
    fprintf(fp, "        e.children.thickness = 2;\n");
    fprintf(fp, "        e.children.foreground = 0;\n");
    fprintf(fp, "        e.children.background = 2;\n");
    fprintf(fp, "        a = gca();\n");
    fprintf(fp, "        a.data_bounds=[xmin,0;xmax,ymax];\n");
    fprintf(fp, "        a.x_label.text = iStr(ii);\n");
    fprintf(fp, "        a.x_label.font_size = 3;\n");
    fprintf(fp, "        a.x_label.font_style = 4;\n");
    fprintf(fp, "        a.grid = [1 1];\n");
    fprintf(fp, "        a.y_label.text = Str(jj);\n");
    fprintf(fp, "        a.y_label.font_size = 3;\n");
    fprintf(fp, "        a.y_label.font_style = 4;\n");
    fprintf(fp, "        a.thickness = 2;\n");
    fprintf(fp, "        a.font_size = 3;\n");
    fprintf(fp, "        a.font_style = 4;\n");
    fprintf(fp, "        a.box = \"on\";\n");
  }
  else
  {
    fprintf(fp,"       axis([xmin xmax 0 ymax])\n");
    fprintf(fp,"       set(gca,'linewidth',2)\n");
    fprintf(fp,"       set(gca,'fontweight','bold')\n");
    fprintf(fp,"       set(gca,'fontsize',12)\n");
    fprintf(fp,"       xlabel(iStr(ii),'FontWeight','bold','FontSize',12)\n");
    fprintf(fp,"       ylabel('Probabilities','FontWeight','bold',");
    fprintf(fp,"'FontSize',12)\n");
    fprintf(fp,"       grid on\n");
    fprintf(fp,"       box on\n");
  }
  fprintf(fp,"      else\n");
  fprintf(fp,"        n = length(X(jj,:));\n");
  fprintf(fp,"        XT = X(jj,:);\n");
  fprintf(fp,"        YT = X(ii,:);\n");
  fprintf(fp,"        HX = (XT(n) - XT(1)) / (n-1);\n");
  fprintf(fp,"        HY = (YT(n) - YT(1)) / (n-1);\n");
  fprintf(fp,"        ZZ = squeeze(NC(ii,jj,:,:));\n");
  fprintf(fp,"        for kk = 1 : ns\n");
  fprintf(fp,"          ZZ1 = ZZ;\n");
  fprintf(fp,"          for ll = 2 : n-1\n");
  fprintf(fp,"            for mm = 2 : n-1\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll+1,mm);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll-1,mm);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll,mm+1);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll,mm-1);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll+1,mm+1);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll-1,mm-1);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll-1,mm+1);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) + ZZ1(ll+1,mm-1);\n");
  fprintf(fp,"              ZZ(ll,mm) = ZZ(ll,mm) / 9;\n");
  fprintf(fp,"            end;\n");
  fprintf(fp,"          end;\n");
  fprintf(fp,"        end;\n");
  fprintf(fp,"        ZZ = ZZ / (sum(sum(ZZ)));\n");
  if (plotScilab())
  {
    fprintf(fp, "        XX = [XT(1):HX:XT(n)];\n");
    fprintf(fp, "        YY = [YT(1):HY:YT(n)];\n");
    fprintf(fp, "        DD = splin2d(XX,YY,ZZ);\n");
    fprintf(fp, "        HX = 0.01 * (XT(n) - XT(1));\n");
    fprintf(fp, "        HY = 0.01 * (YT(n) - YT(1));\n");
    fprintf(fp, "        X2 = [XT(1):HX:XT(n)];\n");
    fprintf(fp, "        Y2 = [YT(1):HY:YT(n)];\n");
    fprintf(fp, "        [XI, YI] = ndgrid(X2, Y2);\n");
    fprintf(fp, "        disp('interpolation')\n");
    fprintf(fp, "        ZI =interp2d(XI, YI, XX, YY, DD, \"natural\");\n");
    fprintf(fp, "        disp('interpolation done')\n");
    fprintf(fp, "        ZB = ZI;\n");
    fprintf(fp, "        nX = length(X2);\n");
    fprintf(fp, "        nY = length(Y2);\n");
    fprintf(fp, "        for ii = 1 : nX\n");
    fprintf(fp, "          for jj = 1 : nY\n");
    fprintf(fp, "            ZI(ii,jj) = ZB(ii,nY-jj+1);\n");
    fprintf(fp, "          end;\n");
    fprintf(fp, "        end;\n");
    fprintf(fp, "        zmax = max(max(ZI));\n");
    fprintf(fp, "        zmin = min(min(ZI)) / zmax;\n");
    fprintf(fp, "        ZI   = ZI / zmax;\n");
    fprintf(fp, "        zmax = 1;\n");
    fprintf(fp, "        Matplot1((ZI-zmin)/(zmax-zmin)*64,[L(ii2),");
    fprintf(fp, "L(ii),U(ii2),U(ii)]);\n");
    fprintf(fp, "        xset(\"colormap\",jetcolormap(64));\n");
    fprintf(fp, "        colorbar(zmin,zmax);\n");
    fprintf(fp, "        contour2d(X2,Y2,ZB,5,rect=[L(ii2),L(ii),");
    fprintf(fp, "U(ii2),U(ii)]);\n");
    fprintf(fp, "        xset(\"fpf\",\" \");\n");
    fprintf(fp, "        a = gca();\n");
    fprintf(fp, "        a.x_label.text = iStr(jj);\n");
    fprintf(fp, "        a.x_label.font_size = 3;\n");
    fprintf(fp, "        a.x_label.font_style = 4;\n");
    fprintf(fp, "        a.y_label.text = iStr(ii);\n");
    fprintf(fp, "        a.y_label.font_size = 3;\n");
    fprintf(fp, "        a.y_label.font_style = 4;\n");
    fwritePlotAxesNoGrid(fp);
  }
  else
  {
    fprintf(fp,"%%      [YY,XX]=meshgrid(XT(1):HX:XT(n),YT(1):HY:YT(n));\n");
    fprintf(fp,"%%      HX = 0.01 * (XT(n) - XT(1));\n");
    fprintf(fp,"%%      HY = 0.01 * (YT(n) - YT(1));\n");
    fprintf(fp,"%%      [YI,XI]=meshgrid(XT(1):HX:XT(n),YT(1):HY:YT(n));\n");
    fprintf(fp,"%%      ZI=interp2(YY, XX, ZZ, YI, XI, 'spline');\n");
    fprintf(fp,"%%      pcolor(XI,YI,ZI)\n");
    fprintf(fp,"%%      shading interp\n");
    fprintf(fp,"%%      hold on\n");
    fprintf(fp,"%%      contour(XI,YI,ZI,5,'k')\n");
    fprintf(fp,"        imagesc(ZZ')\n");
    fprintf(fp,"        xtick = L(ii):(U(ii)-L(ii))/2:U(ii);\n");
    fprintf(fp,"        set(gca,'XTick',0:n/2:n);\n");
    fprintf(fp,"        set(gca,'XTickLabel', xtick);\n");
    fprintf(fp,"        ytick = L(jj):(U(jj)-L(jj))/2:U(jj);\n");
    fprintf(fp,"        set(gca,'YTick',0:n/2:n);\n");
    fprintf(fp,"        set(gca,'YTickLabel', ytick);\n");
    fprintf(fp,"        set(gca,'YDir', 'normal');\n");
    fprintf(fp,"        xlabel(iStr(jj),'FontWeight','bold','FontSize',12)\n");
    fprintf(fp,"        ylabel(iStr(ii),'FontWeight','bold','FontSize',12)\n");
    fwritePlotAxesNoGrid(fp);
  }
  fprintf(fp, "      end;\n");
  fprintf(fp, "    end;\n");
  fprintf(fp, "  end;\n");
  fprintf(fp, "end;\n");
  double minData=PSUADE_UNDEFINED;
  for (int iChain = 0; iChain < nChains; iChain++) 
  { 
    if (chainStatus[iChain] == 0)
    {
      for (jj = 0; jj < chainCnt; jj++) 
      {
        ddata = XChains[iChain][jj*(nInputs+1)+nInputs];
        if (ddata < minData) minData = ddata;
      }
    }
  }
  fprintf(fp, "      subplot(nPlots,nPlots,1)\n");
  if (plotMatlab())
  {
    fprintf(fp,"set(gcf,'NextPlot','add');\n");
    fprintf(fp,"axes;\n");
    fprintf(fp,
      "h=title('MCMC Posterior Distributions (best -loglikelihood=%e)',",
      minData);
    fprintf(fp,"'fontSize',12,'fontWeight','bold');\n");
    fprintf(fp,"set(gca,'Visible','off');\n");
    fprintf(fp,"set(h,'Visible','on');\n");
  }
  fprintf(fp,"negll = %e;\n", minData);
  fclose(fp);
  if (plotScilab())
       printOutTS(PL_INFO, "MCMC: scilabmcmc2.sci file has been created.\n");
  else printOutTS(PL_INFO, "MCMC: matlabmcmc2.m file has been created.\n");
  return 0;
}

// ************************************************************************
// write to another matlab file the mse of the posterior sample with the
// experimental data
// ------------------------------------------------------------------------
int PMCMCAnalyzer::genPostLikelihood(int nInputs, double *lower, 
                     double *upper, double *XRange, int numChains, 
                     int chainCnt, double **XChains, int *chainStatus, 
                     int chainLimit, int *rsIndices, double *rsValues, 
                     int *designParams, int dnInputs, int dnSamples, 
                     double *dSamInputs, FuncApprox **faPtrs, 
                     FuncApprox **faPtrs1,int nOutputs,double *discOutputs,
                     double *discFuncConstantMeans, double *dSamMeans,
                     double *dSamStdevs)
{
  int    iChain, ii, jj, ii2, kk2, dcnt;  
  double ddata,*XGuessS,*XDesignS,*YGuessS,*YDesignS,Ytemp,Ytemp2,ddata2;
  FILE   *fp;

  fp = fopen("matlabpostlikelihood.m", "w");
  if (fp == NULL) return -1;

  XGuessS  = new double[dnSamples * nInputs];
  XDesignS = new double[dnSamples * nInputs];
  YGuessS  = new double[dnSamples * nOutputs];
  YDesignS = new double[dnSamples * nOutputs];
  checkAllocate(YDesignS, "YDesignS in PMCMC::genPostLikelihood");
  fprintf(fp, "A = [\n");
  for (iChain = 0; iChain < numChains; iChain++)
  {
    if (chainStatus[iChain] == 0)
    {
      for (ii = chainCnt-chainLimit; ii < chainCnt; ii++)
      {
        for (jj = 0; jj < nInputs; jj++)
        {
          if ((rsIndices == NULL || rsIndices[jj] >= 0) &&
              (designParams == NULL || designParams[jj] == 0))
          {
            ddata = XChains[iChain][ii*(nInputs+1)+jj] * XRange[jj] + lower[jj];
            fprintf(fp, "%e ", ddata);
          }
          else if (rsIndices != NULL && rsIndices[jj] == 0)
            fprintf(fp, "%e ", rsValues[jj]);
          else if (designParams != NULL && designParams[jj] != 0)
            fprintf(fp, "%e ", 0.5 * (upper[jj] + lower[jj]));
        }
        //**/ fill in the input (including design) values for all experiments
        for (kk2 = 0; kk2 < dnSamples; kk2++)
        {
          dcnt = 0;
          for (ii2 = 0; ii2 < nInputs; ii2++)
          {
            XGuessS[kk2*nInputs+ii2] = XChains[iChain][ii*(nInputs+1)+ii2] * 
                                       XRange[ii2] + lower[ii2];
            if (designParams != NULL && designParams[ii2] == 1)
            {
              XGuessS[kk2*nInputs+ii2] = dSamInputs[kk2*dnInputs+dcnt];
              XDesignS[kk2*dnInputs+dcnt] = dSamInputs[kk2*dnInputs+dcnt];
              dcnt++;
            }
          }
        }
        //**/ evaluate the sample point through the response surface
        for (ii2 = 0; ii2 < nOutputs; ii2++)
        {
          faPtrs[ii2]->evaluatePoint(dnSamples,XGuessS,&YGuessS[ii2*dnSamples]);
          if (faPtrs1 != NULL && faPtrs1[ii2] != NULL)
          {
            for (kk2 = 0; kk2 < dnSamples; kk2++)
              YDesignS[ii2*dnSamples+kk2] = discOutputs[ii2*dnSamples+kk2];
          }
          else if (discFuncConstantMeans != NULL &&
                   discFuncConstantMeans[0] != PSUADE_UNDEFINED)
          {
            for (kk2 = 0; kk2 < dnSamples; kk2++)
              YDesignS[ii2*dnSamples+kk2] = discFuncConstantMeans[ii2];
          }
          else
          {
            for (kk2 = 0; kk2 < dnSamples; kk2++)
              YDesignS[ii2*dnSamples+kk2] = 0.0;
          }
        }
        //**/ compute the rms (weighted and unweighted)
        ddata = ddata2 = 0.0;
        for (ii2 = 0; ii2 < nOutputs; ii2++)
        {
          for (kk2 = 0; kk2 < dnSamples; kk2++)
          {
            Ytemp = YGuessS[ii2*dnSamples+kk2] + YDesignS[ii2*dnSamples+kk2];
            Ytemp2 = pow((Ytemp-dSamMeans[kk2*nOutputs+ii2]),2.0) /
                     (pow(dSamStdevs[kk2*nOutputs+ii2],2.0));
            ddata += Ytemp2;
            Ytemp2 = pow((Ytemp-dSamMeans[kk2*nOutputs+ii2]),2.0); 
            ddata2 += Ytemp2;
          }
        }
        ddata /= (dnSamples*nOutputs);
        ddata2 /= (dnSamples*nOutputs);
        fprintf(fp, "%e %e\n", ddata, ddata2);
      }
    }
  }
  fprintf(fp,"];\n");
  fprintf(fp,"figure(1)\n");
  fprintf(fp,"Y = A(:,%d);\n", nInputs+1);
  fprintf(fp,"subplot(1,2,1)\n");
  fprintf(fp,"hist(Y, 20);\n");
  fprintf(fp,"set(gca,'linewidth',2)\n");
  fprintf(fp,"set(gca,'fontweight','bold')\n");
  fprintf(fp,"set(gca,'fontsize',12)\n");
  fprintf(fp, 
   "xlabel('Weighted Mean Square Errors','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"ylabel('Frequencies','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"grid on\n");
  fprintf(fp,"box on\n");
  fprintf(fp,"subplot(1,2,2)\n");
  fprintf(fp,"plot(Y, 'lineWidth', 2);\n");
  fprintf(fp,"set(gca,'linewidth',2)\n");
  fprintf(fp,"set(gca,'fontweight','bold')\n");
  fprintf(fp,"set(gca,'fontsize',12)\n");
  fprintf(fp,"xlabel('Sample number','FontWeight','bold','FontSize',12)\n");
  fprintf(fp, 
   "ylabel('Weighted Mean Square Errors','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"grid on\n");
  fprintf(fp,"box on\n");
  fprintf(fp,"figure(2)\n");
  fprintf(fp,"Y2 = A(:,%d);\n", nInputs+2);
  fprintf(fp,"subplot(1,2,1)\n");
  fprintf(fp,"hist(Y2, 20);\n");
  fprintf(fp,"set(gca,'linewidth',2)\n");
  fprintf(fp,"set(gca,'fontweight','bold')\n");
  fprintf(fp,"set(gca,'fontsize',12)\n");
  fprintf(fp,"xlabel('Mean Square Errors','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"ylabel('Frequencies','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"grid on\n");
  fprintf(fp,"box on\n");
  fprintf(fp,"subplot(1,2,2)\n");
  fprintf(fp,"plot(Y2, 'lineWidth', 2);\n");
  fprintf(fp,"set(gca,'linewidth',2)\n");
  fprintf(fp,"set(gca,'fontweight','bold')\n");
  fprintf(fp,"set(gca,'fontsize',12)\n");
  fprintf(fp,"xlabel('Sample number','FontWeight','bold','FontSize',12)\n");
  fprintf(fp, 
   "ylabel('Weighted Mean Square Errors','FontWeight','bold','FontSize',12)\n");
  fprintf(fp,"grid on\n");
  fprintf(fp,"box on\n");
  fclose(fp);
  delete [] XGuessS;
  delete [] XDesignS;
  delete [] YGuessS;
  delete [] YDesignS;
  printOutTS(PL_INFO,"PMCMC: matlabpostlikelihood.m file has been created.\n");
  return 0;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
int PMCMCAnalyzer::setParams(int argc, char **argv)
{
   char  *request = (char *) argv[0];
   if (!strcmp(request, "setsim")) mode_ = 1;
   else
   {
      printOutTS(PL_ERROR,"PMCMCAnalyzer ERROR: setParams - not valid.\n");
      exit(1);
   }
   return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
PMCMCAnalyzer& PMCMCAnalyzer::operator=(const PMCMCAnalyzer &)
{
   printOutTS(PL_ERROR,
        "PMCMCAnalyzer operator= ERROR: operation not allowed.\n");
   exit(1);
   return (*this);
}

