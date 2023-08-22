// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team.
// All rights reserved.
//
// Please see the COPYRIGHT_and_LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public License.
//
// PSUADE is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License (as published by the Free Software
// Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Functions for the class PsuadeBase
// AUTHOR : CHARLES TONG
// DATE   : 2014
// ************************************************************************

// ------------------------------------------------------------------------
// system includes
// ------------------------------------------------------------------------
#ifdef HAVE_MPICH
#include <mpi.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "CommManager.h"
#include <PsuadeCmakeConfig.h>

#ifdef WINDOWS
#include <windows.h>
#endif

// ------------------------------------------------------------------------
// local includes : class definition and utilities
// ------------------------------------------------------------------------
#include "Psuade.h"
#include "PsuadeBase.h"
#include "dtype.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PsuadeConfig.h"

// ------------------------------------------------------------------------
// local includes : function approximator and others
// ------------------------------------------------------------------------
#include "PMCMCAnalyzer.h"
#include "PRSFuncApproxAnalyzer.h"
#include "FuncApprox.h"
#include "AnalysisManager.h"
#include "Sampling.h"
#include "FunctionInterface.h"
#include "PsuadeData.h"
#include "Optimizer.h"
#include "PsuadeSession.h"
#include "PKriging.h"
#include "aData.h"
#include "PrintingTS.h"

// ------------------------------------------------------------------------
// local defines 
// ------------------------------------------------------------------------
#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// interpret command from interactive session
// ------------------------------------------------------------------------
int PsuadeBase::interpretInteractiveParallel()
{
  int    mypid, nprocs, strLeng, iOne=1, commandCnt, ii, status, outputID;
  int    commFlag, proc, jj, kk, ss, targc, faType;
  char   command[101],dataFile[101], winput[501],lineIn[501],pString[501];
  char   *targv[3];
  double ddata;
  FILE   *fp;
  aData  aPtr;
  pData  pPtr;
  PsuadeSession *currSession=NULL;

  //**/ ----------------------------------------------------------------
  //**/ get communicator information
  //**/ ----------------------------------------------------------------
  if (psCommMgr_ == NULL)
  {
    printf("PSUADE ERROR: MPI communicator not instantiated.\n");
  }
  else
  {
    mypid  = psCommMgr_->getPID();
    nprocs = psCommMgr_->getNumProcs();
  }

  //**/ ----------------------------------------------------------------
  // loop on the command interpreter
  //**/ ----------------------------------------------------------------
  if (mypid == 0)
  {
    printf("PSUADE - A Problem Solving environment for \n");
    printf("         Uncertainty Analysis and Design Exploration (%d.%d.%dP)\n",
           psuade_VERSION_MAJOR, psuade_VERSION_MINOR, psuade_VERSION_PATCH);
    printf("(for help, enter <help>)\n");
    printEquals(PL_INFO, 0);
  }
  commandCnt = 0;
  while (1)
  {
    for (ii = 0; ii < 500; ii++) lineIn[ii] = '\0';
    winput[0] = '\0';
    pString[0] = '\0';
    command[0] = '\0';
    //**/ -------------------------------------------------------------
    //**/ read in command
    //**/ -------------------------------------------------------------
    if (mypid == 0)
    {
      printf("ppsuade> ");
      fgets(lineIn,500,stdin); 
      sscanf(lineIn, "%s", command);
      if (!strcmp(command, "\0"))
      {
        commandCnt++;
        if (commandCnt >= 10)
        {
          printf("Enter carriage return > 10 times ==> terminate.\n");
          strcpy(command, "return");
          strLeng = 6;
          psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
          psCommMgr_->bcast((void *) command, strLeng, CHAR, 0);
          return 0;
        }
      }
      else commandCnt = 0;
      strLeng = strlen(command) + 1; 
      command[strLeng-1] = '\0';
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) command, strLeng, CHAR, 0);
    }
    else
    {
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) command, strLeng, CHAR, 0);
      if (!strcmp(command, "return")) return 0;
    }

    //**/ -------------------------------------------------------------
    //**/ help in command
    //**/ -------------------------------------------------------------
    if (!strcmp(command, "help") || !strcmp(command, "h"))
    {
      if (mypid == 0)
      {
        strcpy(winput, "\0");
        sscanf(lineIn, "%s %s %s", command, winput, pString);
        if (!strcmp(winput, "io"))
        {
          printf("Commands for reading/write/updating data to/from files:\n");
          printf("(to see details of each command, use '-h' option)\n");
          printf("\tload <file> (load a data file in PSUADE format) \n");
        }
        else if (!strcmp(winput, "rs"))
        {
          printf("Commands for response surface analysis:\n");
          printf("(to see details of each command, use '-h' option)\n");
          printf("\trscheck (response surface validation) \n");
          printf("\trskrig  (response surface generation with Kriging)\n");
        }
        else if (!strcmp(winput, "qsa"))
        {
          printf("Commands for RS-based uncertainty/sensitivity analysis:\n");
          printf("(to see more qsa commands, use 'help qsa long')\n");
          printf("(to see details of each command, use '-h' option)\n");
          printf("(BS == bootstrapping)\n");
          printf("\trsuab      (RS-based UA with bootstrapping)\n");
          printf("\trsmeb      (RS-based McKay main effect with BS)\n");
          printf("\trsieb      (RS-based 2-way effect with BS)\n");
          printf("\trssobol1b  (RS-based Sobol' main effect with BS)\n");
          printf("\trssobol2b  (RS-based Sobol' interaction effect with BS)\n");
          printf("\trssoboltsib(RS-based Sobol' total effect with BS)\n");
        }
        else if (!strcmp(winput, "calibration"))
        {
          printf("Commands for optimization/calibration:\n");
          printf("(to see details of each command, use '-h' option)\n");
          printf("\trsmcmc     (RS-based Bayesian inversion using MCMC)\n");
        }
        else if (!strcmp(winput, "misc"))
        {
          printf("Miscellaneous commands:\n");
          printf("\tquit (exit)    (terminate command line session)\n");
          printf("\trsmax <d>      (set maximum number of data points for RS)\n");
          printf("\tprintlevel <d> (set print level)\n");
        }
        else if (!strcmp(winput, "advanced"))
        {
          printf("Advanced analysis and control commands:\n");
          printf("\tio_expert  (turn on/off IO expert mode)\n");
          printf("\trs_expert  (turn on/off response surface expert mode)\n");
          printf("\tana_expert (turn on/off analysis expert mode)\n");
        }
        else
        {
          printf("Help topics:\n");
          printf("\tio           (file read/write commands)\n");
          printf("\trs           (response surface analysis commands)\n");
          printf("\tqsa          (quantitative sensitivity analysis commands)\n");
          printf("\tcalibration  (Bayesian calibration/optimization commands)\n");
          printf("\tmisc         (miscellaneous commands)\n");
          printf("\tadvanced     (advanced analysis and control commands)\n");
          printf("\t<command -h> (help for a specific command)\n");
        }
        strcpy(winput,"psuadeNoOp");
        strLeng = 11;
        dataFile[strLeng-1] = '\0';
      }
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) winput, strLeng, CHAR, 0);
    }

    //**/ =============================================================
    // Input/output commands
    //**/ =============================================================
    //**/ =============================================================
    //**/ -------------------------------------------------------------
    // +++ load 
    //**/ load data from a psuade file. 
    //**/ -------------------------------------------------------------
    if (!strcmp(command, "load"))
    {
      if (mypid == 0)
      {
        sscanf(lineIn,"%s %s",command,winput);
        if (!strcmp(winput, "-h"))
        { 
          printf("load: load PSUADE data from a file to local memory\n");
          printf("syntax: load <filename>.\n");
          printf("where <filename> is a PSUADE data file.\n");
          strcpy(dataFile,"psuadeNoOp");
          strLeng = 11;
          dataFile[strLeng-1] = '\0';
        }
        else
        {
          strcpy(dataFile, "\0");
          sscanf(lineIn,"%s %s",command,dataFile);
          if (!strcmp(dataFile, "psuadeNoOp")) 
          {
            printf("ERROR: file name psuadeNoOP not allowed.\n");
            strLeng = 11;
          }
          else 
          {
            if (psuadeIO_ != NULL) delete psuadeIO_;
            psuadeIO_ = new PsuadeData();
            status = psuadeIO_->readPsuadeFile(dataFile);
            if (status != 0)
            {
              printf("file %s not found or format invalid.\n", dataFile);
              strcpy(dataFile,"psuadeNoOp");
              strLeng = 11;
              dataFile[strLeng-1] = '\0';
            }
            else
            {
              strLeng = strlen(dataFile) + 1;
              dataFile[strLeng-1] = '\0';
            }
            delete psuadeIO_;
            psuadeIO_ = NULL;
          }
        }
      }
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) dataFile, strLeng, CHAR, 0);
      if (!strcmp(dataFile, "psuadeNoOp")) continue;

      if (currSession != NULL) delete currSession;
      currSession = new PsuadeSession();
      if (psuadeIO_ != NULL) delete psuadeIO_;
      psuadeIO_ = new PsuadeData();
      status = psuadeIO_->readPsuadeFile(dataFile);
      psuadeIO_->getSession(currSession);

      if (currSession->vecSamInputs_.length() == 0 ||
          currSession->vecSamOutputs_.length() == 0)
      {
        if (mypid == 0)
        {
          printf("WARNING: no sample matrix nor output found.\n");
        }
        delete currSession;
        currSession = NULL;
        psuadeIO_ = NULL;
      }
      else
      {
        if (mypid == 0)
        {
          printf("load complete : nSamples = %d\n",currSession->nSamples_);
          printf("                nInputs  = %d\n",currSession->nInputs_);
          printf("                nOutputs = %d\n",currSession->nOutputs_);
        }
        currSession->psuadeIO_ = psuadeIO_;
      }
    }

    //**/ -------------------------------------------------------------
    // +++ rsmcmc
    //**/ response surface based MCMC
    //**/ -------------------------------------------------------------
    else if (!strcmp(command, "rsmcmc"))
    {
      if (mypid == 0)
      {
        strcpy(winput,"psuadeOkay");
        strLeng = 11;
        winput[strLeng-1] = '\0';
        sscanf(lineIn,"%s %s",command,winput);
        if (!strcmp(winput, "-h"))
        {
          printf("rsmcmc: perform MCMC on response surfaces\n");
          printf("syntax: rsmcmc (no argument needed)\n");
          printf("The response surface is constructed from the sample\n");
          printf("already loaded into local memory.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        if (currSession->nInputs_ <= 0 || psuadeIO_ == NULL)
        {
          printf("ERROR: data not loaded yet.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
      }
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) winput, strLeng, CHAR, 0);
      if (!strcmp(winput, "psuadeNoOp")) continue;
      PMCMCAnalyzer *mcmc = new PMCMCAnalyzer(psCommMgr_);
      aPtr.nInputs_ = currSession->nInputs_;
      aPtr.nOutputs_ = currSession->nOutputs_;
      aPtr.nSamples_ = currSession->nSamples_;
      aPtr.sampleInputs_ = currSession->vecSamInputs_.getDVector();
      aPtr.sampleOutputs_ = currSession->vecSamOutputs_.getDVector();
      aPtr.sampleStates_ = currSession->vecSamStates_.getIVector();
      aPtr.iLowerB_ = currSession->vecInpLBounds_.getDVector();
      aPtr.iUpperB_ = currSession->vecInpUBounds_.getDVector();
      aPtr.inputPDFs_ = currSession->vecInpPDFs_.getIVector();
      aPtr.inputMeans_ = currSession->vecInpMeans_.getDVector();
      aPtr.inputStdevs_ = currSession->vecInpStds_.getDVector();
      aPtr.ioPtr_ = psuadeIO_;
      aPtr.printLevel_ = outputLevel_;
      mcmc->analyze(aPtr);
    }

    //**/ ----------------------------------------------------------
    // +++ rssobol1b 
    //**/ rssobol1 with bootstrap
    //**/ ----------------------------------------------------------
    else if (!strcmp(command, "rssobol1b"))
    {
      if (mypid == 0)
      {
        strcpy(winput,"psuadeOkay");
        strLeng = 11;
        winput[strLeng-1] = '\0';
        sscanf(lineIn,"%s %s",command,winput);
        if (!strcmp(winput, "-h"))
        {
          printf("rssobol1b: RS-based Sobol' sensitivity analysis\n");
          printf("syntax: rssobol1b (no argument needed)\n");
          printf("Note: This command uses bootstrapped samples\n");
          printf("      multiple times to get the errors in\n");
          printf("      Sobol' index estimation.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
          return 0;
        }
        if (currSession->nInputs_ <= 0 || psuadeIO_ == NULL)
        {
          printf("ERROR: no data (load data first).\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        if (currSession->nSamples_ < 5)
        {
          printf("ERROR: This command is not suitable for small\n");
          printf("       samples. nSamples needs to be at least 5.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
      }
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) winput, strLeng, CHAR, 0);
      if (!strcmp(winput, "psuadeNoOp")) continue;
      //**/ get output information
      if (mypid == 0)
      {
        printf("rssobol1b INFO: MAKE SURE YOU SET YOUR DESIRED\n");
        printf("          RESPONSE SURFACE TYPE IN YOUR DATA FILE.\n");
        snprintf(pString,100,"Enter output number (1 - %d) : ", 
                 currSession->nOutputs_);
        outputID = getInt(1, currSession->nOutputs_, pString);
        outputID--;
      }
      psCommMgr_->bcast((void *) &outputID, iOne, INT, 0);

      //**/ set up analysis manager
      int analysisMethod = PSUADE_ANA_RSSOBOL1;
      AnalysisManager *anaManager = new AnalysisManager();
      anaManager->setup(analysisMethod, 0);

      //**/ save parameters resident in psuadeIO (to be restored later)
      psuadeIO_->getParameter("ana_diagnostics",pPtr);
      int saveDiag = pPtr.intData_;
      psuadeIO_->updateAnalysisSection(-1,-1,-1,-2,-1,-1);

      //**/ if rstype = MarsBagg, change to MARS
      psuadeIO_->getParameter("ana_rstype", pPtr);
      int saveRS = pPtr.intData_;
      if (saveRS == PSUADE_RS_MARSB)
      {
        if (mypid == 0)
        {
          printf("rssobol1b INFO: MarsBagg response surface selected\n");
          printf("          but it is redundant - set to MARS.\n");
        }
        psuadeIO_->updateAnalysisSection(-1,-1,PSUADE_RS_MARS,-3,-1,-1);
      }

      //**/ need to turn off expert modes, so save them first 
      psConfig_.AnaExpertModeSaveAndReset();
      psConfig_.RSExpertModeSaveAndReset();
      if (mypid == 0)
      {
        printf("rssobol1b INFO: RS expert mode disabled. You can use\n");
        printf("          config file to set special RS options.\n");
      }

      //**/ set up storage for the response surface samples
      int    nInps = currSession->nInputs_;
      int    nOuts = currSession->nOutputs_;
      int    nSams = currSession->nSamples_;
      int    *tempI = new int[nSams];
      int    *states = new int[nSams];
      int    nbs, nSams2, ind;
      double *tempX = new double[nSams*nInps];
      double *tempY = new double[nSams];
      double *tempT, *tempM, *tempV;
      double *SamInps = currSession->vecSamInputs_.getDVector();
      double *SamOuts = currSession->vecSamOutputs_.getDVector();
      pData  *pdata=NULL;

      //**/ set up for iterations
      if (mypid == 0)
      {
        snprintf(pString,100,
                "How many bootstrapped samples to use (10 - 300) : ");
        nbs = getInt(10, 300, pString);
        if (nbs < nprocs)
        {
          nbs = nprocs;
          printf("INFO: number of bootstraps set to %d\n",nbs);
        }
        if (nbs % nprocs != 0) nbs = (nbs / nprocs + 1) * nprocs;
      }
      psCommMgr_->bcast((void *) &nbs, iOne, INT, 0);
      tempT = new double[nbs*nInps];

      //**/ iterate
      initializePrintingTS(-1, NULL, -1);
      for (kk = 0; kk < nbs; kk++)
      {
        if (kk % nprocs == mypid)
        {
          printf("Proc %5d: rssobol1b iteration = %d (of %d)\n",
                 mypid,kk+1,nbs);
          //**/ random draw
          for (jj = 0; jj < nSams; jj++) tempI[jj] = 0;
          ss = nSams2 = 0;
          while (ss < nSams)
          {
            ind = PSUADE_rand() % nSams;
            if (tempI[ind] == 0)
            {
              for (ii = 0; ii < nInps; ii++)
                tempX[nSams2*nInps+ii] = 
                           SamInps[ind*nInps+ii];
              tempY[nSams2] = SamOuts[ind*nOuts+outputID];
              states[nSams2] = currSession->vecSamStates_[ind];
              tempI[ind] = 1;
              nSams2++;
            }
            ss++;
          }
          psuadeIO_->updateInputSection(nSams2,nInps,NULL,NULL,
                               NULL,tempX,NULL,NULL,NULL,NULL,NULL);
          psuadeIO_->updateOutputSection(nSams2,1,tempY,states,
                               currSession->outputNames_.getStrings());
          psuadeIO_->updateMethodSection(PSUADE_SAMP_MC,nSams2,
                               -1,-1,-1);
      
          //**/ analyze the result
          anaManager->analyze(psuadeIO_, 0, NULL, 0);
          pdata = psuadeIO_->getAuxData();
          commFlag = 0;
          if (pdata->nDbles_ != nInps) commFlag = 1;
          psCommMgr_->bcast((void *) &commFlag, iOne, INT, 0);
          if (commFlag == 1)
          {
            if (mypid == 0)
            {
              printf("ERROR: nInputs do not match.\n");
              printf("       Consult PSUADE developers.\n");
            }
            delete [] tempT;
            delete [] tempX;
            delete [] tempY;
            delete [] tempI;
            delete [] states;
            return -1;
          }
          //**/ get the statistics
          if (pdata->dbleData_ > 0)
            for (ii = 0; ii < nInps; ii++)
              tempT[kk*nInps+ii] =
                          pdata->dbleArray_[ii]/pdata->dbleData_;
          else
            for (ii = 0; ii < nInps; ii++)
              tempT[kk*nInps+ii] = pdata->dbleArray_[ii];

          //**/ clean up
          pdata->clean();
        }
        else
        {
          for (jj = 0; jj < nSams; jj++) ind = PSUADE_rand();
        }
      }
      delete [] tempX;
      delete [] tempY;
      delete [] tempI;
      delete [] states;

      commFlag = 1;
      psCommMgr_->bcast((void *) &commFlag, iOne, INT, 0);
      for (kk = 0; kk < nbs; kk++)
      {
        if (mypid == 0 && kk % nprocs != mypid)
        {
          proc = kk % nprocs;
          psCommMgr_->recv((void *) &tempT[kk*nInps],nInps,DOUBLE,kk,proc);
        }
        if (mypid != 0 && kk % nprocs == mypid)
        {
          psCommMgr_->send((void *) &tempT[kk*nInps],nInps,DOUBLE,kk,0);
        }
      }
      psConfig_.AnaExpertModeRestore();
      psConfig_.RSExpertModeRestore();
      if (mypid == 0)
      {
        tempM = new double[nInps];
        for (ii = 0; ii < nInps; ii++)
        {
          tempM[ii] = tempT[ii];
          for (jj = 1; jj < nbs; jj++) tempM[ii] += tempT[jj*nInps+ii];
          tempM[ii] /= (double) nbs;
        }
        tempV = new double[nInps];
        for (ii = 0; ii < nInps; ii++)
        {
          tempV[ii] = pow(tempT[ii]-tempM[ii], 2.0);
          for (jj = 1; jj < nbs; jj++)
            tempV[ii] += pow(tempT[jj*nInps+ii]-tempM[ii],2.0);
          tempV[ii] /= (double) (nbs - 1);
          tempV[ii] = sqrt(tempV[ii]);
        }
        printEquals(PL_INFO, 0);
        printf("Sobol1 Statistics (based on %d replications): \n",nbs);
        for (ii = 0; ii < nInps; ii++)
          printf("Input %4d: mean = %16.8e, std = %16.8e\n",ii+1,
                 tempM[ii],tempV[ii]);

        //**/ generate matlab/scilab file
        if (plotScilab()) fp = fopen("scilabrssobol1b.sci","w");
        else              fp = fopen("matlabrssobol1b.m","w");
        if (fp == NULL) printf("ERROR: cannot open plot file.\n");
        else
        {
          strcpy(pString," This file contains first order Sobol' indices");
          fwriteComment(fp, pString);
          strcpy(pString," with error bars coming from bootstrapping.");
          fwriteComment(fp, pString);
          strcpy(pString," to select the most important ones to display,");
          fwriteComment(fp, pString);
          strcpy(pString," set sortFlag = 1 and set nn to be the number");
          fwriteComment(fp, pString);
          strcpy(pString," of inputs to display.\n");
          fwriteComment(fp, pString);
          fprintf(fp, "sortFlag = 0;\n");
          fprintf(fp, "nn = %d;\n", nInps);
          fprintf(fp, "Means = [\n");
          for (ii = 0; ii < nInps; ii++) fprintf(fp,"%24.16e\n",tempM[ii]);
          fprintf(fp, "];\n");
          fprintf(fp, "Stds = [\n");
          for (ii = 0; ii < nInps; ii++) fprintf(fp,"%24.16e\n",tempV[ii]);
          fprintf(fp, "];\n");
          if (plotScilab()) fprintf(fp, "  Str = [");
          else              fprintf(fp, "  Str = [");
          for (ii = 0; ii < nInps-1; ii++)
            fprintf(fp,"'%s',",currSession->inputNames_[ii]);
          if (plotScilab()) 
               fprintf(fp,"'%s'];\n",currSession->inputNames_[nInps-1]);
          else fprintf(fp,"'%s'};\n",currSession->inputNames_[nInps-1]);
          fwriteHold(fp, 0);
          fprintf(fp, "if (sortFlag == 1)\n");
          if (plotScilab())
               fprintf(fp, "  [Means, I2] = gsort(Means);\n");
          else fprintf(fp, "  [Means, I2] = sort(Means,'descend');\n");
          fprintf(fp, "  Stds = Stds(I2);\n");
          fprintf(fp, "  I2 = I2(1:nn);\n");
          fprintf(fp, "  Means = Means(1:nn);\n");
          fprintf(fp, "  Stds = Stds(1:nn);\n");
          fprintf(fp, "  Str  = Str(I2);\n");
          fprintf(fp, "end\n");
          fprintf(fp, "ymin = min(Means-Stds);\n");
          fprintf(fp, "ymax = max(Means+Stds);\n");
          fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
          if (plotScilab()) fprintf(fp, "drawlater\n");
          fprintf(fp, "bar(Means,0.8);\n");
          fprintf(fp, "for ii = 1:nn\n");
          fprintf(fp, "   if (ii == 1)\n");
          fwriteHold(fp, 1);
          fprintf(fp, "   end;\n");
          fprintf(fp, "   XX = [ii ii];\n");
          fprintf(fp, "   d1 = Means(ii)-Stds(ii);\n");
          fprintf(fp, "   d2 = Means(ii)+Stds(ii);\n");
          fprintf(fp, "   if (d1 < 0)\n");
          fprintf(fp, "      d1 = 0.0;\n");
          fprintf(fp, "   end;\n");
          fprintf(fp, "   YY = [d1 d2];\n");
          fprintf(fp, "   plot(XX,YY,'-ko','LineWidth',3.0,");
          fprintf(fp, "'MarkerEdgeColor',");
          fprintf(fp, "'k','MarkerFaceColor','g','MarkerSize',13)\n");
          fprintf(fp, "end;\n");
          fwritePlotAxes(fp);
          if (plotScilab())
          {
            fprintf(fp,"a=gca();\n");
            fprintf(fp,"a.data_bounds=[0, ymin; nn+1, ymax];\n");
            fprintf(fp,"newtick = a.x_ticks;\n");
            fprintf(fp,"newtick(2) = [1:nn]';\n");
            fprintf(fp,"newtick(3) = Str';\n");
            fprintf(fp,"a.x_ticks = newtick;\n");
            fprintf(fp,"a.x_label.font_size = 3;\n");
            fprintf(fp,"a.x_label.font_style = 4;\n");
          }
          else
          {
            fprintf(fp,"axis([0  nn+1 ymin ymax])\n");
            fprintf(fp,"set(gca,'XTickLabel',[]);\n");
            fprintf(fp,"th=text(1:nn, repmat(ymin-0.05*(ymax-ymin)");
            fprintf(fp,",nn,1),Str,");
            fprintf(fp,"'HorizontalAlignment','left','rotation',90);\n");
            fprintf(fp,"set(th, 'fontsize', 12)\n");
            fprintf(fp,"set(th, 'fontweight', 'bold')\n");
          }
          fwritePlotTitle(fp,"First Order Sobol Indices (with bootstrap)");
          fwritePlotYLabel(fp,"First Order Sobol Index (Normalized)");
          if (plotScilab())
          {
            fprintf(fp, "drawnow\n");
            printf("rssobol1b plot file = scilabrssobol1b.sci\n");
          }
          else printf("rssobol1b plot file = matlabrssobol1b.m\n");
          fclose(fp);
        }
        delete [] tempM;
        delete [] tempV;
      }

      //**/ clean up
      delete anaManager;
      delete [] tempT;

      //**/ restore previous settings
      if (saveRS == PSUADE_RS_MARSB)
        psuadeIO_->updateAnalysisSection(-1,-1,saveRS,-3,-1,-1);
      psuadeIO_->updateAnalysisSection(-1,-1,-1,saveDiag,-1,-1);
    }

    //**/ ----------------------------------------------------------
    // +++ rssobol2b 
    //**/ rssobol2 with bootstrap
    //**/ ----------------------------------------------------------
    else if (!strcmp(command, "rssobol2b"))
    {
      if (mypid == 0)
      {
        strcpy(winput,"psuadeOkay");
        strLeng = 11;
        winput[strLeng-1] = '\0';
        sscanf(lineIn,"%s %s",command,winput);
        if (!strcmp(winput, "-h"))
        {
          printf("rssobol2b: RS-based Sobol' sensitivity analysis\n");
          printf("syntax: rssobol2b (no argument needed)\n");
          printf("Note: This command uses bootstrapped samples\n");
          printf("      multiple times to get the errors in\n");
          printf("      Sobol' index estimation.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
          return 0;
        }
        if (currSession->nInputs_ <= 0 || psuadeIO_ == NULL)
        {
          printf("ERROR: no data (load data first).\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        if (currSession->nInputs_ <= 2)
        {
          printf("INFO: no point doing this analysis for nInputs <= 2.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        if (currSession->nSamples_ < 5)
        {
          printf("ERROR: This command is not suitable for small\n");
          printf("       samples. nSamples needs to be at least 5.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
      }
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) winput, strLeng, CHAR, 0);
      if (!strcmp(winput, "psuadeNoOp")) continue;
      //**/ get output information
      if (mypid == 0)
      {
        printf("rssobol2b INFO: MAKE SURE YOU SET YOUR DESIRED\n");
        printf("          RESPONSE SURFACE TYPE IN YOUR DATA FILE.\n");
        snprintf(pString,100,"Enter output number (1 - %d) : ", 
                 currSession->nOutputs_);
        outputID = getInt(1, currSession->nOutputs_, pString);
        outputID--;
      }
      psCommMgr_->bcast((void *) &outputID, iOne, INT, 0);

      //**/ set up analysis manager
      int analysisMethod = PSUADE_ANA_RSSOBOL2;
      AnalysisManager *anaManager = new AnalysisManager();
      anaManager->setup(analysisMethod, 0);

      //**/ save parameters resident in psuadeIO (to be restored later)
      psuadeIO_->getParameter("ana_diagnostics",pPtr);
      int saveDiag = pPtr.intData_;
      psuadeIO_->updateAnalysisSection(-1,-1,-1,-2,-1,-1);

      //**/ if rstype = MarsBagg, change to MARS
      psuadeIO_->getParameter("ana_rstype", pPtr);
      int saveRS = pPtr.intData_;
      if (saveRS == PSUADE_RS_MARSB)
      {
        if (mypid == 0)
        {
          printf("rssobol1b INFO: MarsBagg response surface selected\n");
          printf("          but it is redundant - set to MARS.\n");
        }
        psuadeIO_->updateAnalysisSection(-1,-1,PSUADE_RS_MARS,-3,-1,-1);
      }

      //**/ need to turn off expert modes, so save them first 
      psConfig_.AnaExpertModeSaveAndReset();
      psConfig_.RSExpertModeSaveAndReset();
      if (mypid == 0)
      {
        printf("rssobol2b INFO: RS expert mode disabled. You can use\n");
        printf("          config file to set special RS options.\n");
      }

      //**/ set up storage for the response surface samples
      int    nInps = currSession->nInputs_;
      int    nOuts = currSession->nOutputs_;
      int    nSams = currSession->nSamples_;
      int    *tempI = new int[nSams];
      int    *states = new int[nSams];
      int    nbs, nSams2, ind;
      double *tempX = new double[nSams*nInps];
      double *tempY = new double[nSams];
      double *tempT, *tempM, *tempV;
      double *SamInps = currSession->vecSamInputs_.getDVector();
      double *SamOuts = currSession->vecSamOutputs_.getDVector();
      pData  *pdata=NULL;

      //**/ set up for iterations
      if (mypid == 0)
      {
        snprintf(pString,100,
                "How many bootstrapped samples to use (10 - 300) : ");
        nbs = getInt(10, 300, pString);
        if (nbs < nprocs)
        {
          nbs = nprocs;
          printf("INFO: number of bootstraps set to %d\n",nbs);
        }
        if (nbs % nprocs != 0) nbs = (nbs / nprocs + 1) * nprocs;
      }
      psCommMgr_->bcast((void *) &nbs, iOne, INT, 0);
      tempT = new double[(nbs+3)*nInps*nInps];

      //**/ iterate
      initializePrintingTS(-1, NULL, -1);
      for (kk = 0; kk < nbs; kk++)
      {
        if (kk % nprocs == mypid)
        {
          printf("Proc %5d: rssoboltsib iteration = %d (of %d)\n",
                 mypid,kk+1,nbs);
          //**/ random draw
          for (jj = 0; jj < nSams; jj++) tempI[jj] = 0;
          ss = nSams2 = 0;
          while (ss < nSams)
          {
            ind = PSUADE_rand() % nSams;
            if (tempI[ind] == 0)
            {
              for (ii = 0; ii < nInps; ii++)
                tempX[nSams2*nInps+ii] = SamInps[ind*nInps+ii];
              tempY[nSams2] = SamOuts[ind*nOuts+outputID];
              states[nSams2] = currSession->vecSamStates_[ind];
              tempI[ind] = 1;
              nSams2++;
            }
            ss++;
          }
          psuadeIO_->updateInputSection(nSams2,nInps,NULL,NULL,
                               NULL,tempX,NULL,NULL,NULL,NULL,NULL);
          psuadeIO_->updateOutputSection(nSams2,1,tempY,states,
                               currSession->outputNames_.getStrings());
          psuadeIO_->updateMethodSection(PSUADE_SAMP_MC,nSams2,-1,-1,-1);
      
          //**/ analyze the result
          anaManager->analyze(psuadeIO_, 0, NULL, 0);
          pdata = psuadeIO_->getAuxData();
          commFlag = 0;
          if (pdata->nDbles_ < nInps) commFlag = 1;
          psCommMgr_->bcast((void *) &commFlag, iOne, INT, 0);
          if (commFlag == 1)
          {
            if (mypid == 0)
            {
              printf("ERROR: nInputs do not match.\n");
              printf("       Consult PSUADE developers.\n");
            }
            delete [] tempT;
            delete [] tempX;
            delete [] tempY;
            delete [] tempI;
            delete [] states;
            return -1;
          }
          //**/ get the statistics
          if (pdata->dbleData_ > 0)
          {
            for (ii = 0; ii < nInps*nInps; ii++)
              tempT[kk*nInps*nInps+ii] =
                        pdata->dbleArray_[ii]/pdata->dbleData_;
          }
          else
          {
            for (ii = 0; ii < nInps*nInps; ii++)
              tempT[kk*nInps*nInps+ii] = pdata->dbleArray_[ii];
          }
          //**/ clean up
          pdata->clean();
        }
        else
        {
          for (jj = 0; jj < nSams; jj++) ind = PSUADE_rand();
        }
      }
      delete [] tempX;
      delete [] tempY;
      delete [] tempI;
      delete [] states;
      commFlag = 1;
      psCommMgr_->bcast((void *) &commFlag, iOne, INT, 0);
      for (kk = 0; kk < nbs; kk++)
      {
        if (mypid == 0 && kk % nprocs != mypid)
        {
          proc = kk % nprocs;
          psCommMgr_->recv((void *) &tempT[kk*nInps*nInps],nInps*nInps,
                           DOUBLE,kk,proc);
        }
        if (mypid != 0 && kk % nprocs == mypid)
        {
          psCommMgr_->send((void *) &tempT[kk*nInps*nInps],nInps*nInps,
                           DOUBLE,kk,0);
        }
      }
      psConfig_.AnaExpertModeRestore();
      psConfig_.RSExpertModeRestore();
      if (mypid == 0)
      {
        //**/ compute mean and std dev
        for (ii = 0; ii < nInps; ii++)
        {
          for (jj = 0; jj <= ii; jj++)
            tempT[nbs*nInps*nInps+ii*nInps+jj] = 0.0;
          for (jj = ii+1; jj < nInps; jj++)
          {
            ddata = 0.0;
            for (kk = 0; kk < nbs; kk++)
              ddata += tempT[kk*nInps*nInps+ii*nInps+jj];
            tempT[nbs*nInps*nInps+ii*nInps+jj] = ddata/(double) nbs;
            tempT[(nbs+1)*nInps*nInps+ii*nInps+jj] = tempT[ii*nInps+jj];
            tempT[(nbs+2)*nInps*nInps+ii*nInps+jj] = tempT[ii*nInps+jj];
            for (kk = 1; kk < nbs; kk++)
            {
              ddata = tempT[kk*nInps*nInps+ii*nInps+jj];
              if (ddata < tempT[(nbs+1)*nInps*nInps+ii*nInps+jj])
                tempT[(nbs+1)*nInps*nInps+ii*nInps+jj] = ddata;
              if (ddata > tempT[(nbs+2)*nInps*nInps+ii*nInps+jj])
                tempT[(nbs+2)*nInps*nInps+ii*nInps+jj] = ddata;
            }
          }
        }
        printDashes(PL_INFO, 0);
        printf("Sobol2b Statistics (based on %d replications): \n", nbs);
        printf("Note: Quantities are normalized. \n");
        printf("      Multiply by variance to compute actual.\n");
        printEquals(PL_INFO, 0);
        for (ii = 0; ii < nInps; ii++)
        {
          for (jj = 0; jj <= ii; jj++) tempT[ii*nInps+jj] = 0.0;
          for (jj = ii+1; jj < nInps; jj++)
          {
            ddata = 0.0;
            for (kk = 0; kk < nbs; kk++)
            {
              tempT[kk*nInps*nInps+ii*nInps+jj] -=
                       tempT[nbs*nInps*nInps+ii*nInps+jj];
              ddata += pow(tempT[kk*nInps*nInps+ii*nInps+jj],2.0);
            }
            ddata /= (double) (nbs - 1);
            tempT[ii*nInps+jj] = ddata;
            printf("Input (%4d %4d): mean = %16.8e, std = %16.8e\n",ii+1,
                   jj+1,tempT[nbs*nInps*nInps+ii*nInps+jj],ddata);
            printf("Input (%4d %4d): min  = %16.8e, max = %16.8e\n",ii+1,
                   jj+1,tempT[(nbs+1)*nInps*nInps+ii*nInps+jj],
                   tempT[(nbs+2)*nInps*nInps+ii*nInps+jj]);
          }
        }
        printEquals(PL_INFO, 0);
        if (plotScilab()) fp = fopen("scilabrssobol2b.sci", "w");
        else              fp = fopen("matlabrssobol2b.m", "w");
        if (fp == NULL) printf("ERROR: cannot open file scilabrssobol2b\n");
        else
        {
          snprintf(pString,100,"this file contains Sobol' 2nd order\n");
          fwriteComment(fp,pString);
          snprintf(pString,100,"indices. Set sortFlag = 1 and set nn to\n");
          fwriteComment(fp,pString);
          snprintf(pString,100,"be the number of inputs to display.\n");
          fwriteComment(fp,pString);
        }
        if (fp != NULL)
        {
          fprintf(fp, "sortFlag = 0;\n");
          fprintf(fp, "nn = %d;\n", nInps);
          fprintf(fp, "Means = [\n");
          for (ii = 0; ii < nInps*nInps; ii++)
            fprintf(fp,"%24.16e\n", tempT[nbs*nInps*nInps+ii]);
          fprintf(fp, "];\n");
          fprintf(fp, "Stds = [\n");
          for (ii = 0; ii < nInps*nInps; ii++)
            fprintf(fp,"%24.16e\n", tempT[ii]);
          fprintf(fp, "];\n");
          fprintf(fp, "Lows = [\n");
          for (ii = 0; ii < nInps*nInps; ii++)
            fprintf(fp,"%24.16e\n", tempT[(nbs+1)*nInps*nInps+ii]);
          fprintf(fp, "];\n");
          fprintf(fp, "Highs = [\n");
          for (ii = 0; ii < nInps*nInps; ii++)
            fprintf(fp,"%24.16e\n", tempT[(nbs+2)*nInps*nInps+ii]);
          fprintf(fp, "];\n");
          if (plotScilab()) fprintf(fp, "Str = [");
          else              fprintf(fp, "Str = {");
          for (ii = 0; ii < nInps-1; ii++)
            fprintf(fp,"'%s',",currSession->inputNames_[ii]);
          if (plotScilab()) 
               fprintf(fp,"'%s'];\n",currSession->inputNames_[nInps-1]);
          else fprintf(fp,"'%s'};\n",currSession->inputNames_[nInps-1]);
          fwriteHold(fp, 0);
          fprintf(fp, "ymin = min(Means-Stds);\n");
          fprintf(fp, "ymax = max(Means+Stds);\n");
          fprintf(fp, "ymin = min(Lows);\n");
          fprintf(fp, "ymax = max(Highs);\n");
          fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
          if (plotScilab())
          {
            fprintf(fp, "nn    = %d;\n",nInps);
            fprintf(fp, "Means = matrix(Means, nn, nn);\n");
            fprintf(fp, "Means = Means';\n");
            fprintf(fp, "Stds  = matrix(Stds, nn, nn);\n");
            fprintf(fp, "Stds  = Stds';\n");
            fprintf(fp, "Lows  = matrix(Lows, nn, nn);\n");
            fprintf(fp, "Lows  = Lows';\n");
            fprintf(fp, "Highs = matrix(Highs, nn, nn);\n");
            fprintf(fp, "Highs = Highs';\n");
            fprintf(fp, "drawlater\n");
            fprintf(fp, "hist3d(Means);\n");
            fprintf(fp, "set(gca(),\"auto_clear\",\"off\")\n");
            fprintf(fp, "a=gca();\n");
            fprintf(fp, "a.data_bounds=[0, 0, 0; nn, nn+1, ymax];\n");
            fprintf(fp, "newtick = a.x_ticks;\n");
            fprintf(fp, "newtick(2) = [1:nn]';\n");
            fprintf(fp, "newtick(3) = Str';\n");
            fprintf(fp, "a.x_ticks = newtick;\n");
            fprintf(fp, "a.x_label.font_size = 3;\n");
            fprintf(fp, "a.x_label.font_style = 4;\n");
            fprintf(fp, "a.y_ticks = newtick;\n");
            fprintf(fp, "a.y_label.font_size = 3;\n");
            fprintf(fp, "a.y_label.font_style = 4;\n");
            fprintf(fp, "a.rotation_angles = [5 -70];\n");
            fprintf(fp, "drawnow\n");
          }
          else
          {
            fprintf(fp, "nn    = %d;\n",nInps);
            fprintf(fp, "Means = reshape(Means, nn, nn);\n");
            fprintf(fp, "Means = Means';\n");
            fprintf(fp, "Stds  = reshape(Stds, nn, nn);\n");
            fprintf(fp, "Stds  = Stds';\n");
            fprintf(fp, "Lows  = reshape(Lows, nn, nn);\n");
            fprintf(fp, "Lows  = Lows';\n");
            fprintf(fp, "Highs = reshape(Highs, nn, nn);\n");
            fprintf(fp, "Highs = Highs';\n");
            fprintf(fp, "hh = bar3(Means,0.8);\n");
            fprintf(fp, "alpha = 0.2;\n");
            fprintf(fp, "set(hh,'FaceColor','b','facea',alpha);\n");
            fprintf(fp, "Lstds = Means - Stds;\n");
            fprintf(fp, "Ustds = Means + Stds;\n");
            fprintf(fp, "Lstds = Lows;\n");
            fprintf(fp, "Ustds = Highs;\n");
            fprintf(fp, "[X,Y] = meshgrid(1:nn,1:nn);\n");
            fwriteHold(fp, 1);
            fprintf(fp,"for k = 1:nn\n");
            fprintf(fp,"  for l = k:nn\n");
            fprintf(fp,"    mkl = Means(k,l);\n");
            fprintf(fp,"    ukl = Ustds(k,l);\n");
            fprintf(fp,"    lkl = Lstds(k,l);\n");
            fprintf(fp,"    if (mkl > .02 & (ukl-lkl)/mkl > .02)\n");
            fprintf(fp,"      xkl = [X(k,l), X(k,l)];\n");
            fprintf(fp,"      ykl = [Y(k,l), Y(k,l)];\n");
            fprintf(fp,"      zkl = [lkl, ukl];\n");
            fprintf(fp,"      plot3(xkl,ykl,zkl,'-mo',...\n");
            fprintf(fp,"      'LineWidth',5,'MarkerEdgeColor','k',...\n");
            fprintf(fp,"      'MarkerFaceColor','k','MarkerSize',10);\n");
            fprintf(fp,"    end\n");
            fprintf(fp,"  end\n");
            fprintf(fp,"end\n");
            fwriteHold(fp, 0);
            fprintf(fp, "axis([0.5 nn+0.5 0.5 nn+0.5 0 ymax])\n");
            fprintf(fp, "set(gca,'XTickLabel',Str);\n");
            fprintf(fp, "set(gca,'YTickLabel',Str);\n");
            fprintf(fp, "set(gca, 'fontsize', 12)\n");
            fprintf(fp, "set(gca, 'fontweight', 'bold')\n");
            fprintf(fp, "set(gca, 'linewidth', 2)\n");
          }
          fwritePlotAxes(fp);
          fwritePlotTitle(fp,"Sobol 1st+2nd Order Indices (with bootstrap)");
          fwritePlotZLabel(fp, "Sobol Indices (Normalized)");
          fwritePlotXLabel(fp, "Inputs");
          fwritePlotYLabel(fp, "Inputs");
          fclose(fp);
          if (plotScilab())
               printf("rssobol2b plot file = scilabrssobol2b.sci\n");
          else printf("rssobol2b plot file = matlabrssobol2b.m\n");
        }
      }
      //**/ restore previous settings
      if (saveRS == PSUADE_RS_MARSB)
        psuadeIO_->updateAnalysisSection(-1,-1,saveRS,-3,-1,-1);
      psuadeIO_->updateAnalysisSection(-1,-1,-1,saveDiag,-1,-1);
      delete anaManager;
      delete [] tempT;
    }

    //**/ ----------------------------------------------------------
    // +++ rssoboltsib 
    //**/ rssoboltsi with bootstrap
    //**/ ----------------------------------------------------------
    else if (!strcmp(command, "rssoboltsib"))
    {
      if (mypid == 0)
      {
        strcpy(winput,"psuadeOkay");
        strLeng = 11;
        winput[strLeng-1] = '\0';
        sscanf(lineIn,"%s %s",command,winput);
        if (!strcmp(winput, "-h"))
        {
          printf("rssoboltsib: RS-based Sobol' sensitivity analysis\n");
          printf("syntax: rssoboltsib (no argument needed)\n");
          printf("Note: This command uses bootstrapped samples\n");
          printf("      multiple times to get the errors in\n");
          printf("      Sobol' index estimation.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        if (currSession->nInputs_ <= 0 || psuadeIO_ == NULL)
        {
          printf("ERROR: no data (load data first).\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        if (currSession->nInputs_ <= 2)
        {
          printf("INFO: this analysis is not suitable for nInputs <= 2.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        if (currSession->nSamples_ < 5)
        {
          printf("ERROR: This command is not suitable for small\n");
          printf("       samples. nSamples needs to be at least 5.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
      }
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) winput, strLeng, CHAR, 0);
      if (!strcmp(winput, "psuadeNoOp")) continue;
      //**/ get output information
      if (mypid == 0)
      {
        printf("rssoboltsib INFO: MAKE SURE YOU SET YOUR DESIRED\n");
        printf("          RESPONSE SURFACE TYPE IN YOUR DATA FILE.\n");
        snprintf(pString,100,"Enter output number (1 - %d) : ", 
                 currSession->nOutputs_);
        outputID = getInt(1, currSession->nOutputs_, pString);
        outputID--;
      }
      psCommMgr_->bcast((void *) &outputID, iOne, INT, 0);

      //**/ set up analysis manager
      int analysisMethod = PSUADE_ANA_RSSOBOLTSI;
      AnalysisManager *anaManager = new AnalysisManager();
      anaManager->setup(analysisMethod, 0);

      //**/ save parameters resident in psuadeIO (to be restored later)
      psuadeIO_->getParameter("ana_diagnostics",pPtr);
      int saveDiag = pPtr.intData_;
      psuadeIO_->updateAnalysisSection(-1,-1,-1,-2,-1,-1);

      //**/ if rstype = MarsBagg, change to MARS
      psuadeIO_->getParameter("ana_rstype", pPtr);
      int saveRS = pPtr.intData_;
      if (saveRS == PSUADE_RS_MARSB)
      {
        if (mypid == 0)
        {
          printf("rssobol1b INFO: MarsBagg response surface selected\n");
          printf("          but it is redundant - set to MARS.\n");
        }
        psuadeIO_->updateAnalysisSection(-1,-1,PSUADE_RS_MARS,-3,-1,-1);
      }

      //**/ need to turn off expert modes, so save them first 
      psConfig_.AnaExpertModeSaveAndReset();
      psConfig_.RSExpertModeSaveAndReset();
      if (mypid == 0)
      {
        printf("rssoboltsib INFO: RS expert mode disabled. You can use\n");
        printf("            config file to set special RS options.\n");
      }

      //**/ set up storage for the response surface samples
      int    nInps = currSession->nInputs_;
      int    nOuts = currSession->nOutputs_;
      int    nSams = currSession->nSamples_;
      int    *tempI = new int[nSams];
      int    *states = new int[nSams];
      int    nbs, nSams2, ind;
      double *tempX = new double[nSams*nInps];
      double *tempY = new double[nSams];
      double *tempT, *tempM, *tempV;
      double *SamInps = currSession->vecSamInputs_.getDVector();
      double *SamOuts = currSession->vecSamOutputs_.getDVector();
      pData  *pdata=NULL;

      //**/ set up for iterations
      if (mypid == 0)
      {
        snprintf(pString,100,
                "How many bootstrapped samples to use (10 - 300) : ");
        nbs = getInt(10, 300, pString);
        if (nbs < nprocs)
        {
          nbs = nprocs;
          printf("INFO: number of bootstraps set to %d\n",nbs);
        }
        if (nbs % nprocs != 0) nbs = (nbs / nprocs + 1) * nprocs;
      }
      psCommMgr_->bcast((void *) &nbs, iOne, INT, 0);
      tempT = new double[nbs*nInps];

      //**/ iterate
      initializePrintingTS(-1, NULL, -1);
      for (kk = 0; kk < nbs; kk++)
      {
        if (kk % nprocs == mypid)
        {
          printf("Proc %5d: rssoboltsib iteration = %d (of %d)\n",
                 mypid,kk+1,nbs);
          //**/ random draw
          for (jj = 0; jj < nSams; jj++) tempI[jj] = 0;
          ss = nSams2 = 0;
          while (ss < nSams)
          {
            ind = PSUADE_rand() % nSams;
            if (tempI[ind] == 0)
            {
              for (ii = 0; ii < nInps; ii++)
                tempX[nSams2*nInps+ii] = SamInps[ind*nInps+ii];
              tempY[nSams2] = SamOuts[ind*nOuts+outputID];
              states[nSams2] = currSession->vecSamStates_[ind];
              tempI[ind] = 1;
              nSams2++;
            }
            ss++;
          }
          psuadeIO_->updateInputSection(nSams2,nInps,NULL,NULL,
                               NULL,tempX,NULL,NULL,NULL,NULL,NULL);
          psuadeIO_->updateOutputSection(nSams2,1,tempY,states,
                               currSession->outputNames_.getStrings());
          psuadeIO_->updateMethodSection(PSUADE_SAMP_MC,nSams2,
                               -1,-1,-1);
      
          //**/ analyze the result
          anaManager->analyze(psuadeIO_, 0, NULL, 0);
          pdata = psuadeIO_->getAuxData();
          commFlag = 0;
          if (pdata->nDbles_ != nInps) commFlag = 1;
          psCommMgr_->bcast((void *) &commFlag, iOne, INT, 0);
          if (commFlag == 1)
          {
            if (mypid == 0)
            {
              printf("ERROR: nInputs do not match.\n");
              printf("       Consult PSUADE developers.\n");
            }
            delete [] tempT;
            delete [] tempX;
            delete [] tempY;
            delete [] tempI;
            delete [] states;
            return -1;
          }
          //**/ get the statistics
          if (pdata->dbleData_ > 0)
            for (ii = 0; ii < nInps; ii++)
              tempT[kk*nInps+ii] =
                          pdata->dbleArray_[ii]/pdata->dbleData_;
          else
            for (ii = 0; ii < nInps; ii++)
              tempT[kk*nInps+ii] = pdata->dbleArray_[ii];

          //**/ clean up
          pdata->clean();
        }
        else
        {
          for (jj = 0; jj < nSams; jj++) ind = PSUADE_rand();
        }
      }
      delete [] tempX;
      delete [] tempY;
      delete [] tempI;
      delete [] states;

      commFlag = 1;
      psCommMgr_->bcast((void *) &commFlag, iOne, INT, 0);
      for (kk = 0; kk < nbs; kk++)
      {
        if (mypid == 0 && kk % nprocs != mypid)
        {
          proc = kk % nprocs;
          psCommMgr_->recv((void *) &tempT[kk*nInps],nInps,DOUBLE,kk,proc);
        }
        if (mypid != 0 && kk % nprocs == mypid)
        {
          psCommMgr_->send((void *) &tempT[kk*nInps],nInps,DOUBLE,kk,0);
        }
      }
      psConfig_.AnaExpertModeRestore();
      psConfig_.RSExpertModeRestore();
      if (mypid == 0)
      {
        tempM = new double[nInps];
        for (ii = 0; ii < nInps; ii++)
        {
          tempM[ii] = tempT[ii];
          for (jj = 1; jj < nbs; jj++) tempM[ii] += tempT[jj*nInps+ii];
          tempM[ii] /= (double) nbs;
        }
        tempV = new double[nInps];
        for (ii = 0; ii < nInps; ii++)
        {
          tempV[ii] = pow(tempT[ii]-tempM[ii], 2.0);
          for (jj = 1; jj < nbs; jj++)
            tempV[ii] += pow(tempT[jj*nInps+ii]-tempM[ii],2.0);
          tempV[ii] /= (double) (nbs - 1);
          tempV[ii] = sqrt(tempV[ii]);
        }
        printEquals(PL_INFO, 0);
        printf("SobolTSI Statistics (based on %d replications): \n",nbs);
        for (ii = 0; ii < nInps; ii++)
          printf("Input %4d: mean = %16.8e, std = %16.8e\n",ii+1,
                 tempM[ii],tempV[ii]);

        //**/ generate matlab/scilab file
        if (plotScilab()) fp = fopen("scilabrssoboltsib.sci","w");
        else              fp = fopen("matlabrssoboltsib.m","w");
        if (fp == NULL) printf("ERROR: cannot open plot file.\n");
        else
        {
          strcpy(pString," This file contains total order Sobol' indices");
          fwriteComment(fp, pString);
          strcpy(pString," with error bars coming from bootstrapping.");
          fwriteComment(fp, pString);
          strcpy(pString," to select the most important ones to display,");
          fwriteComment(fp, pString);
          strcpy(pString," set sortFlag = 1 and set nn to be the number");
          fwriteComment(fp, pString);
          strcpy(pString," of inputs to display.\n");
          fwriteComment(fp, pString);
          fprintf(fp, "sortFlag = 0;\n");
          fprintf(fp, "nn = %d;\n", nInps);
          fprintf(fp, "Means = [\n");
          for (ii = 0; ii < nInps; ii++) fprintf(fp,"%24.16e\n",tempM[ii]);
          fprintf(fp, "];\n");
          fprintf(fp, "Stds = [\n");
          for (ii = 0; ii < nInps; ii++) fprintf(fp,"%24.16e\n",tempV[ii]);
          fprintf(fp, "];\n");
          if (plotScilab()) fprintf(fp, "  Str = [");
          else              fprintf(fp, "  Str = {");
          for (ii = 0; ii < nInps-1; ii++)
            fprintf(fp,"'%s',",currSession->inputNames_[ii]);
          if (plotScilab()) 
               fprintf(fp,"'%s'];\n",currSession->inputNames_[nInps-1]);
          else fprintf(fp,"'%s'};\n",currSession->inputNames_[nInps-1]);
          fwriteHold(fp, 0);
          fprintf(fp, "if (sortFlag == 1)\n");
          if (plotScilab())
               fprintf(fp, "  [Means, I2] = gsort(Means);\n");
          else fprintf(fp, "  [Means, I2] = sort(Means,'descend');\n");
          fprintf(fp, "  Stds = Stds(I2);\n");
          fprintf(fp, "  I2 = I2(1:nn);\n");
          fprintf(fp, "  Means = Means(1:nn);\n");
          fprintf(fp, "  Stds = Stds(1:nn);\n");
          fprintf(fp, "  Str  = Str(I2);\n");
          fprintf(fp, "end\n");
          fprintf(fp, "ymin = min(Means-Stds);\n");
          fprintf(fp, "ymax = max(Means+Stds);\n");
          fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
          if (plotScilab()) fprintf(fp, "drawlater\n");
          fprintf(fp, "bar(Means,0.8);\n");
          fprintf(fp, "for ii = 1:nn\n");
          fprintf(fp, "   if (ii == 1)\n");
          fwriteHold(fp, 1);
          fprintf(fp, "   end;\n");
          fprintf(fp, "   XX = [ii ii];\n");
          fprintf(fp, "   d1 = Means(ii)-Stds(ii);\n");
          fprintf(fp, "   d2 = Means(ii)+Stds(ii);\n");
          fprintf(fp, "   if (d1 < 0)\n");
          fprintf(fp, "      d1 = 0.0;\n");
          fprintf(fp, "   end;\n");
          fprintf(fp, "   YY = [d1 d2];\n");
          fprintf(fp, "   plot(XX,YY,'-ko','LineWidth',3.0,");
          fprintf(fp, "'MarkerEdgeColor',");
          fprintf(fp, "'k','MarkerFaceColor','g','MarkerSize',13)\n");
          fprintf(fp, "end;\n");
          fwritePlotAxes(fp);
          if (plotScilab())
          {
            fprintf(fp,"a=gca();\n");
            fprintf(fp,"a.data_bounds=[0, ymin; nn+1, ymax];\n");
            fprintf(fp,"newtick = a.x_ticks;\n");
            fprintf(fp,"newtick(2) = [1:nn]';\n");
            fprintf(fp,"newtick(3) = Str';\n");
            fprintf(fp,"a.x_ticks = newtick;\n");
            fprintf(fp,"a.x_label.font_size = 3;\n");
            fprintf(fp,"a.x_label.font_style = 4;\n");
          }
          else
          {
            fprintf(fp,"axis([0  nn+1 ymin ymax])\n");
            fprintf(fp,"set(gca,'XTickLabel',[]);\n");
            fprintf(fp,"th=text(1:nn, repmat(ymin-0.05*(ymax-ymin)");
            fprintf(fp,",nn,1),Str,");
            fprintf(fp,"'HorizontalAlignment','left','rotation',90);\n");
            fprintf(fp,"set(th, 'fontsize', 12)\n");
            fprintf(fp,"set(th, 'fontweight', 'bold')\n");
          }
          fwritePlotTitle(fp,"Total Order Sobol Indices (with bootstrap)");
          fwritePlotYLabel(fp,"Total Order Sobol Index (Normalized)");
          if (plotScilab())
          {
            fprintf(fp, "drawnow\n");
            printf("rssoboltsib plot file = scilabrssoboltsib.sci\n");
          }
          else printf("rssoboltsib plot file = matlabrssoboltsib.m\n");
          fclose(fp);
        }
        delete [] tempM;
        delete [] tempV;
      }

      //**/ clean up
      delete anaManager;
      delete [] tempT;

      //**/ restore previous settings
      if (saveRS == PSUADE_RS_MARSB)
        psuadeIO_->updateAnalysisSection(-1,-1,saveRS,-3,-1,-1);
      psuadeIO_->updateAnalysisSection(-1,-1,-1,saveDiag,-1,-1);
    }

    //**/ -------------------------------------------------------------
    // +++ rskrig 
    //**/ check quality of the response surface using Kriging 
    //**/ -------------------------------------------------------------
    else if (!strcmp(command, "rskrig"))
    {
      if (mypid == 0)
      {
        strcpy(winput,"psuadeOkay");
        strLeng = 11;
        winput[strLeng-1] = '\0';
        sscanf(lineIn,"%s %s",command,winput);
        if (!strcmp(winput, "-h"))
        {
          printf("rskrig: create Kriging response surface.\n");
          printf("syntax: rskrig (no argument needed)\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        if (currSession->nInputs_ <= 0 || psuadeIO_ == NULL)
        {
          printf("ERROR: no data to analyze (load data first).\n");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
      }
      else psConfig_.InteractiveSaveAndReset();

      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) winput, strLeng, CHAR, 0);
      if (!strcmp(winput, "psuadeNoOp")) continue;
      PKriging *pkri = new PKriging(currSession->nInputs_,
                               currSession->nSamples_,psCommMgr_); 
      pkri->setOutputLevel(outputLevel_);
      pkri->initialize(currSession->vecSamInputs_.getDVector(),
                       currSession->vecSamOutputs_.getDVector());
      delete pkri;
      if (mypid != 0) psConfig_.InteractiveRestore();
    }

    //**/ -------------------------------------------------------------
    // +++ rscheck 
    //**/ check quality of the response surface 
    //**/ -------------------------------------------------------------
    else if (!strcmp(command, "rscheck") || 
             !strcmp(command, "rsvalidate"))
    {
      if (mypid == 0)
      {
        strcpy(winput,"psuadeOkay");
        strLeng = 11;
        winput[strLeng-1] = '\0';
        sscanf(lineIn,"%s %s",command,winput);
        if (!strcmp(winput, "-h"))
        {
          printf("rscheck: check the quality of the RS (training errors\n");
          printf("         and cross validation errors)\n");
          printf("syntax: rscheck (no argument needed)\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        if (currSession->nInputs_ <= 0 || psuadeIO_ == NULL)
        {
          printf("ERROR: no data to analyze (load data first).\n");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
      }
      else psConfig_.InteractiveSaveAndReset();
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) winput, strLeng, CHAR, 0);
      if (!strcmp(winput, "psuadeNoOp")) continue;
      if (mypid == 0)
      {
        faType = -1;
        snprintf(pString,100,"Enter your choice ? ");
        while (faType < 0 || faType > PSUADE_NUM_RS)
        {
          writeFAInfo(outputLevel_);
          faType = getFAType(pString);
        }
        snprintf(pString,100,"Enter output number (1 - %d) = ",
                 currSession->nOutputs_);
        outputID = getInt(1, currSession->nOutputs_, pString);
        outputID--;
      }
      psCommMgr_->bcast((void *) &faType, iOne, INT, 0);
      psCommMgr_->bcast((void *) &outputID, iOne, INT, 0);

      PRSFuncApproxAnalyzer *prs = new PRSFuncApproxAnalyzer(psCommMgr_);
      targc = 2;
      strcpy(pString, "rstype");
      targv[0] = (char *) pString;
      targv[1] = (char *) &faType;
      prs->setParams(targc, targv);

      //**/ save parameters resident in psuadeIO (to be restored later)
      psuadeIO_->getParameter("ana_diagnostics",pPtr);
      ii = pPtr.intData_;
      if (mypid == 0)
           psuadeIO_->updateAnalysisSection(-1,-1,-1,outputLevel_,-1,-1);
      else psuadeIO_->updateAnalysisSection(-1,-1,-1,-2,-1,-1);
      aPtr.outputID_ = outputID;
      aPtr.nInputs_ = currSession->nInputs_;
      aPtr.nOutputs_ = currSession->nOutputs_;
      aPtr.nSamples_ = currSession->nSamples_;
      aPtr.sampleInputs_ = currSession->vecSamInputs_.getDVector();
      aPtr.sampleOutputs_ = currSession->vecSamOutputs_.getDVector();
      aPtr.sampleStates_ = currSession->vecSamStates_.getIVector();
      aPtr.iLowerB_ = currSession->vecInpLBounds_.getDVector();
      aPtr.iUpperB_ = currSession->vecInpUBounds_.getDVector();
      aPtr.inputPDFs_ = NULL;
      aPtr.inputMeans_ = NULL;
      aPtr.inputStdevs_ = NULL;
      aPtr.ioPtr_ = psuadeIO_;
      if (mypid == 0) aPtr.printLevel_ = outputLevel_;
      else            aPtr.printLevel_ = 0;
      prs->analyze(aPtr);
      if (aPtr.sampleErrors_ != NULL) delete [] aPtr.sampleErrors_;
      aPtr.sampleErrors_ = NULL;
      psuadeIO_->updateAnalysisSection(-1,-1,-1,ii,-1,-1);
      delete prs;
      if (mypid != 0) psConfig_.InteractiveRestore();
    }

    //**/ -------------------------------------------------------------
    // +++ printlevel 
    //**/ set diagnostics output level
    //**/ -------------------------------------------------------------
    else if (!strcmp(command, "printlevel"))
    {
      if (mypid == 0)
      {
        sscanf(lineIn,"%s %s",command,winput);
        if (!strcmp(winput, "-h"))
        {
          printf("printlevel: set printlevel (0 - 5)\n");
          strcpy(winput, "psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        else
        {
          strcpy(winput, "psuadeOkay");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
      }
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) winput, strLeng, CHAR, 0);
      if (!strcmp(winput, "psuadeNoOp")) continue;

      if (mypid == 0)
      {
        sscanf(lineIn, "%s %d", command, &ii);
        if (ii >= PL_MIN && ii <= PL_MAX)
        {
          outputLevel_ = ii;
          printf("psuade printlevel set to %d\n", ii);
        }
        else
        {
          printf("printlevel out of range; set to %d\n", PL_INTERACTIVE);
          outputLevel_ = PL_INTERACTIVE;
        }
      }
      psCommMgr_->bcast((void *) &outputLevel_, iOne, INT, 0);
      setPrintLevelTS(outputLevel_);
    }

    //**/ -------------------------------------------------------------
    // +++ io_expert 
    //**/ set IO expert mode
    //**/ -------------------------------------------------------------
    else if (!strcmp(command, "io_expert"))
    {
      if (mypid == 0)
      {
        sscanf(lineIn,"%s %s",command,winput);
        if (!strcmp(winput, "-h"))
        {
          printf("This command turns on/off I/O expert mode.\n");
          printf("This mode, when turned on, will trigger prompts for\n");
          printf("additonal information regarding data input/output.\n");
          printf("When this mode is off, default setting will be used.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        else
        {
          strLeng = strlen(winput);
          strLeng++;
          winput[strLeng-1] = '\0';
        }
      }
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) winput, strLeng, CHAR, 0);
      if (!strcmp(winput, "psuadeNoOp")) continue;
         
      if (!strcmp(winput, "on"))  psConfig_.IOExpertModeOff();
      if (!strcmp(winput, "off")) psConfig_.IOExpertModeOn();
      if (!psConfig_.IOExpertModeIsOn())
      {
        psConfig_.IOExpertModeOn();
        if (mypid == 0) printf("IO expert mode on\n");
      }
      else
      {
        psConfig_.IOExpertModeOff();
        if (mypid == 0) printf("IO expert mode off\n");
      }
    }

    //**/ -------------------------------------------------------------
    // +++ rs_expert 
    //**/ set response surface expert mode
    //**/ -------------------------------------------------------------
    else if (!strcmp(command, "rs_expert"))
    {
      if (mypid == 0)
      {
        sscanf(lineIn,"%s %s",command,winput);
        if (!strcmp(winput, "-h"))
        {
          printf("This command turns on/off response surface expert mode.\n");
          printf("This mode, when turned on, will trigger prompts for\n");
          printf("additional information regarding creation of response\n");
          printf("surfaces.\n");
          printf("When this mode is off, default setting will be used.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        else
        {
          strLeng = strlen(winput);
          strLeng++;
          winput[strLeng-1] = '\0';
        }
      }
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) winput, strLeng, CHAR, 0);
      if (!strcmp(winput, "psuadeNoOp")) continue;

      if (!strcmp(winput, "on"))  psConfig_.RSExpertModeOff();
      if (!strcmp(winput, "off")) psConfig_.RSExpertModeOn();
      if (!psConfig_.RSExpertModeIsOn())
      {
        psConfig_.RSExpertModeOn();
        if (mypid == 0) printf("response surface expert mode on\n");
      }
      else
      {
        psConfig_.RSExpertModeOff();
        if (mypid == 0) printf("response surface expert mode off\n");
      }
    }

    //**/ -------------------------------------------------------------
    // +++ ana_expert 
    //**/ set analysis expert mode
    //**/ -------------------------------------------------------------
    else if (!strcmp(command, "ana_expert"))
    {
      if (mypid == 0)
      {
        sscanf(lineIn,"%s %s",command,winput);
        if (!strcmp(winput, "-h"))
        {
          printf("This command turns on/off data analysis expert mode.\n");
          printf("This mode, when turned on, will trigger prompts for\n");
          printf("additional information regarding data analysis.\n");
          printf("When this mode is off, default setting will be used.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        else
        {
          strLeng = strlen(winput);
          strLeng++;
          winput[strLeng-1] = '\0';
        }
      }
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) winput, strLeng, CHAR, 0);
      if (!strcmp(winput, "psuadeNoOp")) continue;

      if (!strcmp(winput, "on"))  psConfig_.AnaExpertModeOff();
      if (!strcmp(winput, "off")) psConfig_.AnaExpertModeOn();
      if (!psConfig_.AnaExpertModeIsOn())
      {
        psConfig_.AnaExpertModeOn();
        if (mypid == 0) printf("analysis expert mode on\n");
      }
      else
      {
        psConfig_.AnaExpertModeOff();
        if (mypid == 0) printf("analysis expert mode off\n");
      }
    }

    //**/ -------------------------------------------------------------
    // +++ rsmax 
    //**/ set max data size 
    //**/ -------------------------------------------------------------
    else if (!strcmp(command, "rsmax"))
    {
      if (mypid == 0)
      {
        sscanf(lineIn,"%s %s",command,winput);
        if (!strcmp(winput, "-h"))
        {
          printf("rsmax: set maximum sample size for response surface\n"); 
          printf("     (used to guard against large sample sizes that\n");
          printf("     make response surface generation very expensive.)\n");
          printf("     The default max is 5000.\n");
          strcpy(winput,"psuadeNoOp");
          strLeng = 11;
          winput[strLeng-1] = '\0';
        }
        else
        {
          strLeng = strlen(winput);
          strLeng++;
          winput[strLeng-1] = '\0';
        }
      }
      psCommMgr_->bcast((void *) &strLeng, iOne, INT, 0);
      psCommMgr_->bcast((void *) winput, strLeng, CHAR, 0);
      if (!strcmp(winput, "psuadeNoOp")) continue;

      if (mypid == 0)
      {
        sscanf(lineIn, "%s %d", command, &kk);
        if (kk <= 1 || kk > 100000)
             psConfig_.RSMaxPts_ = 100000;
        else psConfig_.RSMaxPts_ = kk;
        printf("psuade max data size for response surface set to %d\n",
               psConfig_.RSMaxPts_);
      }
      psCommMgr_->bcast((void *) &psConfig_.RSMaxPts_, iOne, INT, 0);
    }

    //**/ -------------------------------------------------------------
    // +++ quit 
    //**/ quit psuade interactive session
    //**/ -------------------------------------------------------------
    else if ((!strcmp(command, "quit")) || (!strcmp(command, "q")))
    {
      break;
    }
    else if ((!strcmp(command, "exit")))
    {
      break;
    }
    else if (command[0] == '#')
    {
      if (mypid == 0) printf("#\n");
    }
  }

  //**/ ----------------------------------------------------------------
  // quit psuade interactive session
  //**/ ----------------------------------------------------------------
  if (currSession != NULL) delete currSession;
  currSession = NULL;
  return 0;
}

