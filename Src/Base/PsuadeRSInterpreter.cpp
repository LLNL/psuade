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
// DATE   : 2005
// ************************************************************************
//
// ------------------------------------------------------------------------
// system includes
// ------------------------------------------------------------------------
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
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
#include "PrintingTS.h"
#include "PsuadeConfig.h"
#include "psVector.h"
#include "ProbMatrix.h"
#include "MainEffectAnalyzer.h"
#include "TwoParamAnalyzer.h"
#include "RSMEntropy1Analyzer.h"
#include "RSMEntropy2Analyzer.h"
#include "RSMEntropyGAnalyzer.h"
#include "RSMSobol1Analyzer.h"
#include "RSMSobol2Analyzer.h"
#include "RSMSobolGAnalyzer.h"
#include "RSMSobolTSIAnalyzer.h"
#include "ShapleyAnalyzer.h"

// ------------------------------------------------------------------------
// local includes : function approximator and others
// ------------------------------------------------------------------------
#include "FuncApprox.h"
#include "pData.h"
#include "PDFManager.h"
#include "PDFNormal.h"

// ------------------------------------------------------------------------
// local defines 
// ------------------------------------------------------------------------
#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// interpret command from interactive session
// ------------------------------------------------------------------------
int PsuadeBase::RSBasedAnalysis(char *lineIn, PsuadeSession *session)
{
  int    ss, ii, jj, kk, status, outputID, flag, faType;
  int    nSamples, nInputs, nOutputs, *sampleStates=NULL;
  double ddata, *sampleInputs=NULL, *sampleOutputs=NULL; 
  double *iLowerB=NULL, *iUpperB=NULL;
  char   command[1001], winput[1001], pString[1001], dataFile[1001];
  char   lineIn2[1001], **inputNames=NULL, **outputNames=NULL;
  FILE   *fp=NULL;
  PsuadeData *psuadeIO=NULL;
  pData      pPtr;

  //**/ -------------------------------------------------------------
  // read in command and data from main interpreter 
  //**/ -------------------------------------------------------------
  winput[0] = '\0';
  sscanf(lineIn,"%s", command);
  //**/ session == NULL when help (-h) is issued
  if (session == NULL)
  {
    nSamples = nInputs = nOutputs = 0;
    sampleInputs = sampleOutputs = NULL;
    sampleStates = NULL;
    psuadeIO = NULL;
    inputNames = outputNames = NULL;
    iLowerB = iUpperB = NULL;
  }
  else
  {
    nSamples = session->nSamples_;
    nInputs = session->nInputs_;
    nOutputs = session->nOutputs_;
    sampleInputs  = session->vecSamInputs_.getDVector();
    sampleOutputs = session->vecSamOutputs_.getDVector();
    sampleStates  = session->vecSamStates_.getIVector();
    psuadeIO = (PsuadeData *) session->psuadeIO_;
    inputNames = session->inputNames_.getStrings();
    outputNames = session->outputNames_.getStrings();
    iLowerB = session->vecInpLBounds_.getDVector();
    iUpperB = session->vecInpUBounds_.getDVector();
  }
  
  //**/ -------------------------------------------------------------
  // +++ rsua 
  //**/ uncertainty analysis on response surface 
  //**/ RS uncertainties introduced by the use of stochastic RS
  //**/ Worst-case analysis (optional: average case analysis)
  // ----------------------------------------------------------------
  if (!strcmp(command, "rsua"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsua: uncertainty analysis on response surface\n");
      printf("Syntax: rsua (no argument needed)\n");
      printf("This command performs uncertainty analysis ");
      printf("using the response surface\n");
      printf("constructed from the LOADED sample. The ");
      printf("sample for forward propagation\n");
      printf("of uncertainties can be either a user-provided ");
      printf("sample in PSUADE data\n");
      printf("format or a PSUADE-generated sample. If a ");
      printf("stochastic response surface\n");
      printf("(e.g. Kriging, MARSB, or regression) is selected, ");
      printf("the RS uncertainties\n");
      printf("will also be shown in the PDF and CDF plots ");
      printf("produced by this command.\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }

    printAsterisks(PL_INFO, 0);
    printf("* Response surface-based Uncertainty Analysis\n");
    printDashes(PL_INFO, 0);
    printf("* To include response surface uncertainties, use ");
    printf("stochastic response\n");
    printf("* surface such as polynomial regression, MARSB, Kriging, ");
    printf(".. (specified\n");
    printf("* in your loaded data file).\n");
    printf("* Option: compute average or worst case RS uncertainties.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ query user for output ID
    sscanf(lineIn,"%s %s", command, winput);
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    //**/ request or generate a sample for evaluation
    printf("A sample is needed from you to propagate through the RS.\n");
    printf("Select between the two options below: \n");
    printf("1. PSUADE will generate the sample\n");
    printf("2. User will provide the sample (in PSUADE data format)\n");
    snprintf(pString,100,"Enter 1 or 2 : ");
    int    samSrc = getInt(1, 2, pString);
    int    uaNSams;
    psVector  vecUAInps, vecUAOuts;
    psIVector vecUAStas;
    if (samSrc == 1)
    {
      printf("PSUADE will generate a sample for uncertainty analysis.\n");
      snprintf(pString,100,"Sample size ? (10000 - 100000) ");
      uaNSams = getInt(10000, 100000, pString);
      vecUAInps.setLength(uaNSams * nInputs);
      psuadeIO->getParameter("ana_use_input_pdfs", pPtr);
      int usePDFs = pPtr.intData_;
      if (usePDFs == 1)
      {
        printf("NOTE: Some inputs have non-uniform PDFs.\n");
        printf("      A MC sample will be created with these PDFs.\n");
        psuadeIO->getParameter("method_sampling", pPtr);
        kk = pPtr.intData_;
        psuadeIO->updateMethodSection(PSUADE_SAMP_MC,-1,-1,-1,-1);
        PDFManager *pdfman = new PDFManager();
        pdfman->initialize(psuadeIO);
        vecUAInps.setLength(uaNSams*nInputs);
        psVector vecLs, vecUs;
        vecUs.load(nInputs, iUpperB);
        vecLs.load(nInputs, iLowerB);
        pdfman->genSample(uaNSams, vecUAInps, vecLs, vecUs);
        psuadeIO->updateMethodSection(kk,-1,-1,-1,-1);
        delete pdfman;
      }
      else
      {
        printAsterisks(PL_INFO, 0);
        printf("NOTE: Uniform distribution is assumed for all inputs. ");
        printf("To use other\n");
        printf("      than uniform distributions, prescribe them in ");
        printf("the sample file\n");
        printf("      and set use_input_pdfs in the ANALYSIS section.\n");
        printAsterisks(PL_INFO, 0);
        Sampling *samPtr;
        if (nInputs < 51)
             samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LPTAU);
        else samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
        samPtr->setPrintLevel(0);
        samPtr->setInputBounds(nInputs, iLowerB, iUpperB);
        samPtr->setOutputParams(1);
        samPtr->setSamplingParams(uaNSams, -1, -1);
        samPtr->initialize(0);
        vecUAOuts.setLength(uaNSams);
        vecUAStas.setLength(uaNSams);
        samPtr->getSamples(uaNSams,nInputs,1,vecUAInps.getDVector(),
                     vecUAOuts.getDVector(),vecUAStas.getIVector());
        delete samPtr;
      }
    }
    else if (samSrc == 2)
    {
      printf("Enter UA sample file name (in PSUADE data format): ");
      char uaFileName[1001];
      scanf("%s", uaFileName);
      fgets(lineIn2, 500, stdin);
      PsuadeData *sampleIO = new PsuadeData();
      status = sampleIO->readPsuadeFile(uaFileName);
      if (status != 0)
      {
        printf("ERROR: Cannot read sample file.\n");
        delete sampleIO;
        return 1;
      }
      sampleIO->getParameter("input_ninputs", pPtr);
      kk = pPtr.intData_;
      if (kk != nInputs)
      {
        printf("ERROR: Sample nInputs mismatch.\n");
        printf(":      Input size in workspace     = %d.\n",nInputs);
        printf(":      Input size from your sample = %d.\n",kk);
        delete sampleIO;
        return 1;
      }
      sampleIO->getParameter("method_nsamples", pPtr);
      uaNSams = pPtr.intData_;
      if (uaNSams < 1000)
      {
        printf("ERROR: Your sample size should be at least 1000 to give\n");
        printf("       any reasonable UA results.\n");
        delete sampleIO;
        return 1;
      }
      sampleIO->getParameter("input_sample", pPtr);
      vecUAInps.load(uaNSams * nInputs, pPtr.dbleArray_);
      pPtr.clean();
      delete sampleIO;
    }

    //**/ ====================================================================
    // ask which method to us
    //**/ ====================================================================
    printf("The default is to perform the average case analysis (1): \n");
    printf(" - For each sample point, evaluation using stochastic ");
    printf("RS gives a mean\n");
    printf("   and a std deviation. Average case analysis take ");
    printf("these quantities\n");
    printf("   and creates a small sample for each sample point.  ");
    printf("Afterward, it\n");
    printf("   creates a probability distribution based on ");
    printf("this enlarged sample.\n");
    printf("However, you can also perform a worst case analysis (2): \n");
    printf(" - For each sample point, evaluation using stochastic ");
    printf("RS gives a mean\n");
    printf("   and a standard deviation. Worst case analysis takes the ");
    printf("max and min\n");
    printf("   at each sample point as the +/- 3 std dev. Afterward, ");
    printf("it creates a\n");
    printf("   probability distribution enveloped by the max/min ");
    printf("distributions.\n");
    snprintf(pString,100,
             "Enter 1 (average case) or 2 (worst case) analysis : ");
    int uaMethod = getInt(1,2,pString);
    uaMethod--;

    //**/ ====================================================================
    // perform UA
    //**/ ====================================================================
    psVector vecUAStds;
    vecUAOuts.setLength(uaNSams);
    vecUAStds.setLength(uaNSams);

    //**/ ----------------------------------------------
    // stochastic RS with average case analysis
    //**/ ----------------------------------------------
    FuncApprox *faPtrUA=NULL;
    //psConfig_.InteractiveSaveAndReset();
    if (uaMethod == 0)
    {
      //**/ create response surface
      printf("** CREATING RESPONSE SURFACE\n");
      faPtrUA = genFA(-1, nInputs, -1, nSamples);
      if (faPtrUA == NULL)
      {
        printf("ERROR: Cannot create response surface.\n");
        return 1;
      }
      faPtrUA->setBounds(iLowerB, iUpperB);
      faPtrUA->setOutputLevel(outputLevel_);
      psVector vecYOut;
      vecYOut.setLength(nSamples);
      for (ss = 0; ss < nSamples; ss++)
        vecYOut[ss] = sampleOutputs[ss*nOutputs+outputID];
      status = faPtrUA->initialize(sampleInputs,vecYOut.getDVector());
      if (status != 0)
      {
        printf("ERROR: Cannot initialize response surface.\n");
        if (faPtrUA != NULL) delete faPtrUA;
        return 1;
      }
      //**/ evaluate response surface
      printf("** EVALUATING RESPONSE SURFACE (fuzzy)\n");
      faPtrUA->evaluatePointFuzzy(uaNSams,vecUAInps.getDVector(),
                                  vecUAOuts.getDVector(),
                                  vecUAStds.getDVector());
      fp = fopen("rsua_sample","w");
      if (fp != NULL)
      {
        fprintf(fp,"%% This file is primarily for diagnostics and \n");
        fprintf(fp,"%% expert analysis\n");
        fprintf(fp,"%% First line: nSamples nInputs\n");
        fprintf(fp,"%% All inputs, output(Y), Y-3*sigma,Y+3*sigma\n");
        fprintf(fp,"%d %d 3\n", uaNSams, nInputs);
        for (ss = 0; ss < uaNSams; ss++)
        {
          for (ii = 0; ii < nInputs; ii++)
            fprintf(fp, "%e ", vecUAInps[ss*nInputs+ii]);
          fprintf(fp, "%e ", vecUAOuts[ss]);
          fprintf(fp, "%e ", vecUAOuts[ss]-3*vecUAStds[ss]);
          fprintf(fp, "%e\n", vecUAOuts[ss]+3*vecUAStds[ss]);
        }
        fclose(fp);
        printf("The outputs and std deviations of the evaluation ");
        printf("sample has been\n");
        printf("written into 'rsua_sample'.\n");
      }

      //**/ first set of statistics 
      double mean=0, stdev=0;
      for (ss = 0; ss < uaNSams; ss++) mean += vecUAOuts[ss];
      mean /= (double) uaNSams;
      for (ss = 0; ss < uaNSams; ss++)
        stdev += pow(vecUAOuts[ss] - mean, 2.0);
      stdev = sqrt(stdev/(double) uaNSams);
      printAsterisks(PL_INFO, 0);
      printf("Sample mean    = %e (RS uncertainties not included)\n", 
             mean);
      printf("Sample std dev = %e (RS uncertainties not included)\n", 
             stdev);
      printEquals(PL_INFO, 0);

      //**/ initialize for binning 
      int    nbins = 100, ntimes=20;
      int    **Fcounts = new int*[ntimes+1];
      double Fmax=-PSUADE_UNDEFINED, Fmin=PSUADE_UNDEFINED;
      for (ss = 0; ss < uaNSams; ss++)
      {
        if (vecUAOuts[ss]+3*vecUAStds[ss] > Fmax)
          Fmax = vecUAOuts[ss] + 3 * vecUAStds[ss];
        if (vecUAOuts[ss]-3*vecUAStds[ss] < Fmin)
          Fmin = vecUAOuts[ss] - 3 * vecUAStds[ss];
      }
      Fmax = Fmax + 0.1 * (Fmax - Fmin);
      Fmin = Fmin - 0.1 * (Fmax - Fmin);
      if (Fmax == Fmin)
      {
        Fmax = Fmax + 0.1 * PABS(Fmax);
        Fmin = Fmin - 0.1 * PABS(Fmin);
      }
      for (ii = 0; ii <= ntimes; ii++)
      {
        Fcounts[ii] = new int[nbins];
        for (kk = 0; kk < nbins; kk++) Fcounts[ii][kk] = 0;
      }

      //**/ generate stochastic RS and bin
      psVector vecSamOutTime, vecSamOutSave;
      vecSamOutTime.setLength(ntimes*nInputs);
      vecSamOutSave.setLength(ntimes*uaNSams);
      for (ss = 0; ss < uaNSams; ss++)
      {
        if (vecUAStds[ss] == 0)
        {
          for (ii = 0; ii < ntimes; ii++) 
            vecSamOutTime[ii] = vecUAOuts[ss];
        }
        else
        {
          ddata = 2.0 * vecUAStds[ss] / (ntimes - 1);
          for (ii = 0; ii < ntimes; ii++) 
            vecSamOutTime[ii] = vecUAOuts[ss]+ii*ddata-
                                vecUAStds[ss];
        }
        for (ii = 0; ii < ntimes; ii++) 
          vecSamOutSave[ss*ntimes+ii] = vecSamOutTime[ii];

        //**/ bin the original sample
        ddata = vecUAOuts[ss] - Fmin;
        if (Fmax > Fmin) ddata = ddata / ((Fmax - Fmin) / nbins);
        else             ddata = nbins / 2;
        kk = (int) ddata;
        if (kk < 0)      kk = 0;
        if (kk >= nbins) kk = nbins - 1;
        Fcounts[ntimes][kk]++;

        //**/ bin the perturbed sample
        for (ii = 0; ii < ntimes; ii++)
        {
          ddata = vecSamOutTime[ii] - Fmin;
          if (Fmax > Fmin)
               ddata = ddata / ((Fmax - Fmin) / nbins);
          else ddata = nbins / 2;
          kk = (int) ddata;
          if (kk < 0)      kk = 0;
          if (kk >= nbins) kk = nbins - 1;
          Fcounts[ii][kk]++;
        }
      }
      double mean2=0, stdev2=0;
      for (ss = 0; ss < uaNSams*ntimes; ss++) 
        mean2 += vecSamOutSave[ss];
      mean2 /= (double) (uaNSams*ntimes);
      stdev2 = 0.0;
      for (ss = 0; ss < uaNSams*ntimes; ss++)
        stdev2 += pow(vecSamOutSave[ss] - mean2, 2.0);
      stdev2 = sqrt(stdev2/(double) (uaNSams*ntimes));
      printf("Sample mean    = %e (RS uncertainties included)\n", 
             mean2);
      printf("Sample std dev = %e (RS uncertainties included)\n", 
             stdev2);
      printAsterisks(PL_INFO, 0);

      //**/ write to file
      double dsum = 0.0;
      for (ss = 0; ss < uaNSams; ss++) dsum += vecUAStds[ss];
      if (plotMatlab()) fp = fopen("matlabrsua.m", "w");
      else              fp = fopen("scilabrsua.sci", "w");
      if (fp == NULL)
      {
        printf("INFO: Cannot write the PDFs/CDFs to matlab file.\n");
      }
      else
      {
        fwriteHold(fp, 0);
        fprintf(fp, "X = [\n");
        for (kk = 0; kk < nbins; kk++)
          fprintf(fp, "%e\n",(Fmax-Fmin)/nbins*(0.5+kk)+Fmin);
        fprintf(fp, "];\n");
        for (ii = 0; ii <= ntimes; ii++)
        {
          fprintf(fp, "N%d = [\n", ii+1);
          for (kk = 0; kk < nbins; kk++)
            fprintf(fp, "%d\n",  Fcounts[ii][kk]);
          fprintf(fp, "];\n");
        }
        fprintf(fp, "N = [");
        for (ii = 0; ii <= ntimes; ii++)
          fprintf(fp, "N%d/sum(N%d) ", ii+1, ii+1);
        fprintf(fp, "];\n");
        fprintf(fp, "NA = N(:,%d+1);\n",ntimes);
        fprintf(fp, "NA = NA / sum(NA);\n");
        if (plotMatlab())
             fprintf(fp, "NB = sum(N(:,1:%d)');\n",ntimes);
        else fprintf(fp, "NB = sum(N(:,1:%d)',1);\n",ntimes);
        fprintf(fp, "NB = NB' / sum(NB);\n");
        fprintf(fp, "NN = [NA NB];\n");
        fprintf(fp, "subplot(2,2,1)\n");
        fprintf(fp, "bar(X,NA,1.0)\n");
        fprintf(fp, "xmin = min(X);\n");
        fprintf(fp, "xmax = max(X);\n");
        fprintf(fp, "ymin = min(min(NA),min(NB));\n");
        fprintf(fp, "ymax = max(max(NA),max(NB));\n");
        fwritePlotScales2D(fp);
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Prob. Dist. (means of RS)");
        fwritePlotXLabel(fp, "Output Value");
        fwritePlotYLabel(fp, "Probabilities)");
        if (plotMatlab())
        {
          fprintf(fp,"text(0.05,0.9,'Mean = %12.4e','sc','FontSize',11)\n",
                  mean);
          fprintf(fp,"text(0.05,0.85,'Std  = %12.4e','sc','FontSize',11)\n",
                  stdev);
        }
        if (dsum == 0.0)
        {
          printf("Deterministic RS used ==> no RS uncertainties.\n");
          fprintf(fp,"subplot(2,2,3)\n");
          fwritePlotAxes(fp);
          fwritePlotTitle(fp,"Prob. Dist. (RS with uncertainties)");
          if (plotMatlab())
          {
            fprintf(fp,"text(0.01,0.5,'No RS uncertainty info.','sc',");
            fprintf(fp,"'FontSize',11)\n");
            fprintf(fp,"text(0.01,0.4,'Deterministic RS used.','sc',");
            fprintf(fp,"'FontSize',11)\n");
          }
        }
        else
        {
          fprintf(fp,"subplot(2,2,3)\n");
          fprintf(fp,"bar(X,NB,1.0)\n");
          fwritePlotScales2D(fp);
          fwritePlotAxes(fp);
          fwritePlotTitle(fp,"Prob. Dist. (RS with uncertainties)");
          fwritePlotXLabel(fp,"Output Value");
          fwritePlotYLabel(fp,"Probabilities");
          if (plotMatlab())
          {
            fprintf(fp,"text(0.05,0.9,'Mean = %12.4e','sc','FontSize',11)\n",
                    mean2);
            fprintf(fp,"text(0.05,0.85,'Std  = %12.4e','sc','FontSize',11)\n",
                    stdev2);
          }
        }
        for (ii = 0; ii <= ntimes; ii++)
        {
          fprintf(fp,"for ii = 2 : %d\n", nbins);
          fprintf(fp,"  N%d(ii) = N%d(ii) + N%d(ii-1);\n",ii+1,ii+1,ii+1);
          fprintf(fp,"end;\n");
        }
        fprintf(fp, "N = [");
        for (ii = 0; ii <= ntimes; ii++)
          fprintf(fp,"N%d/N%d(%d) ", ii+1, ii+1, nbins);
        fprintf(fp, "];\n");
        if (plotMatlab()) fprintf(fp, "subplot(2,2,[2 4])\n");
        else              fprintf(fp, "subplot(1,2,2)\n");
        fprintf(fp, "NA = N(:,%d+1);\n",ntimes);
        fprintf(fp, "NA = NA / NA(%d);\n",nbins);
        if (plotMatlab()) 
             fprintf(fp, "NB = sum(N(:,1:%d)');\n",ntimes);
        else fprintf(fp, "NB = sum(N(:,1:%d)',1);\n",ntimes);
        fprintf(fp, "NB = NB' / NB(%d);\n", nbins);
        fprintf(fp, "NN = [NA NB];\n");
        if (dsum == 0.0)
        {
          fprintf(fp, "plot(X,NA,'linewidth',3)\n");
          fwritePlotTitle(fp,"Cum. Dist.: (b) mean; (g) with uncertainties");
        }
        else
        {
          fprintf(fp, "plot(X,NN,'linewidth',3)\n");
          fwritePlotTitle(fp,"Cum. Dist.: (*) uncertainties unavailable");
        }
        fwritePlotAxes(fp);
        fwritePlotXLabel(fp, "Output Value");
        fwritePlotYLabel(fp, "Probabilities");
        fclose(fp);
        if (plotMatlab())
          printf("Output distribution plots file = matlabrsua.m.\n");
        else
          printf("Output distribution plots file = scilabrsua.sci.\n");
      }
      for (ii = 0; ii <= ntimes; ii++) delete [] Fcounts[ii];
      delete [] Fcounts;
    }

    //**/ ----------------------------------------------
    // stochastic RS with worst case analysis
    //**/ ----------------------------------------------
    else if (uaMethod == 1)
    {
      //**/ create response surface
      printf("** CREATING RESPONSE SURFACE\n");
      faPtrUA = genFA(-1, nInputs, -1, nSamples);
      if (faPtrUA == NULL)
      {
        printf("ERROR: Cannot generate response surface.\n");
        return 1;
      }
      faPtrUA->setBounds(iLowerB, iUpperB);
      faPtrUA->setOutputLevel(0);
      psVector vecYOut;
      vecYOut.setLength(nSamples);
      for (ss = 0; ss < nSamples; ss++)
        vecYOut[ss] = sampleOutputs[ss*nOutputs+outputID];
      status = faPtrUA->initialize(sampleInputs,vecYOut.getDVector());
      if (status != 0)
      {
        printf("ERROR: Cannot initialize response surface.\n");
        if (faPtrUA != NULL) delete faPtrUA;
        return 1;
      }
       
      //**/ create response surface
      faPtrUA->evaluatePointFuzzy(uaNSams,vecUAInps.getDVector(),
                   vecUAOuts.getDVector(),vecUAStds.getDVector());
      
      //**/ first set of statistics 
      double mean=0, stdev=0;
      for (ss = 0; ss < uaNSams; ss++) mean += vecUAOuts[ss];
      mean /= (double) uaNSams;
      for (ss = 0; ss < uaNSams; ss++)
         stdev += pow(vecUAOuts[ss]-mean, 2.0);
      stdev = sqrt(stdev/(double) uaNSams);
      printAsterisks(PL_INFO, 0);
      printf("Sample mean    = %e (RS uncertainties not included)\n",
             mean);
      printf("Sample std dev = %e (RS uncertainties not included)\n",
             stdev);
      printEquals(PL_INFO, 0);

      fp = fopen("rsua_sample","w");
      fprintf(fp,"%% This file is primarily for diagnostics and \n");
      fprintf(fp,"%% expert analysis\n");
      fprintf(fp,"%% First line: nSamples nInputs\n");
      fprintf(fp,"%% All inputs, output(Y), Y-3*sigma, Y+3*sigma\n");
      fprintf(fp,"%d %d 3\n", uaNSams, nInputs);
      for (ss = 0; ss < uaNSams; ss++)
      {
        for (ii = 0; ii < nInputs; ii++)
          fprintf(fp, "%e ", vecUAInps[ss*nInputs+ii]);
        fprintf(fp,"%e ", vecUAOuts[ss]);
        fprintf(fp,"%e ", vecUAOuts[ss]-3*vecUAStds[ss]);
        fprintf(fp,"%e\n", vecUAOuts[ss]+3*vecUAStds[ss]);
      }
      fclose(fp);
      printf("The outputs and std deviations of the evaluation ");
      printf("sample has been\n");
      printf("written into 'rsua_sample'.\n");
      
      //**/ initialize for binning 
      int    nbins = 100, ntimes=7;
      int    **Fcounts = new int*[ntimes+1];
      double Fmax=-PSUADE_UNDEFINED;
      double Fmin=PSUADE_UNDEFINED;
      PDFNormal *rsPDF=NULL;
      for (ss = 0; ss < uaNSams; ss++)
      {
        if (vecUAOuts[ss]+3*vecUAStds[ss] > Fmax)
           Fmax = vecUAOuts[ss] + 3 * vecUAStds[ss];
        if (vecUAOuts[ss]-3*vecUAStds[ss] < Fmin)
           Fmin = vecUAOuts[ss] - 3 * vecUAStds[ss];
      }
      Fmax = Fmax + 0.1 * (Fmax - Fmin);
      Fmin = Fmin - 0.1 * (Fmax - Fmin);
      if (Fmax == Fmin)
      {
        Fmax = Fmax + 0.1 * PABS(Fmax);
        Fmin = Fmin - 0.1 * PABS(Fmin);
      }
      for (ii = 0; ii <= ntimes; ii++)
      {
        Fcounts[ii] = new int[nbins];
        for (kk = 0; kk < nbins; kk++) Fcounts[ii][kk] = 0;
      }

      //**/ binning 
      double dsum = 0.0;
      for (ss = 0; ss < uaNSams; ss++)
      {
        for (ii = 0; ii < ntimes; ii++)
        {
          ddata = vecUAOuts[ss]+vecUAStds[ss]*(ii-3) - Fmin;
          if (Fmax > Fmin)
               ddata = ddata / ((Fmax - Fmin) / nbins);
          else ddata = nbins / 2;
          kk = (int) ddata;
          if (kk < 0)      kk = 0;
          if (kk >= nbins) kk = nbins - 1;
          Fcounts[ii][kk]++;
        }
        dsum += vecUAStds[ss];
      }

      if (plotMatlab()) fp = fopen("matlabrsua.m", "w");
      else              fp = fopen("scilabrsua.sci", "w");
      if (fp == NULL)
      {
        printf("INFO: Cannot write the PDFs/CDFs to matlab/scilab file.\n");
      }
      else
      {
        fwriteHold(fp, 0);
        strcpy(pString, "worst case analysis\n");
        fwriteComment(fp, pString);
        fprintf(fp, "X = [\n");
        for (kk = 0; kk < nbins; kk++)
          fprintf(fp, "%e\n", (Fmax-Fmin)/nbins*(0.5+kk)+Fmin);
        fprintf(fp, "];\n");
        for (ii = 0; ii < ntimes; ii++)
        {
          fprintf(fp, "E%d = [\n", ii+1);
          for (kk = 0; kk < nbins; kk++) 
            fprintf(fp, "%d\n",  Fcounts[ii][kk]);
          fprintf(fp, "];\n");
        }
        fprintf(fp, "EE = [");
        for (ii = 0; ii < ntimes; ii++)
          fprintf(fp, "E%d/sum(E%d) ", ii+1, ii+1);
        fprintf(fp, "];\n");
        fprintf(fp, "subplot(2,2,1)\n");
        fprintf(fp, "bar(X,EE(:,4),1.0)\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Prob. Dist. (means of RS)");
        fwritePlotXLabel(fp, "Output Value");
        fwritePlotYLabel(fp, "Probabilities)");
        if (plotMatlab())
        {
          fprintf(fp,"text(0.05,0.9,'Mean = %12.4e','sc','FontSize',11)\n",
                  mean);
          fprintf(fp,"text(0.05,0.85,'Std  = %12.4e','sc','FontSize',11)\n",
                  stdev);
        }
        fprintf(fp,"xmin = min(X);\n");
        fprintf(fp,"xmax = max(X);\n");
        fprintf(fp,"ymin = min(min(EE));\n");
        fprintf(fp,"ymax = max(max(EE));\n");
        fwritePlotScales2D(fp);

        if (dsum == 0.0)
        {
          printf("Deterministic RS used ==> no RS uncertainties.\n");
          fprintf(fp,"subplot(2,2,3)\n");
          fwritePlotAxes(fp);
          fwritePlotTitle(fp,"Prob. Dist. (RS with uncertainties)");
          if (plotMatlab())
          {
            fprintf(fp,"text(0.01,0.5,'No RS uncertainty info.','sc',");
            fprintf(fp,"'FontSize',11)\n");
            fprintf(fp,"text(0.01,0.4,'Deterministic RS used.','sc',");
            fprintf(fp,"'FontSize',11)\n");
          }
        }
        else
        {
          fprintf(fp,"subplot(2,2,3)\n");
          fprintf(fp,"plot(X,EE,'lineWidth',2)\n");
          fwritePlotScales2D(fp);
          fwritePlotAxes(fp);
          fwritePlotTitle(fp,"Prob. Dist. (-3,2,1,0,1,2,3 std.)");
          fwritePlotXLabel(fp,"Output Value");
          fwritePlotYLabel(fp,"Probabilities");
        }
        if (plotMatlab()) fprintf(fp, "subplot(2,2,[2 4])\n");
        else              fprintf(fp, "subplot(1,2,2)\n");
        for (ii = 0; ii < ntimes; ii++)
        {
          fprintf(fp,"for ii = 2 : %d\n", nbins);
          fprintf(fp,"   E%d(ii) = E%d(ii) + E%d(ii-1);\n",ii+1,ii+1,ii+1);
          fprintf(fp,"end;\n");
        }
        fprintf(fp, "EE = [");
        for (ii = 0; ii < ntimes; ii++)
          fprintf(fp, "E%d/E%d(%d) ", ii+1, ii+1, nbins);
        fprintf(fp, "];\n");
        fprintf(fp, "plot(X,EE,'linewidth',2)\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Cum. Dist. (-3,2,1,0,1,2,3 std.)");
        fwritePlotXLabel(fp, "Output Value");
        fwritePlotYLabel(fp, "Probabilities");
        fclose(fp);
        if (plotMatlab()) 
          printf("Output distribution plot file = matlabrsua.m\n");
        else
          printf("Output distribution plot file = scilabrsua.sci\n");
        for (ii = 0; ii < ntimes; ii++) delete [] Fcounts[ii];
        delete [] Fcounts;
      }
    }
    if (faPtrUA != NULL) delete faPtrUA;
    //psConfig_.InteractiveRestore();
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsuab 
  //**/ RS-based UA with bootstrap and can be with posterior
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsuab"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsuab: uncertainty analysis on bootstrapped RS\n");
      printf("Syntax: rsuab (no argument needed)\n");
      printf("This command perform uncertainty analysis on the ");
      printf("RS built from the\n");
      printf("loaded sample. It is similar to the rsua command, ");
      printf("except that the RS\n");
      printf("uncertainties in this case is induced by bootstrapping ");
      printf("rather than\n");
      printf("predicted by the RS itself.\n");
      return 0;
    }
    printAsterisks(PL_INFO, 0);
    printf("* Response surface-based Uncertainty Analysis (with bootstrap)\n");
    printDashes(PL_INFO, 0);
    printf("It is similar to 'rsua' except that the RS uncertainties ");
    printf("in this case\n");
    printf("is induced by bootstrapping rather than predicted");
    printf("by the RS.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }

    //**/ query user for output ID
    sscanf(lineIn,"%s %s", command, winput);
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    //**/ request or generate a sample for evaluation
    printf("A sample is needed from you to propagate through the RS.\n");
    printf("Select between the two options below: \n");
    printf("1. PSUADE will generate the sample\n");
    printf("2. User will provide the sample (in PSUADE data format)\n");
    snprintf(pString,100,"Enter 1 or 2 : ");
    int samSrc = getInt(1, 2, pString);

    //**/ generate a sample or get from user a sample for evaluation 
    //**/ ==> usNSams, vecUAInps
    int uaNSams;
    psVector  vecUAInps, vecUAOuts;
    psIVector vecUAStas;
    if (samSrc == 1)
    {
      printf("PSUADE will generate a sample for uncertainty analysis.\n");
      snprintf(pString,100,"Sample size ? (10000 - 100000) ");
      uaNSams = getInt(10000, 100000, pString);
      vecUAInps.setLength(uaNSams * nInputs);
      psuadeIO->getParameter("ana_use_input_pdfs", pPtr);
      int usePDFs = pPtr.intData_;
      if (usePDFs == 1)
      {
        printf("NOTE: Some inputs have non-uniform PDFs.\n");
        printf("      A MC sample will be created with these PDFs.\n");
        psuadeIO->getParameter("method_sampling", pPtr);
        kk = pPtr.intData_;
        psuadeIO->updateMethodSection(PSUADE_SAMP_MC,-1,-1,-1,-1);
        PDFManager *pdfman = new PDFManager();
        pdfman->initialize(psuadeIO);
        vecUAInps.setLength(uaNSams*nInputs);
        psVector vecLs, vecUs;
        vecUs.load(nInputs, iUpperB);
        vecLs.load(nInputs, iLowerB);
        pdfman->genSample(uaNSams, vecUAInps, vecLs, vecUs);
        psuadeIO->updateMethodSection(kk,-1,-1,-1,-1);
        delete pdfman;
      }
      else
      {
        printAsterisks(PL_INFO, 0);
        printf("NOTE: Uniform distribution is assumed for all inputs. ");
        printf("To use other\n");
        printf("      than uniform distributions, prescribe them in ");
        printf("the sample file\n");
        printf("      and set use_input_pdfs in the ANALYSIS section.\n");
        printAsterisks(PL_INFO, 0);
        Sampling *samPtr;
        if (nInputs < 51)
             samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LPTAU);
        else samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
        samPtr->setPrintLevel(0);
        samPtr->setInputBounds(nInputs, iLowerB, iUpperB);
        samPtr->setOutputParams(1);
        samPtr->setSamplingParams(uaNSams, -1, -1);
        samPtr->initialize(0);
        vecUAOuts.setLength(uaNSams);
        vecUAStas.setLength(uaNSams);
        samPtr->getSamples(uaNSams,nInputs,1,vecUAInps.getDVector(),
                     vecUAOuts.getDVector(),vecUAStas.getIVector());
        delete samPtr;
      }
    }
    else
    {
      printf("Enter UA sample file name (in PSUADE data format): ");
      char uaFileName[1001];
      scanf("%s", uaFileName);
      fgets(lineIn2, 500, stdin);
      PsuadeData *sampleIO = new PsuadeData();
      status = sampleIO->readPsuadeFile(uaFileName);
      if (status != 0)
      {
        printf("ERROR: Cannot read sample file.\n");
        delete sampleIO;
        return 1;
      }
      sampleIO->getParameter("input_ninputs", pPtr);
      kk = pPtr.intData_;
      if (kk != nInputs)
      {
        printf("ERROR: Sample nInputs mismatch.\n");
        printf(":      Input size in workspace     = %d.\n",nInputs);
        printf(":      Input size from your sample = %d.\n",kk);
        delete sampleIO;
        return 1;
      }
      sampleIO->getParameter("method_nsamples", pPtr);
      uaNSams = pPtr.intData_;
      if (uaNSams < 1000)
      {
        printf("ERROR: Your sample size should be at least 1000 to give\n");
        printf("       any reasonable UA results.\n");
        delete sampleIO;
        return 1;
      }
      sampleIO->getParameter("input_sample", pPtr);
      vecUAInps.load(uaNSams * nInputs, pPtr.dbleArray_);
      pPtr.clean();
      delete sampleIO;
    }

    //**/ need information to get bootstrapped sample
    snprintf(pString,100, 
      "How many bootstraps to create from the loaded sample (10 - 300) : ");
    int numBS = getInt(10, 300, pString);
    double bsPC=0;
    if (psConfig_.MasterModeIsOn())
    {
      printf("Bootstrapped samples will be created from randomly\n");
      printf("drawing from your RS sample. Normally a random draw\n");
      printf("may include around 60%% of the original sample. You\n");
      printf("may increase this percentage below.\n");
      kk = 0;
      while ((bsPC < 60 || bsPC > 90) && kk < 10)
      {
        printf("Enter percentage (60-90) : ");
        scanf("%lg", &bsPC);
        kk++;
      }
      if (bsPC < 60 || bsPC > 90 || kk >= 10) bsPC = 0; 
      else                                    bsPC *= 0.01;
    }

    //**/ ===========================================================
    //**/ perform UA
    //**/ ===========================================================
    vecUAOuts.setLength(uaNSams);

    //**/ ----------------------------------------------
    // bootstrapped method
    //**/ ----------------------------------------------
    //**/ create response surface place holder 
    printf("** CREATING RESPONSE SURFACE\n");
    FuncApprox *faPtrUAB = genFA(-1, nInputs, -1, nSamples);
    if (faPtrUAB == NULL)
    {
      printf("ERROR: Cannot generate response surface.\n");
      return 1;
    }
    int rsMethod = faPtrUAB->getID(); 
    delete faPtrUAB;
    faPtrUAB = NULL;
    
    //**/ for each bootstrap, initialize and evaluate response surface 
    int its;
    psVector vecBsSamInps, vecBsSamOuts, vecBsMeans, vecBsStds;
    vecBsSamInps.setLength(nSamples*nInputs);
    vecBsSamOuts.setLength(nSamples);
    vecBsMeans.setLength(numBS);
    vecBsStds.setLength(numBS);
    psIVector vecUseFlags;
    vecUseFlags.setLength(nSamples);

    if (plotMatlab()) fp = fopen("matlabrsuab.m", "w");
    else              fp = fopen("scilabrsuab.sci", "w");
    psConfig_.InteractiveSaveAndReset();
    for (its = 0; its < numBS; its++)
    {
      for (ss = 0; ss < nSamples; ss++) vecUseFlags[ss] = 0;
      //**/ generate bootstrapped sample
      int bsnSams = 0;
      kk = 0;
      while (kk < nSamples || (1.0*bsnSams/nSamples) < bsPC)
      {
        jj = PSUADE_rand() % nSamples;
        if (vecUseFlags[jj] == 0)
        {
          for (ii = 0; ii < nInputs; ii++)
             vecBsSamInps[bsnSams*nInputs+ii] = sampleInputs[jj*nInputs+ii];
          vecBsSamOuts[bsnSams] = sampleOutputs[jj*nOutputs+outputID];
          vecUseFlags[jj] = 1;
          bsnSams++;
        }
        kk++;
      }
      printf("Bootstrap %d has sample size = %d (drawn from %d)\n",its+1,
             bsnSams,nSamples);
      //**/ initialize response surface
      faPtrUAB = genFA(rsMethod, nInputs, -1, bsnSams);
      faPtrUAB->setBounds(iLowerB, iUpperB);
      faPtrUAB->setOutputLevel(0);
      status = faPtrUAB->initialize(vecBsSamInps.getDVector(),
                                    vecBsSamOuts.getDVector());
      if (status != 0)
      {
        printf("ERROR: In initializing response surface (1).\n");
        if (faPtrUAB != NULL) delete faPtrUAB;
        return 1;
      } 

      //**/ evaluate the user sample
      faPtrUAB->evaluatePoint(uaNSams,vecUAInps.getDVector(),
                              vecUAOuts.getDVector());
      delete faPtrUAB;

      //**/ compute statistics
      vecBsMeans[its] = vecBsStds[its] = 0.0;
      for (ss = 0; ss < uaNSams; ss++) 
        vecBsMeans[its] += vecUAOuts[ss];
      vecBsMeans[its] /= (double) uaNSams;
      for (ss = 0; ss < uaNSams; ss++)
        vecBsStds[its] += pow(vecUAOuts[ss] - vecBsMeans[its], 2.0);
      vecBsStds[its] = sqrt(vecBsStds[its] / uaNSams);
      if (fp != NULL)
      {
        fprintf(fp, "Y = [\n");
        for (ss = 0; ss < uaNSams; ss++) 
          fprintf(fp,"%e\n",vecUAOuts[ss]);
        fprintf(fp, "];\n");
        fprintf(fp, "Y%d = sort(Y);\n",its+1);
        fprintf(fp, "X%d = (1 : %d)';\n", its+1, uaNSams);
        fprintf(fp, "X%d = X%d / %d;\n", its+1, its+1, uaNSams);
        if (its == 0)
        {
          fprintf(fp, "YY = Y%d;\n", its+1);
          fprintf(fp, "XX = X%d;\n", its+1);
        }
        else
        {
          fprintf(fp, "YY = [YY Y%d];\n", its+1);
          fprintf(fp, "XX = [XX X%d];\n", its+1);
        }
      }
    }
    //**/ RS means
    faPtrUAB = genFA(rsMethod, nInputs, -1, nSamples);
    faPtrUAB->setBounds(iLowerB, iUpperB);
    faPtrUAB->setOutputLevel(0);
    for (ss = 0; ss < nSamples; ss++)
      vecBsSamOuts[ss] = sampleOutputs[ss*nOutputs+outputID];
    status = faPtrUAB->initialize(sampleInputs,
                                  vecBsSamOuts.getDVector());
    faPtrUAB->evaluatePoint(uaNSams,vecUAInps.getDVector(),
                            vecUAOuts.getDVector());
    delete faPtrUAB;
    psConfig_.InteractiveRestore();

    //**/ compute statistics 
    printAsterisks(PL_INFO, 0);
    double mean, stdev;
    mean = stdev = 0.0;
    for (its = 0; its < numBS; its++) mean += vecBsMeans[its];
    mean /= (double) numBS;
    for (ss = 0; ss < numBS; ss++) stdev += pow(vecBsMeans[ss]-mean, 2.0);
    stdev = sqrt(stdev/(double) numBS);
    printf("Sample mean    = %e (std = %e)\n", mean, stdev);
    mean = stdev = 0.0;
    for (its = 0; its < numBS; its++) mean += vecBsStds[its];
    mean /= (double) numBS;
    for (ss = 0; ss < numBS; ss++) stdev += pow(vecBsStds[ss]-mean, 2.0);
    stdev = sqrt(stdev/(double) numBS);
    printf("Sample std dev = %e (std = %e)\n", mean, stdev);
    printEquals(PL_INFO, 0);
    if (fp != NULL)
    {
      fprintf(fp, "Y0 = [\n");
      for (ss = 0; ss < uaNSams; ss++) 
        fprintf(fp,"%e\n",vecUAOuts[ss]);
      fprintf(fp, "];\n");
      fwriteHold(fp, 0);
      fprintf(fp,"subplot(2,2,3);\n");
      fprintf(fp,"YYY = reshape(YY,%d,1);\n",uaNSams*numBS);
      fprintf(fp,"[nk,xk] = hist(YYY,50);\n");
      fprintf(fp,"nk = nk / %d;\n",uaNSams);
      fprintf(fp,"bar(xk,nk,1.0)\n");
      fwritePlotAxes(fp);
      fwritePlotTitle(fp, "Prob. Dist. (RS with uncertainties)");
      fwritePlotXLabel(fp,"Output Value");
      fwritePlotYLabel(fp,"Probabilities");
      fprintf(fp,"subplot(2,2,1);\n");
      fprintf(fp,"[nk0,xk0] = hist(Y0,xk,50);\n");
      fprintf(fp,"nk0 = nk0 / %d;\n",uaNSams);
      fprintf(fp,"bar(xk0,nk0,1.0)\n");
      fwritePlotAxes(fp);
      fwritePlotTitle(fp, "Prob. Dist. (Original Sample)");
      fwritePlotXLabel(fp,"Output Value");
      fwritePlotYLabel(fp,"Probabilities");
      mean = stdev = 0.0;
      for (ss = 0; ss < uaNSams; ss++) mean += vecUAOuts[ss];
      mean /= (double) uaNSams;
      for (ss = 0; ss < uaNSams; ss++)
        stdev += pow(vecUAOuts[ss] - mean, 2.0);
      stdev = sqrt(stdev / uaNSams);
      fprintf(fp,"text(0.05,0.9,'Mean = %12.4e','sc','FontSize',11)\n",mean);
      fprintf(fp,"text(0.05,0.85,'Std  = %12.4e','sc','FontSize',11)\n",stdev);
      fprintf(fp,"subplot(2,2,[2 4]);\n");
      fprintf(fp,"plot(YY, XX, 'b-', 'lineWidth',3)\n");
      fwritePlotAxes(fp);
      fwritePlotTitle(fp, "Cumulative Distributions");
      fwritePlotXLabel(fp, "Output Value");
      fwritePlotYLabel(fp, "Probabilities");
      fclose(fp);
      printf("Output distribution plot file = matlabrsuab.m.\n");
    }
    return 0;
  }

  //**/ 12/2023------------------------------------------------------
  //**/ -------------------------------------------------------------
  // +++ rssobol1 (rsvce1_ni) + rsvce1_sobol 
  //**/ Sobol main effect
  //**/ These commands will replace rssobol1 and rssobol1b
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rssobol1") || 
           !strcmp(command, "rsvce1_ni") || 
           !strcmp(command, "rsvce1_sobol"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      if (!strcmp(command, "rsvce1_ni")) 
      {
        printf("rsvce1_ni: compute RS-based Sobol' first-order indices\n");
        printf("Syntax: rsvce1_ni (no argument needed)\n");
      }
      if (!strcmp(command, "rsvce1_sobol")) 
      {
        printf("rsvce1_sobol: compute RS-based Sobol' first-order indices\n");
        printf("Syntax: rsvce1_sobol (no argument needed)\n");
      }
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nInputs < 2)
    {
      printf("INFO: No point doing this for nInputs < 2.\n");
      return 1;
    }

    printAsterisks(PL_INFO, 0);
    printf("This command computes first-order sensitivity ");
    printf("indices using the\n");
    printf("response surface constructed from the loaded sample.\n");
    if (!strcmp(command, "rsvce1_ni")) 
    {
      printf("* rsvce1_ni uses numerical integration\n");
      printf("* rsvce1_ni supports 'non'-uniform input ");
      printf("distributions with no input\n");
      printf("  correlations (rsvce1_bin handles ");
      printf("all input distributions and\n");
      printf("  constraints).\n");
      printf("* rsvce1_ni supports input constraints.\n");
    }
    if (!strcmp(command, "rsvce1_sobol")) 
    {
      printf("* rsvce1_sobol uses the Sobol' method\n");
      printf("* rsvce1_sobol supports 'non'-uniform ");
      printf("input distributions with no\n");
      printf("  input correlations (rsvce1_bin handles ");
      printf("all input distributions and\n");
      printf("  constraints).\n");
      printf("* rsvce1_sobol supports input constraints.\n");
    }
    printf("NOTE: Turn on ana_expert mode for bootstrapped analysis.\n");
    printAsterisks(PL_INFO, 0);

    //**/ make recommendation if inputs have other than 
    //**/ uniform distributions ==> check distributions
    int hasCorr = 0;
    if (inputCMat_ != NULL)
    {
      for (ii = 0; ii < nInputs_; ii++)
        for (jj = 0; jj < ii; jj++)
          if (inputCMat_->getEntry(ii,jj) != 0) hasCorr = 1;
    }
    int hasPDF = VecInpPDFs_.sum();
    if (hasCorr != 0)
    {
      printf("NOTE: Since inputs have correlations, we ");
      printf("recommend using rsvce1_bin\n");
      printf("      instead (both rsvce1_ni and rsvce1_sobol ");
      printf("cannot handle input\n");
      printf("      correlations except those induced by ");
      printf("inequality constraints).\n");
      printDashes(PL_INFO, 0);
    }
    else if (hasPDF == 1 and !strcmp(command, "rsvce1_ni")) 
    {
      printf("* NOTE: Since inputs have other than uniform ");
      printf("distributions, rsvce1_ni\n");
      printf("        requires large samples to give higher ");
      printf("accuracy. Hence, we \n");
      printf("        recommend using rsvce1_sobol.\n"); 
      printDashes(PL_INFO, 0);
    }
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ ask user to select which output
    snprintf(pString,100,"Enter output number (1 - %d) : ", 
             nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    //**/ ask user to select response surface
    faType = -1;
    snprintf(pString,100,"Enter your response surface choice ? ");
    while (faType < 0 || faType > PSUADE_NUM_RS)
    {
      writeFAInfo(outputLevel_);
      faType = getFAType(pString);
    }
    if (faType < 0) 
    {
      printf("ERROR: Wrong response surface type (%d).\n",faType);
      return 1;
    }

    //**/ number of bootstrapped runs (in ana_expert mode only)
    int numBS=1;
    if (psConfig_.AnaExpertModeIsOn())
    {
      printf("INFO: You may use bootstrapping to estimate ");
      printf("errors in this analysis\n");
      printf("      (computing multiple sets of metrics ");
      printf("on multiple response\n");
      printf("      surfaces built from bootstrapped samples).\n");
      snprintf(pString,100,"Number of bootstrapped runs (1 - 50) : ");
      numBS = getInt(1, 50, pString);
    }

    //**/ select numerical integration or Sobol'
    RSMSobol1Analyzer *sobol1 = new RSMSobol1Analyzer();
    if (!strcmp(command, "rsvce1_ni"))
    {
      snprintf(pString, 16, "ana_rssobol1_ni");
      char *targv[1];
      targv[0] = (char *) pString;
      sobol1->setParam(1, targv); 
    }
    else if (!strcmp(command, "rsvce1_sobol") || !strcmp(command, "rssobol1"))
    {
      snprintf(pString, 19, "ana_rssobol1_sobol");
      char *targv[1];
      targv[0] = (char *) pString;
      sobol1->setParam(1, targv); 
    }

    //**/ prepare aData object to pass parameters to analyzer
    aData aPtr;
    aPtr.nSamples_ = nSamples_;
    aPtr.nInputs_ = nInputs_;
    aPtr.nOutputs_ = 1;
    aPtr.outputID_ = 0;
    aPtr.iLowerB_ = iLowerB;
    aPtr.iUpperB_ = iUpperB;
    aPtr.sampleInputs_ = sampleInputs;
    aPtr.sampleOutputs_ = sampleOutputs;
    aPtr.sampleStates_ = sampleStates;
    aPtr.printLevel_ = outputLevel_;
    aPtr.ioPtr_ = psuadeIO;
    aPtr.inputPDFs_ = VecInpPDFs_.getIVector();
    aPtr.inputMeans_ = VecInpMeans_.getDVector();
    aPtr.inputStdevs_ = VecInpStds_.getDVector();
    aPtr.faType_ = faType;

    //**/ run multiple times
    int ind, nSamples2;
    double totVar=0;
    pData *pdata = NULL;
    psVector  vecXT, vecYT, vecVCES;
    psIVector vecST, vecIT;
    vecVCES.setLength(numBS*nInputs);
    vecXT.setLength(nSamples*nInputs);
    vecYT.setLength(nSamples);
    vecST.setLength(nSamples);
    vecIT.setLength(nSamples);
    for (kk = 0; kk < numBS; kk++)
    {
      if (numBS > 1) 
        printf("* Performing analysis on bootstrap %d (of %d)\n",
               kk+1,numBS);
      //**/ random draw (at least 80%)
      if (numBS == 1)
      {
        for (ss = 0; ss < nSamples*nInputs; ss++)
          vecXT[ss] = sampleInputs[ss];
        for (ss = 0; ss < nSamples; ss++)
        {
          vecYT[ss] = sampleOutputs[ss];
          vecST[ss] = sampleStates[ss];
        } 
        nSamples2 = nSamples;
      }
      else
      {
        for (jj = 0; jj < nSamples; jj++) vecIT[jj] = 0;
        ss = nSamples2 = 0;
        while (nSamples2 < nSamples*0.8)
        {
          ind = PSUADE_rand() % nSamples;
          if (vecIT[ind] == 0)
          {
            for (ii = 0; ii < nInputs; ii++)
              vecXT[nSamples2*nInputs+ii] = sampleInputs[ind*nInputs+ii];
            vecYT[nSamples2] = sampleOutputs[ind*nOutputs+outputID];
            vecST[nSamples2] = sampleStates[ind];
            vecIT[ind] = 1;
            nSamples2++;
          }
          ss++;
        }
      }

      //**/ update bootstrapped sample
      aPtr.nSamples_ = nSamples2;
      aPtr.sampleInputs_ = vecXT.getDVector();
      aPtr.sampleOutputs_ = vecYT.getDVector();
      aPtr.sampleStates_ = vecST.getIVector();

      //**/ compute measure
      if (!psConfig_.AnaExpertModeIsOn() || numBS > 1)
        psConfig_.InteractiveSaveAndReset();
      sobol1->analyze(aPtr);
      if (!psConfig_.AnaExpertModeIsOn() || numBS > 1)
        psConfig_.InteractiveRestore();

      //**/ fetch the result
      pdata = psuadeIO->getAuxData();
      if (pdata->nDbles_ >= nInputs)
      {
        if (pdata->dbleData_ > 0)
        {
          for (ii = 0; ii < nInputs; ii++)
          {
            //**/ normalize
            ddata = pdata->dbleArray_[ii] / pdata->dbleData_;
            vecVCES[kk*nInputs+ii] = ddata;
          }
          totVar += pdata->dbleData_;
        }
        if (outputLevel_ > 1)
        {
          for (ii = 0; ii < nInputs; ii++)
            printOutTS(PL_INFO,
              "VCE(%d) = %10.3e\n",ii+1,vecVCES[kk*nInputs+ii]);
        }
      }
      else
      {
        printf("ERROR: Cannot fetch results (Consult developers).\n");
      }
    }

    //**/ clean up
    delete sobol1;
    aPtr.inputPDFs_ = NULL;
    aPtr.inputMeans_ = NULL;
    aPtr.inputStdevs_ = NULL;
    aPtr.iLowerB_ = NULL;
    aPtr.iUpperB_ = NULL;
    aPtr.sampleInputs_ = NULL;
    aPtr.sampleOutputs_ = NULL;
    aPtr.sampleStates_ = NULL;
    aPtr.ioPtr_ = NULL;

    //**/ compute statistics
    psVector vecMeans, vecStdvs, vecMaxs, vecMins;
    vecMeans.setLength(nInputs);
    vecStdvs.setLength(nInputs);
    vecMins.setLength(nInputs);
    vecMaxs.setLength(nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      ddata = 0;
      for (kk = 0; kk < numBS; kk++) ddata += vecVCES[kk*nInputs+ii];
      ddata /= (double) numBS;
      vecMeans[ii] = ddata;
      vecMins[ii] = PSUADE_UNDEFINED;
      vecMaxs[ii] = -PSUADE_UNDEFINED;
    }
    if (numBS > 1)
    {
      for (ii = 0; ii < nInputs; ii++)
      {
        ddata = 0;
        for (kk = 0; kk < numBS; kk++) 
        {
          ddata += pow(vecVCES[kk*nInputs+ii]-vecMeans[ii],2.0);
          if (vecVCES[kk*nInputs+ii] < vecMins[ii])
            vecMins[ii] = vecVCES[kk*nInputs+ii];
          if (vecVCES[kk*nInputs+ii] > vecMaxs[ii])
            vecMaxs[ii] = vecVCES[kk*nInputs+ii];
        }
        ddata /= (double) numBS;
        vecStdvs[ii] = ddata;
      }
    }
    totVar /= (double) numBS;
    if (totVar > 0)
    {
      printAsterisks(PL_INFO, 0);
      if (!strcmp(command, "rsvce1_ni"))
        printf("Sobol' 1st-order indices (normalized, 'num. int.'):\n");
      if (!strcmp(command, "rsvce1_sobol"))
        printf("Sobol' 1st-order indices (normalized, 'sobol'):\n");
      for (ii = 0; ii < nInputs; ii++)
      {
        if (numBS > 1)
          printOutTS(PL_INFO,
            "VCE for input %3d = %10.3e, s.d. = %10.3e\n",
            ii+1, vecMeans[ii], vecStdvs[ii]);
        else
          printOutTS(PL_INFO,
            "VCE for input %3d = %10.3e\n",ii+1,vecMeans[ii]);
      }
      printf("Total variance = %e\n", totVar);
      printAsterisks(PL_INFO, 0);
    }
    else
    {
      printf("Total variance = %e ==> Cannot compute VCEs\n",totVar);
      return 1;
    }
 
    //**/ create matlab or scilab plots
    if (!strcmp(command, "rsvce1_ni") || 
        !strcmp(command, "rssobol1"))
    {
      if (plotScilab())
           fp = fopen("scilabrsvce1_ni.sci", "w");
      else fp = fopen("matlabrsvce1_ni.m", "w");
    }
    if (!strcmp(command, "rsvce1_sobol"))
    {
      if (plotScilab())
           fp = fopen("scilabrsvce1_sobol.sci", "w");
      else fp = fopen("matlabrsvce1_sobol.m", "w");
    }
    if (fp == NULL) 
      printf("RSTSI ERROR: Cannot open file to save data\n");
    else
    {
      strcpy(pString,"This file contains Sobol' indices");
      fwriteComment(fp, pString);
      strcpy(pString,"set sortFlag = 1 and set nn to be the number");
      fwriteComment(fp, pString);
      strcpy(pString,"of inputs to display.");
      fwriteComment(fp, pString);

      fprintf(fp, "sortFlag = 0;\n");
      fprintf(fp, "nn = %d;\n", nInputs);
      fprintf(fp, "Mids = [\n");
      for (ii = 0; ii < nInputs; ii++) 
        fprintf(fp,"%24.16e\n", vecMeans[ii]*totVar);
      fprintf(fp, "];\n");
      if (numBS > 1)
      {
        fprintf(fp, "Mins = [\n");
        for (ii = 0; ii < nInputs; ii++) 
          fprintf(fp,"%24.16e\n",vecMins[ii]*totVar);
        fprintf(fp, "];\n");
        fprintf(fp, "Maxs = [\n");
        for (ii = 0; ii < nInputs; ii++) 
          fprintf(fp,"%24.16e\n",vecMaxs[ii]*totVar);
        fprintf(fp, "];\n");
      }
      if (inputNames == NULL)
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
          if (inputNames[ii] != NULL) 
               fprintf(fp,"'%s',",inputNames[ii]);
          else fprintf(fp,"'X%d',",ii+1);
        }
        if (plotScilab()) 
        {
          if (inputNames[nInputs-1] != NULL)
               fprintf(fp,"'%s'];\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'];\n",nInputs);
        }
        else
        {
          if (inputNames[nInputs-1] != NULL)
               fprintf(fp,"'%s'};\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'};\n",nInputs);
        }
      }
      fwriteHold(fp, 0);
      fprintf(fp, "if (sortFlag == 1)\n");
      if (plotScilab())
           fprintf(fp, "  [Mids, I2] = gsort(Mids);\n");
      else fprintf(fp, "  [Mids, I2] = sort(Mids,'descend');\n");
      if (numBS > 1)
      {
        fprintf(fp, "  Maxs = Maxs(I2);\n");
        fprintf(fp, "  Mins = Mins(I2);\n");
      }
      fprintf(fp, "  Str  = Str(I2);\n");
      fprintf(fp, "  I2 = I2(1:nn);\n");
      fprintf(fp, "  Mids = Mids(1:nn);\n");
      if (numBS > 1)
      {
        fprintf(fp, "  Maxs = Maxs(1:nn);\n");
        fprintf(fp, "  Mins = Mins(1:nn);\n");
      }
      fprintf(fp, "  Str  = Str(1:nn);\n");
      fprintf(fp, "end\n");
      if (numBS > 1)
      {
        fprintf(fp, "ymin = min(Mins);\n");
        fprintf(fp, "ymax = max(Maxs);\n");
      }
      else
      {
        fprintf(fp, "ymin = min(Mids);\n");
        fprintf(fp, "ymax = max(Mids);\n");
      }
      fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
      if (plotScilab()) fprintf(fp, "drawlater\n");
      fprintf(fp, "bar(Mids,0.8);\n");
      if (numBS > 1)
      {
        fprintf(fp,"for ii = 1:nn\n");
        if (plotScilab())
          fprintf(fp,
              "// h = plot(ii,Means(ii),'r*','MarkerSize',13);\n");
        else 
          fprintf(fp,
              "%% h = plot(ii,Means(ii),'r*','MarkerSize',13);\n");
        fprintf(fp,"   if (ii == 1)\n");
        fwriteHold(fp, 1);
        fprintf(fp,"   end;\n");
        fprintf(fp,"   XX = [ii ii];\n");
        fprintf(fp,"   YY = [Mins(ii) Maxs(ii)];\n");
        fprintf(fp,
           "   plot(XX,YY,'-ko','LineWidth',3.0,'MarkerEdgeColor',");
        fprintf(fp,"'k','MarkerFaceColor','g','MarkerSize',13)\n");
        fprintf(fp,"end;\n");
      }
      fwritePlotAxes(fp);
      fprintf(fp,"ymin=0;\n");
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
        fprintf(fp,"axis([0 nn+1 ymin ymax])\n");
        fprintf(fp,"set(gca,'XTickLabel',[]);\n");
        fprintf(fp,
          "th=text(1:nn, repmat(ymin-0.07*(ymax-ymin),nn,1),Str,");
        fprintf(fp,"'HorizontalAlignment','left','rotation',90);\n");
        fprintf(fp,"set(th, 'fontsize', 12)\n");
        fprintf(fp,"set(th, 'fontweight', 'bold')\n");
      }
      fwritePlotTitle(fp,"Sobol First Order Indices");
      fwritePlotYLabel(fp, "Sobol Indices");
      fwriteHold(fp, 0);
      if (plotScilab())
      {
        fprintf(fp, "drawnow\n");
        if (!strcmp(command, "rsvce1_ni"))
          printf("RSMVCE1 plot file = scilabrsvce1_ni.sci\n");
        else
          printf("RSMVCE1 plot file = scilabrsvce1_sobol.sci\n");
      }
      else 
      {
        if (!strcmp(command, "rsvce1_ni"))
          printf("RSMVCE1 plot file = matlabrsvce1_ni.m\n");
        else
          printf("RSMVCE1 plot file = matlabrsvce1_sobol.m\n");
      }
      fclose(fp);
    }
    return 0;
  }

  //**/ 12/2023------------------------------------------------------
  //**/ -------------------------------------------------------------
  // +++ rssobol2 (rsvce2_ni) + rsvce2_sobol
  //**/ Sobol interaction effect
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rssobol2") || 
           !strcmp(command, "rsvce2_ni") ||  
           !strcmp(command, "rsvce2_sobol"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      if (!strcmp(command, "rsvce2_ni"))
      {
        printf("rsvce2_ni: compute RS-based Sobol' second-order indices\n");
        printf("Syntax: rsvce2_ni (no argument needed)\n");
      }
      if (!strcmp(command, "rsvce2_sobol"))
      {
        printf("rsvce2_sobol: compute RS-based Sobol' second-order indices\n");
        printf("Syntax: rsvce2_sobol (no argument needed)\n");
      }
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nInputs <= 2)
    {
      printf("INFO: No point doing this for nInputs <= 2.\n");
      return 1;
    }

    printAsterisks(PL_INFO, 0);
    printf("This command computes second-order sensitivity ");
    printf("indices (input pairs)\n");
    printf("using a response surface constructed from the ");
    printf("loaded sample.\n");
    if (!strcmp(command, "rsvce2_ni"))
    {
      printf("* rsvce2_ni uses numerical integration\n");
      printf("* rsvce2_ni supports 'non'-uniform input ");
      printf("distributions with no input\n");
      printf("  correlations (rsvce2_bin handles ");
      printf("all input distributions and\n");
      printf("  constraints).\n");
      printf("* rsvce2_ni supports input constraints.\n");
    }
    if (!strcmp(command, "rsvce2_sobol"))
    {
      printf("* rsvce2_sobol uses the Sobol' method\n");
      printf("* rsvce2_sobol supports 'non'-uniform ");
      printf("input distributions with no\n");
      printf("  input correlations (rsvce2_bin handles ");
      printf("all input distributions and\n");
      printf("  constraints).\n");
      printf("* rsvce2_sobol supports input constraints.\n");
    }
    printf("NOTE: Turn on ana_expert mode for bootstrapped analysis.\n");
    printAsterisks(PL_INFO, 0);

    //**/ make recommendation if inputs have other than
    //**/ uniform distributions ==> check distributions
    int hasCorr = 0;
    if (inputCMat_ != NULL)
    {
      for (ii = 0; ii < nInputs_; ii++)
        for (jj = 0; jj < ii; jj++)
          if (inputCMat_->getEntry(ii,jj) != 0) hasCorr = 1;
    }
    int hasPDF = VecInpPDFs_.sum();
    if (hasCorr != 0)
    {
      printf("NOTE: Since inputs have correlations, we ");
      printf("recommend using rsvce2_bin\n");
      printf("      instead (both rsvce2_ni and rsvce2_sobol ");
      printf("cannot handle input\n");
      printf("      correlations except those induced by ");
      printf("inequality constraints).\n");
      printDashes(PL_INFO, 0);
    }
    else if (hasPDF == 1 and !strcmp(command, "rsvce2_ni"))
    {
      printf("* NOTE: Since inputs have other than uniform ");
      printf("distributions, rsvce2_ni\n");
      printf("        requires large samples to give higher ");
      printf("accuracy. Hence, we \n");
      printf("        recommend using rsvce2_sobol.\n");
      printDashes(PL_INFO, 0);
    }
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ ask user to select which output
    snprintf(pString,100,"Enter output number (1 - %d) : ", 
             nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    //**/ get RS type from user
    faType = -1;
    snprintf(pString,100,"Enter your response surface choice ? ");
    while (faType < 0 || faType > PSUADE_NUM_RS)
    {
      writeFAInfo(outputLevel_);
      faType = getFAType(pString);
    }
    if (faType < 0) 
    {
      printf("ERROR: Wrong response surface type (%d).\n",faType);
      return 1;
    }

    //**/ number of bootstrapped runs (in ana_expert mode only)
    int numBS=1;
    if (psConfig_.AnaExpertModeIsOn())
    {
      printf("You may estimate the error of the results ");
      printf("using bootstrapped runs\n");
      printf("(generate multiple bootstrapped samples for ");
      printf("building response surface)\n");
      snprintf(pString,100,"Number of bootstrapped runs (1 - 50) : ");
      numBS = getInt(1, 50, pString);
    }

    //**/ select numerical integration or Sobol'
    RSMSobol2Analyzer *sobol2 = new RSMSobol2Analyzer();
    if (!strcmp(command, "rsvce2_ni"))
    {
      snprintf(pString, 16, "ana_rssobol2_ni");
      char *targv[1];
      targv[0] = (char *) pString;
      sobol2->setParam(1, targv);
    }
    else if (!strcmp(command, "rsvce2_sobol") || !strcmp(command, "rssobol2"))
    {
      snprintf(pString, 19, "ana_rssobol2_sobol");
      char *targv[1];
      targv[0] = (char *) pString;
      sobol2->setParam(1, targv);
    }

    //**/ prepare aData object to pass parameters to analyzer
    aData aPtr;
    aPtr.nSamples_ = nSamples_;
    aPtr.nInputs_ = nInputs_;
    aPtr.nOutputs_ = 1;
    aPtr.outputID_ = 0;
    aPtr.iLowerB_ = iLowerB;
    aPtr.iUpperB_ = iUpperB;
    aPtr.sampleInputs_ = sampleInputs;
    aPtr.sampleOutputs_ = sampleOutputs;
    aPtr.sampleStates_ = sampleStates;
    aPtr.printLevel_ = outputLevel_;
    aPtr.ioPtr_ = psuadeIO;
    aPtr.inputPDFs_ = VecInpPDFs_.getIVector();
    aPtr.inputMeans_ = VecInpMeans_.getDVector();
    aPtr.inputStdevs_ = VecInpStds_.getDVector();
    aPtr.faType_ = faType;

    //**/ run multiple times
    int ind, nSamples2;
    double totVar=0;
    pData *pdata = NULL;
    psVector  vecXT, vecYT, vecVCES2;
    psIVector vecST, vecIT;
    vecVCES2.setLength(numBS*nInputs*nInputs);
    vecXT.setLength(nSamples*nInputs);
    vecYT.setLength(nSamples);
    vecST.setLength(nSamples);
    vecIT.setLength(nSamples);

    for (kk = 0; kk < numBS; kk++)
    {
      if (numBS > 1) 
        printf("* Performing analysis on bootstrap %d (of %d)\n",
               kk+1,numBS);
      //**/ random draw (at least 80%)
      if (numBS == 1)
      {
        for (ss = 0; ss < nSamples*nInputs; ss++)
          vecXT[ss] = sampleInputs[ss];
        for (ss = 0; ss < nSamples; ss++)
        {
          vecYT[ss] = sampleOutputs[ss];
          vecST[ss] = sampleStates[ss];
        } 
        nSamples2 = nSamples;
      }
      else
      {
        for (jj = 0; jj < nSamples; jj++) vecIT[jj] = 0;
        ss = nSamples2 = 0;
        while (nSamples2 < nSamples*0.8)
        {
          ind = PSUADE_rand() % nSamples;
          if (vecIT[ind] == 0)
          {
            for (ii = 0; ii < nInputs; ii++)
              vecXT[nSamples2*nInputs+ii] = sampleInputs[ind*nInputs+ii];
            vecYT[nSamples2] = sampleOutputs[ind*nOutputs+outputID];
            vecST[nSamples2] = sampleStates[ind];
            vecIT[ind] = 1;
            nSamples2++;
          }
          ss++;
        }
      }

      //**/ update bootstrapped sample
      aPtr.nSamples_ = nSamples2;
      aPtr.sampleInputs_ = vecXT.getDVector();
      aPtr.sampleOutputs_ = vecYT.getDVector();
      aPtr.sampleStates_ = vecST.getIVector();

      //**/ compute measure
      if (!psConfig_.AnaExpertModeIsOn() || numBS > 1)
        psConfig_.InteractiveSaveAndReset();
      sobol2->analyze(aPtr);
      if (!psConfig_.AnaExpertModeIsOn() || numBS > 1)
        psConfig_.InteractiveRestore();

      //**/ fetch the result
      pdata = psuadeIO->getAuxData();
      if (pdata->nDbles_ >= nInputs*nInputs)
      {
        if (pdata->dbleData_ > 0)
        {
          for (ii = 0; ii < nInputs*nInputs; ii++)
          {
            //**/ normalize
            ddata = pdata->dbleArray_[ii] / pdata->dbleData_;
            vecVCES2[kk*nInputs*nInputs+ii] = ddata;
          }
          totVar += pdata->dbleData_;
          if (outputLevel_ > 1)
          {
            for (ii = 0; ii < nInputs; ii++)
              for (jj = ii+1; jj < nInputs; jj++)
                printOutTS(PL_INFO,
                  "(1st+2nd) VCE(%d,%d) = %10.3e\n",ii+1,jj+1,
                  vecVCES2[kk*nInputs*nInputs+ii*nInputs+jj]);
          }
        }
      }
      else
      {
        printf("ERROR: Cannot fetch results (Consult developers).\n");
      }
    }

    //**/ clean up
    delete sobol2;
    aPtr.inputPDFs_ = NULL;
    aPtr.inputMeans_ = NULL;
    aPtr.inputStdevs_ = NULL;
    aPtr.iLowerB_ = NULL;
    aPtr.iUpperB_ = NULL;
    aPtr.sampleInputs_ = NULL;
    aPtr.sampleOutputs_ = NULL;
    aPtr.sampleStates_ = NULL;
    aPtr.ioPtr_ = NULL;

    //**/ compute statistics
    printAsterisks(PL_INFO, 0);
    psVector vecMeans2, vecStdvs2, vecMaxs2, vecMins2;
    vecMeans2.setLength(nInputs*nInputs);
    vecStdvs2.setLength(nInputs*nInputs);
    vecMins2.setLength(nInputs*nInputs);
    vecMaxs2.setLength(nInputs*nInputs);
    for (ii = 0; ii < nInputs*nInputs; ii++)
    {
      ddata = 0;
      for (kk = 0; kk < numBS; kk++) 
        ddata += vecVCES2[kk*nInputs*nInputs+ii];
      ddata /= (double) numBS;
      vecMeans2[ii] = ddata;
      vecMins2[ii] = PSUADE_UNDEFINED;
      vecMaxs2[ii] = -PSUADE_UNDEFINED;
    }
    if (numBS > 1)
    {
      for (ii = 0; ii < nInputs*nInputs; ii++)
      {
        ddata = 0;
        for (kk = 0; kk < numBS; kk++)
        {
          ddata += pow(vecVCES2[kk*nInputs*nInputs+ii]-
                       vecMeans2[ii],2.0);
          if (vecVCES2[kk*nInputs*nInputs+ii] < vecMins2[ii])
            vecMins2[ii] = vecVCES2[kk*nInputs*nInputs+ii];
          if (vecVCES2[kk*nInputs*nInputs+ii] > vecMaxs2[ii])
            vecMaxs2[ii] = vecVCES2[kk*nInputs*nInputs+ii];
        }
        ddata /= (double) numBS;
        vecStdvs2[ii] = sqrt(ddata);
      }
    }
    totVar /= (double) numBS;
    if (totVar > 0)
    {
      if (!strcmp(command, "rsvce2_ni") ||
          !strcmp(command, "rssobol2"))
        printf("Sobol' input-pair indices (normalized):ni\n");
      else
        printf("Sobol' input-pair indices (normalized):sobol\n");
      for (ii = 0; ii < nInputs; ii++)
      {
        for (jj = ii+1; jj < nInputs; jj++)
        {
          if (numBS > 1)
            printOutTS(PL_INFO,
             "VCE (1st+2nd) for input %3d,%3d = %10.3e, s.d. = %10.3e\n",
             ii+1,jj+1,vecMeans2[ii*nInputs+jj],vecStdvs2[ii*nInputs+jj]);
          else
            printOutTS(PL_INFO,
             "VCE (1st+2nd) for input %3d,%3d = %10.3e\n",ii+1,jj+1,
             vecMeans2[ii*nInputs+jj]);
        }
      }
      printf("Total variance = %e\n", totVar);
    }
    else
    {
      printf("Total variance = %e ==> Cannot compute VCEs\n", totVar);
      return 1;
    }
    printAsterisks(PL_INFO, 0);

    //**/ create matlab or scilab plots
    if (!strcmp(command, "rsvce2_ni") ||
        !strcmp(command, "rssobol2"))
    {
      if (plotScilab())
           fp = fopen("scilabrsvce2_ni.sci", "w");
      else fp = fopen("matlabrsvce2_ni.m", "w");
    }
    else if (!strcmp(command, "rsvce2_sobol"))
    {
      if (plotScilab())
           fp = fopen("scilabrsvce2_sobol.sci", "w");
      else fp = fopen("matlabrsvce2_sobol.m", "w");
    }

    if (fp == NULL)
      printf("RSVCE2 ERROR : Cannot open file to save data.\n");
    else
    {
      strcpy(pString,"This file contains Sobol' indices (unnormalized)");
      fwriteComment(fp, pString);
      strcpy(pString,"set sortFlag = 1 and set nn to be the number");
      fwriteComment(fp, pString);
      strcpy(pString,"of inputs to display.");
      fwriteComment(fp, pString);

      fprintf(fp, "sortFlag = 0;\n");
      fprintf(fp, "nn = %d;\n", nInputs);
      fprintf(fp, "Mids = [\n");
      for (ii = 0; ii < nInputs; ii++) 
      {
        for (jj = 0; jj <= ii; jj++) fprintf(fp,"0.0\n");
        for (jj = ii+1; jj < nInputs; jj++) 
          fprintf(fp,"%24.16e\n",vecMeans2[ii*nInputs+jj]*totVar);
        fprintf(fp, "];\n");
        fprintf(fp, "Mids = Mids';\n");
        if (inputNames == NULL)
        {
          if (plotScilab()) fprintf(fp, "Str = [");
          else              fprintf(fp, "Str = {");
          for (ii = 0; ii < nInputs-1; ii++) 
            fprintf(fp,"'X%d',",ii+1);
          if (plotScilab()) fprintf(fp,"'X%d'];\n",nInputs);
          else              fprintf(fp,"'X%d'};\n",nInputs);
        }
        else
        {
          if (plotScilab()) fprintf(fp, "Str = [");
          else              fprintf(fp, "Str = {");
          for (ii = 0; ii < nInputs-1; ii++)
          {
            if (inputNames[ii] != NULL) 
                 fprintf(fp,"'%s',",inputNames[ii]);
            else fprintf(fp,"'X%d',",ii+1);
          }
          if (plotScilab()) 
          {
            if (inputNames[nInputs-1] != NULL)
                 fprintf(fp,"'%s'];\n",inputNames[nInputs-1]);
            else fprintf(fp,"'X%d'];\n",nInputs);
          }
          else
          {
            if (inputNames[nInputs-1] != NULL)
                 fprintf(fp,"'%s'};\n",inputNames[nInputs-1]);
            else fprintf(fp,"'X%d'};\n",nInputs);
          }
        }
        fwriteHold(fp, 0);
        fprintf(fp, "ymin = min(Mids);\n");
        fprintf(fp, "ymax = max(Mids);\n");
        fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
        if (plotScilab())
        {
          fprintf(fp, "Mids = matrix(Mids, %d, %d);\n",
                  nInputs,nInputs);
          fprintf(fp, "Mids = Mids';\n");
          fprintf(fp, "drawlater\n");
          fprintf(fp, "hist3d(Mids);\n");
          fprintf(fp, "a=gca();\n");
          fprintf(fp, "a.data_bounds=[0, 0, 0; %d+1, %d+1, ymax];\n",
                  nInputs,nInputs);
          fprintf(fp, "newtick = a.x_ticks;\n");
          fprintf(fp, "newtick(2) = [1:%d]';\n",nInputs);
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
          fprintf(fp, "Mids = reshape(Mids, %d, %d);\n",nInputs,nInputs);
          fprintf(fp, "Mids = Mids';\n");
          fprintf(fp, "ymin = min(min(Mids));\n");
          fprintf(fp, "ymax = max(max(Mids));\n");
          fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
          fprintf(fp, "hh = bar3(Mids,0.8);\n");
          fprintf(fp, "alpha = 0.2;\n");
          fprintf(fp, "set(hh,'FaceColor','b','facea',alpha);\n");
          fprintf(fp, "axis([0.5 %d+0.5 0.5 %d+0.5 0 ymax])\n",
                  nInputs,nInputs);
          //fprintf(fp, "bar3(Mids,0.8);\n");
          //fprintf(fp, "axis([0 %d+1 0 %d+1 0 ymax])\n",nInputs,nInputs);
          fprintf(fp, "set(gca,'XTickLabel',Str);\n");
          fprintf(fp, "set(gca,'YTickLabel',Str);\n");
          fwritePlotAxesNoGrid(fp);
        }
        fwritePlotAxes(fp);
        fwritePlotTitle(fp,"Sobol Second Order Indices (+first order)");
        fwritePlotZLabel(fp, "Sobol Indices");
        fwritePlotXLabel(fp, "Inputs");
        fwritePlotYLabel(fp, "Inputs");
        fclose(fp);
        if (plotScilab())
        {
          if (!strcmp(command, "rsvce2_ni"))
            printf("RSMVCE2 plot file = scilabrsvce2_ni.sci\n");
          else
            printf("RSMVCE2 plot file = scilabrsvce2_sobol.sci\n");
        }
        else 
        {
          if (!strcmp(command, "rsvce2_ni"))
            printf("RSMVCE2 plot file = matlabrsvce2_ni.m\n");
          else
            printf("RSMVCE2 plot file = matlabrsvce2_sobol.m\n");

        }
      }
    }
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rssoboltsi (or rstsi_ni) and rstsi_sobol
  //**/ Sobol total sensitivity effect
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rssoboltsi") ||
           !strcmp(command, "rstsi_ni") ||
           !strcmp(command, "rstsi_bin") ||
           !strcmp(command, "rstsi_sobol"))

  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      if (!strcmp(command, "rstsi_ni") || !strcmp(command, "rssoboltsi"))
      {
        printf("rstsi_ni: compute RS-based Sobol' total-order indices\n");
        printf("Syntax: rstsi_ni (no argument needed)\n");
      }
      if (!strcmp(command, "rstsi_sobol"))
      {
        printf("rstsi_sobol: compute RS-based Sobol' total-order indices\n");
        printf("Syntax: rstsi_sobol (no argument needed)\n");
      }
      if (!strcmp(command, "rstsi_bin"))
      {
        printf("rstsi_bin: compute RS-based Sobol' total-order indices\n");
        printf("Syntax: rstsi_bin (no argument needed)\n");
      }
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nInputs < 2)
    {
      printf("INFO: No point doing this for nInputs < 2.\n");
      return 1;
    }

    printAsterisks(PL_INFO, 0);
    printf("This command computes total-order sensitivity ");
    printf("indices using the\n");
    printf("response surface constructed from the loaded sample.\n");
    if (!strcmp(command, "rstsi_ni") || 
        !strcmp(command, "rssoboltsi"))
    {
      printf("* rstsi_ni uses numerical integration\n");
      printf("* rstsi_ni supports 'non'-uniform input ");
      printf("distributions with no input\n");
      printf("  correlations (rstsi_bin handles all ");
      printf("input distributions and\n");
      printf("  constraints).\n");
      printf("* rstsi_ni supports input constraints.\n");
    }
    if (!strcmp(command, "rstsi_sobol"))
    {
      printf("* rstsi_sobol uses the Sobol' method\n");
      printf("* rstsi_sobol supports 'non'-uniform ");
      printf("input distributions with no\n");
      printf("  input correlations (rstsi_bin handles ");
      printf("all input distributions and\n");
      printf("  constraints).\n");
      printf("* rstsi_sobol supports input constraints.\n");
    }
    if (!strcmp(command, "rstsi_bin"))
    {
      printf("* rstsi_bin uses the binning method\n");
      printf("* rstsi_bin supports all types of input distributions.\n");
      printf("* rstsi_bin supports input constraints.\n");
      printf("* rstsi_bin does not work for nInputs > 21.\n");
    }
    printf("NOTE: Turn on ana_expert mode for bootstrapped analysis.\n");
    printAsterisks(PL_INFO, 0);

    //**/ make recommendation if inputs have other than
    //**/ uniform distributions ==> check distributions
    int hasCorr = 0;
    if (inputCMat_ != NULL)
    {
      for (ii = 0; ii < nInputs_; ii++)
        for (jj = 0; jj < ii; jj++)
          if (inputCMat_->getEntry(ii,jj) != 0) hasCorr = 1;
    }
    int hasPDF = VecInpPDFs_.sum();
    if (hasCorr != 0 && (!strcmp(command, "rssoboltsi") || 
                         !strcmp(command, "rstsi_ni") || 
                         !strcmp(command, "rstsi_sobol")))
    {
      printf("NOTE: Since inputs have correlations, we ");
      printf("recommend using rstsi_bin\n");
      printf("      instead (both rstsi_ni and rstsi_sobol ");
      printf("cannot handle input\n");
      printf("      correlations except those induced by ");
      printf("inequality constraints).\n");
      printDashes(PL_INFO, 0);
    }
    else if (hasPDF == 1 && hasCorr == 0 && 
             (!strcmp(command, "rssoboltsi") ||
              !strcmp(command, "rstsi_ni")))
    {
      printf("* NOTE: Since inputs have other than uniform ");
      printf("distributions, rstsi_ni\n");
      printf("        requires large samples to give higher ");
      printf("accuracy. Hence, we \n");
      printf("        recommend using rstsi_sobol.\n");
      printDashes(PL_INFO, 0);
    }
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    snprintf(pString,100,"Enter output number (1 - %d) : ", 
             nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    faType = -1;
    snprintf(pString,100,"Enter your response surface choice ? ");
    while (faType < 0 || faType > PSUADE_NUM_RS)
    {
      writeFAInfo(outputLevel_);
      faType = getFAType(pString);
    }
    if (faType < 0) 
    {
      printf("ERROR: Wrong response surface type (%d).\n",faType);
      return 1;
    }

    //**/ number of bootstrapped runs (in ana_expert mode only)
    int numBS=1;
    if (psConfig_.AnaExpertModeIsOn())
    {
      printf("INFO: You may use bootstrapping to estimate ");
      printf("errors in this analysis\n");
      printf("      (computing multiple sets of metrics ");
      printf("on multiple response\n");
      printf("      surfaces built from bootstrapped samples).\n");
      snprintf(pString,100,"Number of bootstrapped runs (1 - 50) : ");
      numBS = getInt(1, 50, pString);
    }

    //**/ select numerical integration or Sobol'
    RSMSobolTSIAnalyzer *soboltsi = new RSMSobolTSIAnalyzer();
    if (!strcmp(command, "rstsi_ni"))
    {
      snprintf(pString, 18, "ana_rssoboltsi_ni");
      char *targv[1];
      targv[0] = (char *) pString;
      soboltsi->setParam(1, targv);
    }
    else if (!strcmp(command, "rstsi_bin"))
    {
      snprintf(pString, 19, "ana_rssoboltsi_bin");
      char *targv[1];
      targv[0] = (char *) pString;
      soboltsi->setParam(1, targv);
    }
    else if (!strcmp(command, "rstsi_sobol") || !strcmp(command, "rssoboltsi"))
    {
      snprintf(pString, 21, "ana_rssoboltsi_sobol");
      char *targv[1];
      targv[0] = (char *) pString;
      soboltsi->setParam(1, targv);
    }

    //**/ prepare aData object to pass parameters to analyzer
    aData aPtr;
    aPtr.nSamples_ = nSamples_;
    aPtr.nInputs_ = nInputs_;
    aPtr.nOutputs_ = 1;
    aPtr.outputID_ = 0;
    aPtr.iLowerB_ = iLowerB;
    aPtr.iUpperB_ = iUpperB;
    aPtr.sampleInputs_ = sampleInputs;
    aPtr.sampleOutputs_ = sampleOutputs;
    aPtr.sampleStates_ = sampleStates;
    aPtr.printLevel_ = outputLevel_;
    aPtr.ioPtr_ = psuadeIO;
    aPtr.inputPDFs_ = VecInpPDFs_.getIVector();
    aPtr.inputMeans_ = VecInpMeans_.getDVector();
    aPtr.inputStdevs_ = VecInpStds_.getDVector();
    aPtr.faType_ = faType;

    //**/ run multiple times
    int ind, nSamples2;
    double totVar=0;
    pData *pdata = NULL;
    psVector  vecXT, vecYT, vecTSIs;
    psIVector vecST, vecIT;
    vecTSIs.setLength(numBS*nInputs);
    vecXT.setLength(nSamples*nInputs);
    vecYT.setLength(nSamples);
    vecST.setLength(nSamples);
    vecIT.setLength(nSamples);

    for (kk = 0; kk < numBS; kk++)
    {
      if (numBS > 1)
        printf("* Performing analysis on bootstrap %d (of %d)\n",
               kk+1,numBS);
      //**/ random draw (at least 80%)
      if (numBS == 1)
      {
        for (ss = 0; ss < nSamples*nInputs; ss++)
          vecXT[ss] = sampleInputs[ss];
        for (ss = 0; ss < nSamples; ss++)
        {
          vecYT[ss] = sampleOutputs[ss];
          vecST[ss] = sampleStates[ss];
        }
        nSamples2 = nSamples;
      }
      else
      {
        for (jj = 0; jj < nSamples; jj++) vecIT[jj] = 0;
        ss = nSamples2 = 0;
        while (nSamples2 < nSamples*0.8)
        {
          ind = PSUADE_rand() % nSamples;
          if (vecIT[ind] == 0)
          {
            for (ii = 0; ii < nInputs; ii++)
              vecXT[nSamples2*nInputs+ii] = sampleInputs[ind*nInputs+ii];
            vecYT[nSamples2] = sampleOutputs[ind*nOutputs+outputID];
            vecST[nSamples2] = sampleStates[ind];
            vecIT[ind] = 1;
            nSamples2++;
          }
          ss++;
        }
      }

      //**/ update bootstrapped sample
      aPtr.nSamples_ = nSamples2;
      aPtr.sampleInputs_ = vecXT.getDVector();
      aPtr.sampleOutputs_ = vecYT.getDVector();
      aPtr.sampleStates_ = vecST.getIVector();

      //**/ compute measure
      if (!psConfig_.AnaExpertModeIsOn() || numBS > 1)
        psConfig_.InteractiveSaveAndReset();
      soboltsi->analyze(aPtr);
      if (!psConfig_.AnaExpertModeIsOn() || numBS > 1)
        psConfig_.InteractiveRestore();

      //**/ fetch the results
      pdata = psuadeIO->getAuxData();
      if (pdata->nDbles_ >= nInputs)
      {
        if (pdata->dbleData_ > 0)
        {
          for (ii = 0; ii < nInputs; ii++)
          {
            ddata = pdata->dbleArray_[ii] / pdata->dbleData_;
            vecTSIs[kk*nInputs+ii] = ddata;
          }
          totVar += pdata->dbleData_;
          if (outputLevel_ > 1)
          {
            for (ii = 0; ii < nInputs; ii++)
              printOutTS(PL_INFO,"TSI(%d) = %10.3e\n",ii+1,
                  vecTSIs[kk*nInputs+ii]);
          }
        }
      }
      else
      {
        printf("ERROR: Cannot fetch results (Consult developers).\n");
      }
    }

    //**/ clean up
    delete soboltsi;
    aPtr.inputPDFs_ = NULL;
    aPtr.inputMeans_ = NULL;
    aPtr.inputStdevs_ = NULL;
    aPtr.iLowerB_ = NULL;
    aPtr.iUpperB_ = NULL;
    aPtr.sampleInputs_ = NULL;
    aPtr.sampleOutputs_ = NULL;
    aPtr.sampleStates_ = NULL;
    aPtr.ioPtr_ = NULL;

    //**/ compute statistics
    printAsterisks(PL_INFO, 0);
    psVector vecMeans, vecStdvs, vecMaxs, vecMins;
    vecMeans.setLength(nInputs);
    vecStdvs.setLength(nInputs);
    vecMins.setLength(nInputs);
    vecMaxs.setLength(nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      ddata = 0;
      for (kk = 0; kk < numBS; kk++) 
        ddata += vecTSIs[kk*nInputs+ii];
      ddata /= (double) numBS;
      vecMeans[ii] = ddata;
      vecMins[ii] = PSUADE_UNDEFINED;
      vecMaxs[ii] = -PSUADE_UNDEFINED;
    }
    if (numBS > 1)
    {
      for (ii = 0; ii < nInputs; ii++)
      {
        ddata = 0;
        for (kk = 0; kk < numBS; kk++)
        {
          ddata += pow(vecTSIs[kk*nInputs+ii]-vecMeans[ii],2.0);
          if (vecTSIs[kk*nInputs+ii] < vecMins[ii])
            vecMins[ii] = vecTSIs[kk*nInputs+ii];
          if (vecTSIs[kk*nInputs+ii] > vecMaxs[ii])
            vecMaxs[ii] = vecTSIs[kk*nInputs+ii];
        }
        ddata /= (double) numBS;
        vecStdvs[ii] = ddata;
      }
    }
    totVar /= (double) numBS;
    if (totVar > 0)
    {
      printAsterisks(PL_INFO, 0);
      if (!strcmp(command, "rstsi_ni") || !strcmp(command, "rssoboltsi"))
        printf("Sobol' total-order indices (normalized, 'num. int.'):\n");
      if (!strcmp(command, "rstsi_sobol"))
        printf("Sobol' total-order indices (normalized, 'sobol'):\n");
      if (!strcmp(command, "rstsi_bin"))
        printf("Sobol' total-order indices (normalized, 'bin'):\n");
      for (ii = 0; ii < nInputs; ii++)
      {
        if (numBS > 1)
          printOutTS(PL_INFO,
            "TSI for input %3d = %10.3e, s.d. = %10.3e\n",
            ii+1, vecMeans[ii], vecStdvs[ii]);
        else
          printOutTS(PL_INFO,
            "TSI for input %3d = %10.3e\n",ii+1,vecMeans[ii]);
      }
      printf("Total variance = %e\n", totVar);
      printAsterisks(PL_INFO, 0);
    }
    else
    {
      printf("Total variance = %e ==> cannot compute TSIs\n", totVar);
    }
    if (totVar == 0)
    {
      printf("INFO: Total variance = 0 ==> no plot.\n");
      return 1;
    }

    //**/ create matlab or scilab plots
    if (!strcmp(command, "rstsi_ni") ||
        !strcmp(command, "rssoboltsi"))
    {
      if (plotScilab())
           fp = fopen("scilabrstsi_ni.sci", "w");
      else fp = fopen("matlabrstsi_ni.m", "w");
    }
    else if (!strcmp(command, "rstsi_sobol"))
    {
      if (plotScilab())
           fp = fopen("scilabrstsi_sobol.sci", "w");
      else fp = fopen("matlabrstsi_sobol.m", "w");
    }
    else if (!strcmp(command, "rstsi_bin"))
    {
      if (plotScilab())
           fp = fopen("scilabrstsi_bin.sci", "w");
      else fp = fopen("matlabrstsi_bin.m", "w");
    }
    if (fp == NULL)
      printf("RSTSI ERROR: Cannot open file to save data\n");
    else
    {
      strcpy(pString,"This file contains Sobol' indices");
      fwriteComment(fp, pString);
      strcpy(pString,"set sortFlag = 1 and set nn to be the number");
      fwriteComment(fp, pString);
      strcpy(pString,"of inputs to display.");
      fwriteComment(fp, pString);

      fprintf(fp, "sortFlag = 0;\n");
      fprintf(fp, "nn = %d;\n", nInputs);
      fprintf(fp, "Mids = [\n");
      for (ii = 0; ii < nInputs; ii++)
        fprintf(fp,"%24.16e\n", vecMeans[ii]*totVar);
      fprintf(fp, "];\n");
      if (numBS > 1)
      {
        fprintf(fp, "Mins = [\n");
        for (ii = 0; ii < nInputs; ii++)
          fprintf(fp,"%24.16e\n",vecMins[ii]*totVar);
        fprintf(fp, "];\n");
        fprintf(fp, "Maxs = [\n");
        for (ii = 0; ii < nInputs; ii++)
          fprintf(fp,"%24.16e\n",vecMaxs[ii]*totVar);
        fprintf(fp, "];\n");
      }
      if (inputNames == NULL)
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
          if (inputNames[ii] != NULL)
               fprintf(fp,"'%s',",inputNames[ii]);
          else fprintf(fp,"'X%d',",ii+1);
        }
        if (plotScilab())
        {
          if (inputNames[nInputs-1] != NULL)
               fprintf(fp,"'%s'];\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'];\n",nInputs);
        }
        else
        {
          if (inputNames[nInputs-1] != NULL)
               fprintf(fp,"'%s'};\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'};\n",nInputs);
        }
      }
      fwriteHold(fp, 0);
      fprintf(fp, "if (sortFlag == 1)\n");
      if (plotScilab())
           fprintf(fp, "  [Mids, I2] = gsort(Mids);\n");
      else fprintf(fp, "  [Mids, I2] = sort(Mids,'descend');\n");
      if (numBS > 1)
      {
        fprintf(fp, "  Maxs = Maxs(I2);\n");
        fprintf(fp, "  Mins = Mins(I2);\n");
      }
      fprintf(fp, "  Str  = Str(I2);\n");
      fprintf(fp, "  I2 = I2(1:nn);\n");
      fprintf(fp, "  Mids = Mids(1:nn);\n");
      if (numBS > 1)
      {
        fprintf(fp, "  Maxs = Maxs(1:nn);\n");
        fprintf(fp, "  Mins = Mins(1:nn);\n");
      }
      fprintf(fp, "  Str  = Str(1:nn);\n");
      fprintf(fp, "end\n");
      if (numBS > 1)
      {
        fprintf(fp, "ymin = min(Mins);\n");
        fprintf(fp, "ymax = max(Maxs);\n");
      }
      else
      {
        fprintf(fp, "ymin = min(Mids);\n");
        fprintf(fp, "ymax = max(Mids);\n");
      }
      fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
      if (plotScilab()) fprintf(fp, "drawlater\n");
      fprintf(fp, "bar(Mids,0.8);\n");
      if (numBS > 1)
      {
        fprintf(fp,"for ii = 1:nn\n");
        if (plotScilab())
          fprintf(fp,
              "// h = plot(ii,Means(ii),'r*','MarkerSize',13);\n");
        else
          fprintf(fp,
              "%% h = plot(ii,Means(ii),'r*','MarkerSize',13);\n");
        fprintf(fp,"   if (ii == 1)\n");
        fwriteHold(fp, 1);
        fprintf(fp,"   end;\n");
        fprintf(fp,"   XX = [ii ii];\n");
        fprintf(fp,"   YY = [Mins(ii) Maxs(ii)];\n");
        fprintf(fp,
           "   plot(XX,YY,'-ko','LineWidth',3.0,'MarkerEdgeColor',");
        fprintf(fp,"'k','MarkerFaceColor','g','MarkerSize',13)\n");
        fprintf(fp,"end;\n");
      }
      fwritePlotAxes(fp);
      fprintf(fp,"ymin=0;\n");
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
        fprintf(fp,"axis([0 nn+1 ymin ymax])\n");
        fprintf(fp,"set(gca,'XTickLabel',[]);\n");
        fprintf(fp,
          "th=text(1:nn, repmat(ymin-0.07*(ymax-ymin),nn,1),Str,");
        fprintf(fp,"'HorizontalAlignment','left','rotation',90);\n");
        fprintf(fp,"set(th, 'fontsize', 12)\n");
        fprintf(fp,"set(th, 'fontweight', 'bold')\n");
      }
      fwritePlotTitle(fp,"Sobol Total Order Indices");
      fwritePlotYLabel(fp, "Sobol Indices");
      fwriteHold(fp, 0);
      if (plotScilab())
      {
        fprintf(fp, "drawnow\n");
        if (!strcmp(command, "rstsi_ni"))
          printf("RSMTSI plot file = scilabrstsi_ni.sci\n");
        else if (!strcmp(command, "rstsi_sobol"))
          printf("RSMTSI plot file = scilabrstsi_sobol.sci\n");
        else if (!strcmp(command, "rstsi_bin"))
          printf("RSMTSI plot file = scilabrstsi_bin.sci\n");
      }
      else
      {
        if (!strcmp(command, "rstsi_ni"))
          printf("RSMTSI plot file = matlabrstsi_ni.m\n");
        else if (!strcmp(command, "rstsi_sobol"))
          printf("RSMTSI plot file = matlabrstsi_sobol.m\n");
        else if (!strcmp(command, "rstsi_bin"))
          printf("RSMTSI plot file = matlabrstsi_bin.m\n");
      }
      fclose(fp);
    }
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsvceg_ni and rsvceg_sobol
  //**/ Sobol group main effect
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsvceg_ni") || 
           !strcmp(command, "rsvceg_sobol"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      if (!strcmp(command, "rsvceg_ni")) 
      {
        printf("rsvceg_ni: RS-based Sobol' group-order indices\n");
        printf("Syntax: rsvceg_ni (no argument needed)\n");
      }
      else if (!strcmp(command, "rsvceg_sobol")) 
      {
        printf("rsvceg_sobol: RS-based Sobol' group-order indices\n");
        printf("Syntax: rsvceg_sobol (no argument needed)\n");
      }
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }

    printAsterisks(PL_INFO, 0);
    printf("This command computes group-order sensitivity ");
    printf("indices using the\n");
    printf("response surface constructed from the loaded sample.\n");
    if (!strcmp(command, "rsvceg_ni"))
    {
      printf("* rsvceg_ni uses numerical integration\n");
      printf("* rsvceg_ni supports 'non'-uniform input ");
      printf("distributions with no input\n");
      printf("  correlations.\n");
      printf("* rsvceg_ni supports input constraints.\n");
    }
    if (!strcmp(command, "rsvceg_sobol"))
    {
      printf("* rsvceg_sobol uses the Sobol' method\n");
      printf("* rsvceg_sobol supports 'non'-uniform ");
      printf("input distributions with no\n");
      printf("  input correlations.\n");
      printf("* rsvceg_sobol supports input constraints.\n");
    }
    printf("NOTE: Turn on ana_expert mode for bootstrapped analysis.\n");
    printDashes(PL_INFO, 0);

    //**/ make recommendation if inputs have other than 
    //**/ uniform distributions ==> check distributions
    int hasCorr = 0;
    if (inputCMat_ != NULL)
    {
      for (ii = 0; ii < nInputs_; ii++)
        for (jj = 0; jj < ii; jj++) 
          if (inputCMat_->getEntry(ii,jj) != 0) hasCorr = 1;
    }
    int hasPDF = VecInpPDFs_.sum();
    if (hasCorr != 0)
    {
      printf("NOTE: Since inputs have correlations, we ");
      printf("recommend using rsvceg_bin\n");
      printf("      instead (both rsvceg_ni and rsvceg_sobol ");
      printf("cannot handle input\n");
      printf("      correlations except those induced by ");
      printf("inequality constraints).\n");
      printDashes(PL_INFO, 0);
    }
    else if (hasPDF == 1 and !strcmp(command, "rsvceg_ni"))
    {
      printf("* NOTE: Since inputs have other than uniform ");
      printf("distributions, rsvceg_ni\n");
      printf("        requires large samples to give higher ");
      printf("accuracy. Hence, we \n");
      printf("        recommend using rsvceg_sobol.\n");
      printDashes(PL_INFO, 0);
    }
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ ask user to select response surface
    snprintf(pString,100,"Enter output number (1 - %d) : ", 
             nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    //**/ ask user to select response surface
    faType = -1;
    snprintf(pString,100,"Enter your response surface choice ? ");
    while (faType < 0 || faType > PSUADE_NUM_RS)
    {
      writeFAInfo(outputLevel_);
      faType = getFAType(pString);
    }
    if (faType < 0) 
    {
      printf("ERROR: Response surface type not currently available.\n");
      return 1;
    }

    //**/ number of bootstrapped runs (in ana_expert mode only)
    int numBS=1;
    if (psConfig_.AnaExpertModeIsOn())
    {
      printf("INFO: You may use bootstrapping to estimate ");
      printf("errors in this analysis\n");
      printf("      (computing multiple sets of metrics ");
      printf("on multiple response\n");
      printf("      surfaces built from bootstrapped samples).\n");
      snprintf(pString,100,"Number of bootstrapped runs (1 - 50) : ");
      numBS = getInt(1, 50, pString);
    }

    printf("This group sensitivity analysis requires a file specifying\n");
    printf("group information in the form of : \n");
    printf("line 1: PSUADE_BEGIN\n");
    printf("line 2: <d> specifying the number of groups\n");
    printf("line 3 to line <d>+2: group number, size, input numbers\n");
    printf("last line: PSUADE_END\n");
    printf("Now enter the name of the group file : ");
    scanf("%s", dataFile);
    fgets(winput,5000,stdin);

    //**/ select numerical integration or Sobol'
    RSMSobolGAnalyzer *sobolg = new RSMSobolGAnalyzer();
    if (!strcmp(command, "rsvceg_ni"))
    {
      snprintf(pString, 16, "ana_rssobolg_ni");
      char *targv[1];
      targv[0] = (char *) pString;
      sobolg->setParam(1, targv);
    }
    else if (!strcmp(command, "rsvceg_sobol"))
    {
      snprintf(pString, 19, "ana_rssobolg_sobol");
      char *targv[1];
      targv[0] = (char *) pString;
      sobolg->setParam(1, targv);
    }

    //**/ prepare aData object to pass parameters to analyzer
    aData aPtr;
    aPtr.nSamples_ = nSamples_;
    aPtr.nInputs_ = nInputs_;
    aPtr.nOutputs_ = 1;
    aPtr.outputID_ = 0;
    aPtr.iLowerB_ = iLowerB;
    aPtr.iUpperB_ = iUpperB;
    aPtr.sampleInputs_ = sampleInputs;
    aPtr.sampleOutputs_ = sampleOutputs;
    aPtr.sampleStates_ = sampleStates;
    aPtr.printLevel_ = outputLevel_;
    aPtr.ioPtr_ = psuadeIO;
    aPtr.inputPDFs_ = VecInpPDFs_.getIVector();
    aPtr.inputMeans_ = VecInpMeans_.getDVector();
    aPtr.inputStdevs_ = VecInpStds_.getDVector();
    aPtr.faType_ = faType;
    kk = strlen(dataFile);
    dataFile[strlen(dataFile)] = '\0';
    strncpy(aPtr.grpFileName_, dataFile, kk+1);

    //**/ run multiple times
    int ind, nSamples2, nGroups;
    double totVar=0;
    pData *pdata = NULL;
    psVector  vecXT, vecYT, vecVCEG;
    psIVector vecST, vecIT;
    vecVCEG.setLength(numBS*nInputs);
    vecXT.setLength(nSamples*nInputs);
    vecYT.setLength(nSamples);
    vecST.setLength(nSamples);
    vecIT.setLength(nSamples);
    for (kk = 0; kk < numBS; kk++)
    {
      if (numBS > 1)
        printf("* Performing analysis on bootstrap %d (of %d)\n",
               kk+1,numBS);
      //**/ random draw (at least 80%)
      if (numBS == 1)
      {
        for (ss = 0; ss < nSamples*nInputs; ss++)
          vecXT[ss] = sampleInputs[ss];
        for (ss = 0; ss < nSamples; ss++)
        {
          vecYT[ss] = sampleOutputs[ss];
          vecST[ss] = sampleStates[ss];
        }
        nSamples2 = nSamples;
      }
      else
      {
        for (jj = 0; jj < nSamples; jj++) vecIT[jj] = 0;
        ss = nSamples2 = 0;
        while (nSamples2 < nSamples*0.8)
        {
          ind = PSUADE_rand() % nSamples;
          if (vecIT[ind] == 0)
          {
            for (ii = 0; ii < nInputs; ii++)
              vecXT[nSamples2*nInputs+ii] = sampleInputs[ind*nInputs+ii];
            vecYT[nSamples2] = sampleOutputs[ind*nOutputs+outputID];
            vecST[nSamples2] = sampleStates[ind];
            vecIT[ind] = 1;
            nSamples2++;
          }
          ss++;
        }
      }

      //**/ update bootstrapped sample
      aPtr.nSamples_ = nSamples2;
      aPtr.sampleInputs_ = vecXT.getDVector();
      aPtr.sampleOutputs_ = vecYT.getDVector();
      aPtr.sampleStates_ = vecST.getIVector();

      //**/ compute measure
      if (!psConfig_.AnaExpertModeIsOn() || numBS > 1)
        psConfig_.InteractiveSaveAndReset();
      sobolg->analyze(aPtr);
      if (!psConfig_.AnaExpertModeIsOn() || numBS > 1)
        psConfig_.InteractiveRestore();

      //**/ fetch the result
      pdata = psuadeIO->getAuxData();
      nGroups = pdata->nDbles_;
      //**/ need to revise this vector length as it is unknown
      if (vecVCEG.length() < numBS*nGroups) 
        vecVCEG.setLength(numBS*nGroups);
      if (nGroups > 0)
      {
        if (pdata->dbleData_ > 0)
        {
          for (ii = 0; ii < nGroups; ii++)
          {
            //**/ normalize
            ddata = pdata->dbleArray_[ii] / pdata->dbleData_;
            vecVCEG[kk*nGroups+ii] = ddata;
          }
          totVar += pdata->dbleData_;
        }
        if (outputLevel_ > 1)
        {
          for (ii = 0; ii < nGroups; ii++)
            printOutTS(PL_INFO,
              "VCE(%d) = %10.3e\n",ii+1,vecVCEG[kk*nGroups+ii]);
        }
      }
      else
      {
        printf("ERROR: Cannot fetch results (Consult developers).\n");
      }
    }
    //**/ clean up
    delete sobolg;
    aPtr.inputPDFs_ = NULL;
    aPtr.inputMeans_ = NULL;
    aPtr.inputStdevs_ = NULL;
    aPtr.iLowerB_ = NULL;
    aPtr.iUpperB_ = NULL;
    aPtr.sampleInputs_ = NULL;
    aPtr.sampleOutputs_ = NULL;
    aPtr.sampleStates_ = NULL;
    aPtr.ioPtr_ = NULL;

    //**/ compute statistics
    psVector vecMeans, vecStdvs, vecMaxs, vecMins;
    vecMeans.setLength(nGroups);
    vecStdvs.setLength(nGroups);
    vecMins.setLength(nGroups);
    vecMaxs.setLength(nGroups);
    for (ii = 0; ii < nGroups; ii++)
    {
      ddata = 0;
      for (kk = 0; kk < numBS; kk++) ddata += vecVCEG[kk*nGroups+ii];
      ddata /= (double) numBS;
      vecMeans[ii] = ddata;
      vecMins[ii] = PSUADE_UNDEFINED;
      vecMaxs[ii] = -PSUADE_UNDEFINED;
    }
    if (numBS > 1)
    {
      for (ii = 0; ii < nGroups; ii++)
      {
        ddata = 0;
        for (kk = 0; kk < numBS; kk++)
        {
          ddata += pow(vecVCEG[kk*nGroups+ii]-vecMeans[ii],2.0);
          if (vecVCEG[kk*nGroups+ii] < vecMins[ii])
            vecMins[ii] = vecVCEG[kk*nGroups+ii];
          if (vecVCEG[kk*nGroups+ii] > vecMaxs[ii])
            vecMaxs[ii] = vecVCEG[kk*nGroups+ii];
        }
        ddata /= (double) numBS;
        vecStdvs[ii] = ddata;
      }
    }
    totVar /= (double) numBS;
    if (totVar > 0)
    {
      printAsterisks(PL_INFO, 0);
      if (!strcmp(command, "rsvceg_ni"))
        printf("Sobol' Group-order indices (normalized, 'num. int.'):\n");
      if (!strcmp(command, "rsvceg_sobol"))
        printf("Sobol' Group-order indices (normalized, 'sobol'):\n");
      for (ii = 0; ii < nGroups; ii++)
      {
        if (numBS > 1)
          printOutTS(PL_INFO,
            "VCE for Group %3d = %10.3e, s.d. = %10.3e\n",
            ii+1, vecMeans[ii], vecStdvs[ii]);
        else
          printOutTS(PL_INFO,
            "VCE for Group %3d = %10.3e\n",ii+1,vecMeans[ii]);
      }
      printf("Total variance = %e\n", totVar);
      printAsterisks(PL_INFO, 0);
    }
    else
    {
      printf("Total variance = %e ==> cannot compute VCEs\n", totVar);
      printf("INFO: Total variance = 0 ==> no plot.\n");
      return 1;
    }
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rssobolg 
  //**/ Sobol group main effect
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rssobolg"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rssobolg: RS-based Sobol' group-order indices\n");
      printf("Syntax: rssobolg (no argument needed)\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }

    printAsterisks(PL_INFO, 0);
    printf("This command computes group-order sensitivity ");
    printf("indices using the\n");
    printf("response surface constructed from the loaded sample.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    faType = -1;
    snprintf(pString,100,"Enter your response surface choice ? ");
    while (faType < 0 || faType > PSUADE_NUM_RS)
    {
      writeFAInfo(outputLevel_);
      faType = getFAType(pString);
    }
    if (faType < 0) 
    {
      printf("ERROR: Response surface type not currently available.\n");
      return 1;
    }

    int analysisMethod = PSUADE_ANA_RSSOBOLG;
    AnalysisManager *anaManager = new AnalysisManager();
    anaManager->setup(analysisMethod, 0);
    psuadeIO->getParameter("ana_diagnostics",pPtr);
    int saveDiag = pPtr.intData_;
    psuadeIO->getParameter("ana_rstype",pPtr);
    int saveRS = pPtr.intData_;
    psuadeIO->updateAnalysisSection(-1,-1,faType,outputLevel_,-1,-1);
    anaManager->analyze(psuadeIO, 0, NULL, outputID);
    psuadeIO->updateAnalysisSection(-1,-1,saveRS,saveDiag,-1,-1);
    delete anaManager;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rssobol1b (rsvce1b_ni) -- OBSOLETE as of Dec 2023
  //**/ rssobol1 with bootstrap
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rssobol1b") ||
           !strcmp(command, "rsvce1b_ni"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsvce1b_ni: compute RS-based Sobol' first-order indices\n");
      printf("Syntax: rsvce1b_ni (no argument needed)\n");
      printf("NOTE: This command computes the first-order ");
      printf("Sobol' indices using\n");
      printf("      response surface constructed from the ");
      printf("loaded sample. It \n");
      printf("      estimates prediction uncertainty using ");
      printf("bootstrapping.\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nSamples < nInputs+1)
    {
      printf("INFO: This command is not suitable for small samples.\n");
      printf("      nSamples needs to be at least %d.\n",1+nInputs);
      return 1;
    }

    //**/ print usage information
    printAsterisks(PL_INFO, 0);
    printf("This command computes first-order sensitivity ");
    printf("indices using an\n");
    printf("ensemble of response surfaces constructed from ");
    printf("the loaded sample.\n");
    printf("Evaluations from the ensemble response surfaces ");
    printf("give error estimates\n");
    printf("for the sensitivity indices.\n");
#ifndef PSUADE_OMP
    printf("Advice: this command can be accelerated if you use OpenMP.\n");
#endif
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ get which output to analyze
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    //**/ get which response surface to use
    faType = -1;
    snprintf(pString,100,"Enter your response surface choice ? ");
    while (faType < 0 || faType > PSUADE_NUM_RS)
    {
      writeFAInfo(outputLevel_);
      faType = getFAType(pString);
#ifdef PSUADE_OMP
      if (faType == PSUADE_RS_MARS || faType == PSUADE_RS_GP1 || 
          faType >= PSUADE_RS_ACOSSO || faType == PSUADE_RS_SVM ||
          faType == PSUADE_RS_TGP || faType == PSUADE_RS_KR)
      {
        printf("These RS does not work in OpenMP mode. Select another one.\n");
        printf("- MARS and MARS-based RS\n");
        printf("- Multi-domain RS\n");
        printf("- GP1, TGP, SVM\n");
        printf("- SVM\n");
        printf("- Kriging (it uses Fortran optimizer)\n");
        faType = -1;
      }
      if (faType == PSUADE_RS_REGRL)
        printf("Legendre polynomial of order=2\n");
#endif
    }
    if (faType < 0) 
    {
      printf("ERROR: Response surface type not currently available.\n");
      return 1;
    }
    if (faType == PSUADE_RS_MARSB)
    {
      printf("rsvce1b_ni INFO: MarsBagg response surface selected but\n");
      printf("                 it is redundant - use MARS instead.\n");
      faType = PSUADE_RS_MARS;
    }

    //**/ set up for iterations
    psVector vecTT;
    snprintf(pString,100, 
          "How many bootstrapped samples to use (5 - 300) : ");
    int numBS = getInt(5, 300, pString);
    vecTT.setLength(numBS*nInputs);

    //**/ need to turn off expert modes, so save them first 
    psConfig_.AnaExpertModeSaveAndReset();
    psConfig_.RSExpertModeSaveAndReset();
    psConfig_.InteractiveSaveAndReset();

    //**/ set up analysis manager
    int analysisMethod = PSUADE_ANA_RSSOBOL1;
    printEquals(PL_INFO, 0);

    //**/ set up for bootstrapping
    int nSamples2, ind;
    psVector  vecXT, vecYT;
    psIVector vecIT, vecST;
    PsuadeData *psIO = NULL;
    AnalysisManager *anaManager;
    pData pNames, pIpdfs, pImeans, pIstds, pIcor;

    //**/ iterate
#pragma omp parallel shared(vecTT,sampleInputs,sampleStates,sampleOutputs,psuadeIO) \
    private(kk,jj,ss,ii,nSamples2,ind,vecXT,vecYT,vecST,vecIT,psIO,anaManager,pNames,pIpdfs,pImeans,pIstds,pIcor)
{
#pragma omp for
    for (kk = 0; kk < numBS; kk++)
    {
      vecXT.setLength(nSamples*nInputs);
      vecYT.setLength(nSamples);
      vecST.setLength(nSamples);
      vecIT.setLength(nSamples);
      anaManager = new AnalysisManager();
      anaManager->setup(analysisMethod, 0);
      psIO = new PsuadeData();
      //**/ random draw
      for (jj = 0; jj < nSamples; jj++) vecIT[jj] = 0;
      ss = nSamples2 = 0;
      while (ss < nSamples)
      {
        ind = PSUADE_rand() % nSamples;
        if (vecIT[ind] == 0)
        {
          for (ii = 0; ii < nInputs; ii++)
            vecXT[nSamples2*nInputs+ii] = sampleInputs[ind*nInputs+ii];
          vecYT[nSamples2] = sampleOutputs[ind*nOutputs+outputID];
          vecST[nSamples2] = sampleStates[ind];
          vecIT[ind] = 1;
          nSamples2++;
        }
        ss++;
      }
      printf("rsvce1b_ni: Bootstrap %d begins (sample size = %d)\n",
             kk+1,nSamples2);
      psuadeIO->getParameter("input_names", pNames);
      psuadeIO->getParameter("input_pdfs", pIpdfs);
      psuadeIO->getParameter("input_means", pImeans);
      psuadeIO->getParameter("input_stdevs", pIstds);
      psuadeIO->getParameter("input_cor_matrix", pIcor);
      psIO->updateInputSection(nSamples2,nInputs,NULL,iLowerB,
                  iUpperB,vecXT.getDVector(),pNames.strArray_,
                  pIpdfs.intArray_,pImeans.dbleArray_,pIstds.dbleArray_,
                  (psMatrix *) pIcor.psObject_); 
      psIO->updateOutputSection(nSamples2,1,vecYT.getDVector(),
                  vecST.getIVector(),&(outputNames[outputID]));
      psIO->updateMethodSection(PSUADE_SAMP_MC,nSamples2,-1,-1,-1);
      psIO->updateAnalysisSection(-1,-1,faType,-3,-1,-1);

      //**/ analyze the result
      anaManager->analyze(psIO, 0, NULL, 0);

      //**/ get the statistics
      pData *pdata = psIO->getAuxData(); 
      if (pdata->dbleData_ > 0)
      {
        for (ii = 0; ii < nInputs; ii++)
          vecTT[kk*nInputs+ii] = 
                  pdata->dbleArray_[ii]/pdata->dbleData_;
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          vecTT[kk*nInputs+ii] = pdata->dbleArray_[ii];
      }

      //**/ clean up
      pdata->clean();
      delete anaManager;
      delete psIO;
    }
} /* omp */

    //**/ postprocessing
    psVector vecMT;
    vecMT.setLength(nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      vecMT[ii] = vecTT[ii];
      for (jj = 1; jj < numBS; jj++) vecMT[ii] += vecTT[jj*nInputs+ii];
      vecMT[ii] /= (double) numBS;
    }
    psVector vecVT;
    vecVT.setLength(nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      vecVT[ii] = pow(vecTT[ii]-vecMT[ii], 2.0);
      for (jj = 1; jj < numBS; jj++) 
        vecVT[ii] += pow(vecTT[jj*nInputs+ii]-vecMT[ii],2.0);
      vecVT[ii] /= (double) (numBS - 1);
      vecVT[ii] = sqrt(vecVT[ii]);
    }
    printAsterisks(PL_INFO, 0);
    printf("rsvce1b_ni Statistics (based on %d replications): \n",numBS);
    printf("Quantities are normalized.\n");
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < nInputs; ii++)
      printf("   Input %4d: mean = %10.3e, std = %10.3e\n",ii+1,
             vecMT[ii],vecVT[ii]);
    printAsterisks(PL_INFO, 0);

    //**/ generate matlab/scilab file
    if (plotScilab()) fp = fopen("scilabrsvce1b_ni.sci","w");
    else              fp = fopen("matlabrsvce1b_ni.m","w");
    if (fp == NULL) printf("ERROR: Cannot open plot file.\n");
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
      fprintf(fp, "nn = %d;\n", nInputs);
      fprintf(fp, "Means = [\n");
      for (ii = 0; ii < nInputs; ii++) fprintf(fp,"%24.16e\n",vecMT[ii]);
      fprintf(fp, "];\n");
      fprintf(fp, "Stds = [\n");
      for (ii = 0; ii < nInputs; ii++) fprintf(fp,"%24.16e\n",vecVT[ii]);
      fprintf(fp, "];\n");
      if (inputNames == NULL)
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
          if (inputNames[ii] != NULL) 
               fprintf(fp,"'%s',",inputNames[ii]);
          else fprintf(fp,"'X%d',",ii+1);
        }
        if (plotScilab()) 
        {
          if (inputNames[nInputs-1] != NULL) 
               fprintf(fp,"'%s'];\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'];\n",nInputs);
        }
        else
        {
          if (inputNames[nInputs-1] != NULL) 
               fprintf(fp,"'%s'};\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'};\n",nInputs);
        }
      }
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
      fprintf(fp, "if ymin < 0 \n");
      fprintf(fp, "   ymin = 0;\n");
      fprintf(fp, "end;\n");
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
      fprintf(fp, "   plot(XX,YY,'-ko','LineWidth',3.0,'MarkerEdgeColor',");
      fprintf(fp, "'k','MarkerFaceColor','g','MarkerSize',13)\n");
      fprintf(fp, "end;\n");
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
        fprintf(fp,"axis([0  nn+1 ymin ymax])\n");
        fprintf(fp,"set(gca,'XTickLabel',[]);\n");
        fprintf(fp,"th=text(1:nn, repmat(ymin-0.05*(ymax-ymin),nn,1),Str,");
        fprintf(fp,"'HorizontalAlignment','left','rotation',90);\n");
        fprintf(fp,"set(th, 'fontsize', 12)\n");
        fprintf(fp,"set(th, 'fontweight', 'bold')\n");
      }
      fwritePlotTitle(fp,"First Order Sobol Indices (with bootstrap)");
      fwritePlotYLabel(fp, "First Order Sobol Index (Normalized)");
      if (plotScilab())
      {
        fprintf(fp, "drawnow\n");
        printf("rsvce1b_ni plot file = scilabrsvce1b_ni.sci\n");
      }
      else printf("rsvce1b_ni plot file = matlabrsvce1b_ni.m\n");
      fclose(fp);
    }

    //**/ restore previous settings
    psConfig_.AnaExpertModeRestore();
    psConfig_.RSExpertModeRestore();
    psConfig_.InteractiveRestore();
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rssobol2b 
  //**/ rssobol2 with bootstrap
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rssobol2b"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rssobol2b: compute RS-based Sobol' second-order indices\n");
      printf("Syntax: rssobol2b (no argument needed)\n");
      printf("NOTE: This command computes the second-order ");
      printf("Sobol' indices using\n");
      printf("      response surface constructed from the ");
      printf("loaded sample. It \n");
      printf("      estimates prediction uncertainty using ");
      printf("bootstrapping.\n");
      return 0;
    }
    if (nInputs <= 0 || psuadeIO == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nInputs <= 2)
    {
      printf("INFO: No point doing this for nInputs <= 2.\n");
      return 1;
    }
    if (nSamples < (nInputs+1)*5/3+1)
    {
      printf("INFO: This command is not suitable for small samples.\n");
      printf("      nSamples needs to be at least %d.\n",(nInputs+1)/3+1);
      printf("      nSamples needs to be at least %d.\n",(nInputs+1)/3+1);
      return 1;
    }

    //**/ print usage information
    printAsterisks(PL_INFO, 0);
    printf("This command computes input-pair sensitivity ");
    printf("indices using an\n");
    printf("ensemble of response surfaces constructed from ");
    printf("the loaded sample.\n");
    printf("Evaluations from the ensemble response surfaces ");
    printf("give error estimates\n");
    printf("for the sensitivity indices.\n");
#ifndef PSUADE_OMP
    printf("Advice: this command can be accelerated if you use OpenMP.\n");
#endif
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ get output
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    //**/ get which response surface to use
    faType = -1;
    snprintf(pString,100,"Enter your response surface choice ? ");
    while (faType < 0 || faType > PSUADE_NUM_RS)
    {
      writeFAInfo(outputLevel_);
      faType = getFAType(pString);
#ifdef PSUADE_OMP
      if (faType == PSUADE_RS_MARS || faType == PSUADE_RS_GP1 || 
          faType >= PSUADE_RS_ACOSSO || faType == PSUADE_RS_SVM ||
          faType == PSUADE_RS_TGP || faType == PSUADE_RS_KR)
      {
        printf("These RS does not work in OpenMP mode. Select another one.\n");
        printf("- MARS and MARS-based RS\n");
        printf("- Multi-domain RS\n");
        printf("- GP1, TGP, SVM\n");
        printf("- SVM\n");
        printf("- Kriging (it uses Fortran optimizer)\n");
        faType = -1;
      }
      if (faType == PSUADE_RS_REGRL)
        printf("Legendre polynomial of order=2\n");
#endif
    }
    if (faType < 0) 
    {
      printf("ERROR: Response surface type not currently available.\n");
      return 1;
    }
    if (faType == PSUADE_RS_MARSB)
    {
      printf("rssobol2b INFO: MarsBagg response surface selected but\n");
      printf("                it is redundant - use MARS instead.\n");
      faType = PSUADE_RS_MARS;
    }

    //**/ set up for iterations
    snprintf(pString,100,"How many bootstrapped samples to use (5 - 300) : ");
    int numBS = getInt(5, 300, pString);

    //**/ need to turn off expert modes, so save them first
    psConfig_.AnaExpertModeSaveAndReset();
    psConfig_.RSExpertModeSaveAndReset();
    psConfig_.InteractiveSaveAndReset();

    //**/ set up analysis manager
    int analysisMethod = PSUADE_ANA_RSSOBOL2;
    printEquals(PL_INFO, 0);

    //**/ set up storage for the response surface samples
    int ind, nSamples2;
    psVector  vecXT, vecYT, vecTT;
    psIVector vecIT, vecST;
    PsuadeData *psIO=NULL;
    vecTT.setLength((numBS+3)*nInputs*nInputs);
    AnalysisManager *anaManager;
    pData pNames, pIpdfs, pImeans, pIstds, pIcor;
  
    //**/ iterate
#pragma omp parallel shared(vecTT,sampleInputs,sampleStates,sampleOutputs,psuadeIO) \
    private(kk,jj,ss,ii,nSamples2,ind,vecXT,vecYT,vecST,vecIT,psIO,anaManager,pNames,pIpdfs,pImeans,pIstds,pIcor)
{
#pragma omp for
    for (kk = 0; kk < numBS; kk++)
    {
      vecXT.setLength(nSamples*nInputs);
      vecYT.setLength(nSamples);
      vecIT.setLength(nSamples);
      vecST.setLength(nSamples);
      anaManager = new AnalysisManager();
      anaManager->setup(analysisMethod, 0);
      psIO = new PsuadeData();

      //**/ random draw
      for (jj = 0; jj < nSamples; jj++) vecIT[jj] = 0;
      ss = nSamples2 = 0;
      while (ss < nSamples)
      {
        ind = PSUADE_rand() % nSamples;
        if (vecIT[ind] == 0)
        {
          for (ii = 0; ii < nInputs; ii++)
            vecXT[nSamples2*nInputs+ii] = sampleInputs[ind*nInputs+ii];
          vecYT[nSamples2] = sampleOutputs[ind*nOutputs+outputID];
          vecST[nSamples2] = sampleStates[ind];
          vecIT[ind] = 1;
          nSamples2++;
        }
        ss++;
      }
      printf("rssobol2b: bootstrap %d begins (sample size = %d)\n",
             kk+1,nSamples2);
      psuadeIO->getParameter("input_names", pNames);
      psuadeIO->getParameter("input_pdfs", pIpdfs);
      psuadeIO->getParameter("input_means", pImeans);
      psuadeIO->getParameter("input_stdevs", pIstds);
      psuadeIO->getParameter("input_cor_matrix", pIcor);
      psIO->updateInputSection(nSamples2,nInputs,NULL,iLowerB,iUpperB,
                  vecXT.getDVector(),pNames.strArray_,pIpdfs.intArray_,
                  pImeans.dbleArray_,pIstds.dbleArray_,
                  (psMatrix *) pIcor.psObject_);
      psIO->updateOutputSection(nSamples2,1,vecYT.getDVector(),
                  vecST.getIVector(),&(outputNames[outputID]));
      psIO->updateMethodSection(PSUADE_SAMP_MC,nSamples2,-1,-1,-1);
      psIO->updateAnalysisSection(-1,-1,faType,-3,-1,-1);

      //**/ analyze 
      status = anaManager->analyze(psIO, 0, NULL, 0);

      //**/ get statistics
      pData *pdata = psIO->getAuxData();
      if (pdata->dbleData_ > 0)
      {
        for (ii = 0; ii < nInputs*nInputs; ii++)
          vecTT[kk*nInputs*nInputs+ii] =
              pdata->dbleArray_[ii]/pdata->dbleData_;
      }
      else
      {
        for (ii = 0; ii < nInputs*nInputs; ii++)
          vecTT[kk*nInputs*nInputs+ii] = pdata->dbleArray_[ii];
      }
      //**/ clean up
      pdata->clean();
      delete anaManager;
      delete psIO;
    }
} /* omp */
    
    //**/ compute mean and std dev
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = 0; jj <= ii; jj++) 
        vecTT[numBS*nInputs*nInputs+ii*nInputs+jj] = 0.0;
      for (jj = ii+1; jj < nInputs; jj++)
      {
        ddata = 0.0;
        for (kk = 0; kk < numBS; kk++)
          ddata += vecTT[kk*nInputs*nInputs+ii*nInputs+jj]; 
        vecTT[numBS*nInputs*nInputs+ii*nInputs+jj] = ddata/(double) numBS;
        vecTT[(numBS+1)*nInputs*nInputs+ii*nInputs+jj] =  
                          vecTT[ii*nInputs+jj]; 
        vecTT[(numBS+2)*nInputs*nInputs+ii*nInputs+jj] =  
                         vecTT[ii*nInputs+jj]; 
        for (kk = 1; kk < numBS; kk++)
        {
          ddata = vecTT[kk*nInputs*nInputs+ii*nInputs+jj]; 
          if (ddata < vecTT[(numBS+1)*nInputs*nInputs+ii*nInputs+jj]) 
            vecTT[(numBS+1)*nInputs*nInputs+ii*nInputs+jj] = ddata; 
          if (ddata > vecTT[(numBS+2)*nInputs*nInputs+ii*nInputs+jj]) 
            vecTT[(numBS+2)*nInputs*nInputs+ii*nInputs+jj] = ddata; 
        }
      }
    }
    printAsterisks(PL_INFO, 0);
    printf("RSSobol2b Statistics (based on %d replications): \n", numBS);
    printf("Quantities are normalized.\n");
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = 0; jj <= ii; jj++) vecTT[ii*nInputs+jj] = 0.0;
      for (jj = ii+1; jj < nInputs; jj++)
      {
        ddata = 0.0;
        for (kk = 0; kk < numBS; kk++)
        {
          vecTT[kk*nInputs*nInputs+ii*nInputs+jj] -=  
             vecTT[numBS*nInputs*nInputs+ii*nInputs+jj];
          ddata += pow(vecTT[kk*nInputs*nInputs+ii*nInputs+jj],2.0);
        }
        ddata /= (double) (numBS - 1);
        vecTT[ii*nInputs+jj] = ddata;
        printf("RSSobol2 Inputs (%4d %4d): mean = %10.3e, std = %10.3e\n",
               ii+1,jj+1,vecTT[numBS*nInputs*nInputs+ii*nInputs+jj],ddata);
        //printf("2Param Input (%4d %4d): min  = %10.3e, max = %10.3e\n",ii+1,
        //      jj+1,vecTT[(numBS+1)*nInputs*nInputs+ii*nInputs+jj],
        //      vecTT[(numBS+2)*nInputs*nInputs+ii*nInputs+jj]);
      }
    }
    printAsterisks(PL_INFO, 0);
    if (plotScilab())
    {
      fp = fopen("scilabrssobol2b.sci", "w");
      if (fp == NULL) 
        printf("ERROR: Cannot open file scilabrssobol2b.sci\n");
    }
    else
    {
      fp = fopen("matlabrssobol2b.m", "w");
      if (fp == NULL) 
        printf("ERROR: Cannot open file matlabrssobol2b.sci\n");
    }
    if (fp != NULL) 
    {
      strcpy(pString,"This file contains Sobol' 2nd order indices");
      fwriteComment(fp, pString);
      strcpy(pString,"set sortFlag = 1 and set nn to be the number");
      fwriteComment(fp, pString);
      strcpy(pString," of inputs to display.");
      fwriteComment(fp, pString);

      fprintf(fp, "sortFlag = 0;\n");
      fprintf(fp, "nn = %d;\n", nInputs);
      fprintf(fp, "Means = [\n");
      for (ii = 0; ii < nInputs*nInputs; ii++) 
        fprintf(fp,"%24.16e\n", vecTT[numBS*nInputs*nInputs+ii]);
      fprintf(fp, "];\n");
      fprintf(fp, "Stds = [\n");
      for (ii = 0; ii < nInputs*nInputs; ii++) 
        fprintf(fp,"%24.16e\n", vecTT[ii]);
      fprintf(fp, "];\n");
      fprintf(fp, "Lows = [\n");
      for (ii = 0; ii < nInputs*nInputs; ii++) 
        fprintf(fp,"%24.16e\n", vecTT[(numBS+1)*nInputs*nInputs+ii]);
      fprintf(fp, "];\n");
      fprintf(fp, "Highs = [\n");
      for (ii = 0; ii < nInputs*nInputs; ii++) 
        fprintf(fp,"%24.16e\n", vecTT[(numBS+2)*nInputs*nInputs+ii]);
      fprintf(fp, "];\n");
      if (inputNames == NULL)
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
          if (inputNames[ii] != NULL) 
               fprintf(fp,"'%s',",inputNames[ii]);
          else fprintf(fp,"'X%d',",ii+1);
        }
        if (plotScilab())
        {
          if (inputNames[nInputs-1] != NULL)
               fprintf(fp,"'%s'];\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'];\n",nInputs);
        }
        else
        {
          if (inputNames[nInputs-1] != NULL)
               fprintf(fp,"'%s'};\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'};\n",nInputs);
        }
      }
      fwriteHold(fp, 0);
      fprintf(fp, "ymin = min(Means-Stds);\n");
      fprintf(fp, "ymax = max(Means+Stds);\n");
      fprintf(fp, "ymin = min(Lows);\n");
      fprintf(fp, "ymax = max(Highs);\n");
      fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
      if (plotScilab())
      {
        fprintf(fp, "nn    = %d;\n",nInputs);
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
        fprintf(fp, "//for ii = 1:nn\n");
        fprintf(fp, "//  for jj = ii+1:nn\n");
        fprintf(fp, "//    XX = [ii ii];\n");
        fprintf(fp, "//    YY = [jj jj];\n");
        fprintf(fp, "//    MM = Means(ii,jj);\n");
        fprintf(fp, "//    SS = Stds(ii,jj);\n");
        fprintf(fp, "//    ZZ = [MM-SS MM+SS];\n");
        fprintf(fp, "//    plot3d(XX,YY,ZZ,'-ko','LineWidth',3.0,");
        fprintf(fp, "//      'MarkerEdgeColor','k','MarkerFaceColor',");
        fprintf(fp, "//      'g','MarkerSize',13)\n");
        fprintf(fp, "//  end;\n");
        fprintf(fp, "//end;\n");
        fprintf(fp, "//a=gca();\n");
        fprintf(fp, "//a.data_bounds=[0, 0, 0; nn, nn+1, ymax];\n");
        fprintf(fp, "//newtick = a.x_ticks;\n");
        fprintf(fp, "//newtick(2) = [1:nn]';\n");
        fprintf(fp, "//drawlater\n");
        fprintf(fp, "//hist3d(Means);\n");
        fprintf(fp, "//set(gca(),\"auto_clear\",\"off\")\n");
        fprintf(fp, "//for ii = 1:nn\n");
        fprintf(fp, "//  for jj = ii+1:nn\n");
        fprintf(fp, "//    XX = [ii ii];\n");
        fprintf(fp, "//    YY = [jj jj];\n");
        fprintf(fp, "//    MM = Means(ii,jj);\n");
        fprintf(fp, "//    SS = Stds(ii,jj);\n");
        fprintf(fp, "//    ZZ = [MM-SS MM+SS];\n");
        fprintf(fp, "//    plot3d(XX,YY,ZZ,'-ko','LineWidth',3.0,");
        fprintf(fp, "//      'MarkerEdgeColor','k','MarkerFaceColor',");
        fprintf(fp, "//      'g','MarkerSize',13)\n");
        fprintf(fp, "//  end;\n");
        fprintf(fp, "//end;\n");
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
        fprintf(fp, "nn    = %d;\n",nInputs);
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
        fprintf(fp, "for k = 1:nn\n");
        fprintf(fp, "  for l = k:nn\n");
        fprintf(fp, "    mkl = Means(k,l);\n");
        fprintf(fp, "    ukl = Ustds(k,l);\n");
        fprintf(fp, "    lkl = Lstds(k,l);\n");
        fprintf(fp, "    if (mkl > .02 & (ukl-lkl)/mkl > .02)\n");
        fprintf(fp, "      xkl = [X(k,l), X(k,l)];\n");
        fprintf(fp, "      ykl = [Y(k,l), Y(k,l)];\n");
        fprintf(fp, "      zkl = [lkl, ukl];\n");
        fprintf(fp, "      plot3(xkl,ykl,zkl,'-mo',...\n");
        fprintf(fp, "        'LineWidth',5,'MarkerEdgeColor','k',...\n");
        fprintf(fp, "        'MarkerFaceColor','k','MarkerSize',10);\n");
        fprintf(fp, "    end\n");
        fprintf(fp, "  end\n");
        fprintf(fp, "end\n");
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

    //**/ restore previous settings
    psConfig_.AnaExpertModeRestore();
    psConfig_.RSExpertModeRestore();
    psConfig_.InteractiveRestore();
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rssoboltsib 
  //**/ rssoboltsi with bootstrap
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rssoboltsib"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rssoboltsib: compute RS-based Sobol' total-order indices\n");
      printf("Syntax: rssoboltsib (no argument needed)\n");
      printf("NOTE: This command computes the total-order ");
      printf("Sobol' indices using\n");
      printf("      response surface constructed from the ");
      printf("loaded sample. It \n");
      printf("      estimates prediction uncertainty using ");
      printf("bootstrapping.\n");
      return 0;
    }
    if (nInputs <= 0 || psuadeIO == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nInputs < 2)
    {
      printf("INFO: No point doing this for nInputs < 2.\n");
      return 1;
    }
    if (nSamples < 10)
    {
      printf("WARNING: Your sample size is quite small.\n");
      printf("         Bootstrapped samples will be smaller.\n");
      return 1;
    }

    //**/ print usage information
    printAsterisks(PL_INFO, 0);
    printf("This command computes total-order sensitivity ");
    printf("indices using an\n");
    printf("ensemble of response surfaces constructed from ");
    printf("the loaded sample.\n");
    printf("Evaluations from the ensemble response surfaces ");
    printf("give error estimates\n");
    printf("for the sensitivity indices.\n");
#ifndef PSUADE_OMP
    printf("Advice: this command can be accelerated if you use OpenMP.\n");
#endif
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ get output information
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    //**/ get which response surface to use
    faType = -1;
    snprintf(pString,100,"Enter your response surface choice ? ");
    while (faType < 0 || faType > PSUADE_NUM_RS)
    {
      writeFAInfo(outputLevel_);
      faType = getFAType(pString);
#ifdef PSUADE_OMP
      if (faType == PSUADE_RS_MARS || faType == PSUADE_RS_GP1 ||
          faType >= PSUADE_RS_ACOSSO || faType == PSUADE_RS_SVM ||
          faType == PSUADE_RS_TGP || faType == PSUADE_RS_KR)
      {
        printf("These RS does not work in OpenMP mode. Select another one.\n");
        printf("- MARS and MARS-based RS\n");
        printf("- Multi-domain RS\n");
        printf("- GP1, TGP, SVM\n");
        printf("- SVM\n");
        printf("- Kriging (it uses Fortran optimizer)\n");
        faType = -1;
      }
      if (faType == PSUADE_RS_REGRL)
        printf("Legendre polynomial of order=2\n");
#endif
    }
    if (faType < 0)
    {
      printf("ERROR: Response surface type not currently available.\n");
      return 1;
    }
    if (faType == PSUADE_RS_MARSB)
    {
      printf("rssoboltsib INFO: MarsBagg response surface selected but\n");
      printf("                  it is redundant - use MARS instead.\n");
      faType = PSUADE_RS_MARS;
    }

    //**/ set up for iterations
    snprintf(pString,100,"How many bootstrapped samples to use (5 - 300) : ");
    int numBS = getInt(5, 300, pString);

    //**/ need to turn off expert modes, so save them first
    psConfig_.AnaExpertModeSaveAndReset();
    psConfig_.RSExpertModeSaveAndReset();
    psConfig_.InteractiveSaveAndReset();

    //**/ set up analysis manager
    int analysisMethod = PSUADE_ANA_RSSOBOLTSI;
    printEquals(PL_INFO, 0);

    //**/ set up storage for the response surface samples
    int ind, nSamples2;
    psVector  vecXT, vecYT, vecTT;
    psIVector vecIT, vecST;
    PsuadeData *psIO=NULL;
    AnalysisManager *anaManager;
    pData pNames, pIpdfs, pImeans, pIstds, pIcor;
    vecTT.setLength(numBS*nInputs);

    //**/ iterate
#pragma omp parallel shared(vecTT,sampleInputs,sampleStates,sampleOutputs,psuadeIO) \
    private(kk,jj,ss,ii,nSamples2,ind,vecXT,vecYT,vecST,vecIT,psIO,anaManager,pNames,pIpdfs,pImeans,pIstds,pIcor)
#pragma omp for
    for (kk = 0; kk < numBS; kk++)
    {
      vecXT.setLength(nSamples*nInputs);
      vecYT.setLength(nSamples);
      vecIT.setLength(nSamples);
      vecST.setLength(nSamples);
      anaManager = new AnalysisManager();
      anaManager->setup(analysisMethod, 0);
      psIO = new PsuadeData();

      //**/ random draw
      for (jj = 0; jj < nSamples; jj++) vecIT[jj] = 0;
      ss = nSamples2 = 0;
      while (ss < nSamples)
      {
        ind = PSUADE_rand() % nSamples;
        if (vecIT[ind] == 0)
        {
          for (ii = 0; ii < nInputs; ii++)
            vecXT[nSamples2*nInputs+ii] = 
                                   sampleInputs[ind*nInputs+ii];
          vecYT[nSamples2] = sampleOutputs[ind*nOutputs+outputID];
          vecST[nSamples2] = sampleStates[ind];
          vecIT[ind] = 1;
          nSamples2++;
        }
        ss++;
      }
      printf("rssoboltsib: Bootstrap %d begins (sample size = %d)\n",
             kk+1,nSamples2);
      psuadeIO->getParameter("input_names", pNames);
      psuadeIO->getParameter("input_pdfs", pIpdfs);
      psuadeIO->getParameter("input_means", pImeans);
      psuadeIO->getParameter("input_stdevs", pIstds);
      psuadeIO->getParameter("input_cor_matrix", pIcor);
      psIO->updateInputSection(nSamples2,nInputs,NULL,iLowerB,iUpperB,
                  vecXT.getDVector(),pNames.strArray_,pIpdfs.intArray_,
                  pImeans.dbleArray_,pIstds.dbleArray_,
                  (psMatrix *) pIcor.psObject_);
      psIO->updateOutputSection(nSamples2,1,vecYT.getDVector(),
                  vecST.getIVector(),&(outputNames[outputID]));
      psIO->updateMethodSection(PSUADE_SAMP_MC,nSamples2,-1,-1,-1);
      psIO->updateAnalysisSection(-1,-1,faType,-3,-1,-1);

      //**/ analyze the result
      anaManager->analyze(psIO, 0, NULL, 0);

      //**/ get the statistics
      pData *pdata = psIO->getAuxData(); 
      if (pdata->dbleData_ > 0)
      {
        for (ii = 0; ii < nInputs; ii++)
          vecTT[kk*nInputs+ii] = 
                 pdata->dbleArray_[ii]/pdata->dbleData_;
      }
      else
      {
        for (ii = 0; ii < nInputs; ii++)
          vecTT[kk*nInputs+ii] = pdata->dbleArray_[ii];
      }

      //**/ clean up
      pdata->clean();
      delete anaManager;
      delete psIO;
    }

    //**/ postprocessing
    psVector vecMT, vecVT;
    vecMT.setLength(nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      vecMT[ii] = vecTT[ii];
      for (jj = 1; jj < numBS; jj++) vecMT[ii] += vecTT[jj*nInputs+ii];
      vecMT[ii] /= (double) numBS;
    }
    vecVT.setLength(nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      vecVT[ii] = pow(vecTT[ii]-vecMT[ii], 2.0);
      for (jj = 1; jj < numBS; jj++) 
        vecVT[ii] += pow(vecTT[jj*nInputs+ii]-vecMT[ii],2.0);
      vecVT[ii] /= (double) (numBS - 1);
      vecVT[ii] = sqrt(vecVT[ii]);
    }
    printAsterisks(PL_INFO, 0);
    printf("RSSobolTSIb' Statistics (based on %d replications): \n",
           numBS);
    printf("Quantities are normalized.\n");
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < nInputs; ii++)
      printf("RSSobolTSI Input %4d: mean = %10.3e, std = %10.3e\n",
             ii+1,vecMT[ii],vecVT[ii]);
    printAsterisks(PL_INFO, 0);

    //**/ generate matlab/scilab file
    if (plotScilab())
    {
      fp = fopen("scilabrssoboltsib.sci","w");
      if (fp == NULL) 
        printf("ERROR: Cannot open file scilabrssoboltsib.sci\n");
      else
      {
        fprintf(fp,"// This file contains total order Sobol' indices\n");
        fprintf(fp,"// with error bars coming from bootstrapping.\n");
        fprintf(fp,"// to select the most important ones to display,\n");
        fprintf(fp,"// set sortFlag = 1 and set nn to be the number\n");
        fprintf(fp,"// of inputs to display.\n");
      }
    }
    else
    {
      fp = fopen("matlabrssoboltsib.m","w");
      if (fp == NULL) 
        printf("ERROR: Cannot open file matlabrssoboltsib.sci\n");
      else
      {
        fprintf(fp,"%% This file contains total order Sobol' indices\n");
        fprintf(fp,"%% with error bars coming from bootstrapping.\n");
        fprintf(fp,"%% to select the most important ones to display,\n");
        fprintf(fp,"%% set sortFlag = 1 and set nn to be the number\n");
        fprintf(fp,"%% of inputs to display.\n");
      }
    }
    if (fp != NULL)
    {
      fprintf(fp, "sortFlag = 0;\n");
      fprintf(fp, "nn = %d;\n", nInputs);
      fprintf(fp, "Means = [\n");
      for (ii = 0; ii < nInputs; ii++) fprintf(fp,"%24.16e\n",vecMT[ii]);
      fprintf(fp, "];\n");
      fprintf(fp, "Stds = [\n");
      for (ii = 0; ii < nInputs; ii++) fprintf(fp,"%24.16e\n",vecVT[ii]);
      fprintf(fp, "];\n");
      if (inputNames == NULL)
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
          if (inputNames[ii] != NULL) 
               fprintf(fp,"'%s',",inputNames[ii]);
          else fprintf(fp,"'X%d',",ii+1);
        }
        if (plotScilab()) 
        {
          if (inputNames[nInputs-1] != NULL) 
               fprintf(fp,"'%s'];\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'];\n",nInputs);
        }
        else
        {
          if (inputNames[nInputs-1] != NULL) 
               fprintf(fp,"'%s'};\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'};\n",nInputs);
        }
      }
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
      fprintf(fp, "if ymin < 0 \n");
      fprintf(fp, "    ymin = 0;\n");
      fprintf(fp, "end;\n");
      fprintf(fp, "ymax = max(Means+Stds);\n");
      fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
      if (plotScilab()) fprintf(fp, "drawlater\n");
      fprintf(fp, "bar(Means,0.8);\n");
      fprintf(fp, "for ii = 1:nn\n");
      fprintf(fp, "   if (ii == 1)\n");
      fwriteHold(fp, 1);
      fprintf(fp, "   end;\n");
      fprintf(fp, "   XX = [ii ii];\n");
      fprintf(fp, "   YY = [Means(ii)-Stds(ii) Means(ii)+Stds(ii)];\n");
      fprintf(fp, "   if YY(1) < 0 \n");
      fprintf(fp, "      YY(1) = 0;\n");
      fprintf(fp, "   end;\n");
      fprintf(fp, "   plot(XX,YY,'-ko','LineWidth',3.0,'MarkerEdgeColor',");
      fprintf(fp, "'k','MarkerFaceColor','g','MarkerSize',12)\n");
      fprintf(fp, "end;\n");
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
        fprintf(fp,"axis([0  nn+1 ymin ymax])\n");
        fprintf(fp,"set(gca,'XTickLabel',[]);\n");
        fprintf(fp,"th=text(1:nn, repmat(ymin-0.05*(ymax-ymin),nn,1),");
        fprintf(fp,"Str,'HorizontalAlignment','left','rotation',90);\n");
        fprintf(fp,"set(th, 'fontsize', 12)\n");
        fprintf(fp,"set(th, 'fontweight', 'bold')\n");
      }
      fwritePlotTitle(fp,"Total Order Sobol Indices (with bootstrap)");
      fwritePlotYLabel(fp, "Total Order Sobol Index (Normalized)");
      fwriteHold(fp, 0);
      if (plotScilab())
      {
        fprintf(fp, "drawnow\n");
        printf("rssoboltsib plot file = scilabrssoboltsib.sci\n");
      }
      else printf("rssoboltsib plot file = matlabrssoboltsib.m\n");
      fclose(fp);
    }

    //**/ restore previous settings
    psConfig_.AnaExpertModeRestore();
    psConfig_.RSExpertModeRestore();
    psConfig_.InteractiveRestore();
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsua2
  //**/ uncertainty analysis on fuzzy response surface
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsua2") || !strcmp(command, "rs_ua"))
  {
    printf("This command has been replaced by rsua\n");
    return 0;
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsua2: uncertainty analysis on response surface\n");
      printf("Syntax: rsua2 (no argument needed)\n");
      printf("This command perform uncertainty analysis on the response\n");
      printf("surface built from the loaded sample. If you select a\n");
      printf("stochastic response surface type (Kriging, MARSB, or\n");
      printf("polynomial regression, the effect of response surface\n");
      printf("uncertainty (in the average sense) will be shown on the \n");
      printf("PDF and CDF plots.\n");
      printf("NOTE: This analysis supports non-uniform distributions for\n");
      printf("      the inputs. Simply prescribe PDF in the data file\n");
      printf("      and turn on use_input_pdfs in ANALYSIS.\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
     
    //**/ query user
    Sampling   *samPtr;
    FuncApprox *faPtr;
    PDFManager *pdfman;
    psVector   vecOut, vecLower, vecUpper;

    psuadeIO->getParameter("ana_rstype", pPtr);
    faType = pPtr.intData_;
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;
    snprintf(pString,100,
        "Sample size for generating distribution? (10000 - 100000) ");
    int nSamp = getInt(10000, 100000, pString);
    int saveFlag = 0;
    printf("Save the generated sample in a file? (y or n) ");
    fgets(winput,10,stdin); 
    if (winput[0] == 'y') saveFlag = 1; 
    int rsUncertaintyFlag = 0;
    printf("Include RS uncertainty in uncertainty analysis? (y or n) ");
    fgets(winput,10,stdin); 
    if (winput[0] == 'y') rsUncertaintyFlag = 1; 

    //**/ create response surface ==> faPtr
    printf("Phase 1 of 4: create response surface\n");
    faType = -1;
    faPtr = genFA(faType, nInputs, -1, nSamples);
    faPtr->setNPtsPerDim(32);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);
    psVector vecYT;
    if (nOutputs > 1)
    {
      vecYT.setLength(nSamples);
      for (ss = 0; ss < nSamples; ss++) 
        vecYT[ss] = sampleOutputs[ss*nOutputs+outputID];
    }
    else vecYT.load(nSamples, sampleOutputs);
    faPtr->initialize(sampleInputs,vecYT.getDVector());
           
    //**/ create a MC sample => nSamp, samInputs
    printf("Phase 2 of 4: create MC sample\n");
    psVector vecSamInps, vecSamOuts;
    vecSamInps.setLength(nInputs*nSamp);
    vecSamOuts.setLength(nSamp);
    psuadeIO->getParameter("ana_use_input_pdfs", pPtr);
    int usePDFs = pPtr.intData_;
    if (usePDFs == 1)
    {
      printf("NOTE: Some inputs have non-uniform PDFs.\n");
      printf("      A MC sample will be created with these PDFs.\n");
      psuadeIO->getParameter("method_sampling", pPtr);
      kk = pPtr.intData_;
      psuadeIO->updateMethodSection(PSUADE_SAMP_MC,-1,-1,-1,-1);
      pdfman = new PDFManager();
      pdfman->initialize(psuadeIO);
      vecOut.setLength(nSamp*nInputs);
      vecUpper.load(nInputs, iUpperB);
      vecLower.load(nInputs, iLowerB);
      pdfman->genSample(nSamp, vecOut, vecLower, vecUpper);
      for (ii = 0; ii < nSamp*nInputs; ii++) vecSamInps[ii] = vecOut[ii];
      psuadeIO->updateMethodSection(kk,-1,-1,-1,-1);
      delete pdfman;
    }
    else
    {
      printf("NOTE: Uniform distributions will be used for all inputs.\n");
      printf("      To use other than uniform distributions, prescribe\n");
      printf("      them in the data file and set use_input_pdfs in the\n");
      printf("      ANALYSIS section.\n");
      if (nInputs < 51)
           samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LPTAU);
      else samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
      samPtr->setPrintLevel(0);
      samPtr->setInputBounds(nInputs, iLowerB, iUpperB);
      samPtr->setOutputParams(1);
      samPtr->setSamplingParams(nSamp, -1, -1);
      samPtr->initialize(0);
      psIVector vecSamStates;
      vecSamStates.setLength(nSamp);
      samPtr->getSamples(nSamp,nInputs,1,vecSamInps.getDVector(),
                     vecSamOuts.getDVector(),vecSamStates.getIVector());
      delete samPtr;
      samPtr = NULL;
    }

    //**/ evaluate the sample => samOutputs, samStds
    psVector vecSamStds;
    vecSamStds.setLength(nSamp);
    printf("Phase 3 of 4: evaluate sample\n");
    if (rsUncertaintyFlag == 1) 
      faPtr->evaluatePointFuzzy(nSamp,vecSamInps.getDVector(),
                vecSamOuts.getDVector(),vecSamStds.getDVector()); 
    else
      faPtr->evaluatePoint(nSamp, vecSamInps.getDVector(),
                vecSamOuts.getDVector()); 
    delete faPtr;
    faPtr = NULL;
    if (saveFlag == 1)
    {
      if (!strcmp(command, "rsua2")) fp = fopen("rsua2_sample","w");
      else                           fp = fopen("rsua_sample","w");
      fprintf(fp, "%% inputs, output, output-3 sigma, output+3sigma\n");
      fprintf(fp, "%d %d 3\n", nSamp, nInputs);
      for (ss = 0; ss < nSamp; ss++)
      {
        for (ii = 0; ii < nInputs; ii++) 
          fprintf(fp, "%e ", vecSamInps[ss*nInputs+ii]);
        fprintf(fp, "%e ", vecSamOuts[ss]);
        fprintf(fp, "%e ", vecSamOuts[ss]-3*vecSamStds[ss]);
        fprintf(fp, "%e\n", vecSamOuts[ss]+3*vecSamStds[ss]);
      }
      fclose(fp);
      if (!strcmp(command, "rsua2"))
        printf("A MC sample has been written to the file 'rsua2_sample'\n");
      else
        printf("A MC sample has been written to the file 'rsua_sample'\n");
    }

    //**/ get the bounds for binning purposes => Fcounts
    int    nbins = 100, ntimes, **Fcounts;
    double mean=0, stdev=0;
    double FmaxO=-PSUADE_UNDEFINED, FminO=PSUADE_UNDEFINED;

    printf("Phase 4 of 4: binning\n");
    //**/ bin the original sample and compute statistics
    if (rsUncertaintyFlag == 0) ntimes = 1; 
    else                        ntimes = 21;
    Fcounts = new int*[ntimes+1];
    for (ii = 0; ii <= ntimes; ii++)
    {
      Fcounts[ii] = new int[nbins];
      for (kk = 0; kk < nbins; kk++) Fcounts[ii][kk] = 0;
    }
    for (ss = 0; ss < nSamp; ss++)
    {
      if (vecSamOuts[ss] > FmaxO) FmaxO = vecSamOuts[ss];
      if (vecSamOuts[ss] < FminO) FminO = vecSamOuts[ss];
      if (vecSamOuts[ss] > FmaxO) FmaxO = vecSamOuts[ss];
      if (vecSamOuts[ss] < FminO) FminO = vecSamOuts[ss];
    }
    FmaxO = FmaxO + 0.1 * (FmaxO - FminO);
    FminO = FminO - 0.1 * (FmaxO - FminO);
    if (FmaxO == FminO)
    {
      FmaxO = FmaxO + 0.1 * PABS(FmaxO);
      FminO = FminO - 0.1 * PABS(FminO);
    }
    for (ss = 0; ss < nSamp; ss++)
    {
      ddata = vecSamOuts[ss] - FminO;
      if (FmaxO > FminO) ddata = ddata / ((FmaxO - FminO) / nbins);
      else               ddata = nbins / 2;
      kk = (int) ddata;
      if (kk < 0)      kk = 0;
      if (kk >= nbins) kk = nbins - 1;
      Fcounts[ntimes][kk]++;
    }
    for (ss = 0; ss < nSamp; ss++) mean += vecSamOuts[ss];
    mean /= (double) nSamp;
    for (ss = 0; ss < nSamp; ss++) 
      stdev += pow(vecSamOuts[ss]-mean, 2.0);
    stdev = sqrt(stdev/(double) nSamp);
    printAsterisks(PL_INFO, 0);
    printf("Sample mean    = %e (RS uncertainties not included)\n",mean);
    printf("Sample std dev = %e (RS uncertainties not included)\n",stdev);
    printEquals(PL_INFO, 0);

    //**/ bin the rest 
    double mean2=0, stdev2=0;
    double Fmax=-PSUADE_UNDEFINED, Fmin=PSUADE_UNDEFINED;
    if (rsUncertaintyFlag == 1) 
    {
      psVector  vecSamFuzzy, vecSamOutSave;
      PDFNormal *rsPDF;
      vecSamFuzzy.setLength(ntimes*nInputs);
      vecSamOutSave.setLength(nSamp*ntimes);
      for (ss = 0; ss < nSamp; ss++)
      {
        if (vecSamStds[ss] == 0)
        {
          for (ii = 0; ii < ntimes; ii++) 
            vecSamFuzzy[ii] = vecSamOuts[ss];
        }
        else
        {
          ddata = 6.0 * vecSamStds[ss] / (ntimes - 1.0);
          for (ii = 0; ii < ntimes; ii++) 
            vecSamFuzzy[ii] = vecSamOuts[ss]+ii*ddata-3*vecSamStds[ss];
        }
        for (ii = 0; ii < ntimes; ii++) 
          vecSamOutSave[ss*ntimes+ii] = vecSamFuzzy[ii];
        if (ss % (nSamp / 8) == 0)
        {
          printf(".");
          fflush(stdout);
        }
      }
      for (ss = 0; ss < nSamp*ntimes; ss++)
      {
        if (vecSamOutSave[ss] < Fmin) Fmin = vecSamOutSave[ss];
        if (vecSamOutSave[ss] > Fmax) Fmax = vecSamOutSave[ss];
      }
      Fmax = Fmax + 0.1 * (Fmax - Fmin);
      Fmin = Fmin - 0.1 * (Fmax - Fmin);
      if (Fmax == Fmin)
      {
        Fmax = Fmax + 0.1 * PABS(Fmax);
        Fmin = Fmin - 0.1 * PABS(Fmin);
      }
      for (ss = 0; ss < nSamp; ss++)
      {
        for (ii = 0; ii < ntimes; ii++)
        {
          ddata = vecSamOutSave[ss*ntimes+ii] - Fmin;
          ddata = ddata / ((Fmax - Fmin) / nbins);
          kk = (int) ddata;
          if (kk < 0)      kk = 0;
          if (kk >= nbins) kk = nbins - 1;
          Fcounts[ii][kk]++;
        }
      }
      for (ss = 0; ss < nSamp*ntimes; ss++) mean2 += vecSamOutSave[ss];
      mean2 /= (double) (nSamp*ntimes);
      for (ss = 0; ss < nSamp*ntimes; ss++) 
        stdev2 += pow(vecSamOutSave[ss] - mean2, 2.0);
      stdev2 = sqrt(stdev2/(double) (nSamp*ntimes));
      printf("Sample mean    = %e (RS uncertainties included)\n",mean2);
      printf("Sample std dev = %e (RS uncertainties included)\n",stdev2);
      printAsterisks(PL_INFO, 0);
    }

    //**/ generate matlab/scilab file
    if (plotScilab())
    {
      if (!strcmp(command, "rsua2")) fp = fopen("scilabrsua2.sci", "w");
      else                           fp = fopen("scilabrsua.sci", "w");
      if (fp == NULL)
      {
        printf("rsua2 ERROR: Cannot open scilab file.\n");
        for (ii = 0; ii <= ntimes; ii++) delete [] Fcounts[ii];
        delete [] Fcounts;
        return 1;
      }
    }
    else
    {
      if (!strcmp(command, "rsua2")) fp = fopen("matlabrsua2.m", "w");
      else                           fp = fopen("matlabrsua.m", "w");
      if (fp == NULL)
      {
        printf("rsua2 ERROR: Cannot open matlab file.\n");
        for (ii = 0; ii <= ntimes; ii++) delete [] Fcounts[ii];
        delete [] Fcounts;
        return 1;
      }
    }
    fwriteHold(fp, 0);
    fprintf(fp, "subplot(2,2,1)\n");
    fprintf(fp, "XO = [\n");
    for (kk = 0; kk < nbins; kk++)
      fprintf(fp, "%e\n", (FmaxO-FminO)/nbins*(0.5+kk)+FminO);
    fprintf(fp, "];\n");
    fprintf(fp, "X = [\n");
    for (kk = 0; kk < nbins; kk++)
      fprintf(fp, "%e\n", (Fmax-Fmin)/nbins*(0.5+kk)+Fmin);
    fprintf(fp, "];\n");
    for (ii = 0; ii <= ntimes; ii++)
    {
      fprintf(fp, "N%d = [\n", ii+1);
      for (kk = 0; kk < nbins; kk++)
        fprintf(fp, "%d\n", Fcounts[ii][kk]);
      fprintf(fp, "];\n");
    }
    fprintf(fp, "N = [");
    for (ii = 0; ii <= ntimes; ii++)
      fprintf(fp, "N%d/sum(N%d) ", ii+1, ii+1);
    fprintf(fp, "];\n");
    fprintf(fp, "NA = N(:,%d+1);\n",ntimes);
    fprintf(fp, "NA = NA / sum(NA);\n");
    fprintf(fp, "bar(XO,NA,1.0)\n");
    fprintf(fp, "ymin = 0;\n");
    fprintf(fp, "ymax = max(NA);\n");
    fprintf(fp, "axis([min(XO) max(XO) ymin ymax])\n");
    fwritePlotAxes(fp);
    fwritePlotTitle(fp, "Prob. Dist. (means of RS)");
    fwritePlotXLabel(fp, "Output Value");
    fwritePlotYLabel(fp, "Probabilities)");
    if (plotMatlab())
    {
      fprintf(fp,"text(0.05,0.9,'Mean = %12.4e','sc','FontSize',11)\n",mean);
      fprintf(fp,"text(0.05,0.85,'Std  = %12.4e','sc','FontSize',11)\n",
              stdev);
    }
    if (rsUncertaintyFlag == 1) 
    {
      fprintf(fp, "NB = sum(N(:,1:%d)');\n",ntimes);
      fprintf(fp, "NB = NB' / sum(NB);\n");
      fprintf(fp, "subplot(2,2,3)\n");
      fprintf(fp, "bar(X,NB,1.0)\n");
      fprintf(fp, "ymax = max(max(NA),max(NB));\n");
      fprintf(fp, "axis([min(X) max(X) ymin ymax])\n");
      fwritePlotAxes(fp);
      fwritePlotTitle(fp, "Prob. Dist. (RS with uncertainties)");
      fwritePlotXLabel(fp, "Output Value");
      fwritePlotYLabel(fp, "Probabilities)");
      if (plotMatlab())
      {
        fprintf(fp,"text(0.05,0.9,'Mean = %12.4e','sc','FontSize',11)\n",
                mean2);
        fprintf(fp,"text(0.05,0.85,'Std  = %12.4e','sc','FontSize',11)\n",
                   stdev2);
      }
    }
    else
    {
      printf("Deterministic RS used ==> no RS uncertainties.\n");
      fprintf(fp, "subplot(2,2,3)\n");
      fwritePlotTitle(fp, "Prob. Dist. (RS with uncertainties)");
      snprintf(pString,100,"Deterministic RS: no RS uncertainties");
      fwriteComment(fp, pString);
    }
    if (faType == PSUADE_RS_MARS || faType == PSUADE_RS_RBF ||
        rsUncertaintyFlag == 0) 
    {
      fprintf(fp, "subplot(2,2,[2 4])\n");
      fprintf(fp, "plot(X,NA,'linewidth',3)\n");
      fwritePlotTitle(fp,"Cum. Dist.: (*) uncertainties unavailable");
    }
    else
    {
      for (ii = 0; ii <= ntimes; ii++)
      {
        fprintf(fp, "for ii = 2 : %d\n", nbins);
        fprintf(fp, "  N%d(ii) = N%d(ii) + N%d(ii-1);\n",ii+1,ii+1,ii+1);
        fprintf(fp, "end;\n");
      }
      fprintf(fp, "N = [");
      for (ii = 0; ii <= ntimes; ii++)
        fprintf(fp, "N%d/N%d(%d) ", ii+1, ii+1, nbins);
      fprintf(fp, "];\n");
      fprintf(fp, "subplot(2,2,[2 4])\n");
      fwriteHold(fp, 1);
      fprintf(fp, "for ii = 1 : %d\n", ntimes);
      fprintf(fp, "  if (ii == %d)\n", (ntimes+1)/2);
      fprintf(fp, "    plot(X,N(:,ii),'b-','linewidth',3)\n");
      fprintf(fp, "  else\n");
      fprintf(fp, "    plot(X,N(:,ii),'r-','linewidth',1)\n");
      fprintf(fp, "  end\n");
      fprintf(fp, "end\n");
      fwritePlotTitle(fp,"Cum. Dist.: (b) mean; (r) with uncertainties");
    }
    fwritePlotAxes(fp);
    fwritePlotXLabel(fp, "Output Value");
    fwritePlotYLabel(fp, "Probabilities");
    fclose(fp);
    if (!strcmp(command, "rsua2")) 
    {
      if (plotScilab())
           printf("Output distribution plots is in scilabrsua2.sci.\n");
      else printf("Output distribution plots is in matlabrsua2.m.\n");
    }
    else
    {
      if (plotScilab())
           printf("Output distribution plot file = scilabrsua.sci.\n");
      else printf("Output distribution plot file = matlabrsua.m.\n");
    }
    for (ii = 0; ii < ntimes; ii++) delete [] Fcounts[ii];
    delete [] Fcounts;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsuab2 
  //**/ uncertainty analysis on fuzzy response surface (New 2/2014)
  //**/ This is similar to rsuab but support psuadeData format
  //**/ This will be replaced by rsua + rsuab
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsuab2"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsuab2: uncertainty analysis on response surface\n");
      printf("Syntax: rsuab2 (no argument needed)\n");
      printf("This command performs uncertainty analysis on the ");
      printf("response surface\n");
      printf("constructed from the LOADED sample. Uncertainty analysis ");
      printf("is performed\n");
      printf("using a user-provided sample in PSUADE data format ");
      printf("(created by running\n");
      printf("psuade on an input file). If you select a stochastic ");
      printf("response surface\n");
      printf("(Kriging, MARSB, or polynomial regression), the effect ");
      printf("of response\n");
      printf("surface uncertainty will be shown on the PDF and CDF plots.)\n");
      printf("NOTE: This command is more general than rsua and rsuab by ");
      printf("allowing\n");
      printf("      users to add a discrepancy model.\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    int   discFile=0, nInps, nOuts, dnInps;
    char  discFileName[1001], uaFileName[1001];
    PsuadeData *discIO=NULL, *sampleIO=NULL;
    FuncApprox **faPtrsRsEval=NULL;
     
    //**/ query user for output ID
    sscanf(lineIn,"%s %s", command, winput);
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    //**/ optional: discrepancy model
    faPtrsRsEval = new FuncApprox*[2];
    faPtrsRsEval[0] = NULL;
    faPtrsRsEval[1] = NULL;
    printf("Use discrepancy model (in PSUADE data format)? ('y' or 'n') ");
    scanf("%s", winput);
    fgets(lineIn2, 500, stdin);
    if (winput[0] == 'y')
    {
      discFile = 1;
      printf("Enter discrepancy model file (in PSUADE data format): ");
      scanf("%s", discFileName);
      fgets(lineIn2, 500, stdin);
      discIO = new PsuadeData();
      status = discIO->readPsuadeFile(discFileName);
      if (status != 0)
      {
        printf("ERROR: Cannot read discrepancy model file.\n");
        delete [] faPtrsRsEval;
        delete discIO;
        return 1;
      }
      discIO->getParameter("input_ninputs", pPtr);
      dnInps = pPtr.intData_;
      if (dnInps < nInputs)
      {
        printf("Discrepancy model has %d inputs. So the first\n", dnInps);
        printf("%d inputs in the model file will be assumed to\n", dnInps);
        printf("be associated with the inputs of the discrepancy model.\n");
      }
      discIO->getParameter("output_noutputs", pPtr);
      nOuts = pPtr.intData_;
      if (nOuts > 1)
      {
        printf("The discrepancy model has nOutputs > 1.\n");
        printf("This is currently not supported.\n");
        printf("Use 'odelete' to modify your discrepancy model file.\n");
        delete [] faPtrsRsEval;
        delete discIO;
        return 1;
      }
      printf("** CREATING RESPONSE SURFACE FOR DISCREPANCY MODEL\n");
      faPtrsRsEval[1] = genFAInteractive(discIO, 3);
      delete discIO;
      discIO = NULL;
    }

    //**/ request or generate a sample for evaluation
    printf("A sample is needed from you to propagate through the RS.\n");
    printf("Select between the two options below: \n");
    printf("1. PSUADE will generate the sample\n");
    printf("2. User will provide the sample (in PSUADE data format)\n");
    snprintf(pString,100,"Enter 1 or 2 : ");
    int samSrc = getInt(1, 2, pString);

    //**/ generate a sample or get from user a sample for evaluation 
    //**/ ==> usNSams, vecUAInps
    int uaNSams;
    psVector  vecUAInps, vecUAOuts;
    psIVector vecUAStas;
    if (samSrc == 1)
    {
      printf("PSUADE will generate a sample for uncertainty analysis.\n");
      snprintf(pString,100,"Sample size ? (10000 - 100000) ");
      uaNSams = getInt(10000, 100000, pString);
      vecUAInps.setLength(uaNSams * nInputs);
      psuadeIO->getParameter("ana_use_input_pdfs", pPtr);
      int usePDFs = pPtr.intData_;
      if (usePDFs == 1)
      {
        printf("NOTE: Some inputs have non-uniform PDFs.\n");
        printf("      A MC sample will be created with these PDFs.\n");
        psuadeIO->getParameter("method_sampling", pPtr);
        kk = pPtr.intData_;
        psuadeIO->updateMethodSection(PSUADE_SAMP_MC,-1,-1,-1,-1);
        PDFManager *pdfman = new PDFManager();
        pdfman->initialize(psuadeIO);
        vecUAInps.setLength(uaNSams*nInputs);
        psVector vecLs, vecUs;
        vecUs.load(nInputs, iUpperB);
        vecLs.load(nInputs, iLowerB);
        pdfman->genSample(uaNSams, vecUAInps, vecLs, vecUs);
        psuadeIO->updateMethodSection(kk,-1,-1,-1,-1);
        delete pdfman;
      }
      else
      {
        printAsterisks(PL_INFO, 0);
        printf("NOTE: Uniform distribution is assumed for all inputs. ");
        printf("To use other\n");
        printf("      than uniform distributions, prescribe them in ");
        printf("the sample file\n");
        printf("      and set use_input_pdfs in the ANALYSIS section.\n");
        printAsterisks(PL_INFO, 0);
        Sampling *samPtr;
        if (nInputs < 51)
             samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LPTAU);
        else samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
        samPtr->setPrintLevel(0);
        samPtr->setInputBounds(nInputs, iLowerB, iUpperB);
        samPtr->setOutputParams(1);
        samPtr->setSamplingParams(uaNSams, -1, -1);
        samPtr->initialize(0);
        vecUAOuts.setLength(uaNSams);
        vecUAStas.setLength(uaNSams);
        samPtr->getSamples(uaNSams,nInputs,1,vecUAInps.getDVector(),
                     vecUAOuts.getDVector(),vecUAStas.getIVector());
        delete samPtr;
      }
    }
    else
    {
      printf("Enter UA sample file name (in PSUADE data format): ");
      char uaFileName[1001];
      scanf("%s", uaFileName);
      fgets(lineIn2, 500, stdin);
      PsuadeData *sampleIO = new PsuadeData();
      status = sampleIO->readPsuadeFile(uaFileName);
      if (status != 0)
      {
        printf("ERROR: Cannot read sample file.\n");
        delete sampleIO;
        if (faPtrsRsEval[1] == NULL) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        return 1;
      }
      sampleIO->getParameter("input_ninputs", pPtr);
      kk = pPtr.intData_;
      if (kk != nInputs)
      {
        printf("ERROR: Sample nInputs mismatch.\n");
        printf(":      Input size in workspace     = %d.\n",nInputs);
        printf(":      Input size from your sample = %d.\n",kk);
        delete sampleIO;
        if (faPtrsRsEval[1] == NULL) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        return 1;
      }
      sampleIO->getParameter("method_nsamples", pPtr);
      uaNSams = pPtr.intData_;
      if (uaNSams < 1000)
      {
        printf("ERROR: Your sample size should be at least 1000 to give\n");
        printf("       any reasonable UA results.\n");
        delete sampleIO;
        if (faPtrsRsEval[0] == NULL) delete faPtrsRsEval[0];
        if (faPtrsRsEval[1] == NULL) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        return 1;
      }
      sampleIO->getParameter("input_sample", pPtr);
      vecUAInps.load(uaNSams * nInputs, pPtr.dbleArray_);
      pPtr.clean();
      delete sampleIO;
    }

    //**/ ask for how to do UQ
    int uaMethod=0;
    int includeRSErr=0, numBS=1;
    printf("Include response surface uncertainties in UA? (y or n) ");
    scanf("%s", winput);
    fgets(lineIn2, 500, stdin);
    if (winput[0] == 'y')
    {
      includeRSErr = 1;
      printf("Three options are available for including RS uncertainties:\n");
      printf("1. use bootstrapping + RS (or deterministic RS, e.g. MARS)\n");
      printf("2. use stochastic RS (Kriging, MARS+Bootstrap, regression)\n");
      printf("3. use (2) but perform worst-case analysis (2 - average case)\n");
      snprintf(pString,100,"Select 1, 2, or 3 : ");
      uaMethod = getInt(1, 3, pString);
      if (uaMethod == 1)
      {
        snprintf(pString,100,"How many bootstrapped samples to use (10 - 300) : ");
        numBS = getInt(10, 300, pString);
      }
    }

    //**/ ====================================================================
    // perform UA
    //**/ ====================================================================
    psVector vecUAStds;
    vecUAOuts.setLength(uaNSams);
    vecUAStds.setLength(uaNSams);

    //**/ ----------------------------------------------
    // use deterministic 
    //**/ ----------------------------------------------
    if (uaMethod == 0)
    {
      //**/ generate response surface 
      printf("** CREATING RESPONSE SURFACE FOR PRIMARY MODEL\n");
      faPtrsRsEval[0] = genFA(-1, nInputs, -1, nSamples);
      if (faPtrsRsEval[0] == NULL)
      {
        printf("ERROR: Cannot generate response surface.\n");
        if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        delete sampleIO;
        return 1;
      }
      faPtrsRsEval[0]->setBounds(iLowerB, iUpperB);
      faPtrsRsEval[0]->setOutputLevel(0);
      psConfig_.InteractiveSaveAndReset();
      status = faPtrsRsEval[0]->initialize(sampleInputs,sampleOutputs);
      psConfig_.InteractiveRestore();
      if (status != 0)
      {
        printf("ERROR: Cannot initialize response surface.\n");
        if (faPtrsRsEval[0] != NULL) delete faPtrsRsEval[0];
        if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        delete sampleIO;
        return 1;
      }
      //**/ evaluate response surface at the user sample points
      faPtrsRsEval[0]->evaluatePoint(uaNSams,vecUAInps.getDVector(),
                                     vecUAOuts.getDVector());
      //**/ add discrepancy, if available
      double *UAInps = vecUAInps.getDVector();
      if (discFile == 1)
      {
        for (ss = 0; ss < uaNSams; ss++)
        {
          ddata = faPtrsRsEval[1]->evaluatePoint(&UAInps[ss*nInputs]);
          vecUAOuts[ss] += ddata;
        }
      }
       
      //**/ compute statistics 
      double mean=0, stdev=0;
      for (ss = 0; ss < uaNSams; ss++) mean += vecUAOuts[ss];
      mean /= (double) uaNSams;
      for (ss = 0; ss < uaNSams; ss++)
        stdev += pow(vecUAOuts[ss]-mean, 2.0);
      stdev = sqrt(stdev/(double) uaNSams);
      printAsterisks(PL_INFO, 0);
      printf("Sample mean    = %e (without RS uncertainties)\n", mean);
      printf("Sample std dev = %e (without RS uncertainties)\n", stdev);
      printEquals(PL_INFO, 0);

      //**/ generate matlab file
      fp = NULL;
      fp = fopen("matlabrsuab2.m", "w");
      if (fp != NULL)
      {
        fprintf(fp,"Y = [\n");
        for (ss = 0; ss < uaNSams; ss++) 
          fprintf(fp,"%e\n",vecUAOuts[ss]);
        fprintf(fp, "];\n");
        fwriteHold(fp, 0);
        fprintf(fp,"subplot(1,2,1);\n");
        fprintf(fp,"[nk,xk] = hist(Y,50);\n");
        fprintf(fp,"nk = nk / %d;\n", uaNSams);
        fprintf(fp,"bar(xk,nk,1.0);\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Probability Distribution");
        fwritePlotXLabel(fp,"Output Value");
        fwritePlotYLabel(fp,"Probabilities");
        fprintf(fp,"text(0.05,0.9,'Mean = %12.4e','sc','FontSize',11)\n",
                mean);
        fprintf(fp,"text(0.05,0.85,'Std  = %12.4e','sc','FontSize',11)\n",
                stdev);
        fprintf(fp,"subplot(1,2,2);\n");
        fprintf(fp, "Y = sort(Y);\n");
        fprintf(fp, "X = (1 : %d)' / %d;\n", uaNSams, uaNSams);
        fprintf(fp,"plot(Y, X, 'lineWidth',3)\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Cumulative Distribution");
        fwritePlotXLabel(fp, "Output Value");
        fwritePlotYLabel(fp, "Cum. Prob.");
        fclose(fp);
        printf("Output distribution plots are in matlabrsuab2.m.\n");
      }
    }

    //**/ ----------------------------------------------
    // bootstrapped method
    //**/ ----------------------------------------------
    if (uaMethod == 1)
    {
      int bsnSams, rsMethod;
      //**/ create response surface place holder 
      printf("** CREATING RESPONSE SURFACE FOR PRIMARY MODEL\n");
      faPtrsRsEval[0] = genFA(-1, nInputs, -1, nSamples);
      if (faPtrsRsEval[0] == NULL)
      {
        printf("ERROR: Cannot generate primary response surface.\n");
        if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        delete sampleIO;
        return 1;
      }
      rsMethod = faPtrsRsEval[0]->getID(); 
      delete faPtrsRsEval[0];
      faPtrsRsEval[0] = NULL;
        
      //**/ for each bootstrap, initialize and evaluate response surface 
      int its;
      psVector vecBsSamInps, vecBsSamOuts, vecBsMeans, vecBsStds;
      vecBsSamInps.setLength(nSamples*nInputs);
      vecBsSamOuts.setLength(nSamples);
      vecBsMeans.setLength(numBS);
      vecBsStds.setLength(numBS);
      psIVector vecUseFlags;
      vecUseFlags.setLength(nSamples);

      fp = NULL;
      fp = fopen("matlabrsbua.m", "w");
      for (its = 0; its < numBS; its++)
      {
        for (ss = 0; ss < nSamples; ss++) vecUseFlags[ss] = 0;
        //**/ generate bootstrapped sample
        bsnSams = 0;
        for (ss = 0; ss < nSamples; ss++) 
        {
          jj = PSUADE_rand() % nSamples;
          if (vecUseFlags[jj] == 0)
          {
            for (ii = 0; ii < nInputs; ii++)
              vecBsSamInps[bsnSams*nInputs+ii] = sampleInputs[jj*nInputs+ii];
            vecBsSamOuts[bsnSams] = sampleOutputs[jj*nOutputs+outputID];
            vecUseFlags[jj] = 1;
            bsnSams++;
          }
        }
        printf("Bootstrap %d has sample size = %d (%d)\n",its+1,bsnSams,
               nSamples);
        //**/ initialize response surface
        psConfig_.InteractiveSaveAndReset();
        faPtrsRsEval[0] = genFA(rsMethod, nInputs, -1, bsnSams);
        faPtrsRsEval[0]->setBounds(iLowerB, iUpperB);
        faPtrsRsEval[0]->setOutputLevel(0);
        status = faPtrsRsEval[0]->initialize(vecBsSamInps.getDVector(),
                                             vecBsSamOuts.getDVector());
        psConfig_.InteractiveRestore();
        if (status != 0)
        {
          printf("ERROR: In initializing response surface (1).\n");
          if (faPtrsRsEval[0] != NULL) delete faPtrsRsEval[0];
          if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
          delete [] faPtrsRsEval;
          delete sampleIO;
          return 1;
        } 
        //**/ evaluate the user sample
        faPtrsRsEval[0]->evaluatePoint(uaNSams,vecUAInps.getDVector(),
                                       vecUAOuts.getDVector());
        delete faPtrsRsEval[0];
        faPtrsRsEval[0] = NULL;
        //**/ add discrepancy to evaluated sample
        double *UAInps = vecUAInps.getDVector();
        if (discFile == 1)
        {
          for (ss = 0; ss < uaNSams; ss++)
          {
            ddata = faPtrsRsEval[1]->evaluatePoint(&UAInps[ss*nInputs]);
            vecUAOuts[ss] += ddata;
          }
        }
        //**/ compute statistics
        vecBsMeans[its] = vecBsStds[its] = 0.0;
        for (ss = 0; ss < uaNSams; ss++) 
          vecBsMeans[its] += vecUAOuts[ss];
        vecBsMeans[its] /= (double) uaNSams;
        for (ss = 0; ss < uaNSams; ss++)
          vecBsStds[its] += pow(vecUAOuts[ss] - vecBsMeans[its], 2.0);
        vecBsStds[its] = sqrt(vecBsStds[its] / uaNSams);
        if (fp != NULL)
        {
          fprintf(fp, "%% bootstrapped samples\n");
          fprintf(fp, "Y = [\n");
          for (ss = 0; ss < uaNSams; ss++) fprintf(fp,"%e\n",vecUAOuts[ss]);
          fprintf(fp, "];\n");
          fprintf(fp, "Y%d = sort(Y);\n",its+1);
          fprintf(fp, "X%d = (1 : %d)';\n", its+1, uaNSams);
          fprintf(fp, "X%d = X%d / %d;\n", its+1, its+1, uaNSams);
          if (its == 0)
          {
            fprintf(fp, "YY = Y%d;\n", its+1);
            fprintf(fp, "XX = X%d;\n", its+1);
          }
          else
          {
            fprintf(fp, "YY = [YY Y%d];\n", its+1);
            fprintf(fp, "XX = [XX X%d];\n", its+1);
          }
        }
      }
      //**/ compute statistics 
      printAsterisks(PL_INFO, 0);
      double mean, stdev;
      mean = stdev = 0.0;
      for (its = 0; its < numBS; its++) mean += vecBsMeans[its];
      mean /= (double) numBS;
      for (ss = 0; ss < numBS; ss++) stdev += pow(vecBsMeans[ss]-mean, 2.0);
      stdev = sqrt(stdev/(double) numBS);
      printf("Sample mean    = %e (std = %e)\n", mean, stdev);
      mean = stdev = 0.0;
      for (its = 0; its < numBS; its++) mean += vecBsStds[its];
      mean /= (double) numBS;
      for (ss = 0; ss < numBS; ss++) stdev += pow(vecBsStds[ss]-mean, 2.0);
      stdev = sqrt(stdev/(double) numBS);
      printf("Sample std dev = %e (std = %e)\n", mean, stdev);
      printEquals(PL_INFO, 0);
      if (fp != NULL)
      {
        fwriteHold(fp, 0);
        fprintf(fp,"subplot(1,2,1);\n");
        fprintf(fp,"[nk,xk] = hist(YY,50);\n");
        fprintf(fp,"plot(xk,nk, 'lineWidth',2)\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Probability Distribution");
        fwritePlotXLabel(fp,"Output Value");
        fwritePlotYLabel(fp,"Probabilities");
        fprintf(fp,"subplot(1,2,2);\n");
        fprintf(fp,"plot(YY, XX, 'lineWidth',3)\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Cumulative Distribution");
        fwritePlotXLabel(fp, "Output Value");
        fwritePlotYLabel(fp, "Probabilities");
        fclose(fp);
        printf("Output distribution plot file = matlabrsbua2.m.\n");
      }
    }
     
    //**/ ----------------------------------------------
    // use stochastic response surface with average case analysis
    //**/ ----------------------------------------------
    if (uaMethod == 2)
    {
      //**/ create response surface
      psConfig_.InteractiveSaveAndReset();
      printf("** CREATING RESPONSE SURFACE FOR PRIMARY MODEL\n");
      faPtrsRsEval[0] = genFA(-1, nInputs, -1, nSamples);
      if (faPtrsRsEval[0] == NULL)
      {
        printf("ERROR: Cannot generate response surface.\n");
        if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        delete sampleIO;
        return 1;
      }
      faPtrsRsEval[0]->setBounds(iLowerB, iUpperB);
      faPtrsRsEval[0]->setOutputLevel(0);
      status = faPtrsRsEval[0]->initialize(sampleInputs,sampleOutputs);
      psConfig_.InteractiveRestore();
      if (status != 0)
      {
        printf("ERROR: Cannot initialize response surface.\n");
        if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        delete sampleIO;
        return 1;
      }
      //**/ evaluate response surface
      faPtrsRsEval[0]->evaluatePointFuzzy(uaNSams,vecUAInps.getDVector(),
                          vecUAOuts.getDVector(),vecUAStds.getDVector());
      //**/ add discrepancy to original sample
      double discSamStd, *UAInps = vecUAInps.getDVector();
      if (discFile == 1)
      {
        for (ss = 0; ss < uaNSams; ss++)
        {
          vecUAOuts[ss] += 
            faPtrsRsEval[1]->evaluatePointFuzzy(&UAInps[ss*nInputs],
                                                discSamStd);
          ddata = pow(vecUAStds[ss],2.0) + discSamStd * discSamStd;
          vecUAStds[ss] = sqrt(ddata);
        }
      }

      fp = fopen("rsuab2_sample","w");
      if (fp != NULL)
      {
        fprintf(fp,"%% This file is primarily for diagnostics and\n");
        fprintf(fp,"%% expert analysis\n");
        fprintf(fp,"%% First line: nSamples nInputs\n");
        fprintf(fp,"%% All inputs, output, output-3*sigma, output+3*sigma\n");
        fprintf(fp,"%d %d 3\n", uaNSams, nInputs);
        for (ss = 0; ss < uaNSams; ss++)
        {
          for (ii = 0; ii < nInputs; ii++)
            fprintf(fp, "%e ", vecUAInps[ss*nInputs+ii]);
          fprintf(fp, "%e ",  vecUAOuts[ss]);
          fprintf(fp, "%e ",  vecUAOuts[ss]-3*vecUAStds[ss]);
          fprintf(fp, "%e\n", vecUAOuts[ss]+3*vecUAStds[ss]);
        }
        fclose(fp);
        printf("The outputs and stds of your sample has been written ");
        printf("to 'rsuab2_sample'.\n");
      }

      //**/ first set of statistics 
      double mean=0, stdev=0;
      for (ss = 0; ss < uaNSams; ss++) mean += vecUAOuts[ss];
      mean /= (double) uaNSams;
      for (ss = 0; ss < uaNSams; ss++)
        stdev += pow(vecUAOuts[ss]-mean, 2.0);
      stdev = sqrt(stdev/(double) uaNSams);
      printAsterisks(PL_INFO, 0);
      printf("Sample mean    = %e (RS uncertainties not included)\n", mean);
      printf("Sample std dev = %e (RS uncertainties not included)\n", stdev);
      printEquals(PL_INFO, 0);

      //**/ initialize for binning 
      int    nbins = 100, ntimes=20;
      int    **Fcounts = new int*[ntimes+1];
      double Fmax=-PSUADE_UNDEFINED;
      double Fmin=PSUADE_UNDEFINED;
      PDFNormal *rsPDF=NULL;
      for (ss = 0; ss < uaNSams; ss++)
      {
        if (vecUAOuts[ss]+3*vecUAStds[ss] > Fmax)
          Fmax = vecUAOuts[ss] + 3 * vecUAStds[ss];
        if (vecUAOuts[ss]-3*vecUAStds[ss] < Fmin)
          Fmin = vecUAOuts[ss] - 3 * vecUAStds[ss];
      }
      Fmax = Fmax + 0.1 * (Fmax - Fmin);
      Fmin = Fmin - 0.1 * (Fmax - Fmin);
      if (Fmax == Fmin)
      {
        Fmax = Fmax + 0.1 * PABS(Fmax);
        Fmin = Fmin - 0.1 * PABS(Fmin);
      }
      for (ii = 0; ii <= ntimes; ii++)
      {
        Fcounts[ii] = new int[nbins];
        for (kk = 0; kk < nbins; kk++) Fcounts[ii][kk] = 0;
      }

      //**/ generate stochastic RS and bin
      double d1, d2;
      psVector vecSamOutTime, vecSamOutSave;
      vecSamOutTime.setLength(ntimes*nInputs);
      vecSamOutSave.setLength(ntimes*uaNSams);
      for (ss = 0; ss < uaNSams; ss++)
      {
        if (vecUAStds[ss] == 0)
        {
          for (ii = 0; ii < ntimes; ii++) 
            vecSamOutTime[ii] = vecUAOuts[ss];
        }
        else
        {
          rsPDF = new PDFNormal(vecUAOuts[ss],vecUAStds[ss]);
          d1 = vecUAOuts[ss] - 3.0 * vecUAStds[ss];
          d2 = vecUAOuts[ss] + 3.0 * vecUAStds[ss];
          rsPDF->genSample(ntimes,vecSamOutTime.getDVector(),&d1,&d2);
          delete rsPDF;
        }
        for (ii = 0; ii < ntimes; ii++) 
          vecSamOutSave[ss*ntimes+ii] = vecSamOutTime[ii];

        //**/ bin the original sample
        ddata = vecUAOuts[ss] - Fmin;
        if (Fmax > Fmin) ddata = ddata / ((Fmax - Fmin) / nbins);
        else             ddata = nbins / 2;
        kk = (int) ddata;
        if (kk < 0)      kk = 0;
        if (kk >= nbins) kk = nbins - 1;
        Fcounts[ntimes][kk]++;

        //**/ bin the perturbed sample
        for (ii = 0; ii < ntimes; ii++)
        {
          ddata = vecSamOutTime[ii] - Fmin;
          if (Fmax > Fmin)
               ddata = ddata / ((Fmax - Fmin) / nbins);
          else ddata = nbins / 2;
          kk = (int) ddata;
          if (kk < 0)      kk = 0;
          if (kk >= nbins) kk = nbins - 1;
          Fcounts[ii][kk]++;
        }
      }
      double mean2=0, stdev2=0;
      for (ss = 0; ss < uaNSams*ntimes; ss++) mean2 += vecSamOutSave[ss];
      mean2 /= (double) (uaNSams*ntimes);
      stdev2 = 0.0;
      for (ss = 0; ss < uaNSams*ntimes; ss++)
        stdev2 += pow(vecSamOutSave[ss] - mean2, 2.0);
      stdev2 = sqrt(stdev2/(double) (uaNSams*ntimes));
      printf("Sample mean    = %e (RS uncertainties included)\n", mean2);
      printf("Sample std dev = %e (RS uncertainties included)\n", stdev2);
      printAsterisks(PL_INFO, 0);

      //**/ write to file
      fp = fopen("matlabrsuab2.m", "w");
      if (fp == NULL)
      {
        printf("INFO: Cannot write the PDFs/CDFs to matlab file.\n");
      }
      else
      {
        fwriteHold(fp, 0);
        fprintf(fp, "subplot(2,2,1)\n");
        fprintf(fp, "X = [\n");
        for (kk = 0; kk < nbins; kk++)
          fprintf(fp, "%e\n",(Fmax-Fmin)/nbins*(0.5+kk)+Fmin);
        fprintf(fp, "];\n");
        for (ii = 0; ii <= ntimes; ii++)
        {
          fprintf(fp, "N%d = [\n", ii+1);
          for (kk = 0; kk < nbins; kk++)
            fprintf(fp, "%d\n",  Fcounts[ii][kk]);
          fprintf(fp, "];\n");
        }
        fprintf(fp, "N = [");
        for (ii = 0; ii <= ntimes; ii++)
          fprintf(fp, "N%d/sum(N%d) ", ii+1, ii+1);
        fprintf(fp, "];\n");
        fprintf(fp, "NA = N(:,%d+1);\n",ntimes);
        fprintf(fp, "NA = NA / sum(NA);\n");
        fprintf(fp, "NB = sum(N(:,1:%d)');\n",ntimes);
        fprintf(fp, "NB = NB' / sum(NB);\n");
        fprintf(fp, "NN = [NA NB];\n");
        fprintf(fp, "bar(X,NA,1.0)\n");
        fprintf(fp, "ymin = min(min(NA),min(NB));\n");
        fprintf(fp, "ymax = max(max(NA),max(NB));\n");
        fprintf(fp, "axis([min(X) max(X) ymin ymax])\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Prob. Dist. (means of RS)");
        fwritePlotXLabel(fp, "Output Value");
        fwritePlotYLabel(fp, "Probabilities)");
        fprintf(fp,"text(0.05,0.9,'Mean = %12.4e','sc','FontSize',11)\n",
                mean);
        fprintf(fp,"text(0.05,0.85,'Std  = %12.4e','sc','FontSize',11)\n",
                stdev);
        if (faType == PSUADE_RS_MARS)
        {
          printf("Deterministic RS used ==> no RS uncertainties.\n");
        }
        else
        {
          fprintf(fp,"subplot(2,2,3)\n");
          fprintf(fp,"bar(X,NB,1.0)\n");
          fprintf(fp,"axis([min(X) max(X) ymin ymax])\n");
          fwritePlotAxes(fp);
          fwritePlotTitle(fp,"Prob. Dist. (RS with uncertainties)");
          fwritePlotXLabel(fp,"Output Value");
          fwritePlotYLabel(fp,"Probabilities)");
          fprintf(fp,"text(0.05,0.9,'Mean = %12.4e','sc','FontSize',11)\n",
                  mean2);
          fprintf(fp,"text(0.05,0.85,'Std  = %12.4e','sc','FontSize',11)\n",
                  stdev2);
        }
        for (ii = 0; ii <= ntimes; ii++)
        {
          fprintf(fp,"for ii = 2 : %d\n", nbins);
          fprintf(fp,"  N%d(ii) = N%d(ii) + N%d(ii-1);\n",ii+1,ii+1,ii+1);
          fprintf(fp,"end;\n");
        }
        fprintf(fp, "N = [");
        for (ii = 0; ii <= ntimes; ii++)
          fprintf(fp,"N%d/N%d(%d) ", ii+1, ii+1, nbins);
        fprintf(fp, "];\n");
        fprintf(fp, "subplot(2,2,[2 4])\n");
        fprintf(fp, "NA = N(:,%d+1);\n",ntimes);
        fprintf(fp, "NA = NA / NA(%d);\n",nbins);
        fprintf(fp, "NB = sum(N(:,1:%d)');\n",ntimes);
        fprintf(fp, "NB = NB' / NB(%d);\n", nbins);
        fprintf(fp, "NN = [NA NB];\n");
        if (faType == PSUADE_RS_MARS)
        {
          fprintf(fp, "plot(X,NA,'linewidth',3)\n");
          fwritePlotTitle(fp,"Cum. Dist.: (b) mean; (g) with uncertainties");
        }
        else
        {
          fprintf(fp, "plot(X,NN,'linewidth',3)\n");
          fwritePlotTitle(fp,"Cum. Dist.: (*) uncertainties unavailable");
        }
        fwritePlotAxes(fp);
        fwritePlotXLabel(fp, "Output Value");
        fwritePlotYLabel(fp, "Probabilities");
        fclose(fp);
        printf("Output distribution plot file = matlabrsbua.m.\n");
      }
      for (ii = 0; ii <= ntimes; ii++) delete [] Fcounts[ii];
      delete [] Fcounts;
    }

    //**/ ----------------------------------------------
    // use stochastic response surface with worst case analysis
    //**/ ----------------------------------------------
    if (uaMethod == 3)
    {
      //**/ create response surface
      printf("** CREATING RESPONSE SURFACE FOR PRIMARY MODEL\n");
      psConfig_.InteractiveSaveAndReset();
      faPtrsRsEval[0] = genFA(-1, nInputs, -1, nSamples);
      if (faPtrsRsEval[0] == NULL)
      {
        printf("ERROR: Cannot generate response surface.\n");
        if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        delete sampleIO;
        return 1;
      }
      faPtrsRsEval[0]->setBounds(iLowerB, iUpperB);
      faPtrsRsEval[0]->setOutputLevel(0);
      status = faPtrsRsEval[0]->initialize(sampleInputs,sampleOutputs);
      psConfig_.InteractiveRestore();
      if (status != 0)
      {
        printf("ERROR: Cannot initialize response surface.\n");
        if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        delete sampleIO;
        return 1;
      }
        
      //**/ create response surface
      faPtrsRsEval[0]->evaluatePointFuzzy(uaNSams,vecUAInps.getDVector(),
                          vecUAOuts.getDVector(),vecUAStds.getDVector());
        
      //**/ add discrepancy to original sample
      double discSamStd, *UAInps = vecUAInps.getDVector();
      if (discFile == 1)
      {
        for (ss = 0; ss < uaNSams; ss++)
        {
          vecUAOuts[ss] += 
            faPtrsRsEval[1]->evaluatePointFuzzy(&UAInps[ss*nInputs],
                                                discSamStd);
          ddata = pow(vecUAStds[ss],2.0) + discSamStd * discSamStd;
          vecUAStds[ss] = sqrt(ddata);
        }
      }
        
      //**/ first set of statistics 
      double mean=0, stdev=0;
      for (ss = 0; ss < uaNSams; ss++) mean += vecUAOuts[ss];
      mean /= (double) uaNSams;
      for (ss = 0; ss < uaNSams; ss++)
        stdev += pow(vecUAOuts[ss]-mean, 2.0);
      stdev = sqrt(stdev/(double) uaNSams);
      printAsterisks(PL_INFO, 0);
      printf("Sample mean    = %e (RS uncertainties not included)\n",mean);
      printf("Sample std dev = %e (RS uncertainties not included)\n",stdev);
      printEquals(PL_INFO, 0);

      fp = fopen("rsuab2_sample","w");
      fprintf(fp, "%% This file is primarily for diagnostics and \n");
      fprintf(fp, "%% expert analysis\n");
      fprintf(fp, "%% First line: nSamples nInputs\n");
      fprintf(fp, "%% All inputs, output, output-3*sigma, output+3*sigma\n");
      fprintf(fp, "%d %d 3\n", uaNSams, nInputs);
      for (ss = 0; ss < uaNSams; ss++)
      {
        for (ii = 0; ii < nInputs; ii++)
          fprintf(fp, "%e ", vecUAInps[ss*nInputs+ii]);
        fprintf(fp, "%e ", vecUAOuts[ss]);
        fprintf(fp, "%e ", vecUAOuts[ss]-3*vecUAStds[ss]);
        fprintf(fp, "%e\n", vecUAOuts[ss]+3*vecUAStds[ss]);
      }
      fclose(fp);
        
      //**/ initialize for binning 
      int    nbins = 100, ntimes=7;
      int    **Fcounts = new int*[ntimes+1];
      double Fmax=-PSUADE_UNDEFINED;
      double Fmin=PSUADE_UNDEFINED;
      PDFNormal *rsPDF=NULL;
      for (ss = 0; ss < uaNSams; ss++)
      {
        if (vecUAOuts[ss]+3*vecUAStds[ss] > Fmax)
          Fmax = vecUAOuts[ss] + 3 * vecUAStds[ss];
        if (vecUAOuts[ss]-3*vecUAStds[ss] < Fmin)
          Fmin = vecUAOuts[ss] - 3 * vecUAStds[ss];
      }
      Fmax = Fmax + 0.1 * (Fmax - Fmin);
      Fmin = Fmin - 0.1 * (Fmax - Fmin);
      if (Fmax == Fmin)
      {
        Fmax = Fmax + 0.1 * PABS(Fmax);
        Fmin = Fmin - 0.1 * PABS(Fmin);
      }
      for (ii = 0; ii <= ntimes; ii++)
      {
        Fcounts[ii] = new int[nbins];
        for (kk = 0; kk < nbins; kk++) Fcounts[ii][kk] = 0;
      }

      //**/ binning 
      for (ss = 0; ss < uaNSams; ss++)
      {
        for (ii = 0; ii < ntimes; ii++)
        {
          ddata = vecUAOuts[ss]+vecUAStds[ss]*(ii-3) - Fmin;
          if (Fmax > Fmin)
               ddata = ddata / ((Fmax - Fmin) / nbins);
          else ddata = nbins / 2;
          kk = (int) ddata;
          if (kk < 0)      kk = 0;
          if (kk >= nbins) kk = nbins - 1;
          Fcounts[ii][kk]++;
        }
      }

      fp = fopen("matlabrsuab2.m", "w");
      if (fp == NULL)
      {
        printf("INFO: Cannot write the PDFs/CDFs to matlab file.\n");
      }
      else
      {
        fwriteHold(fp, 0);
        fprintf(fp, "%% worst case analysis\n");
        fprintf(fp, "X = [\n");
        for (kk = 0; kk < nbins; kk++)
          fprintf(fp, "%e\n", (Fmax-Fmin)/nbins*(0.5+kk)+Fmin);
        fprintf(fp, "];\n");
        for (ii = 0; ii < ntimes; ii++)
        {
          fprintf(fp, "E%d = [\n", ii+1);
          for (kk = 0; kk < nbins; kk++) 
            fprintf(fp, "%d\n",Fcounts[ii][kk]);
          fprintf(fp, "];\n");
        }
        fprintf(fp, "EE = [");
        for (ii = 0; ii < ntimes; ii++)
          fprintf(fp, "E%d/sum(E%d) ", ii+1, ii+1);
        fprintf(fp, "];\n");
        fprintf(fp, "subplot(1,2,1)\n");
        fprintf(fp, "plot(X,EE,'lineWidth',2)\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Prob. Dist. (-3,2,1,0,1,2,3 std.)");
        fwritePlotXLabel(fp, "Output Value");
        fwritePlotYLabel(fp, "Probabilities");
        fprintf(fp, "subplot(1,2,2)\n");
        for (ii = 0; ii < ntimes; ii++)
        {
          fprintf(fp, "for ii = 2 : %d\n", nbins);
          fprintf(fp, "   E%d(ii) = E%d(ii) + E%d(ii-1);\n",ii+1,ii+1,ii+1);
          fprintf(fp, "end;\n");
        }
        fprintf(fp, "EE = [");
        for (ii = 0; ii < ntimes; ii++)
          fprintf(fp, "E%d/E%d(%d) ", ii+1, ii+1, nbins);
        fprintf(fp, "];\n");
        fprintf(fp, "plot(X,EE,'linewidth',2)\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Cum. Dist. (-3,2,1,0,1,2,3 std.)");
        fwritePlotXLabel(fp, "Output Value");
        fwritePlotYLabel(fp, "Probabilities");
        fclose(fp);
        printf("Output distribution plot file = matlabrsuab2.m.\n");
        for (ii = 0; ii < ntimes; ii++) delete [] Fcounts[ii];
        delete [] Fcounts;
      }
    }
    if (faPtrsRsEval[0] != NULL) delete faPtrsRsEval[0];
    if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
    delete [] faPtrsRsEval;
    if (sampleIO != NULL) delete sampleIO;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rs_ua2 
  //**/ uncertainty analysis on fuzzy response surface (worst case)
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rs_ua2"))
  {
    printf("This command has been replaced by rsua or rsuab.\n");
    return 0;
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rs_ua2: uncertainty analysis on response surface (worst case)\n");
      printf("syntax: rs_ua2 (no argument needed)\n");
      printf("This command perform uncertainty analysis on the response\n");
      printf("surface built from the loaded sample. If you select a\n");
      printf("stochastic response surface type (Kriging, MARSB, or\n");
      printf("polynomial regression, the effect of response surface\n");
      printf("uncertainty will be shown on the PDF and CDF plots.\n");
      printf("This is a worst case analysis in the sense that the each\n");
      printf("histogram is constructed from perturbing each sample point\n");
      printf("with the same fraction of its standard deviation.\n");
      printf("NOTE: This analysis supports non-uniform distributions\n");
      printf("      for the inputs. Simply prescribe the distributions in\n");
      printf("      the data file and turn on use_input_pdfs in ANALYSIS.\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    //**/ fetch sample information
    Sampling   *samPtr;
    FuncApprox *faPtr;
    PDFManager *pdfman;
    psVector   vecOut, vecLower, vecUpper;
    psuadeIO->getParameter("ana_rstype", pPtr);
    faType = pPtr.intData_;
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;
    snprintf(pString,100,
           "Sample size for generating distribution? (10000 - 100000) ");
    int nSamp = getInt(10000, 100000, pString);
    flag = 0;
    printf("Save the generated sample in a file? (y or n) ");
    fgets(winput,10,stdin); 
    if (winput[0] == 'y') flag = 1; 

    //**/ create response surface ==> faPtr
    printf("Phase 1 of 4: create response surface\n");
    printf("NOTE: the response surface type is taken from your data file.\n");
    faPtr = genFA(faType, nInputs, -1, nSamples);
    faPtr->setNPtsPerDim(32);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);
    psVector vecYT;
    if (nOutputs > 1)
    {
      vecYT.setLength(nSamples);
      for (ss = 0; ss < nSamples; ss++) 
        vecYT[ss] = sampleOutputs[ss*nOutputs+outputID];
    }
    else vecYT.load(nSamples,sampleOutputs);
    faPtr->initialize(sampleInputs,vecYT.getDVector());
           
    //**/ create a MC sample => nSamp, samInputs
    printEquals(PL_INFO, 0);
    printf("Phase 2 of 4: create MC sample\n");
    psVector vecSamInps, vecSamOuts, vecSamStds;
    vecSamInps.setLength(nSamp*nInputs);
    vecSamOuts.setLength(nSamp);
    vecSamStds.setLength(nSamp);
    double *samInputs  = new double[nInputs*nSamp];
    double *samOutputs = new double[nSamp];
    double *samStds    = new double[nSamp];
    psuadeIO->getParameter("ana_use_input_pdfs", pPtr);
    int usePDFs = pPtr.intData_;
    if (usePDFs == 1)
    {
      printf("NOTE: Some inputs have non-uniform PDFs.\n");
      printf("      A MC sample will be created with these PDFs.\n");
      psuadeIO->getParameter("method_sampling", pPtr);
      kk = pPtr.intData_;
      psuadeIO->updateMethodSection(PSUADE_SAMP_MC,-1,-1,-1,-1);
      pdfman = new PDFManager();
      pdfman->initialize(psuadeIO);
      vecOut.setLength(nSamp*nInputs);
      vecUpper.load(nInputs, iUpperB);
      vecLower.load(nInputs, iLowerB);
      pdfman->genSample(nSamp, vecOut, vecLower, vecUpper);
      for (ii = 0; ii < nSamp*nInputs; ii++) vecSamInps[ii] = vecOut[ii];
      psuadeIO->updateMethodSection(kk,-1,-1,-1,-1);
    }
    else
    {
      printf("NOTE: Uniform distributions will be used for all inputs.\n");
      printf("      To use other than uniform distributions, prescribe\n");
      printf("      them in the data file and set use_input_pdfs in the\n");
      printf("      ANALYSIS section.\n");
      psIVector vecSamStas;
      vecSamStas.setLength(nSamp);
      if (nInputs < 51)
           samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LPTAU);
      else samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
      samPtr->setPrintLevel(0);
      samPtr->setInputBounds(nInputs, iLowerB, iUpperB);
      samPtr->setOutputParams(1);
      samPtr->setSamplingParams(nSamp, -1, -1);
      samPtr->initialize(0);
      samPtr->getSamples(nSamp,nInputs,1,vecSamInps.getDVector(),
                         vecSamOuts.getDVector(),vecSamStas.getIVector());
      delete samPtr;
      samPtr = NULL;
    }

    //**/ evaluate the sample => samOutputs, samStds
    printf("Phase 3 of 4: evaluate sample\n");
    faPtr->evaluatePointFuzzy(nSamp,vecSamInps.getDVector(),
                vecSamOuts.getDVector(),vecSamStds.getDVector()); 
    delete faPtr;
    faPtr = NULL;
    if (flag == 1)
    {
      fp = fopen("rsua2_sample","w");
      fprintf(fp, "%% inputs, output, output-3 sigma, output+3sigma\n");
      fprintf(fp, "%d %d 3\n", nSamp, nInputs);
      for (ss = 0; ss < nSamp; ss++)
      {
        for (ii = 0; ii < nInputs; ii++) 
          fprintf(fp, "%e ", vecSamInps[ss*nInputs+ii]);
        fprintf(fp, "%e ", vecSamOuts[ss]);
        fprintf(fp, "%e ", vecSamOuts[ss]-3*vecSamStds[ss]);
        fprintf(fp, "%e\n", vecSamOuts[ss]+3*vecSamStds[ss]);
      }
      fclose(fp);
      printf("A MC sample has been written to the file 'rsua2_sample'.\n");
    }

    //**/ compute statistics
    double mean=0, stdev=0;
    for (ss = 0; ss < nSamp; ss++) mean += vecSamOuts[ss];
    mean /= (double) nSamp;
    for (ss = 0; ss < nSamp; ss++) 
      stdev += pow(vecSamOuts[ss]-mean, 2.0);
    stdev = sqrt(stdev/(double) nSamp);
    printAsterisks(PL_INFO, 0);
    printf("Sample mean    = %e (RS uncertainties not included)\n", mean);
    printf("Sample std dev = %e (RS uncertainties not included)\n", stdev);
    printEquals(PL_INFO, 0);
    printf("Phase 4 of 4: binning\n");

    //**/ get the bounds for binning purposes => Fcounts
    int    nbins = 100, ntimes=7;
    int    **Fcounts = new int*[ntimes];
    double Fmax=-PSUADE_UNDEFINED;
    double Fmin=PSUADE_UNDEFINED;

    for (ss = 0; ss < nSamp; ss++)
    {
      if (vecSamOuts[ss] > Fmax) Fmax = vecSamOuts[ss];
      if (vecSamOuts[ss] < Fmin) Fmin = vecSamOuts[ss];
      if (vecSamOuts[ss]+3*vecSamStds[ss] > Fmax) 
        Fmax = vecSamOuts[ss] + 3 * vecSamStds[ss];
      if (vecSamOuts[ss]-3*vecSamStds[ss] < Fmin) 
        Fmin = vecSamOuts[ss] - 3 * vecSamStds[ss];
    }
    Fmax = Fmax + 0.1 * (Fmax - Fmin);
    Fmin = Fmin - 0.1 * (Fmax - Fmin);
    if (Fmax == Fmin)
    {
      Fmax = Fmax + 0.1 * PABS(Fmax);
      Fmin = Fmin - 0.1 * PABS(Fmin);
    }
    for (ii = 0; ii < ntimes; ii++)
    {
      Fcounts[ii] = new int[nbins];
      for (kk = 0; kk < nbins; kk++) Fcounts[ii][kk] = 0;
    }

    //**/ get the worst case envelope
    for (ss = 0; ss < nSamp; ss++)
    {
      //**/ bin the samples from stochastic RS
      for (ii = 0; ii < ntimes; ii++) 
      {
        ddata = vecSamOuts[ss]+vecSamStds[ss]*(ii-3) - Fmin;
        if (Fmax > Fmin)
             ddata = ddata / ((Fmax - Fmin) / nbins);
        else ddata = nbins / 2;
        kk = (int) ddata;
        if (kk < 0)      kk = 0;
        if (kk >= nbins) kk = nbins - 1;
        Fcounts[ii][kk]++;
      }
    }

    //**/ partial clean up : everything except Fcounts

    //**/ generate matlab/scilab file
    if (plotScilab())
    {
      fp = fopen("scilabrsua2.sci", "w");
      if (fp == NULL)
      {
         printf("ERROR: Cannot open scilab file.\n");
         for (ii = 0; ii < ntimes; ii++) delete [] Fcounts[ii];
         delete [] Fcounts;
         return 1;
      }
    }
    else
    {
      fp = fopen("matlabrsua2.m", "w");
      if (fp == NULL)
      {
        printf("ERROR: Cannot open matlab file.\n");
        for (ii = 0; ii < ntimes; ii++) delete [] Fcounts[ii];
        delete [] Fcounts;
        return 1;
      }
    }
    fprintf(fp, "X = [\n");
    for (kk = 0; kk < nbins; kk++)
      fprintf(fp, "%e\n", (Fmax-Fmin)/nbins*(0.5+kk)+Fmin);
    fprintf(fp, "];\n");
    fwriteHold(fp,0);
    for (ii = 0; ii < ntimes; ii++)
    {
      fprintf(fp, "E%d = [\n", ii+1);
      for (kk = 0; kk < nbins; kk++) fprintf(fp, "%d\n",  Fcounts[ii][kk]);
      fprintf(fp, "];\n");
    }
    fprintf(fp, "EE = [");
    for (ii = 0; ii < ntimes; ii++)
      fprintf(fp, "E%d/sum(E%d) ", ii+1, ii+1);
    fprintf(fp, "];\n");
    fprintf(fp, "subplot(1,2,1)\n");
    fprintf(fp, "plot(X,EE,'lineWidth',2)\n");
    fwritePlotAxes(fp);
    fwritePlotTitle(fp, "Prob. Dist. (-3,2,1,0,1,2,3 std.)");
    fwritePlotXLabel(fp, "Output Value");
    fwritePlotYLabel(fp, "Probabilities");
    fprintf(fp, "subplot(1,2,2)\n");
    for (ii = 0; ii < ntimes; ii++)
    {
      fprintf(fp, "for ii = 2 : %d\n", nbins);
      fprintf(fp, "   E%d(ii) = E%d(ii) + E%d(ii-1);\n",ii+1,ii+1,ii+1);
      fprintf(fp, "end;\n");
    }
    fprintf(fp, "EE = [");
    for (ii = 0; ii < ntimes; ii++)
      fprintf(fp, "E%d/E%d(%d) ", ii+1, ii+1, nbins);
    fprintf(fp, "];\n");
    fprintf(fp, "plot(X,EE,'linewidth',2)\n");
    fwritePlotAxes(fp);
    fwritePlotTitle(fp, "Cum. Dist. (-3,2,1,0,1,2,3 std.)");
    fwritePlotXLabel(fp, "Output Value");
    fwritePlotYLabel(fp, "Probabilities");
    fclose(fp);
    if (plotScilab())
         printf("Output distribution plot file = scilabrsua2.sci.\n");
    else printf("Output distribution plot file = matlabrsua2.m.\n");
    for (ii = 0; ii < ntimes; ii++) delete [] Fcounts[ii];
    delete [] Fcounts;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsuab3 
  //**/ RS-based UA with bootstrap and can be used with posterior
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsuab3"))
  {
    printf("This command has been replaced by rsua or rsuab.\n");
    return 0;
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsuab3: This is a generic RS-based command that ");
      printf("can accommodate a\n");
      printf("        discrepancy model, a pre-generated sample ");
      printf("(in a file), and\n");
      printf("        bootstrapping. The sample file should ");
      printf("have the following format: \n");
      printf("PSUADE_BEGIN \n");
      printf("<nPts> <nInputs> \n");
      printf("1 <input 1> <input 2> ... \n");
      printf("2 <input 1> <input 2> ... \n");
      printf("...... \n");
      printf("PSUADE_END \n");
      printf("syntax: rs_uab (no argument needed)\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nOutputs > 1)
    {
      printf("Currently this command does not support nOutputs > 1.\n");
      printf("Use 'write' to generate a one-output data file first.\n");
      return 1;
    }
    else
    {
      int    discFile=1,nInps,nOuts,dnSamp,it,ind,nSamples2,*tempI, nbs;
      double mean=0.0, stdev=0.0, dtemp;
      double *outVals, *tempX, *tempY, *tempV, *tempW, *inputVals=NULL;
      PsuadeData *localIO = NULL;
      FuncApprox **faPtrsRsEval=NULL;
      //**/ first set up for response surfaces (2- original and discrepancy)
      outputID = 0;
      faPtrsRsEval = new FuncApprox*[2];
      faPtrsRsEval[0] = NULL;
      faPtrsRsEval[1] = NULL;

      //**/ read the discrepancy model file
      printf("Enter discrepancy model PSUADE file (if none, just 'n'): ");
      scanf("%s", winput);
      fgets(lineIn2,500,stdin); 
      if (winput[0] == 'n') discFile = 0; 
      else
      {
        localIO = new PsuadeData();
        status = localIO->readPsuadeFile(winput);
        if (status == 0)
        {
          localIO->getParameter("input_ninputs", pPtr);
          nInps = pPtr.intData_;
          if (nInps < nInputs)
          {
            printf("Discrepancy model has %d inputs.\n", nInps);
            printf("So the first %d inputs in the model file ",nInps);
            printf("are assumed to associate with the inputs of\n");
            printf("the discrepancy model.\n");
          }
          localIO->getParameter("output_noutputs", pPtr);
          nOuts = pPtr.intData_;
          if (nOuts > 1)
          {
            printf("The discrepancy model has nOutputs > 1.\n");
            printf("This is currently not supported.\n");
            delete [] faPtrsRsEval;
            delete localIO;
            return 1;
          }
          printf("** CREATING RESPONSE SURFACE FOR DISCREPANCY MODEL\n");
          faPtrsRsEval[1] = genFAInteractive(localIO, 3);
          delete localIO;
          localIO = NULL;
        }
        else
        {
          printf("ERROR: In reading the discrepancy model file %s.\n",
                 winput);
          discFile = 0;
          delete [] faPtrsRsEval;
          faPtrsRsEval = NULL;
          delete localIO;
          localIO = NULL;
          return 1;
        }
      }

      //**/ read the user-generated sample
      printf("Enter sample file (in some standard format): ");
      scanf("%s", dataFile);
      fgets(lineIn2,500,stdin); 
      fp = fopen(dataFile, "r");
      if (fp == NULL)
      {
        printf("ERROR: Sample data file %s not found.\n", dataFile);
        if (discFile == 1) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        faPtrsRsEval = NULL;
        if (localIO != NULL) delete localIO;
        localIO = NULL;
        return 1;
      }
      else
      {
        fscanf(fp, "%s", winput);
        if (strcmp(winput, "PSUADE_BEGIN"))
        {
          printf("ERROR: File must begin with PSUADE_BEGIN\n");
          fclose(fp);
          printf("File format: \n");
          printf("PSUADE_BEGIN \n");
          printf("<nPts> <nInputs> \n");
          printf("1 <input 1> <input 2> ... \n");
          printf("2 <input 1> <input 2> ... \n");
          printf("...... \n");
          printf("PSUADE_END \n");
          delete [] faPtrsRsEval;
          faPtrsRsEval = NULL;
          if (localIO != NULL) delete localIO;
          localIO = NULL;
          return 1;
        }
        else
        {
          fscanf(fp, "%d %d", &dnSamp, &kk);
          if (dnSamp <= 0)
          {
            printf("ERROR: Invalid sample size\n");
            fclose(fp);
            delete [] faPtrsRsEval;
            faPtrsRsEval = NULL;
            if (localIO != NULL) delete localIO;
            localIO = NULL;
            return 1;
          }
          if (kk != nInputs)
          {
            printf("ERROR: Input size does not match nInputs.\n");
            printf("       Input size in local memory = %d.\n", 
                   nInputs);
            printf("       Input size from file       = %d.\n",kk);
            fclose(fp);
            delete [] faPtrsRsEval;
            if (localIO != NULL) delete localIO;
            faPtrsRsEval = NULL;
            localIO = NULL;
            return 1;
          }
          psVector vecInpVals;
          vecInpVals.setLength(dnSamp*nInputs);
          inputVals = vecInpVals.getDVector();
          for (jj = 0; jj < dnSamp; jj++)
          {
            fscanf(fp, "%d", &ind);
            if (ind != (jj+1))
            {
              printf("ERROR: Input index mismatch (%d,%d)\n",jj+1,ind);
              printf("       Read     index = %d\n", ind);
              printf("       Expected index = %d\n", jj+1);
              printf("File format: \n");
              printf("PSUADE_BEGIN \n");
              printf("<nPts> <nInputs> \n");
              printf("1 <input 1> <input 2> ... \n");
              printf("2 <input 1> <input 2> ... \n");
              printf("...... \n");
              printf("PSUADE_END \n");
              delete [] faPtrsRsEval;
              if (localIO != NULL) delete localIO;
              faPtrsRsEval = NULL;
              localIO = NULL;
              fclose(fp);
              return 1;
            }
            for (ii = 0; ii < nInputs; ii++)
              fscanf(fp, "%lg", &(inputVals[jj*nInputs+ii]));
          }
          if (jj != dnSamp)
          {
            fscanf(fp, "%s", winput);
            fscanf(fp, "%s", winput);
            if (strcmp(winput, "PSUADE_END"))
            {
              fclose(fp);
              printf("ERROR: File must end with PSUADE_END\n");
              delete [] faPtrsRsEval;
              if (localIO != NULL) delete localIO;
              faPtrsRsEval = NULL;
              localIO = NULL;
              return 1;
            }
          }
          fclose(fp);
        }
           
        //**/ set up for iterations
        snprintf(pString,100,"How many bootstrapped samples to use (10 - 300) : ");
        nbs = getInt(1, 300, pString);
        printf("Write the CDFs to a matlab/scilab file? (y or n) ");
        scanf("%s", winput);
        fgets(lineIn,500,stdin); 
        flag = 0;
        fp = NULL;
        if (winput[0] == 'y')
        {
          if (dnSamp > 50000)
          {
            printf("INFO: Sample size %d too large (>50000) for matlab\n",
                   dnSamp);
            printf("      plot. CDF plots not to be generated.\n");
          }
          else
          {
            flag = 1; 
            if (plotScilab())
            {
              fp = fopen("scilabrsuab_cdf.sci", "w");
              if (fp == NULL)
              {
                printf("ERROR: Cannot open file.\n");
                flag = 0;
              }
              else
              {
                fprintf(fp, "// CDFs for rs_uab\n");
                fwritePlotCLF(fp);
              }
            }
            else
            {
              fp = fopen("matlabrsuab_cdf.m", "w");
              if (fp == NULL)
              {
                printf("ERROR: Cannot open file.\n");
                flag = 0;
              }
              else
              {
                fprintf(fp, "%% CDFs for rs_uab\n");
                fwritePlotCLF(fp);
              }
            }
          }
        }
        //**/ iterate
        if (nbs == 1) nSamples2 = nSamples;
        else
        {
          nSamples2 = (int) (0.9 * nSamples);
          if ((double) nSamples2 / (double) nSamples < 0.9) nSamples2++;
        }
        faPtrsRsEval[0] = genFA(-1, nInputs, -1, nSamples2);
        if (faPtrsRsEval[0] == NULL)
        {
          printf("ERROR: Cannot generate response surface.\n");
          delete [] faPtrsRsEval;
          faPtrsRsEval = NULL;
          if (localIO != NULL) delete localIO;
          localIO = NULL;
          return 1;
        }
        faPtrsRsEval[0]->setBounds(iLowerB, iUpperB);
        faPtrsRsEval[0]->setOutputLevel(0);
        psVector  vecOutVals, vecXT, vecYT, vecWT, vecVT;
        psIVector vecIT;
        vecOutVals.setLength(dnSamp);
        vecXT.setLength(nSamples*nInputs);
        vecYT.setLength(nSamples);
        vecWT.setLength(nbs);
        vecVT.setLength(nbs);
        for (it = 0; it < nbs; it++)
        {
          printf("rs_uab: ITERATION %d\n", it+1);
          //**/ random draw
          if (nbs == 1)
          {
            for (jj = 0; jj < nSamples*nInputs; jj++)
              vecXT[jj] = sampleInputs[jj];
            for (jj = 0; jj < nSamples; jj++)
              vecYT[jj] = sampleOutputs[jj*nOutputs+outputID];
          }
          else
          {   
            for (jj = 0; jj < nSamples; jj++) vecIT[jj] = 0;
            kk = 0;
            while (kk < nSamples2)
            {
              ind = PSUADE_rand() % nSamples;
              if (vecIT[ind] == 0)
              {
                for (ii = 0; ii < nInputs; ii++)
                  vecXT[kk*nInputs+ii] = sampleInputs[ind*nInputs+ii];
                vecYT[kk] = sampleOutputs[ind*nOutputs+outputID];
                vecIT[ind] = 1;
                kk++;
              }
            }
          }
          //**/ add discrepancy to sample
          if (discFile == 1)
          {
            for (jj = 0; jj < nSamples2; jj++)
            {
              double *tx = vecXT.getDVector();
              dtemp = faPtrsRsEval[1]->evaluatePoint(tx);
              vecYT[jj] += dtemp;
            }
          }
          //**/ generate response surface
          status = faPtrsRsEval[0]->initialize(vecXT.getDVector(),
                                               vecYT.getDVector());
          //**/ evaluate the response surface using the user example
          faPtrsRsEval[0]->evaluatePoint(dnSamp, inputVals, 
                                         vecOutVals.getDVector());
          mean = stdev = 0.0;
          for (jj = 0; jj < dnSamp; jj++) mean += vecOutVals[jj];
          mean /= (double) dnSamp;
          for (jj = 0; jj < dnSamp; jj++)
            stdev += pow(vecOutVals[jj] - mean, 2.0);
          stdev = sqrt(stdev / dnSamp);
          vecVT[it] = mean;
          vecWT[it] = stdev;
          if (fp != NULL && flag == 1)
          {
            fprintf(fp, "Y = [\n");
            for (jj = 0; jj < dnSamp; jj++) 
              fprintf(fp,"%e\n",vecOutVals[jj]);
            fprintf(fp, "];\n");
            fprintf(fp, "Y%d = sort(Y);\n",it+1);
            fprintf(fp, "X%d = (1 : %d)';\n", it+1, dnSamp);
            fprintf(fp, "X%d = X%d / %d;\n", it+1, it+1, dnSamp);
            if (it == 0)
            {
              fprintf(fp, "YY = Y%d;\n", it+1);
              fprintf(fp, "XX = X%d;\n", it+1);
            }
            else
            {
              fprintf(fp, "YY = [YY Y%d];\n", it+1);
              fprintf(fp, "XX = [XX X%d];\n", it+1);
            }
          }
        }
        if (fp != NULL)
        {
          fprintf(fp, "plot(YY, XX, 'lineWidth',3)\n");
          fwritePlotTitle(fp, "Cumulative Distribution");
          fwritePlotAxes(fp);
          fwritePlotXLabel(fp, "Output Value");
          fwritePlotYLabel(fp, "Probabilities");
          fclose(fp);
          if (plotScilab())
               printf("rs_uab: scilabrsuab_cdf.sci has the CDF plots.\n");
          else printf("rs_uab: matlabrsuab_cdf.m has the CDF plots.\n");
        }
        delete faPtrsRsEval[0];
        faPtrsRsEval[0] = NULL;
        if (discFile == 1) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        faPtrsRsEval = NULL;
        if (it == nbs && nbs > 1)
        {
          mean = 0.0;
          for (jj = 0; jj < nbs; jj++) mean += vecVT[jj];
          mean /= (double) nbs;
          for (jj = 0; jj < nbs; jj++)
            stdev += pow(vecVT[jj] - mean, 2.0);
          stdev = sqrt(stdev / nbs);
          printf("rs_uab: Sample Mean  = %e (%e)\n", mean, stdev);
          mean = 0.0;
          for (jj = 0; jj < nbs; jj++) mean += vecWT[jj];
          mean /= (double) nbs;
          for (jj = 0; jj < nbs; jj++)
            stdev += pow(vecWT[jj] - mean, 2.0);
          stdev = sqrt(stdev / nbs);
          printf("rs_uab: Sample Stdev = %e (%e)\n", mean, stdev);
        }
        else if (kk == nbs && nbs == 1)
        {
          printf("rs_uab: Sample Mean  = %e\n", vecVT[0]);
          printf("rs_uab: Sample Stdev = %e\n", vecWT[0]);
        }
      }
    }
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsuap 
  //**/ RS-based UA with posterior sample
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsuap_disable"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsuap: This command is similar to 'rsua' except that ");
      printf("it is especially\n");
      printf("       made for uncertainty analyis using a ");
      printf("posterior sample (e.g.\n");
      printf("       MCMCPostSample) and optionally a discrepancy model ");
      printf("produced by\n");
      printf("       the 'rsmcmc' command.\n");
      printf("Syntax: rsuap (no argument needed)\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nOutputs > 1)
    {
      printf("Currently this command does not support nOutputs > 1.\n");
      printf("Use 'write' to choose one output for processing.\n");
      return 1;
    }
    printAsterisks(PL_INFO, 0);
    printf("* Response surface-based Post-MCMC Uncertainty Analysis\n");
    printDashes(PL_INFO, 0);
    printf("This command is similar to 'rsua' except that ");
    printf("it is especially made for\n");
    printf("uncertainty analyis using a posterior sample ");
    printf("(e.g. MCMCPostSample) and\n");
    printf("optionally a discrepancy model produced by 'rsmcmc'.\n");
    printAsterisks(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ construct response surface
    int discFile=1, nInps, nOuts, faFlag;
    FuncApprox **faPtrsRsEval=NULL;
    PsuadeData *localIO=NULL;
    faFlag = 3;
    psuadeIO->getParameter("ana_outputid", pPtr);
    kk = pPtr.intData_;
    outputID = 0;
    faPtrsRsEval = new FuncApprox*[2];
    psuadeIO->updateAnalysisSection(-1, -1, -1, -1, outputID, -1);
    faPtrsRsEval[0] = genFAInteractive(psuadeIO, faFlag);
    faPtrsRsEval[1] = NULL;
    psuadeIO->updateAnalysisSection(-1, -1, -1, -1, kk, -1);

    //**/ read the discrepancy model file
    printf("Enter discrepancy model PSUADE file (if none, enter NONE): ");
    scanf("%s", winput);
    fgets(lineIn2,500,stdin); 
    if (!strcmp(winput, "NONE") || !strcmp(winput,"none") ||
      winput[0] == 'n') discFile = 0; 
    else
    {
      localIO = new PsuadeData();
      status = localIO->readPsuadeFile(winput);
      if (status == 0)
      {
        localIO->getParameter("input_ninputs", pPtr);
        nInps = pPtr.intData_;
        if (nInps < nInputs)
        {
          printf("Discrepancy model has %d inputs.\n", nInps);
          printf("So the first %d inputs in the model file ",nInps);
          printf("are assumed to associate with the inputs of\n");
          printf("the discrepancy model.\n");
        }
        localIO->getParameter("output_noutputs", pPtr);
        nOuts = pPtr.intData_;
        if (nOuts > 1)
        {
          printf("The discrepancy model has nOutputs > 1.\n");
          printf("This is currently not supported.\n");
          delete localIO;
          if (faPtrsRsEval[0] != NULL) delete faPtrsRsEval[0];
          delete [] faPtrsRsEval;
          return 1;
        }
        printf("** CREATING RESPONSE SURFACE FOR DISCREPANCY MODEL\n");
        faPtrsRsEval[1] = genFAInteractive(localIO, 3);
        delete localIO;
      }
      else
      {
        printf("ERROR: In reading the discrepancy model file %s.\n",
               winput);
        discFile = 0;
        delete localIO;
        delete faPtrsRsEval[0];
        delete [] faPtrsRsEval;
        return 1;
      }
      localIO = NULL;
    }

    //**/ read the sample 
    int    dnInps, dnSamp, ind;
    double *inputVals=NULL;
    char   dataFile[1001];
    psVector vecInpVals, vecOutVals;
    printf("Enter sample file (in iread format): ");
    scanf("%s", dataFile);
    fgets(lineIn2,500,stdin); 
    fp = fopen(dataFile, "r");
    if (fp == NULL)
    {
      printf("ERROR: Sample data file %s not found.\n", dataFile);
      if (faPtrsRsEval[0] != NULL) delete faPtrsRsEval[0];
      if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
      delete [] faPtrsRsEval;
      return 1;
    }
    else
    {
      fscanf(fp, "%s", winput);
      if (strcmp(winput, "PSUADE_BEGIN"))
      {
        printf("INFO: no first line found with PSUADE_BEGIN\n");
        fclose(fp);
        fp = fopen(dataFile, "r");
      }
      while (1)
      {
        kk = getc(fp);
        if (kk != '#')
        {
          ungetc(kk, fp);
          break;
        }
      }
      fscanf(fp, "%d %d", &dnSamp, &dnInps);
      if (dnSamp <= 0)
      {
        printf("ERROR: Invalid sample size\n");
        fclose(fp);
        if (faPtrsRsEval[0] != NULL) delete faPtrsRsEval[0];
        if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        return 1;
      }
      printf("Sample size read = %d\n", dnSamp);
      if (dnInps != nInputs)
      {
        printf("ERROR: Input size does not match nInputs.\n");
        printf("       Input size in local memory = %d.\n",nInputs);
        printf("       Input size from file       = %d.\n",dnInps);
        fclose(fp);
        if (faPtrsRsEval[0] != NULL) delete faPtrsRsEval[0];
        if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
        delete [] faPtrsRsEval;
        return 1;
      }
      fgets(lineIn2, 500, fp);
      while (1)
      {
        kk = getc(fp);
        if (kk != '#')
        {
          ungetc(kk, fp);
          break;
        }
        else fgets(lineIn2, 500, fp);
      }
      vecInpVals.setLength(dnSamp*dnInps);
      vecOutVals.setLength(dnSamp);
      inputVals = vecInpVals.getDVector();
      for (jj = 0; jj < dnSamp; jj++)
      {
        fscanf(fp, "%d", &ind);
        if (ind != (jj+1))
        {
          printf("ERROR: Input index mismatch (%d,%d)\n",jj+1,ind);
          printf("       Read     index = %d\n", ind);
          printf("       Expected index = %d\n", jj+1);
          if (faPtrsRsEval[0] != NULL) delete faPtrsRsEval[0];
          if (faPtrsRsEval[1] != NULL) delete faPtrsRsEval[1];
          delete [] faPtrsRsEval;
          return 1;
        }
        for (ii = 0; ii < nInputs; ii++)
          fscanf(fp, "%lg", &(inputVals[jj*dnInps+ii]));
      }
      fgets(lineIn2, 500, fp);
      fscanf(fp, "%s", winput);
      if (strcmp(winput, "PSUADE_END"))
        printf("INFO: PSUADE_END not found as the last line\n");
      fclose(fp);
    }
    //**/ now the sample has been read (inputVals, dnSamp)
    //**/ the response surface is ready (faPtrsRsEval[0])
    //**/ the discrepancy response surface is ready (faPtrsRsEval[1])
    faPtrsRsEval[0]->evaluatePoint(dnSamp, inputVals, 
                                   vecOutVals.getDVector());
    if (discFile == 1)
    {
      for (jj = 0; jj < dnSamp; jj++)
      {
        ddata = faPtrsRsEval[1]->evaluatePoint(&inputVals[jj*nInputs]);
        vecOutVals[jj] += ddata;
      }
    }
    double mean=0.0, stdev=0.0;
    for (ss = 0; ss < dnSamp; ss++) mean += vecOutVals[ss];
    mean /= (double) dnSamp;
    for (ss = 0; ss < dnSamp; ss++) stdev += pow(vecOutVals[ss]-mean,2.0);
    stdev = sqrt(stdev / dnSamp);
    if (plotScilab())
    {
      fp = fopen("scilabrsuap.sci", "w");
      if (fp == NULL)
        printf("ERROR: Cannot open scilabrsuap.sci file.\n");
    }
    else
    {
      fp = fopen("matlabrsuap.m", "w");
      if (fp == NULL)
        printf("ERROR: Cannot open matlabrsuap.m file.\n");
    }
    if (fp != NULL)
    {
      fprintf(fp, "Y = [ \n");
      for (jj = 0; jj < dnSamp; jj++)
        fprintf(fp, "%16.8e\n", vecOutVals[jj]);
      fprintf(fp, "];\n");
      if (plotScilab())
      {
        fprintf(fp, "histplot(10, Y, style=2);\n");
        fprintf(fp, "a = gce();\n");
        fprintf(fp, "a.children.fill_mode = \"on\";\n");
        fprintf(fp, "a.children.thickness = 2;\n");
        fprintf(fp, "a.children.foreground = 0;\n");
        fprintf(fp, "a.children.background = 2;\n");
      }
      else
      {
        fprintf(fp, "[nk,xk]=hist(Y,10);\n");
        fprintf(fp, "bar(xk,nk/%d,1.0)\n",dnSamp);
      }
      fwritePlotAxes(fp);
      fwritePlotTitle(fp, "Probability Distribution");
      fwritePlotXLabel(fp, "Output Value");
      fwritePlotYLabel(fp, "Probabilities");
      fclose(fp);
      if (plotScilab())
           printf("distribution plot file = scilabrsuap.sci\n");
      else printf("distribution plot file = matlabrsuap.m.\n");
      printAsterisks(PL_INFO, 0);
      printf("**             Summary Statistics\n");
      printEquals(PL_INFO, 0);
      printf("** Sample mean  = %e\n", mean);
      printf("** Sample stdev = %e\n", stdev);
      printAsterisks(PL_INFO, 0);
      delete faPtrsRsEval[0];
      if (discFile == 1) delete faPtrsRsEval[1];
      delete [] faPtrsRsEval;
      faPtrsRsEval = NULL;
    }
    return 0;
  }

  //**/ -------------------------------------------------------------
  // several qsa methods 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rs_qsa"))
  {
    printf("This command has been replaced by vce1_sobol and\n");
    printf("tsi_sobol.\n");
    return 0;
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rs_qsa: RS-based quantitative sensitivity analysis\n");
      printf("Syntax: rs_qsa (no argument needed)\n");
      printf("NOTE: to facilitate processing, all expert modes have\n");
      printf("      been suppressed.\n");
      printf("NOTE: This command differs from vce1_sobol, tsi_sobol,\n");
      printf("      and the command 'vce1_bin' in that it uses bootstrapped\n");
      printf("      samples multiple times to get the errors in Sobol'\n");
      printf("      indices due to response surface errors.\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }

    printAsterisks(PL_INFO, 0);
    printf("This command computes first-order sensitivity ");
    printf("indices using the\n");
    printf("response surface constructed from the loaded ");
    printf("sample (with bootstrapping).\n");
    printf("This is an alternative to rssobol1b (for cross-checking).\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ get output information
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    //**/ set up analysis manager
    printf("Which main or total effect analyzer ? \n");
    printf("1. Sobol' main effect using McKay's method (replicated LH).\n");
    printf("2. Sobol' main effect method using on numerical integration. \n");
    printf("3. Sobol' total sensitivity using numerical integration.\n");
    snprintf(pString,100,"Which  method (1, 2, or 3) ? ");
    int method = getInt(1, 3, pString);
    int analysisMethod, iOne=1;
    if      (method == 1) analysisMethod = PSUADE_ANA_ME;
    else if (method == 2) analysisMethod = PSUADE_ANA_RSSOBOL1;
    else if (method == 3) analysisMethod = PSUADE_ANA_RSSOBOLTSI;
    AnalysisManager *anaManager = new AnalysisManager();
    anaManager->setup(analysisMethod, 0);

    //**/ set up MARS response surface
    FuncApprox *faPtr;
    if (method == 1)
    {
      faType = -1;
      faPtr = genFA(faType, nInputs, iOne, nSamples);
      faPtr->setBounds(iLowerB, iUpperB);
      faPtr->setOutputLevel(0);
    }

    //**/ save parameters resident in psuadeIO (to be restored later)
    psuadeIO->getParameter("ana_diagnostics",pPtr);
    int saveDiag = pPtr.intData_;
    psuadeIO->updateAnalysisSection(-1,-1,-1,-1,-1,-1);
    psConfig_.AnaExpertModeSaveAndReset();
    psuadeIO->getParameter("method_sampling", pPtr);
    int saveMethod = pPtr.intData_;

    //**/ set up storage for the response surface samples
    int usePDFs, ind, count2=100000, nReps=200;
    Sampling *sampPtr;
    psVector vecUpper, vecLower, vecOut;
    psVector  vecMT, vecVT, vecXT, vecYT, vecTT;
    psIVector vecIT, vecST;
    vecMT.setLength(nSamples*nInputs);
    vecVT.setLength(nSamples*nInputs);
    vecXT.setLength(count2*nInputs);
    vecYT.setLength(count2);
    vecST.setLength(nSamples);
    vecIT.setLength(count2);
    for (ii = 0; ii < count2*nInputs; ii++) vecXT[ii] = 0.0;
    for (ii = 0; ii < count2; ii++) vecYT[ii] = 0.0;
    for (ii = 0; ii < count2; ii++) vecIT[ii] = 1;

    //**/ set up for iterations
    snprintf(pString,100,"How many times to run it (10 - 1000) : ");
    int count = getInt(10, 1000, pString);
    vecTT.setLength(count*nInputs);
    psuadeIO->getParameter("ana_use_input_pdfs", pPtr);
    usePDFs = pPtr.intData_;
    PDFManager *pdfman = NULL;
    if (usePDFs == 1 && method == 1)
    {
      printf("NOTE: Some inputs have non-uniform distributions.\n");
      pdfman = new PDFManager();
      pdfman->initialize(psuadeIO);
      vecOut.setLength(count2*nInputs);
      vecUpper.load(nInputs, iUpperB);
      vecLower.load(nInputs, iLowerB);
    }

    //**/ iterate
    for (kk = 0; kk < count; kk++)
    {
      printf("rq_qsa: ITERATION %d\n", kk+1);
      //**/ random draw
      for (ss = 0; ss < nSamples; ss++)
      {
        ind = PSUADE_rand() % nSamples;
        for (ii = 0; ii < nInputs; ii++)
          vecMT[ss*nInputs+ii] = sampleInputs[ind*nInputs+ii];
        vecVT[ss] = sampleOutputs[ind*nOutputs+outputID];
        vecST[ss] = sampleStates[ss];
      }
      //**/ only for McKay's main effect, not rssobol1 and others
      if (method == 1)
      {
        //**/ use sample to create a response surface
        status = faPtr->initialize(vecMT.getDVector(),vecVT.getDVector());

        //**/ generate a LH sample
        sampPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
        sampPtr->setPrintLevel(0);
        sampPtr->setInputBounds(nInputs, iLowerB, iUpperB);
        sampPtr->setOutputParams(1);
        sampPtr->setSamplingParams(count2, nReps, 1);
        sampPtr->initialize(0);
        sampPtr->getSamples(count2,nInputs,1,vecXT.getDVector(),
                            vecYT.getDVector(), vecIT.getIVector());
        //**/ if use_input_pdfs set in analysis section
        if (usePDFs == 1)
        {
          pdfman->invCDF(count2, vecXT, vecOut, vecLower, vecUpper);
          vecXT = vecOut;
        }

        //**/ evaluate the LH sample
        faPtr->evaluatePoint(count2,vecXT.getDVector(),vecXT.getDVector());

        //**/ write the LH sample into a PsuadeData object
        psuadeIO->updateInputSection(count2,nInputs,NULL,NULL,NULL,
                    vecXT.getDVector(),NULL,NULL,NULL,NULL,NULL);
        psuadeIO->updateOutputSection(count2,1,vecYT.getDVector(),
                                      vecIT.getIVector(),NULL);
        psuadeIO->updateMethodSection(PSUADE_SAMP_LHS,count2,
                                      nReps,-1,-1);
      }
      else
      {
        psuadeIO->updateInputSection(nSamples,nInputs,NULL,NULL,
                       NULL,vecMT.getDVector(),NULL,NULL,NULL,NULL,NULL);
        psuadeIO->updateOutputSection(nSamples,1,vecVT.getDVector(),
                       vecST.getIVector(), outputNames);
        psuadeIO->updateMethodSection(PSUADE_SAMP_MC,nSamples,
                                      -1,-1,-1);
      }
      //**/ analyze the result
      anaManager->analyze(psuadeIO, 0, NULL, 0);
      pData *pdata = psuadeIO->getAuxData();
      if (pdata->nDbles_ != nInputs)
      {
        printf("ERROR: nInputs do not match (%d, %d).\n",
               pdata->nDbles_, nInputs);
        printf("       Consult PSUADE developers.\n");
        if (method == 1) delete sampPtr;
        return 1;
      }

      //**/ get the statistics
      if (pdata->dbleData_ > 0)
        for (ii = 0; ii < nInputs; ii++)
          vecTT[kk*nInputs+ii] =
                pdata->dbleArray_[ii]/pdata->dbleData_;
      else
        for (ii = 0; ii < nInputs; ii++)
          vecTT[kk*nInputs+ii] = pdata->dbleArray_[ii];

      //**/ clean up
      pdata->clean();
      if (method == 1) delete sampPtr;
    }
    if (usePDFs == 1 && method == 1) delete pdfman;
    vecMT.setLength(nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      vecMT[ii] = vecTT[ii];
      for (jj = 1; jj < count; jj++) vecMT[ii] += vecTT[jj*nInputs+ii];
      vecMT[ii] /= (double) count;
    }
    vecVT.setLength(nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      vecVT[ii] = pow(vecTT[ii]-vecMT[ii], 2.0);
      for (jj = 1; jj < count; jj++)
        vecVT[ii] += pow(vecTT[jj*nInputs+ii]-vecMT[ii],2.0);
      vecVT[ii] /= (double) (count - 1);
      vecVT[ii] = sqrt(vecVT[ii]);
    }
    printEquals(PL_INFO, 0);
    printf("Statistics (based on %d replications): \n", count);
    for (ii = 0; ii < nInputs; ii++)
      printf("Input %4d: mean = %16.8e, std = %16.8e\n",ii+1,
             vecMT[ii],vecVT[ii]);
    delete anaManager;
    if (faPtr != NULL) delete faPtr;

    //**/ restore previous settings
    psuadeIO->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                                 sampleInputs,NULL,NULL,NULL,NULL,NULL);
    psuadeIO->updateOutputSection(nSamples,nOutputs,sampleOutputs,
                                   sampleStates,outputNames);
    psuadeIO->updateMethodSection(saveMethod,nSamples,-1,-1,-1);
    psuadeIO->updateAnalysisSection(-1,-1,-1,saveDiag,-1,-1);
    psConfig_.AnaExpertModeRestore();
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsplot1
  //**/ generate response surface of any one inputs and write the
  //**/ grid data to file for display with matlab
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsplot1") ||
           !strcmp(command, "rs1"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsplot1: response surface plot in one parameter\n");
      printf("syntax: rsplot1 (no argument needed)\n");
      return 1;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    printAsterisks(PL_INFO, 0);
    if (plotMatlab())
      printf("This command creates a Matlab 2D plot (1 input/1 output).\n");
    else 
      printf("This command creates a Scilab 2D plot (1 input/1 output).\n");
    printf("The selected input  will be in the X axis.\n");
    printf("The selected output will be in the Y axis.\n");
    printf("The other inputs are set at their midpoints or user-specified.\n");
    printf("You will be asked to select a response surface type.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ set up the function approximator
    int nPtsPerDim = 256;
    int faFlag = 1;
    FuncApprox *faPtr = genFAInteractive(psuadeIO, faFlag);
    if (faPtr == NULL) {printf("ERROR detected.\n"); return 1;}
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);
    psVector vecInpSettings;
    vecInpSettings.setLength(nInputs);

    int iplot1, iInd1, jplot, sInd;
    snprintf(pString,100,"Enter the input for x axis (1 - %d) : ", nInputs);
    iplot1 = getInt(1, nInputs, pString);
    iplot1--;

    if (nInputs > 1)
    {
      snprintf(pString,100,"Set other inputs at their mid points? (y or n) ");
      getString(pString, winput);
      if (winput[0] == 'y')
      {
        for (iInd1 = 0; iInd1 < nInputs; iInd1++)
        {
          if (iInd1 != iplot1)
               vecInpSettings[iInd1] = 0.5*(iLowerB[iInd1]+iUpperB[iInd1]);
          else vecInpSettings[iInd1] = 1.0;
        }
      }
      else
      {
        for (iInd1 = 0; iInd1 < nInputs; iInd1++)
        {
          if (iInd1 != iplot1)
          {
            snprintf(pString,100,
                    "Enter nominal value for input %d (%e - %e): ", 
                    iInd1+1, iLowerB[iInd1], iUpperB[iInd1]);
            vecInpSettings[iInd1] = getDouble(pString);
          }
          else vecInpSettings[iInd1] = 1.0;
        }
      }
    }
    jplot = 0;
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    jplot = getInt(1, nOutputs, pString);
    jplot--;

    //**/ generate 1D data
    int    faLeng=0;
    double *faXOut=NULL, *faYOut=NULL;
    psVector vecFaYIn;
    vecFaYIn.setLength(nSamples);
    for (sInd = 0; sInd < nSamples; sInd++)
      vecFaYIn[sInd] = sampleOutputs[sInd*nOutputs+jplot];

    faPtr->gen1DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,
                  vecInpSettings.getDVector(), &faLeng, &faXOut,&faYOut);

    //**/ write to scilab file
    if (plotScilab())
    {
      fp = fopen("scilabrsplot1.sci", "w");
      if (fp == NULL)
      {
        printf("ERROR: Cannot open file scilabrsplot1.sci.\n");
        delete [] faXOut;
        delete [] faYOut;
        delete faPtr;
        return 1;
      }
      fwritePlotCLF(fp);
      if (nInputs == 1)
      {
        fprintf(fp, "XX = [\n");
        for (sInd = 0; sInd < nSamples; sInd++)
          fprintf(fp, "%e\n", sampleInputs[sInd]);
        fprintf(fp, "];\n");
        fprintf(fp, "YY = [\n");
        for (sInd = 0; sInd < nSamples; sInd++)
          fprintf(fp, "%e\n", vecFaYIn[sInd]);
        fprintf(fp, "];\n");
      }
      fprintf(fp, "A = [\n");
      for (sInd = 0; sInd < faLeng; sInd++)
        fprintf(fp, "%e\n", faYOut[sInd]);
      fprintf(fp, "];\n");
      fprintf(fp, "X = [\n");
      for (sInd = 0; sInd < faLeng; sInd++)
        fprintf(fp, "%e\n", faXOut[sInd]);
      fprintf(fp, "];\n");
      fprintf(fp, "plot(X,A,'-')\n");
      fprintf(fp, "a = gca();\n");
      fprintf(fp, "a.children.children.thickness = 4;\n");
      if (nInputs == 1)
      {
        fwriteHold(fp,1);
        fprintf(fp,"plot(XX,YY,'*');\n");
      }
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, inputNames[iplot1]);
      fwritePlotYLabel(fp, outputNames[jplot]);
      snprintf(winput,100,"Plot for %s", outputNames[jplot]);
      fwritePlotTitle(fp, winput);
      fclose(fp);
      printf("rsplot1 plot file = scilabrsplot1.sci\n");
    }
    else
    {
      //**/ write to matlab file
      fp = fopen("matlabrsplot1.m", "w");
      if (fp == NULL)
      {
        printf("ERROR: Cannot open file matlabrsplot1.m.\n");
        delete [] faXOut;
        delete [] faYOut;
        delete faPtr;
        return 1;
      }
      fwritePlotCLF(fp);
      if (nInputs == 1)
      {
        fprintf(fp, "XX = [\n");
        for (sInd = 0; sInd < nSamples; sInd++)
          fprintf(fp, "%e\n", sampleInputs[sInd]);
        fprintf(fp, "];\n");
        fprintf(fp, "YY = [\n");
        for (sInd = 0; sInd < nSamples; sInd++)
          fprintf(fp, "%e\n", vecFaYIn[sInd]);
        fprintf(fp, "];\n");
      }
      fprintf(fp, "A = [\n");
      for (sInd = 0; sInd < faLeng; sInd++)
        fprintf(fp, "%e\n", faYOut[sInd]);
      fprintf(fp, "];\n");
      fprintf(fp, "X = [\n");
      for (sInd = 0; sInd < faLeng; sInd++)
        fprintf(fp, "%e\n", faXOut[sInd]);
      fprintf(fp, "];\n");
      fprintf(fp, "plot(X,A,'-','lineWidth',4)\n");
      fprintf(fp, "hold on\n");
      double Ymin = faYOut[0];
      for (sInd = 1; sInd < faLeng; sInd++)
        if (faYOut[sInd] < Ymin) Ymin = faYOut[sInd];
      double Ymax = faYOut[0];
      for (sInd = 1; sInd < faLeng; sInd++)
        if (faYOut[sInd] > Ymax) Ymax = faYOut[sInd];
      printf("Ymin and Ymax found = %e %e.\n", Ymin, Ymax);
      printf("You can set thresholds to cut out certain regions.\n");
      snprintf(pString,100,"Set lower threshold for output? (y or n) : ");
      getString(pString, winput);
      fprintf(fp, "yminFlag = 0;\n");
      double thresh;
      if (winput[0] == 'y')
      {
        snprintf(pString,100,"Enter the lower threshold (min = %e) : ", 
                Ymin);
        thresh = getDouble(pString);
        fprintf(fp, "ymin = %e;\n", thresh);
        fprintf(fp, "plot(X,ones(%d,1)*ymin,'r-')\n",faLeng);
      }
      snprintf(pString,100,"Set upper threshold for output? (y or n) : ");
      getString(pString, winput);
      if (winput[0] == 'y')
      {
        snprintf(pString,100,"Enter the upper threshold (max = %e) : ", 
                 Ymax);
        thresh = getDouble(pString);
        fprintf(fp, "ymax = %e;\n", thresh);
        fprintf(fp, "plot(X,ones(%d,1)*ymax,'r-')\n",faLeng);
      }
      if (nInputs == 1)
      {
        fwriteHold(fp,1);
        fprintf(fp,"plot(XX,YY,'k*','markersize',13);\n");
      }
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, inputNames[iplot1]);
      fwritePlotYLabel(fp, outputNames[jplot]);
      snprintf(winput,100,"Plot for %s", outputNames[jplot]);
      fwritePlotTitle(fp, winput);
      fclose(fp);
      printf("rsplot1 plot file = matlabrsplot1.m\n");
    }
    delete [] faXOut;
    delete [] faYOut;
    delete faPtr;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsplot1s
  //**/ generate response surface of any one inputs and write the
  //**/ grid data to file for display with matlab (include 
  //**/ uncertainties)
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsplot1s"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsplot1s: 1-parameter RS surface (with uncertainty) plot\n");
      printf("Syntax: rsplot1s (no argument needed)\n");
      return 1;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    printAsterisks(PL_INFO, 0);
    if (plotMatlab())
         printf("Create a Matlab plot (with output uncertainty).\n");
    else printf("Create a Scilab plot (with output uncertainty).\n");
    printEquals(PL_INFO, 0);

    //**/ set up the function approximator
    int nPtsPerDim = 128;
    int faFlag = 1;
    FuncApprox *faPtr = genFAInteractive(psuadeIO, faFlag);
    if (faPtr == NULL) {printf("ERROR detected.\n"); return 1;}
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);
    psVector vecInpSettings;
    vecInpSettings.setLength(nInputs);
    int    iplot1, iInd1, jplot, sInd;

    snprintf(pString,100,"Enter the input for x axis (1 - %d) : ", nInputs);
    iplot1 = getInt(1, nInputs, pString);
    iplot1--;
    if (nInputs > 1)
    {
      snprintf(pString,100,"Set other inputs at their mid points? (y or n) ");
      getString(pString, winput);
      if (winput[0] == 'y')
      {
        for (iInd1 = 0; iInd1 < nInputs; iInd1++)
        {
          if (iInd1 != iplot1)
               vecInpSettings[iInd1] = 0.5*(iLowerB[iInd1]+iUpperB[iInd1]);
          else vecInpSettings[iInd1] = 1.0;
        }
      }
      else
      {
        for (iInd1 = 0; iInd1 < nInputs; iInd1++)
        {
          if (iInd1 != iplot1)
          {
            snprintf(pString,100,
                  "Enter nominal value for input %d (%e - %e): ", 
                  iInd1+1, iLowerB[iInd1], iUpperB[iInd1]);
            vecInpSettings[iInd1] = getDouble(pString);
          }
          else vecInpSettings[iInd1] = 1.0;
        }
      }
    }
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    jplot = getInt(1, nOutputs, pString);
    jplot--;

    //**/ generate 1D data
    psVector vecFaYIn;
    vecFaYIn.setLength(nSamples);
    for (sInd = 0; sInd < nSamples; sInd++)
      vecFaYIn[sInd] = sampleOutputs[sInd*nOutputs+jplot];
    faPtr->initialize(sampleInputs,vecFaYIn.getDVector());

    double hx = (iUpperB[iplot1]-iLowerB[iplot1])/(double) (nPtsPerDim-1.0);
    psVector vecFaXOut, vecFaYOut, vecFaYStd;
    vecFaXOut.setLength(nPtsPerDim*nInputs);
    for (ii = 0; ii < nPtsPerDim; ii++)
    {
      for (jj = 0; jj < nInputs; jj++)
        vecFaXOut[ii*nInputs+jj] = vecInpSettings[jj];
      vecFaXOut[ii*nInputs+iplot1] = hx * ii + iLowerB[iplot1];
    }
    vecFaYOut.setLength(nPtsPerDim);
    vecFaYStd.setLength(nPtsPerDim);

    faPtr->evaluatePointFuzzy(nPtsPerDim, vecFaXOut.getDVector(), 
                      vecFaYOut.getDVector(),vecFaYStd.getDVector());

    //**/ write to scilab file
    if (plotScilab())
    {
      fp = fopen("scilabrsplot1s.sci", "w");
      if (fp == NULL)
      {
        printf("ERROR: Cannot open file scilabrsplot1s.sci.\n");
        delete faPtr;
        return 1;
      }
      fwritePlotCLF(fp);
      fprintf(fp, "A = [\n");
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
        fprintf(fp, "%16.8e %16.8e\n", vecFaYOut[sInd], vecFaYStd[sInd]);
      fprintf(fp, "];\n");
      fprintf(fp, "X = [\n");
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
        fprintf(fp, "%e\n", vecFaXOut[sInd*nInputs+iplot1]);
      fprintf(fp, "];\n");
      fprintf(fp, "plot(X,A(:,1),'k-')\n");
      fprintf(fp, "a = gca();\n");
      fprintf(fp, "a.children.children.thickness = 4;\n");
      fwriteHold(fp, 1);
      fprintf(fp, "for ii = 1 : %d\n", nPtsPerDim);
      fprintf(fp, "  xx = [X(ii) X(ii)];\n");
      fprintf(fp, "  yy = [A(ii,1)-2*A(ii,2) A(ii,1)+2*A(ii,2)];\n");
      fprintf(fp, "  plot(xx,yy,'b-','lineWidth',1)\n");
      fprintf(fp, "end\n");
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, inputNames[iplot1]);
      fwritePlotYLabel(fp, outputNames[jplot]);
      snprintf(winput,100,"Plot for %s", outputNames[jplot]);
      fwritePlotTitle(fp, winput);
      fclose(fp);
      printf("rsplot1s plot file = scilabrsplot1s.sci\n");
    }
    else
    {
      //**/ write to matlab file
      fp = fopen("matlabrsplot1s.m", "w");
      if (fp == NULL)
      {
        printf("ERROR: Cannot open file matlabrsplot1s.m.\n");
        delete faPtr;
        return 1;
      }
      fwritePlotCLF(fp);
      fprintf(fp, "%% 1D plot with +/- 2 std dev\n");
      fprintf(fp, "A = [\n");
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
        fprintf(fp, "%16.8e %16.8e\n", vecFaYOut[sInd], vecFaYStd[sInd]);
      fprintf(fp, "];\n");
      fprintf(fp, "X = [\n");
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
        fprintf(fp, "%e\n", vecFaXOut[sInd*nInputs+iplot1]);
      fprintf(fp, "];\n");
      fprintf(fp, "plot(X,A(:,1),'-','lineWidth',4)\n");
      fprintf(fp, "hold on\n");
      fprintf(fp, "for ii = 1 : %d\n", nPtsPerDim);
      fprintf(fp, "  xx = [X(ii) X(ii)];\n");
      fprintf(fp, "  yy = [A(ii,1)-2*A(ii,2) A(ii,1)+2*A(ii,2)];\n");
      fprintf(fp, "  plot(xx,yy,'b-','lineWidth',1)\n");
      fprintf(fp, "end\n");
      
      double Ymin = vecFaYOut[0];
      for (sInd = 1; sInd < nPtsPerDim; sInd++)
        if (vecFaYOut[sInd] < Ymin) Ymin = vecFaYOut[sInd];
      double Ymax = vecFaYOut[0];
      for (sInd = 1; sInd < nPtsPerDim; sInd++)
        if (vecFaYOut[sInd] > Ymax) Ymax = vecFaYOut[sInd];
      printf("Ymin and Ymax found = %e %e.\n", Ymin, Ymax);
      printf("You can set thresholds to cut out certain regions.\n");
      snprintf(pString,100,"Set lower threshold for output? (y or n) : ");
      getString(pString, winput);
      fprintf(fp, "yminFlag = 0;\n");
      double thresh;
      if (winput[0] == 'y')
      {
        snprintf(pString,100,"Enter the lower threshold (min = %e) : ", 
                 Ymin);
        thresh = getDouble(pString);
        fprintf(fp, "ymin = %e;\n", thresh);
        fprintf(fp, "plot(X,ones(%d,1)*ymin,'r-')\n",nPtsPerDim);
      }
      snprintf(pString,100,"Set upper threshold for output? (y or n) : ");
      getString(pString, winput);
      if (winput[0] == 'y')
      {
        snprintf(pString,100,"Enter the upper threshold (max = %e) : ", 
                 Ymax);
        thresh = getDouble(pString);
        fprintf(fp, "ymax = %e;\n", thresh);
        fprintf(fp, "plot(X,ones(%d,1)*ymax,'r-')\n",nPtsPerDim);
      }
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, inputNames[iplot1]);
      fwritePlotYLabel(fp, outputNames[jplot]);
      snprintf(winput,100,"Plot for %s", outputNames[jplot]);
      fwritePlotTitle(fp, winput);
      fclose(fp);
      printf("rsplot1s plot file = matlabrsplot1s.m\n");
    }
    delete faPtr;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rs2 or rsplot2
  //**/ generate response surface of any two inputs and write the
  //**/ grid data to file for display with matlab
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rs2") ||
           !strcmp(command, "rsplot2"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsplot2: response surface plot in two parameters\n");
      printf("syntax: rsplot2 (no argument needed)\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nInputs < 2)
    {
      printf("ERROR: rsplot2 requires 2 or more inputs.\n");
      return 1;
    }
    printAsterisks(PL_INFO, 0);
    if (plotMatlab())
      printf("This command creates a Matlab 3D plot (2 inputs/1 output).\n");
    else 
      printf("This command creates a Scilab 3D plot (2 inputs/1 output).\n");
    printf("The selected inputs will be in the X and Y axes.\n");
    printf("The selected output will be in the Z axis.\n");
    printf("The other inputs are set at their midpoints or user-specified.\n");
    printf("You will be asked to select a response surface type.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ set up the function approximator
    int nPtsPerDim = 64;
    snprintf(pString,100,"Grid resolution ? (32 - 256) ");
    nPtsPerDim = getInt(32, 256, pString);
    int faFlag = 1;
    FuncApprox *faPtr = genFAInteractive(psuadeIO, faFlag);
    if (faPtr == NULL) {printf("ERROR detected.\n"); return 1;}
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);
    psVector vecInpSettings;
    vecInpSettings.setLength(nInputs);

    int iplot1, iplot2, iInd1, sInd, jplot;
    snprintf(pString,100,"Enter the input for x axis (1 - %d) : ", nInputs);
    iplot1 = getInt(1, nInputs, pString);
    iplot1--;
    snprintf(pString,100,"Enter the input for y axis (1 - %d) : ", nInputs);
    iplot2 = getInt(1, nInputs, pString);
    iplot2--;

    int p2cnt=1;
    if (iplot1 != iplot2) p2cnt++;
    if (nInputs-p2cnt > 0)
    {
      snprintf(pString,100,
              "Set other inputs at their mid points? (y or n) ");
      getString(pString, winput);
      if (winput[0] == 'y')
      {
        for (iInd1 = 0; iInd1 < nInputs; iInd1++)
        {
          if (iInd1 != iplot1 && iInd1 != iplot2)
               vecInpSettings[iInd1] = 0.5*(iLowerB[iInd1]+iUpperB[iInd1]);
          else vecInpSettings[iInd1] = 1.0;
        }
      }
      else
      {
        for (iInd1 = 0; iInd1 < nInputs; iInd1++)
        {
          if (iInd1 != iplot1 && iInd1 != iplot2)
          {
            snprintf(pString,100,
                  "Enter nominal value for input %d (%e - %e): ", 
                  iInd1+1, iLowerB[iInd1], iUpperB[iInd1]);
            vecInpSettings[iInd1] = getDouble(pString);
          }
          else vecInpSettings[iInd1] = 1.0;
        }
      }
    }
    jplot = 0;
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    jplot = getInt(1, nOutputs, pString);
    jplot--;

    //**/ generate 2D data
    int    faLeng=0;
    double *faXOut=NULL, *faYOut=NULL;
    psVector vecFaYIn;
    vecFaYIn.setLength(nSamples);
    for (sInd = 0; sInd < nSamples; sInd++)
       vecFaYIn[sInd] = sampleOutputs[sInd*nOutputs+jplot];

    if (outputLevel_ > 1)
       printf("Please wait while generating RS ....\n");
    faPtr->gen2DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,iplot2,
              vecInpSettings.getDVector(), &faLeng, &faXOut,&faYOut);

    //**/ write to matlab/scilab file
    if (plotScilab())
    {
      fp = fopen("scilabrsplot2.sci", "w");
      if (fp == NULL)
      {
        printf("ERROR: Cannot open file scilabrsplot2.sci.\n");
        delete [] faXOut;
        delete [] faYOut;
        delete faPtr;
        return 1;
      }
      fprintf(fp,"twoPlots = 1;\n");
      fprintf(fp,"A = [\n");
      for (sInd = 0; sInd < faLeng; sInd++)
        fprintf(fp, "%e\n", faYOut[sInd]);
      fprintf(fp,"];\n");
      fprintf(fp,"A = matrix(A,%d,%d);\n", nPtsPerDim, nPtsPerDim);
      fprintf(fp,"x = [\n");
      for (sInd = 0; sInd < faLeng; sInd+=nPtsPerDim)
        fprintf(fp, "%e\n", faXOut[sInd*2]);
      fprintf(fp,"];\n");
      fprintf(fp,"y = [\n");
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
        fprintf(fp, "%e\n", faXOut[sInd*2+1]);
      fprintf(fp,"];\n");
      fwritePlotCLF(fp);
      fprintf(fp,"if twoPlots == 1\n");
      fprintf(fp,"drawlater\n");
      fprintf(fp,"subplot(1,2,1)\n");
      fprintf(fp,"mesh(x,y,A)\n");
      fprintf(fp,"h = get(\"hdl\");\n");
      fprintf(fp,"h.color_flag=1;\n");
      fprintf(fp,"h.color_mode=-2;\n");
      fprintf(fp,"bmin = min(min(A)); bmax = max(max(A));\n");
      fprintf(fp,"xset(\"colormap\",jetcolormap(64));\n");
      fprintf(fp,"colorbar(bmin,bmax);\n");
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, inputNames[iplot1]);
      fwritePlotYLabel(fp, inputNames[iplot2]);
      fwritePlotZLabel(fp, outputNames[jplot]);
      snprintf(winput,100,"Mesh Plot for %s", outputNames[jplot]);
      fwritePlotTitle(fp, winput);
      fprintf(fp,"a=gca();\n");
      fprintf(fp,"a.data_bounds=[%e,%e;%e,%e];\n",iLowerB[iplot1],
              iLowerB[iplot2], iUpperB[iplot1], iUpperB[iplot2]);
      fprintf(fp,"a.axes_visible=\"on\";\n");
      fprintf(fp,"drawnow\n");
      fprintf(fp,"subplot(1,2,2)\n");
      fprintf(fp,"end;\n");
      fprintf(fp,"drawlater\n");
      fprintf(fp,"B = A;\n");
      fprintf(fp,"nX = length(x);\n");
      fprintf(fp,"nY = length(y);\n");
      fprintf(fp,"for ii = 1 : nX\n");
      fprintf(fp,"for jj = 1 : nY\n");
      fprintf(fp,"B(ii,jj) = A(nX-ii+1,jj);\n");
      fprintf(fp,"end;\n");
      fprintf(fp,"end;\n");
      fprintf(fp,"a=gca();\n");
      fprintf(fp,"a.data_bounds=[%e,%e;%e,%e];\n",iLowerB[iplot1],
              iLowerB[iplot2], iUpperB[iplot1], iUpperB[iplot2]);
      fprintf(fp,"bmin = min(min(B)); bmax = max(max(B));\n");
      fprintf(fp,"Matplot1((B-bmin)/(bmax-bmin)*64,[%e,%e,%e,%e])\n",
              iLowerB[iplot1],iLowerB[iplot2],iUpperB[iplot1], 
              iUpperB[iplot2]);
      fprintf(fp,"set(gca(),\"auto_clear\",\"off\")\n");
      fprintf(fp,"//contour2d(x,y,flipdim(B',1),6);\n");
      fprintf(fp,"xset(\"colormap\",jetcolormap(64));\n");
      fprintf(fp,"colorbar(bmin,bmax);\n");
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, inputNames[iplot1]);
      fwritePlotYLabel(fp, inputNames[iplot2]);
      snprintf(winput,100,"Contour Plot for %s", outputNames[jplot]);
      fwritePlotTitle(fp, winput);
      fprintf(fp,"drawnow\n");
      fclose(fp);
      printf("rsplot2 plot file = scilabrsplot2.sci\n");
    }
    else
    {
      fp = fopen("matlabrsplot2.m", "w");
      if (fp == NULL)
      {
        printf("ERROR: Cannot open file matlabrsplot2.m.\n");
        delete [] faXOut;
        delete [] faYOut;
        delete faPtr;
        return 1;
      }
      fwritePlotCLF(fp);
      fprintf(fp, "twoPlots = 1;\n");
      fprintf(fp, "A = [\n");
      for (sInd = 0; sInd < faLeng; sInd++)
        fprintf(fp, "%e\n", faYOut[sInd]);
      fprintf(fp, "];\n");
      fprintf(fp, "A = reshape(A,%d,%d);\n", nPtsPerDim, nPtsPerDim);
      fprintf(fp, "x = [\n");
      for (sInd = 0; sInd < faLeng; sInd+=nPtsPerDim)
        fprintf(fp, "%e\n", faXOut[sInd*2]);
      fprintf(fp, "];\n");
      fprintf(fp, "y = [\n");
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
        fprintf(fp, "%e\n", faXOut[sInd*2+1]);
      fprintf(fp, "];\n");
      if (nInputs == 2)
      {
        fprintf(fp, "xx = [\n");
        for (sInd = 0; sInd < nSamples; sInd++)
          fprintf(fp, "%e\n", sampleInputs[sInd*2+iplot1]);
        fprintf(fp, "];\n");
        fprintf(fp, "yy = [\n");
        for (sInd = 0; sInd < nSamples; sInd++)
          fprintf(fp, "%e\n", sampleInputs[sInd*2+iplot2]);
        fprintf(fp, "];\n");
        fprintf(fp, "zz = [\n");
        for (sInd = 0; sInd < nSamples; sInd++)
          fprintf(fp, "%e\n", vecFaYIn[sInd]);
        fprintf(fp, "];\n");
      }
      fprintf(fp, "B = A;\n");
      double Ymin = faYOut[0];
      for (sInd = 1; sInd < faLeng; sInd++)
        if (faYOut[sInd] < Ymin) Ymin = faYOut[sInd];
      double Ymax = faYOut[0];
      for (sInd = 1; sInd < faLeng; sInd++)
        if (faYOut[sInd] > Ymax) Ymax = faYOut[sInd];
      printf("Ymin and Ymax found = %e %e.\n", Ymin, Ymax);
      printf("You can set thresholds to cut out certain regions.\n");
      snprintf(pString,100,"Set lower threshold for output? (y or n) : ");
      getString(pString, winput);
      fprintf(fp, "n1 = 0;\n");
      fprintf(fp, "n2 = 0;\n");
      double thresh;
      if (winput[0] == 'y')
      {
        snprintf(pString,100,"Enter the lower threshold (min = %e) : ",Ymin);
        thresh = getDouble(pString);
        fprintf(fp, "ymin = %e;\n", thresh);
        fprintf(fp, "[ia,ja,aa] = find(A<ymin);\n");
        fprintf(fp, "for ii = 1 : length(ia)\n");
        //fprintf(fp, "   B(ia(ii),ja(ii)) = %e;\n",Ymin-PABS(Ymin)*0.9);
        fprintf(fp, "   B(ia(ii),ja(ii)) = NaN;\n");
        fprintf(fp, "end;\n");
        fprintf(fp, "n1 = length(ia);\n");
      }
      snprintf(pString,100,"Set upper threshold for output? (y or n) : ");
      getString(pString, winput);
      if (winput[0] == 'y')
      {
        snprintf(pString,100,"Enter the upper threshold (max = %e) : ",Ymax);
        thresh = getDouble(pString);
        fprintf(fp, "ymax = %e;\n", thresh);
        fprintf(fp, "[ia,ja,aa] = find(A>ymax);\n");
        fprintf(fp, "for ii = 1 : length(ia)\n");
        //fprintf(fp, "   B(ia(ii),ja(ii)) = %e;\n",Ymin-PABS(Ymin)*0.9);
        fprintf(fp, "   B(ia(ii),ja(ii)) = NaN;\n");
        fprintf(fp, "end;\n");
        fprintf(fp, "n2 = length(ia);\n");
      }
      fprintf(fp, "nB = size(B,1);\n");
      fprintf(fp, "if (n1 + n2 == nB * nB)\n");
      fprintf(fp, "   B(1,1) = 0;\n");
      fprintf(fp, "   B(%d,%d) = 1;\n",nPtsPerDim,nPtsPerDim);
      fprintf(fp, "end\n");
      fprintf(fp, "if twoPlots == 1\n");
      fprintf(fp, "subplot(1,2,1), mesh(x,y,A)\n");
      if (nInputs == 2)
      {
        fwriteHold(fp,1);
        fprintf(fp,"plot3(xx,yy,zz,'k*','markersize',13);\n");
      } 
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, inputNames[iplot1]);
      fwritePlotYLabel(fp, inputNames[iplot2]);
      fwritePlotZLabel(fp, outputNames[jplot]);
      snprintf(winput,100,"Mesh Plot for %s", outputNames[jplot]);
      fwritePlotTitle(fp, winput);
      fprintf(fp,"colorbar\n");
      fprintf(fp,"subplot(1,2,2)\n");
      fprintf(fp,"end\n");
      fprintf(fp,"contourf(x,y,B)\n");
      if (nInputs == 2)
      {
        fwriteHold(fp,1);
        fprintf(fp,"plot(xx,yy,'k*','markersize',13);\n");
      } 
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, inputNames[iplot1]);
      fwritePlotYLabel(fp, inputNames[iplot2]);
      fprintf(fp,"colorbar\n");
      fprintf(fp,"colormap(jet)\n");
      snprintf(winput,100,"Contour Plot for %s", outputNames[jplot]);
      fwritePlotTitle(fp, winput);
      fclose(fp);
      printf("rsplot2 plot file = matlabrsplot2.m\n");
    }
    delete [] faXOut;
    delete [] faYOut;
    delete faPtr;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rs3 or rsplot3 
  //**/ generate response surface of any 3 inputs and write the
  //**/ grid data to file for display with matlab
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rs3") ||
           !strcmp(command, "rsplot3"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsplot3: response surface plot in three parameters\n");
      printf("syntax: rsplot3 (no argument needed)\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nInputs < 3)
    {
      printf("ERROR: rsplot3 requires 3 or more inputs.\n");
      return 1;
    }
    if (plotScilab())
    {
      printf("INFO: rsplot3 is currently not available in scilab.\n");
      return 1;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command creates a Matlab 3D plot (3 input/1 output).\n");
    printf("The selected inputs will be in X, Y, and Z axes.\n");
    printf("The output values will be displayed as different colors.\n");
    printf("The other inputs are set at their midpoints or user-specified.\n");
    printf("You will be asked to select a response surface type.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ set up the function approximator
    int nPtsPerDim = 16;
    snprintf(pString,100,"Grid resolution ? (16 - 32) ");
    nPtsPerDim = getInt(16, 32, pString);
    int faFlag = 1;
    FuncApprox *faPtr = genFAInteractive(psuadeIO, faFlag);
    if (faPtr == NULL) {printf("ERROR detected.\n"); return 1;}
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);

    //**/ ask users to specify the three inputs and one output
    int iplot1, iplot2, iplot3;
    psVector vecInpSettings;
    vecInpSettings.setLength(nInputs);
    snprintf(pString,100,"Enter the input for x axis (1 - %d) : ", nInputs);
    iplot1 = getInt(1, nInputs, pString);
    iplot1--;
    snprintf(pString,100,"Enter the input for y axis (1 - %d) : ", nInputs);
    iplot2 = getInt(1, nInputs, pString);
    iplot2--;
    snprintf(pString,100,"Enter the input for z axis (1 - %d) : ", nInputs);
    iplot3 = getInt(1, nInputs, pString);
    iplot3--;

    int pcnt=1, iInd1, sInd, jplot;
    if (iplot2 != iplot1) pcnt++;
    if (iplot3 != iplot1 && iplot3 != iplot2) pcnt++;
    if (nInputs-pcnt > 0)
    {
      snprintf(pString,100,
              "Set other inputs at their mid points? (y or n) ");
      getString(pString, winput);
      if (winput[0] == 'y')
      {
        for (iInd1 = 0; iInd1 < nInputs; iInd1++)
        {
          if (iInd1 != iplot1 && iInd1 != iplot2 && iInd1 != iplot3)
               vecInpSettings[iInd1] = 0.5*(iLowerB[iInd1]+iUpperB[iInd1]);
          else vecInpSettings[iInd1] = 1.0;
        }
      }
      else
      {
        for (iInd1 = 0; iInd1 < nInputs; iInd1++)
        {
          if (iInd1 != iplot1 && iInd1 != iplot2 && iInd1 != iplot3)
          {
            vecInpSettings[iInd1] = iLowerB[iInd1] - 1.0;
            snprintf(pString,100,
                  "Enter nominal value for input %d (%e - %e): ", 
                  iInd1+1, iLowerB[iInd1], iUpperB[iInd1]);
            while (vecInpSettings[iInd1] < iLowerB[iInd1] ||
                   vecInpSettings[iInd1] > iUpperB[iInd1])
               vecInpSettings[iInd1] = getDouble(pString);
          }
          else vecInpSettings[iInd1] = 1.0;
        }
      }
    }
    snprintf(pString,100,"Enter the output number (1 - %d) : ", nOutputs);
    jplot = getInt(1, nOutputs, pString);
    jplot--;

    int    faLeng=0;
    double *faXOut=NULL, *faYOut=NULL;
    psVector vecFaYIn;
    vecFaYIn.setLength(nSamples);
    for (sInd = 0; sInd < nSamples; sInd++)
      vecFaYIn[sInd] = sampleOutputs[sInd*nOutputs+jplot];

    //**/ begin generating 3D data
    printf("Please wait while generating the RS data \n");
    faPtr->gen3DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,iplot2, 
              iplot3,vecInpSettings.getDVector(), &faLeng, &faXOut,&faYOut);

    //**/ ask for lower and upper threshold only once
    double GYmin = faYOut[0];
    for (sInd = 1; sInd < faLeng; sInd++)
      if (faYOut[sInd] < GYmin) GYmin = faYOut[sInd];
    double GYmax = faYOut[0];
    for (sInd = 1; sInd < faLeng; sInd++)
      if (faYOut[sInd] > GYmax) GYmax = faYOut[sInd];
    printf("\nYmin and Ymax found = %e %e.\n", GYmin, GYmax);
    double threshL = GYmin - 0.2 * PABS(GYmax-GYmin);
    double gamma = threshL;
    printf("You can set thresholds to cut out certain regions.\n");
    snprintf(pString,100,"Set lower threshold for output? (y or n) ");
    getString(pString, winput);
    if (winput[0] == 'y')
    {
      snprintf(pString,100,"Enter the lower threshold (min = %e): ",GYmin);
      threshL = getDouble(pString);
      if (threshL < GYmin)
      {
        threshL = GYmin;
        printf("rsplot3 INFO: Lower threshold set to %e.\n", threshL);
      }
    }
    int    ind;
    double threshU = GYmax + 0.2 * PABS(GYmax-GYmin);
    snprintf(pString,100,"Set upper threshold for output? (y or n) ");
    getString(pString, winput);
    if (winput[0] == 'y')
    {
      snprintf(pString,100,"Enter the upper threshold (max = %e): ",GYmax);
      threshU = getDouble(pString);
      if (threshU > GYmax)
      {
        threshU = GYmax;
        printf("rsplot3 INFO: Upper threshold set to %e.\n", threshU);
      }
    }
    if (threshL >= threshU)
    {
      printf("ERROR: Lower threshold (%e) >= Upper threshold (%e)\n",
             threshL, threshU);
      delete [] faXOut;
      delete [] faYOut;
      delete faPtr;
      return 1;
    }
    fp = fopen("matlabrsplot3.m", "w");
    if (fp == NULL)
    {
      printf("ERROR: Cannot open file matlabrsplot3.m.\n");
      delete [] faXOut;
      delete [] faYOut;
      delete faPtr;
      return 1;
    }
    fwritePlotCLF(fp);
    fprintf(fp,"xlo = %e; \n", iLowerB[iplot2]);
    fprintf(fp,"xhi = %e; \n", iUpperB[iplot2]);
    fprintf(fp,"ylo = %e; \n", iLowerB[iplot1]);
    fprintf(fp,"yhi = %e; \n", iUpperB[iplot1]);
    fprintf(fp,"zlo = %e; \n", iLowerB[iplot3]);
    fprintf(fp,"zhi = %e; \n", iUpperB[iplot3]);
    fprintf(fp,"X=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
    fprintf(fp,"Y=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
    fprintf(fp,"Z=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
    fprintf(fp,"V=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
    for (jj = 0; jj < nPtsPerDim; jj++)
    {
      fprintf(fp,"Y(:,:,%d) = [\n", jj + 1);
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
      {
        for (ii = 0; ii < nPtsPerDim; ii++)
        {
          ind = sInd*nPtsPerDim*nPtsPerDim+ii*nPtsPerDim+jj;
          fprintf(fp,"%e ", faXOut[ind*3]);
        }
        fprintf(fp,"\n");
      }
      fprintf(fp, "];\n");
      fprintf(fp, "X(:,:,%d) = [\n", jj + 1);
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
      {
        for (ii = 0; ii < nPtsPerDim; ii++)
        {
          ind = sInd*nPtsPerDim*nPtsPerDim+ii*nPtsPerDim+jj;
          fprintf(fp, "%e ", faXOut[ind*3+1]);
        }
        fprintf(fp, "\n");
      }
      fprintf(fp, "];\n");
      fprintf(fp, "Z(:,:,%d) = [\n", jj + 1);
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
      {
        for (ii = 0; ii < nPtsPerDim; ii++)
        {
          ind = sInd*nPtsPerDim*nPtsPerDim+ii*nPtsPerDim+jj;
          fprintf(fp, "%e ", faXOut[ind*3+2]);
        }
        fprintf(fp, "\n");
      }
      fprintf(fp, "];\n");
    }
    int count=0;
    for (jj = 0; jj < nPtsPerDim; jj++)
    {
      fprintf(fp, "V(:,:,%d) = [\n", jj + 1);
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
      {
        for (ii = 0; ii < nPtsPerDim; ii++)
        {
          ind = sInd*nPtsPerDim*nPtsPerDim+ii*nPtsPerDim+jj;
          if (faYOut[ind] < threshL)
          {
            fprintf(fp, "%e ", gamma);
            count++;
          }
          else if (faYOut[ind] > threshU)
          {
            fprintf(fp, "%e ", gamma);
            count++;
          }
          else fprintf(fp, "%e ", faYOut[ind]);
        }
        fprintf(fp, "\n");
      }
      fprintf(fp, "];\n");
    }
    if (count == nPtsPerDim*nPtsPerDim*nPtsPerDim)
    {
      fprintf(fp, "V(1,1,1)=0;\n");
      fprintf(fp, "V(%d,%d,%d)=1;\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
    }
    fprintf(fp,"xt = [%e:%e:%e];\n", iLowerB[iplot2],
            (iUpperB[iplot2]-iLowerB[iplot2])*0.01, iUpperB[iplot2]);
    fprintf(fp,"yt = [%e:%e:%e];\n", iLowerB[iplot1],
            (iUpperB[iplot1]-iLowerB[iplot1])*0.01, iUpperB[iplot1]);
    fprintf(fp,"zt = [%e:%e:%e];\n", iLowerB[iplot3],
            (iUpperB[iplot3]-iLowerB[iplot3])*0.01, iUpperB[iplot3]);
    fprintf(fp,"isoval = %e;\n", gamma);
    fprintf(fp,"h = patch(isosurface(X,Y,Z,V,isoval),... \n");
    fprintf(fp,"          'FaceColor', 'blue', ... \n");
    fprintf(fp,"          'EdgeColor', 'none', ... \n");
    fprintf(fp,"          'AmbientStrength', 0.2, ... \n");
    fprintf(fp,"          'SpecularStrength', 0.7, ... \n");
    fprintf(fp,"          'DiffuseStrength', 0.4);\n");
    fprintf(fp,"isonormals(X,Y,Z,V,h);\n");
    fprintf(fp,"patch(isocaps(X,Y,Z,V,isoval), ...\n");
    fprintf(fp,"      'FaceColor', 'interp', ... \n");
    fprintf(fp,"      'EdgeColor', 'none'); \n");
    fprintf(fp,"axis([xlo xhi ylo yhi zlo zhi])\n");
    fprintf(fp,"daspect([%e,%e,%e])\n",iUpperB[iplot2]-iLowerB[iplot2],
            iUpperB[iplot1]-iLowerB[iplot1],
            iUpperB[iplot3]-iLowerB[iplot3]);
    fprintf(fp,"   xlabel('%s','FontSize',12,'FontWeight','bold')\n",
            inputNames[iplot2]);
    fprintf(fp,"   ylabel('%s','Fontsize',12,'FontWeight','bold')\n",
            inputNames[iplot1]);
    fprintf(fp,"   zlabel('%s','Fontsize',12,'FontWeight','bold')\n",
            inputNames[iplot3]);
    fprintf(fp,"   title('%s','Fontsize',12,'FontWeight','bold')\n", 
            outputNames[jplot]);
    fwritePlotAxes(fp);
    fprintf(fp,"colormap('default'); colorbar\n");
    fprintf(fp,"%%axis tight\n");
    fprintf(fp,"view(3) \n");
    fprintf(fp,"set(gcf,'Renderer','zbuffer')\n");
    fprintf(fp,"lighting phong\n");
    fprintf(fp,"cin = input('generate slices ? (y or n) ','s');\n");
    fprintf(fp,"if (cin == 'y')\n");
    fprintf(fp,"xin = input('axis to slide through ? (x,y,z) ','s');\n");
    fprintf(fp,"for i = 1 : 101\n");
    fprintf(fp,"   display(['displaying ' int2str(i) ' of 100'])\n");
    fprintf(fp,"   if (xin == 'y')\n");
    fprintf(fp,"      h = slice(X,Y,Z,V,xt(i),[],[]);\n");
    fprintf(fp,"   elseif (xin == 'x')\n");
    fprintf(fp,"      h = slice(X,Y,Z,V,[],yt(i),[]);\n");
    fprintf(fp,"   elseif (xin == 'z')\n");
    fprintf(fp,"      h = slice(X,Y,Z,V,[],[],zt(i));\n");
    fprintf(fp,"   end\n");
    fprintf(fp,"   axis([%11.4e %11.4e %11.4e %11.4e %11.4e %11.4e ",
            iLowerB[iplot2], iUpperB[iplot2], iLowerB[iplot1],
            iUpperB[iplot1], iLowerB[iplot3], iUpperB[iplot3]);
    fprintf(fp,"%11.4e %11.4e])\n",
            threshL-0.2*(threshU-threshL),threshU+0.2*(threshU-threshL));
    fwritePlotAxes(fp);
    fprintf(fp,"   xlabel('%s','FontSize',12,'FontWeight','bold')\n",
            inputNames[iplot2]);
    fprintf(fp,"   ylabel('%s','Fontsize',12,'FontWeight','bold')\n",
            inputNames[iplot1]);
    fprintf(fp,"   zlabel('%s','Fontsize',12,'FontWeight','bold')\n",
            inputNames[iplot3]);
    fprintf(fp,"   title('3D Contour Plot',");
    fprintf(fp,"'FontWeight','bold','FontSize',12)\n");
    fprintf(fp,"   view(3)\n");
    fprintf(fp,"   colorbar\n");
    fprintf(fp,"   pause(1)\n");
    fprintf(fp,"   if (i < 101)\n");
    fprintf(fp,"      clf\n");
    fprintf(fp,"   end\n");
    fprintf(fp,"end\n");
    fprintf(fp,"end\n");
    fclose(fp);
    printf("rsplot3 plot file = matlabrsplot3.m\n");
    delete [] faXOut;
    delete [] faYOut;
    delete faPtr;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rs3m or rsplot3m
  //**/ generate response surface of any 3 inputs and write the
  //**/ grid data to file for display with matlab (movie)
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rs3m") ||
           !strcmp(command, "rsplot3m"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsplot3m: 3-input response surface in movie mode\n");
      printf("syntax: rsplot3m (no argument needed)\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nInputs < 3)
    {
      printf("ERROR: rsplot3m requires 3 or more inputs.\n");
      return 1;
    }
    if (plotScilab())
    {
      printf("INFO: rsplot3m is currently not available in scilab.\n");
      return 1;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command creates a Matlab 3D movie (3 inputs/1 output).\n");
    printf("The selected inputs will be in X, Y, and Z axes.\n");
    printf("The output will be displayed in the time axis.\n");
    printf("The other inputs are set at their midpoints or user-specified.\n");
    printf("You will be asked to select a response surface type.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ set up the function approximator
    int nPtsPerDim = 16;
    snprintf(pString,100,"Grid resolution ? (16 - 32) ");
    nPtsPerDim = getInt(16, 32, pString);
    int faFlag = 1;
    FuncApprox *faPtr = genFAInteractive(psuadeIO, faFlag);
    if (faPtr == NULL) {printf("ERROR detected.\n"); return 1;}
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);

    //**/ ask users to specify the three inputs and one output
    int iplot1, iplot2, iplot3, jplot, iInd1, sInd;
    psVector vecInpSettings;
    vecInpSettings.setLength(nInputs);
    snprintf(pString,100,"Enter the input for x axis (1 - %d) : ", nInputs);
    iplot1 = getInt(1, nInputs, pString);
    iplot1--;
    snprintf(pString,100,"Enter the input for y axis (1 - %d) : ", nInputs);
    iplot2 = getInt(1, nInputs, pString);
    iplot2--;
    snprintf(pString,100,"Enter the input for z axis (1 - %d) : ", nInputs);
    iplot3 = getInt(1, nInputs, pString);
    iplot3--;
    int pmcnt=1;
    if (iplot2 != iplot1) pmcnt++;
    if (iplot3 != iplot1 && iplot3 != iplot2) pmcnt++;
    if (nInputs-pmcnt > 0)
    {
      snprintf(pString,100,"Set other inputs at their mid points ? (y or n) ");
      getString(pString, winput);
      if (winput[0] == 'y')
      {
        for (iInd1 = 0; iInd1 < nInputs; iInd1++)
        {
          if (iInd1 != iplot1 && iInd1 != iplot2 && iInd1 != iplot3)
               vecInpSettings[iInd1] = 0.5*(iLowerB[iInd1]+iUpperB[iInd1]);
          else vecInpSettings[iInd1] = 1.0;
        }
      }
      else
      {
        for (iInd1 = 0; iInd1 < nInputs; iInd1++)
        {
          if (iInd1 != iplot1 && iInd1 != iplot2 && iInd1 != iplot3)
          {
            snprintf(pString,100,"Enter setting for input %d (%e - %e): ", 
                    iInd1+1, iLowerB[iInd1], iUpperB[iInd1]);
            vecInpSettings[iInd1] = getDouble(pString);
          }
          else vecInpSettings[iInd1] = 1.0;
        }
      }
    }
    snprintf(pString,100,"Enter the output number (1 - %d) : ", nOutputs);
    jplot = getInt(1, nOutputs, pString);
    jplot--;

    psVector vecFaYIn;
    vecFaYIn.setLength(nSamples);
    for (sInd = 0; sInd < nSamples; sInd++)
      vecFaYIn[sInd] = sampleOutputs[sInd*nOutputs+jplot];

    //**/ begin generating 2D data
    fp = fopen("matlabrsplot3m.m", "w");
    if (fp == NULL)
    {
      printf("ERROR: Cannot open file matlabrsplot3m.m.\n");
      delete faPtr;
      return 1;
    }

    //**/ begin generating 3D data
    int faLeng=0;
    double *faXOut=NULL, *faYOut=NULL;
    faPtr->gen3DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,iplot2, 
              iplot3,vecInpSettings.getDVector(), &faLeng, &faXOut,&faYOut);
    double GYmin = faYOut[0];
    for (sInd = 1; sInd < faLeng; sInd++)
      if (faYOut[sInd] < GYmin) GYmin = faYOut[sInd];
    double GYmax = faYOut[0];
    for (sInd = 1; sInd < faLeng; sInd++)
      if (faYOut[sInd] > GYmax) GYmax = faYOut[sInd];
    printf("\nYmin and Ymax found = %e %e.\n", GYmin, GYmax);
    double threshL = GYmin - 0.2 * PABS(GYmin);
    printf("You can set thresholds to cut out certain regions.\n");
    snprintf(pString,100,"Set lower threshold for output? (y or n) ");
    getString(pString, winput);
    if (winput[0] == 'y')
    {
      snprintf(pString,100,"Enter the lower threshold (min = %e): ",GYmin);
      threshL = getDouble(pString);
    }
    double threshU = GYmax + 0.2 * PABS(GYmax);
    snprintf(pString,100,"Set upper threshold for output? (y or n) ");
    getString(pString, winput);
    if (winput[0] == 'y')
    {
      snprintf(pString,100,"Enter the upper threshold (min = %e): ",GYmax);
      threshU = getDouble(pString);
    }
    fprintf(fp,"twoPlots = 1;\n");
    fprintf(fp,"disp(\'Please wait while loading data.\')\n");
    fprintf(fp,"hold off\n");
    fwritePlotCLF(fp);

    //**/ generating visualization data
    for (ii = 0; ii < nPtsPerDim; ii++)
    {
      vecInpSettings[iplot3] = (iUpperB[iplot3] - iLowerB[iplot3]) / 
                              (nPtsPerDim - 1.0) * ii + iLowerB[iplot3];
      //**/ x and y data only needed to output once
      fprintf(fp,"x = [\n");
      for (sInd = 0; sInd < faLeng; sInd+=nPtsPerDim*nPtsPerDim)
        fprintf(fp, "%e\n", faXOut[sInd*3]);
      fprintf(fp,"];\n");
      fprintf(fp,"y = [\n");
      for (sInd = 0; sInd < nPtsPerDim*nPtsPerDim; sInd+=nPtsPerDim)
        fprintf(fp, "%e\n", faXOut[sInd*3+1]);
      fprintf(fp,"];\n");

      //**/ output the response data data
      fprintf(fp,"A%d = [\n", ii + 1);
      for (sInd = 0; sInd < faLeng; sInd+=nPtsPerDim)
        fprintf(fp, "%e\n", faYOut[sInd+ii]);
      fprintf(fp,"];\n");
      fprintf(fp,"A%d = reshape(A%d,%d,%d);\n", ii+1, ii+1,
              nPtsPerDim, nPtsPerDim);
      fprintf(fp,"disp(\'Plotting frame %d of %d\')\n",ii+1,nPtsPerDim);
      fprintf(fp,"B%d = A%d;\n", ii+1, ii+1);
      fprintf(fp,"yLo = %e;\n", threshL);
      fprintf(fp,"yHi = %e;\n", threshU);
      fprintf(fp,"nA  = size(A%d,1);\n", ii+1);
      fprintf(fp,"[ia,ja,aa] = find(A%d<yLo);\n", ii+1);
      fprintf(fp,"for ii = 1 : length(ia)\n");
      fprintf(fp,"   B%d(ia(ii),ja(ii)) = NaN;\n", ii+1);
      fprintf(fp,"end;\n");
      fprintf(fp,"n1 = length(ia);\n");
      fprintf(fp,"[ia,ja,aa] = find(A%d>yHi);\n", ii+1);
      fprintf(fp,"for ii = 1 : length(ia)\n");
      fprintf(fp,"   B%d(ia(ii),ja(ii)) = NaN;\n", ii+1);
      fprintf(fp,"end;\n");
      fprintf(fp,"n2 = length(ia);\n");
      fprintf(fp,"if (n1 + n2 == nA*nA)\n");
      fprintf(fp,"   B%d(1,1) = 0;\n",ii+1);
      fprintf(fp,"   B%d(%d,%d) = 1;\n",ii+1,nPtsPerDim,nPtsPerDim);
      fprintf(fp,"end;\n");
      fprintf(fp,"if twoPlots == 1\n");
      fprintf(fp,"subplot(1,2,1), surf(x,y,A%d)\n", ii+1);
      fprintf(fp,"axis([%e %e %e %e %e %e])\n",iLowerB[iplot1],
              iUpperB[iplot1],iLowerB[iplot2],iUpperB[iplot2],GYmin, GYmax); 
      fwritePlotAxes(fp);
      fprintf(fp,"xlabel('%s','FontSize',12,'FontWeight','bold')\n",
              inputNames[iplot1]);
      fprintf(fp,"ylabel('%s','Fontsize',12,'FontWeight','bold')\n",
              inputNames[iplot2]);
      fprintf(fp,"zlabel('%s','Fontsize',12,'FontWeight','bold')\n",
              outputNames[jplot]);
      fprintf(fp,"colorbar\n");
      fprintf(fp,"title(\'%s Mesh plot, val(3) = %14.7e\',",
              outputNames[jplot], vecInpSettings[iplot3]);
      fprintf(fp,"'FontWeight','bold','FontSize',12)\n");
      fprintf(fp,"subplot(1,2,2)\n");
      fprintf(fp,"end\n");
      fprintf(fp,"contourf(x,y,B%d)\n",ii+1);
      fprintf(fp,"axis([%e %e %e %e])\n",iLowerB[iplot1],
              iUpperB[iplot1],iLowerB[iplot2],iUpperB[iplot2]);
      fwritePlotAxes(fp);
      fprintf(fp,"colorbar\n");
      fprintf(fp,"colormap(jet)\n");
      fprintf(fp,"caxis([%e %e])\n",GYmin, GYmax);
      fprintf(fp,"title(\'%s contour plot, val(3) = %14.7e\',",
              outputNames[jplot], vecInpSettings[iplot3]);
      fprintf(fp,"'FontWeight','bold','FontSize',12)\n");
      fprintf(fp,"pause(1)\n");
    }
    fprintf(fp,"rotate3d on\n");
    fclose(fp);
    printf("rsplot3m plot file = matlabrsplot3m.m\n");
    delete [] faXOut;
    delete [] faYOut;
    delete faPtr;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rs4 or rsplot4m
  //**/ generate response surface of any 4 inputs and write the
  //**/ grid data to file for display with matlab
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rs4") ||
          !strcmp(command, "rsplot4m"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsplot4m: 4-input-1-output response surface plot.\n");
      printf("syntax: rsplot4 (no argument needed)\n");
      return 0;
    }
    if (nInputs <= 0 || psuadeIO == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nInputs < 4)
    {
      printf("ERROR: rsplot4m requires 4 or more inputs.\n");
      return 1;
    }
    if (plotScilab())
    {
      printf("INFO: rsplot4m is currently not available in scilab.\n");
      return 1;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command creates a Matlab 3D movie (4 inputs/1 output).\n");
    printf("3 selected inputs will be in X, Y, and Z axes.\n");
    printf("The 4th input and output will be displayed in the time axis.\n");
    printf("The other inputs are set at their midpoints or user-specified.\n");
    printf("You will be asked to select a response surface type.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ set up the function approximator
    int nPtsPerDim = 16;
    printf("NOTE: if matlab crashes, it may be due to high grid resolution\n");
    snprintf(pString,100,"Grid resolution ? (16 - 32) ");
    nPtsPerDim = getInt(16, 32, pString);
    int faFlag = 1;
    FuncApprox *faPtr = genFAInteractive(psuadeIO, faFlag);
    if (faPtr == NULL) {printf("ERROR detected.\n"); return 1;}
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);

    //**/ ask users to specify the three inputs and one output
    int iplot1, iplot2, iplot3, iplot4, iInd1;
    psVector vecInpSettings;
    vecInpSettings.setLength(nInputs);
    snprintf(pString,100,"Enter the input for x axis (1 - %d) : ", nInputs);
    iplot1 = getInt(1, nInputs, pString);
    iplot1--;
    snprintf(pString,100,"Enter the input for y axis (1 - %d) : ", nInputs);
    iplot2 = getInt(1, nInputs, pString);
    iplot2--;
    snprintf(pString,100,"Enter the input for z axis (1 - %d) : ", nInputs);
    iplot3 = getInt(1, nInputs, pString);
    iplot3--;
    snprintf(pString,100,"Enter the input for t axis (1 - %d) : ", nInputs);
    iplot4 = getInt(1, nInputs, pString);
    iplot4--;

    int p4cnt=1;
    if (iplot2 != iplot1) p4cnt++;
    if (iplot3 != iplot1 && iplot3 != iplot2) p4cnt++;
    if (iplot4 != iplot1 && iplot4 != iplot2 && iplot4 != iplot3) p4cnt++;

    if (nInputs-p4cnt > 0)
    {
      snprintf(pString,100,"Set other inputs at their mid points ? (y/n) ");
      getString(pString, winput);
      if (winput[0] == 'y')
      {
        for (iInd1 = 0; iInd1 < nInputs; iInd1++)
        {
          if (iInd1 != iplot1 && iInd1 != iplot2 && iInd1 != iplot3 &&
              iInd1 != iplot4)
               vecInpSettings[iInd1] = 0.5*(iLowerB[iInd1]+iUpperB[iInd1]);
          else vecInpSettings[iInd1] = 1.0;
        }
      }
      else
      {
        for (iInd1 = 0; iInd1 < nInputs; iInd1++)
        {
          if (iInd1 != iplot1 && iInd1 != iplot2 && iInd1 != iplot3 &&
              iInd1 != iplot4)
          {
             vecInpSettings[iInd1] = iLowerB[iInd1] - 1.0;
             snprintf(pString,100,
                   "Enter nominal value for input %d (%e - %e): ", 
                   iInd1+1, iLowerB[iInd1], iUpperB[iInd1]);
             while (vecInpSettings[iInd1] < iLowerB[iInd1] ||
                    vecInpSettings[iInd1] > iUpperB[iInd1])
               vecInpSettings[iInd1] = getDouble(pString);
          }
          else vecInpSettings[iInd1] = 1.0;
        }
      }
    }
    int jplot=0, sInd;
    snprintf(pString,100,"Enter the output number (1 - %d) : ", nOutputs);
    jplot = getInt(1, nOutputs, pString);
    jplot--;

    psVector vecFaYIn;
    vecFaYIn.setLength(nSamples);
    for (sInd = 0; sInd < nSamples; sInd++)
      vecFaYIn[sInd] = sampleOutputs[sInd*nOutputs+jplot];

    //**/ search for extrema
    int    faLeng=0;
    double *faXOut=NULL, *faYOut=NULL;
    faPtr->gen4DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,iplot2,
              iplot3,iplot4,vecInpSettings.getDVector(),&faLeng,&faXOut,
              &faYOut);
    double GYmin =   PSUADE_UNDEFINED;
    double GYmax = - PSUADE_UNDEFINED;
    for (sInd = 0; sInd < faLeng; sInd++)
      if (faYOut[sInd] < GYmin) GYmin = faYOut[sInd];
    for (sInd = 0; sInd < faLeng; sInd++)
    if (faYOut[sInd] > GYmax) GYmax = faYOut[sInd];
    printf("\nYmin and Ymax found = %e %e.\n", GYmin, GYmax);
    double threshL = GYmin - 0.2 * PABS(GYmax - GYmin);
    printf("You can set thresholds to cut out certain regions.\n");
    snprintf(pString,100,"Set lower threshold for output? (y or n) ");
    double gamma = threshL;
    getString(pString, winput);
    if (winput[0] == 'y')
    {
      snprintf(pString,100,"Enter the lower threshold (min = %e): ",GYmin);
      threshL = getDouble(pString);
    }
    double threshU = GYmax + 0.2 * PABS(GYmax - GYmin);
    snprintf(pString,100,"Set upper threshold for output? (y or n) ");
    getString(pString, winput);
    if (winput[0] == 'y')
    {
      snprintf(pString,100,"Enter the upper threshold (max = %e): ",GYmax);
      threshU = getDouble(pString);
    }

    //**/ begin generating data
    fp = fopen("matlabrsplot4m.m", "w");
    if (fp == NULL)
    {
      printf("ERROR: Cannot open file matlabrsplot4m.m.\n");
      if (faXOut != NULL) delete [] faXOut;
      if (faYOut != NULL) delete [] faYOut;
      if (faPtr  != NULL) delete faPtr;
      return 1;
    }

    fprintf(fp,"%% user adjustable parameter section begins *****\n");
    fprintf(fp,"%% use nSubplots, nSubNx and nSubNy to spread \n");
    fprintf(fp,"%% the movie frames into a number of subplots.\n");
    fprintf(fp,"nSubplots = 1;\n");
    fprintf(fp,"nSubNx = 1;\n");
    fprintf(fp,"nSubNy = 1;\n");
    fprintf(fp,"%% user adjustable parameter section ends *****\n");
    fwritePlotCLF(fp);
    fprintf(fp,"nFrames = %d;\n", nPtsPerDim);
    fprintf(fp,"nSubCnt = 0;\n");
    fprintf(fp,"isoval = %e;\n", threshL);
    fprintf(fp,"xlo = %e; \n", iLowerB[iplot2]);
    fprintf(fp,"xhi = %e; \n", iUpperB[iplot2]);
    fprintf(fp,"ylo = %e; \n", iLowerB[iplot1]);
    fprintf(fp,"yhi = %e; \n", iUpperB[iplot1]);
    fprintf(fp,"zlo = %e; \n", iLowerB[iplot3]);
    fprintf(fp,"zhi = %e; \n", iUpperB[iplot3]);
    fprintf(fp,"X=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
    fprintf(fp,"Y=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
    fprintf(fp,"Z=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
    fprintf(fp,"V=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
    int ind;
    for (ii = 0; ii < nPtsPerDim; ii++)
    {
      //**/ x and y data only needed to output once
      fprintf(fp,"Y(:,:,%d) = [\n", ii + 1);
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
      {
        for (jj = 0; jj < nPtsPerDim; jj++)
        {
          ind = (sInd*nPtsPerDim*nPtsPerDim+jj*nPtsPerDim+ii)*nPtsPerDim;
          fprintf(fp, "%e ", faXOut[ind*4]);
        }
        fprintf(fp, "\n");
      }
      fprintf(fp,"];\n");
      fprintf(fp,"X(:,:,%d) = [\n", ii + 1);
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
      {
        for (jj = 0; jj < nPtsPerDim; jj++)
        {
          ind = (sInd*nPtsPerDim*nPtsPerDim+jj*nPtsPerDim+ii)*nPtsPerDim;
          fprintf(fp, "%e ", faXOut[ind*4+1]);
        }
        fprintf(fp, "\n");
      }
      fprintf(fp,"];\n");
      fprintf(fp,"Z(:,:,%d) = [\n", ii + 1);
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
      {
        for (jj = 0; jj < nPtsPerDim; jj++)
        {
          ind = (sInd*nPtsPerDim*nPtsPerDim+jj*nPtsPerDim+ii)*nPtsPerDim;
          fprintf(fp,"%e ", faXOut[ind*4+2]);
        }
        fprintf(fp, "\n");
      }
      fprintf(fp, "];\n");
    }
    //**/fprintf(fp, "xt = [%e:%e:%e];\n", iLowerB[iplot2],
    //**/        (iUpperB[iplot2]-iLowerB[iplot2])*0.05, iUpperB[iplot2]);
    //**/fprintf(fp, "yt = [%e:%e:%e];\n", iLowerB[iplot1],
    //**/        (iUpperB[iplot1]-iLowerB[iplot1])*0.05, iUpperB[iplot1]);
    //**/fprintf(fp, "zt = [%e:%e:%e];\n", iLowerB[iplot3],
    //**/        (iUpperB[iplot3]-iLowerB[iplot3])*0.05, iUpperB[iplot3]);
    int count, ll;
    for (ll = 0; ll < nPtsPerDim; ll++)
    {
      for (ii = 0; ii < nPtsPerDim; ii++)
      {
        count = 0;
        //**/ x and y data only needed to output once
        fprintf(fp,"V(:,:,%d) = [\n", ii + 1);
        for (sInd = 0; sInd < nPtsPerDim; sInd++)
        {
          for (jj = 0; jj < nPtsPerDim; jj++)
          {
            ind = ((sInd*nPtsPerDim+jj)*nPtsPerDim+ii)*nPtsPerDim+ll;
            if (faYOut[ind] < threshL)
            {
              fprintf(fp, "%e ", threshL);
              count++;
            }
            else if (faYOut[ind] > threshU)
            {
              //**/fprintf(fp, "%e ", threshU);
              //**/ set it at isoval
              fprintf(fp, "%e ", threshL);
              count++;
            }
            else fprintf(fp, "%e ", faYOut[ind]);
          }
          fprintf(fp, "\n");
        }
        fprintf(fp,"];\n");
        if (count == nPtsPerDim*nPtsPerDim)
        {
          if (threshL-0.2*(threshU-threshL) > gamma)
             fprintf(fp,"V(:,:,%d) = %e * ones(%d,%d);\n",ii+1,gamma,
                     nPtsPerDim, nPtsPerDim);
          else
             fprintf(fp,"V(:,:,%d) = %e * ones(%d,%d);\n",ii+1,
                     threshL-0.2*(threshU-threshL),
                     nPtsPerDim,nPtsPerDim);
          printf("Frame %d, slice %d nonfeasible -> set to ground.\n",
                 ll+1, ii+1);
        }
      }
      fprintf(fp,"frame = %d;\n", ll+1);
      fprintf(fp,"if nSubplots > 1\n");
      fprintf(fp,"   if frame <= 2\n");
      fprintf(fp,"      nSubCnt = nSubCnt + 1;\n");
      fprintf(fp,"      subplot(nSubNx, nSubNy, nSubCnt)\n");
      fprintf(fp,"   elseif frame == nFrames\n");
      fprintf(fp,"      subplot(nSubNx, nSubNy, nSubplots)\n");
      fprintf(fp,"   else\n");
      fprintf(fp,"      ft1 = (nFrames-1) / (nSubplots-1);\n");
      fprintf(fp,"      ft2 = round(ft1 * (nSubCnt-1)) + 2;\n");
      fprintf(fp,"      if frame == ft2\n");
      fprintf(fp,"         nSubCnt = nSubCnt + 1;\n");
      fprintf(fp,"         subplot(nSubNx, nSubNy, nSubCnt)\n");
      fprintf(fp,"      end\n");
      fprintf(fp,"   end\n");
      fprintf(fp,"else\n");
      fprintf(fp,"   clf\n");
      fprintf(fp,"end\n");
      fprintf(fp,"disp('Frame %d of %d')\n", ll+1, nPtsPerDim);
      //**/ Nov 25, 2008: old, new using isosurface is better
      //**/ fprintf(fp, "h = contourslice(x,y,z,v,xt,yt,zt,21);\n");
      //**/ fprintf(fp, "axis([min(min(min(x))) max(max(max(x))) ");
      //**/ fprintf(fp, "min(min(min(y))) max(max(max(y))) ");
      //**/ fprintf(fp, "min(min(min(z))) max(max(max(z))) ");
      //**/ if (threshL-0.2*(threshU-threshL) > gamma)
      //**/    fprintf(fp, " %e %e])\n",gamma,threshU+0.2*(threshU-threshL));
      //**/ else
      //**/    fprintf(fp, " %e %e])\n", threshL-0.2*(threshU-threshL), 
      //**/            threshU+0.2*(threshU-threshL));
      //**/ fprintf(fp, "view(-40,60)\n");
      //**/ fprintf(fp, "set(h, 'Linewidth', 5)\n");
      //**/ fprintf(fp, "box on\n");
      //**/ fprintf(fp, "grid on\n");
      fprintf(fp,"h = patch(isosurface(X,Y,Z,V,isoval),... \n");
      fprintf(fp,"          'FaceColor', 'blue', ... \n");
      fprintf(fp,"          'EdgeColor', 'none', ... \n");
      fprintf(fp,"          'AmbientStrength', 0.2, ... \n");
      fprintf(fp,"          'SpecularStrength', 0.7, ... \n");
      fprintf(fp,"          'DiffuseStrength', 0.4);\n");
      fprintf(fp,"isonormals(X,Y,Z,V,h);\n");
      fprintf(fp,"patch(isocaps(X,Y,Z,V,isoval), ...\n");
      fprintf(fp,"      'FaceColor', 'interp', ... \n");
      fprintf(fp,"      'EdgeColor', 'none'); \n");
      fprintf(fp,"axis([xlo xhi ylo yhi zlo zhi])\n");
      fprintf(fp,"daspect([xhi-xlo, yhi-ylo, zhi-zlo])\n");
      fprintf(fp,"colormap('default')\n");
      fprintf(fp,"if nSubplots == 1\n");
      fprintf(fp,"   colorbar\n");
      fprintf(fp,"end\n");
      fprintf(fp,"%%axis tight\n");
      fprintf(fp,"view(3) \n");
      //**/ fprintf(fp, "camlight right \n");
      //**/ fprintf(fp, "camlight left \n");
      fprintf(fp,"set(gcf,'Renderer','zbuffer')\n");
      fprintf(fp,"box on\n");
      fprintf(fp,"grid on\n");
      fprintf(fp,"lighting phong\n");
      fwritePlotAxes(fp);
      if (ll == 0)
      {
        fprintf(fp,"xlabel('%s','FontSize',12,'FontWeight','bold')\n",
                inputNames[iplot2]);
        fprintf(fp,"ylabel('%s','Fontsize',12,'FontWeight','bold')\n",
                inputNames[iplot1]);
        fprintf(fp,"zlabel('%s','Fontsize',12,'FontWeight','bold')\n",
                inputNames[iplot3]);
      }
      fprintf(fp,"title('%s=%12.4e',",inputNames[iplot4],faXOut[ll*4+3]);
      fprintf(fp,"'FontWeight','bold','FontSize',12)\n");
      fprintf(fp,"pause(1)\n");
    }
    fclose(fp);
    if (faXOut != NULL) delete [] faXOut;
    if (faYOut != NULL) delete [] faYOut;
    if (faPtr != NULL) delete faPtr;
    printf("rsplot4m plot file = matlabrsplot4m.m\n");
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rssd  or rsplot_sd
  //**/ generate standard deviation response surface and write the
  //**/ grid data to file for display with matlab
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rssd") ||
           !strcmp(command, "rsplot_sd"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsplot_sd: RS of prediction std. dev. (1D-4D).\n");
      printf("INFO: rsplot_sd not available for >2 inputs for scilab.\n");
      printf("syntax: rsplot_sd (no argument needed.\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command creates a Matlab response surface plot ");
    printf("for the prediction\n");
    printf("uncertainties in the selected input space.\n");
    printf("You can select up to 4 inputs.\n");
    printf("The other inputs are set at their midpoints ");
    printf("or user-specified.\n");
    printf("You will be asked to select a response ");
    printf("surface (RS) type.\n");
    printf("The selected RS should give prediction ");
    printf("uncertainty (e.g. GP).\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;
    
    //**/ ask users to specify the three inputs and one output
    int iplot1, iplot2, iplot3, iplot4, count, iInd1, sInd, ind, ll;
    psVector vecInpSettings;
    vecInpSettings.setLength(nInputs);
    iplot1 = iplot2 = iplot3 = iplot4 = -1;
    snprintf(pString,100,"Enter the input for x axis (1 - %d) : ", 
             nInputs);
    iplot1 = getInt(1, nInputs, pString);
    iplot1--;
    count = 1;
    if (nInputs == 2) iplot2 = nInputs - iplot1 - 1;
    else if (nInputs > 2)
    {
      iplot2 = iplot1;
      while (iplot1 == iplot2)
      {
        snprintf(pString,100,
           "Y-axis input ? (1-%d, 0 if not used, not %d) ",
           nInputs,iplot1+1);
        iplot2 = getInt(0, nInputs, pString);
        iplot2--;
        if (iplot2 == -1) break;
        if (iplot1 == iplot2)
          printf("ERROR: Duplicate input number %d.\n",iplot2+1);
      }
    }
    if (iplot2 != -1) count++;
    if (plotMatlab() && iplot2 != -1)
    {
      if (nInputs == 3) iplot3 = nInputs - iplot1 - iplot2;
      else if (nInputs > 3)
      {
        iplot3 = iplot1;
        while (iplot3 == iplot1 || iplot3 == iplot2)
        {
          snprintf(pString,100,
             "Z axis input ? (1-%d, 0 if not used, not %d nor %d) ",
             nInputs, iplot1+1, iplot2+1);
          iplot3 = getInt(0, nInputs, pString);
          iplot3--;
          if (iplot3 == -1) break;
          if (iplot3 == iplot1 || iplot3 == iplot2)
             printf("ERROR: Duplicate input number %d.\n",iplot3+1);
        }
      }
      if (iplot3 != -1) count++;
      if (nInputs >= 4 && iplot3 != -1)
      {
        while (iplot4 < 0 || iplot4 == iplot1 || iplot4 == iplot2 ||  
               iplot4 == iplot3)
        {
          snprintf(pString,100,
             "Enter the input for t axis (1 - %d), not %d nor %d,%d: ",
                  nInputs, iplot1+1, iplot2+1, iplot3+1);
          iplot4 = getInt(1, nInputs, pString);
          iplot4--;
          if (iplot4 == iplot1 || iplot4 == iplot2 || iplot4 == iplot3)
             printf("ERROR: Duplicate input number %d.\n",iplot4+1);
        }
      } 
      if (iplot4 != -1) count++;
    }
    strcpy(winput, "y");
    if (nInputs > count)
    {
      snprintf(pString,100,"Set other inputs at their mid points ? (y/n) ");
      getString(pString, winput);
    }
    if (winput[0] == 'y')
    {
      for (iInd1 = 0; iInd1 < nInputs; iInd1++)
      {
        if (iInd1 != iplot1 && iInd1 != iplot2 && iInd1 != iplot3 &&
            iInd1 != iplot4)
             vecInpSettings[iInd1] = 0.5*(iLowerB[iInd1]+iUpperB[iInd1]);
        else vecInpSettings[iInd1] = 1.0;
      }
    }
    else
    {
      for (iInd1 = 0; iInd1 < nInputs; iInd1++)
      {
        if (iInd1 != iplot1 && iInd1 != iplot2 && iInd1 != iplot3 &&
            iInd1 != iplot4)
        {
          vecInpSettings[iInd1] = iLowerB[iInd1] - 1.0;
          snprintf(pString,100,
                  "Enter nominal value for input %d (%e - %e): ", 
                  iInd1+1, iLowerB[iInd1], iUpperB[iInd1]);
          while (vecInpSettings[iInd1] < iLowerB[iInd1] ||
                 vecInpSettings[iInd1] > iUpperB[iInd1])
            vecInpSettings[iInd1] = getDouble(pString);
        }
        else vecInpSettings[iInd1] = 1.0;
      }
    }
    //**/ set up the function approximator
    int nPtsPerDim;
    if      (iplot2 == -1) nPtsPerDim = 1024;
    else if (iplot3 == -1) nPtsPerDim = 128;
    else if (iplot4 == -1) nPtsPerDim = 24;
    else                   nPtsPerDim = 10;
    printf("This command works with the following response surfaces:\n");
    printf("1. Linear    regression\n");
    printf("2. Quadratic regression\n");
    printf("3. cubic     regression\n");
    printf("4. quartic   regression\n");
    printf("5. GP1 (MacKay)\n");
    printf("6. GP3 (Tong)\n");
    printf("7. MarsBagg\n");
    printf("8. Tree GP\n");
    printf("9. Kriging\n");
    snprintf(pString,100,"Enter your choice: (1, 2, ..., 9) ");
    int faType = getInt(1, 9, pString);
    if      (faType == 1) faType = PSUADE_RS_REGR1;
    else if (faType == 2) faType = PSUADE_RS_REGR2;
    else if (faType == 3) faType = PSUADE_RS_REGR3;
    else if (faType == 4) faType = PSUADE_RS_REGR4;
    else if (faType == 5) faType = PSUADE_RS_GP1;
    else if (faType == 6) faType = PSUADE_RS_GP3;
    else if (faType == 7) faType = PSUADE_RS_MARSB;
    else if (faType == 8) faType = PSUADE_RS_TGP;
    else if (faType == 9) faType = PSUADE_RS_KR;
    int faFlag = 1, iOne=1;
    FuncApprox *faPtr = genFA(faType, nInputs, iOne, nSamples);
    if (faPtr == NULL) {printf("ERROR detected.\n"); return 1;}
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);

    //**/ ask users to specify the three inputs and one output
    int jplot = 0;
    snprintf(pString,100,
             "Enter the output number (1 - %d) : ",nOutputs);
    jplot = getInt(1, nOutputs, pString);
    jplot--;

    psVector vecFaYIn;
    vecFaYIn.setLength(nSamples);
    for (sInd = 0; sInd < nSamples; sInd++)
      vecFaYIn[sInd] = sampleOutputs[sInd*nOutputs+jplot];

    //**/ generate data points 
    int faLeng = 0;
    double *faXOut=NULL, *faYOut=NULL;
    if (iplot2 == -1)
      faPtr->gen1DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,
               vecInpSettings.getDVector(), &faLeng, &faXOut,&faYOut);
    else if (iplot3 == -1)
      faPtr->gen2DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,
               iplot2,vecInpSettings.getDVector(),&faLeng,&faXOut,&faYOut);
    else if (iplot4 == -1)
      faPtr->gen3DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,
               iplot2,iplot3,vecInpSettings.getDVector(),&faLeng,&faXOut,
               &faYOut);
    else
      faPtr->gen4DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,
               iplot2,iplot3,iplot4,vecInpSettings.getDVector(),&faLeng, 
               &faXOut,&faYOut);

    //**/ re-generate to include standard deviation 
    psVector vecWT, vecXT;
    vecWT.setLength(faLeng);
    vecXT.setLength(faLeng*nInputs);
    for (sInd = 0; sInd < faLeng; sInd++)
      for (jj = 0; jj < nInputs; jj++)
        vecXT[sInd*nInputs+jj] = vecInpSettings[jj];
    for (sInd = 0; sInd < faLeng; sInd++)
    {
      vecXT[sInd*nInputs+iplot1] = faXOut[sInd*count];
      if (iplot2 != -1)
        vecXT[sInd*nInputs+iplot2] = faXOut[sInd*count+1];
      if (iplot3 != -1)
        vecXT[sInd*nInputs+iplot3] = faXOut[sInd*count+2];
      if (iplot4 != -1)
        vecXT[sInd*nInputs+iplot4] = faXOut[sInd*count+3];
    }
    faPtr->evaluatePointFuzzy(faLeng, vecXT.getDVector(), faYOut, 
                              vecWT.getDVector());
    double gamma = PSUADE_UNDEFINED;
    for (sInd = 0; sInd < faLeng; sInd++)
      if (vecWT[sInd] < gamma) gamma = vecWT[sInd];
        
    //**/ begin generating data
    if (plotScilab())
    {
      fp = fopen("scilabrsplot_sd.sci", "w");
      if (fp == NULL)
      {
        printf("ERROR: Cannot open file scilabrsplot_sd.sci.\n");
        delete [] faXOut;
        delete [] faYOut;
        delete faPtr;
        return 1;
      }
    }
    else
    {
      fp = fopen("matlabrsplot_sd.m", "w");
      if (fp == NULL)
      {
        printf("ERROR: Cannot open file matlabrsplot_sd.m.\n");
        delete [] faXOut;
        delete [] faYOut;
        delete faPtr;
        return 1;
      }
    }
    fwritePlotCLF(fp);
    if (count == 1)
    {
      fprintf(fp, "A = [\n");
      for (sInd = 0; sInd < faLeng; sInd++)
        fprintf(fp, "%e\n", vecWT[sInd]);
      fprintf(fp, "];\n");
      fprintf(fp, "X = [\n");
      for (sInd = 0; sInd < faLeng; sInd++)
        fprintf(fp, "%e\n", faXOut[sInd]);
      fprintf(fp, "];\n");
      if (plotScilab())
      {
        fprintf(fp, "plot(X,A);");
        fprintf(fp, "a = gca();\n");
        fprintf(fp, "a.children.children.thickness = 4;\n");
        fprintf(fp, "set(gca(),\"auto_clear\",\"off\")\n");
      }
      else 
      {
        fprintf(fp, "plot(X,A,'lineWidth',4)\n");
        fprintf(fp, "hold on\n");
      }
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, inputNames[iplot1]);
      fwritePlotYLabel(fp, outputNames[jplot]);
      snprintf(winput,100,"Std. Dev. Plot for %s", outputNames[jplot]);
      fwritePlotTitle(fp, winput);
    }
    else if (count == 2)
    {
      if (plotMatlab()) fprintf(fp, "twoPlots = 1;\n");
      fprintf(fp, "A = [\n");
      for (sInd = 0; sInd < faLeng; sInd++)
        fprintf(fp, "%e\n", vecWT[sInd]);
      fprintf(fp, "];\n");
      if (plotScilab()) 
        fprintf(fp, "A = matrix(A,%d,%d);\n", nPtsPerDim, nPtsPerDim);
      else
        fprintf(fp, "A = reshape(A,%d,%d);\n", nPtsPerDim, nPtsPerDim);
      fprintf(fp, "X = [\n");
      for (sInd = 0; sInd < faLeng; sInd+=nPtsPerDim)
        fprintf(fp, "%e\n", faXOut[sInd*2]);
      fprintf(fp, "];\n");
      fprintf(fp, "Y = [\n");
      for (sInd = 0; sInd < nPtsPerDim; sInd++)
        fprintf(fp, "%e\n", faXOut[sInd*2+1]);
      fprintf(fp, "];\n");
      if (plotScilab())
      {
        fprintf(fp, "mesh(X,Y,A)\n");
        fprintf(fp, "h = get(\"hdl\");\n");
        fprintf(fp, "h.color_flag=1;\n");
        fprintf(fp, "h.color_mode=-2;\n");
        fprintf(fp, "bmin = min(min(A)); bmax = max(max(A));\n");
        fprintf(fp, "xset(\"colormap\",jetcolormap(64));\n");
        fprintf(fp, "colorbar(bmin,bmax);\n");
        fwritePlotAxes(fp);
        fwritePlotXLabel(fp, inputNames[iplot1]);
        fwritePlotYLabel(fp, inputNames[iplot2]);
        fwritePlotZLabel(fp, outputNames[jplot]);
        snprintf(winput,100,"Std. Dev. Plot for %s", outputNames[jplot]);
        fwritePlotTitle(fp, winput);
        fprintf(fp, "scf(2);\n");
        fprintf(fp, "a=gca();\n");
        fprintf(fp, "a.data_bounds=[%e,%e;%e,%e];\n",
                iLowerB[iplot1], iLowerB[iplot2],
                iUpperB[iplot1], iUpperB[iplot2]);
        fprintf(fp, "a.axes_visible=\"on\";\n");
        fprintf(fp, "B = A;\n");
        fprintf(fp, "nX = length(X);\n");
        fprintf(fp, "nY = length(Y);\n");
        fprintf(fp, "for ii = 1 : nX\n");
        fprintf(fp, "for jj = 1 : nY\n");
        fprintf(fp, "B(ii,jj) = A(nX-ii+1,jj);\n");
        fprintf(fp, "end;\n");
        fprintf(fp, "end;\n");
        fprintf(fp, "Matplot1((B-bmin)/(bmax-bmin)*64,[%e,%e,%e,%e])\n",
                iLowerB[iplot1],iLowerB[iplot2],
                iUpperB[iplot1], iUpperB[iplot2]);
        fprintf(fp, "xset(\"colormap\",jetcolormap(64));\n");
        fprintf(fp, "colorbar(bmin,bmax);\n");
        fprintf(fp, "a.thickness = 2;\n");
        fprintf(fp, "a.font_size = 3;\n");
        fprintf(fp, "a.font_style = 4;\n");
        fprintf(fp, "a.box = \"on\";\n");
        fprintf(fp, "a.grid = [1 1];\n");
        fwritePlotAxes(fp);
        fwritePlotXLabel(fp, inputNames[iplot1]);
        fwritePlotYLabel(fp, inputNames[iplot2]);
        snprintf(winput,100,"Std. Dev. Plot for %s", outputNames[jplot]);
        fwritePlotTitle(fp, winput);
      }
      else
      { 
        fprintf(fp, "if twoPlots == 1\n");
        fprintf(fp, "subplot(1,2,1), surf(X,Y,A)\n");
        fwritePlotAxes(fp);
        fwritePlotXLabel(fp, inputNames[iplot1]);
        fwritePlotYLabel(fp, inputNames[iplot2]);
        fwritePlotZLabel(fp, outputNames[jplot]);
        snprintf(winput,100,"Std. Dev. Plot for %s", outputNames[jplot]);
        fwritePlotTitle(fp, winput);
        fprintf(fp, "colorbar\n");
        fprintf(fp, "subplot(1,2,2)\n");
        fprintf(fp, "end\n");
        fprintf(fp, "contourf(X,Y,A)\n");
        fwritePlotAxes(fp);
        fwritePlotXLabel(fp, inputNames[iplot1]);
        fwritePlotYLabel(fp, inputNames[iplot2]);
        snprintf(winput,100,"Std. Dev. Plot for %s", outputNames[jplot]);
        fwritePlotTitle(fp, winput);
        fprintf(fp, "colorbar\n");
        fprintf(fp, "colormap(jet)\n");
      }
    }
    else if (count == 3)
    {
      fprintf(fp,"xlo = %e; \n", iLowerB[iplot2]);
      fprintf(fp,"xhi = %e; \n", iUpperB[iplot2]);
      fprintf(fp,"ylo = %e; \n", iLowerB[iplot1]);
      fprintf(fp,"yhi = %e; \n", iUpperB[iplot1]);
      fprintf(fp,"zlo = %e; \n", iLowerB[iplot3]);
      fprintf(fp,"zhi = %e; \n", iUpperB[iplot3]);
      fprintf(fp,"X=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
      fprintf(fp,"Y=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
      fprintf(fp,"Z=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
      fprintf(fp,"V=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
      for (jj = 0; jj < nPtsPerDim; jj++)
      {
        fprintf(fp, "Y(:,:,%d) = [\n", jj + 1);
        for (sInd = 0; sInd < nPtsPerDim; sInd++)
        {
          for (ii = 0; ii < nPtsPerDim; ii++)
          {
            ind = sInd*nPtsPerDim*nPtsPerDim+ii*nPtsPerDim+jj;
            fprintf(fp, "%e ", faXOut[ind*3]);
          }
          fprintf(fp, "\n");
        }
        fprintf(fp, "];\n");
        fprintf(fp, "X(:,:,%d) = [\n", jj + 1);
        for (sInd = 0; sInd < nPtsPerDim; sInd++)
        {
          for (ii = 0; ii < nPtsPerDim; ii++)
          {
            ind = sInd*nPtsPerDim*nPtsPerDim+ii*nPtsPerDim+jj;
            fprintf(fp, "%e ", faXOut[ind*3+1]);
          }
          fprintf(fp, "\n");
        }
        fprintf(fp, "];\n");
        fprintf(fp, "Z(:,:,%d) = [\n", jj + 1);
        for (sInd = 0; sInd < nPtsPerDim; sInd++)
        {
          for (ii = 0; ii < nPtsPerDim; ii++)
          {
            ind = sInd*nPtsPerDim*nPtsPerDim+ii*nPtsPerDim+jj;
            fprintf(fp, "%e ", faXOut[ind*3+2]);
          }
          fprintf(fp, "\n");
        }
        fprintf(fp, "];\n");
      }
      double GYmax = - PSUADE_UNDEFINED;
      double GYmin =   PSUADE_UNDEFINED;
      for (jj = 0; jj < nPtsPerDim; jj++)
      {
        fprintf(fp, "V(:,:,%d) = [\n", jj + 1);
        for (sInd = 0; sInd < nPtsPerDim; sInd++)
        {
          for (ii = 0; ii < nPtsPerDim; ii++)
          {
            ind = sInd*nPtsPerDim*nPtsPerDim+ii*nPtsPerDim+jj;
            fprintf(fp, "%e ", vecWT[ind]);
            if (vecWT[ind] > GYmax) GYmax = vecWT[ind];
            if (vecWT[ind] < GYmin) GYmin = vecWT[ind];
          }
          fprintf(fp, "\n");
        }
        fprintf(fp, "];\n");
      }
      fprintf(fp, "xt = [%e:%e:%e];\n", iLowerB[iplot2],
              (iUpperB[iplot2]-iLowerB[iplot2])*0.01, iUpperB[iplot2]);
      fprintf(fp, "yt = [%e:%e:%e];\n", iLowerB[iplot1],
              (iUpperB[iplot1]-iLowerB[iplot1])*0.01, iUpperB[iplot1]);
      fprintf(fp, "zt = [%e:%e:%e];\n", iLowerB[iplot3],
              (iUpperB[iplot3]-iLowerB[iplot3])*0.01, iUpperB[iplot3]);
      fwritePlotCLF(fp);
      fprintf(fp, "isoval = %e;\n", gamma);
      fprintf(fp, "h = patch(isosurface(X,Y,Z,V,isoval),... \n");
      fprintf(fp, "          'FaceColor', 'blue', ... \n");
      fprintf(fp, "          'EdgeColor', 'none', ... \n");
      fprintf(fp, "          'AmbientStrength', 0.2, ... \n");
      fprintf(fp, "          'SpecularStrength', 0.7, ... \n");
      fprintf(fp, "          'DiffuseStrength', 0.4);\n");
      fprintf(fp, "isonormals(X,Y,Z,V,h);\n");
      fprintf(fp, "patch(isocaps(X,Y,Z,V,isoval), ...\n");
      fprintf(fp, "      'FaceColor', 'interp', ... \n");
      fprintf(fp, "      'EdgeColor', 'none'); \n");
      fprintf(fp, "axis([xlo xhi ylo yhi zlo zhi])\n");
      fprintf(fp, "daspect([xhi-xlo, yhi-ylo, zhi-zlo])\n");
      fprintf(fp, "colormap('default'); colorbar\n");
      fprintf(fp, "%%axis tight\n");
      fprintf(fp, "view(3) \n");
      fprintf(fp, "set(gcf,'Renderer','zbuffer')\n");
      fprintf(fp, "box on\n");
      fprintf(fp, "grid on\n");
      fprintf(fp, "lighting phong\n");
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, inputNames[iplot2]);
      fwritePlotYLabel(fp, inputNames[iplot1]);
      fwritePlotZLabel(fp, inputNames[iplot3]);
      snprintf(winput,100,"Std. Dev. Plot for %s", outputNames[jplot]);
      fwritePlotTitle(fp, winput);
      fprintf(fp,"cin = input('generate slices ? (y or n) ','s');\n");
      fprintf(fp,"if (cin == 'y')\n");
      fprintf(fp,"xin = input('axis to slide through? (x,y,z) ','s');\n");
      fprintf(fp,"N = 101;\n");
      fprintf(fp,"for i = 1 : N\n");
      fprintf(fp,"   display(['displaying ' int2str(i) ' of 101'])\n");
      fprintf(fp,"   if (xin == 'y')\n");
      fprintf(fp,"      h = slice(X,Y,Z,V,xt(i),[],[]);\n");
      fprintf(fp,"   elseif (xin == 'x')\n");
      fprintf(fp,"      h = slice(X,Y,Z,V,[],yt(i),[]);\n");
      fprintf(fp,"   elseif (xin == 'z')\n");
      fprintf(fp,"      h = slice(X,Y,Z,V,[],[],zt(i));\n");
      fprintf(fp,"   end\n");
      fprintf(fp,"   axis([%11.4e %11.4e %11.4e %11.4e %11.4e %11.4e ",
              iLowerB[iplot2], iUpperB[iplot2], iLowerB[iplot1],
              iUpperB[iplot1], iLowerB[iplot3], iUpperB[iplot3]);
      fprintf(fp, "%11.4e %11.4e])\n",
              GYmin-0.1*(GYmax-GYmin),GYmax+0.1*(GYmax-GYmin));
      //**/fprintf(fp,"   if (xin == 'y')\n");
      //**/fprintf(fp,"      h = contourslice(X,Y,Z,V,xt(i),[],[],N);\n");
      //**/fprintf(fp,"   elseif (xin == 'y')\n");
      //**/fprintf(fp,"      h = contourslice(X,Y,Z,V,[],yt(i),[],N);\n");
      //**/fprintf(fp,"   elseif (xin == 'z')\n");
      //**/fprintf(fp,"      h = contourslice(X,Y,Z,V,[],[],zt(i),N);\n");
      //**/fprintf(fp,"   end\n");
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, inputNames[iplot2]);
      fwritePlotYLabel(fp, inputNames[iplot1]);
      fwritePlotZLabel(fp, inputNames[iplot3]);
      snprintf(winput,100,"Std. Dev. Slice Plot for %s", outputNames[jplot]);
      fwritePlotTitle(fp, winput);
      fprintf(fp, "   view(3)\n");
      fprintf(fp, "   colorbar\n");
      fprintf(fp, "   pause(1)\n");
      fprintf(fp, "   if (i < 101)\n");
      fprintf(fp, "      clf\n");
      fprintf(fp, "   end\n");
      fprintf(fp, "end\n");
      fprintf(fp, "end\n");
    }
    else if (count == 4)
    {
      fprintf(fp,"xlo = %e; \n", iLowerB[iplot2]);
      fprintf(fp,"xhi = %e; \n", iUpperB[iplot2]);
      fprintf(fp,"ylo = %e; \n", iLowerB[iplot1]);
      fprintf(fp,"yhi = %e; \n", iUpperB[iplot1]);
      fprintf(fp,"zlo = %e; \n", iLowerB[iplot3]);
      fprintf(fp,"zhi = %e; \n", iUpperB[iplot3]);
      fprintf(fp,"X=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
      fprintf(fp,"Y=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
      fprintf(fp,"Z=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
      fprintf(fp,"V=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
      for (ii = 0; ii < nPtsPerDim; ii++)
      {
        //**/ x and y data only needed to output once
        fprintf(fp, "Y(:,:,%d) = [\n", ii + 1);
        for (sInd = 0; sInd < nPtsPerDim; sInd++)
        {
          for (jj = 0; jj < nPtsPerDim; jj++)
          {
            ind = (sInd*nPtsPerDim*nPtsPerDim+jj*nPtsPerDim+ii)*
                  nPtsPerDim;
            fprintf(fp, "%e ", faXOut[ind*4]);
          }
          fprintf(fp, "\n");
        }
        fprintf(fp, "];\n");
        fprintf(fp, "X(:,:,%d) = [\n", ii + 1);
        for (sInd = 0; sInd < nPtsPerDim; sInd++)
        {
          for (jj = 0; jj < nPtsPerDim; jj++)
          {
            ind = (sInd*nPtsPerDim*nPtsPerDim+jj*nPtsPerDim+ii)*
                  nPtsPerDim;
            fprintf(fp, "%e ", faXOut[ind*4+1]);
          }
          fprintf(fp, "\n");
        }
        fprintf(fp, "];\n");
        fprintf(fp, "Z(:,:,%d) = [\n", ii + 1);
        for (sInd = 0; sInd < nPtsPerDim; sInd++)
        {
          for (jj = 0; jj < nPtsPerDim; jj++)
          {
            ind = (sInd*nPtsPerDim*nPtsPerDim+jj*nPtsPerDim+ii)*
                  nPtsPerDim;
            fprintf(fp, "%e ", faXOut[ind*4+2]);
          }
          fprintf(fp, "\n");
        }
        fprintf(fp, "];\n");
      }
      fprintf(fp, "xt = [%e:%e:%e];\n", iLowerB[iplot2],
              (iUpperB[iplot2]-iLowerB[iplot2])*0.05, iUpperB[iplot2]);
      fprintf(fp, "yt = [%e:%e:%e];\n", iLowerB[iplot1],
              (iUpperB[iplot1]-iLowerB[iplot1])*0.05, iUpperB[iplot1]);
      fprintf(fp, "zt = [%e:%e:%e];\n", iLowerB[iplot3],
              (iUpperB[iplot3]-iLowerB[iplot3])*0.05, iUpperB[iplot3]);
      for (ll = 0; ll < nPtsPerDim; ll++)
      {
        for (ii = 0; ii < nPtsPerDim; ii++)
        {
          //**/ x and y data only needed to output once
          fprintf(fp, "V(:,:,%d) = [\n", ii + 1);
          for (sInd = 0; sInd < nPtsPerDim; sInd++)
          {
            for (jj = 0; jj < nPtsPerDim; jj++)
            {
              ind=((sInd*nPtsPerDim+jj)*nPtsPerDim+ii)*nPtsPerDim+ll;
              fprintf(fp, "%e ", vecWT[ind]);
            }
            fprintf(fp, "\n");
          }
          fprintf(fp, "];\n");
        }
        fprintf(fp, "disp('Frame %d of %d')\n", ll+1, nPtsPerDim);
        fwritePlotCLF(fp);
        fprintf(fp, "isoval = %e;\n", gamma);
        fprintf(fp, "h = patch(isosurface(X,Y,Z,V,isoval),... \n");
        fprintf(fp, "          'FaceColor', 'blue', ... \n");
        fprintf(fp, "          'EdgeColor', 'none', ... \n");
        fprintf(fp, "          'AmbientStrength', 0.2, ... \n");
        fprintf(fp, "          'SpecularStrength', 0.7, ... \n");
        fprintf(fp, "          'DiffuseStrength', 0.4);\n");
        fprintf(fp, "isonormals(X,Y,Z,V,h);\n");
        fprintf(fp, "patch(isocaps(X,Y,Z,V,isoval), ...\n");
        fprintf(fp, "      'FaceColor', 'interp', ... \n");
        fprintf(fp, "      'EdgeColor', 'none'); \n");
        fprintf(fp, "axis([xlo xhi ylo yhi zlo zhi])\n");
        fprintf(fp, "daspect([xhi-xlo, yhi-ylo, zhi-zlo])\n");
        fprintf(fp, "colormap('default'); colorbar\n");
        fprintf(fp, "%%axis tight\n");
        fprintf(fp, "view(3) \n");
        fprintf(fp, "set(gcf,'Renderer','zbuffer')\n");
        fprintf(fp, "lighting phong\n");
        fwritePlotAxes(fp);
        fwritePlotXLabel(fp, inputNames[iplot2]);
        fwritePlotYLabel(fp, inputNames[iplot1]);
        fwritePlotZLabel(fp, inputNames[iplot3]);
        fprintf(fp, "title('3D Std Dev Isosurface Plot at %s=%e',",
                inputNames[iplot4],faXOut[ll*4+3]);
        fprintf(fp, "'FontWeight','bold','FontSize',12)\n");
        fprintf(fp, "pause(1)\n");
      }
    }
    fclose(fp);
    if (plotScilab())
         printf("rsplot_sd plot file = scilabrsplot_sd.sci\n");
    else printf("rsplot_sd plot file = matlabrsplot_sd.m\n");
    delete [] faXOut;
    delete [] faYOut;
    delete faPtr;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsi2 or rsplot2_int
  //**/ generate intersection surfaces for multiple outputs for 
  //**/ display with matlab
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsi2") ||
           !strcmp(command, "rsplot2_int"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsplot2_int: create intersection surfaces for >1 outputs.\n");
      printf("syntax: rsplot2_int (no argument needed).\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet..\n");
      return 1;
    }
    if (nInputs < 2)
    {
      printf("ERROR: rsplot2_int requires 2 or more inputs.\n");
      return 1;
    }
    if (nOutputs < 2)
    {
      printf("ERROR: rsplot2_int requires 2 or more outputs.\n");
      return 1;
    }
    if (plotScilab())
    {
      printf("INFO: rsplot2_int is currently not available for scilab.\n");
      return 1;
    }

    printAsterisks(PL_INFO, 0);
    printf("This command first creates 2 or more response surfaces for 2 ");
    printf("selected\n");
    printf("inputs. Regions in each RS surface falling inside some ");
    printf("user-specified\n");
    printf("interval are carved out, and the degree of overlap ");
    printf("(intersection)\n");
    printf("between them will be displayed with different colors ");
    printf("(blank for no\n");
    printf("overlap).\n");
    printf("If there are more than 2 inputs, the other inputs are set at ");
    printf("their\n");
    printf("midpoints or are user-specified.\n");
    printf("You will be asked to select a response surface (RS) type.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);

    //**/ set up the function approximator
    int nPtsPerDim = 32;
    snprintf(pString,100,"Grid resolution ? (32 - 256) ");
    nPtsPerDim = getInt(32, 256, pString);
    int faFlag = 1;
    FuncApprox *faPtr = genFAInteractive(psuadeIO, faFlag);
    if (faPtr == NULL) {printf("ERROR detected.\n"); return 1;}
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);

    //**/ ask users to specify the 2 inputs 
    int iplot1, iplot2, iInd1;
    psVector vecInpSettings;
    vecInpSettings.setLength(nInputs);
    iplot1 = iplot2 = -1;
    snprintf(pString,100,"Enter the input for x axis (1 - %d) : ", nInputs);
    iplot1 = getInt(1, nInputs, pString);
    iplot1--;
    iplot2 = iplot1;
    while (iplot1 == iplot2)
    {
      snprintf(pString,100,"Enter the input for y axis (1 - %d), not %d : ",
             nInputs, iplot1+1);
      iplot2 = getInt(1, nInputs, pString);
      iplot2--;
      if (iplot1 == iplot2)
        printf("ERROR: Duplicate input number %d.\n", iplot2+1);
    }
    snprintf(pString,100,"Set other inputs at their mid points ? (y or n) ");
    getString(pString, winput);
    if (winput[0] == 'y')
    {
      for (iInd1 = 0; iInd1 < nInputs; iInd1++)
      {
        if (iInd1 != iplot1 && iInd1 != iplot2)
             vecInpSettings[iInd1] = 0.5*(iLowerB[iInd1]+iUpperB[iInd1]);
        else vecInpSettings[iInd1] = 1.0;
      }
    }
    else
    {
      for (iInd1 = 0; iInd1 < nInputs; iInd1++)
      {
        if (iInd1 != iplot1 && iInd1 != iplot2)
        {
          vecInpSettings[iInd1] = iLowerB[iInd1] - 1.0;
          while (vecInpSettings[iInd1] < iLowerB[iInd1] ||
                 vecInpSettings[iInd1] > iUpperB[iInd1])
          {
            snprintf(pString,100,
                    "Enter nominal value for input %d (%e - %e):",
                    iInd1+1, iLowerB[iInd1], iUpperB[iInd1]);
            vecInpSettings[iInd1] = getDouble(pString);
          }
        }
        else vecInpSettings[iInd1] = 1.0;
      }
    }
    fp = fopen("matlabrsplot2_int.m", "w");
    if (fp == NULL)
    {
      printf("ERROR: Cannot open file matlabrsplot2_int.m.\n");
      delete faPtr;
      return 1;
    }
    fprintf(fp, "twoPlots = 1;\n");
    fprintf(fp, "fs = 10;\n");
    fwritePlotCLF(fp);
    
    //**/ ask users to specify the output set
    int rsiNOutputs = 2;
    snprintf(pString,100,"How many outputs to use ? (2 - %d) ",nOutputs);
    rsiNOutputs = getInt(2, nOutputs, pString);

    //**/ get the collection of output set
    psIVector vecRsiSet;
    vecRsiSet.setLength(rsiNOutputs);
    if (rsiNOutputs == nOutputs)
    {
      for (ii = 0; ii < rsiNOutputs; ii++) vecRsiSet[ii] = ii;
    }
    else
    {
      for (ii = 0; ii < rsiNOutputs; ii++)
      {
        snprintf(pString,100,"Enter the %d-th output index (1 - %d) : ",
                ii+1, nOutputs);
        vecRsiSet[ii] = getInt(1, nOutputs, pString);
        vecRsiSet[ii]--;
      }
    }
    if (rsiNOutputs > 5)
    {
      printf("INFO: rsplot2_int only shows the constrained RS for the\n");
      printf("      first 5 outputs and then the aggregate.\n");
    }
     
    psVector vecFaYIn;
    vecFaYIn.setLength(nSamples);
    int **rsiMatrix = new int*[nPtsPerDim];
    for (ii = 0; ii < nPtsPerDim; ii++)
    {
      rsiMatrix[ii] = new int[nPtsPerDim];
      for (jj = 0; jj < nPtsPerDim; jj++)
        rsiMatrix[ii][jj] = rsiNOutputs;
    }

    //**/ interpolate
    int    jplot, ind, ind2, sInd, faLeng=0, count;
    double Ymin, Ymax, threshU, threshL, *faXOut=NULL, *faYOut=NULL;
    for (ii = 0; ii < rsiNOutputs; ii++)
    {
       jplot = vecRsiSet[ii];
       for (sInd = 0; sInd < nSamples; sInd++)
         vecFaYIn[sInd] = sampleOutputs[sInd*nOutputs+jplot];

       faPtr->gen2DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,iplot2, 
                 vecInpSettings.getDVector(), &faLeng, &faXOut,&faYOut);

       Ymin = faYOut[0];
       for (sInd = 1; sInd < faLeng; sInd++)
         if (faYOut[sInd] < Ymin) Ymin = faYOut[sInd];
       Ymax = faYOut[0];
       for (sInd = 1; sInd < faLeng; sInd++)
         if (faYOut[sInd] > Ymax) Ymax = faYOut[sInd];

       printf("Ymin and Ymax = %e %e\n", Ymin, Ymax);
       snprintf(pString,100,
               "Enter the lower threshold for output %d (min = %16.8e) : ",
               jplot+1, Ymin);
       threshL = getDouble(pString);
       snprintf(pString,100,
               "Enter the upper threshold for output %d (max = %16.8e) : ",
               jplot+1, Ymax);
       threshU = getDouble(pString);

       if (ii == 0)
       {
         fprintf(fp, "x = [\n");
         for (sInd = 0; sInd < faLeng; sInd+=nPtsPerDim)
           fprintf(fp, "%e\n", faXOut[sInd*2]);
         fprintf(fp, "];\n");
         fprintf(fp, "y = [\n");
         for (sInd = 0; sInd < nPtsPerDim; sInd++)
           fprintf(fp, "%e\n", faXOut[sInd*2+1]);
         fprintf(fp, "];\n");
       }
       if (ii < 5)
       {
         fprintf(fp, "A%d = [\n", ii+1);
         for (sInd = 0; sInd < faLeng; sInd++)
           fprintf(fp, "%e\n", faYOut[sInd]);
         fprintf(fp, "];\n");
         fprintf(fp, "A%d = reshape(A%d,%d,%d);\n",ii+1,ii+1,
                 nPtsPerDim,nPtsPerDim);
         fprintf(fp, "yLo = %e;\n", threshL);
         fprintf(fp, "yHi = %e;\n", threshU);
         fprintf(fp, "nA  = size(A%d,1);\n", ii+1);
         fprintf(fp, "[ia,ja,aa] = find(A%d<yLo);\n", ii+1);
         fprintf(fp, "for ii = 1 : length(ia)\n");
         fprintf(fp, "   A%d(ia(ii),ja(ii)) = NaN;\n", ii+1); 
         fprintf(fp, "end;\n");
         fprintf(fp, "n1 = length(ia);\n");
         fprintf(fp, "[ia,ja,aa] = find(A%d>yHi);\n", ii+1);
         fprintf(fp, "for ii = 1 : length(ia)\n");
         fprintf(fp, "   A%d(ia(ii),ja(ii)) = NaN;\n", ii+1); 
         fprintf(fp, "end;\n");
         fprintf(fp, "n2 = length(ia);\n");
         fprintf(fp, "if (n1 + n2 == nA*nA)\n");
         fprintf(fp, "   A%d(1,1) = 0;\n",ii+1);
         fprintf(fp, "   A%d(%d,%d) = 1;\n",ii+1,nPtsPerDim,
                 nPtsPerDim);
         fprintf(fp, "end;\n");
         if (ii == 0) fprintf(fp, "subplot(2,3,1)\n");
         if (ii == 1) fprintf(fp, "subplot(2,3,2)\n");
         if (ii == 2) fprintf(fp, "subplot(2,3,3)\n");
         if (ii == 3) fprintf(fp, "subplot(2,3,4)\n");
         if (ii == 4) fprintf(fp, "subplot(2,3,5)\n");
         fprintf(fp, "contourf(x,y,A%d)\n", ii+1);
         fprintf(fp, "axis([%e %e %e %e])\n",iLowerB[iplot1],
                 iUpperB[iplot1],iLowerB[iplot2],iUpperB[iplot2]);
         fwritePlotAxes(fp);
         fprintf(fp, "xlabel('%s','FontSize',fs,'FontWeight','bold')\n",
                 inputNames[iplot1]);
         fprintf(fp, "ylabel('%s','Fontsize',fs,'FontWeight','bold')\n",
                 inputNames[iplot2]);
         fprintf(fp, "title('%s',",outputNames[jplot]);
         fprintf(fp, "'FontWeight','bold','FontSize',fs)\n");
         fprintf(fp, "colorbar\n");
       }

       for (sInd = 0; sInd < faLeng; sInd++)
       {
          ind  = sInd % nPtsPerDim;
          ind2 = sInd / nPtsPerDim;
          if (faYOut[sInd] < threshL) rsiMatrix[ind][ind2]--;
          if (faYOut[sInd] > threshU) rsiMatrix[ind][ind2]--;
       }
       delete [] faXOut;
       delete [] faYOut;
    }

    //**/ write data to a matlab file

    fprintf(fp, "A = [\n");
    count = 0;
    for (ii = 0;  ii < nPtsPerDim; ii++)
      for (jj = 0;  jj < nPtsPerDim; jj++)
        if (rsiMatrix[jj][ii] == 0) count++;
    if (count == nPtsPerDim*nPtsPerDim)
    {
      for (ii = 0;  ii < nPtsPerDim; ii++)
        for (jj = 0;  jj < nPtsPerDim; jj++) fprintf(fp, "0\n");
    }
    else
    {
      for (ii = 0;  ii < nPtsPerDim; ii++)
      {
        for (jj = 0;  jj < nPtsPerDim; jj++)
        {
          if (rsiMatrix[jj][ii] == 0) fprintf(fp, "NaN\n");
          else fprintf(fp, "%d\n", rsiMatrix[jj][ii]);
        }
      }
    }
    fprintf(fp, "];\n");
    fprintf(fp, "A = reshape(A,%d,%d);\n",nPtsPerDim, nPtsPerDim);
    fprintf(fp, "A(%d,%d) = %e;\n", nPtsPerDim, nPtsPerDim, 
            (double) rsiNOutputs);
    //**/ delete (9/2016)
    //fprintf(fp, "if twoPlots == 1\n");
    //fprintf(fp, "subplot(2,3,5), mesh(y,x,A)\n");
    //fprintf(fp, "axis([%e %e %e %e])\n",iLowerB[iplot1],
    //        iUpperB[iplot1],iLowerB[iplot2],iUpperB[iplot2]);
    //fwritePlotAxes(fp);
    //fprintf(fp, "xlabel('%s','FontSize',12,'FontWeight','bold')\n",
    //        inputNames[iplot1]);
    //fprintf(fp, "ylabel('%s','Fontsize',12,'FontWeight','bold')\n",
    //        inputNames[iplot2]);
    //fprintf(fp, "title('Intersection Plot','FontWeight',");
    //fprintf(fp, "'bold','FontSize',12)\n");
    //fprintf(fp, "colorbar\n");
    //fprintf(fp, "colormap(cool)\n");
    //fprintf(fp, "end\n");
    fprintf(fp,"subplot(2,3,6), contourf(x,y,A)\n");
    fprintf(fp,"axis([%e %e %e %e])\n",iLowerB[iplot1],
            iUpperB[iplot1],iLowerB[iplot2],iUpperB[iplot2]);
    fwritePlotAxes(fp);
    fprintf(fp,"xlabel('%s','FontSize',fs,'FontWeight','bold')\n",
            inputNames[iplot1]);
    fprintf(fp,"ylabel('%s','Fontsize',fs,'FontWeight','bold')\n",
            inputNames[iplot2]);
    fprintf(fp,"title('Intersection (color=deg of overlap)','FontWeight',");
    fprintf(fp,"'bold','FontSize',fs)\n");
    fprintf(fp,"colorbar\n");
    fprintf(fp,"colormap(cool)\n");
    fprintf(fp,"disp('On intersection plot, if a region has a color value");
    fprintf(fp," of 2, it means it is feasible for 2 outputs.')\n");
    fclose(fp);
    printf("rsplot2_int plot file = matlabrsplot2_int.m\n");

    delete faPtr;
    for (ii = 0; ii < nPtsPerDim; ii++) delete [] rsiMatrix[ii];
      delete [] rsiMatrix;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsi3 or rsplot3_int 
  //**/ generate 3D response surface and write the grid data to file
  //**/ for display with matlab
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsi3") ||
           !strcmp(command, "rsplot3_int"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsplot3_int: create intersection surfaces for >1 outputs\n");
      printf("syntax: rsplot3_int (no argument needed).\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nInputs < 3)
    {
      printf("ERROR: rsplot3_int requires 3 or more inputs.\n");
      return 1;
    }
    if (nOutputs < 2)
    {
      printf("ERROR: rsplot3_int requires 2 or more outputs.\n");
      return 1;
    }
    if (plotScilab())
    {
      printf("INFO: rsplot3_int is currently not available for scilab.\n");
      return 1;
    }

    printAsterisks(PL_INFO, 0);
    printf("This command first creates 2 or more response surfaces for 3 ");
    printf("selected\n");
    printf("inputs. Regions in each RS surface falling inside some ");
    printf("user-specified\n");
    printf("interval are carved out, and the degree of overlap ");
    printf("(in input space)\n");
    printf("between them will be displayed with different colors ");
    printf("(blank for no\n");
    printf("overlap). The 3 selected inputs are in the X, Y, Z axes.\n");
    printf("If there are more than 3 inputs, the other inputs are set at ");
    printf("their\n");
    printf("midpoints or are user-specified.\n");
    printf("You will be asked to select a response surface (RS) type.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);

    //**/ set up the function approximator
    int nPtsPerDim = 24;
    snprintf(pString,100,"Grid resolution ? (16 - 32) ");
    nPtsPerDim = getInt(16, 32, pString);
    int faFlag = 1;
    FuncApprox *faPtr = genFAInteractive(psuadeIO, faFlag);
    if (faPtr == NULL) {printf("ERROR detected.\n"); return 1;}
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);

    //**/ ask users to specify the three inputs and one output
    int iplot1, iplot2, iplot3, iInd1, sInd;
    psVector vecInpSettings;
    vecInpSettings.setLength(nInputs);
    iplot1 = iplot2 = iplot3 = -1;
    snprintf(pString,100,"Enter the input for x axis (1 - %d) : ", nInputs);
    iplot1 = getInt(1, nInputs, pString);
    iplot1--;
    iplot2 = iplot1;
    while (iplot1 == iplot2)
    {
      snprintf(pString,100,"Enter the input for y axis (1 - %d), not %d : ",
              nInputs, iplot1+1);
      iplot2 = getInt(1, nInputs, pString);
      iplot2--;
      if (iplot1 == iplot2)
        printf("ERROR: Duplicate input number %d.\n",iplot2+1);
    }
    if (nInputs == 3) iplot3 = 3 - iplot1 - iplot2;
    while (iplot3 < 0 || iplot3 == iplot1 || iplot3 == iplot2)
    {
      snprintf(pString,100,
              "Enter the input for z axis (1 - %d), not %d nor %d: ",
              nInputs, iplot1+1, iplot2+1);
      iplot3 = getInt(1, nInputs, pString);
      iplot3--;
      if (iplot3 == iplot1 || iplot3 == iplot2)
        printf("ERROR: Duplicate input number %d.\n",iplot3+1);
    }
    snprintf(pString,100,"Set other inputs at their mid points ? (y or n) ");
    getString(pString, winput);
    if (winput[0] == 'y')
    {
      for (iInd1 = 0; iInd1 < nInputs; iInd1++)
      {
        if (iInd1 != iplot1 && iInd1 != iplot2 && iInd1 != iplot3)
             vecInpSettings[iInd1] = 0.5*(iLowerB[iInd1]+iUpperB[iInd1]);
        else vecInpSettings[iInd1] = 1.0;
      }
    }
    else
    {
      for (iInd1 = 0; iInd1 < nInputs; iInd1++)
      {
        if (iInd1 != iplot1 && iInd1 != iplot2 && iInd1 != iplot3)
        {
          vecInpSettings[iInd1] = iLowerB[iInd1] - 1.0;
          snprintf(pString,100,
                  "Enter nominal value for input %d (%e - %e): ", 
                  iInd1+1, iLowerB[iInd1], iUpperB[iInd1]);
          while (vecInpSettings[iInd1] < iLowerB[iInd1] ||
                 vecInpSettings[iInd1] > iUpperB[iInd1])
            vecInpSettings[iInd1] = getDouble(pString);
        }
        else vecInpSettings[iInd1] = 1.0;
      }
    }
    fp = fopen("matlabrsplot3_int.m", "w");
    if (fp == NULL)
    {
      printf("ERROR: Cannot open file matlabrsplot3_int.m.\n");
      delete faPtr;
    }
    fwritePlotCLF(fp);

    //**/ ask users to specify the output set
    int rsiNOutputs = 1;
    snprintf(pString,100,"How many outputs to use ? (2 - %d) ",nOutputs);
    rsiNOutputs = getInt(2, nOutputs, pString);

    psVector vecThreshLs, vecThreshUs;
    vecThreshLs.setLength(rsiNOutputs);
    vecThreshUs.setLength(rsiNOutputs);

    //**/ get the collection of output set
    int *rsiSet = new int[rsiNOutputs];
    if (rsiNOutputs == nOutputs)
    {
      for (ii = 0; ii < rsiNOutputs; ii++) rsiSet[ii] = ii;
    }
    else
    {
      for (ii = 0; ii < rsiNOutputs; ii++)
      {
        snprintf(pString,100,"Enter the %d-th output index (1 - %d) : ",
                ii+1, nOutputs);
        rsiSet[ii] = getInt(1, nOutputs, pString);
        rsiSet[ii]--;
      }
    }
    psVector vecFaYIn;
    vecFaYIn.setLength(nSamples);

    //**/ generate and write response surface data
    printf("Please wait while generating the RS data \n");
    fprintf(fp, "xlo = %e; \n", iLowerB[iplot2]);
    fprintf(fp, "xhi = %e; \n", iUpperB[iplot2]);
    fprintf(fp, "ylo = %e; \n", iLowerB[iplot1]);
    fprintf(fp, "yhi = %e; \n", iUpperB[iplot1]);
    fprintf(fp, "zlo = %e; \n", iLowerB[iplot3]);
    fprintf(fp, "zhi = %e; \n", iUpperB[iplot3]);
    fprintf(fp, "X=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
    fprintf(fp, "Y=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
    fprintf(fp, "Z=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);
    fprintf(fp, "V=zeros(%d,%d,%d);\n",nPtsPerDim,nPtsPerDim,nPtsPerDim);

    //**/ get max and min and write X, Y, Z coordinates
    int    faLeng, jplot, ind, count;
    double GYmax, GYmin, gamma, *faXOut=NULL, *faYOut=NULL;
    for (ii = 0; ii < rsiNOutputs; ii++)
    {
      jplot = rsiSet[ii];
      for (sInd = 0; sInd < nSamples; sInd++)
         vecFaYIn[sInd] = sampleOutputs[sInd*nOutputs+jplot];

      faLeng = 0;
      faPtr->gen3DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,iplot2,
                iplot3,vecInpSettings.getDVector(),&faLeng,&faXOut,&faYOut);
      GYmin = faYOut[0];
      for (sInd = 1; sInd < faLeng; sInd++)
        if (faYOut[sInd] < GYmin) GYmin = faYOut[sInd];
      GYmax = faYOut[0];
      for (sInd = 1; sInd < faLeng; sInd++)
        if (faYOut[sInd] > GYmax) GYmax = faYOut[sInd];

      printf("\nOutput %d : Ymin and Ymax found = %e %e.\n", jplot+1,
             GYmin, GYmax);
      //**/vecThreshL = GYmin - 0.2 * PABS(GYmin);
      snprintf(pString,100,"Enter the lower threshold (min = %e) : ", GYmin);
      vecThreshLs[ii] = getDouble(pString);
      //**/vecThreshU = GYmax + 0.2 * PABS(GYmax);
      snprintf(pString,100,"Enter the upper threshold (max = %e) : ", GYmax);
      vecThreshUs[ii] = getDouble(pString);
      if (ii == 0) gamma = vecThreshLs[ii];  
      else         gamma = (gamma<vecThreshLs[ii]) ? gamma:vecThreshLs[ii];

      if (ii == 0)
      {
        for (jj = 0; jj < nPtsPerDim; jj++)
        {
          fprintf(fp, "Y(:,:,%d) = [\n", jj + 1);
          for (sInd = 0; sInd < nPtsPerDim; sInd++)
          {
            for (kk = 0; kk < nPtsPerDim; kk++)
            {
              ind = sInd*nPtsPerDim*nPtsPerDim+kk*nPtsPerDim+jj;
              fprintf(fp, "%e ", faXOut[ind*3]);
            }
            fprintf(fp, "\n");
          }
          fprintf(fp, "];\n");
          fprintf(fp, "X(:,:,%d) = [\n", jj + 1);
          for (sInd = 0; sInd < nPtsPerDim; sInd++)
          {
            for (kk = 0; kk < nPtsPerDim; kk++)
            {
              ind = sInd*nPtsPerDim*nPtsPerDim+kk*nPtsPerDim+jj;
              fprintf(fp, "%e ", faXOut[ind*3+1]);
            }
            fprintf(fp, "\n");
          }
          fprintf(fp, "];\n");
          fprintf(fp, "Z(:,:,%d) = [\n", jj + 1);
          for (sInd = 0; sInd < nPtsPerDim; sInd++)
          {
            for (kk = 0; kk < nPtsPerDim; kk++)
            {
              ind = sInd*nPtsPerDim*nPtsPerDim+kk*nPtsPerDim+jj;
              fprintf(fp, "%e ", faXOut[ind*3+2]);
            }
            fprintf(fp, "\n");
          }
          fprintf(fp, "];\n");
        }
      }
      delete [] faXOut;
      delete [] faYOut;
    }

    //**/ invalidate cells not inside feasible region
    for (ii = 0; ii < rsiNOutputs; ii++)
    {
      jplot = rsiSet[ii];
      for (sInd = 0; sInd < nSamples; sInd++)
        vecFaYIn[sInd] = sampleOutputs[sInd*nOutputs+jplot];

      faLeng = 0;
      faPtr->gen3DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,iplot2, 
               iplot3,vecInpSettings.getDVector(),&faLeng,&faXOut,&faYOut);
      for (jj = 0; jj < nPtsPerDim; jj++)
      {
        fprintf(fp, "V%d(:,:,%d) = [\n", ii+1, jj+1);
        for (sInd = 0; sInd < nPtsPerDim; sInd++)
        {
          for (kk = 0; kk < nPtsPerDim; kk++)
          {
            ind = sInd*nPtsPerDim*nPtsPerDim+kk*nPtsPerDim+jj;
            if (faYOut[ind] < vecThreshLs[ii])
            {
              fprintf(fp, "%e ", gamma);
              count++;
            }
            else if (faYOut[ind] > vecThreshUs[ii])
            {
              fprintf(fp, "%e ", gamma);
              count++;
            }
            else fprintf(fp, "%e ", faYOut[ind]);
          }
          fprintf(fp, "\n");
        }
        fprintf(fp, "];\n");
      }
      delete [] faXOut;
      delete [] faYOut;
      if (ii == 0) fprintf(fp, "V = V%d;\n", ii+1);
      else         fprintf(fp, "V = min(V, V%d);\n", ii+1);
    }

    //**/ prepare matlab script
    double threshL, threshU;
    threshL = vecThreshLs[0];
    for (ii = 1; ii < rsiNOutputs; ii++)
      if (vecThreshLs[ii] < threshL) threshL = vecThreshLs[ii];
    threshU = vecThreshUs[0];
    for (ii = 1; ii < rsiNOutputs; ii++)
      if (vecThreshUs[ii] > threshU) threshU = vecThreshUs[ii];
    fprintf(fp, "xt = [%e:%e:%e];\n", iLowerB[iplot2],
            (iUpperB[iplot2]-iLowerB[iplot2])*0.01, iUpperB[iplot2]);
    fprintf(fp, "yt = [%e:%e:%e];\n", iLowerB[iplot1],
            (iUpperB[iplot1]-iLowerB[iplot1])*0.01, iUpperB[iplot1]);
    fprintf(fp, "zt = [%e:%e:%e];\n", iLowerB[iplot3],
            (iUpperB[iplot3]-iLowerB[iplot3])*0.01, iUpperB[iplot3]);
    fprintf(fp, "isoval = %e;\n", gamma);
    fprintf(fp, "h = patch(isosurface(X,Y,Z,V,isoval),... \n");
    fprintf(fp, "          'FaceColor', 'blue', ... \n");
    fprintf(fp, "          'EdgeColor', 'none', ... \n");
    fprintf(fp, "          'AmbientStrength', 0.2, ... \n");
    fprintf(fp, "          'SpecularStrength', 0.7, ... \n");
    fprintf(fp, "          'DiffuseStrength', 0.4);\n");
    fprintf(fp, "isonormals(X,Y,Z,V,h);\n");
    fprintf(fp, "patch(isocaps(X,Y,Z,V,isoval), ...\n");
    fprintf(fp, "      'FaceColor', 'interp', ... \n");
    fprintf(fp, "      'EdgeColor', 'none'); \n");
    fprintf(fp, "axis([xlo xhi ylo yhi zlo zhi])\n");
    fprintf(fp, "daspect([%e,%e,%e])\n",iUpperB[iplot2]-iLowerB[iplot2],
            iUpperB[iplot1]-iLowerB[iplot1],
            iUpperB[iplot3]-iLowerB[iplot3]);
    fprintf(fp, "   xlabel('%s','FontSize',12,'FontWeight','bold')\n",
            inputNames[iplot2]);
    fprintf(fp, "   ylabel('%s','Fontsize',12,'FontWeight','bold')\n",
            inputNames[iplot1]);
    fprintf(fp, "   zlabel('%s','Fontsize',12,'FontWeight','bold')\n",
            inputNames[iplot3]);
    fwritePlotAxes(fp);
    fprintf(fp, "%%colormap('default'); colorbar\n");
    fprintf(fp, "%%axis tight\n");
    fprintf(fp, "view(3) \n");
    fprintf(fp, "set(gcf,'Renderer','zbuffer')\n");
    fprintf(fp, "lighting phong\n");
    fprintf(fp, "cin = input('generate slices ? (y or n) ','s');\n");
    fprintf(fp, "if (cin == 'y')\n");
    fprintf(fp, "xin = input('axis to slide through ? (x,y,z) ','s');\n");
    fprintf(fp, "for i = 1 : 101\n");
    fprintf(fp, "   if (xin == 'y')\n");
    fprintf(fp, "      h = contourslice(X,Y,Z,V,xt(i),[],[],101);\n");
    fprintf(fp, "   elseif (xin == 'x')\n");
    fprintf(fp, "      h = contourslice(X,Y,Z,V,[],yt(i),[],101);\n");
    fprintf(fp, "   elseif (xin == 'z')\n");
    fprintf(fp, "      h = contourslice(X,Y,Z,V,[],[],zt(i),101);\n");
    fprintf(fp, "   end\n");
    fprintf(fp, "   axis([%11.4e %11.4e %11.4e %11.4e %11.4e %11.4e ",
            iLowerB[iplot2], iUpperB[iplot2], iLowerB[iplot1],
            iUpperB[iplot1], iLowerB[iplot3], iUpperB[iplot3]);
    fprintf(fp, "%11.4e %11.4e])\n",
            threshL-0.2*(threshU-threshL),threshU+0.2*(threshU-threshL));
    fwritePlotAxes(fp);
    fprintf(fp, "   xlabel('%s','FontSize',12,'FontWeight','bold')\n",
            inputNames[iplot2]);
    fprintf(fp, "   ylabel('%s','Fontsize',12,'FontWeight','bold')\n",
            inputNames[iplot1]);
    fprintf(fp, "   zlabel('%s','Fontsize',12,'FontWeight','bold')\n",
            inputNames[iplot3]);
    fprintf(fp, "colormap('default'); colorbar\n");
    fprintf(fp, "view(3) \n");
    fprintf(fp, "set(gcf,'Renderer','zbuffer')\n");
    fprintf(fp, "lighting phong\n");
    fprintf(fp, "pause(1)\n");
    fprintf(fp," if (i < 101)\n");
    fprintf(fp,"   clf\n");
    fprintf(fp," end\n");
    fprintf(fp, "end\n");
    fprintf(fp, "end\n");
    fclose(fp);
    printf("rsplot3_int plot file = matlabrsplot3_int.m\n");

    delete [] rsiSet;
    delete faPtr;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsi3m or rsplot3m_int
  //**/ generate 3D response surfaces and find their intersection
  //**/ for display with matlab (movie mode)
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsi3m") ||
           !strcmp(command, "rsplot3m_int"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsplot3m_int: create intersection surfaces for >1 outputs\n");
      printf("syntax: rsplot3m_int (no argument needed).\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nInputs < 3)
    {
      printf("ERROR: rsplot3m_int requires 3 or more inputs.\n");
      return 1;
    }
    if (nOutputs < 2)
    {
      printf("ERROR: rsplot3m_int requires 2 or more outputs.\n");
      return 1;
    }
    if (plotScilab())
    {
      printf("INFO: rsplot3m_int is currently not available for scilab.\n");
      return 1;
    }

    printAsterisks(PL_INFO, 0);
    printf("This command first creates 2 or more response ");
    printf("surfaces for 3 selected\n");
    printf("inputs. Regions in each RS surface falling inside ");
    printf("some user-specified\n");
    printf("interval are carved out, and the degree of ");
    printf("overlap (intersection)\n");
    printf("between them will be displayed with different ");
    printf("colors (blank for no\n");
    printf("overlap). The difference between rsplot3m_int and ");
    printf("rsplot3m is that, ");
    printf("instead of using X,Y,Z axes for the 3 inputs, ");
    printf("rsplot3m_int uses X, Y\n");
    printf("axes for 2 inputs and time axis for a third input ");
    printf("so that it produces\n");
    printf("a movie of 2D intersection plots.\n");
    printf("If there are more than 3 inputs, the other inputs ");
    printf("are set at their\n");
    printf("midpoints or are user-specified.\n");
    printf("You will be asked to select a response surface (RS) type.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);

    //**/ set up the function approximator
    int nPtsPerDim = 24;
    snprintf(pString,100,"Grid resolution ? (16 - 32) ");
    nPtsPerDim = getInt(16, 32, pString);
    int faFlag = 1;
    FuncApprox *faPtr = genFAInteractive(psuadeIO, faFlag);
    if (faPtr == NULL) {printf("ERROR detected.\n"); return 1;}
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);

    //**/ ask users to specify the three inputs and one output
    int iplot1, iplot2, iplot3, iInd1, sInd;
    psVector vecInpSettings;
    vecInpSettings.setLength(nInputs);
    iplot1 = iplot2 = iplot3 = -1;
    snprintf(pString,100,"Enter the input for x axis (1 - %d) : ", nInputs);
    iplot1 = getInt(1, nInputs, pString);
    iplot1--;
    iplot2 = iplot1;
    while (iplot1 == iplot2)
    {
      snprintf(pString,100,"Enter the input for y axis (1 - %d), not %d : ",
              nInputs, iplot1+1);
      iplot2 = getInt(1, nInputs, pString);
      iplot2--;
      if (iplot1 == iplot2)
        printf("ERROR: Duplicate input number %d.\n",iplot2+1);
    }
    if (nInputs == 3) iplot3 = 3 - iplot1 - iplot2;
    while (iplot3 < 0 || iplot3 == iplot1 || iplot3 == iplot2)
    {
      snprintf(pString,100,
              "Enter the input for t axis (1 - %d), not %d nor %d: ",
              nInputs, iplot1+1, iplot2+1);
      iplot3 = getInt(1, nInputs, pString);
      iplot3--;
      if (iplot3 == iplot1 || iplot3 == iplot2)
        printf("ERROR: Duplicate input number %d.\n",iplot3+1);
    }
    snprintf(pString,100,"Set other inputs at their mid points ? (y or n) ");
    getString(pString, winput);
    if (winput[0] == 'y')
    {
      for (iInd1 = 0; iInd1 < nInputs; iInd1++)
      {
        if (iInd1 != iplot1 && iInd1 != iplot2 && iInd1 != iplot3)
             vecInpSettings[iInd1] = 0.5*(iLowerB[iInd1]+iUpperB[iInd1]);
        else vecInpSettings[iInd1] = 1.0;
      }
    }
    else
    {
      for (iInd1 = 0; iInd1 < nInputs; iInd1++)
      {
        if (iInd1 != iplot1 && iInd1 != iplot2 && iInd1 != iplot3)
        {
          vecInpSettings[iInd1] = iLowerB[iInd1] - 1.0;
          snprintf(pString,100,
                  "Enter nominal value for input %d (%e - %e): ", 
                  iInd1+1, iLowerB[iInd1], iUpperB[iInd1]);
          while (vecInpSettings[iInd1] < iLowerB[iInd1] ||
                 vecInpSettings[iInd1] > iUpperB[iInd1])
            vecInpSettings[iInd1] = getDouble(pString);
        }
        else vecInpSettings[iInd1] = 1.0;
      }
    }
    fp = fopen("matlabrsplot3m_int.m", "w");
    if (fp == NULL)
    {
      printf("ERROR: Cannot open file matlabrsplot3m_int.m.\n");
      delete faPtr;
      return 1;
    }
    fprintf(fp, "hold off\n");
    fwritePlotCLF(fp);
    fprintf(fp, "disp(\'Please wait while loading.\')\n");
    fprintf(fp, "pause(1)\n");
     
    //**/ ask users to specify the output set
    int rsiNOutputs = 2;
    snprintf(pString,100,"How many outputs to use ? (2 - %d) ",nOutputs);
    rsiNOutputs = getInt(2, nOutputs, pString);

    //**/ get the collection of output set
    int *rsiSet = new int[rsiNOutputs];
    if (rsiNOutputs == nOutputs)
    {
      for (ii = 0; ii < rsiNOutputs; ii++) rsiSet[ii] = ii;
    }
    else
    {
      for (ii = 0; ii < rsiNOutputs; ii++)
      {
        snprintf(pString,100,"Enter the %d-th output index (1 - %d) : ",
                ii+1, nOutputs);
        rsiSet[ii] = getInt(1, nOutputs, pString);
        rsiSet[ii]--;
      }
    }
    int    jplot, faLeng;
    double *faXOut=NULL, *faYOut=NULL, GYmax, GYmin, threshL, threshU;
    psVector vecFaYIn;
    vecFaYIn.setLength(nSamples);

    //**/ generate and write response surface data
    printf("Please wait while generating the RS data \n");
    for (jj = 0; jj < nPtsPerDim; jj++)
      fprintf(fp, "M%d = %e * ones(%d);\n", jj+1, 1.0*rsiNOutputs, 
              nPtsPerDim);
    for (ii = 0; ii < rsiNOutputs; ii++)
    {
      jplot = rsiSet[ii];
      for (sInd = 0; sInd < nSamples; sInd++)
         vecFaYIn[sInd] = sampleOutputs[sInd*nOutputs+jplot];

      faLeng = 0;
      faPtr->gen3DGridData(sampleInputs,vecFaYIn.getDVector(),iplot1,iplot2, 
               iplot3,vecInpSettings.getDVector(),&faLeng,&faXOut,&faYOut);
      GYmin = faYOut[0];
      for (sInd = 1; sInd < faLeng; sInd++)
        if (faYOut[sInd] < GYmin) GYmin = faYOut[sInd];
      GYmax = faYOut[0];
      for (sInd = 1; sInd < faLeng; sInd++)
        if (faYOut[sInd] > GYmax) GYmax = faYOut[sInd];

      for (jj = 0; jj < nPtsPerDim; jj++)
      {
        printf(".");
        fflush(stdout);

        //**/ output the response surface data
        fprintf(fp, "A%d_%d = [\n", ii+1, jj+1);
        for (sInd = 0; sInd < faLeng; sInd+=nPtsPerDim)
          fprintf(fp, "%e\n", faYOut[sInd+jj]);
        fprintf(fp, "];\n");
        fprintf(fp, "A%d_%d = reshape(A%d_%d,%d,%d);\n", ii+1, jj+1,
                ii+1, jj+1, nPtsPerDim, nPtsPerDim);

        //**/ x and y data only needed to output once
        if (ii == 0 && jj == 0)
        {
          fprintf(fp, "x = [\n");
          for (sInd = 0; sInd < faLeng; sInd+=nPtsPerDim*nPtsPerDim)
             fprintf(fp, "%e\n", faXOut[sInd*3]);
          fprintf(fp, "];\n");
          fprintf(fp, "y = [\n");
          for (sInd = 0; sInd < nPtsPerDim*nPtsPerDim; sInd+=nPtsPerDim)
            fprintf(fp, "%e\n", faXOut[sInd*3+1]);
          fprintf(fp, "];\n");
        }
      }
      delete [] faXOut;
      delete [] faYOut;
      printf("\nOutput %d : Ymin and Ymax found = %e %e.\n", jplot+1,
             GYmin, GYmax);
      //**/threshL = GYmin - 0.2 * PABS(GYmin);
      snprintf(pString,100,"Enter the lower threshold (min = %e) : ", GYmin);
      threshL = getDouble(pString);
      //**/threshU = GYmax + 0.2 * PABS(GYmax);
      snprintf(pString,100,"Enter the upper threshold (max = %e) : ", GYmax);
      threshU = getDouble(pString);

      for (jj = 0; jj < nPtsPerDim; jj++)
      {
        fprintf(fp, "B%d_%d = A%d_%d;\n",ii+1,jj+1,ii+1,jj+1);
        fprintf(fp, "nA  = size(A%d_%d,1);\n", ii+1, jj+1);
        fprintf(fp, "n1 = 0;\n");
        fprintf(fp, "n2 = 0;\n");
        if (threshL > GYmin)
        { 
          fprintf(fp, "yLo = %e;\n", threshL);
          fprintf(fp, "[ia,ja,aa] = find(A%d_%d<yLo);\n",ii+1,jj+1);
          fprintf(fp, "for ii = 1 : length(ia)\n");
          fprintf(fp, "   B%d_%d(ia(ii),ja(ii))=NaN;\n",ii+1,jj+1);
          fprintf(fp, "   M%d(ia(ii),ja(ii))=M%d(ia(ii),ja(ii))-1;\n", 
                  jj+1,jj+1);
          fprintf(fp, "end;\n");
          fprintf(fp, "n1 = length(ia);\n");
        }
        if (threshU < GYmax)
        { 
          fprintf(fp, "yHi = %e;\n", threshU);
          fprintf(fp, "[ia,ja,aa] = find(A%d_%d>yHi);\n",ii+1,jj+1);
          fprintf(fp, "for ii = 1 : length(ia)\n");
          fprintf(fp, "   B%d_%d(ia(ii),ja(ii))=NaN;\n",ii+1,jj+1);
          fprintf(fp, "   M%d(ia(ii),ja(ii))=M%d(ia(ii),ja(ii))-1;\n", 
                  jj+1,jj+1);
          fprintf(fp, "end;\n");
          fprintf(fp, "n1 = length(ia);\n");
        }
        fprintf(fp, "if (n1+n2 == nA*nA)\n");
        fprintf(fp, "   B%d_%d(1,1)=0;\n",ii+1,jj+1);
        fprintf(fp, "   B%d_%d(%d,%d)=1;\n",ii+1,jj+1,
                nPtsPerDim,nPtsPerDim);
        fprintf(fp, "end;\n");
      }
    }
    for (jj  = 0; jj < nPtsPerDim; jj++)
    {
      fprintf(fp, "[ia,ja,aa] = find(M%d == 0);\n", jj+1);
      fprintf(fp, "nM  = size(M%d,1);\n", jj+1);
      fprintf(fp, "for ii = 1 : length(ia)\n");
      fprintf(fp, "   M%d(ia(ii),ja(ii)) = NaN;\n", jj+1);
      fprintf(fp, "end;\n");
      fprintf(fp, "if (length(ia) == nM*nM)\n");
      fprintf(fp, "   M%d(1,1) = 0;\n", jj+1);
      fprintf(fp, "   M%d(nM,nM) = %e;\n", jj+1, 1.0*rsiNOutputs);
      fprintf(fp, "end;\n");
      fprintf(fp, "Mmax = max(max(M%d));\n", jj+1);
      fprintf(fp, "if (Mmax ~= %d)\n", rsiNOutputs);
      fprintf(fp, "   M%d(%d,%d) = %d;\n", jj+1, nPtsPerDim,
              nPtsPerDim, rsiNOutputs);
      fprintf(fp, "end;\n");
      fprintf(fp, "Mmin = min(min(M%d));\n", jj+1);
      fprintf(fp, "if (Mmin ~= 0)\n");
      fprintf(fp, "   M%d(1,1) = 0;\n", jj+1);
      fprintf(fp, "end;\n");
    }

    //**/ create matlab movie
    for (jj = 0; jj < nPtsPerDim; jj++)
    {
      vecInpSettings[iplot3] = (iUpperB[iplot3] - iLowerB[iplot3]) *
                              jj / (nPtsPerDim - 1.0) + iLowerB[iplot3];
      fprintf(fp, "disp(\'Plotting frame %d of %d\')\n",jj+1,nPtsPerDim);
      fprintf(fp, "subplot(2,3,1), contourf(x,y,B1_%d)\n", jj+1);
      fwritePlotAxes(fp);
      fprintf(fp,"title(\'Contour Plot for %s\',",outputNames[rsiSet[0]]);
      fprintf(fp, "'FontSize',12,'FontWeight','bold')\n"); 
      fprintf(fp, "xlabel('%s','FontSize',12,'FontWeight','bold')\n",
              inputNames[iplot1]);
      fprintf(fp, "ylabel('%s','FontSize',12,'FontWeight','bold');\n",
              inputNames[iplot2]);
      fprintf(fp, "subplot(2,3,2), contourf(x,y,B2_%d)\n", jj+1);
      fwritePlotAxes(fp);
      fprintf(fp,"title(\'Contour Plot for %s\',",outputNames[rsiSet[1]]);
      fprintf(fp, "'FontSize',12,'FontWeight','bold')\n"); 
      fprintf(fp, "xlabel('%s','FontSize',12,'FontWeight','bold')\n",
              inputNames[iplot1]);
      fprintf(fp, "ylabel('%s','FontSize',12,'FontWeight','bold');\n",
              inputNames[iplot2]);
      if (rsiNOutputs > 2)
      {
        fprintf(fp, "subplot(2,3,3), contourf(x,y,B3_%d)\n", jj+1);
        fwritePlotAxes(fp);
        fprintf(fp,"title(\'Contour Plot for %s\',",
                outputNames[rsiSet[2]]);
        fprintf(fp, "'FontSize',12,'FontWeight','bold')\n"); 
        fprintf(fp, "xlabel('%s','FontSize',12,'FontWeight','bold')\n",
                inputNames[iplot1]);
        fprintf(fp, "ylabel('%s','FontSize',12,'FontWeight','bold');\n",
                inputNames[iplot2]);
      }
      if (rsiNOutputs > 3)
      {
        fprintf(fp, "subplot(2,3,4), contourf(x,y,B4_%d)\n", jj+1);
        fwritePlotAxes(fp);
        fprintf(fp, "title(\'Contour Plot for ");
        fprintf(fp, "%s\','FontSize',12,'FontWeight','bold')\n", 
                outputNames[rsiSet[3]]);
        fprintf(fp, "xlabel('%s','FontSize',12,'FontWeight','bold')\n",
                inputNames[iplot1]);
        fprintf(fp, "ylabel('%s','FontSize',12,'FontWeight','bold');\n",
                inputNames[iplot2]);
      }
      if (rsiNOutputs > 4)
      {
        fprintf(fp, "subplot(2,3,5), contourf(x,y,B5_%d)\n", jj+1);
        fwritePlotAxes(fp);
        fprintf(fp, "title(\'Contour Plot for ");
        fprintf(fp, "%s\','FontSize',12,'FontWeight','bold')\n", 
                outputNames[rsiSet[4]]);
        fprintf(fp, "xlabel('%s','FontSize',12,'FontWeight','bold')\n",
                inputNames[iplot1]);
        fprintf(fp, "ylabel('%s','FontSize',12,'FontWeight','bold');\n",
                inputNames[iplot2]);
      }
      //**/if (rsiNOutputs <= 3)
      //**/   fprintf(fp, "subplot(2,3,[4 5]), surfl(x,y,M%d)\n", jj+1);
      //**/else
      //**/   fprintf(fp, "subplot(2,3,5), surfl(x,y,M%d)\n", jj+1);
      //**/fwritePlotAxes(fp);
      //**/fprintf(fp,"xlabel('%s','FontSize',12,'FontWeight','bold')\n",
      //**/        inputNames[iplot1]);
      //**/fprintf(fp,"ylabel('%s','FontSize',12,'FontWeight','bold');\n",
      //**/        inputNames[iplot2]);
      //**/fprintf(fp, "title('Intersection','FontSize',12,");
      //**/fprintf(fp, "'FontWeight','bold')\n");
      //**/fprintf(fp, "colorbar\n");
      fprintf(fp, "subplot(2,3,6), contourf(x,y,M%d)\n",jj+1);
      fwritePlotAxes(fp);
      fwritePlotXLabel(fp, inputNames[iplot1]);
      fwritePlotYLabel(fp, inputNames[iplot2]);
      fprintf(fp, "title('Intersection: Input %s = %11.4e',",
              inputNames[iplot3], vecInpSettings[iplot3]);
      fprintf(fp, "'FontSize',12,'FontWeight','bold')\n");
      fprintf(fp, "colorbar\n");
      fprintf(fp, "colormap(jet)\n");
      fprintf(fp,"pause(1)\n");
    }
    //**/ Sept 2010: not good enough, create rsplot3m_int
    //**/fprintf(fp,"disp(\'Press enter to view 3D shape plot\')\n");
    //**/fprintf(fp,"pause\n");
    //**/fprintf(fp,"hold off\n");
    //**/fprintf(fp,"clf\n");
    //**/fprintf(fp, "yLo = %e;\n", threshL);
    //**/fprintf(fp, "yHi = %e;\n", threshU);
    //**/for (jj = 0; jj < nPtsPerDim; jj++)
    //**/{
    //**/   vecInpSettings[iplot3] = (iUpperB[iplot3] - iLowerB[iplot3]) *
    //**/                       jj/(nPtsPerDim - 1.0) + iLowerB[iplot3];
    //**/   fprintf(fp, "B%d = ones(%d) * NaN;\n", jj+1, nPtsPerDim);
    //**/   fprintf(fp, "[ia,ja,aa] = find(M%d==%d);\n", jj+1,rsiNOutputs);
    //**/   fprintf(fp, "for ii = 1 : length(ia)\n");
    //**/   fprintf(fp, "   B%d(ia(ii),ja(ii)) = %e;\n", jj+1,
    //**/           vecInpSettings[iplot3]);
    //**/   fprintf(fp, "end;\n");
    //**/   fprintf(fp, "if (length(ia) == 0)\n");
    //**/   fprintf(fp, "   B%d(1,1) = 0;\n", jj+1);
    //**/   fprintf(fp, "   B%d(%d,%d) = 1;\n",jj+1,nPtsPerDim,nPtsPerDim);
    //**/   fprintf(fp, "end;\n");
    //**/   //**/fprintf(fp, "surf(x,y,B%d,M%d)\n",jj+1,jj+1);
    //**/   fprintf(fp, "surf(x,y,B%d)\n",jj+1);
    //**/   if (jj == 0) fprintf(fp, "hold on;\n");
    //**/}
    //**/fprintf(fp, "rotate3d on\n");
    fclose(fp);
    printf("rsplot3m_int plot file = matlabrsplot3m_int.m\n");

    delete [] rsiSet;
    delete faPtr;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rssd_ua or rsua_sd_plot
  //**/ uncertainty analysis of standard deviations from RS fit 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rssd_ua") ||
           !strcmp(command, "rsua_sd_plot"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsua_sd_plot: create pdf for RS std. dev.\n");
      printf("syntax: rsua_sd_plot (no argument needed).\n");
      return 0;
    }
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: no data to analyze (load sample first).\n");
      return 1;
    }

    printAsterisks(PL_INFO, 0);
    printf("This command first creates a response surface for a ");
    printf("selected output.\n");
    printf("It then creates a large sample from the input ");
    printf("distributions and\n");
    printf("propagates it through the response surface, ");
    printf("collecting at each sample\n");
    printf("point its prediction uncertainty (thus, this ");
    printf("command requires the use\n");
    printf("of a stochastic response surface such as regression, ");
    printf("GP or Kriging),\n");
    printf("and finally creating a histogram of the prediction ");
    printf("uncertainties.\n");
    printf("You will be asked to select a response surface ");
    printf("(RS) type.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ choose function approximator
    printf("This command works with the following RS types:\n");
    printf("1. Linear    regression\n");
    printf("2. Quadratic regression\n");
    printf("3. cubic     regression\n");
    printf("4. quartic   regression\n");
#ifdef HAVE_TPROS
    printf("5. GP1 (MacKay implementation)\n");
    printf("6. GP3 (Tong implementation)\n");
    printf("7. MarsBagg\n");
    printf("8. Tree GP\n");
    printf("9. Kriging\n");
    snprintf(pString,100,"Enter your choice: (1, 2, ..., 9) ");
    int faType = getInt(1, 9, pString);
#else
    printf("5. GP3 (Tong implementation)\n");
    printf("6. MarsBagg\n");
    printf("7. Tree GP\n");
    printf("8. Kriging\n");
    snprintf(pString,100,"Enter your choice: (1, 2, ..., 8) ");
    int faType = getInt(1, 8, pString);
#endif
    if      (faType == 1) faType = PSUADE_RS_REGR1;
    else if (faType == 2) faType = PSUADE_RS_REGR2;
    else if (faType == 3) faType = PSUADE_RS_REGR3;
    else if (faType == 4) faType = PSUADE_RS_REGR4;
#ifdef HAVE_TPROS
    else if (faType == 5) faType = PSUADE_RS_GP1;
    else if (faType == 6) faType = PSUADE_RS_GP3;
    else if (faType == 7) faType = PSUADE_RS_MARSB;
    else if (faType == 8) faType = PSUADE_RS_TGP;
    else if (faType == 9) faType = PSUADE_RS_KR;
#else
    else if (faType == 5) faType = PSUADE_RS_GP3;
    else if (faType == 6) faType = PSUADE_RS_MARSB;
    else if (faType == 7) faType = PSUADE_RS_TGP;
    else if (faType == 8) faType = PSUADE_RS_KR;
#endif

    //**/ ask users to specify one output
    int jplot = 0;
    snprintf(pString,100,"Enter the output number (1 - %d) : ",nOutputs);
    jplot = getInt(1, nOutputs, pString);
    jplot--;

    //**/ set up function approximator
    printf("rsua_sd_plot: setting up function approximator\n");
    int iOne=1, sInd;
    FuncApprox *faPtr = genFA(faType, nInputs, iOne, nSamples);
    if (faPtr == NULL) {printf("ERROR detected in RS.\n"); return 1;}
    psVector vecFaYIn;
    vecFaYIn.setLength(nSamples);
    for (sInd = 0; sInd < nSamples; sInd++)
      vecFaYIn[sInd] = sampleOutputs[sInd*nOutputs+jplot];
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);
    faPtr->initialize(sampleInputs,vecFaYIn.getDVector());

    //**/ generate a quasi-Monte Carlo sample
    printf("rsua_sd_plot: creating a large sample for building PDF\n");
    Sampling *sampPtr = 
         (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LPTAU);
    sampPtr->setPrintLevel(0);
    sampPtr->setInputBounds(nInputs, iLowerB, iUpperB);
    sampPtr->setOutputParams(1);
    int count = 100000;
    sampPtr->setSamplingParams(count, -1, 1);
    sampPtr->initialize(0);
    psVector  vecXT, vecYT, vecWT;
    psIVector vecST;
    vecXT.setLength(count*nInputs);
    vecYT.setLength(count);
    vecWT.setLength(count);
    vecST.setLength(count);
    sampPtr->getSamples(count,nInputs,1,vecXT.getDVector(),
                        vecYT.getDVector(), vecST.getIVector());
    faPtr->evaluatePointFuzzy(count,vecXT.getDVector(),vecYT.getDVector(), 
                              vecWT.getDVector());

    //**/ put the scale results in vecYT
    psVector vecSW;
    vecSW.setLength(count);
    for (ii = 0; ii < count; ii++)
    {
      if (vecYT[ii] == 0.0) vecSW[ii] = vecWT[ii]; 
      else                  vecSW[ii] = vecWT[ii]/vecYT[ii];
    }

    //**/ plot the result 
    char fname[100];
    if (plotScilab()) strcpy(fname, "scilabrsua_sd_plot.sci");
    else              strcpy(fname, "matlabrsua_sd_plot.m");
    fp = fopen(fname, "w");
    if (fp != NULL)
    {
      strcpy(pString," Col 1: std dev, Col 2: scaled std dev, Col 3: YOut");
      fwriteComment(fp, pString);
      fprintf(fp, "Y = [\n");
      for (ss = 0; ss < nSamples; ss++)
        fprintf(fp, "  %24.16e %24.16e %24.16e\n",vecWT[ss],vecSW[ss],
                vecYT[ss]);
      fprintf(fp, "];\n");
      if (plotScilab())
      {
        fwritePlotCLF(fp);
        fprintf(fp, "subplot(1,2,1)\n");
        fprintf(fp, "ymin = min(Y(:,1));\n");
        fprintf(fp, "ymax = max(Y(:,1));\n");
        fprintf(fp, "ywid = 0.1 * (ymax - ymin);\n");
        fprintf(fp, "if (ywid < 1.0e-12)\n");
        fprintf(fp, "   disp('range too small.')\n");
        fprintf(fp, "   halt\n");
        fprintf(fp, "end;\n");
        fprintf(fp, "histplot(10, Y(:,1)/ywid, style=2);\n");
        fprintf(fp, "a = gce();\n");
        fprintf(fp, "a.children.fill_mode = \"on\";\n");
        fprintf(fp, "a.children.thickness = 2;\n");
        fprintf(fp, "a.children.foreground = 0;\n");
        fprintf(fp, "a.children.background = 2;\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Prediction Std. Dev. Distribution");
        fwritePlotXLabel(fp, "Output Std. Dev.");
        fwritePlotYLabel(fp, "Probabilities");
        fprintf(fp, "subplot(1,2,2)\n");
        fprintf(fp, "ymin = min(Y(:,2));\n");
        fprintf(fp, "ymax = max(Y(:,2));\n");
        fprintf(fp, "ywid = 0.1 * (ymax - ymin);\n");
        fprintf(fp, "if (ywid < 1.0e-12)\n");
        fprintf(fp, "   disp('range too small.')\n");
        fprintf(fp, "   halt\n");
        fprintf(fp, "end;\n");
        fprintf(fp, "histplot(10, Y(:,2)/ywid, style=2);\n");
        fprintf(fp, "a = gce();\n");
        fprintf(fp, "a.children.fill_mode = \"on\";\n");
        fprintf(fp, "a.children.thickness = 2;\n");
        fprintf(fp, "a.children.foreground = 0;\n");
        fprintf(fp, "a.children.background = 2;\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Prediction Scaled Std. Dev. Distribution");
        fwritePlotXLabel(fp, "Output Scaled Std. Dev.");
        fwritePlotYLabel(fp, "Probabilities");
      }
      else
      {
        fwritePlotCLF(fp);
        fprintf(fp, "twoPlots = 0;\n");
        fprintf(fp, "if (twoPlots == 1)\n");
        fprintf(fp, "subplot(1,2,1)\n");
        if (nSamples > 500) fprintf(fp, "[nk,xk]=hist(Y(:,1),20);\n");
        else                fprintf(fp, "[nk,xk]=hist(Y(:,1),10);\n");
        fprintf(fp, "bar(xk,nk/%d,1.0)\n",nSamples);
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Prediction Std. Dev. Distribution");
        fwritePlotXLabel(fp, "Output Std. Dev.");
        fwritePlotYLabel(fp, "Probabilities");
        fprintf(fp, "subplot(1,2,2)\n");
        fprintf(fp, "end;\n");
        if (nSamples > 500) fprintf(fp, "[nk,xk]=hist(Y(:,2),20);\n");
        else                fprintf(fp, "[nk,xk]=hist(Y(:,2),10);\n");
        fprintf(fp, "bar(xk,nk/%d,1.0)\n",nSamples);
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Prediction Scaled Std. Dev. Distribution");
        fwritePlotXLabel(fp, "Output Scaled Std. Dev.");
        fwritePlotYLabel(fp, "Probabilities");

        fprintf(fp, "if (twoPlots == 1)\n");
        fprintf(fp, "figure(2)\n");
        fprintf(fp, "Yk = sort(Y(:,1));\n");
        fprintf(fp, "Xk = 1 : %d;\n", nSamples);
        fprintf(fp, "Xk = Xk / %d;\n", nSamples);
        fprintf(fp, "subplot(1,2,2)\n");
        fprintf(fp, "plot(Yk, Xk, 'lineWidth',3)\n");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Cumulative Std Dev Distribution");
        fwritePlotXLabel(fp, "Cumulative Std. Dev.");
        fwritePlotYLabel(fp, "Probabilities");
        fprintf(fp, "end;\n");
      }
      printf("rsua_sd_plot distribution plot file = %s\n",fname);
      fclose(fp);
    }
    else 
    {
      printOutTS(PL_ERROR,"ERROR: Cannot open file %s.\n", fname);
    }

    //**/ clean up
    delete sampPtr;
    delete faPtr;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsmeb (or rsvce1b_bin)
  //**/ bootstrapped main effect analysis using replicated LH
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsmeb") || 
           !strcmp(command, "rsvce1_bin"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsvce1_bin: Sobol' first-order analysis on ");
      printf("response surface\n");
      printf("Syntax: rsvce1_bin (no argument needed)\n");
      return 0;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command computes first-order Sobol' indices using ");
    printf("the response\n");
    printf("surface built from the loaded sample.\n");
    printf("NOTE: This analysis supports all types of input ");
    printf("distributions:\n");
    printf("   - use rLHS for inputs with uniform distributions\n");
    printf("   - use binning for inputs with other distributions\n");
    printf("   - use binning if there are input constraints\n");
    printf("NOTE: Turn on ana_expert mode to run bootstrapped analysis.\n");
    printf("NOTE: Turn on ana_expert mode to change sample size.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    //**/ error checking
    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }

    //**/ select which output to analyze
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    //**/ select response surface
    faType = -1;
    snprintf(pString,100,"Enter your response surface choice ? ");
    while (faType < 0 || faType > PSUADE_NUM_RS)
    {
      writeFAInfo(outputLevel_);
      faType = getFAType(pString);
    }
    if (faType < 0)
    {
      printf("ERROR: Response surface type not currently available.\n");
      return 1;
    }
    if (faType == PSUADE_RS_MARSB) faType = PSUADE_RS_MARS;

    //**/ ask user for sample size
    int nLHS=2000000;
    if (psConfig_.AnaExpertModeIsOn())
    {
      snprintf(pString,100,
        "Sample size for generating distribution? (1M - 5M, default=2M) ");
      nLHS = getInt(100000, 5000000, pString);
    }

    //**/ number of bootstrapped runs (in ana_expert mode only)
    int numBS = 1;
    if (psConfig_.AnaExpertModeIsOn())
    {
      printf("INFO: You may use bootstrapping to estimate ");
      printf("errors in this analysis\n");
      printf("      (computing multiple sets of metrics ");
      printf("on multiple response\n");
      printf("      surfaces built from bootstrapped samples).\n");
      snprintf(pString,100,"Number of bootstrapped runs (1 - 50) : ");
      numBS = getInt(1, 50, pString);
    }

    //**/ detect any non-normal input distributions
    int hasPDF=0;
    if (VecInpPDFs_.length() != 0)
    {
      for (ii = 0; ii < nInputs; ii++)
        if (VecInpPDFs_[ii] != 0) hasPDF++; 
    }
    else
    {
      printf("NOTE: All input distributions are uniform. ");
      printf("You may also use\n");
      printf("      rsvce1_sobol or rsvce1_ni to compute ");
      printf("first-order Sobol'\n");
      printf("      indices (or for comparison purposes).\n");
    }

    //**/ if distributions are uniform, create replicated LH sample 
    int nReps=1000, iOne=1, iZero=0;
    psVector  vecLHSInps, vecLHSOuts;
    if (hasPDF == 0)
      printf("Phase 1: Create a replicated LH sample (%d,r=%d)\n",
             nLHS,nReps);
    else
      printf("Phase 1: Create a large random sample (%d)\n",nLHS);
    nLHS = nLHS / nReps * nReps;
    vecLHSInps.setLength(nLHS*nInputs);
    vecLHSOuts.setLength(nLHS);
    printf("NOTE: Sample size to be used for analysis = %d (r=%d).\n",
           nLHS,nReps);
    if (hasPDF == 0)
    {
      Sampling *samPtr;
      psIVector vecLHSStas;
      vecLHSStas.setLength(nLHS);
      samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
      samPtr->setPrintLevel(0);
      samPtr->setInputBounds(nInputs, iLowerB, iUpperB);
      samPtr->setInputParams(nInputs, NULL, NULL, NULL);
      samPtr->setOutputParams(iOne);
      samPtr->setSamplingParams(nLHS, nReps, iZero);
      psConfig_.SamExpertModeSaveAndReset();
      samPtr->initialize(0);
      psConfig_.SamExpertModeRestore();
      samPtr->getSamples(nLHS, nInputs, iOne, vecLHSInps.getDVector(),
                         vecLHSOuts.getDVector(),vecLHSStas.getIVector());
      delete samPtr;
    }
    //**/ if non-uniform, create random sample
    else
    {
      PDFManager *pdfman = new PDFManager();
      pdfman->initialize(nInputs,VecInpPDFs_.getIVector(),
                    VecInpMeans_.getDVector(),VecInpStds_.getDVector(),
                    *inputCMat_,SamPDFFiles_,
                    VecSamPDFIndices_.getIVector());
      psVector vecLs, vecUs;
      vecUs.load(nInputs, iUpperB);
      vecLs.load(nInputs, iLowerB);
      pdfman->genSample(nLHS, vecLHSInps, vecLs, vecUs);
      delete pdfman;
    }

    //**/ create response surface ==> faPtr
    printEquals(PL_INFO, 0);
    printf("Phase 2: Create a response surface\n");
    psVector vecYT;
    if (nOutputs > 1)
    {
      vecYT.setLength(nSamples);
      for (ss = 0; ss < nSamples; ss++)
        vecYT[ss] = sampleOutputs[ss*nOutputs+outputID];
    }
    else vecYT.load(nSamples,sampleOutputs);

    //**/ prepare to call MainEffectAnalyzer.cpp
    int ind, nSamples2;
    psIVector vecMebInds;
    psVector  vecTmpInps, vecTmpOuts, vecMeStore;
    vecTmpInps.setLength(nSamples*nInputs);
    vecTmpOuts.setLength(nSamples);
    vecMebInds.setLength(nSamples);
    vecMeStore.setLength((numBS+2)*nInputs);
    MainEffectAnalyzer *meAnalyzer = new MainEffectAnalyzer();
    pData *pd = NULL;
    aData adata;
    adata.nInputs_ = nInputs;
    adata.nOutputs_ = 1;
    adata.nSamples_ = nLHS;
    adata.outputID_ = 0;
    adata.sampleInputs_ = vecLHSInps.getDVector();
    adata.sampleOutputs_ = vecLHSOuts.getDVector();
    adata.nSubSamples_ = nLHS / nReps;
    adata.iLowerB_ = iLowerB;
    adata.iUpperB_ = iUpperB;
    adata.printLevel_ = -1;
    adata.ioPtr_ = psuadeIO;
    psConfig_.AnaExpertModeSaveAndReset();
    psConfig_.RSExpertModeSaveAndReset();

    printf("Phase 3: Compute first-order Sobol' indices\n");
    FuncApprox *faPtr=NULL;
    double totalVar = 0;
    for (kk = 0; kk < numBS; kk++)
    {
      if (numBS > 1)
        printf("* Performing analysis on bootstrap %d (of %d)\n", 
               kk+1, numBS);
      //**/ random draw
      if (numBS > 1)
      {
        for (ss = 0; ss < nSamples; ss++) vecMebInds[ss] = 0;
        ss = nSamples2 = 0;
        while (ss < nSamples && nSamples2 < 0.8*nSamples)
        {
          ind = PSUADE_rand() % nSamples;
          if (vecMebInds[ind] == 0)
          {
            for (ii = 0; ii < nInputs; ii++)
              vecTmpInps[nSamples2*nInputs+ii] = 
                            sampleInputs[ind*nInputs+ii];
            vecTmpOuts[nSamples2] = vecYT[ind];
            vecMebInds[ind] = 1;
            nSamples2++;
          }
          ss++;
        }
      }
      else
      {
        nSamples2 = nSamples;
        for (ss = 0; ss < nSamples; ss++) 
        {
          for (ii = 0; ii < nInputs; ii++) 
            vecTmpInps[ss*nInputs+ii] = sampleInputs[ss*nInputs+ii];
          vecTmpOuts[ss] = vecYT[ss];
        }
      }
      if (!psConfig_.AnaExpertModeIsOn() || numBS > 1)
        psConfig_.InteractiveSaveAndReset();
      faPtr = genFA(faType, nInputs, -1, nSamples2);
      faPtr->setNPtsPerDim(32);
      faPtr->setBounds(iLowerB, iUpperB);
      faPtr->setOutputLevel(0);
      faPtr->initialize(vecTmpInps.getDVector(),vecTmpOuts.getDVector());
      faPtr->evaluatePoint(nLHS,vecLHSInps.getDVector(),
                           vecLHSOuts.getDVector());
      //**/ perform main effect analysis
      meAnalyzer->analyze(adata);
      if (!psConfig_.AnaExpertModeIsOn() || numBS > 1)
        psConfig_.InteractiveRestore();

      pd = psuadeIO->getAuxData();
      totalVar += pd->dbleData_;
      for (ii = 0; ii < nInputs; ii++)
      {
        if (pd->dbleData_ > 0)
             vecMeStore[kk*nInputs+ii] = 
                pd->dbleArray_[ii]/pd->dbleData_;
        else vecMeStore[kk*nInputs+ii] = pd->dbleArray_[ii];
      }
      if (outputLevel_ > 2 && numBS > 1)
      {
        printf("Bootstrap %d (of %d):\n",kk+1,numBS); 
        for (ii = 0; ii < nInputs; ii++)
          printf("   VCE(%d) = %10.3e\n",ii+1,
                  vecMeStore[kk*nInputs+ii]);
      }
      pd->clean();
      delete faPtr;
    }
    totalVar /= (double) numBS;

    //**/ compute main effects
    double mean, stdev;
    printAsterisks(PL_INFO, 0);
    printf("Sobol' 1st-order indices (normalized, 'bin')\n");
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < nInputs; ii++)
    {
      mean = 0.0;
      for (kk = 0; kk < numBS; kk++) mean += vecMeStore[kk*nInputs+ii];
      mean /= numBS;
      vecMeStore[numBS*nInputs+ii] = mean;
      stdev = 0.0;
      if (numBS > 1)
      {
        for (kk = 0; kk < numBS; kk++)
          stdev += pow(vecMeStore[kk*nInputs+ii]-mean, 2.0);
        stdev = sqrt(stdev/(numBS-1));
      }
      vecMeStore[(numBS+1)*nInputs+ii] = stdev;
      if (numBS > 1)
        printf("VCE for input %3d = %10.3e, s.d. = %10.3e\n",ii+1,
               mean, stdev);
      else
        printf("VCE for input %3d = %10.3e\n",ii+1,mean);
    }
    printf("Total variance = %10.3e\n",totalVar);
    printAsterisks(PL_INFO, 0);

    //**/ generate matlab/scilab file
    fp = NULL;
    if (plotScilab()) fp = fopen("scilabrsvce1_bin.sci","w");
    else              fp = fopen("matlabrsvce1_bin.m","w");
    if (fp == NULL) printf("ERROR: Cannot open plot file.\n");
    else
    {
      strcpy(pString," This file contains first-order Sobol' effect ");
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
      fprintf(fp, "nn = %d;\n", nInputs);
      fprintf(fp, "Means = [\n");
      for (ii = 0; ii < nInputs; ii++)
         fprintf(fp,"%24.16e\n",vecMeStore[numBS*nInputs+ii]);
      fprintf(fp, "];\n");
      fprintf(fp, "Stds = [\n");
      for (ii = 0; ii < nInputs; ii++)
        fprintf(fp,"%24.16e\n",vecMeStore[(numBS+1)*nInputs+ii]);
      fprintf(fp, "];\n");
      if (inputNames == NULL)
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
          if (inputNames[ii] != NULL) 
               fprintf(fp,"'%s',",inputNames[ii]);
          else fprintf(fp,"'X%d',",ii+1);
        }
        if (plotScilab())
        {
          if (inputNames[nInputs-1] != NULL) 
             fprintf(fp,"'%s'];\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'];\n",nInputs);
        }
        else
        {
          if (inputNames[nInputs-1] != NULL) 
             fprintf(fp,"'%s'};\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'};\n",nInputs);
        }
      }
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
      fprintf(fp,"   end;\n");
      fprintf(fp,"   XX = [ii ii];\n");
      fprintf(fp,"   d1 = Means(ii)-Stds(ii);\n");
      fprintf(fp,"   d2 = Means(ii)+Stds(ii);\n");
      fprintf(fp,"   if (d1 < 0)\n");
      fprintf(fp,"      d1 = 0.0;\n");
      fprintf(fp,"   end;\n");
      fprintf(fp,"   YY = [d1 d2];\n");
      fprintf(fp,
        "   plot(XX,YY,'-ko','LineWidth',3.0,'MarkerEdgeColor',");
      fprintf(fp,"'k','MarkerFaceColor','g','MarkerSize',13)\n");
      fprintf(fp,"end;\n");
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
        fprintf(fp,"th=text(1:nn, repmat(ymin-0.05*(ymax-ymin),nn,1),Str,");
        fprintf(fp,"'HorizontalAlignment','left','rotation',90);\n");
        fprintf(fp,"set(th, 'fontsize', 12)\n");
        fprintf(fp,"set(th, 'fontweight', 'bold')\n");
      }
      fwritePlotTitle(fp,"First-order Sobol'Indices (with bootstrap)");
      fwritePlotYLabel(fp, "First-order Sobol'Indices (Normalized)");
      if (plotScilab())
      {
        fprintf(fp, "drawnow\n");
        printf("Scilab main effect plot file = scilabrsvce1_bin.sci\n");
      }
      else 
      {
        printf("Matlab main effect plot file = matlabrsvce1_bin.m\n");
      }
      fclose(fp);
    }
    delete meAnalyzer;
    faPtr = NULL;
    psConfig_.AnaExpertModeRestore();
    psConfig_.RSExpertModeRestore();
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsieb (or rsvce2_bin)
  //**/ bootstrapped pairwise effect analysis using replicated OA
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsieb") ||
           !strcmp(command, "rsvce2_bin"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsvce2_bin: Sobol' second-order analysis on ");
      printf("response surface\n");
      printf("Syntax: rsvce2_bin (no argument needed)\n");
      return 0;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command computes second-order Sobol' indices using ");
    printf("the response\n");
    printf("surface built from the loaded sample.\n");
    printf("NOTE: This analysis supports all types of input ");
    printf("distributions:\n");
    printf("   - use rOA for inputs with uniform distributions\n");
    printf("   - use binning for inputs with other distributions\n");
    printf("NOTE: Turn on ana_expert mode to run bootstrapped analysis.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (psuadeIO == NULL || sampleOutputs == NULL)
    {
      printf("ERROR: Sample not loaded yet.\n");
      return 1;
    }
    if (nInputs <= 2)
    {
      printf("INFO: nInputs <=2 -> no need to perform this analysis.\n");
      return 1;
    }

    //**/ query user for which output and which response surface type
    snprintf(pString,100,"Enter output number (1 - %d) : ", nOutputs);
    outputID = getInt(1, nOutputs, pString);
    outputID--;

    faType = -1;
    snprintf(pString,100,"Enter your response surface choice ? ");
    while (faType < 0 || faType > PSUADE_NUM_RS)
    {
      writeFAInfo(outputLevel_);
      faType = getFAType(pString);
    }
    if (faType < 0)
    {
      printf("ERROR: Response surface type not currently available.\n");
      return 1;
    }
    if (faType == PSUADE_RS_MARSB) faType = PSUADE_RS_MARS;

    //**/ number of bootstrapped runs (in ana_expert mode only)
    int numBS = 1;
    if (psConfig_.AnaExpertModeIsOn())
    {
      printf("INFO: You may use bootstrapping to estimate ");
      printf("errors in this analysis\n");
      printf("      (computing multiple sets of metrics ");
      printf("on multiple response\n");
      printf("      surfaces built from bootstrapped samples).\n");
      snprintf(pString,100,"Number of bootstrapped runs (1 - 50) : ");
      numBS = getInt(1, 50, pString);
    }
    printEquals(PL_INFO, 0);

    //**/ detect any non-normal input distributions
    int hasPDF=0;
    if (VecInpPDFs_.length() != 0)
    {
      for (ii = 0; ii < nInputs; ii++)
        if (VecInpPDFs_[ii] != 0) hasPDF++; 
    }

    //**/ create sample: distributions are uniform, create 
    //**/ replicated OA sample. If not, use QMC 
    if (hasPDF == 0)
      printf("Phase 1: Create a replicated OA sample\n");
    else
      printf("Phase 1: Create a large random sample\n");
    int nSamp, iOne=1, iZero=0, nReps=100, nOA, nOA1=317;
    psVector vecOAInps, vecOAOuts;
    nOA   = nOA1 * nOA1;
    nSamp = nOA * nReps;
    vecOAInps.setLength(nInputs*nSamp);
    vecOAOuts.setLength(nSamp);
    if (hasPDF == 0)
    {
      printf("NOTE: Sample size to be used for analysis = %d.\n",
             nSamp);
      Sampling *samPtr;
      psIVector vecOAStas;
      vecOAStas.setLength(nSamp);
      samPtr = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_OA);
      samPtr->setPrintLevel(outputLevel_);
      samPtr->setInputBounds(nInputs, iLowerB, iUpperB);
      samPtr->setInputParams(nInputs, NULL, NULL, NULL);
      samPtr->setOutputParams(iOne);
      samPtr->setSamplingParams(nSamp, nReps, iZero);
      samPtr->initialize(0);
      samPtr->getSamples(nSamp,nInputs,iOne,vecOAInps.getDVector(),
                 vecOAOuts.getDVector(), vecOAStas.getIVector());
      delete samPtr;
    }
    else
    {
      PDFManager *pdfman = new PDFManager();
      pdfman->initialize(nInputs,VecInpPDFs_.getIVector(),
                VecInpMeans_.getDVector(),VecInpStds_.getDVector(),
                *inputCMat_,NULL,NULL);
      psVector vecLs, vecUs;
      vecUs.load(nInputs, iUpperB);
      vecLs.load(nInputs, iLowerB);
      printf("NOTE: Sample size to be used for analysis = %d.\n",nSamp);
      pdfman->genSample(nSamp, vecOAInps, vecLs, vecUs);
      delete pdfman;
    }

    //**/ acquire and prepare for response surface generation
    printEquals(PL_INFO, 0);
    printf("Phase 2: Create response surface and compute metric\n");
    psVector vecYT;
    if (nOutputs > 1)
    {
      vecYT.setLength(nSamples);
      for (ss = 0; ss < nSamples; ss++)
        vecYT[ss] = sampleOutputs[ss*nOutputs+outputID];
    }
    else vecYT.load(nSamples,sampleOutputs);

#if 0
    //**/ set up storage space for analysis
    int    ii1, ii2, jj1, jj2, kk2, bin1, bin2, ind, nSamples2;
    double totVar=0, width1, width2, iemean, ievar, rsieMean;
    psIVector vecIeCount, vecIebInds;
    psVector  vecIeMeans, vecIeVars;
    psVector  vecIeStore, vecTmpInps, vecTmpOuts;
    vecTmpInps.setLength(nSamples*nInputs);
    vecTmpOuts.setLength(nSamples);
    vecIeStore.setLength((numBS+2)*nInputs*nInputs);
    vecIebInds.setLength(nSamples);
    vecIebInds.setLength(nSamples);
    vecIeCount.setLength(nOA1*nOA1);
    vecIeMeans.setLength(nOA1*nOA1);
    vecIeVars.setLength(nOA1*nOA1);
    FuncApprox *faPtr=NULL;

    //*/ create multiple bootstraps and evaluate and bin  
    for (kk = 0; kk < numBS; kk++)
    {
      if (numBS > 1)
        printf("* ITERATION %d (of %d)\n", kk+1, numBS);
      //**/ random draw for building RS (if numBS=1, use all)
      if (numBS > 1)
      {
        for (ss = 0; ss < nSamples; ss++) vecIebInds[ss] = 0;
        ss = nSamples2 = 0;
        while (ss < nSamples)
        {
          ind = PSUADE_rand() % nSamples;
          if (vecIebInds[ind] == 0)
          {
            for (ii = 0; ii < nInputs; ii++)
              vecTmpInps[nSamples2*nInputs+ii] = 
                         sampleInputs[ind*nInputs+ii];
            vecTmpOuts[nSamples2] = vecYT[ind];
            vecIebInds[ind] = 1;
            nSamples2++;
          }
          ss++;
        }
      }
      else
      {
        nSamples2 = nSamples;
        for (ss = 0; ss < nSamples; ss++) 
        {
          for (ii = 0; ii < nInputs; ii++) 
            vecTmpInps[ss*nInputs+ii] = sampleInputs[ss*nInputs+ii];
          vecTmpOuts[ss] = vecYT[ss];
        }
      }

      //**/ create response surface on bootstrap
      psConfig_.InteractiveSaveAndReset();
      faPtr = genFA(faType, nInputs, -1, nSamples2);
      faPtr->setNPtsPerDim(32);
      faPtr->setBounds(iLowerB, iUpperB);
      faPtr->setOutputLevel(0);
      faPtr->initialize(vecTmpInps.getDVector(),
                        vecTmpOuts.getDVector());
      faPtr->evaluatePoint(nSamp,vecSamInps.getDVector(),
                           vecSamOuts.getDVector());
      psConfig_.InteractiveRestore();
      delete faPtr;

      //**/ compute mean and variance
      if (kk == 0)
      {
        rsieMean = 0;
        for (ii1 = 0; ii1 < nSamp; ii1++) 
          rsieMean += vecSamOuts[ii1]; 
        rsieMean /= (double) nSamp;
        ddata = 0;
        for (ii1 = 0; ii1 < nSamp; ii1++) 
          ddata += pow(vecSamOuts[ii1] - rsieMean, 2.0); 
        ddata /= (nSamp -1);
        totVar += ddata; 
      }

      //**/ perform interaction effect analysis
      //**/ this code could have been in TwoParameter.cpp, but
      //**/ to refactor it will be too cumbersome so just leave it
      for (ii1 = 0; ii1 < nInputs; ii1++)
      {
        vecIeStore[kk*nInputs*nInputs+ii1*nInputs+ii1] = 0.0;
        width1 = (iUpperB[ii1] - iLowerB[ii1]) / (nOA1 - 1);
        for (ii2 = ii1+1; ii2 < nInputs; ii2++)
        {
          width2 = (iUpperB[ii2] - iLowerB[ii2]) / (nOA1 - 1);
          for (kk2 = 0; kk2 < nOA1*nOA1; kk2++) 
          {
            vecIeCount[kk2] = 0;
            vecIeMeans[kk2] = 0.0;
            vecIeVars[kk2] = 0.0;
          }
          for (kk2 = 0; kk2 < nSamp; kk2++)
          {
            bin1 = (int) ((vecSamInps[kk2*nInputs+ii1]-
                           iLowerB[ii1]+1.0e-12)/width1);
            bin2 = (int) ((vecSamInps[kk2*nInputs+ii2]-
                           iLowerB[ii2]+1.0e-12)/width2);
            vecIeMeans[bin1*nOA1+bin2] += vecSamOuts[kk2];
            vecIeCount[bin1*nOA1+bin2]++;
          }
          for (kk2 = 0; kk2 < nOA1*nOA1; kk2++)
            if (vecIeCount[kk2] > 0) vecIeMeans[kk2] /= vecIeCount[kk2];
          for (kk2 = 0; kk2 < nOA; kk2++)
          {
            bin1 = (int) ((vecSamInps[kk2*nInputs+ii1]-
                           iLowerB[ii1]+1.0e-12)/width1);
            bin2 = (int) ((vecSamInps[kk2*nInputs+ii2]-
                           iLowerB[ii2]+1.0e-12)/width2);
            vecIeVars[bin1*nOA1+bin2] += 
                    pow(vecSamOuts[kk2]-vecIeMeans[bin1*nOA1+bin2],2.0);
          }
          for (kk2 = 0; kk2 < nOA1*nOA1; kk2++)
            if (vecIeCount[kk2] > 0) vecIeVars[kk2] /= vecIeCount[kk2];
          iemean = 0.0;
          for (kk2 = 0; kk2 < nOA1*nOA1; kk2++) iemean += vecIeMeans[kk2];
          iemean /= (double) (nOA1 * nOA1);
          ievar = 0.0;
          for (kk2 = 0; kk2 < nOA1*nOA1; kk2++) 
            ievar += pow(vecIeMeans[kk2]-iemean,2.0);
          ievar /= (double) (nOA1 * nOA1);
          iemean = 0.0;
          for (kk2 = 0; kk2 < nOA1*nOA1; kk2++) iemean += vecIeVars[kk2];
          iemean /= (double) (nOA1 * nOA1);
          ievar -= iemean / nReps;
          if (ievar < 0) ievar = 0;
          vecIeStore[kk*nInputs*nInputs+ii1*nInputs+ii2] = ievar;
          vecIeStore[kk*nInputs*nInputs+ii2*nInputs+ii1] = ievar;
        }
      }
    } /* kk < numBS */
#else
    int    ind, nSamples2;
    double totVar=0;
    FuncApprox *faPtr=NULL;
    psIVector vecIebInds;
    psVector  vecTmpInps, vecTmpOuts, vecIeStore;
    vecTmpInps.setLength(nSamples*nInputs);
    vecTmpOuts.setLength(nSamples);
    vecIebInds.setLength(nSamples);
    vecIeStore.setLength((numBS+2)*nInputs*nInputs);
    TwoParamAnalyzer *ieAnalyzer = new TwoParamAnalyzer();
    pData *pd = NULL;
    aData adata;
    adata.nInputs_ = nInputs;
    adata.nOutputs_ = 1;
    adata.nSamples_ = nSamp;
    adata.outputID_ = 0;
    adata.sampleInputs_ = vecOAInps.getDVector();
    adata.sampleOutputs_ = vecOAOuts.getDVector();
    adata.nSubSamples_ = nOA;
    adata.iLowerB_ = iLowerB;
    adata.iUpperB_ = iUpperB;
    adata.printLevel_ = -1;
    adata.ioPtr_ = psuadeIO;

    //*/ create multiple bootstraps and evaluate and bin  
    printf("Phase 3: Compute second-order Sobol' indices\n");
    for (kk = 0; kk < numBS; kk++)
    {
      if (numBS > 1)
        printf("* Performing analysis on bootstrap %d (of %d)\n", 
               kk+1, numBS);
      //**/ random draw for building RS (if numBS=1, use all)
      if (numBS == 1)
      {
        nSamples2 = nSamples;
        for (ss = 0; ss < nSamples; ss++) 
        {
          for (ii = 0; ii < nInputs; ii++) 
            vecTmpInps[ss*nInputs+ii] = sampleInputs[ss*nInputs+ii];
          vecTmpOuts[ss] = vecYT[ss];
        }
      }
      else
      {
        for (ss = 0; ss < nSamples; ss++) vecIebInds[ss] = 0;
        ss = nSamples2 = 0;
        while (ss < nSamples && nSamples2 < 0.8*nSamples)
        {
          ind = PSUADE_rand() % nSamples;
          if (vecIebInds[ind] == 0)
          {
            for (ii = 0; ii < nInputs; ii++)
              vecTmpInps[nSamples2*nInputs+ii] = 
                         sampleInputs[ind*nInputs+ii];
            vecTmpOuts[nSamples2] = vecYT[ind];
            vecIebInds[ind] = 1;
            nSamples2++;
          }
          ss++;
        }
      }

      //**/ create response surface on bootstrap
      if (!psConfig_.AnaExpertModeIsOn() || numBS > 1)
        psConfig_.InteractiveSaveAndReset();
      faPtr = genFA(faType, nInputs, -1, nSamples2);
      faPtr->setNPtsPerDim(32);
      faPtr->setBounds(iLowerB, iUpperB);
      faPtr->setOutputLevel(0);
      faPtr->initialize(vecTmpInps.getDVector(),
                        vecTmpOuts.getDVector());
      faPtr->evaluatePoint(nSamp,vecOAInps.getDVector(),
                           vecOAOuts.getDVector());
      ieAnalyzer->analyze(adata);
      if (!psConfig_.AnaExpertModeIsOn() || numBS > 1)
        psConfig_.InteractiveRestore();
      pd = psuadeIO->getAuxData();
      for (ii = 0; ii < nInputs*nInputs; ii++)
      {
        if (pd->dbleData_ > 0)
          vecIeStore[kk*nInputs*nInputs+ii] = 
                pd->dbleArray_[ii]/pd->dbleData_;
        else 
          vecIeStore[kk*nInputs*nInputs+ii] = pd->dbleArray_[ii];
      }
      //for (ii = 0; ii < nInputs; ii++)
      //  for (jj = ii+1; jj < nInputs; jj++)
      //    printf("(1st+2nd) VCE(%d,%d) = %10.3e\n",ii+1,jj+1,
      //           vecIeStore[kk*nInputs*nInputs+ii*nInputs+jj]);
      totVar += pd->dbleData_;
      pd->clean();
      delete faPtr;
    }
#endif
 
    //**/ compute aggregate statistics
    double mean, stdev;
    totVar /= (double) numBS;
    printAsterisks(PL_INFO, 0);
    printf("Sobol' second-order indices (normalized):bin\n");
    printEquals(PL_INFO, 0);
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = ii+1; jj < nInputs; jj++)
      {
        mean = 0.0;
        for (kk = 0; kk < numBS; kk++)
          mean += vecIeStore[kk*nInputs*nInputs+ii*nInputs+jj];
        mean /= numBS;
        vecIeStore[numBS*nInputs*nInputs+ii*nInputs+jj] = mean;
        vecIeStore[numBS*nInputs*nInputs+jj*nInputs+ii] = 0.0;
        stdev = 0.0;
        if (numBS > 1)
        {
          for (kk = 0; kk < numBS; kk++)
            stdev += pow(vecIeStore[kk*nInputs*nInputs+ii*nInputs+jj]-
                     mean,2.0);
          stdev = sqrt(stdev/(numBS-1));
        }
        vecIeStore[(numBS+1)*nInputs*nInputs+ii*nInputs+jj] = stdev;
        vecIeStore[(numBS+1)*nInputs*nInputs+jj*nInputs+ii] = 0.0;
        if (numBS > 1)
          printf("VCE (1st+2nd) for Input %3d,%3d = %10.3e (s.d. = %10.3e)\n",
                 ii+1, jj+1,mean,stdev);
        else
          printf("(1st+2nd)VCE for Input %3d,%3d = %10.3e\n",ii+1,jj+1,mean);
      }
    }
    printf("Total variance = %e\n",totVar);
    printAsterisks(PL_INFO, 0);

    //**/ generate graphics
    if (plotScilab())
    {
      fp = fopen("scilabrsvce2_bin.sci", "w");
      if (fp == NULL) 
           printf("ERROR: Cannot open file scilabvce2b_bin.sci\n");
      else
      {
        fprintf(fp,
              "// This file contains 2nd order sensitivity indices\n");
        fprintf(fp,"// set sortFlag = 1 and set nn to be the number\n");
        fprintf(fp,"// of inputs to display.\n");
      }
    }
    else
    {
      fp = fopen("matlabrsvce2_bin.m", "w");
      if (fp == NULL) 
           printf("ERROR: Cannot open file matlabrsvce2_bin.sci\n");
      else
      {
        fprintf(fp,
              "%% This file contains 2nd order sensitivity indices\n");
        fprintf(fp,"%% set sortFlag = 1 and set nn to be the number\n");
        fprintf(fp,"%% of inputs to display.\n");
      }
    }
    if (fp != NULL) 
    {
      fprintf(fp, "sortFlag = 0;\n");
      fprintf(fp, "nn = %d;\n", nInputs);
      fprintf(fp, "Means = [\n");
      for (ii = 0; ii < nInputs*nInputs; ii++) 
        fprintf(fp,"%24.16e\n", vecIeStore[numBS*nInputs*nInputs+ii]);
      fprintf(fp, "];\n");
      fprintf(fp, "Stds = [\n");
      for (ii = 0; ii < nInputs*nInputs; ii++) 
        fprintf(fp,"%24.16e\n",vecIeStore[(numBS+1)*nInputs*nInputs+ii]);
      fprintf(fp, "];\n");
      if (inputNames == NULL)
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
          if (inputNames[ii] != NULL) 
               fprintf(fp,"'%s',",inputNames[ii]);
          else fprintf(fp,"'X%d',",ii+1);
        }
        if (plotScilab()) 
        {
          if (inputNames[nInputs-1] != NULL)
               fprintf(fp,"'%s'];\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'];\n",nInputs);
        }
        else
        {
          if (inputNames[nInputs-1] != NULL)
               fprintf(fp,"'%s'};\n",inputNames[nInputs-1]);
          else fprintf(fp,"'X%d'};\n",nInputs);
        }
      }
      fwriteHold(fp, 0);
      fprintf(fp, "ymin = min(Means-Stds);\n");
      fprintf(fp, "ymax = max(Means+Stds);\n");
      fprintf(fp, "h2 = 0.05 * (ymax - ymin);\n");
      if (plotScilab())
      {
        fprintf(fp, "nn    = %d;\n",nInputs);
        fprintf(fp, "Means = matrix(Means, nn, nn);\n");
        fprintf(fp, "Means = Means';\n");
        fprintf(fp, "Stds  = matrix(Stds, nn, nn);\n");
        fprintf(fp, "Stds  = Stds';\n");
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
        fprintf(fp, "nn    = %d;\n",nInputs);
        fprintf(fp, "Means = reshape(Means, nn, nn);\n");
        fprintf(fp, "Means = Means';\n");
        fprintf(fp, "Stds  = reshape(Stds, nn, nn);\n");
        fprintf(fp, "Stds  = Stds';\n");
        fprintf(fp, "hh = bar3(Means,0.8);\n");
        fprintf(fp, "alpha = 0.2;\n");
        fprintf(fp, "set(hh,'FaceColor','b','facea',alpha);\n");
        fprintf(fp, "Lstds = Means - Stds;\n");
        fprintf(fp, "Ustds = Means + Stds;\n");
        fprintf(fp, "[X,Y] = meshgrid(1:nn,1:nn);\n");
        fwriteHold(fp, 1);
        fprintf(fp, "for k = 1:nn\n");
        fprintf(fp, "  for l = k:nn\n");
        fprintf(fp, "    mkl = Means(k,l);\n");
        fprintf(fp, "    ukl = Ustds(k,l);\n");
        fprintf(fp, "    lkl = Lstds(k,l);\n");
        fprintf(fp, "    if (mkl > .02 & (ukl-lkl)/mkl > .02)\n");
        fprintf(fp, "      xkl = [X(k,l), X(k,l)];\n");
        fprintf(fp, "      ykl = [Y(k,l), Y(k,l)];\n");
        fprintf(fp, "      zkl = [lkl, ukl];\n");
        fprintf(fp, "      plot3(xkl,ykl,zkl,'-mo',...\n");
        fprintf(fp, "        'LineWidth',5,'MarkerEdgeColor','k',...\n");
        fprintf(fp, "        'MarkerFaceColor','k','MarkerSize',10);\n");
        fprintf(fp, "    end\n");
        fprintf(fp, "  end\n");
        fprintf(fp, "end\n");
        fwriteHold(fp, 0);
        fprintf(fp, "axis([0.5 nn+0.5 0.5 nn+0.5 0 ymax])\n");
        fprintf(fp, "set(gca,'XTickLabel',Str);\n");
        fprintf(fp, "set(gca,'YTickLabel',Str);\n");
        fprintf(fp, "set(gca, 'fontsize', 12)\n");
        fprintf(fp, "set(gca, 'fontweight', 'bold')\n");
        fprintf(fp, "set(gca, 'linewidth', 2)\n");
      }
      fwritePlotAxes(fp);
      fwritePlotTitle(fp,
            "1st+2nd Order Sensitivity Indices (with bootstrap)");
      fwritePlotZLabel(fp,"2nd Order Sensitivity Indices (Normalized)");
      fwritePlotXLabel(fp,"Inputs");
      fwritePlotYLabel(fp,"Inputs");
      fclose(fp);
      if (plotScilab())
           printf("rsvce2_bin plot file = scilabrsvce2_bin.sci\n");
      else printf("rsvce2_bin plot file = matlabrsvce2_bin.m\n");
    }
    faPtr = NULL;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsua_sa or aeua   
  //**/ UQ for aleatoric and epistemic UQ using inner-outer iteration
  //**/ -------------------------------------------------------------
  else if ((!strcmp(command, "rsua_ae")) || (!strcmp(command, "aeua")))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsua_ae: UQ for aleatoric-epistemic uncertainty analysis\n");
      printf("Syntax: rsua_ae (no argument needed).\n");
      return 0;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command performs uncertainty analysis ");
    printf("on the response surface\n");
    printf("constructed from the loaded sample whereby ");
    printf("some of the sample inputs\n");
    printf("are aleatory (with prescribed distributions) ");
    printf("and some are epistemic\n");
    printf("(with prescribed ranges). These distributions ");
    printf("and ranges are taken\n");
    printf("from the loaded sample file. The result is a ");
    printf("bundle of CDFs each\n");
    printf("representing the output distribution based ");
    printf("on the aleatory inputs\n");
    printf("at some given epistemic input values. This bundle ");
    printf("of CDFs can be\n");
    printf("enveloped and the envelop is called a p-box.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0 || psuadeIO_ == NULL || nSamples_ <= 0)
    {
      printf("ERROR: Sample data have not been loaded.\n");
      return 1;
    }
    if (nInputs_ == 1)
    {
      printf("ERROR: nInputs must be 2 or more.\n");
      return 1;
    }

    //**/ select output
    snprintf(pString,100,"Enter output number (1 - %d) : ",nOutputs_);
    outputID = getInt(1, nOutputs_, pString);
    outputID--;

    //**/ select epistemic uncertain parameters
    printf("Step 1: Select aleatoric and epistemic parameters\n");
    int nEpistemic=0;
    psIVector vecUTypes;
    vecUTypes.setLength(nInputs_);
    //**/ set all parameters to be aleatoric
    for (ii = 0; ii < nInputs_; ii++) vecUTypes[ii] = 0;
    kk = 1;
    while (kk > 0)
    {
      snprintf(pString,100,
              "Select epistemic parameters (1 - %d, 0 if done) : ",
              nInputs_); 
      kk = getInt(0, nInputs_, pString);
      if (kk > 0)
      {
        vecUTypes[kk-1] = 1;
        nEpistemic++;
      }
    } 
    if (nEpistemic == 0 || nEpistemic == nInputs_)
    {
      printf("At least 1 and at most %d epistemic parameters are\n",
             nInputs_-1);
      printf("required for this command.\n");
      return 0;
    }
    printf("You have specified %d epistemic parameters.\n",nEpistemic);

    //**/ set up the function approximator
    printf("Step 2: Construct response surface\n");
    psuadeIO_->getParameter("ana_outputid", pPtr);
    int iSave = pPtr.intData_;
    psuadeIO_->updateAnalysisSection(-1,-1,-1,-1,outputID,-1);
    faType = -1;
    FuncApprox *faPtr = genFA(faType,nInputs_,outputLevel_,nSamples_);
    if (faPtr == NULL) 
    {
      printf("ERROR: Cannot construct response surface.\n"); 
      return 0;
    }
    int nPtsPerDim = 64;
    psVector vecYT;
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(VecILowerBs_.getDVector(),VecIUpperBs_.getDVector());
    faPtr->setOutputLevel(outputLevel_);
    vecYT.setLength(nSamples_);
    for (ii = 0; ii < nSamples_; ii++) 
      vecYT[ii] = VecSamOutputs_[ii*nOutputs_+outputID];
    status = faPtr->initialize(VecSamInputs_.getDVector(),vecYT.getDVector());

    //**/ generate a sample for outer iteration (uniform)
    printf("Step 3: Construct CDFs via outer-inner iterations\n");
    printf("   Outer iteration: iterate on the epistemic samples\n");
    printf("   Inner iteration: iterate on the aleatoric samples\n");
    printf("   Objective: iterate 1000 times in the outer loop and ");
    printf("look for the\n");
    printf("              lower and upper edges of the p-box.\n");
    int nSams=20000;
    psIVector vecPDFs1;
    psVector  vecStds1, vecMeans1, vecSamInps, veclbs, vecubs;
    psVector  vecLBs, vecUBs;
    psMatrix  corMat1;

    vecPDFs1.setLength(nEpistemic);
    vecMeans1.setLength(nEpistemic);
    vecStds1.setLength(nEpistemic);
    veclbs.setLength(nEpistemic);
    vecubs.setLength(nEpistemic);
    ddata = 1.0;
    kk = 0;
    corMat1.setDim(nEpistemic,nEpistemic);
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (vecUTypes[ii] == 1)
      {
        corMat1.setEntry(kk,kk,ddata);
        vecPDFs1[kk] = VecInpPDFs_[ii]; 
        vecMeans1[kk] = VecInpMeans_[ii]; 
        vecStds1[kk] = VecInpStds_[ii]; 
        veclbs[kk] = VecILowerBs_[ii];
        vecubs[kk] = VecIUpperBs_[ii];
        kk++;
      }
    }
    PDFManager *pdfman = new PDFManager();
    pdfman->initialize(nEpistemic,vecPDFs1.getIVector(),
            vecMeans1.getDVector(),vecStds1.getDVector(),corMat1,
            SamPDFFiles_,VecSamPDFIndices_.getIVector());
    vecLBs.load(nEpistemic, veclbs.getDVector());
    vecUBs.load(nEpistemic, vecubs.getDVector());
    vecSamInps.setLength(nSams*nEpistemic);
    pdfman->genSample(nSams, vecSamInps, vecLBs, vecUBs);
    delete pdfman;

    //**/ generate a sample for the inner iterations
    int nAleatoric = nInputs_ - nEpistemic, nSams2=2000;
    psIVector vecPDFs2;
    psVector  vecStds2, vecMeans2, vecInps;
    psMatrix  corMat2;

    vecPDFs2.setLength(nAleatoric);
    vecMeans2.setLength(nAleatoric);
    vecStds2.setLength(nAleatoric);
    veclbs.setLength(nAleatoric);
    vecubs.setLength(nAleatoric);
    ddata = 1.0;
    kk = 0;
    corMat2.setDim(nAleatoric,nAleatoric);
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (vecUTypes[ii] != 1)
      {
        corMat2.setEntry(kk,kk,ddata);
        vecPDFs2[kk] = VecInpPDFs_[ii]; 
        vecMeans2[kk] = VecInpMeans_[ii]; 
        vecStds2[kk] = VecInpStds_[ii]; 
        veclbs[kk] = VecILowerBs_[ii];
        vecubs[kk] = VecIUpperBs_[ii];
        kk++;
      }
    }
    pdfman = new PDFManager();
    pdfman->initialize(nAleatoric,vecPDFs2.getIVector(),
            vecMeans2.getDVector(),vecStds2.getDVector(),corMat2,
            SamPDFFiles_,VecSamPDFIndices_.getIVector());
    vecLBs.load(nAleatoric, veclbs.getDVector());
    vecUBs.load(nAleatoric, vecubs.getDVector());
    vecInps.setLength(nSams2*nAleatoric);
    pdfman->genSample(nSams2, vecInps, vecLBs, vecUBs);
    delete pdfman;

    if (plotScilab()) fp = fopen("scilabrsua_ae.sci", "w");
    else              fp = fopen("matlabrsua_ae.m", "w");
    if (fp == NULL)
    {
      printf("rsua_ae ERROR: Cannot open plot file.\n");
      return 1;
    }
    fwritePlotCLF(fp);

    //**/ outer iteration
    psVector vecYmaxs, vecYmins, vec1Sample;
    vec1Sample.setLength(nInputs_);
    vecYmaxs.setLength(nSams2);
    vecYmins.setLength(nSams2);
    for (ss = 0; ss < nSams2; ss++)
    {
      vecYmaxs[ss] = -PSUADE_UNDEFINED;
      vecYmins[ss] =  PSUADE_UNDEFINED;
    }
    int count, converged = 0;
    int upperCnt, lowerCnt, convergedCnt=0;
    double upperAcc, lowerAcc, dtemp;
    ss = 0;
    while (ss < nSams && (converged == 0 || ss < 1000))
    {
      if (outputLevel_ > 2) printf("Epistemic sample %d\n",ss+1);
      if (ss < 50) fprintf(fp, "Y%d = [\n",ss+1);
      if (ss > 0 && (ss % 100 == 0))
      {
        printf("%5.1f%% ", 0.1*ss);
        fflush(stdout);
      }
      int count2 = 0;
      for (ii = 0; ii < nInputs_; ii++)
      {
        if (vecUTypes[ii] == 1)
        {
          vec1Sample[ii] = vecSamInps[ss*nEpistemic+count2];
          count2++;
        }
      }
      count = lowerCnt = upperCnt = 0;
      upperAcc = lowerAcc = 0;
      for (kk = 0; kk < nSams2; kk++)
      {
        int flag = 0;
        for (ii = 0; ii < nAleatoric; ii++)
        {
          if (vecInps[kk*nAleatoric+ii] < veclbs[ii] ||
              vecInps[kk*nAleatoric+ii] > vecubs[ii]) flag++;
        }
        if (flag == 0)
        {
          count2 = 0;
          for (ii = 0; ii < nInputs_; ii++)
          {
            if (vecUTypes[ii] == 0)
            {
              vec1Sample[ii] = vecInps[kk*nAleatoric+count2];
              count2++;
            }
          }
          dtemp = faPtr->evaluatePoint(vec1Sample.getDVector());
          if (ss < 50) fprintf(fp, "%e\n", dtemp);
          if (dtemp > vecYmaxs[kk]) 
          {
            if (vecYmaxs[kk] != 0) 
              upperAcc += PABS((vecYmaxs[kk]-dtemp)/vecYmaxs[kk]);
            vecYmaxs[kk] = dtemp;
            upperCnt++;
          }
          if (dtemp < vecYmins[kk])
          {
            if (vecYmins[kk] != 0) 
              lowerAcc += PABS((vecYmins[kk]-dtemp)/vecYmins[kk]);
            vecYmins[kk] = dtemp;
            lowerCnt++;
          }
          count++;
        }
      }
      ddata = 100.0 * upperAcc / nSams;
      if (outputLevel_ > 2 && ddata > 0.0 && ss > 0) 
        printf("  Upper lifted  %7.3f %% on average from last time\n",
               ddata);
      ddata = 100.0 * lowerAcc / nSams;
      if (outputLevel_ > 2 && ddata > 0.0 && ss > 0) 
        printf("  Lower dropped %7.3f %% on average from last time\n",
               ddata);
      if (upperCnt+lowerCnt == 0) convergedCnt++;
      else                        convergedCnt = converged = 0;
      if (outputLevel_ > 3) 
        printf("  Convergence indicator = %5d (20 times => converged)\n",
               upperCnt+lowerCnt);
      if (convergedCnt >= 20) converged = 1;
      if (count < 0.5 * nSams2)
      {
        printf("WARNING: < half of the points are within bounds.\n");
        printf("         Input ranges may need to be widened.\n");
      }
      if (count == 0)
      {
        printf("ERROR: None of the sample points are within bounds.\n");
        return 1;
      }
      if (ss < 50) fprintf(fp, "];\n");
      if (ss < 50) 
      {
        if (plotScilab()) 
             fprintf(fp,"Y%d = gsort(Y%d,'g','i');\n",ss+1,ss+1);
        else fprintf(fp,"Y%d = sort(Y%d);\n",ss+1,ss+1);
        fprintf(fp, "X = 1:%d;\n",count);
        fprintf(fp, "X = X' / %d;\n", count);
        fprintf(fp, "plot(Y%d,X)\n",ss+1);
        fprintf(fp, "drawnow\n");
        if (ss == 0)
        {
          snprintf(winput,100,"Cumulative Distributions");
          fwritePlotTitle(fp, winput);
          fwritePlotAxes(fp);
          if (StrOutNames_[outputID] != NULL) 
               snprintf(winput,100,"%s",StrOutNames_[outputID]);
          else snprintf(winput,100,"Output Values");
          fwritePlotXLabel(fp, winput);
          snprintf(winput,100,"Probabilities");
          fwritePlotYLabel(fp, winput);
          if (plotScilab())
               fprintf(fp, "set(gca(),\"auto_clear\",\"off\")\n");
          else fprintf(fp, "hold on\n");
        }
      }
      ss++;
    }
    printf("\n");
    count = 0;
    fprintf(fp, "YU = [\n");
    for (ss = 0; ss < nSams2; ss++)
    {
      if (vecYmaxs[ss] > -PSUADE_UNDEFINED) 
      {
        fprintf(fp, "%e\n", vecYmaxs[ss]);
        count++;
      }
    }
    fprintf(fp, "];\n");
    if (plotScilab()) 
         fprintf(fp,"YU = gsort(YU,'g','i');\n");
    else fprintf(fp,"YU = sort(YU);\n");
    fprintf(fp, "X = 1:%d;\n",count);
    fprintf(fp, "X = X' / %d;\n", count);
    fprintf(fp, "plot(YU,X,'r-','lineWidth',3)\n");

    count = 0;
    fprintf(fp, "YL = [\n");
    for (ss = 0; ss < nSams2; ss++)
    {
      if (vecYmins[ss] < PSUADE_UNDEFINED) 
      {
        fprintf(fp, "%e\n", vecYmins[ss]);
        count++;
      }
    }
    fprintf(fp, "];\n");
    if (plotScilab()) 
         fprintf(fp,"YL = gsort(YL,'g','i');\n");
    else fprintf(fp,"YL = sort(YL);\n");
    fprintf(fp, "X = 1:%d;\n",count);
    fprintf(fp, "X = X' / %d;\n", count);
    fprintf(fp, "plot(YL,X,'r-','lineWidth',3)\n");
    fclose(fp);
    printf("Plot file for aleatoric-epistemic analysis = ");
    if (plotScilab()) printf("scilabrsua_ae.sci.\n");
    else              printf("matlabrsua_ae.m.\n");
    psuadeIO_->updateAnalysisSection(-1,-1,-1,-1,iSave,-1);
    delete faPtr;
    faPtr = NULL;
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsentropy 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsentropy"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsentropy: compute total output entropy");
      printf("Syntax: rsentropy (no argument needed).\n");
      return 0;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command computes total output entropy ");
    printf("from the response surface\n");
    printf("constructed from the loaded sample.\n");
    if (nSamples_ > 0 && nSamples_ > 10000)
    {
      printf("The loaded sample has sample size > 10k, which ");
      printf("may be computationally\n");
      printf("prohibitive for most response surface methods. ");
    }
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0 || psuadeIO_ == NULL || nSamples_ <= 0)
    {
      printf("ERROR: Sample data have not been loaded.\n");
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ select output
    //**/ -----------------------------------------------------
    snprintf(pString,100,
             "Enter output number (1 - %d) : ",nOutputs_);
    outputID = getInt(1, nOutputs_, pString);
    outputID--;

    psVector vecY;
    vecY.setLength(nSamples_);
    for (ss = 0; ss < nSamples_; ss++)
      vecY[ss] = VecSamOutputs_[ss*nOutputs_+outputID];
    aData adata;
    adata.nInputs_  = nInputs_;
    adata.nOutputs_ = 1;
    adata.nSamples_ = nSamples_;
    adata.iLowerB_  = VecILowerBs_.getDVector();
    adata.iUpperB_  = VecIUpperBs_.getDVector(); 
    adata.sampleInputs_  = VecSamInputs_.getDVector();
    adata.sampleOutputs_ = vecY.getDVector();
    adata.outputID_   = 0;
    adata.printLevel_ = outputLevel_;
    adata.ioPtr_ = psuadeIO_;
    RSMEntropy1Analyzer analyzer;
    snprintf(pString, 22, "ana_entropy1_tentropy");
    char *targv[1];
    targv[0] = (char *) pString;
    analyzer.setParam(1, targv); 
    analyzer.analyze(adata);
    printf("rsentropy results:\n");
    printDashes(PL_INFO, 0);
    printf("Total output entropy  = %e\n",
           analyzer.get_outputEntropy());
    printEquals(PL_INFO, 0);
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsentropy1 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsentropy1"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsentropy1: entropy-based sensitivity analysis");
      printf("Syntax: rsentropy1 (no argument needed).\n");
      return 0;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command performs first-order sensitivity ");
    printf("analysis using entropy.\n");
    if (nSamples_ > 0 && nSamples_ > 10000)
    {
      printf("The loaded sample has sample size > 10k, which ");
      printf("may be computationally\n");
      printf("prohibitive for most response surface methods.\n");
    }
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0 || psuadeIO_ == NULL || nSamples_ <= 0)
    {
      printf("ERROR: Sample data have not been loaded.\n");
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ select output
    //**/ -----------------------------------------------------
    snprintf(pString,100,"Enter output number (1 - %d) : ",
             nOutputs_);
    outputID = getInt(1, nOutputs_, pString);
    outputID--;

    psVector vecY;
    vecY.setLength(nSamples_);
    for (ss = 0; ss < nSamples_; ss++)
      vecY[ss] = VecSamOutputs_[ss*nOutputs_+outputID];
    aData adata;
    adata.nInputs_  = nInputs_;
    adata.nOutputs_ = 1;
    adata.nSamples_ = nSamples_;
    adata.iLowerB_  = VecILowerBs_.getDVector();
    adata.iUpperB_  = VecIUpperBs_.getDVector(); 
    adata.sampleInputs_  = VecSamInputs_.getDVector();
    adata.sampleOutputs_ = vecY.getDVector();
    adata.outputID_   = 0;
    adata.printLevel_ = outputLevel_;
    adata.ioPtr_ = psuadeIO_;
    RSMEntropy1Analyzer analyzer;
    snprintf(pString, 21, "ana_entropy1_entropy");
    char *targv[1];
    targv[0] = (char *) pString;
    analyzer.setParam(1, targv); 
    analyzer.analyze(adata);
    printf("rsentropy1 results:\n");
    printDashes(PL_INFO, 0);
    double totEnt = analyzer.get_outputEntropy();
    for (ii = 0; ii < nInputs_; ii++)
    {
      ddata = analyzer.get_entropy1(ii);
      if (ddata < 0) ddata = 0;
      if (totEnt != 0)
        printf("Input %3d sensitivity = %10.3e (scaled=%10.3e)\n",
               ii+1,ddata,ddata/PABS(totEnt));
      else
        printf("Input %3d sensitivity = %10.3e\n",ii+1,ddata);
    }
    printf("Sensitivity measure = H(Y) - H(Y|X) (also = M.I.)\n"); 
    printf("Scaled measure = 1 - H(Y|X)/H(Y) if H(Y) != 0\n"); 
    printf("Total output entropy  = %e\n",totEnt);
    printEquals(PL_INFO, 0);
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsdelta1 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsdelta1"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsdelta1: expected output variation induced by ");
      printf("individual inputs\n");
      printf("Syntax: rsdelta1 (no argument needed).\n");
      return 0;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command computes average output ");
    printf("variation when varying individual\n");
    printf("inputs. As such, this measure may be useful when ");
    printf("input distribution is\n");
    printf("unknown, and thus non-informative distribution ");
    printf("should be assumed. This\n");
    printf("is done by, instead of using the distribution of the ");
    printf("output Y, finding\n");
    printf("the difference of the maximum and minimum of Y induced ");
    printf("by an input\n");
    printf("in the presence of variation in all other inputs. ");
    printf("Mathematically, this\n");
    printf("measure is Delta(E[Y|X]). Large measure for ");
    printf("a given input means that\n");
    printf("this input is important.\n"); 
    printDashes(PL_INFO, 0);
    if (nSamples_ >= 10000)
    {
      printf("\nWARNING: \n");
      printf("The loaded sample has sample size >= 10k, which ");
      printf("may be computationally\n");
      printf("prohibitive for most response surface methods. ");
    }

    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0 || psuadeIO_ == NULL || nSamples_ <= 0)
    {
      printf("ERROR: Sample data have not been loaded.\n");
      return 1;
    }
    //**/ -----------------------------------------------------
    //**/ select output
    //**/ -----------------------------------------------------
    snprintf(pString,100,
             "Enter output number (1 - %d) : ",nOutputs_);
    outputID = getInt(1, nOutputs_, pString);
    outputID--;

    psVector vecY;
    vecY.setLength(nSamples_);
    for (ss = 0; ss < nSamples_; ss++)
      vecY[ss] = VecSamOutputs_[ss*nOutputs_+outputID];
    aData adata;
    adata.nInputs_  = nInputs_;
    adata.nOutputs_ = 1;
    adata.nSamples_ = nSamples_;
    adata.iLowerB_  = VecILowerBs_.getDVector();
    adata.iUpperB_  = VecIUpperBs_.getDVector(); 
    adata.sampleInputs_  = VecSamInputs_.getDVector();
    adata.sampleOutputs_ = vecY.getDVector();
    adata.outputID_   = 0;
    adata.printLevel_ = outputLevel_;
    adata.ioPtr_ = psuadeIO_;
    RSMEntropy1Analyzer analyzer;
    snprintf(pString, 19, "ana_entropy1_delta");
    char *targv[1];
    targv[0] = (char *) pString;
    analyzer.setParam(1, targv); 
    analyzer.analyze(adata);
    printf("rsdelta1 results: (D(E[Y|X]))\n"); 
    printDashes(PL_INFO, 0);
    for (ii = 0; ii < nInputs_; ii++)
    {
      ddata = analyzer.get_delta1(ii);
      printf("Input %3d sensitivity = %9.3e\n",ii+1,ddata);
    }
    printf("Total output delta  = %e\n",
           analyzer.get_outputDelta());
    printEquals(PL_INFO, 0);
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsentropy2 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsentropy2"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsentropy2: 2nd-order sensitivity analysis using ");
      printf("entropy measure\n");
      printf("Syntax: rsentropy2 (no argument needed).\n");
      return 0;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command performs input pair sensitivity ");
    printf("analysis based on entropy\n");
    if (nSamples_ < 20000)
    {
      printf("measure using the response surface ");
      printf("constructed from the loaded sample.\n");
    }
    else
    {
      printf("measure operating directly on the loaded sample.\n");
    }
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0 || psuadeIO_ == NULL || nSamples_ <= 0)
    {
      printf("ERROR: Sample data have not been loaded.\n");
      return 1;
    }
    if (nInputs_ <= 2)
    {
      printf("ERROR: This command is for nInputs > 2 only.\n");
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ select output
    //**/ -----------------------------------------------------
    snprintf(pString,100,
             "Enter output number (1 - %d) : ", nOutputs_);
    outputID = getInt(1, nOutputs_, pString);
    outputID--;

    psVector vecY;
    vecY.setLength(nSamples_);
    for (ss = 0; ss < nSamples_; ss++)
      vecY[ss] = VecSamOutputs_[ss*nOutputs_+outputID];
    aData adata;
    adata.nInputs_  = nInputs_;
    adata.nOutputs_ = 1;
    adata.nSamples_ = nSamples_;
    adata.iLowerB_  = VecILowerBs_.getDVector();
    adata.iUpperB_  = VecIUpperBs_.getDVector(); 
    adata.sampleInputs_  = VecSamInputs_.getDVector();
    adata.sampleOutputs_ = vecY.getDVector();
    adata.outputID_   = 0;
    adata.printLevel_ = outputLevel_;
    adata.ioPtr_ = psuadeIO_;
    RSMEntropy2Analyzer analyzer;
    snprintf(pString, 21, "ana_entropy2_entropy");
    char *targv[1];
    targv[0] = (char *) pString;
    analyzer.setParam(1, targv); 
    analyzer.analyze(adata);
    printf("rsentropy2 results:\n");
    printDashes(PL_INFO, 0);
    double totEnt = analyzer.get_outputEntropy();
    for (ii = 0; ii < nInputs_; ii++)
    {
      for (jj = ii+1; jj < nInputs_; jj++)
      {
        ddata = analyzer.get_entropy2(ii,jj);
        if (ddata < 0) ddata = 0;
        if (totEnt != 0)
          printf("Input (%3d,%3d) sensitivity = %10.3e (scaled=%10.3e)\n",
                 ii+1,jj+1,ddata,ddata/PABS(totEnt));
        else
          printf("Input (%3d,%3d) sensitivity = %10.3e\n",ii+1,
                 jj+1,ddata);
      }
    }
    printf("Sensitivity measure = H(Y) - H(Y|X) (also = M.I.)\n"); 
    printf("Scaled measure = 1 - H(Y|X)/H(Y) if H(Y) != 0\n"); 
    printf("Total output entropy = %e \n",
           analyzer.get_outputEntropy());
    printEquals(PL_INFO, 0);
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsdelta2 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsdelta2"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsdelta2: expected output variation induced by ");
      printf("pairs of inputs\n");
      printf("Syntax: rsdelta2 (no argument needed).\n");
      return 0;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command computes average output ");
    printf("variation when varying all pairs\n");
    printf("of inputs.  As such, this measure is ");
    printf("useful when input distributions\n");
    printf("are unknown, and thus non-informative ");
    printf("distribution is assumed. This\n");
    printf("is done by, instead of using the ");
    printf("distribution of the output Y, finding\n");
    printf("the difference of the maximum and minimum ");
    printf("of Y induced by inputs X1,X2\n");
    printf("in the presence of variation in all other ");
    printf("inputs. Mathematically, this\n");
    printf("measure is Delta(E[Y|X]). Large measure ");
    printf("for a given input means that\n");
    printf("this input is important.\n"); 
    printDashes(PL_INFO, 0);
    if (nSamples_ >= 10000)
    {
      printf("\nWARNING: \n");
      printf("The loaded sample has sample size >= 10k, which ");
      printf("may be computationally\n");
      printf("prohibitive for most response surface methods. ");
    }

    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0 || psuadeIO_ == NULL || nSamples_ <= 0)
    {
      printf("ERROR: Sample data have not been loaded.\n");
      return 1;
    }
    //**/ -----------------------------------------------------
    //**/ select output
    //**/ -----------------------------------------------------
    snprintf(pString,100,
             "Enter output number (1 - %d) : ",nOutputs_);
    outputID = getInt(1, nOutputs_, pString);
    outputID--;

    psVector vecY;
    vecY.setLength(nSamples_);
    for (ss = 0; ss < nSamples_; ss++)
      vecY[ss] = VecSamOutputs_[ss*nOutputs_+outputID];
    aData adata;
    adata.nInputs_  = nInputs_;
    adata.nOutputs_ = 1;
    adata.nSamples_ = nSamples_;
    adata.iLowerB_  = VecILowerBs_.getDVector();
    adata.iUpperB_  = VecIUpperBs_.getDVector(); 
    adata.sampleInputs_  = VecSamInputs_.getDVector();
    adata.sampleOutputs_ = vecY.getDVector();
    adata.outputID_   = 0;
    adata.printLevel_ = outputLevel_;
    adata.ioPtr_ = psuadeIO_;
    RSMEntropy2Analyzer analyzer;
    snprintf(pString, 19, "ana_entropy2_delta");
    char *targv[1];
    targv[0] = (char *) pString;
    analyzer.setParam(1, targv); 
    analyzer.analyze(adata);
    printf("rsdelta2 results: (D(E[Y|X]))\n"); 
    printDashes(PL_INFO, 0);
    for (ii = 0; ii < nInputs_; ii++)
    {
      for (jj = ii+1; jj < nInputs_; jj++)
      {
        printf("Input (%3d,%3d) sensitivity = %10.3e\n",
               ii+1,jj+1,analyzer.get_delta2(ii,jj));
      }
    }
    printf("Overall output Ymax-Ymin = %e\n",
           analyzer.get_outputDelta());
    printEquals(PL_INFO, 0);
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsentropyg 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsentropyg"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsentropyg: group-order sensitivity analysis using ");
      printf("entropy measure\n");
      printf("Syntax: rsentropyg (no argument needed).\n");
      return 0;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command performs group-order sensitivity ");
    printf("analysis based on entropy\n");
    if (nSamples_ < 20000)
    {
      printf("measure using the response surface ");
      printf("constructed from the loaded sample.\n");
    }
    else
    {
      printf("measure operating directly on the loaded sample.\n");
    }
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0 || psuadeIO_ == NULL || nSamples_ <= 0)
    {
      printf("ERROR: Sample data have not been loaded.\n");
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ select output
    //**/ -----------------------------------------------------
    snprintf(pString,100,
             "Enter output number (1 - %d) : ", nOutputs_);
    outputID = getInt(1, nOutputs_, pString);
    outputID--;

    psVector vecY;
    vecY.setLength(nSamples_);
    for (ss = 0; ss < nSamples_; ss++)
      vecY[ss] = VecSamOutputs_[ss*nOutputs_+outputID];
    aData adata;
    adata.nInputs_  = nInputs_;
    adata.nOutputs_ = 1;
    adata.nSamples_ = nSamples_;
    adata.iLowerB_  = VecILowerBs_.getDVector();
    adata.iUpperB_  = VecIUpperBs_.getDVector(); 
    adata.sampleInputs_  = VecSamInputs_.getDVector();
    adata.sampleOutputs_ = vecY.getDVector();
    adata.outputID_   = 0;
    adata.printLevel_ = outputLevel_;
    adata.ioPtr_ = psuadeIO_;
    RSMEntropyGAnalyzer analyzer;
    snprintf(pString, 21, "ana_entropyg_entropy");
    char *targv[1];
    targv[0] = (char *) pString;
    analyzer.setParam(1, targv); 
    analyzer.analyze(adata);
    printf("rsentropyg results:\n"); 
    printDashes(PL_INFO, 0);
    double totEnt = analyzer.get_outputEntropy();
    int nGroups = analyzer.get_ngroups();
    for (ii = 0; ii < nGroups; ii++)
    {
      ddata = analyzer.get_entropyG(ii);
      if (ddata < 0) ddata = 0;
      if (totEnt != 0)
        printf("Group %3d Sensitivity = %10.3e (scaled=%10.3e)\n",
               ii+1,ddata,ddata/PABS(totEnt));
      else
        printf("Group %3d Sensitivity = %10.3e ",ii+1,ddata);
    }
    printf("Sensitivity measure = H(Y) - H(Y|X) (also = M.I.)\n"); 
    printf("Scaled measure = 1 - H(Y|X)/H(Y) if H(Y) != 0\n"); 
    printf("Total output entropy = %e \n",
           analyzer.get_outputEntropy());
    printEquals(PL_INFO, 0);
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsdeltag 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsdeltag"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsdeltag: group-order sensitivity analysis using ");
      printf("delta measure\n");
      printf("Syntax: rsdeltag (no argument needed).\n");
      return 0;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command computes average output ");
    printf("variation when varying groups of\n");
    printf("inputs. As such, this measure may be useful when ");
    printf("input distribution is\n");
    printf("unknown, and thus non-informative distribution ");
    printf("should be assumed. This\n");
    printf("is done by, instead of using the distribution of the ");
    printf("output Y, finding\n");
    printf("the difference of the maximum and minimum of Y induced ");
    printf("by a group\n");
    printf("in the presence of variation in all other inputs. ");
    printf("Mathematically, this\n");
    printf("measure is Delta(E[Y|X]). Large measure for ");
    printf("a given input means that\n");
    printf("this input is important.\n"); 
    printDashes(PL_INFO, 0);
    if (nSamples_ >= 10000)
    {
      printf("\nWARNING: \n");
      printf("The loaded sample has sample size >= 10k, which ");
      printf("may be computationally\n");
      printf("prohibitive for most response surface methods. ");
    }
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0 || psuadeIO_ == NULL || nSamples_ <= 0)
    {
      printf("ERROR: Sample data have not been loaded.\n");
      return 1;
    }

    //**/ -----------------------------------------------------
    //**/ select output
    //**/ -----------------------------------------------------
    snprintf(pString,100,
             "Enter output number (1 - %d) : ", nOutputs_);
    outputID = getInt(1, nOutputs_, pString);
    outputID--;

    psVector vecY;
    vecY.setLength(nSamples_);
    for (ss = 0; ss < nSamples_; ss++)
      vecY[ss] = VecSamOutputs_[ss*nOutputs_+outputID];
    aData adata;
    adata.nInputs_  = nInputs_;
    adata.nOutputs_ = 1;
    adata.nSamples_ = nSamples_;
    adata.iLowerB_  = VecILowerBs_.getDVector();
    adata.iUpperB_  = VecIUpperBs_.getDVector(); 
    adata.sampleInputs_  = VecSamInputs_.getDVector();
    adata.sampleOutputs_ = vecY.getDVector();
    adata.outputID_   = 0;
    adata.printLevel_ = outputLevel_;
    adata.ioPtr_ = psuadeIO_;
    RSMEntropyGAnalyzer analyzer;
    snprintf(pString, 21, "ana_entropyg_delta");
    char *targv[1];
    targv[0] = (char *) pString;
    analyzer.setParam(1, targv); 
    analyzer.analyze(adata);
    printf("rsdeltag results: (D(E[Y|X]))\n"); 
    printDashes(PL_INFO, 0);
    int nGroups = analyzer.get_ngroups();
    for (ii = 0; ii < nGroups; ii++)
    {
      ddata = analyzer.get_deltaG(ii);
      printf("Group %3d sensitivity = %10.3e\n",ii+1,ddata);
    }
    printf("Overall output Ymax-Ymin = %e\n",
           analyzer.get_outputDelta());
    printEquals(PL_INFO, 0);
    return 0;
  }

  //**/ -------------------------------------------------------------
  // +++ rsshapley1 
  //**/ -------------------------------------------------------------
  else if (!strcmp(command, "rsshapley1"))
  {
    sscanf(lineIn,"%s %s",command,winput);
    if (!strcmp(winput, "-h"))
    {
      printf("rsshapley1: Sensitivity analysis based on Shapley values\n");
      printf("Syntax: rsshapley1 (no argument needed).\n");
      return 0;
    }
    printAsterisks(PL_INFO, 0);
    printf("This command computes Shapley values as sensitivity ");
    printf("measures for each\n");
    printf("input using the response surface constructed from ");
    printf("the loaded sample.\n");
    printf("You can choose between 3 different cost functions, ");
    printf("namely,\n");
    printf("1. Sobol'-based main effect (VCE),\n");
    printf("2. Sobol'-based total effect (TSI), and\n");
    printf("3. Entropy-based single-parameter effect (Ent).\n");
    printf("The default is TSI-based method. To select others, ");
    printf("turn on analysis\n");
    printf("expert mode before running this command.\n");
    printf("NOTE: An advantage of variance-based Shapley values ");
    printf("as compared to\n");
    printf("      Sobol' indices is that the sum of Shapley ");
    printf("values for all inputs\n");
    printf("      is equal to the total variance. This makes ");
    printf("it easier to estimate\n");
    printf("      the percentage of total variance that can ");
    printf("be attributed to each\n");
    printf("      input.\n");
    printDashes(PL_INFO, 0);
    printf("Proceed ? (y or n to abort) ");
    scanf("%s", lineIn2);
    fgets(winput,5000,stdin);
    if (lineIn2[0] != 'y') return 0;

    if (nInputs_ <= 0 || psuadeIO_ == NULL || nSamples_ <= 0)
    {
      printf("ERROR: Sample data have not been loaded.\n");
      return 1;
    }
    snprintf(pString,100,"Enter output number (1 - %d) : ",nOutputs_);
    outputID = getInt(1, nOutputs_, pString);
    outputID--;

    psVector vecY;
    vecY.setLength(nSamples_);
    for (ss = 0; ss < nSamples_; ss++)
      vecY[ss] = VecSamOutputs_[ss*nOutputs_+outputID];
    aData adata;
    adata.nInputs_  = nInputs_;
    adata.nOutputs_ = 1;
    adata.nSamples_ = nSamples_;
    adata.iLowerB_  = VecILowerBs_.getDVector();
    adata.iUpperB_  = VecIUpperBs_.getDVector();
    adata.sampleInputs_  = VecSamInputs_.getDVector();
    adata.sampleOutputs_ = vecY.getDVector();
    adata.sampleStates_  = VecSamStates_.getIVector();
    adata.outputID_   = 0;
    adata.printLevel_ = outputLevel_;
    adata.ioPtr_ = psuadeIO_;
    ShapleyAnalyzer analyzer;
    analyzer.analyze(adata);
    return 0;
  }
  return 0;
}

