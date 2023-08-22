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
// Functions for the class PRSFuncApproxAnalyzer  
// AUTHOR : CHARLES TONG
// DATE   : 2014
// ************************************************************************
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"
#include "FuncApprox.h"
#include "PRSFuncApproxAnalyzer.h"
#include "PrintingTS.h"
#include "dtype.h"

#define PABS(x) (((x) > 0.0) ? (x) : -(x))

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
PRSFuncApproxAnalyzer::PRSFuncApproxAnalyzer(CommManager *comm) 
{
   //**/ default is MARS
   rsType_ = PSUADE_RS_MARS;
   commMgr_ = comm;
   if (comm == NULL)
   {
      printOutTS(PL_ERROR,"PRSFA ERROR: no communicator.\n");
      exit(1);
   }
   mypid_  = psCommMgr_->getPID();
   nprocs_ = psCommMgr_->getNumProcs();
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
PRSFuncApproxAnalyzer::~PRSFuncApproxAnalyzer() 
{ 
} 

// ************************************************************************ 
// perform analysis 
// ------------------------------------------------------------------------
double PRSFuncApproxAnalyzer::analyze(aData &adata)
{
   int        nInputs, nOutputs, nSamples, outputID, status,  ii;
   int        ss, ss2, count, nPtsPerDim=64, ranFlag=0, printLevel, wgtID;
   int        nSubSamples, *iArray, *iArray2, iOne=1, commLeng, commFlag;
   int        numCVGroups, pindex;
   double     ddata, ymax, ymin, *YLocal, *X, *Y, *X2, *Y2;
   double     *lower, *upper, *eArray, *YT, *WW, *wgts, sdata, *XX, *YY;
   double     cvErr1, cvErr1s, cvErr2, cvErr2s,cvMax, cvMaxs, cvMaxBase;
   double     cvMaxBases, maxBase, maxBases, *sArray, *sigmas, *ST;
   double     sumErr1, sumErr1s, sumErr2, sumErr2s, maxErr, maxErrs;
   double     sumErr11, sumErr11s, ymean, yvar, ydiff, ssum;
   char       commBuffer[1001], pString[500], winput[500];
   FILE       *fpData, *fpErr;
   FuncApprox *faPtr=NULL;

   //**/ ---------------------------------------------------------------
   //**/ extract data
   //**/ ---------------------------------------------------------------
   printLevel = adata.printLevel_;
   nInputs   = adata.nInputs_;
   nOutputs  = adata.nOutputs_;
   nSamples  = adata.nSamples_;
   lower     = adata.iLowerB_;
   upper     = adata.iUpperB_;
   X         = adata.sampleInputs_;
   Y         = adata.sampleOutputs_;
   outputID  = adata.outputID_;
   wgtID     = adata.regWgtID_;

   //**/ ---------------------------------------------------------------
   //**/ error checking and diagnostics
   //**/ ---------------------------------------------------------------
   if (nInputs <= 0 || nOutputs <= 0 || nSamples <= 0)
   {
      if (mypid_ == 0)
      {
         printOutTS(PL_ERROR,"RSAnalyzer ERROR: invalid arguments.\n");
         printOutTS(PL_ERROR,"   nInputs  = %d\n", nInputs);
         printOutTS(PL_ERROR,"   nOutputs = %d\n", nOutputs);
         printOutTS(PL_ERROR,"   nSamples = %d\n", nSamples);
      }
      return PSUADE_UNDEFINED;
   }
   status = 0;
   for (ss = 0; ss < nSamples; ss++)
      if (Y[nOutputs*ss+outputID] > 0.9*PSUADE_UNDEFINED) status = 1;
   if (status == 1)
   {
      if (mypid_ == 0)
      {
         printOutTS(PL_ERROR,"PRSA ERROR: Some outputs are undefined.\n");
         printOutTS(PL_ERROR,"     Prune the undefined sample points first.\n");
      }
      return PSUADE_UNDEFINED;
   }
   if (mypid_ == 0 && printLevel > 0)
   {
      printAsterisks(PL_INFO, 0);
      printThisFA(rsType_);
      printEquals(PL_INFO, 0);
   }
   
   //**/ ---------------------------------------------------------------
   //**/ copy the selected output data to a new array and find max
   //**/ ---------------------------------------------------------------
   YLocal = new double[nSamples];
   checkAllocate(YLocal, "YLocal in PRSFuncApprox::analyze");
   for (ss = 0; ss < nSamples; ss++) YLocal[ss] = Y[ss*nOutputs+outputID];
   ymax = 0.0;
   ymin = PSUADE_UNDEFINED;
   for (ss = 0; ss < nSamples; ss++)
   {
      if (PABS(YLocal[ss]) > ymax) ymax = PABS(YLocal[ss]);
      if (PABS(YLocal[ss]) < ymin) ymin = PABS(YLocal[ss]);
   }
   if (mypid_ == 0)
   {
      printAsterisks(PL_INFO,0);
      printOutTS(PL_INFO,"     Parallel Response Surface Analysis.\n");
      printEquals(PL_INFO,0);
      printOutTS(PL_INFO,"PRSA: Output ID = %d\n", outputID+1);
      printOutTS(PL_INFO,"PRSA: Output Maximum/Minimum = %14.6e %14.6e\n",ymax,ymin);
   }
   if (ymax == PSUADE_UNDEFINED)
   {
      if (mypid_ == 0)
      {
         printOutTS(PL_ERROR,"PRSA ERROR: some outputs are undefined.\n");
         printOutTS(PL_ERROR, "           Prune them first before analyze.\n");
      }
      delete [] YLocal;
      return PSUADE_UNDEFINED;
   }

   //**/ ---------------------------------------------------------------
   //**/ generate response surfaces 
   //**/ ---------------------------------------------------------------
   if (mypid_ == 0)
   {
      YT = new double[nSamples];
      checkAllocate(YT, "YT in PRSFuncApprox::analyze");
      faPtr = genFA(rsType_, nInputs, iOne, nSamples);
      if (faPtr == NULL)
      {
         printOutTS(PL_INFO,"PRSA INFO: cannot create response surface.\n");
         delete [] YLocal;
         delete faPtr;
         delete [] YT;
         strcpy(commBuffer,"psuadeError");
         commLeng = 12;
         commBuffer[commLeng-1] = '\0';
         commMgr_->bcast((void *) &commLeng, iOne, INT, 0);
         commMgr_->bcast((void *) commBuffer, commLeng, CHAR, 0);
         return PSUADE_UNDEFINED;
      }
      faPtr->setNPtsPerDim(nPtsPerDim);
      faPtr->setBounds(lower, upper);
      faPtr->setOutputLevel(printLevel);
      if (wgtID >= 0 && wgtID < nOutputs)
      {
         wgts = new double[nSamples];
         checkAllocate(wgts, "wgts in PRSFuncApprox::analyze");
         for (ss = 0; ss < nSamples; ss++) wgts[ss] = Y[ss*nOutputs+wgtID];
         faPtr->loadWeights(nSamples, wgts);
         delete [] wgts;
      }
      status = faPtr->initialize(X, YLocal);
      if (status != 0)
      {
         printOutTS(PL_ERROR,"PRSA ERROR: something wrong in FA initialize.\n");
         delete [] YLocal;
         delete faPtr;
         delete [] YT;
         strcpy(commBuffer,"psuadeError");
         commLeng = 12;
         commBuffer[commLeng-1] = '\0';
         commMgr_->bcast((void *) &commLeng, iOne, INT, 0);
         commMgr_->bcast((void *) commBuffer, commLeng, CHAR, 0);
         return PSUADE_UNDEFINED;
      }

      //**/ ------------------------------------------------------------
      //**/ check RS data against the original data if more than 1 level.
      //**/ For one level, a final check will be done later.
      //**/ ------------------------------------------------------------
      sumErr1  = sumErr2 = maxErr = sumErr1s = sumErr2s = maxErrs = 0.0;
      sumErr11 = sumErr11s = ydiff = 0.0;
      maxBase = maxBases = ymean = yvar = 0.0;
      faPtr->evaluatePoint(nSamples, X, YT);
      for (ss = 0; ss < nSamples; ss++)
      {
         ddata = PABS(YT[ss] - YLocal[ss]);
         if (YLocal[ss] != 0.0) sdata = ddata / PABS(YLocal[ss]);
         else                   sdata = ddata;
         sumErr1   += ddata;
         sumErr1s  += sdata;
         sumErr11  += (YT[ss] - YLocal[ss]);
         if (YLocal[ss] != 0.0)
            sumErr11s += (YT[ss] - YLocal[ss]) / PABS(YLocal[ss]);
         else
            sumErr11s += (YT[ss] - YLocal[ss]);
         if (ddata > maxErr ) {maxErr = ddata;  maxBase = PABS(YLocal[ss]);}
         if (sdata > maxErrs) {maxErrs = sdata; maxBases = PABS(YLocal[ss]);}
         sumErr2  += (ddata * ddata);
         sumErr2s += (sdata * sdata);
         ymean += YLocal[ss]; 
      }
      ymean /= (double) nSamples;
      for (ss = 0; ss < nSamples; ss++)
         yvar += (YLocal[ss] - ymean) * (YLocal[ss] - ymean);
      sumErr1   = sumErr1 / (double) nSamples;
      sumErr1s  = sumErr1s / (double) nSamples;
      sumErr11  = sumErr11 / (double) nSamples;
      sumErr11s = sumErr11s / (double) nSamples;
      sumErr2  = sqrt(sumErr2 / (double) nSamples);
      sumErr2s = sqrt(sumErr2s / (double) nSamples);
      printOutTS(PL_INFO,"PRSA: Interpolation error on training set \n");
      printOutTS(PL_INFO,"      avg error far from 0 ==> systematic bias.\n");
      printOutTS(PL_INFO,
         "      rms error large      ==> average   error large.\n");
      printOutTS(PL_INFO,
         "      max error large      ==> pointwise error large.\n");
      printOutTS(PL_INFO,
         "      R-square may not always be a reliable measure.\n");
      printOutTS(PL_INFO,"  avg error   = %11.3e (unscaled)\n", sumErr11);
      printOutTS(PL_INFO,"  avg error   = %11.3e (scaled)\n", sumErr11s);
      printOutTS(PL_INFO,"  rms error   = %11.3e (unscaled)\n", sumErr2);
      printOutTS(PL_INFO,"  rms error   = %11.3e (scaled)\n", sumErr2s);
      printOutTS(PL_INFO,"  max error   = %11.3e (unscaled, BASE=%9.3e)\n",
                 maxErr, maxBase);
      printOutTS(PL_INFO,"  max error   = %11.3e (  scaled, BASE=%9.3e)\n",
                 maxErrs, maxBases);
      printOutTS(PL_INFO,
         "  R-square    = %16.8e\n",1.0-sumErr2*sumErr2*nSamples/yvar);
      printOutTS(PL_INFO,"Based on %d training points.\n",nSamples);
      delete faPtr;
      delete [] YT;
      strcpy(commBuffer,"psuadeOkay");
      commLeng = 11;
      commBuffer[commLeng-1] = '\0';
      commMgr_->bcast((void *) &commLeng, iOne, INT, 0);
      commMgr_->bcast((void *) commBuffer, commLeng, CHAR, 0);
   }
   else
   {
      commMgr_->bcast((void *) &commLeng, iOne, INT, 0);
      commMgr_->bcast((void *) commBuffer, commLeng, CHAR, 0);
      if (!strcmp(commBuffer, "psuadeError")) return PSUADE_UNDEFINED;
   }

   //**/ ---------------------------------------------------------------
   //**/ cross validation
   //**/ ---------------------------------------------------------------
   if (rsType_ == PSUADE_RS_REGRGL || rsType_ == PSUADE_RS_SPLINES ||
       rsType_ == PSUADE_RS_REGSG) 
   {
      if (mypid_ == 0)
      {
         printOutTS(PL_INFO,"Cross validation (CV) cannot be performed on\n");
         printOutTS(PL_INFO,"  gradient-based Legendre polynomial, splines,\n");
         printOutTS(PL_INFO,"  and sparse grid response surface.\n");
      }
      return 0;
   }
   if (mypid_ == 0) 
   {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO,"Cross validation (CV) will be performed next. ");
      printOutTS(PL_INFO,"Since CV iterates as\n");
      printOutTS(PL_INFO,"many times as the number of groups, rs_expert ");
      printOutTS(PL_INFO,"mode will be turned off.\n");
      printOutTS(PL_INFO,"To change default parameters settings for your\n");
      printOutTS(PL_INFO,"selected response \n");
      printOutTS(PL_INFO,"surface, you will need to exit, create a config\n");
      printOutTS(PL_INFO,"file (use genconfigfile in command line mode), ");
      printOutTS(PL_INFO,"and turn on config \n");
      printOutTS(PL_INFO,"option in your data file.\n");
      printDashes(PL_INFO, 0);
      snprintf(pString,100,"Enter the number of groups to validate : (2 - %d) ",
               nSamples);
      numCVGroups = getInt(1, nSamples, pString);
      printOutTS(PL_INFO, "RSFA: number of CV groups = %d\n",numCVGroups);
      nSubSamples = nSamples / numCVGroups;
      if (nSubSamples * numCVGroups < nSamples)
      {
         numCVGroups++;
         printOutTS(PL_INFO,
            "INFO: number of CV groups has been adjusted to %d.\n",
            numCVGroups);
         printOutTS(PL_INFO,"      Each CV group has <= %d sample points\n",
                    nSubSamples);
      }
      snprintf(pString,100,"Random selection of leave-out groups ? (y or n) ");
      getString(pString, winput);
      if (winput[0] == 'y') ranFlag = 1;
      printOutTS(PL_INFO,"PRSA: Cross validation (CV) begins...\n");
   }
   commMgr_->bcast((void *) &numCVGroups, iOne, INT, 0);
   commMgr_->bcast((void *) &nSubSamples, iOne, INT, 0);
   commMgr_->bcast((void *) &ranFlag, iOne, INT, 0);

   //**/ allocate space to store errors
   adata.sampleErrors_ = new double[nSamples];
   for (ss = 0; ss < nSamples; ss++) adata.sampleErrors_[ss] = 0.0;

   //**/ instantiate regressors and set parameters
   faPtr = genFA(rsType_, nInputs, iOne, nSamples-nSubSamples);
   //**/ NULL check by Bill Oliver
   if (faPtr == NULL)
   {
      printOutTS(PL_INFO,"PRSA: genFA returned NULL in file %s line %d\n",
                 __FILE__, __LINE__ );
      exit(1);
   }
   faPtr->setNPtsPerDim(nPtsPerDim);
   faPtr->setBounds(lower, upper);
   if (mypid_ == 0) faPtr->setOutputLevel(printLevel);
   else             faPtr->setOutputLevel(0);
   psConfig_.RSExpertModeSaveAndReset();

   //**/ permute design matrix
   XX = new double[nSamples*nInputs];
   YY = new double[nSamples];
   WW = new double[nSamples];
   iArray  = new int[nSamples];
   iArray2 = new int[nSamples];
   checkAllocate(iArray2, "iArray2 in PRSFuncApprox::analyze");
   for (ss = 0; ss < nSamples; ss++) iArray[ss] = ss;
   if (ranFlag == 1) generateRandomIvector(nSamples, iArray);
   commMgr_->bcast((void *) iArray, nSamples, INT, 0);
   for (ii = 0; ii < nInputs; ii++)
   {
      for (ss = 0; ss < nSamples; ss++)
         XX[iArray[ss]*nInputs+ii] = X[ss*nInputs+ii];
   }
   for (ss = 0; ss < nSamples; ss++) YY[iArray[ss]] = YLocal[ss];
   for (ss = 0; ss < nSamples; ss++) iArray2[iArray[ss]] = ss;
   if (wgtID >= 0 && wgtID < nOutputs)
   {
      for (ss = 0; ss < nSamples; ss++)
         WW[iArray[ss]] = Y[ss*nOutputs+wgtID];
   }

   //**/ iterate on groups
   X2 = new double[nSamples*nInputs];
   Y2 = new double[nSamples];
   YT = new double[nSamples];
   ST = new double[nSamples];
   wgts = new double[nSamples];
   checkAllocate(wgts, "wgts in PRSFuncApprox::analyze");
   for (ss = 0; ss < nSamples; ss+=nSubSamples)
   {
      pindex = ss / nSubSamples;
      pindex = pindex % nprocs_;
      if (pindex == mypid_)
      {
         printf("P%5d : CV processes %d out of %d.\n",
                mypid_,ss/nSubSamples+1, nSamples/nSubSamples);

         //**/ gather the non-excluded data points
         count = 0;
         for (ss2 = 0; ss2 < nSamples; ss2++)
         {
            if (ss2 < ss || ss2 >= (ss+nSubSamples))
            {
               for (ii = 0; ii < nInputs; ii++)
                  X2[count*nInputs+ii] = XX[ss2*nInputs+ii];
               if (wgtID >= 0 && wgtID < nOutputs)
                  wgts[count] = WW[ss2*nOutputs+wgtID];
               Y2[count++] = YY[ss2];
            }
         }
         if (wgtID >= 0 && wgtID < nOutputs)
            faPtr->loadWeights(nSamples-nSubSamples, wgts);

         //**/ generate response surface
         status = faPtr->initialize(X2, Y2);
         if (status == -1) break;
         count = nSubSamples;
         if ((ss + nSubSamples) > nSamples) count = nSamples - ss;
         faPtr->evaluatePointFuzzy(count, &(XX[ss*nInputs]), &YT[ss], &ST[ss]);
      }
   }

   //**/ Communicate the chain information
   commFlag = 1;
   commMgr_->bcast((void *) &commFlag, iOne, INT, 0);
   for (ss = 0; ss < nSamples; ss+=nSubSamples)
   {
      pindex = ss / nSubSamples;
      pindex = pindex % nprocs_;
      count = nSubSamples;
      if ((ss + nSubSamples) > nSamples) count = nSamples - ss;
      if (mypid_ == 0 && pindex != mypid_)
         commMgr_->recv((void *) &YT[ss],count,DOUBLE,pindex,pindex);
      else if (mypid_ != 0 && pindex == mypid_)
         commMgr_->send((void *) &YT[ss],count,DOUBLE,pindex,0);
   }
   for (ss = 0; ss < nSamples; ss+=nSubSamples)
   {
      pindex = ss / nSubSamples;
      pindex = pindex % nprocs_;
      count = nSubSamples;
      if ((ss + nSubSamples) > nSamples) count = nSamples - ss;
      if (mypid_ == 0 && pindex != mypid_)
         commMgr_->recv((void *) &ST[ss],count,DOUBLE,pindex,pindex);
      else if (mypid_ != 0 && pindex == mypid_)
         commMgr_->send((void *) &ST[ss],count,DOUBLE,pindex,0);
   }

   //**/ compute errors
   adata.sampleErrors_ = NULL;
   if (mypid_ == 0)
   {
      eArray = new double[nSamples];
      sArray = new double[nSamples];
      sigmas = new double[nSamples];
      checkAllocate(sigmas, "sigmas in PRSFuncApprox::analyze");
      cvErr1 = cvErr1s = cvErr2 = cvErr2s = cvMax = cvMaxs = 0.0;
      cvMaxBase = cvMaxBases = 0.0;
      for (ss = 0; ss < nSamples; ss++)
      {
         ddata = YT[ss] - YY[ss];
         eArray[iArray2[ss]] = ddata;
         sArray[iArray2[ss]] = YT[ss];
         sigmas[iArray2[ss]] = ST[ss];
         cvErr1  += ddata;
         cvErr2  += (ddata * ddata);
         if (PABS(ddata) > cvMax)
         {
            cvMax  = PABS(ddata);
            cvMaxBase = PABS(YY[ss]);
         }
         if (YY[ss] != 0.0) sdata = ddata / PABS(YY[ss]);
         else               sdata = ddata;
         cvErr1s += sdata;
         cvErr2s += (sdata * sdata);
         if (PABS(sdata) > cvMaxs)
         {
            cvMaxs = PABS(sdata);
            cvMaxBases = PABS(YY[ss]);
         }
      }
      cvErr1  = cvErr1 / (double) nSamples;
      cvErr1s = cvErr1s / (double) nSamples;
      cvErr2  = sqrt(cvErr2 / nSamples);
      cvErr2s = sqrt(cvErr2s / nSamples);
      printOutTS(PL_INFO,
           "PRSA: final CV error  = %11.3e (avg unscaled)\n",cvErr1);
      printOutTS(PL_INFO,
           "PRSA: final CV error  = %11.3e (avg   scaled)\n",cvErr1s);
      printOutTS(PL_INFO,
           "PRSA: final CV error  = %11.3e (rms unscaled)\n",cvErr2);
      printOutTS(PL_INFO,
           "PRSA: final CV error  = %11.3e (rms   scaled)\n",cvErr2s);
      printOutTS(PL_INFO,
           "PRSA: final CV error  = %11.3e (max unscaled,BASE=%9.3e)\n",
           cvMax, cvMaxBase);
      printOutTS(PL_INFO,
           "PRSA: final CV error  = %11.3e (max   scaled,BASE=%9.3e)\n",
           cvMaxs, cvMaxBases);
      if (plotScilab())
      {
         fpData = fopen("RSFA_CV_err.sci", "w");
         if (fpData != NULL)
            printOutTS(PL_INFO, "INFO: cannot open file RSFA_CV_err.sci.\n");
      }
      else
      {
         fpData = fopen("RSFA_CV_err.m", "w");
         if (fpData == NULL)
            printOutTS(PL_INFO, "INFO: cannot open file RSFA_CV_err.m.\n");
      }
      if (fpData != NULL)
      {
         strcpy(pString,"This file stores CV error for each point");
         fwriteComment(fpData, pString);
         strcpy(pString,"Column 1: error (true - predicted)");
         fwriteComment(fpData, pString);
         strcpy(pString,"Column 2: true");
         fwriteComment(fpData, pString);
         strcpy(pString,"Column 3: predicted");
         fwriteComment(fpData, pString);
         strcpy(pString,"Column 4: standard deviations"); 
         fwriteComment(fpData, pString);
         strcpy(pString,"Set morePlots=1 for normalized residual error plot");
         fwriteComment(fpData, pString);
         ssum = 0.0;
         fprintf(fpData, "morePlots = 0;\n");
         fprintf(fpData, "A = [\n");
         for (ss = 0; ss < nSamples; ss++)
         {
            fprintf(fpData, "  %e %e %e %e\n", eArray[ss], YLocal[ss],
                    sArray[ss], sigmas[ss]);
            ssum += PABS(sigmas[ss]);
         }
         fprintf(fpData, "];\n");
         fwriteHold(fpData, 0);
         fwritePlotFigure(fpData, 1);
         fprintf(fpData, "subplot(1,2,1)\n");
         if (plotScilab())
         {
            fprintf(fpData, "ymin = min(A(:,1));\n");
            fprintf(fpData, "ymax = max(A(:,1));\n");
            fprintf(fpData, "ywid = 0.1 * (ymax - ymin);\n");
            fprintf(fpData, "if (ywid < 1.0e-12)\n");
            fprintf(fpData, "   disp('range too small.')\n");
            fprintf(fpData, "   halt\n");
            fprintf(fpData, "end;\n");
            fprintf(fpData, "histplot(10, A(:,1), style=2);\n");
            fprintf(fpData, "a = gce();\n");
            fprintf(fpData, "a.children.fill_mode = \"on\";\n");
            fprintf(fpData, "a.children.thickness = 2;\n");
            fprintf(fpData, "a.children.foreground = 0;\n");
            fprintf(fpData, "a.children.background = 2;\n");
         }
         else
         {
            fprintf(fpData, "[nk, xk] = hist(A(:,1), 10);\n");
            fprintf(fpData, "bar(xk,nk/sum(nk))\n");
         }
         fwritePlotAxes(fpData);
         fwritePlotTitle(fpData, "CV Error Histogram");
         fwritePlotXLabel(fpData, "Error (unnormalized)");
         fwritePlotYLabel(fpData, "Probabilities");
         if (plotMatlab())
         {
            fprintf(fpData,"disp(['Error Mean  = ' num2str(mean(A(:,1)))])\n");
            fprintf(fpData,"disp(['Error stdev = ' num2str(std(A(:,1)))])\n");
         }
         fprintf(fpData, "subplot(1,2,2)\n");
         fprintf(fpData, "xmax = max(A(:,2));\n");
         fprintf(fpData, "xmin = min(A(:,2));\n");
         fprintf(fpData, "if (xmax == xmin) \n");
         fprintf(fpData, "   xmin = 0.9 * xmin; \n");
         fprintf(fpData, "   xmax = 1.1 * xmax; \n");
         fprintf(fpData, "end;\n");
         fprintf(fpData, "if (xmax == xmin) \n");
         fprintf(fpData, "   xmin = -0.1; \n");
         fprintf(fpData, "   xmax = 0.1; \n");
         fprintf(fpData, "end;\n");
         fprintf(fpData, "ymax = max(A(:,3)+A(:,4));\n");
         fprintf(fpData, "ymin = min(A(:,3)-A(:,4));\n");
         fprintf(fpData, "if (ymax == ymin) \n");
         fprintf(fpData, "   ymin = 0.9 * ymin; \n");
         fprintf(fpData, "   ymax = 1.1 * ymax; \n");
         fprintf(fpData, "end;\n");
         fprintf(fpData, "if (ymax == ymin) \n");
         fprintf(fpData, "   ymin = -0.1; \n");
         fprintf(fpData, "   ymax = 0.1; \n");
         fprintf(fpData, "end;\n");
         fprintf(fpData, "xmin = min(xmin, ymin);\n");
         fprintf(fpData, "xmax = max(xmax, ymax);\n");
         fprintf(fpData, "XX = xmin : xmax-xmin : xmax;\n");
         fprintf(fpData, "plot(A(:,2), A(:,3),'*','MarkerSize',12)\n");
         fwriteHold(fpData, 1);
         fprintf(fpData, "plot(XX, XX)\n");
         if (plotScilab())
         {
            fprintf(fpData, "a = get(\"current_axes\");\n");
            fprintf(fpData, "a.data_bounds=[xmin,xmin;xmax,xmax];\n");
         }
         else fprintf(fpData, "axis([xmin xmax xmin xmax])\n");
         if (ssum > 0.0)
         {
            if (plotScilab()) fprintf(fpData,"drawlater\n");
            fprintf(fpData,"for ii = 1 : %d\n", nSamples);
            fprintf(fpData,"  xx = [A(ii,2) A(ii,2)];\n");
            fprintf(fpData,"  d1 = A(ii,3)-A(ii,4);\n");
            fprintf(fpData,"  d2 = A(ii,3)+A(ii,4);\n");
            fprintf(fpData,"  yy = [d1 d2];\n");
            fprintf(fpData,"  if (xx(1) < d1 | xx(1) > d2);\n");
            fprintf(fpData,"    plot(xx, yy, 'r-', 'lineWidth', 1);\n");
            fprintf(fpData,"   else\n");
            fprintf(fpData,"     plot(xx, yy, 'g-', 'lineWidth', 1);\n");
            fprintf(fpData,"  end;\n");
            fprintf(fpData,"%%plot(xx(1), yy(1),'rv','markerSize',10);\n");
            fprintf(fpData,"%%plot(xx(2), yy(2),'r^','markerSize',10);\n");
            fprintf(fpData,"end;\n");
            if (plotScilab()) fprintf(fpData,"drawnow\n");
            else
            {
              fprintf(fpData,"text(0.1,0.9,'RED: prediction outside +/-1 std ");
              fprintf(fpData,"dev','sc','fontSize',11,'fontweight','bold')\n");
            }
         }
         fwriteHold(fpData, 0);
         fwritePlotAxes(fpData);
         fwritePlotTitle(fpData, "Actual vs predicted data with CV");
         fwritePlotXLabel(fpData, "actual data");
         fwritePlotYLabel(fpData, "predicted data");
         fprintf(fpData,"if morePlots == 1\n");
         strcpy(pString," For the following B matrix");
         fwriteComment(fpData, pString);
         strcpy(pString,"Column 1: true values");
         fwriteComment(fpData, pString);
         strcpy(pString,"Column 2: normalized residual");
         fwriteComment(fpData, pString);
         strcpy(pString,"Column 3: predicted values");
         fwriteComment(fpData, pString);
         strcpy(pString,"Column 4-(m+3): inputs");
         fwriteComment(fpData, pString);
         fwritePlotFigure(fpData, 2);
         fprintf(fpData, "B = [\n");
         for (ss = 0; ss < nSamples; ss++)
         {
            if (YLocal[ss] == 0) fprintf(fpData, " %e 0 ",YLocal[ss]);
            else fprintf(fpData," %e %e ",YLocal[ss],eArray[ss]/YLocal[ss]);
            fprintf(fpData," %e ", sArray[ss]);
            for (ss2 = 0; ss2 < nInputs; ss2++)
               fprintf(fpData," %e ",XX[ss*nInputs+ss2]);
            fprintf(fpData,"\n");
         }
         fprintf(fpData, "];\n");
         fprintf(fpData, "AA = B(:,1);\n");
         fprintf(fpData, "BB = B(:,2);\n");
         fprintf(fpData, "CC = B(:,3);\n");
         fprintf(fpData, "plot(AA, BB, '*','markerSize',12);\n");
         fwritePlotAxes(fpData);
         fwritePlotTitle(fpData, "Normalized Resdual Analysis");
         fwritePlotXLabel(fpData, "Actual Data");
         fwritePlotYLabel(fpData, "Normalized Error");
         strcpy(pString,"plot(AA-CC, '*','markerSize',12);");
         fwriteComment(fpData, pString);
         strcpy(pString,"xlabel('Sample Number');");
         fwriteComment(fpData, pString);
         strcpy(pString,"ylabel('Error (true-predicted)');");
         fwriteComment(fpData, pString);
         fprintf(fpData,"end;\n");
         fclose(fpData);
         if (plotScilab())
              printOutTS(PL_INFO, "CV error file is RSFA_CV_err.sci\n");
         else printOutTS(PL_INFO, "CV error file is RSFA_CV_err.m\n");
      }
      delete [] eArray;
      delete [] sArray;
      delete [] sigmas;
   }
   psConfig_.RSExpertModeRestore();
   delete faPtr;
   delete [] XX;
   delete [] YY;
   delete [] YT;
   delete [] WW;
   delete [] X2;
   delete [] Y2;
   delete [] ST;
   delete [] wgts;
   delete [] iArray;
   delete [] iArray2;
   delete [] YLocal;
   return 0;
}

// ************************************************************************ 
// set parameters
// ------------------------------------------------------------------------
int PRSFuncApproxAnalyzer::setParams(int argc, char **argv)
{
   char  *request = (char *) argv[0]; 
   if (!strcmp(request, "rstype"))
   {
      if (argc != 2) printOutTS(PL_WARN,"PRSA WARNING: setParams.\n");
      rsType_ = *(int *) argv[1];
      if (rsType_ < 0 || rsType_ > PSUADE_NUM_RS)
      {
         printOutTS(PL_ERROR,"PRSA ERROR: INVALID rstype, set to REGR2.\n");
         rsType_ = PSUADE_RS_REGR2;
      }
   }
   else
   {
      printOutTS(PL_ERROR,"PRSA ERROR: setParams - not valid.\n");
      exit(1);
   }
   return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
PRSFuncApproxAnalyzer& PRSFuncApproxAnalyzer::operator=(const 
                       PRSFuncApproxAnalyzer &)
{
   printOutTS(PL_ERROR,"PRSA operator= ERROR: operation not allowed.\n");
   exit(1);
   return (*this);
}

