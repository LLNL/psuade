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
// Functions for the class OUU1Optimizer
// AUTHOR : CHARLES TONG
// DATE   : 2015
// ************************************************************************
#include <PsuadeCmakeConfig.h>

#ifdef WINDOWS
#include <windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "OUU1Optimizer.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "sysdef.h"
#include "Sampling.h"
#include "PDFManager.h"

// ************************************************************************
// External functions
// ------------------------------------------------------------------------
extern "C" void bobyqa_(int *,int *, double *, double *, double *, double *,
                        double *, int *, int *, double*);

// ************************************************************************
// Internal 'global' variables
// ------------------------------------------------------------------------
int     psOUU1PrintLevel_=0;
void    *psOUU1Obj_=NULL;
int     psOUU1M1_=-1;
int     psOUU1M2_=-1;
double  *psOUU1M2LowerB_=NULL;
double  *psOUU1M2UpperB_=NULL;
int     psOUU1nSamples_=100;
double  *psOUU1SamInputs_=NULL;
double  *psOUU1SamOutputs_=NULL;
double  *psOUU1XValues_=NULL;
int     psOUU1UseRS_=0;
int     psOUU1Mode_=1;
int     psOUU1Percentile_=0;
double  psOUU1StdevMultiplier_=0;
FuncApprox *psOUU1faPtr_=NULL;
int     psOUU1LargeSampleSize_=0;
double  *psOUU1LargeSamInputs_=NULL;
double  *psOUU1LargeSamOutputs_=NULL;
int     psOUU1Parallel_=0;
#define psOUU1MaxSaved_ 10000
int     psOUU1NSaved_=0;
double  psOUU1SaveX_[psOUU1MaxSaved_*10];
double  psOUU1SaveY_[psOUU1MaxSaved_*10];
int     psOUU1EnsembleEval_ = 0;

#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// ************************************************************************
// resident function to perform evaluation 
// ------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" 
{
#endif
  void *ouu1evalfunc_(int *nInps, double *XValues, double *YValue)
  {
    int    ii, jj, kk, funcID, nDesigns, M, iOne=1, status;
    int    minIndex, *readys, found;
    double mean, stdev, ddata;
    oData  *odata;
    FILE   *fp=NULL;

    //**/ ------ fetch data ------
    nDesigns = (*nInps);
    odata    = (oData *) psOUU1Obj_;
    M        = nDesigns + psOUU1M2_;
    funcID   = odata->numFuncEvals_;

    //**/ ------ detect termination request ------
    fp = fopen("psuade_ouu_stop","r");
    if (fp != NULL && psOUU1NSaved_ > 0)
    {
      fclose(fp);
      printf("OUU1Optimizer: psuade_ouu_stop file found.\n");
      printf("             Abrupt termination.\n");
      unlink("psuade_ouu_stop");
      fp = fopen("psuade_ouu1_history","w");
      if (fp != NULL)
      {
        for (ii = 0; ii < psOUU1NSaved_; ii++)
        {
          fprintf(fp, "999 %d ", M);
          for (kk = 0; kk < M; kk++)
            fprintf(fp, "%24.16e ", psOUU1SaveX_[ii*M+kk]);
          fprintf(fp, "%24.16e\n", psOUU1SaveY_[ii]);
        }
        fclose(fp);
      }
      printf("OUU1Optimizer: history saved in psuade_ouu1_history\n");
      exit(1);
    }

    //**/ ------ run simulations ------
    for (ii = 0; ii < nDesigns; ii++) psOUU1XValues_[ii] = XValues[ii];
    if (psOUU1EnsembleEval_ == 0)
    {
      readys = (int *) malloc(psOUU1nSamples_*sizeof(int));
      for (ii = 0; ii < psOUU1nSamples_; ii++)
      {
        for (jj = 0; jj < psOUU1M2_; jj++) 
          psOUU1XValues_[nDesigns+jj] = psOUU1SamInputs_[ii*psOUU1M2_+jj];

        readys[ii] = -1;
        //**/ check to see if it is in the history
        found = 0;
        for (jj = 0; jj < psOUU1NSaved_; jj++)
        {
          for (kk = 0; kk < M; kk++)
            if (PABS(psOUU1SaveX_[jj*M+kk]-psOUU1XValues_[kk])>1.0e-14) 
              break;
          if (kk == M)
          {
            found = 1;
            if (odata->outputLevel_ > 2)
              printf("OUU1Optimizer: simulation results reuse.\n");
            psOUU1SamOutputs_[ii] = psOUU1SaveY_[jj];
            readys[ii] = 0;
            break;
          }
        }
        //**/ if not, run the simulation
        if (found == 0)
        {
          readys[ii] = odata->funcIO_->evaluate(funcID+ii,M,
                          psOUU1XValues_,iOne,&psOUU1SamOutputs_[ii],0);
          odata->numFuncEvals_++;
          if (readys[ii] == 0)
          {
            //**/ ------ save the data ------
            if ((psOUU1NSaved_+1)*M < psOUU1MaxSaved_*10)
            {
              for (jj = 0; jj < M; jj++)
                psOUU1SaveX_[psOUU1NSaved_*M+jj] = psOUU1XValues_[jj];
              psOUU1SaveY_[psOUU1NSaved_] = psOUU1SamOutputs_[ii];
              psOUU1NSaved_++;
            }
          }
        }
      }
   
      //**/ ------ additional processing for asynchronous mode ------
      if (psOUU1Parallel_ == 1)
      {
        for (ii = 0; ii < psOUU1nSamples_; ii++)
        {
          if (readys[ii] != 0)
          {
            readys[ii] = odata->funcIO_->evaluate(funcID+ii,M,
                           psOUU1XValues_,iOne,&psOUU1SamOutputs_[ii],2);
            while (readys[ii] != 0)
            {
#ifdef WINDOWS
              Sleep(1000);
#else
              sleep(1);
#endif
              readys[ii] = odata->funcIO_->evaluate(funcID+ii,M,
                            psOUU1XValues_,iOne,&psOUU1SamOutputs_[ii],2);
            }
            //**/ ------ save the data ------
            if ((psOUU1NSaved_+1)*M < psOUU1MaxSaved_*10)
            {
              for (jj = 0; jj < M; jj++)
                psOUU1SaveX_[psOUU1NSaved_*M+jj] = psOUU1XValues_[jj];
              psOUU1SaveY_[psOUU1NSaved_] = psOUU1SamOutputs_[ii];
              psOUU1NSaved_++;
            }
          }
        }
      }
    }
    else
    {
      for (ii = 0; ii < psOUU1nSamples_; ii++)
      {
        for (jj = 0; jj < nDesigns; jj++) 
          psOUU1XValues_[ii*M+jj] = XValues[jj];
        for (jj = 0; jj < psOUU1M2_; jj++) 
          psOUU1XValues_[ii*M+nDesigns+jj] = psOUU1SamInputs_[ii*psOUU1M2_+jj];
      }
      //**/ check to see if it is in the history
      found = 0;
      for (ii = 0; ii < psOUU1nSamples_; ii++)
      {
        for (jj = 0; jj < psOUU1NSaved_; jj++)
        {
          for (kk = 0; kk < M; kk++)
            if (PABS(psOUU1SaveX_[jj*M+kk]-psOUU1XValues_[ii*M+kk])>1.0e-14) 
              break;
          if (kk == M)
          {
            found++;
            psOUU1SamOutputs_[ii] = psOUU1SaveY_[jj];
            break;
          }
        }
      }
      //**/ run ensemble simulations
      if (found != psOUU1nSamples_)
      {
        odata->funcIO_->ensembleEvaluate(psOUU1nSamples_,M,psOUU1XValues_,
                                         iOne,psOUU1SamOutputs_,funcID);
        odata->numFuncEvals_ += psOUU1nSamples_;
        //**/ ------ save the data ------
        for (ii = 0; ii < psOUU1nSamples_; ii++)
        {
          if ((psOUU1NSaved_+1)*M < psOUU1MaxSaved_*10)
          {
            for (jj = 0; jj < M; jj++)
              psOUU1SaveX_[psOUU1NSaved_*M+jj] = psOUU1XValues_[ii*M+jj];
            psOUU1SaveY_[psOUU1NSaved_] = psOUU1SamOutputs_[ii];
            psOUU1NSaved_++;
          }
        }
      }
    }

    //**/ ------ compute mean ------
    //**/ Case 1: no RS used
    if (psOUU1UseRS_ == 0)
    {
      if (psOUU1Mode_ == 1 || psOUU1Mode_ == 2)
      {
        mean = 0.0;
        for (ii = 0; ii < psOUU1nSamples_; ii++)
          mean += psOUU1SamOutputs_[ii];
        mean /= (double) psOUU1nSamples_;
        (*YValue) = mean;
      }
      if (psOUU1Mode_ == 2 && psOUU1StdevMultiplier_ != 0)
      {
        stdev = 0.0;
        for (ii = 0; ii < psOUU1nSamples_; ii++)
          stdev += pow(psOUU1SamOutputs_[ii]-mean, 2.0);
        stdev /= (double) psOUU1nSamples_;
        (*YValue) = mean + psOUU1StdevMultiplier_ * stdev;
      }
      if (psOUU1Mode_ == 3)
      {
        sortDbleList(psOUU1nSamples_, psOUU1SamOutputs_);
        kk = (int) ((1.0 - psOUU1Percentile_) * psOUU1nSamples_);
        if (kk >= psOUU1nSamples_) kk = psOUU1nSamples_ - 1;
        (*YValue) = psOUU1SamOutputs_[kk];
      }
      if (psOUU1Mode_ == 4)
      {
        ddata = psOUU1SamOutputs_[0];
        minIndex = 0;
        for (ii = 1; ii < psOUU1nSamples_; ii++)
        {
          if (psOUU1SamOutputs_[ii] < ddata)
          {
            ddata = psOUU1SamOutputs_[ii];
            minIndex = ii;
          }
        }
        (*YValue) = ddata;
      }
    }
    else
    //**/ Case 2: RS used
    {
      if (odata->outputLevel_ > 2)
        printf("OUU1Optimizer: computing objective with response surface.\n");

      status = psOUU1faPtr_->initialize(psOUU1SamInputs_,psOUU1SamOutputs_);
      psOUU1faPtr_->evaluatePoint(psOUU1LargeSampleSize_,
                         psOUU1LargeSamInputs_,psOUU1LargeSamOutputs_);

      if (psOUU1Mode_ == 1 || psOUU1Mode_ == 2)
      {
        mean = 0.0;
        for (ii = 0; ii < psOUU1LargeSampleSize_; ii++)
          mean += psOUU1LargeSamOutputs_[ii];
        mean /= (double) psOUU1LargeSampleSize_;
        (*YValue) = mean;
      }
      if (psOUU1Mode_ == 2 && psOUU1StdevMultiplier_ != 0)
      {
        stdev = 0.0;
        for (ii = 0; ii < psOUU1LargeSampleSize_; ii++)
          stdev += pow(psOUU1LargeSamOutputs_[ii]-mean, 2.0);
        stdev /= (double) psOUU1LargeSampleSize_;
        (*YValue) = mean + psOUU1StdevMultiplier_ * stdev;
      }
      if (psOUU1Mode_ == 3)
      {
        sortDbleList(psOUU1LargeSampleSize_, psOUU1LargeSamOutputs_);
        kk = (int) ((1.0 - psOUU1Percentile_) * psOUU1LargeSampleSize_);
        if (kk >= psOUU1LargeSampleSize_) kk = psOUU1LargeSampleSize_ - 1;
        (*YValue) = psOUU1LargeSamOutputs_[kk];
      }
      if (psOUU1Mode_ == 4)
      {
        ddata = psOUU1LargeSamOutputs_[0];
        for (ii = 1; ii < psOUU1LargeSampleSize_; ii++)
          if (psOUU1LargeSamOutputs_[ii] < ddata)
              ddata = psOUU1LargeSamOutputs_[ii];
        (*YValue) = ddata;
      }
    }

    //**/ ------ diagnostics ------
    if (odata->outputLevel_ > 2)
    {
      printf("OUU1Optimizer %6d : \n", odata->numFuncEvals_);
      for (ii = 0; ii < nDesigns; ii++)
        printf("    X %6d = %16.8e\n", ii+1, XValues[ii]);
      if (psOUU1Mode_ == 4 && psOUU1UseRS_ == 0)
      {
        for (ii = 0; ii < psOUU1M2_; ii++)
          printf("    X %6d = %16.8e\n", ii+1+nDesigns,
                 psOUU1SamInputs_[minIndex*psOUU1M2_+ii]);
      }
      printf("    Y     = %16.8e\n", (*YValue));
    }

    //**/ ------ store optimal information ------
    if ((*YValue) < odata->optimalY_)
    {
      odata->optimalY_ = (*YValue);
      for (ii = 0; ii < nDesigns; ii++) 
        odata->optimalX_[ii] = XValues[ii];
      if (odata->outputLevel_ > 0)
      {
        printf("Number of Function Evaluations = %d\n", 
               odata->numFuncEvals_);
        for (ii = 0; ii < nDesigns; ii++)
          printf("    X %6d = %16.8e\n", ii+1, XValues[ii]);
        if (psOUU1Mode_ == 4 && psOUU1UseRS_ == 0)
        {
          for (ii = 0; ii < psOUU1M2_; ii++)
            printf("    X %6d = %16.8e\n", ii+1+nDesigns,
                   psOUU1SamInputs_[minIndex*psOUU1M2_+ii]);
        }
        printf("    New Ymin = %16.8e\n", odata->optimalY_);
      }
    }
    return NULL;
  }
#ifdef __cplusplus
}
#endif

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
OUU1Optimizer::OUU1Optimizer()
{
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
OUU1Optimizer::~OUU1Optimizer()
{
}

// ************************************************************************
// optimize
// ------------------------------------------------------------------------
void OUU1Optimizer::optimize(oData *odata)
{
  int    nInputs, printLevel=0, M1, M2, ii, kk, maxfun, nPts=0, iOne=1;
  int    iZero=0, nOutputs, bobyqaFlag=3333, printHeader=1, method=1;
  double *XValues, rhobeg=1.0, rhoend=1.0e-4, ddata, *workArray;
  char   lineIn[1000], pString[500];
  FILE   *fp=NULL;
  static int currDriver=-1;

  //**/ ----------------------------------------------------------
  //**/ obtain information from user
  //**/ ----------------------------------------------------------
  printLevel = odata->outputLevel_;
  psOUU1PrintLevel_ = printLevel;
  nInputs = odata->nInputs_;
  if (nInputs <= 1) 
  {
    printf("OUU1Optimizer ERROR: nInputs needs to be larger than 1\n");
    return;
  }
  nOutputs = odata->nOutputs_;
  if (nOutputs != 1) 
  {
    printf("OUU1Optimizer ERROR: nOutputs > 1 currently not supported.\n");
    return;
  }
  if (psOUU1M1_ > 0 && psOUU1M1_ < nInputs && (psOUU1M1_+psOUU1M2_) == nInputs)
    printHeader = 0;
  if (printLevel >= 0 && printHeader == 1)
  {
    printAsterisks(PL_INFO, 0);
    printf("         Optimization Under Uncertainty \n");
    printEquals(PL_INFO, 0);
    printf("This optimization capability solves the following problem:\n");
    printf("\n   minimize_X1 { Phi_X2 [ F(X1,X2) ] } \n\n");
    printf("   subject to bound constraints on X1 and X2\n\n");
    printf("   where\n");
    printf("      mean_X2[*] is the statistical mean with respect to X2\n");
    printf("      minimize_X1 means to minimize with respect to X1\n");
    printf("      Phi_X2 is a functional of F(X1,X2) with respect to X2\n");
    printf("      For example, Phi_X2 can be:\n");
    printf("      1. the mean of F(X1,X2) with respect to X2\n");
    printf("      2. a linear combination of mean[F(X1,X2)] and std. dev.\n");
    printf("      3. F(X1,X2*) such that Prob(F(X1,X2)>F(X1,X2)) = epsilon\n");
    printEquals(PL_INFO, 0);
    printf("Total number of parameters M = %d\n", nInputs);
    printf("These parameters are to be divided into two groups:\n");
    printf("(1) optimization (design) paramters X1 (M1 > 1) \n");
    printf("(2) uncertain paramters X2 (M2 > 1) \n");
    printf("Thus, the first M1 parameters are considered to be\n");
    printf("optimization parameters, followed by M2 uncertain\n");
    printf("parameters so that M = M1 + M2.\n");
    printf("IF YOU ARE READY TO MOVE ON, ENTER 'y' AND RETURN : ");
    scanf("%s", lineIn);
    if (lineIn[0] != 'y')
    {
      odata->optimalY_ = 0;
      for (ii = 0; ii < nInputs; ii++) odata->optimalX_[ii] = 0.0;
      printf("OUU1Optimizer INFO: abrupt termination.\n");
      return;
    }
  }
  if (psOUU1M1_ > 0 && psOUU1M2_ > 0 && (psOUU1M1_+psOUU1M2_) == nInputs)
  {
    M1 = psOUU1M1_;
    M2 = psOUU1M2_;
  }
  else
  {
    M1 = 0;
    while (M1 <= 0 || M1 >= nInputs)
    {
      printf("Enter M1 (between 1 and %d) : ", nInputs-1);
      scanf("%d", &M1);
    }
    fgets(lineIn, 500, stdin);
    psOUU1M1_ = M1;
    M2 = psOUU1M2_ = nInputs - M1;
  }
  if (printLevel >= 0)
  {
    printDashes(PL_INFO, 0);
    printf("Number of optimization parameters X1 = %d\n", M1);
    printf("Number of uncertain    parameters X2 = %d\n", M2);
    printDashes(PL_INFO, 0);
  }

  //**/ ----------------------------------------------------------
  //**/ prepare for optimization
  //**/ ----------------------------------------------------------
  for (ii = 0; ii < nInputs; ii++) odata->optimalX_[ii] = 0.0;
  odata->optimalY_ = 1.0e50;

  //**/ ---- compute optimization tolerance
  rhobeg = odata->upperBounds_[0] - odata->lowerBounds_[0];
  for (ii = 1; ii < M1; ii++) 
  {
    ddata = odata->upperBounds_[ii] - odata->lowerBounds_[ii];
    if (ddata < rhobeg) rhobeg = ddata;
  }
  rhobeg *= 0.5;
  rhoend = rhobeg * odata->tolerance_;
  if (rhobeg < rhoend)
  {
    if (printLevel >= 0)
    {
      printf("OUU1Optimizer WARNING: tolerance too large.\n");
      printf("                      tolerance reset to 1.0e-6.\n");
    }
    rhoend = rhobeg * 1.0e-6;
  }
  if (printLevel >= 0)
  {
    printf("OUU1 optimizer: max fevals = %d\n", odata->maxFEval_);
    printf("OUU1 optimizer: tolerance  = %e\n", odata->tolerance_);
    printDashes(PL_INFO, 0);
  }

  //**/ ----------------------------------------------------------
  //**/ set up uncertain parameter information
  //**/ put away current driver and use default driver
  //**/ ----------------------------------------------------------
  if ((odata->setOptDriver_ & 1))
  {
    if (printLevel >= 0)
      printf("OUU1Optimizer: setting optimization simulation driver.\n");
    currDriver = odata->funcIO_->getDriver();
    odata->funcIO_->setDriver(1);
  }
  psOUU1M2LowerB_ = new double[M2];
  psOUU1M2UpperB_ = new double[M2];
  for (ii = 0; ii < M2; ii++) 
  {
    psOUU1M2LowerB_[ii] = odata->lowerBounds_[M1+ii];
    psOUU1M2UpperB_[ii] = odata->upperBounds_[M1+ii];
  }
  psOUU1Obj_= (void *) odata;

  //**/ ----------------------------------------------------------
  //**/ set up LH/FACT sample for uncertain parameters
  //**/ ----------------------------------------------------------
  psOUU1nSamples_ = 100;
  if (psConfig_.OptExpertModeIsOn())
  {
    printEquals(PL_INFO, 0);
    printf("Select which functional Phi_X2 to use: \n");
    printf("  1. the mean of F(X1,X2) with respect to X2 (default)\n");
    printf("  2. mean of F(X1,X2) + alpha * std dev of F(X1,X2)\n");          
    printf("  3. F(X1,X2*) s.t. Prob(F(X1,X2)>F(X1,X2*)) = epsilon\n");
    printf("  4. min_X2 F(X1,X2) \n");
    snprintf(pString,100,"Enter your preferred functional (1, 2, 3 or 4) : ");
    psOUU1Mode_ = getInt(1, 4, pString);
    if (psOUU1Mode_ == 2)
    {
      snprintf(pString,100,"Enter your desired alpha : ");
      psOUU1StdevMultiplier_ = getDouble(pString);
    }
    if (psOUU1Mode_ == 3)
    {
      psOUU1Percentile_ = 0.0;
      while (psOUU1Percentile_ <= 0.01 || psOUU1Percentile_ > 0.5)    
      {
        snprintf(pString,100,"Enter your desired percentile : (0.01 - 0.5)");
        psOUU1Percentile_ = getDouble(pString);
      }
    } 
    printEquals(PL_INFO, 0);
    printf("OUU1Optimizer uses a sample of X2 to estimate the objective.\n");
    printf("Default sampling method = Latin hypercube\n");
    printf("Default sample size     = %d\n",psOUU1nSamples_);
    printf("Options for sampling method: (1) LHS and (2) factorial.\n");
    snprintf(pString,100,"Enter your preferred method (1 or 2) : ");
    method = getInt(1, 2, pString);
    if (method == 1)
    {
      snprintf(pString,100,
              "Enter your preferred sample size (>=100, <=1000) : ");
      psOUU1nSamples_ = getInt(50, 10000, pString);
    }
    else
    {
      snprintf(pString,100,
              "Enter number of levels per variable (>=3, <=100) : ");
      psOUU1nSamples_ = getInt(3, 100, pString);
      kk = psOUU1nSamples_;
      for (ii = 1; ii < M2; ii++) psOUU1nSamples_ *= kk;
      printf("Factorial design has sample size = %d\n", psOUU1nSamples_);
    }
    printEquals(PL_INFO, 0);
    printf("Use response surface to compute statistics ? (y or n) ");
    scanf("%s", lineIn);
    if (lineIn[0] == 'y') psOUU1UseRS_ = 1;
    printEquals(PL_INFO, 0);
    printf("Use aux opt driver with ensemble run ? (y or n) ");
    scanf("%s", lineIn);
    if (lineIn[0] == 'y') psOUU1EnsembleEval_ = 1;
    printEquals(PL_INFO, 0);
    if (psOUU1EnsembleEval_ == 0)
    {
      printf("Use synchronous (s) or asychronous (p) mode for ensemble runs ? (s or p) ");
      scanf("%s", lineIn);
      if (lineIn[0] == 'p') psOUU1Parallel_ = 1;
        printEquals(PL_INFO, 0);
    }
  }

  //**/ ----------------------------------------------------------
  //**/ generate a sample for computing statistics
  //**/ ----------------------------------------------------------
  if (printLevel > 2) printf("OUU1Optimizer: generating a sample.\n");
  Sampling *sampler=NULL;
  if (method == 1)
       sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
  else sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_FACT);
  sampler->setPrintLevel(0);
  sampler->setInputBounds(M2, psOUU1M2LowerB_, psOUU1M2UpperB_);
  sampler->setOutputParams(iOne);
  sampler->setSamplingParams(psOUU1nSamples_, iOne, iZero);
  sampler->initialize(0);
  psOUU1nSamples_   = sampler->getNumSamples();
  psOUU1SamInputs_  = new double[psOUU1nSamples_ * M2];
  psOUU1SamOutputs_ = new double[psOUU1nSamples_];
  int *samStates   = new int[psOUU1nSamples_];
  sampler->getSamples(psOUU1nSamples_, M2, iOne, psOUU1SamInputs_,
                      psOUU1SamOutputs_, samStates);
  delete [] samStates;
  delete sampler;
  psOUU1faPtr_ = NULL;

  //**/ ----------------------------------------------------------
  //**/ set up response surface constructor and a large sample
  //**/ to be evaluated on the response surface
  //**/ ----------------------------------------------------------
  int        rstype=0, *inputPDFs=NULL;
  double     *inputMeans=NULL, *inputStdevs=NULL;
  PDFManager *pdfman=NULL;
  psVector   vecLB, vecUB, vecOut;
  pData      pdata;
  psMatrix   *corMat1, corMat2;
  if (psOUU1UseRS_ == 1)
  {
    if (printLevel > 2)
      printf("OUU1Optimizer: setting up response surface\n");
    if (method == 2 && ((M2 == 2) || (M2 == 2)))
    {
      rstype = PSUADE_RS_SPLINES;
      printf("OUU1ptimizer: use Splines for factorial design\n");
    }
    if (psOUU1nSamples_ > 300)
    {
      rstype = PSUADE_RS_MARS;
      printf("OUU1Optimizer: use MARS since nSamples > 300\n");
    }
    else
    {
      rstype = PSUADE_RS_KR;
      printf("OUU1Optimizer: use Kriging response surface\n");
    }
    psConfig_.InteractiveSaveAndReset();
    psOUU1faPtr_ = genFA(rstype, M2, -1, psOUU1nSamples_);
    psOUU1faPtr_->setBounds(&(odata->lowerBounds_[M1]),
                            &(odata->upperBounds_[M1]));
    psOUU1faPtr_->setOutputLevel(0);
    psConfig_.InteractiveRestore();

    //**/ check to see if the inputs have other than uniform PDFs
    odata->psIO_->getParameter("input_pdfs", pdata);
    inputPDFs = pdata.intArray_;
    pdata.intArray_ = NULL;
    if (inputPDFs == NULL)
    {
      inputPDFs = new int[nInputs];
      for (ii = 0; ii < M1+M2; ii++) inputPDFs[ii] = 0;
    }
    kk = 0;
    for (ii = M1; ii < M1+M2; ii++) kk += inputPDFs[ii];
    if (kk == 0)
    {
      sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
      sampler->setPrintLevel(0);
      sampler->setInputBounds(M2, &(odata->lowerBounds_[M1]),
                              &(odata->upperBounds_[M1]));
      sampler->setOutputParams(iOne);
      psOUU1LargeSampleSize_ = 10000;
      sampler->setSamplingParams(psOUU1LargeSampleSize_, iOne, iZero);
      sampler->initialize(0);
      psOUU1LargeSampleSize_ = sampler->getNumSamples();
      psOUU1LargeSamInputs_= new double[psOUU1LargeSampleSize_*M2];
      psOUU1LargeSamOutputs_ = new double[psOUU1LargeSampleSize_];
      samStates = new int[psOUU1LargeSampleSize_];
      sampler->getSamples(psOUU1LargeSampleSize_, M2, iOne,
                   psOUU1LargeSamInputs_,psOUU1LargeSamOutputs_,samStates);
      delete [] samStates;
      delete sampler;
    }
    else
    {
      odata->psIO_->getParameter("input_means", pdata);
      inputMeans = pdata.dbleArray_;
      if (inputMeans == NULL)
      {
        inputMeans = new double[nInputs];
        for (ii = 0; ii < M1+M2; ii++) inputMeans[ii] = 0;
      }
      pdata.dbleArray_ = NULL;
      odata->psIO_->getParameter("input_stdevs", pdata);
      inputStdevs = pdata.dbleArray_;
      if (inputStdevs == NULL)
      {
        inputStdevs = new double[nInputs];
        for (ii = 0; ii < M1+M2; ii++) inputStdevs[ii] = 1;
      }
      pdata.dbleArray_ = NULL;
      odata->psIO_->getParameter("input_cor_matrix", pdata);
      corMat1 = (psMatrix *) pdata.psObject_;
      pdata.psObject_ = NULL;
      corMat2.setDim(M2,M2);
      for (ii = 0; ii < M2; ii++)
      {
        for (kk = 0; kk < M2; kk++)
        {
          ddata = corMat1->getEntry(M1+ii,M1+kk);
          corMat2.setEntry(ii,kk, ddata);
        }
      }
      pdfman = new PDFManager();
      pdfman->initialize(M2,&(inputPDFs[M1]),&(inputMeans[M1]),
                         &(inputStdevs[M1]),corMat2,NULL,NULL);
      vecLB.load(M2, &(odata->lowerBounds_[M1]));
      vecUB.load(M2, &(odata->upperBounds_[M1]));
      psOUU1LargeSampleSize_ = 10000;
      vecOut.setLength(psOUU1LargeSampleSize_*M2);
      pdfman->genSample(psOUU1LargeSampleSize_, vecOut, vecLB, vecUB);
      psOUU1LargeSamInputs_= new double[psOUU1LargeSampleSize_*M2];
      psOUU1LargeSamOutputs_ = new double[psOUU1LargeSampleSize_];
      for (ii = 0; ii < psOUU1LargeSampleSize_*M2; ii++)
        psOUU1LargeSamInputs_[ii] = vecOut[ii];
      delete [] inputPDFs;
      delete [] inputMeans;
      delete [] inputStdevs;
      delete pdfman;
    }
  }

  //**/ ----------------------------------------------------------
  //**/ call optimizer 
  //**/ ----------------------------------------------------------
  nPts = (M1 + 1) * (M1 + 2) / 2;
  workArray = new double[(nPts+5)*(nPts+nInputs)+3*nInputs*(nInputs+5)/2+1];
  XValues = new double[nInputs+1];
  psOUU1XValues_ = new double[nInputs*psOUU1nSamples_];
  maxfun = odata->maxFEval_;
  if (psOUU1Parallel_ == 1) odata->funcIO_->setAsynchronousMode();

#ifdef HAVE_BOBYQA
  //**/ ----- retrieve history -----
  fp = fopen("psuade_ouu1_history", "r");
  if (fp != NULL)
  {
    printf("OUU1Optimizer history file found. Use it? (y or n) ");
    scanf("%s", lineIn);
    if (lineIn[0] != 'y') fclose(fp);
    else
    {
      psOUU1NSaved_ = 0;
      while (feof(fp) == 0)
      {
        fscanf(fp, "%d %d", &ii, &kk);
        if (ii != 999 || kk != nInputs)
        {
          break;
        }
        else
        {
          for (ii = 0; ii < nInputs; ii++)
            fscanf(fp, "%lg",&psOUU1SaveX_[psOUU1NSaved_*nInputs+ii]);
          fscanf(fp, "%lg",&psOUU1SaveY_[psOUU1NSaved_]);
          psOUU1NSaved_++;
        }
        if (((psOUU1NSaved_+1)*nInputs > psOUU1MaxSaved_*10) ||
          psOUU1NSaved_ > psOUU1MaxSaved_) break;
      }
      fclose(fp);
    }
  }

  //**/ ----- run optimization -----
  for (ii = 0; ii < M1; ii++)
  {
    XValues[ii] = odata->initialX_[ii];
    if (printLevel >= 0)
      printf("OUU1Optimizer initial X %3d = %e\n", ii+1, XValues[ii]);
  }
  bobyqa_(&M1, &nPts, XValues, odata->lowerBounds_, odata->upperBounds_, 
          &rhobeg, &rhoend, &bobyqaFlag, &maxfun, workArray);
  if (psOUU1Parallel_ == 1) odata->funcIO_->setSynchronousMode();
  if (printLevel >= 0)
     printf("OUU1Optimizer: total number of evaluations = %d\n",
             odata->numFuncEvals_);

  //**/ ----- save history -----
  if (psOUU1NSaved_ > 0)
  {
    fp = fopen("psuade_ouu1_history","w");
    if (fp != NULL)
    {
      for (ii = 0; ii < psOUU1NSaved_; ii++)
      {
        fprintf(fp, "999 %d ", nInputs);
        for (kk = 0; kk < nInputs; kk++)
          fprintf(fp, "%24.16e ", psOUU1SaveX_[ii*nInputs+kk]);
        fprintf(fp, "%24.16e\n", psOUU1SaveY_[ii]);
      }
      fclose(fp);
    }
    printf("OUU1Optimizer: history saved in psuade_ouu1_history\n");
  }
#else
  printf("ERROR : OUU1 optimizer not installed.\n");
  exit(1);
#endif

  //**/ ----------------------------------------------------------
  //**/ set return values and clean up 
  //**/ ----------------------------------------------------------
  if ((odata->setOptDriver_ & 2) && currDriver >= 0)
  {
    if (printLevel >= 0)
      printf("Bobyqa INFO: reverting to original simulation driver.\n");
    odata->funcIO_->setDriver(currDriver);
  }
  delete [] XValues;
  delete [] workArray;
  if (psOUU1SamInputs_ != NULL) delete [] psOUU1SamInputs_;
  if (psOUU1SamOutputs_ != NULL) delete [] psOUU1SamOutputs_;
  if (psOUU1M2LowerB_ != NULL) delete [] psOUU1M2LowerB_;
  if (psOUU1M2UpperB_ != NULL) delete [] psOUU1M2UpperB_;
  if (psOUU1XValues_ != NULL) delete [] psOUU1XValues_;
  if (psOUU1LargeSamInputs_ != NULL) delete [] psOUU1LargeSamInputs_;
  if (psOUU1LargeSamOutputs_ != NULL) delete [] psOUU1LargeSamOutputs_;
  if (psOUU1faPtr_ != NULL) delete psOUU1faPtr_;
  psOUU1LargeSamInputs_ = NULL;
  psOUU1LargeSamOutputs_ = NULL;
  psOUU1M2LowerB_ = NULL;
  psOUU1M2UpperB_ = NULL;
  psOUU1SamInputs_ = NULL;
  psOUU1SamOutputs_ = NULL;
  psOUU1XValues_ = NULL;
  psOUU1LargeSamInputs_ = NULL;
  psOUU1LargeSamOutputs_ = NULL;
  psOUU1faPtr_ = NULL;
  odata->intData_ = M1;
}

// ************************************************************************
// set parameter 
// ------------------------------------------------------------------------
void OUU1Optimizer::setNumDesignParams(int nDesign)
{
  psOUU1M1_ = nDesign;
}

// ************************************************************************
// assign operator
// ------------------------------------------------------------------------
OUU1Optimizer& OUU1Optimizer::operator=(const OUU1Optimizer &)
{
  printf("OUU1Optimizer operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

