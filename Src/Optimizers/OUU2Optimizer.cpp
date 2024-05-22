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
// Functions for the class OUU2Optimizer
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

#include "PDFManager.h"
#include "OUU2Optimizer.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "sysdef.h"
#include "Sampling.h"

// ************************************************************************
// External functions
// ------------------------------------------------------------------------
#ifdef HAVE_BOBYQA
extern "C" void bobyqa_(int *,int *, double *, double *, double *, double *,
                        double *, int *, int *, double*);
extern "C" void obobyqa_(int *,int *, double *, double *, double *, double *,
                        double *, int *, int *, double*);
#endif

// ************************************************************************
// Internal 'global' variables
// ------------------------------------------------------------------------
void    *psOUU2Obj_=NULL;
int     psOUU2M1_=-1;
int     psOUU2M2_=-1;
int     psOUU2M3_=-1;
int     psOUU2nSamples_=-1;
int     psOUU2UserOpt_ = 0;
int     psOUU2UseRS_ = 0;
double  *psOUU2SamInputs_=NULL;
double  *psOUU2SamOutputs_=NULL;
double  *psOUU2SamProbs_=NULL;
double  *psOUU2XValues_=NULL;
double  *psOUU2WValues_=NULL;
double  *psOUU2OptimalX_=NULL;
double  psOUU2OptimalY_=0.0;
FuncApprox *psOUU2faPtr_=NULL;
int     psOUU2LargeSampleSize_=0;
double  *psOUU2LargeSamInputs_=NULL;
double  *psOUU2LargeSamOutputs_=NULL;
int     psOUU2Mode_=1;
int     psOUU2Percentile_=0;
double  psOUU2StdevMultiplier_=0;
int     psOUU2MasterMode_=0;
int     psOUU2Counter_=0;
int     psOUU2Parallel_=0;
#define psOUU2MaxSaved_ 10000
int     psOUU2NSaved_=0;
double  psOUU2SaveX_[psOUU2MaxSaved_*10];
double  psOUU2SaveY_[psOUU2MaxSaved_*10];
int     psOUU2EnsembleEval_=0;

#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// ************************************************************************
// resident function to perform evaluation 
// ------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" 
{
#endif
   /* ****************************************************************** */
   /* This function will evaluate the simulation given M2 variables by   */
   /* the optimizer appended with the M3 variables fixed in ouu2evalfunc */
   /* -------------------------------------------------------------------*/
   void *ouu2evalfunc2_(int *nInps, double *XValues, double *YValue)
   {
      int    ii, kk, M1, M2, M3, M, iOne=1, found, funcID;
      double ddata;
      oData  *odata;

      //**/ ------ fetch data ------
      odata = (oData *) psOUU2Obj_;
      M1    = psOUU2M1_;
      M2    = psOUU2M2_;
      M3    = psOUU2M3_;
      M     = M1 + M2 + M3;

      //**/ ------ set up to run simulation ------
      funcID = odata->numFuncEvals_;
      for (ii = 0; ii < M2; ii++) psOUU2XValues_[M1+ii] = XValues[ii];
      for (ii = 0; ii < M3; ii++) psOUU2XValues_[M1+M2+ii] = psOUU2WValues_[ii];

      //**/ ------ search to see if it has already been evaluated ------
      found = 0;
      for (ii = 0; ii < psOUU2NSaved_; ii++)
      {
         for (kk = 0; kk < M; kk++)
            if (PABS(psOUU2SaveX_[ii*M+kk]-psOUU2XValues_[kk])>1.0e-14) break;
         if (kk == M)
         {
            found = 1;
            if (odata->outputLevel_ > 2)
               printf("OUU2Optimizer: simulation results reuse.\n");
            ddata = psOUU2SaveY_[ii];
            break;
         }
      }

      //**/ ------ run simulation ------
      if (found == 0)
      {
         odata->funcIO_->evaluate(funcID,M,psOUU2XValues_,iOne,&ddata,0);
         odata->numFuncEvals_++;
         if ((psOUU2NSaved_+1)*M < psOUU2MaxSaved_*10)
         {
            for (kk = 0; kk < M; kk++)
               psOUU2SaveX_[psOUU2NSaved_*M+kk] = psOUU2XValues_[kk];
            psOUU2SaveY_[psOUU2NSaved_] = ddata;
            psOUU2NSaved_++;
         }
      }
      (*YValue) = ddata;

      //**/ ------ diagnostics ------
      if (ddata < psOUU2OptimalY_)
      {
         psOUU2OptimalY_ = ddata;
         for (ii = 0; ii < M; ii++) psOUU2OptimalX_[ii] = psOUU2XValues_[ii];
         if (odata->outputLevel_ > 2)
         {
            printf("    OUU2Optimizer inner loop New Ymin = %16.8e (%d)\n",
                   ddata, odata->numFuncEvals_);
            for (ii = M1; ii < M1+M2; ii++) 
               printf("     Input %3d = %e\n", ii+1, psOUU2OptimalX_[ii]);
         }
      }
      return NULL;
   }

   /* ****************************************************************** */
   /* This function will evaluate the simulation with variables          */
   /* corresponding only to the design variables (stage 1)               */
   /* It runs a number of optimizations to find the best M2 variable     */
   /* values for each of the sample points corresponding to M3 variables */
   /* -------------------------------------------------------------------*/
   void *ouu2evalfunc_(int *nInps, double *XValues, double *YValue)
   {
      int    ii, kk, ss, funcID, M, M1, M2, M3, bobyqaFlag=2223, nPts;
      int    maxfun, iOne=1, *readys, status;
      double rhobeg, rhoend, ddata, *workArray, mean, stdev, *XLocal;
      char   winput[1000];
      oData  *odata;
      FILE   *fp=NULL;

      //**/ ------ fetch data (*nInp) = M1 ------
      odata = (oData *) psOUU2Obj_;
      M1    = psOUU2M1_;
      M2    = psOUU2M2_;
      M3    = psOUU2M3_;
      M     = M1 + M2 + M3;

      //**/ ------ set up for ensemble optimization ------
      rhobeg = odata->upperBounds_[M1] - odata->lowerBounds_[M1];
      for (ii = 1; ii < M2; ii++)
      {
         ddata = odata->upperBounds_[M1+ii] - odata->lowerBounds_[M1+ii];
         if (ddata < rhobeg) rhobeg = ddata;
      }
      rhobeg *= 0.5;
      rhoend = rhobeg * odata->tolerance_;
      if (rhobeg < rhoend) rhoend = rhobeg * 1.0e-6;
      maxfun = odata->maxFEval_;
      nPts = (M2 + 1) * (M2 + 2) / 2;
      kk = (nPts + 5) * (nPts + M2) + 3 * M2 * (M2 + 5) / 2 + 1;
      workArray = (double *) malloc(kk * sizeof(double));
      XLocal = (double *) malloc(M*sizeof(double));
      readys = (int *) malloc(psOUU2nSamples_*sizeof(int));

      //**/ ------ run ensemble optimization ------
      for (ii = 0; ii < M1; ii++) psOUU2XValues_[ii] = XValues[ii];
      for (ii = 0; ii < M2; ii++)
      {
         ddata = 0.5*(odata->lowerBounds_[M1+ii]+odata->upperBounds_[M1+ii]);
         XLocal[ii] = ddata;
      } 
      if (odata->outputLevel_ > 1) 
      {
         printf("OUU2Optimizer: Outer optimization loop FuncEval.\n");
         for (ii = 0; ii < M1; ii++)
            printf("    Current input %3d = %e\n", ii+1, XValues[ii]);
      }
      if (psOUU2EnsembleEval_ == 0)
      {
         for (ss = 0; ss < psOUU2nSamples_; ss++)
         {
            if (odata->outputLevel_ > 3) 
               printf("OUU2Optimizer sample %d (of %d)\n",ss+1,psOUU2nSamples_); 
            readys[ss] = -1;
            psOUU2OptimalY_ = PSUADE_UNDEFINED;
            for (ii = 0; ii < M3; ii++)
               psOUU2WValues_[ii] = psOUU2SamInputs_[ss*M3+ii];
            //**/ ------ if user does not provide 2nd level optimizer ------
            if (psOUU2UserOpt_ == 0)
            {
               bobyqaFlag = 1111;
#ifdef HAVE_BOBYQA
               obobyqa_(&M2, &nPts, XLocal, &(odata->lowerBounds_[M1]), 
                       &(odata->upperBounds_[M1]),&rhobeg, &rhoend, 
                       &bobyqaFlag, &maxfun, workArray);
#else
               printf("OUU2Optimzer ERROR: Bobyqa not installed.\n");
               exit(1);
#endif
               psOUU2SamOutputs_[ss] = psOUU2OptimalY_;
               if (odata->outputLevel_ > 3) 
                  printf("OUU2Optimizer (1) sample %d completed, best Y = %e\n",
                         ss+1,psOUU2SamOutputs_[ss]); 
            }
            else
            //**/ ------ if user provides 2nd level optimizer ------
            {
               for (ii = 0; ii < M1; ii++) XLocal[ii] = psOUU2XValues_[ii];
               for (ii = 0; ii < M2; ii++)
               {
                  XLocal[M1+ii] = 0.5 * (odata->lowerBounds_[M1+ii] + 
                                         odata->upperBounds_[M1+ii]);
                  psOUU2XValues_[M1+ii] = XLocal[M1+ii];
               }
               for (ii = 0; ii < M3; ii++)
                  XLocal[M1+M2+ii] = psOUU2SamInputs_[ss*M3+ii];
               funcID = psOUU2Counter_ * psOUU2nSamples_ + ss;
               readys[ss] = odata->funcIO_->evaluate(funcID,M,XLocal,iOne,&ddata,0);
               psOUU2SamOutputs_[ss] = ddata;
               for (ii = 0; ii < M; ii++) psOUU2OptimalX_[ii] = psOUU2XValues_[ii];
               if (odata->outputLevel_ > 3 && readys[ss] == 0) 
                  printf("OUU2Optimizer (2) sample %d completed, best Y = %e\n",
                      ss+1,psOUU2SamOutputs_[ss]); 
               if (psOUU2Parallel_ == 0) odata->numFuncEvals_++;
            }
         } 

         //**/ if asynchronous mode is on, additonal processing
         if (psOUU2UserOpt_ == 1 && psOUU2Parallel_ == 1)
         {
            for (ss = 0; ss < psOUU2nSamples_; ss++)
            {
               funcID = psOUU2Counter_ * psOUU2nSamples_ + ss;
               readys[ss] = odata->funcIO_->evaluate(funcID,M,XLocal,iOne,&ddata,2);
               while (readys[ss] != 0)
               {
#ifdef WINDOWS
                  Sleep(1000);
#else
                  sleep(1);
#endif
                  readys[ss] = odata->funcIO_->evaluate(funcID,M,XLocal,iOne,&ddata,2);
               }
               psOUU2SamOutputs_[ss] = ddata;
               if (odata->outputLevel_ > 3)
                  printf("OUU2Optimizer (3) sample %d completed, best Y = %e\n",
                         ss+1,psOUU2SamOutputs_[ss]); 
               odata->numFuncEvals_++;
            }
         }
 
         //**/ ------ optional: spit out the response surface data ------
         if (psOUU2MasterMode_ == 1)
         {
            fp = fopen("ouu2Sample", "w");
            fprintf(fp, "%d %d 1\n", psOUU2nSamples_, M3);
            for (ss = 0; ss < psOUU2nSamples_; ss++)
            {
               for (ii = 0; ii < M3; ii++)
                  fprintf(fp, "%24.16e ", psOUU2SamInputs_[ss*M3+ii]);
               fprintf(fp, "%24.16e\n", psOUU2SamOutputs_[ss]);
            }
            fclose(fp);
            printf("OUU2Optimizer: file ouu2Sample ready for your inspection.\n");
            printf("Enter 'n' to go to the next one or 'q' to continue without stop : ");
            scanf("%s", winput);
            if (winput[0] == 'q') psOUU2MasterMode_ = 0;
         }
      }
      else
      {
         for (ii = 0; ii < M1; ii++) XLocal[ii] = psOUU2XValues_[ii];
         for (ss = 0; ss < psOUU2nSamples_; ss++)
         {
            for (ii = 0; ii < M1; ii++) 
               psOUU2XValues_[ss*M+ii] = XLocal[ii];
            for (ii = 0; ii < M2; ii++)
               psOUU2XValues_[ss*M+M1+ii] = 0.5*(odata->lowerBounds_[M1+ii] + 
                                                 odata->upperBounds_[M1+ii]);
            for (ii = 0; ii < M3; ii++)
               psOUU2XValues_[ss*M+M1+M2+ii] = psOUU2SamInputs_[ss*M3+ii];
         }
         odata->funcIO_->ensembleEvaluate(psOUU2nSamples_,M,psOUU2XValues_,
                                          iOne,psOUU2SamOutputs_,funcID);
         odata->numFuncEvals_ += psOUU2nSamples_;
         //**/ ------ save the data ------
         for (ss = 0; ss < psOUU2nSamples_; ss++)
         {
            if ((psOUU2NSaved_+1)*M < psOUU2MaxSaved_*10)
            {
               for (kk = 0; kk < M; kk++)
                  psOUU2SaveX_[psOUU2NSaved_*M+kk] = psOUU2XValues_[ss*M+kk];
               psOUU2SaveY_[psOUU2NSaved_] = psOUU2SamOutputs_[ss];
               psOUU2NSaved_++;
            }
         }
         for (ii = 0; ii < M; ii++) psOUU2OptimalX_[ii] = psOUU2XValues_[ii];
      }

      //**/ ------ analyze optimization results ------
      if (psOUU2UseRS_ == 0)
      {
         if (psOUU2Mode_ == 1 || psOUU2Mode_ == 2)
         {
            mean = 0.0;
            for (ss = 0; ss < psOUU2nSamples_; ss++) 
               mean += psOUU2SamOutputs_[ss] * psOUU2SamProbs_[ss];
            (*YValue) = mean;
         }
         if (psOUU2Mode_ == 2 && psOUU2StdevMultiplier_ != 0.0)
         {
            stdev = 0.0;
            for (ss = 0; ss < psOUU2nSamples_; ss++) 
               stdev += pow(psOUU2SamOutputs_[ss]-mean, 2.0)*psOUU2SamProbs_[ss];
            (*YValue) = mean + psOUU2StdevMultiplier_ * stdev;
         }
         if (psOUU2Mode_ == 3)
         {
            sortDbleList(psOUU2nSamples_, psOUU2SamOutputs_);
            kk = (int) ((1.0 - psOUU2Percentile_) * psOUU2nSamples_);
            if (kk >= psOUU2nSamples_) kk = psOUU2nSamples_ - 1;
            (*YValue) = psOUU2SamOutputs_[kk];
         }
         if (psOUU2Mode_ == 4)
         {
            ddata = psOUU2SamOutputs_[0];
            for (ss = 1; ss < psOUU2nSamples_; ss++) 
               if (psOUU2SamOutputs_[ss] < ddata)
                  ddata = psOUU2SamOutputs_[ss];
            (*YValue) = ddata;
         }
      }
      else
      {
         if (odata->outputLevel_ > 2) 
            printf("OUU2Optimizer: computing objective with response surface.\n");
         
         status = psOUU2faPtr_->initialize(psOUU2SamInputs_,psOUU2SamOutputs_);
         psOUU2faPtr_->evaluatePoint(psOUU2LargeSampleSize_,
                            psOUU2LargeSamInputs_,psOUU2LargeSamOutputs_);
         if (psOUU2Mode_ == 1 || psOUU2Mode_ == 2)
         {
            mean = 0.0;
            for (ss = 0; ss < psOUU2LargeSampleSize_; ss++) 
               mean += psOUU2LargeSamOutputs_[ss];
            mean /= (double) psOUU2LargeSampleSize_;
            (*YValue) = mean;
         }
         if (psOUU2Mode_ == 2 && psOUU2StdevMultiplier_ != 0.0)
         {
            stdev = 0.0;
            for (ss = 0; ss < psOUU2LargeSampleSize_; ss++) 
               stdev += pow(psOUU2LargeSamOutputs_[ss] - mean, 2.0);
            stdev /= (double) psOUU2LargeSampleSize_;
            (*YValue) = mean + psOUU2StdevMultiplier_ * stdev;
         }
         if (psOUU2Mode_ == 3)
         {
            sortDbleList(psOUU2LargeSampleSize_, psOUU2LargeSamOutputs_);
            kk = (int) ((1.0 - psOUU2Percentile_) * psOUU2LargeSampleSize_);
            if (kk >= psOUU2LargeSampleSize_) kk = psOUU2LargeSampleSize_ - 1;
            (*YValue) = psOUU2LargeSamOutputs_[kk];
         }
         if (psOUU2Mode_ == 4)
         {
            ddata = psOUU2LargeSamOutputs_[0];
            for (ss = 1; ss < psOUU2LargeSampleSize_; ss++) 
               if (psOUU2LargeSamOutputs_[ss] < ddata)
                  ddata = psOUU2LargeSamOutputs_[ss];
            (*YValue) = ddata;
         }
         if (odata->outputLevel_ > 2) 
         {
            printf("OUU2Optimizer: computed objective (with RS) = %e.\n",
                   (*YValue));
            if (psOUU2Mode_ == 0)
            {
               ddata = 0.0;
               for (ss = 0; ss < psOUU2nSamples_; ss++) 
                  ddata += psOUU2SamOutputs_[ss];
               ddata /= (double) psOUU2nSamples_;
               printf("OUU2Optimizer: computed mean (w/o  RS) = %e.\n",ddata);
            }
         }
      }
 
      //**/ ------ store optimal information ------
      if (mean < odata->optimalY_)
      {
         odata->optimalY_ = mean;
         for (ii = 0; ii < M1+M2+M3; ii++) 
            odata->optimalX_[ii] = psOUU2OptimalX_[ii];
         if (odata->outputLevel_ > 0)
         {
            printf("    OUU2Optimizer outer loop new Ymin = %16.8e\n",mean);
            if (psOUU2UserOpt_ == 0)
            {
               for (ii = 0; ii < M1+M2; ii++) 
                  printf("        Input %3d at min = %e\n", ii+1, 
                         odata->optimalX_[ii]);
            }
         }
      }
      psOUU2Counter_++;

      //**/ ------ clean up and return ------
      free(readys);
      free(XLocal);
      free(workArray);
      return NULL;
   }
#ifdef __cplusplus
}
#endif

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
OUU2Optimizer::OUU2Optimizer()
{
   psOUU2M1_ = -1;
   psOUU2M2_ = -1;
   psOUU2M3_ = -1;
   psOUU2nSamples_ = -1;
   psOUU2UserOpt_  = 0;
   psOUU2Obj_ = NULL;
   psOUU2SamInputs_ = NULL;
   psOUU2SamOutputs_ = NULL;
   psOUU2XValues_ = NULL;
   psOUU2WValues_ = NULL;
   psOUU2OptimalX_ = NULL;
   psOUU2OptimalY_ = 0.0;
   psOUU2UseRS_ = 0;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
OUU2Optimizer::~OUU2Optimizer()
{
}

// ************************************************************************
// optimize
// ------------------------------------------------------------------------
void OUU2Optimizer::optimize(oData *odata)
{
   int    nInputs, printLevel=0, ii, kk, maxfun, nPts=0, bobyqaFlag=2222;
   int    M1, M2, M3, iOne=1, iZero=0, printHeader=1, method=1;
   double *XValues, rhobeg=1.0, rhoend=1.0e-4, ddata, *workArray;
   char   pString[1000], lineIn[1000], filename[1000];
   FILE   *fp=NULL;
   static int currDriver=-1;

   //**/ ----------------------------------------------------------
   //**/ header and fetch data
   //**/ ----------------------------------------------------------
   if (psConfig_.MasterModeIsOn()) psOUU2MasterMode_ = 1;
   psConfig_.MasterModeOff();
   nInputs = odata->nInputs_;
   printLevel = odata->outputLevel_;
   if (psOUU2M1_+psOUU2M2_+psOUU2M3_ == nInputs && psOUU2M1_ > 0 &
       psOUU2M2_ > 0 && psOUU2M3_ > 0) printHeader = 0;
   if (printLevel >= 0 && printHeader == 1)
   {
      printAsterisks(PL_INFO, 0);
      printf("      Two-stage Optimization Under Uncertainty \n");
      printEquals(PL_INFO, 0);
      printf("This optimization capability solves the following problem:\n");
      printf("\n   minimize_X1 { Phi_X3 [ G(X1,X2,X3) ] } \n\n");
      printf("   subject to bound constraints on X1, X2, and X3\n\n");
      printf("   where\n");
      printf("      G(X1,X2*,X3) is minimum value from minimize_X2 F(X1,X2,X3)\n");
      printf("            given X1 and X3, and the minimum is at X2=X2* \n");
      printf("      F(X1,X2,X3) is your simulator (pointed to by 'driver')\n");
      printf("      minimize_X2 means to minimize with respect to X2\n");
      printf("      Phi_X3 is a functional of F(X1,X2,X3) with respect to X3\n");
      printf("      For example, Phi_X3 can be:\n");
      printf("      1. the mean of G(X1,X2,X3) with respect to X3\n"); 
      printf("      2. a linear combination of mean[G(X1,X2,X3)] and std. dev.\n"); 
      printf("      3. G(X1,X2,X3*) s.t. Prob(G(X1,X2,X3)>G(X1,X2,X3*)) = epsilon\n");
      printf("      Internally G(X1,X2*,X3) is computed by BOBYQA\n");
      printf("      You may substitue G(X1,X2,X3) with your own optimizer by\n");
      printf("          selecting the proper option below and point to your\n");
      printf("          optimizer using 'opt_driver = XX' in the PSUADE\n");
      printf("          input file (this option is available in opt_expert\n");
      printf("          mode only.\n");
      printEquals(PL_INFO, 0);
      printf("Total number of parameters M = %d\n", nInputs);
      printf("These parameters are to be divided into three groups:\n");
      printf("(1) Stage 1 optimization paramters X1 (M1 > 1) \n");
      printf("(2) Stage 2 optimization paramters X2 (M2 > 1) \n");
      printf("(3) uncertain paramters X3 (M3 > 1) \n");
      printf("Thus, the first M1+M2 parameters are considered to be\n");
      printf("optimization parameters, followed by M3 uncertain\n");
      printf("parameters so that M = M1 + M2 + M3.\n\n");
      printf("Use master mode to inspect response surface, if desired.\n");
      printf("IF YOU ARE READY TO MOVE ON, ENTER 'y' AND RETURN : ");
      scanf("%s", lineIn);
      if (lineIn[0] != 'y') 
      {
         odata->optimalY_ = 0;
         for (ii = 0; ii < nInputs; ii++) odata->optimalX_[ii] = 0.0;
         printf("OUU2Optimizer INFO: abrupt termination.\n");
         return;
      }
      printEquals(PL_INFO, 0);
   }
   if (psOUU2M1_+psOUU2M2_+psOUU2M3_ == nInputs && psOUU2M1_ > 0 &
       psOUU2M2_ > 0 && psOUU2M3_ > 0)
   {
      M1 = psOUU2M1_;
      M2 = psOUU2M2_;
      M3 = psOUU2M3_;
   }
   else
   {
      M1 = 0;
      while (M1 <= 0 || M1 >= nInputs-1)
      {
         printf("Enter M1 (between 1 and %d) : ", nInputs-2);
         scanf("%d", &M1);
      }
      psOUU2M1_ = M1;
      M2 = 0;
      while (M2 <= 0 || M1+M2 >= nInputs)
      {
         printf("Enter M2 (between 1 and %d) : ", nInputs-1-M1);
         scanf("%d", &M2);
      }
      fgets(lineIn, 500, stdin);
      psOUU2M2_ = M2;
      psOUU2M3_ = M3 = nInputs - M1 - M2;
   }
   if (printLevel >= 0)
   {
      printDashes(PL_INFO, 0);
      printf("Number of first  stage optimization parameters = %d\n", M1);
      printf("Number of second stage optimization parameters = %d\n", M2);
      printf("Number of uncertain parameters                 = %d\n", M3);
      printDashes(PL_INFO, 0);
   }

   //**/ ----------------------------------------------------------
   //**/ prepare for optimization 
   //**/ ----------------------------------------------------------
   for (ii = 0; ii < M1; ii++) odata->optimalX_[ii] = 0.0;
   odata->optimalY_ = 1.0e50;
   XValues = new double[M1+1];
   for (ii = 0; ii < M1; ii++) XValues[ii] = odata->initialX_[ii];
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
      printf("OUU2Optimizer WARNING: tolerance too large.\n");
      printf("                       tolerance reset to 1.0e-6.\n");
      rhoend = rhobeg * 1.0e-6;
   }

   maxfun = odata->maxFEval_;
   if ((odata->setOptDriver_ & 1))
   {
      printf("OUU2Optimizer: setting optimization simulation driver.\n");
      currDriver = odata->funcIO_->getDriver();
      odata->funcIO_->setDriver(1);
   }
   psOUU2Obj_= (void *) odata;
   printAsterisks(PL_INFO, 0);
   printf("OUU2Optimizer: max fevals = %d\n", odata->maxFEval_);
   printf("OUU2Optimizer: tolerance  = %e\n", odata->tolerance_);
   printEquals(PL_INFO, 0);

   //**/ ----------------------------------------------------------
   //**/ set up LH/FACT sample for uncertain parameters
   //**/ ----------------------------------------------------------
   psOUU2nSamples_ = 100;
   psOUU2UserOpt_ = 0;
   psOUU2UseRS_ = 0;
   method = 1;
   if (psConfig_.OptExpertModeIsOn())
   {
     printEquals(PL_INFO, 0);
     printf("Select which functional Phi_X3 to use: \n");
     printf("  1. the mean of G(X1,X2,X3) with respect to X3 (default)\n"); 
     printf("  2. mean of G(X1,X2,X3) + alpha * std dev of G(X1,X2,X3)\n"); 
     printf("  3. G(X1,X2,X3*) s.t. Prob(G(X1,X2,X3)>G(X1,X2,X3*)) = epsilon\n");
     printf("  4. min_X3 G(X1,X2,X3) given X1 and X2\n");
     snprintf(pString,100,"Enter your preferred functional (1, 2, 3 or 4) : ");
     psOUU2Mode_ = getInt(1, 4, pString);
     if (psOUU2Mode_ == 2)
     {
        snprintf(pString,100,"Enter your desired alpha : ");
        psOUU2StdevMultiplier_ = getDouble(pString);
     } 
     if (psOUU2Mode_ == 3)
     {
       psOUU2Percentile_ = 0.0;
       while (psOUU2Percentile_ <= 0.01 || psOUU2Percentile_ > 0.5) 
       { 
         snprintf(pString,100,"Enter your desired percentile : (0.01 - 0.5)");
         psOUU2Percentile_ = getDouble(pString);
       } 
     }
     printEquals(PL_INFO, 0);
     if (psOUU2Mode_ == 1 || psOUU2Mode_ == 2)
     {
       printf("OUU2Optimizer uses a sample of X3 to estimate the objective.\n");
       printf("Default sampling method = Latin hypercube\n");
       printf("Default sample size     = %d\n",psOUU2nSamples_);
       printf("Available sampling method: (1) LHS, (2) factorial, (3) your own.\n");
       snprintf(pString,100,"Select sampling method (1, 2 or 3) : ");
       method = getInt(1, 3, pString);
     }
     if (psOUU2Mode_ == 3 || psOUU2Mode_ == 4)
     {
       printf("OUU2Optimizer uses a sample of X3 to estimate the objective.\n");
       printf("Default sampling method = Latin hypercube\n");
       printf("Default sample size     = %d\n",psOUU2nSamples_);
       printf("Available sampling method: (1) LHS or (2) factorial.\n");
       snprintf(pString,100,"Select sampling method (1 or 2) : ");
       method = getInt(1, 2, pString);
     }
     if (method == 1)
     {
       snprintf(pString,100,
               "Enter your preferred sample size (>=10, <=1000) : ");
       psOUU2nSamples_ = getInt(10, 10000, pString);
     }
     else if (method == 2)
     {
       snprintf(pString,100,
               "Enter number of levels per variable (>=3, <=100) : ");
       psOUU2nSamples_ = getInt(3, 100, pString);
       kk = psOUU2nSamples_;
       for (ii = 1; ii < M3; ii++) psOUU2nSamples_ *= kk;
       printf("Factorial design has sample size = %d\n", psOUU2nSamples_);  
     }
     else if (method == 3)
     {
       printf("Data format of user sample:\n");
       printf("line 1: <nSamples> <nInputs> \n");
       printf("line 2: <sample 1 input 1> <input 2> ... <probability>\n");
       printf("line 3: <sample 2 input 1> <input 2> ... <probability>\n");
       printf("...\n");
       printf("Enter user sample file name : ");
       scanf("%s", filename);
       fp = fopen(filename, "r");
       if (fp == NULL)
       {
         printf("OUU2Optimizer ERROR: user sample file %s not found.\n",
                filename);
         exit(1);
       }
       fgets(lineIn, 5000, fp);
       sscanf(lineIn, "%d %d", &psOUU2nSamples_, &ii);
       if (psOUU2nSamples_ <= 5)
       {
         printf("OUU2Optimizer ERROR: user sample size should be > 5\n");
         fclose(fp);
         exit(1);
       } 
       if (ii != M3)
       {
         printf("OUU2Optimizer ERROR: user sample nInputs %d != %d\n",
                ii, M3);
         fclose(fp);
         exit(1);
       } 
       psOUU2SamInputs_  = new double[psOUU2nSamples_ * M3];
       psOUU2SamOutputs_ = new double[psOUU2nSamples_];
       psOUU2SamProbs_ = new double[psOUU2nSamples_];
       ddata = 0.0;
       for (ii = 0; ii < psOUU2nSamples_; ii++)
       {
         for (kk = 0; kk < M3; kk++)
           fscanf(fp, "%lg", &psOUU2SamInputs_[ii*M3+kk]);
         fscanf(fp, "%lg", &psOUU2SamProbs_[ii]);
         ddata += psOUU2SamProbs_[ii];
       }
       fclose(fp);
       printf("User sample has %d points\n", psOUU2nSamples_);
       printf("User sample CDF = %e (should be ~1)\n", ddata);
     }
     printEquals(PL_INFO, 0);
     printf("Use your own optimizer instead of BOBYQA ? (y or n) ");
     scanf("%s", lineIn);
     if (lineIn[0] == 'y')
     {
       psOUU2UserOpt_ = 1;
       printf("NOTE: Make sure your optimizer executable has been\n");
       printf("      assigned to 'opt_driver' and it optimizes with\n");
       printf("      respect to the %d-th to %d-th parameters.\n", 
              M1+1, M1+M2);
     }   
     if (method != 3)
     {
       printEquals(PL_INFO, 0);
       printf("Use response surface to compute statistics ? (y or n) ");
       scanf("%s", lineIn);
       if (lineIn[0] == 'y') psOUU2UseRS_ = 1;
     }
     printEquals(PL_INFO, 0);
     if (psOUU2UserOpt_ == 1)
     {
       printf("Use aux opt driver with ensemble run ? (y or n) ");
       scanf("%s", lineIn);
       if (lineIn[0] == 'y') psOUU2EnsembleEval_ = 1;
       printEquals(PL_INFO, 0);
     }
     if (psOUU2EnsembleEval_ == 0)
     {
       printf("Use synchronous (s) or asynchronous (p) mode for ensemble runs ? (s or p) ");
       scanf("%s", lineIn);
       if (lineIn[0] == 'p') psOUU2Parallel_ = 1;
       printEquals(PL_INFO, 0);
     }
   }

   //**/ ----------------------------------------------------------
   //**/ generate a sample for computing statistics
   //**/ ----------------------------------------------------------
   if (printLevel > 2) printf("OUU2Optimizer: generating a sample.\n");
   Sampling *sampler = NULL;
   int *samStates;
   if (method != 3)
   {
      if (method == 1)
           sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
      else sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_FACT);
      sampler->setPrintLevel(0);
      sampler->setInputBounds(M3, &(odata->lowerBounds_[M1+M2]), 
                              &(odata->upperBounds_[M1+M2]));
      sampler->setOutputParams(iOne);
      sampler->setSamplingParams(psOUU2nSamples_, iOne, iZero);
      sampler->initialize(0);
      psOUU2nSamples_   = sampler->getNumSamples();
      psOUU2SamInputs_  = new double[psOUU2nSamples_ * M3];
      psOUU2SamOutputs_ = new double[psOUU2nSamples_];
      psOUU2SamProbs_   = new double[psOUU2nSamples_];
      samStates = new int[psOUU2nSamples_];
      sampler->getSamples(psOUU2nSamples_, M3, iOne, psOUU2SamInputs_,
                          psOUU2SamOutputs_, samStates);
      delete [] samStates;
      delete sampler;
      for (ii = 0; ii < psOUU2nSamples_; ii++) 
         psOUU2SamProbs_[ii] = 1.0 / (double) psOUU2nSamples_;
   }
   psOUU2WValues_ = new double[nInputs];
   psOUU2XValues_ = new double[nInputs*psOUU2nSamples_];
   psOUU2OptimalX_ = new double[nInputs];
   psOUU2faPtr_ = NULL;

   //**/ ----------------------------------------------------------
   //**/ set up response surface constructor and large sample
   //**/ ----------------------------------------------------------
   int        rstype=0, *inputPDFs=NULL;
   double     *inputMeans=NULL, *inputStdevs=NULL;
   PDFManager *pdfman=NULL;
   psVector   vecLB, vecUB, vecOut;
   pData      pdata;
   psMatrix   *corMat1, corMat2;
   if (psOUU2UseRS_ == 1)
   {
      if (printLevel > 2) 
         printf("OUU2Optimizer: setting up response surface\n");
      if (method == 2 && ((M3 == 2) || (M3 == 3)))
      {
         rstype = PSUADE_RS_SPLINES;
         printf("OUU2Optimizer: use Splines for factorial design\n");
      }
      else if (psOUU2nSamples_ > 300)
      {
         rstype = PSUADE_RS_MARS;
         printf("OUU2Optimizer: use MARS since nSamples > 300\n");
      }
      else
      {
         rstype = PSUADE_RS_KR;
         printf("OUU2Optimizer: use Kriging response surface\n");
      }
      psConfig_.InteractiveSaveAndReset();
      psOUU2faPtr_ = genFA(rstype, M3, -1, psOUU2nSamples_);
      psOUU2faPtr_->setBounds(&(odata->lowerBounds_[M1+M2]),
                              &(odata->upperBounds_[M1+M2]));
      psOUU2faPtr_->setOutputLevel(0);
      psConfig_.InteractiveRestore();

      odata->psIO_->getParameter("input_pdfs", pdata);
      inputPDFs = pdata.intArray_;
      pdata.intArray_ = NULL;
      if (inputPDFs == NULL)
      {
         inputPDFs = new int[nInputs];
         for (ii = 0; ii < M1+M2+M3; ii++) inputPDFs[ii] = 0;
      }
      kk = 0;
      for (ii = M1+M2; ii < M1+M2+M3; ii++) kk += inputPDFs[ii];
      if (kk == 0)
      {
         sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
         sampler->setPrintLevel(0);
         sampler->setInputBounds(M3, &(odata->lowerBounds_[M1+M2]), 
                                 &(odata->upperBounds_[M1+M2]));
         sampler->setOutputParams(iOne);
         psOUU2LargeSampleSize_ = 10000;
         sampler->setSamplingParams(psOUU2LargeSampleSize_, iOne, iZero);
         sampler->initialize(0);
         psOUU2LargeSampleSize_ = sampler->getNumSamples();
         psOUU2LargeSamInputs_= new double[psOUU2LargeSampleSize_*M3];
         psOUU2LargeSamOutputs_ = new double[psOUU2LargeSampleSize_];
         samStates = new int[psOUU2LargeSampleSize_];
         sampler->getSamples(psOUU2LargeSampleSize_, M3, iOne, 
                      psOUU2LargeSamInputs_,psOUU2LargeSamOutputs_,
                      samStates);
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
            for (ii = 0; ii < M1+M2+M3; ii++) inputMeans[ii] = 0;
         }
         pdata.dbleArray_ = NULL;
         odata->psIO_->getParameter("input_stdevs", pdata);
         inputStdevs = pdata.dbleArray_;
         if (inputStdevs == NULL)
         {
            inputStdevs = new double[nInputs];
            for (ii = 0; ii < M1+M2+M3; ii++) inputStdevs[ii] = 1;
         }
         pdata.dbleArray_ = NULL;
         odata->psIO_->getParameter("input_cor_matrix", pdata);
         corMat1 = (psMatrix *) pdata.psObject_;
         pdata.psObject_ = NULL;
         corMat2.setDim(M3,M3);
         for (ii = 0; ii < M3; ii++)
         {
            for (kk = 0; kk < M3; kk++)
            {
               ddata = corMat1->getEntry(M1+M2+ii,M1+M2+kk);
               corMat2.setEntry(ii,kk, ddata);
            }
         }
         pdfman = new PDFManager();
         pdfman->initialize(M3,&(inputPDFs[M1+M2]),&(inputMeans[M1+M2]),
                            &(inputStdevs[M1+M2]),corMat2,NULL,NULL);
         vecLB.load(M3, &(odata->lowerBounds_[M1+M2])); 
         vecUB.load(M3, &(odata->upperBounds_[M1+M2])); 
         psOUU2LargeSampleSize_ = 10000;
         vecOut.setLength(psOUU2LargeSampleSize_*M3);
         pdfman->genSample(psOUU2LargeSampleSize_, vecOut, vecLB, vecUB);
         psOUU2LargeSamInputs_= new double[psOUU2LargeSampleSize_*M3];
         psOUU2LargeSamOutputs_ = new double[psOUU2LargeSampleSize_];
         for (ii = 0; ii < psOUU2LargeSampleSize_*M3; ii++)
            psOUU2LargeSamInputs_[ii] = vecOut[ii];
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
   workArray = new double[(nPts+5)*(nPts+M1)+3*M1*(M1+5)/2+1];
   psOUU2Counter_ = 0;
   if (psOUU2Parallel_ == 1) odata->funcIO_->setAsynchronousMode();

#ifdef HAVE_BOBYQA
   //**/ ----- retrieve history -----
   fp = fopen("psuade_ouu2_history", "r");
   if (fp != NULL)
   {
      printf("OUU2Optimizer history file found. Use it? (y or n) ");
      scanf("%s", lineIn);
      if (lineIn[0] != 'y') fclose(fp);
      else
      {
         psOUU2NSaved_ = 0;
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
                  fscanf(fp, "%lg",&psOUU2SaveX_[psOUU2NSaved_*nInputs+ii]);
               fscanf(fp, "%lg",&psOUU2SaveY_[psOUU2NSaved_]);
               psOUU2NSaved_++;
            }
            if (((psOUU2NSaved_+1)*nInputs > psOUU2MaxSaved_*10) ||
                psOUU2NSaved_ > psOUU2MaxSaved_) break;
         }
         fclose(fp);
      }
   }

   //**/ ----- run optimizer -----
   for (ii = 0; ii < M1; ii++) 
      printf("OUU2Optimizer initial X %3d = %e\n", ii+1, XValues[ii]);
   bobyqa_(&M1, &nPts, XValues, odata->lowerBounds_,
           odata->upperBounds_, &rhobeg, &rhoend, &bobyqaFlag, &maxfun, 
           workArray);
   printf("OUU2Optimizer: total number of evaluations = %d\n",
           odata->numFuncEvals_);

   //**/ ----- save history -----
   if (psOUU2NSaved_ > 0)
   {
      fp = fopen("psuade_ouu2_history","w");
      if (fp != NULL)
      {
         for (ii = 0; ii < psOUU2NSaved_; ii++)
         {
            fprintf(fp, "999 %d ", nInputs);
            for (kk = 0; kk < nInputs; kk++)
               fprintf(fp, "%24.16e ", psOUU2SaveX_[ii*nInputs+kk]);
            fprintf(fp, "%24.16e\n", psOUU2SaveY_[ii]);
         }
         fclose(fp);
      }
      printf("OUU2Optimizer: history saved in psuade_ouu2_history\n");
   }
#else
   printf("ERROR : Bobyqa optimizer not installed.\n");
   exit(1);
#endif
   if (psOUU2Parallel_ == 1) odata->funcIO_->setSynchronousMode();

   //**/ ----------------------------------------------------------
   //**/ set return values and clean up 
   //**/ ----------------------------------------------------------
   if ((odata->setOptDriver_ & 2) && currDriver >= 0)
   {
      printf("OUU2Optimizer INFO: reverting to original simulation driver.\n");
      odata->funcIO_->setDriver(currDriver);
   }
   delete [] XValues;
   delete [] workArray;
   if (psOUU2SamInputs_ != NULL) delete [] psOUU2SamInputs_;
   if (psOUU2SamOutputs_ != NULL) delete [] psOUU2SamOutputs_;
   if (psOUU2SamProbs_ != NULL) delete [] psOUU2SamProbs_;
   if (psOUU2LargeSamInputs_ != NULL) delete [] psOUU2LargeSamInputs_;
   if (psOUU2LargeSamOutputs_ != NULL) delete [] psOUU2LargeSamOutputs_;
   if (psOUU2XValues_ != NULL) delete [] psOUU2XValues_;
   if (psOUU2WValues_ != NULL) delete [] psOUU2WValues_;
   if (psOUU2OptimalX_ != NULL) delete [] psOUU2OptimalX_;
   if (psOUU2faPtr_ != NULL) delete psOUU2faPtr_;
   psOUU2OptimalX_ = NULL;
   psOUU2SamInputs_ = NULL;
   psOUU2SamOutputs_ = NULL;
   psOUU2SamProbs_ = NULL;
   psOUU2WValues_ = NULL;
   psOUU2XValues_ = NULL;
   psOUU2faPtr_ = NULL;
   psOUU2LargeSamInputs_ = NULL;
   psOUU2LargeSamOutputs_ = NULL;
   odata->intData_ = M1;
}

// ************************************************************************
// assign operator
// ------------------------------------------------------------------------
OUU2Optimizer& OUU2Optimizer::operator=(const OUU2Optimizer &)
{
   printf("OUU2Optimizer operator= ERROR: operation not allowed.\n");
   exit(1);
   return (*this);
}

