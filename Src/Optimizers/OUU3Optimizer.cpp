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
// Functions for the class OUU3Optimizer
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
#include "OUU3Optimizer.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "sysdef.h"
#include "Sampling.h"

// ************************************************************************
// External functions
// ------------------------------------------------------------------------
extern "C" void bobyqa_(int *,int *, double *, double *, double *, double *,
                        double *, int *, int *, double*);
extern "C" void obobyqa_(int *,int *, double *, double *, double *, double *,
                        double *, int *, int *, double*);

// ************************************************************************
// Internal 'global' variables
// ------------------------------------------------------------------------
void    *psOUU3Obj_=NULL;
int     psOUU3M1_=-1;
int     psOUU3M2_=-1;
int     psOUU3M3_=-1;
int     psOUU3M4_=-1;
int     psOUU3X3nSamples_=-1;
int     psOUU3X4nSamples_=-1;
int     psOUU3UserOpt_ = 0;
int     psOUU3UseRS_ = 0;
double  *psOUU3X3SamInputs_=NULL;
double  *psOUU3X4SamInputs_=NULL;
double  *psOUU3SamOutputs_=NULL;
double  *psOUU3SamProbs_=NULL;
double  *psOUU3XValues_=NULL;
double  *psOUU3WValues_=NULL;
double  *psOUU3OptimalX_=NULL;
double  psOUU3OptimalY_=0.0;
FuncApprox *psOUU3faPtr_=NULL;
int     psOUU3LargeSampleSize_=0;
double  *psOUU3LargeSamInputs_=NULL;
double  *psOUU3LargeSamOutputs_=NULL;
int     psOUU3Mode_=1;
int     psOUU3Percentile_=0;
double  psOUU3StdevMultiplier_=0;
int     psOUU3Counter_=0;
int     psOUU3Parallel_=0;
#define psOUU3MaxSaved_ 10000
int     psOUU3NSaved_=0;
double  psOUU3SaveX_[psOUU3MaxSaved_*10];
double  psOUU3SaveY_[psOUU3MaxSaved_];
int     psOUU3EnsembleEval_=0;
int     *psOUU3InputTypes_=NULL;

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
   void *ouu3evalfunc2_(int *nInps, double *XValues, double *YValue)
   {
      int    ii, kk, M1, M2, M3, M4, M, index, iOne=1, found, funcID;
      double ddata;
      oData  *odata;

      //**/ ------ fetch data ------
      odata = (oData *) psOUU3Obj_;
      M1    = psOUU3M1_;
      M2    = psOUU3M2_;
      M3    = psOUU3M3_;
      M4    = psOUU3M4_;
      M     = M1 + M2 + M3 + M4;
      funcID = odata->numFuncEvals_;

      //**/ ------ set the operating variables ------
      //**/ XValues compacted (values for X1 should
      //**/ have been set
      //**/ psOUU3XValues in order
      index = 0;
      for (ii = 0; ii < M; ii++) 
      {
         if (psOUU3InputTypes_[ii] == 1)
         {
            psOUU3XValues_[ii] = XValues[index];
            index++;
         }
      }
      //**/ ------ set the uncertain variables ------
      //**/ psOUU3WValues compacted
      //**/ psOUU3XValues in order
      index = 0;
      for (ii = 0; ii < M; ii++) 
      {
         if (psOUU3InputTypes_[ii] == 2)
         {
            psOUU3XValues_[ii] = psOUU3WValues_[index];
            index++;
         }
      }
      for (ii = 0; ii < M; ii++) 
      {
         if (psOUU3InputTypes_[ii] == 3)
         {
            psOUU3XValues_[ii] = psOUU3WValues_[index];
            index++;
         }
      }

      //**/ ------ search to see if it has already been evaluated ------
      found = 0;
      for (ii = 0; ii < psOUU3NSaved_; ii++)
      {
         for (kk = 0; kk < M; kk++)
            if (PABS(psOUU3SaveX_[ii*M+kk]-psOUU3XValues_[kk])>1.0e-14) 
               break;
         if (kk == M)
         {
            found = 1;
            if (odata->outputLevel_ > 2)
               printf("OUU3Optimizer: simulation results reuse.\n");
            ddata = psOUU3SaveY_[ii];
            break;
         }
      }

      //**/ ------ run simulation ------
      if (found == 0)
      {
         odata->funcIO_->evaluate(funcID,M,psOUU3XValues_,iOne,&ddata,0);
         odata->numFuncEvals_++;
         if ((psOUU3NSaved_+1)*M < psOUU3MaxSaved_*10)
         {
            for (kk = 0; kk < M; kk++)
               psOUU3SaveX_[psOUU3NSaved_*M+kk] = psOUU3XValues_[kk];
            psOUU3SaveY_[psOUU3NSaved_] = ddata;
            psOUU3NSaved_++;
         }
      }
      (*YValue) = ddata;

      //**/ ------ diagnostics ------
      if (ddata < psOUU3OptimalY_)
      {
         psOUU3OptimalY_ = ddata;
         for (ii = 0; ii < M; ii++) psOUU3OptimalX_[ii] = psOUU3XValues_[ii];
         if (odata->outputLevel_ > 2)
         {
            printf("    OUU3Optimizer inner loop New Ymin = %16.8e (%d)\n",
                   ddata, odata->numFuncEvals_);
            //**/ only display values of operating variables
            //**/ since this is the second stage optimization
            for (ii = 0; ii < M; ii++) 
               if (psOUU3InputTypes_[ii] == 1)
                  printf("     Input %3d = %e\n", ii+1, psOUU3OptimalX_[ii]);
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
   void *ouu3evalfunc_(int *nInps, double *XValues, double *YValue)
   {
      int    ii, kk, ss, funcID, M, M1, M2, M3, M4, bobyqaFlag=1112, nPts;
      int    maxfun, iOne=1, *readys, status, index, nSamp;
      double rhobeg, rhoend, ddata, *workArray, mean, stdev, *XLocal;
      double *lowers, *uppers;
      char   winput[1000];
      oData  *odata;
      FILE   *fp=NULL;

      //**/ ------ fetch data (*nInp) = M1 ------
      odata = (oData *) psOUU3Obj_;
      M1    = psOUU3M1_;
      M2    = psOUU3M2_;
      M3    = psOUU3M3_;
      M4    = psOUU3M4_;
      M     = M1 + M2 + M3 + M4;
      nSamp = psOUU3X3nSamples_ * psOUU3X4nSamples_;

      //**/ ------ set up for ensemble optimization ------
      //**/        for operating variables (=1)
      rhobeg = 1.0e35;
      for (ii = 0; ii < M; ii++)
      {
         if (psOUU3InputTypes_[ii] == 1)
         {
            ddata = odata->upperBounds_[ii] - odata->lowerBounds_[ii];
            if (ddata < rhobeg) rhobeg = ddata;
         }
      }
      rhobeg *= 0.5;
      rhoend = rhobeg * odata->tolerance_;
      if (rhobeg < rhoend) rhoend = rhobeg * 1.0e-6;
      maxfun = odata->maxFEval_;
      nPts = (M2 + 1) * (M2 + 2) / 2;
      kk = (nPts + 5) * (nPts + M2) + 3 * M2 * (M2 + 5) / 2 + 1;
      workArray = (double *) malloc(kk * sizeof(double));
      XLocal = (double *) malloc(M*sizeof(double));
      readys = (int *) malloc(nSamp*sizeof(int));
      lowers = (double *) malloc(M*sizeof(double));
      uppers = (double *) malloc(M*sizeof(double));

      //**/ ------ run ensemble optimization ------
      //**/   XValues (design variables - compacted)
      //**/   psOUU3XValues - all variables in order
      //**/   inject the design data to psOUU3XValues
      index = 0;
      for (ii = 0; ii < M; ii++)
      {
         if (psOUU3InputTypes_[ii] == 0)
         {
            psOUU3XValues_[ii] = XValues[index];
            index++;
         }
      }
      if (odata->outputLevel_ > 1) 
      {
         printf("OUU3Optimizer: Outer optimization loop FuncEval.\n");
         //**/ print out the design variables
         index = 0;
         for (ii = 0; ii < M; ii++)
         {
            if (psOUU3InputTypes_[ii] == 0)
            {
               printf("    Current Level 1 input %3d = %e\n", ii+1, 
                      XValues[index]);
               index++;
            }
         }
      }
      if (psOUU3EnsembleEval_ == 0)
      {
         //**/ recall: psOUU3XValues already stuffed with design & initial
         //**/         operating variable data
         for (ss = 0; ss < nSamp; ss++)
         {
            if (odata->outputLevel_ > 3) 
               printf("OUU3Optimizer sample %d (of %d)\n",ss+1,nSamp); 
            readys[ss] = -1;
            psOUU3OptimalY_ = PSUADE_UNDEFINED;
            //**/ initialize psOUU3WValues - uncertain variables (compacted)
            //**/ design values have been set already
            index = 0;
            for (ii = 0; ii < M; ii++)
            {
               if (psOUU3InputTypes_[ii] == 2)
               {
                  kk = ss / psOUU3X4nSamples_;
                  psOUU3WValues_[index] = psOUU3X3SamInputs_[kk*M3+index];
                  index++;
               }
            }
            index = 0;
            for (ii = 0; ii < M; ii++)
            {
               if (psOUU3InputTypes_[ii] == 3)
               {
                  kk = ss % psOUU3X4nSamples_;
                  psOUU3WValues_[index] = psOUU3X4SamInputs_[kk*M4+index];
                  index++;
               }
            }
            //**/ Given design and uncertain parameters (in psOUU3XValues
            //**/  - in order, and in psOUU3WValues - compacted)
            //**/ ------ if user does not provide 2nd level optimizer ------
            if (psOUU3UserOpt_ == 0)
            {
               index = 0;
               for (ii = 0; ii < M; ii++)
               {
                  if (psOUU3InputTypes_[ii] == 1)
                  {
                     lowers[index] = odata->lowerBounds_[ii];
                     uppers[index] = odata->upperBounds_[ii];
                     ddata = 0.5 * (lowers[index] + uppers[index]);
                     XLocal[index] = ddata;
                     index++;
                  }
               }
               bobyqaFlag = 1112;
               obobyqa_(&M2, &nPts, XLocal, lowers, uppers, &rhobeg, &rhoend, 
                        &bobyqaFlag, &maxfun, workArray);
               psOUU3SamOutputs_[ss] = psOUU3OptimalY_;
               if (odata->outputLevel_ > 3) 
                  printf("OUU3Optimizer (1) sample %d completed, best Y = %e\n",
                         ss+1,psOUU3SamOutputs_[ss]); 
            }
            else
            //**/ ------ if user provides 2nd level optimizer ------
            {
               //**/ inject design variable values into XLocal
               index = 0;
               for (ii = 0; ii < M; ii++)
               {
                  if (psOUU3InputTypes_[ii] == 0)
                  {
                     XLocal[ii] = psOUU3XValues_[index];
                     index++;
                  }
               }
               //**/ inject operating variable values (mid point)
               index = 0;
               for (ii = 0; ii < M; ii++)
               {
                  if (psOUU3InputTypes_[ii] == 1)
                  {
                     XLocal[ii] = 0.5 * (odata->lowerBounds_[ii] + 
                                         odata->upperBounds_[ii]);
                     index++;
                  }
               }
               //**/ inject uncertain variable values (from sample)
               index = 0;
               for (ii = 0; ii < M; ii++)
               {
                  if (psOUU3InputTypes_[ii] == 2)
                  {
                     kk = ss / psOUU3X4nSamples_;
                     XLocal[ii] = psOUU3X3SamInputs_[kk*M3+index];
                     index++;
                  }
               }
               index = 0;
               for (ii = 0; ii < M; ii++)
               {
                  if (psOUU3InputTypes_[ii] == 3)
                  {
                     kk = ss % psOUU3X4nSamples_;
                     XLocal[ii] = psOUU3X4SamInputs_[kk*M4+index];
                     index++;
                  }
               }
               //**/ evaluate
               funcID = psOUU3Counter_ * nSamp + ss;
               readys[ss] = odata->funcIO_->evaluate(funcID,M,XLocal,iOne,
                                                     &ddata,0);
               psOUU3SamOutputs_[ss] = ddata;
               for (ii = 0; ii < M; ii++) psOUU3OptimalX_[ii] = XLocal[ii];
               if (odata->outputLevel_ > 3 && readys[ss] == 0) 
                  printf("OUU3Optimizer (2) sample %d completed, best Y = %e\n",
                      ss+1,psOUU3SamOutputs_[ss]); 
               if (psOUU3Parallel_ == 0) odata->numFuncEvals_++;
            }
         } 

         //**/ if asynchronous mode is on, additonal processing
         if (psOUU3UserOpt_ == 1 && psOUU3Parallel_ == 1)
         {
            for (ss = 0; ss < nSamp; ss++)
            {
               funcID = psOUU3Counter_ * nSamp + ss;
               readys[ss] = odata->funcIO_->evaluate(funcID,M,XLocal,iOne,
                                                     &ddata,2);
               while (readys[ss] != 0)
               {
#ifdef WINDOWS
                  Sleep(1000);
#else
                  sleep(1);
#endif
                  readys[ss] = odata->funcIO_->evaluate(funcID,M,XLocal,
                                                        iOne,&ddata,2);
               }
               psOUU3SamOutputs_[ss] = ddata;
               if (odata->outputLevel_ > 3)
                  printf("OUU3Optimizer (3) sample %d completed, best Y = %e\n",
                         ss+1,psOUU3SamOutputs_[ss]); 
               odata->numFuncEvals_++;
            }
         }
 
      }
      else
      {
         //**/ put design variable values to XLocal (compacted)
         index = 0;
         for (ii = 0; ii < M; ii++)
         {
            if (psOUU3InputTypes_[ii] == 0)
            {
               XLocal[index] = psOUU3XValues_[ii];
               index++;
            }
         }
         for (ss = 0; ss < nSamp; ss++)
         {
            //**/ inject design values into psOUU3XValues 
            index = 0;
            for (ii = 0; ii < M; ii++) 
            {
               if (psOUU3InputTypes_[ii] == 0)
               {
                  psOUU3XValues_[ss*M+ii] = XLocal[index];
                  index++;
               }
            }
            //**/ inject operating values into psOUU3XValues
            for (ii = 0; ii < M; ii++)
            {
               if (psOUU3InputTypes_[ii] == 1)
               {
                  psOUU3XValues_[ss*M+ii] = 0.5*(odata->lowerBounds_[ii] + 
                                                 odata->upperBounds_[ii]);
               }
            }
            //**/ inject uncertain values into psOUU3XValues
            index = 0;
            for (ii = 0; ii < M; ii++)
            {
               if (psOUU3InputTypes_[ii] == 2)
               {
                  kk = ss / psOUU3X4nSamples_;
                  psOUU3XValues_[ss*M+ii] = psOUU3X3SamInputs_[kk*M3+index];
                  index++;
               }
            }
            index = 0;
            for (ii = 0; ii < M; ii++)
            {
               if (psOUU3InputTypes_[ii] == 3)
               {
                  kk = ss % psOUU3X4nSamples_;
                  psOUU3XValues_[ss*M+ii] = psOUU3X4SamInputs_[kk*M4+index];
                  index++;
               }
            }
         }
         odata->funcIO_->ensembleEvaluate(nSamp,M,psOUU3XValues_,
                                          iOne,psOUU3SamOutputs_,funcID);
         odata->numFuncEvals_ += nSamp;
         //**/ ------ save the data ------
         for (ss = 0; ss < nSamp; ss++)
         {
            if ((psOUU3NSaved_+1)*M < psOUU3MaxSaved_*10)
            {
               for (kk = 0; kk < M; kk++)
                  psOUU3SaveX_[psOUU3NSaved_*M+kk] = psOUU3XValues_[ss*M+kk];
               psOUU3SaveY_[psOUU3NSaved_] = psOUU3SamOutputs_[ss];
               psOUU3NSaved_++;
            }
         }
         for (ii = 0; ii < M; ii++) psOUU3OptimalX_[ii] = psOUU3XValues_[ii];
      }

      //**/ ------ analyze optimization results ------
      if (psOUU3UseRS_ == 0)
      {
         if (psOUU3Mode_ == 1 || psOUU3Mode_ == 2)
         {
            mean = 0.0;
            if (psOUU3X3nSamples_ == 1)
            {
               for (ss = 0; ss < nSamp; ss++) 
               {
                  index = ss / psOUU3X4nSamples_;
                  mean += psOUU3SamOutputs_[ss] / (double) nSamp;
               }
            }
            else
            {
               for (ss = 0; ss < nSamp; ss++) 
               {
                  index = ss / psOUU3X4nSamples_;
                  mean += psOUU3SamOutputs_[ss] / psOUU3X4nSamples_ * 
                          psOUU3SamProbs_[index];
               }
            }
            (*YValue) = mean;
         }
         if (psOUU3Mode_ == 2 && psOUU3StdevMultiplier_ != 0.0)
         {
            stdev = 0.0;
            for (ss = 0; ss < nSamp; ss++) 
            {
               index = ss / psOUU3X4nSamples_;
               if (psOUU3X3nSamples_ == 1)
                  stdev += pow(psOUU3SamOutputs_[ss]-mean,2.0)/(double) nSamp;
               else
                  stdev += pow(psOUU3SamOutputs_[ss]-mean, 2.0)*
                           psOUU3SamProbs_[index] / (double) psOUU3X4nSamples_;
            }
            (*YValue) = mean + psOUU3StdevMultiplier_ * sqrt(stdev);
         }
         if (psOUU3Mode_ == 3)
         {
            sortDbleList(nSamp, psOUU3SamOutputs_);
            kk = (int) ((1.0 - psOUU3Percentile_) * nSamp);
            if (kk >= nSamp) kk = nSamp - 1;
            (*YValue) = psOUU3SamOutputs_[kk];
         }
         if (psOUU3Mode_ == 4)
         {
            ddata = psOUU3SamOutputs_[0];
            for (ss = 1; ss < nSamp; ss++) 
               if (psOUU3SamOutputs_[ss] < ddata)
                  ddata = psOUU3SamOutputs_[ss];
            (*YValue) = ddata;
         }
      }
      else
      {
         if (odata->outputLevel_ > 2) 
            printf("OUU3Optimizer: computing objective with RS.\n");
         
         double totalMean=0.0, totalStdv=0.0, *resultStore;
         resultStore=(double *) malloc(psOUU3X3nSamples_*sizeof(double));
         for (ii = 0; ii < psOUU3X3nSamples_; ii++)
         {
            status = psOUU3faPtr_->initialize(psOUU3X4SamInputs_,
                                    &psOUU3SamOutputs_[ii*psOUU3X4nSamples_]);
            psOUU3faPtr_->evaluatePoint(psOUU3LargeSampleSize_,
                               psOUU3LargeSamInputs_,psOUU3LargeSamOutputs_);
            if (psOUU3Mode_ == 1 || psOUU3Mode_ == 2)
            {
               mean = 0.0;
               for (ss = 0; ss < psOUU3LargeSampleSize_; ss++) 
                  mean += psOUU3LargeSamOutputs_[ss];
               mean /= (double) psOUU3LargeSampleSize_;
               resultStore[ii] = mean;
            }
            if (psOUU3Mode_ == 2 && psOUU3StdevMultiplier_ != 0.0)
            {
               stdev = 0.0;
               for (ss = 0; ss < psOUU3LargeSampleSize_; ss++) 
                  stdev += pow(psOUU3LargeSamOutputs_[ss] - mean, 2.0);
               stdev /= (double) psOUU3LargeSampleSize_;
               resultStore[ii] = mean + psOUU3StdevMultiplier_ * stdev;
            }
            if (psOUU3Mode_ == 3)
            {
               sortDbleList(psOUU3LargeSampleSize_, psOUU3LargeSamOutputs_);
               kk = (int) ((1.0 - psOUU3Percentile_) * psOUU3LargeSampleSize_);
               if (kk >= psOUU3LargeSampleSize_) 
                  kk = psOUU3LargeSampleSize_ - 1;
               resultStore[ii] = psOUU3LargeSamOutputs_[kk];
            }
            if (psOUU3Mode_ == 4)
            {
               ddata = psOUU3LargeSamOutputs_[0];
               for (ss = 1; ss < psOUU3LargeSampleSize_; ss++) 
                  if (psOUU3LargeSamOutputs_[ss] < ddata)
                     ddata = psOUU3LargeSamOutputs_[ss];
               resultStore[ii] = ddata;
            }
         }
         if (psOUU3Mode_ == 1 || psOUU3Mode_ == 2)
         {
            mean = 0.0;
            for (ii = 0; ii < psOUU3X3nSamples_; ii++) 
            {
               if (psOUU3SamProbs_ == NULL)
                  mean += resultStore[ii] / (double) psOUU3X3nSamples_;
               else
                  mean += resultStore[ii] * psOUU3SamProbs_[ii];
            }
            (*YValue) = mean;
         }
         if (psOUU3Mode_ == 3)
         {
            sortDbleList(psOUU3X3nSamples_, resultStore);
            kk = (int) ((1.0 - psOUU3Percentile_) * psOUU3X3nSamples_);
            if (kk >= psOUU3X3nSamples_) kk = psOUU3X3nSamples_ - 1;
            (*YValue) = resultStore[kk];
         }
         if (psOUU3Mode_ == 4)
         {
            ddata = resultStore[0];
            for (ss = 1; ss < psOUU3X3nSamples_; ss++) 
               if (resultStore[ss] < ddata) ddata = resultStore[ss];
            (*YValue) = ddata;
         }
         if (odata->outputLevel_ > 2) 
         {
            printf("OUU3Optimizer: computed objective (with RS) = %e.\n",
                   (*YValue));
         }
         free(resultStore);
      }
 
      //**/ ------ store optimal information ------
      if (mean < odata->optimalY_)
      {
         odata->optimalY_ = mean;
         for (ii = 0; ii < M; ii++) 
            odata->optimalX_[ii] = psOUU3OptimalX_[ii];
         if (odata->outputLevel_ > 0)
         {
            printf("    OUU3Optimizer outer loop new Ymin = %16.8e\n",mean);
            if (psOUU3UserOpt_ == 0)
            {
               for (ii = 0; ii < M1+M2; ii++) 
                  printf("        Input %3d at min = %e\n", ii+1, 
                         odata->optimalX_[ii]);
            }
         }
      }
      psOUU3Counter_++;

      //**/ ------ clean up and return ------
      free(lowers);
      free(uppers);
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
OUU3Optimizer::OUU3Optimizer()
{
   psOUU3M1_ = -1;
   psOUU3M2_ = -1;
   psOUU3M3_ = -1;
   psOUU3X3nSamples_ = -1;
   psOUU3X4nSamples_ = -1;
   psOUU3UserOpt_  = 0;
   psOUU3Obj_ = NULL;
   psOUU3X3SamInputs_ = NULL;
   psOUU3X4SamInputs_ = NULL;
   psOUU3SamOutputs_ = NULL;
   psOUU3XValues_ = NULL;
   psOUU3WValues_ = NULL;
   psOUU3OptimalX_ = NULL;
   psOUU3OptimalY_ = 0.0;
   psOUU3UseRS_ = 0;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
OUU3Optimizer::~OUU3Optimizer()
{
}

// ************************************************************************
// optimize
// ------------------------------------------------------------------------
void OUU3Optimizer::optimize(oData *odata)
{
   int    nInputs, printLevel=0, ii, kk, maxfun, nPts=0, bobyqaFlag=1111;
   int    M1, M2, M3, M4, index, iOne=1, iZero=0, printHeader=1, count;
   double *XValues, rhobeg=1.0, rhoend=1.0e-4, ddata, *workArray;
   char   pString[1000], lineIn[1000], filename[1000];
   FILE   *fp=NULL;
   static int currDriver=-1;

   //**/ ----------------------------------------------------------
   //**/ header and fetch data
   //**/ ----------------------------------------------------------
   //**/ turning on master mode may cause problem. So turn ff
   psConfig_.MasterModeOff();
   nInputs = odata->nInputs_;
   printLevel = odata->outputLevel_;
   //**/ ----------------------------------------------------------
   //**/ if this has been instantiated before, no need to print header
   //**/ ----------------------------------------------------------
   if (psOUU3M1_+psOUU3M2_+psOUU3M3_ == nInputs && psOUU3M1_ > 0 &
       psOUU3M2_ > 0 && psOUU3M3_ > 0) printHeader = 0;
   if (printLevel >= 0 && printHeader == 1)
   {
      printAsterisks(PL_INFO, 0);
      printf("      Two-stage Optimization Under Uncertainty II \n");
      printEquals(PL_INFO, 0);
      printf("This optimization capability solves the following problem:\n");
      printf("\n   minimize_X1 { Phi_{X3,X4} [ G(X1,X2,X3,X4) ] } \n\n");
      printf("   subject to bound constraints on X1, X2, and X4;\n");
      printf("   X3 is a set of discrete parameters for which a sample\n\n");
      printf("   is to be provided by user; where\n");
      printf("      G(X1,X2*,X3,X4) is minimum value from minimize_X2\n");
      printf("            F(X1,X2,X3,X4) given X1, X3, and X4, and the\n");
      printf("            minimum is at X2=X2* \n");
      printf("      F(X1,X2,X3,X4) is your simulator (pointed to by 'driver')\n");
      printf("      minimize_X2 means to minimize with respect to X2\n");
      printf("      Phi_{X3,X4} is a functional of F(X1,X2,X3,X4) with \n");
      printf("            respect to X3,X4\n");
      printf("      For example, Phi_{X3,X4} can be:\n");
      printf("      1. the mean of G(X1,X2,X3,X4) with respect to X3 and X4\n"); 
      printf("      2. linear combination of mean[G(X1*,X2*,X3,X4)] & std. dev.\n"); 
      printf("      3. G(X1,X2,X3*,X4*) s.t. \n");
      printf("              Prob(G(X1,X2,X3,X4)>G(X1,X2,X3*,X4*)) = epsilon\n");
      printf("      Internally G(X1,X2*,X3,X4) is computed by BOBYQA\n");
      printf("      You may substitue G(X1,X2,X3,X4) with your own optimizer\n");
      printf("          by selecting the proper option below and point to\n");
      printf("          your optimizer using 'opt_driver = XX' in the PSUADE\n");
      printf("          input file (this option is available in opt_expert\n");
      printf("          mode only.\n");
      printEquals(PL_INFO, 0);
      printf("Total number of parameters M = %d\n", nInputs);
      printf("These parameters are to be divided into three groups:\n");
      printf("(1) Stage 1 optimization paramters X1 (M1 > 1) \n");
      printf("(2) Stage 2 optimization paramters X2 (M2 > 1) \n");
      printf("(3) uncertain paramters X3,X4 (M3+M4 > 1) \n");
      printf("Thus, the first M1+M2 parameters are considered to be\n");
      printf("optimization parameters, followed by M3+M4 uncertain\n");
      printf("parameters so that M = M1 + M2 + M3 + M4.\n\n");
      printf("Use master mode to inspect response surface, if desired.\n");
      printf("IF YOU ARE READY TO MOVE ON, ENTER 'y' AND RETURN : ");
      scanf("%s", lineIn);
      if (lineIn[0] != 'y') 
      {
         odata->optimalY_ = 0;
         for (ii = 0; ii < nInputs; ii++) odata->optimalX_[ii] = 0.0;
         printf("OUU3Optimizer INFO: abrupt termination.\n");
         return;
      }
      printEquals(PL_INFO, 0);
   }
   //**/ ----------------------------------------------------------
   //**/ fetch M1, M2, M3, and M4
   //**/ ----------------------------------------------------------
   if (psOUU3M1_+psOUU3M2_+psOUU3M4_ == nInputs && psOUU3M1_ > 0 &
       psOUU3M2_ > 0 && psOUU3M4_ > 0)
   {
      M1 = psOUU3M1_;
      M2 = psOUU3M2_;
      M3 = psOUU3M3_ - psOUU3M4_;
      M4 = psOUU3M4_;
   }
   else
   {
      M1 = 0;
      printf("M1 = number of design (level 1 optimization) parameters\n");
      while (M1 <= 0 || M1 >= nInputs-1)
      {
         printf("Enter M1 (between 1 and %d) : ", nInputs-2);
         scanf("%d", &M1);
      }
      printf("M2 = number of operating (level 2 optimization) parameters\n");
      M2 = 0;
      while (M2 <= 0 || M1+M2 >= nInputs)
      {
         printf("Enter M2 (between 1 and %d) : ", nInputs-1-M1);
         scanf("%d", &M2);
      }
      M3 = -1;
      printf("M3 = number of discrete (scenario) parameters\n");
      while (M3 < 0 || M1+M2+M3 > nInputs)
      {
         printf("Enter M3 (between 0 and %d) : ", nInputs-M1-M2);
         scanf("%d", &M3);
      }
      fgets(lineIn, 500, stdin);
      M4 = nInputs - M1 - M2 - M3;
      psOUU3M1_ = M1;
      psOUU3M2_ = M2;
      psOUU3M3_ = M3;
      psOUU3M4_ = M4;
   }
   if (printLevel >= 0)
   {
      printDashes(PL_INFO, 0);
      printf("Number of first  stage optimization parameters = %d\n", M1);
      printf("Number of second stage optimization parameters = %d\n", M2);
      printf("Number of discrete   uncertain parameters      = %d\n", M3);
      printf("Number of continuous uncertain parameters      = %d\n", M4);
      printDashes(PL_INFO, 0);
   }

   //**/ ----------------------------------------------------------
   //**/ request user to identify variable types
   //**/ ----------------------------------------------------------
   printf("In the following, please select type for each variable:\n");
   printf("  1. design variable\n");
   printf("  2. operating variable\n");
   printf("  3. discrete uncertain variable\n");
   printf("  4. continuous uncertain variable\n");
   psOUU3InputTypes_ = new int[nInputs];
   for (ii = 0; ii < nInputs; ii++)
   {
      snprintf(pString,100,"Type for variable %d ? ", ii+1);
      psOUU3InputTypes_[ii] = getInt(1,4,pString); 
      psOUU3InputTypes_[ii]--;
   }
   printEquals(PL_INFO, 0);
   for (ii = 0; ii < nInputs; ii++)
   {
      if (psOUU3InputTypes_[ii] == 0) 
         printf("Input %4d is a design parameter.\n",ii+1);
      else if (psOUU3InputTypes_[ii] == 1) 
         printf("Input %4d is an operating parameter.\n",ii+1);
      else if (psOUU3InputTypes_[ii] == 2) 
         printf("Input %4d is a discrete uncertain parameter.\n",ii+1);
      else if (psOUU3InputTypes_[ii] == 3) 
         printf("Input %4d is a continuous uncertain parameter.\n",ii+1);
   }
   int c1=0, c2=0, c3=0, c4=0;
   for (ii = 0; ii < nInputs; ii++)
   {
      if (psOUU3InputTypes_[ii] == 0) c1++;
      if (psOUU3InputTypes_[ii] == 1) c2++;
      if (psOUU3InputTypes_[ii] == 2) c3++;
      if (psOUU3InputTypes_[ii] == 3) c4++;
   }
   if (c1 != M1 || c2 != M2 || c3 != M3 || c4 != M4)
   {
      printf("OUU3Optimizer ERROR: input type counts do not match.\n");
      printf("    Number of type 1 = %d (expected = %d)\n",c1,M1);
      printf("    Number of type 2 = %d (expected = %d)\n",c2,M2);
      printf("    Number of type 3 = %d (expected = %d)\n",c3,M3);
      printf("    Number of type 4 = %d (expected = %d)\n",c4,M4);
      exit(1);
   }
   printEquals(PL_INFO, 0);

   //**/ ----------------------------------------------------------
   //**/ prepare for optimization: rhobeg, maxfun, initial guess, 
   //**/ initialize optimal data storage
   //**/ ----------------------------------------------------------
   for (ii = 0; ii < nInputs; ii++) odata->optimalX_[ii] = 0.0;
   odata->optimalY_ = 1.0e50;
   //**/ XValues stores the initial values for design parameters
   XValues = new double[M1+1];
   index = 0;
   for (ii = 0; ii < nInputs; ii++)
   {
      if (psOUU3InputTypes_[ii] == 0)
      {
         XValues[index] = odata->initialX_[ii];
         index++;
      }
   }
   //**/ compute the tolerance criterion (based on design parameters)
   rhobeg = 1e35;
   index = 0;
   for (ii = 0; ii < nInputs; ii++) 
   {
      if (psOUU3InputTypes_[ii] == 0)
      {
         ddata = odata->upperBounds_[ii] - odata->lowerBounds_[ii];
         if (ddata < rhobeg) rhobeg = ddata;
         index++;
      }
   }
   rhobeg *= 0.5;
   rhoend = rhobeg * odata->tolerance_;
   if (rhobeg < rhoend)
   {
      printf("OUU3Optimizer WARNING: tolerance too large.\n");
      printf("                       tolerance reset to 1.0e-6.\n");
      rhoend = rhobeg * 1.0e-6;
   }

   //**/ use the opt_driver code
   maxfun = odata->maxFEval_;
   if ((odata->setOptDriver_ & 1))
   {
      printf("OUU3Optimizer: setting optimization simulation driver.\n");
      currDriver = odata->funcIO_->getDriver();
      odata->funcIO_->setDriver(1);
   }
   psOUU3Obj_= (void *) odata;
   printAsterisks(PL_INFO, 0);
   printf("OUU3Optimizer: max fevals = %d\n", odata->maxFEval_);
   printf("OUU3Optimizer: tolerance  = %e\n", odata->tolerance_);
   printEquals(PL_INFO, 0);

   //**/ ----------------------------------------------------------
   //**/ set default: no user opt code, no RS
   //**/ then ask for type of objective function ==> psOUU3Mode_
   //**/ ----------------------------------------------------------
   psOUU3UserOpt_ = 0;
   psOUU3UseRS_ = 0;
   if (psConfig_.OptExpertModeIsOn())
   {
      printEquals(PL_INFO, 0);
      printf("Select which functional Phi_{X3,X4} to use: \n");
      printf("  1. mean of G(X1,X2,X3,X4) with respect to X3,X4 (default)\n"); 
      printf("  2. mean of G(X1,X2,X3,X4) + alpha * std dev of G(X1,X2,X3,X4)\n"); 
      printf("  3. G(X1,X2,X3*,X4*) such that \n");
      printf("           Prob(G(X1,X2,X3,X4)>G(X1,X2,X3*,X4*)) = epsilon\n");
      printf("  4. min_{X3,X4} G(X1,X2,X3,X4) given X1 and X2\n");
      snprintf(pString,100,"Enter your preferred functional (1, 2, 3 or 4) : ");
      psOUU3Mode_ = getInt(1, 4, pString);
      if (psOUU3Mode_ == 2)
      {
        snprintf(pString,100,"Enter your desired alpha : ");
        psOUU3StdevMultiplier_ = getDouble(pString);
      } 
      if (psOUU3Mode_ == 3)
      {
        psOUU3Percentile_ = 0.0;
        while (psOUU3Percentile_ <= 0.01 || psOUU3Percentile_ > 0.5) 
        { 
          snprintf(pString,100,"Enter your desired percentile : (0.01 - 0.5)");
          psOUU3Percentile_ = getDouble(pString);
        } 
      }
   }

   //**/ ----------------------------------------------------------
   //**/ request users for a sample for X3 
   //**/ ----------------------------------------------------------
   psOUU3X3nSamples_ = 1;
   if (M3 > 0)
   {
      printEquals(PL_INFO, 0);
      printf("A sample for X3 is needed from you. Data format should be :\n");
      printf("line 1: <nSamples> <nInputs> \n");
      printf("line 2: <sample 1 input 1> <input 2> ... <probability>\n");
      printf("line 3: <sample 2 input 1> <input 2> ... <probability>\n");
      printf("...\n");
      printf("Enter user sample file name : ");
      scanf("%s", filename);
      fgets(lineIn, 5000, stdin);
      fp = fopen(filename, "r");
      if (fp == NULL)
      {
         printf("OUU3Optimizer ERROR: user sample file %s not found.\n",
                 filename);
            exit(1);
      }
      fgets(lineIn, 5000, fp);
      sscanf(lineIn, "%d %d", &psOUU3X3nSamples_, &ii);
      if (psOUU3X3nSamples_ <= 5)
      {
         printf("OUU3Optimizer ERROR: user sample size should be > 5\n");
         fclose(fp);
         exit(1);
      } 
      if (ii != M3)
      {
         printf("OUU3Optimizer ERROR: user sample nInputs %d != %d\n",
                ii, M3);
         fclose(fp);
         exit(1);
      } 
      psOUU3X3SamInputs_ = new double[psOUU3X3nSamples_ * M3];
      psOUU3SamProbs_    = new double[psOUU3X3nSamples_];
      ddata = 0.0;
      for (ii = 0; ii < psOUU3X3nSamples_; ii++)
      {
         for (kk = 0; kk < M3; kk++)
            fscanf(fp,"%lg",&psOUU3X3SamInputs_[ii*M3+kk]);
         fscanf(fp, "%lg", &psOUU3SamProbs_[ii]);
         ddata += psOUU3SamProbs_[ii];
      }
      fclose(fp);
      printf("User sample for X3 has %d points\n", psOUU3X3nSamples_);
      printf("User sample for X3 CDF = %e (should be ~1)\n", ddata);
   }

   //**/ ----------------------------------------------------------
   //**/ ask for X4 sampling method (default - LHS) and size
   //**/ ----------------------------------------------------------
   if (printLevel > 2) printf("OUU3Optimizer: generating a sample for X4.\n");
   int    methodX4 = 1, *samStates;
   double *samOuts, *lowers, *uppers;
   psOUU3X4nSamples_ = 1;
   Sampling *sampler = NULL;
   if (M4 > 0)
   {
      psOUU3X4nSamples_ = 100;
      printEquals(PL_INFO, 0);
      printf("OUU3Optimizer uses a sample of X4 to estimate the objective.\n");
      printf("Default sampling method = Latin hypercube\n");
      printf("Default sample size     = %d\n",psOUU3X4nSamples_);
      printf("Available sampling method: (1) LHS or (2) factorial.\n");
      snprintf(pString,100,"Select sampling method (1 or 2) : ");
      methodX4 = getInt(1, 2, pString);
      if (methodX4 == 1)
      {
        snprintf(pString,100,
                "Enter your preferred sample size (>=10, <=1000) : ");
        psOUU3X4nSamples_ = getInt(10, 10000, pString);
        printf("Latin hypercube  has sample size = %d\n", psOUU3X4nSamples_);  
      }
      else if (methodX4 == 2)
      {
        snprintf(pString,100,
                "Enter number of levels per variable (>=3, <=100) : ");
        psOUU3X4nSamples_ = getInt(3, 100, pString);
        kk = psOUU3X4nSamples_;
        for (ii = 1; ii < M4; ii++) psOUU3X4nSamples_ *= kk;
        printf("Factorial design has sample size = %d\n", psOUU3X4nSamples_);  
      }
      if (methodX4 == 1)
           sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
      else sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_FACT);
      sampler->setPrintLevel(0);
      lowers = new double[M4];
      uppers = new double[M4];
      index = 0;
      for (ii = 0; ii < nInputs; ii++)
      {
         if (psOUU3InputTypes_[ii] == 3)
         {
            lowers[index] = odata->lowerBounds_[ii];
            uppers[index] = odata->upperBounds_[ii];
            index++;
         }
      }
      sampler->setInputBounds(M4, lowers, uppers);
      sampler->setOutputParams(iOne);
      sampler->setSamplingParams(psOUU3X4nSamples_, iOne, iZero);
      sampler->initialize(0);
      psOUU3X4nSamples_ = sampler->getNumSamples();
      psOUU3X4SamInputs_ = new double[psOUU3X4nSamples_ * M4];
      samStates = new int[psOUU3X4nSamples_];
      samOuts = new double[psOUU3X4nSamples_];
      sampler->getSamples(psOUU3X4nSamples_, M4, iOne, psOUU3X4SamInputs_, 
                          samOuts, samStates);
      delete sampler;
      delete [] lowers;
      delete [] uppers;
      delete [] samStates;
      delete [] samOuts;
      printEquals(PL_INFO, 0);
   }

   //**/ ----------------------------------------------------------
   //**/ if there are discrete variables X3, read in the sample
   //**/ ----------------------------------------------------------
   int nSamp=psOUU3X3nSamples_*psOUU3X4nSamples_;
   psOUU3SamOutputs_  = new double[nSamp];
   printf("Use your own optimizer instead of BOBYQA ? (y or n) ");
   scanf("%s", lineIn);
   if (lineIn[0] == 'y')
   {
      psOUU3UserOpt_ = 1;
      printf("NOTE: Make sure your optimizer executable has been\n");
      printf("      assigned to 'opt_driver' and it optimizes with\n");
      printf("      respect to the %d-th to %d-th parameters.\n", 
             M1+1, M1+M2);
   }   
   if (M4 > 0)
   {
      printEquals(PL_INFO, 0);
      printf("Use response surface for X4 to compute statistics ? (y or n) ");
      scanf("%s", lineIn);
      if (lineIn[0] == 'y') psOUU3UseRS_ = 1;
   }
   printEquals(PL_INFO, 0);
   if (psOUU3UserOpt_ == 1)
   {
      printf("Use aux opt driver with ensemble run ? (y or n) ");
      scanf("%s", lineIn);
      if (lineIn[0] == 'y') psOUU3EnsembleEval_ = 1;
      printEquals(PL_INFO, 0);
   }
   if (psOUU3EnsembleEval_ == 0)
   {
      printf("Synchronous (s) or asynchronous (p) mode for runs ? (s or p) ");
      scanf("%s", lineIn);
      if (lineIn[0] == 'p') psOUU3Parallel_ = 1;
      printEquals(PL_INFO, 0);
   }
   fgets(lineIn, 500, stdin);

   //**/ ----------------------------------------------------------
   //**/ these variables are for temporary storage at various stages
   //**/ of optimization. psOUU3XValues is for storing all simulations
   //**/ ----------------------------------------------------------
   psOUU3WValues_ = new double[nInputs];
   psOUU3XValues_ = new double[nInputs*nSamp];
   psOUU3OptimalX_ = new double[nInputs];

   //**/ ----------------------------------------------------------
   //**/ set up response surface constructor and large sample
   //**/ ----------------------------------------------------------
   int        rstype=0, *inputPDFs=NULL;
   double     *inputMeans=NULL, *inputStdevs=NULL;
   PDFManager *pdfman=NULL;
   psVector   vecLB, vecUB, vecOut;
   pData      pdata;
   psMatrix   *corMat1, corMat2;
   psOUU3faPtr_ = NULL;
   if (psOUU3UseRS_ == 1 && M4 > 0)
   {
      if (printLevel > 2) 
         printf("OUU3Optimizer: setting up response surface\n");
      if (psOUU3X4nSamples_ > 400)
      {
         rstype = PSUADE_RS_MARS;
         printf("OUU3Optimizer: use MARS since nSamples > 300\n");
      }
      else
      {
         rstype = PSUADE_RS_KR;
         printf("OUU3Optimizer: use Kriging response surface\n");
      }
      //**/ create response surface
      psConfig_.InteractiveSaveAndReset();
      psOUU3faPtr_ = genFA(rstype, M4, -1, psOUU3X4nSamples_);
      lowers = new double[M4];
      uppers = new double[M4];
      index = 0;
      for (ii = 0; ii < nInputs; ii++)
      {
         if (psOUU3InputTypes_[ii] == 3)
         {
            lowers[index] = odata->lowerBounds_[ii];
            uppers[index] = odata->upperBounds_[ii];
            index++;
         }
      }
      psOUU3faPtr_->setBounds(lowers,uppers);
      psOUU3faPtr_->setOutputLevel(0);
      psConfig_.InteractiveRestore();

      //**/ generate a large sample
      odata->psIO_->getParameter("input_pdfs", pdata);
      inputPDFs = pdata.intArray_;
      pdata.intArray_ = NULL;
      if (inputPDFs == NULL)
      {
         inputPDFs = new int[nInputs];
         for (ii = 0; ii < nInputs; ii++) inputPDFs[ii] = 0;
      }
      kk = 0; index = 0;
      for (ii = 0; ii < nInputs; ii++)
      {
         if (psOUU3InputTypes_[ii] == 3)
         {
            kk += inputPDFs[ii];
            index++;
         }
      }
      if (kk == 0)
      {
         sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
         sampler->setPrintLevel(0);
         sampler->setInputBounds(M4, lowers, uppers);
         sampler->setOutputParams(iOne);
         psOUU3LargeSampleSize_ = 10000;
         sampler->setSamplingParams(psOUU3LargeSampleSize_, iOne, iZero);
         sampler->initialize(0);
         psOUU3LargeSampleSize_ = sampler->getNumSamples();
         psOUU3LargeSamInputs_= new double[psOUU3LargeSampleSize_*M4];
         psOUU3LargeSamOutputs_ = new double[psOUU3LargeSampleSize_];
         samStates = new int[psOUU3LargeSampleSize_];
         sampler->getSamples(psOUU3LargeSampleSize_, M4, iOne, 
                      psOUU3LargeSamInputs_,psOUU3LargeSamOutputs_,
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
            for (ii = 0; ii < nInputs; ii++) inputMeans[ii] = 0;
         }
         pdata.dbleArray_ = NULL;
         odata->psIO_->getParameter("input_stdevs", pdata);
         inputStdevs = pdata.dbleArray_;
         if (inputStdevs == NULL)
         {
            inputStdevs = new double[nInputs];
            for (ii = 0; ii < nInputs; ii++) inputStdevs[ii] = 1;
         }
         pdata.dbleArray_ = NULL;
         odata->psIO_->getParameter("input_cor_matrix", pdata);
         corMat1 = (psMatrix *) pdata.psObject_;
         pdata.psObject_ = NULL;

         //**/ compress pdf information
         corMat2.setDim(M4,M4);
         int    *iPdfs  = new int[nInputs];
         double *iMeans = new double[nInputs];
         double *iStdvs = new double[nInputs];
         int index2;
         index = 0;
         for (ii = 0; ii < nInputs; ii++)
         {
            if (psOUU3InputTypes_[ii] == 3)
            { 
               iMeans[index] = inputMeans[ii];
               iStdvs[index] = inputStdevs[ii];
               iPdfs[index]  = inputPDFs[ii];
               index2 = 0;
               for (kk = 0; kk < nInputs; kk++)
               {
                  if (psOUU3InputTypes_[kk] == 3)
                  {
                     ddata = corMat1->getEntry(ii,kk);
                     corMat2.setEntry(index,index2,ddata);
                     index2++;
                  }
               }
               index++;
            }
         }
         pdfman = new PDFManager();
         pdfman->initialize(M4,iPdfs,iMeans,iStdvs,corMat2,NULL,NULL);
         vecLB.load(M4, lowers);
         vecUB.load(M4, uppers);
         psOUU3LargeSampleSize_ = 10000;
         vecOut.setLength(psOUU3LargeSampleSize_*M4);
         pdfman->genSample(psOUU3LargeSampleSize_, vecOut, vecLB, vecUB);
         psOUU3LargeSamInputs_= new double[psOUU3LargeSampleSize_*M4];
         psOUU3LargeSamOutputs_ = new double[psOUU3LargeSampleSize_];
         for (ii = 0; ii < psOUU3LargeSampleSize_*M4; ii++)
            psOUU3LargeSamInputs_[ii] = vecOut[ii];
         delete [] inputPDFs;
         delete [] inputMeans;
         delete [] inputStdevs;
         delete [] iPdfs;
         delete [] iMeans;
         delete [] iStdvs;
         delete pdfman;
      }
      delete [] lowers;
      delete [] uppers;
   }

   //**/ ----------------------------------------------------------
   //**/ optimize 
   //**/ ----------------------------------------------------------
   nPts = (M1 + 1) * (M1 + 2) / 2;
   workArray = new double[(nPts+5)*(nPts+M1)+3*M1*(M1+5)/2+1];
   psOUU3Counter_ = 0;
   if (psOUU3Parallel_ == 1) odata->funcIO_->setAsynchronousMode();

#ifdef HAVE_BOBYQA
   //**/ ----------------------------------------------------------
   //**/ ----- retrieve history -----
   //**/ ----------------------------------------------------------
   fp = fopen("psuade_ouu3_history", "r");
   if (fp != NULL)
   {
      printf("OUU3Optimizer history file found. Use it? (y or n) ");
      scanf("%s", lineIn);
      if (lineIn[0] != 'y') fclose(fp);
      else
      {
         psOUU3NSaved_ = 0;
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
                  fscanf(fp, "%lg",&psOUU3SaveX_[psOUU3NSaved_*nInputs+ii]);
               fscanf(fp, "%lg",&psOUU3SaveY_[psOUU3NSaved_]);
               psOUU3NSaved_++;
            }
            if (((psOUU3NSaved_+1)*nInputs > psOUU3MaxSaved_*10) ||
                psOUU3NSaved_ > psOUU3MaxSaved_) break;
         }
         fclose(fp);
      }
   }

   //**/ ----------------------------------------------------------
   //**/ ----- run optimizer -----
   //**/ ----------------------------------------------------------
   for (ii = 0; ii < M1; ii++) 
      printf("OUU3Optimizer initial X %3d = %e\n", ii+1, XValues[ii]);
   bobyqa_(&M1, &nPts, XValues, odata->lowerBounds_,
           odata->upperBounds_, &rhobeg, &rhoend, &bobyqaFlag, &maxfun, 
           workArray);
   printf("OUU3Optimizer: total number of evaluations = %d\n",
           odata->numFuncEvals_);

   //**/ ----------------------------------------------------------
   //**/ ----- save history -----
   //**/ ----------------------------------------------------------
   if (psOUU3NSaved_ > 0)
   {
      fp = fopen("psuade_ouu3_history","w");
      if (fp != NULL)
      {
         for (ii = 0; ii < psOUU3NSaved_; ii++)
         {
            fprintf(fp, "999 %d ", nInputs);
            for (kk = 0; kk < nInputs; kk++)
               fprintf(fp, "%24.16e ", psOUU3SaveX_[ii*nInputs+kk]);
            fprintf(fp, "%24.16e\n", psOUU3SaveY_[ii]);
         }
         fclose(fp);
      }
      printf("OUU3Optimizer: history saved in psuade_ouu3_history\n");
   }
#else
   printf("ERROR : Bobyqa optimizer not installed.\n");
   exit(1);
#endif
   if (psOUU3Parallel_ == 1) odata->funcIO_->setSynchronousMode();

   //**/ ----------------------------------------------------------
   //**/ set return values and clean up 
   //**/ ----------------------------------------------------------
   if ((odata->setOptDriver_ & 2) && currDriver >= 0)
   {
      printf("OUU3Optimizer INFO: reverting to original simulation driver.\n");
      odata->funcIO_->setDriver(currDriver);
   }
   delete [] XValues;
   delete [] workArray;
   if (psOUU3X3SamInputs_ != NULL) delete [] psOUU3X3SamInputs_;
   if (psOUU3X4SamInputs_ != NULL) delete [] psOUU3X4SamInputs_;
   if (psOUU3SamOutputs_  != NULL) delete [] psOUU3SamOutputs_;
   if (psOUU3SamProbs_    != NULL) delete [] psOUU3SamProbs_;
   if (psOUU3LargeSamInputs_  != NULL) delete [] psOUU3LargeSamInputs_;
   if (psOUU3LargeSamOutputs_ != NULL) delete [] psOUU3LargeSamOutputs_;
   if (psOUU3XValues_  != NULL) delete [] psOUU3XValues_;
   if (psOUU3WValues_  != NULL) delete [] psOUU3WValues_;
   if (psOUU3OptimalX_ != NULL) delete [] psOUU3OptimalX_;
   if (psOUU3InputTypes_ != NULL) delete psOUU3InputTypes_;
   if (psOUU3faPtr_ != NULL) delete psOUU3faPtr_;
   psOUU3X3SamInputs_ = NULL;
   psOUU3X4SamInputs_ = NULL;
   psOUU3SamOutputs_ = NULL;
   psOUU3SamProbs_ = NULL;
   psOUU3LargeSamInputs_ = NULL;
   psOUU3LargeSamOutputs_ = NULL;
   psOUU3XValues_ = NULL;
   psOUU3WValues_ = NULL;
   psOUU3OptimalX_ = NULL;
   psOUU3InputTypes_ = NULL;
   psOUU3faPtr_ = NULL;
   odata->intData_ = M1;
}

// ************************************************************************
// assign operator
// ------------------------------------------------------------------------
OUU3Optimizer& OUU3Optimizer::operator=(const OUU3Optimizer &)
{
   printf("OUU3Optimizer operator= ERROR: operation not allowed.\n");
   exit(1);
   return (*this);
}

