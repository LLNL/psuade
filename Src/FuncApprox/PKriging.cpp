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
// Ref: S. Lophaven H. B. Nielsen, and J. Sondergaard, "DACE: A Matlab
//      Kriging toolbox," Technical Report IMM-TR-2002-12, Informatics
//      and Mathematical Modelling, Technical University of Denmark.
// ************************************************************************
// Functions for the class PKriging
//   This module is intended primarily for parallel generation of 
//      Kriging coefficients. Evaluations should be done by Kriging module.
// AUTHOR : CHARLES TONG
// DATE   : 2014
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "PKriging.h"
#include "sysdef.h"
#include "dtype.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "PsuadeConfig.h"
#include "Sampling.h"
#include "PrintingTS.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// ************************************************************************
// external functions
// ------------------------------------------------------------------------
extern "C" {
   void dtrsv_(char *, char *, char *, int *, double *, int *, double *, int *);
   void dpotrf_(char *, int *, double *, int *, int *);
   void dpotrs_(char *, int *, int *, double *, int *, double *,
                int *, int *);
   void kbobyqa_(int *,int *, double *, double *, double *, double *,
                double *, int *, int *, double*);
}

// ************************************************************************
// ************************************************************************
// internal global variables
// ------------------------------------------------------------------------
int    PKRI_outputLevel=0;
int    PKRI_iter=-1;
int    PKRI_nInputs=-1;
int    PKRI_nSamples=-1;
int    PKRI_pOrder=-1;
double *PKRI_XDists=NULL;
double *PKRI_SMatrix=NULL;
double *PKRI_FMatrix=NULL;
double *PKRI_FMatTmp=NULL;
double *PKRI_MMatrix=NULL;
double *PKRI_X=NULL;
double *PKRI_Y=NULL;
double PKRI_currY=0.0;
double *PKRI_Ytmp=NULL;
double PKRI_OptY=0.0;
double *PKRI_OptThetas=NULL;
double *PKRI_dataStdDevs=NULL;
double PKRI_YStd=1.0;
int    PKRI_terminate=0;
int    PKRI_noProgressCnt=0;

// ************************************************************************
// ************************************************************************
// resident function to perform evaluation 
// ------------------------------------------------------------------------
#ifdef __cplusplus
extern "C"
{
#endif
  void *pkribobyqaevalfunc_(int *nInps, double *XValues, double *YValue)
  {
    int    nBasis, count, Cleng, ii, jj, kk, status;
    double dist, ddata, ddata2;
    char   uplo='L';
    FILE   *fp=NULL;

    //**/ =======================================================
    //**/ termination signal has been detected
    //**/ =======================================================
    if (PKRI_noProgressCnt >= 200)
    {
      if (PKRI_outputLevel >= 1)
      {
        //PKRI_outputLevel = 0;
        printf("\t*** no progress for more than 200 iterations.\n");
      }
      (*YValue) = PSUADE_UNDEFINED;
      return NULL;
    }
    fp = fopen("psuade_stop", "r");
    if (fp != NULL)
    {
      fclose(fp);
      printf("Kriging: psuade_stop file found - terminating ....\n");
      (*YValue) = PKRI_OptY;
      return NULL;
    }
    //**/ =======================================================
    //**/ fill matrix
    //**/ =======================================================
    //**/ figure out the size of the matrix
    PKRI_iter++;
    nBasis = 1;
    if (PKRI_pOrder == 1) nBasis = PKRI_nInputs + 1;
    Cleng = PKRI_nSamples + nBasis;

      //**/ fill in the S matrix
    count = 0;
    for (jj = 0; jj < PKRI_nSamples; jj++)
    {
      PKRI_SMatrix[jj*PKRI_nSamples+jj] = 1.0 + 1e-15 * PKRI_nSamples;
      if (PKRI_nInputs == 2) PKRI_SMatrix[jj*PKRI_nSamples+jj] += 0.01;

      if (PKRI_dataStdDevs != NULL) 
      {
        ddata = pow(PKRI_dataStdDevs[jj]/PKRI_YStd,2.0);
        PKRI_SMatrix[jj*PKRI_nSamples+jj] += ddata;
      }
      for (kk = jj+1; kk < PKRI_nSamples; kk++)
      {
        dist = 0.0;
        for (ii = 0; ii < PKRI_nInputs; ii++)
          dist += pow(PKRI_XDists[count*PKRI_nInputs+ii]/XValues[ii],2.0);
        dist = exp(-dist);
        if (dist < 1.0e-16) dist = 0;
        PKRI_SMatrix[jj*PKRI_nSamples+kk] = dist;
        PKRI_SMatrix[kk*PKRI_nSamples+jj] = dist;
        count++;
      }
    }

    //**/ =======================================================
    //**/ Solve the linear system (note: F has been filled in main
    //**/   | S   F | |alpha | = Y
    //**/   | F^T 0 | |beta  | = 0
    //**/ =======================================================
    //**/ -------------------------------------------------------
    //**/ Create F
    //**/ -------------------------------------------------------
    for (jj = 0; jj < PKRI_nSamples; jj++)
    {
      PKRI_FMatrix[jj] = 1.0;
      PKRI_FMatTmp[jj] = 1.0;
      if (PKRI_pOrder == 1)
      {
        for (kk = 1; kk < PKRI_nInputs+1; kk++)
        {
          ddata = PKRI_X[jj*PKRI_nInputs+kk-1];
          PKRI_FMatTmp[kk*PKRI_nSamples+jj] = ddata;
          PKRI_FMatrix[kk*PKRI_nSamples+jj] = ddata;
        }
      }
    }
    //**/ -------------------------------------------------------
    //**/ S^{-1} F ==> FMatTmp
    //**/ -------------------------------------------------------
    dpotrf_(&uplo, &PKRI_nSamples, PKRI_SMatrix, &PKRI_nSamples, &status);
    kk = nBasis;
    dpotrs_(&uplo, &PKRI_nSamples, &kk, PKRI_SMatrix, &PKRI_nSamples,
            PKRI_FMatTmp, &PKRI_nSamples, &status);
    //**/ -------------------------------------------------------
    //**/ M = F^T S^{-1} F = F^T FMatTmp, create inv(M)
    //**/ -------------------------------------------------------
    for (jj = 0; jj < nBasis; jj++)
    {
      for (kk = 0; kk < nBasis; kk++)
      {
        ddata = 0.0;
        for (ii = 0; ii < PKRI_nSamples; ii++)
          ddata += PKRI_FMatrix[ii+jj*PKRI_nSamples]*
                   PKRI_FMatTmp[ii+kk*PKRI_nSamples];
        PKRI_MMatrix[jj+kk*nBasis] = ddata;
      }
    }
    if (nBasis > 1) dpotrf_(&uplo, &nBasis, PKRI_MMatrix, &nBasis, &status);
    //**/ -------------------------------------------------------
    //**/ Ytmp = S^{-1} Y (Y is the output vector)
    //**/ -------------------------------------------------------
    for (jj = 0; jj < PKRI_nSamples; jj++) PKRI_Ytmp[jj] = PKRI_Y[jj];
    kk = 1;
    dpotrs_(&uplo,&PKRI_nSamples,&kk,PKRI_SMatrix,&PKRI_nSamples,PKRI_Ytmp,
            &PKRI_nSamples, &status);
    //**/ -------------------------------------------------------
    //**/ Ytmp = F^T S^{-1} Y - X = F^T Ytmp (X = 0)
    //**/ -------------------------------------------------------
    for (jj = 0; jj < nBasis; jj++)
    {
       ddata = 0.0;
       for (ii = 0; ii < PKRI_nSamples; ii++)
         ddata += PKRI_FMatrix[ii+jj*PKRI_nSamples]*PKRI_Ytmp[ii];
       PKRI_Ytmp[PKRI_nSamples+jj] = ddata;
    }
    //**/ -------------------------------------------------------
    //**/ Ytmp = (F^T S^{-1} F)^{-1} (F^T S^{-1} Y - X) = M^{-1} Ytmp
    //**/ beta is now in Ytmp
    //**/ -------------------------------------------------------
    if (nBasis == 1)
    {
      if (PKRI_MMatrix[0] == 0)
      {
        printf("PKriging ERROR: divide by 0.\n");
        exit(1);
      }
      PKRI_Ytmp[PKRI_nSamples] /= PKRI_MMatrix[0];
    }
    else
    {
      kk = 1;
      dpotrs_(&uplo,&kk,&kk,PKRI_MMatrix,&nBasis,&PKRI_Ytmp[PKRI_nSamples],
              &kk, &status);
    }
    //**/ -------------------------------------------------------
    //**/ Ytmp = (Y - F beta)
    //**/ -------------------------------------------------------
    for (jj = 0; jj < PKRI_nSamples; jj++)
    {
      ddata = 0.0;
      for (ii = 0; ii < nBasis; ii++)
        ddata += PKRI_FMatrix[jj+ii*PKRI_nSamples]*
                 PKRI_Ytmp[ii+PKRI_nSamples];
      PKRI_Ytmp[jj] = PKRI_Y[jj] - ddata;
    }
    //**/ -------------------------------------------------------
    //**/ alpha = S^{-1} (Y - F beta) = S^{-1} Ytmp
    //**/ now everything is in Ytmp
    //**/ -------------------------------------------------------
    kk = 1;
    dpotrs_(&uplo,&PKRI_nSamples,&kk,PKRI_SMatrix,&PKRI_nSamples,PKRI_Ytmp,
            &PKRI_nSamples, &status);

    //**/ =======================================================
    //**/ compute metric
    //**/ =======================================================
    ddata = 0.0;
    for (jj = 0; jj < PKRI_nSamples; jj++)
      ddata += PKRI_Ytmp[jj] * PKRI_Y[jj];
    ddata /= (double) PKRI_nSamples;

    for (jj = 0; jj < PKRI_nSamples; jj++) 
    {
      ddata2 = PKRI_SMatrix[PKRI_nSamples*jj+jj];
      if (ddata2 < 0.0) ddata2 = - ddata2;
      ddata *= pow(ddata2, 2.0/(double) PKRI_nSamples);
    }

    (*YValue) = PKRI_currY = ddata;

    if (ddata < PKRI_OptY || PKRI_iter == 1)
    {
      PKRI_noProgressCnt = 0;
      PKRI_OptY = ddata;
      for (ii = 0; ii < PKRI_nInputs; ii++)
        PKRI_OptThetas[ii] = XValues[ii];
      if (PKRI_outputLevel > 1)
      {
        printf("\t Kriging : iteration %d\n",PKRI_iter);
        for (ii = 0; ii < PKRI_nInputs; ii++)
          printf("\t    Current best theta %d = %e\n",ii+1,XValues[ii]);
        printf("\t    Current best objective value = %e\n",ddata);
      }
    }
    else PKRI_noProgressCnt++;

    if (psConfig_.RSExpertModeIsOn() && PKRI_outputLevel > 3)
    {
      printf("\t PKriging : iteration %d\n", PKRI_iter);
      for (ii = 0; ii < PKRI_nInputs; ii++)
        printf("\t    Current theta %d = %e\n", ii+1, XValues[ii]);
      printf("\t    Current Kriging objective value = %e\n", ddata);
      printf("\t* For early termination, just create a file\n");
      printf("\t* called 'psuade_stop' in your local directory.\n");
      fp = fopen("psuade_stop", "r");
      if (fp != NULL)
      {
        printf("\t*** psuade_stop file found.\n");
        (*YValue) = 0.0;
        fclose(fp);
      }
    }
    return NULL;
  }
#ifdef __cplusplus
}
#endif

// ************************************************************************
// Constructor for object class Kriging
// ------------------------------------------------------------------------
PKriging::PKriging(int nInputs,int nSamples, CommManager *comm) : 
                   FuncApprox(nInputs,nSamples)
{
  int    ii, jj, pstatus, iOne=1;
  double ddata;
  char   pString[500], winput[500], winput2[500], fname[500], *strPtr;
  FILE   *fp;

  //**/ =======================================================
  // initialize variables and parameters
  //**/ =======================================================
  faID_ = PSUADE_RS_KR;
  commMgr_ = comm;
  if (comm == NULL)
  {
    printOutTS(PL_ERROR,"PKriging ERROR: no communicator.\n");
    exit(1);
  }
  mypid_  = psCommMgr_->getPID();
  nprocs_ = psCommMgr_->getNumProcs();

  pOrder_  = 0;
  optTolerance_ = 1.0e-4;
  VecThetas_.setLength(nInputs_+1);
  for (ii = 0; ii <= nInputs_; ii++) VecThetas_[ii] = 0.01;

  //**/ =======================================================
  // display banner and additonal information
  //**/ =======================================================
  if (mypid_ == 0)
  {
    printAsterisks(PL_INFO, 0);
    printf("*                PKriging Analysis\n");
    printf("* Set printlevel to 1-4 to see Kriging details.\n");
    printf("* Create 'psuade_stop' file to gracefully terminate.\n");
    printEquals(PL_INFO, 0);
  }

  //**/ =======================================================
  // read configure file, if any 
  //**/ =======================================================
  if (mypid_ == 0)
  {
    strPtr = psConfig_.getParameter("RS_KRI_DATA_STDEV_FILE");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %s", winput, winput2, fname);
      fp = fopen(fname, "r");
      if (fp == NULL)
      {
        printf("Kriging INFO: data variance file %s not found.\n",fname);
      }
      else
      {
        fscanf(fp, "%d", &ii); 
        if (ii != nSamples_)
        {
          printf("Kriging ERROR: std. dev. file should have %d entries.\n",
                 nSamples_);
          fclose(fp);
        }
        else
        {
          VecDataStdDevs_.setLength(nSamples_);
          for (ii = 0; ii < nSamples_; ii++)
          {
            fscanf(fp, "%d %lg", &jj, &ddata);
            VecDataStdDevs_[ii] = ddata; 
            if (jj != ii+1)
            {
              printf("Kriging ERROR: line %d in the std dev file.\n",
                     jj+1);
              VecDataStdDevs_.clean();
              break;
            }
          } 
          fclose(fp);
          if (VecDataStdDevs_.length() > 0)
          {
            printf("Kriging INFO: std. dev. file has been read.\n");
            PKRI_dataStdDevs = VecDataStdDevs_.getDVector(); 
          }
        }
      }
    }
  }
  pstatus = nSamples_;
  if (mypid_ == 0 && VecDataStdDevs_.length() == 0) pstatus = 0; 
  psCommMgr_->bcast((void *) &pstatus, iOne, INT, 0);
  if (mypid_ != 0 && pstatus != 0) VecDataStdDevs_.setLength(nSamples_);
  psCommMgr_->bcast((void *) VecDataStdDevs_.getDVector(),pstatus,DOUBLE,0);
  PKRI_dataStdDevs = VecDataStdDevs_.getDVector(); 
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
PKriging::~PKriging()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int PKriging::initialize(double *X, double *Y)
{
  //**/ ---------------------------------------------------------------
  //**/ generate the Gaussian hyperparameters
  //**/ ---------------------------------------------------------------
  train(X,Y);
  return 0;
}
  
// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int PKriging::genNDGridData(double *, double *,int *, double **, double **)
{
  printf("PKriging genNDGridData INFO: use Kriging\n");
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int PKriging::gen1DGridData(double *, double *, int, double *, int *, 
                            double **, double **)
{
  printf("PKriging gen1DGridData INFO: use Kriging\n");
  return 0;
}

// ************************************************************************
// Generate 2D results for display
// ------------------------------------------------------------------------
int PKriging::gen2DGridData(double *, double *, int, int, double *, int *, 
                            double **, double **)
{
  printf("PKriging gen2DGridData INFO: use Kriging\n");
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int PKriging::gen3DGridData(double *, double *, int, int, int, double *, 
                            int *, double **, double **)
{
  printf("PKriging gen3DGridData INFO: use Kriging\n");
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int PKriging::gen4DGridData(double *, double *, int, int, int, int, double *,
                            int *, double **, double **)
{
  printf("PKriging gen4DGridData INFO: use Kriging\n");
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double PKriging::evaluatePoint(double *)
{
  printf("PKriging evaluatePoint INFO: use Kriging\n");
  return 0.0;
}

// ************************************************************************
// Evaluate a number of points 
// ------------------------------------------------------------------------
double PKriging::evaluatePoint(int, double *, double *)
{
  printf("PKriging evaluatePoint INFO: use Kriging\n");
  return 0.0;
}

// ************************************************************************
// Evaluate a given point, return also the standard deviation 
// ------------------------------------------------------------------------
double PKriging::evaluatePointFuzzy(double *, double &)
{
  printf("PKriging evaluatePointFuzzy INFO: use Kriging\n");
  return 0.0;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double PKriging::evaluatePointFuzzy(int, double *, double *, double *)
{
  printf("PKriging evaluatePointFuzzy INFO: use Kriging\n");
  return 0.0;
}

// ************************************************************************
// training 
// ------------------------------------------------------------------------
double PKriging::train(double *X, double *Y)
{
  int    ii, kk;
  double ddata; 
  char   pString[500];

  //**/ ============================================================
  // normalize the input and outputs 
  //**/ ============================================================
  VecNormalX_.setLength(nSamples_*nInputs_);
  initInputScaling(X, VecNormalX_.getDVector(), 1);
  VecNormalY_.setLength(nSamples_);
  initOutputScaling(Y, VecNormalY_.getDVector());
  PKRI_YStd = YStd_;

  //**/ ============================================================
  // compute distances between all pairs of inputs (vecXDists)
  //**/ needed for creating the correlation matrix
  //**/ ============================================================
  psVector vecXDists;
  computeDistances(vecXDists);
  double *XDists = vecXDists.getDVector();

  //**/ ============================================================
  // slow mode: optimize
  //**/ ============================================================
  int    nBasis, maxfun, pLevel, nPts, iOne=1, iZero=0, nSamOpt;
  double rhobeg=1.0, rhoend=1.0e-4, optY=PSUADE_UNDEFINED;
  FILE   *fp=NULL;
  Sampling *sampler;

  //**/ ----------------------------------------------------
  //**/ if slow mode, need to make sure the psuade_stop
  //**/ file has been cleaned up
  //**/ ----------------------------------------------------
  int status = 0;
  fp = fopen("psuade_stop", "r");
  if (mypid_ == 0 && fp != NULL)
  {
    printf("PKriging ERROR: remove the 'psuade_stop' file\n");
    printf("                first and re-do.\n");
    fclose(fp);
    status = 1;
  }
  psCommMgr_->bcast((void *) &status, iOne, INT, 0);
  if (status == 1) exit(1);

  //**/ ----------------------------------------------------
  //**/ get number of basis 
  //**/ ----------------------------------------------------
  if (nSamples_ <= nInputs_ + 1) pOrder_ = 0;
  if (pOrder_ == 1) nBasis = nInputs_ + 1;
  else              nBasis = 1;
  int Cleng = nSamples_ + nBasis;

  //**/ ----------------------------------------------------
  //**/ prepare for optimization 
  //**/ ----------------------------------------------------
  psVector vecTUppers, vecTLowers;
  vecTUppers.setLength(nInputs_+1);
  vecTLowers.setLength(nInputs_+1);
  for (ii = 0; ii < nInputs_; ii++)
  {
    if (VecXMeans_[ii] == 0 && VecXStds_[ii] == 1.0)
    {
      vecTUppers[ii] = 20.0 * (VecUBs_[ii] - VecLBs_[ii]);
      vecTLowers[ii] =  0.1 * (VecUBs_[ii] - VecLBs_[ii]);
    }
    else
    {
      vecTUppers[ii] = 20.0;
      vecTLowers[ii] = 0.1;
    }
  }
  status = 0;
  if (mypid_ == 0 && psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("PKriging: current optimization lower bound for input %d = %e",
            ii+1,vecTLowers[ii]);
    snprintf(pString,100,
             "PKriging: Enter optimization lower bound for input %d : ",ii+1);
    vecTLowers[ii] = getDouble(pString);
    if (vecTLowers[ii] <= 0.0)
    {
      printf("PKriging ERROR: lower bound <= 0\n");
      exit(1);
    }
    printf("PKriging: current optimization upper bound for input %d = %e",
           ii+1,vecTUppers[ii]);
    snprintf(pString,100,
            "PKriging: Enter optimization upper bound for input %d : ",ii+1);
    vecTUppers[ii] = getDouble(pString);
    if (vecTLowers[ii] > vecTUppers[ii])
    {
      printf("PKriging ERROR: lower bound >= upper bound\n");
      status = 1;
    }
  }
  psCommMgr_->bcast((void *) &status, iOne, INT, 0);
  if (status == 1) exit(1);
  kk = nInputs_ + 1;
  psCommMgr_->bcast((void *) vecTUppers.getDVector(), kk, DOUBLE, 0);
  psCommMgr_->bcast((void *) vecTLowers.getDVector(), kk, DOUBLE, 0);

  rhobeg = vecTUppers[0] - vecTLowers[0];
  for (ii = 1; ii < nInputs_; ii++)
  {
    ddata = vecTUppers[ii] - vecTLowers[ii];
    if (ddata < rhobeg) rhobeg = ddata;
  }
  rhobeg *= 0.5;
  rhoend = rhobeg * optTolerance_;
  psVector vecTVals;
  vecTVals.setLength(nInputs_+1);
  double *TValues = vecTVals.getDVector();
  nPts = (nInputs_ + 1) * (nInputs_ + 2) / 2;
  psVector vecWork;
  vecWork.setLength((nPts+5)*(nPts+nInputs_)+3*nInputs_*(nInputs_+5)/2+1);
  double *work = vecWork.getDVector();
  if (mypid_ == 0)
  {
    printEquals(PL_INFO, 0);
    printf("* PKriging optimization tolerance = %e\n", rhoend);
  }

  //**/ ----------------------------------------------------
  //**/ generate a sample
  //**/ ----------------------------------------------------
  if (mypid_ == 0)
    printf("PKriging training begins.... (order = %d)\n",pOrder_);
  nSamOpt = 10;
  if (nprocs_ > nSamOpt) nSamOpt = nprocs_;
  if (psConfig_.RSExpertModeIsOn() && mypid_ == 0)
  {
    printf("To thoroughly explore the parameter space, multi-start\n");
    printf("optimization is to be employed. Please enter the number\n");
    printf("of multi-starts (more the better, but also more expensive.\n");
    printf("Default is max(num. of processors, 10). \n");
    snprintf(pString,100,"Enter number of multi-starts (up to 100): ");
    nSamOpt = getInt(nSamOpt, 100, pString);
  }
  if (nInputs_ > 50)
       sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
  else sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LPTAU);
 
  sampler->setPrintLevel(0);
  sampler->setInputBounds(nInputs_, vecTLowers.getDVector(), 
                          vecTUppers.getDVector());
  sampler->setOutputParams(iOne);
  sampler->setSamplingParams(nSamOpt, iOne, iZero);
  sampler->initialize(0);
  nSamOpt = sampler->getNumSamples();

  psVector vecSamInps, vecSamOuts;
  vecSamInps.setLength(nSamOpt*nInputs_);
  vecSamOuts.setLength(nSamOpt);
  double *samInputs  = vecSamInps.getDVector();
  double *samOutputs = vecSamOuts.getDVector();
  
  psIVector vecSamStas;
  vecSamStas.setLength(nSamOpt);
  int *samStates = vecSamStas.getIVector();

  sampler->getSamples(nSamOpt, nInputs_, iOne, samInputs,
                      samOutputs, samStates);
  delete sampler;

  //**/ ----------------------------------------------------
  //**/ pass data to global variables for external calls
  //**/ ----------------------------------------------------
  PKRI_XDists = XDists;
  PKRI_nSamples = nSamples_;
  PKRI_nInputs = nInputs_;
  PKRI_pOrder = pOrder_;
  PKRI_X = VecNormalX_.getDVector();
  PKRI_Y = VecNormalY_.getDVector();
  psVector vecYtmp;
  vecYtmp.setLength(Cleng);
  PKRI_Ytmp = vecYtmp.getDVector();
  PKRI_OptThetas = VecThetas_.getDVector();
  maxfun = 5000;
  PKRI_OptY = PSUADE_UNDEFINED;
  PKRI_outputLevel = outputLevel_;
  
  psVector vecOptThetas;
  vecOptThetas.setLength(nInputs_);
  double *optThetas = vecOptThetas.getDVector();

  psVector vecSmat, vecFmat, vecFtmp, vecMmat;
  vecSmat.setLength(nSamples_*nSamples_);
  vecFmat.setLength(nSamples_*nBasis);
  vecFtmp.setLength(nSamples_*nBasis);
  vecMmat.setLength(nBasis*nBasis);
  PKRI_SMatrix = vecSmat.getDVector();
  PKRI_FMatrix = vecFmat.getDVector();
  PKRI_FMatTmp = vecFtmp.getDVector();
  PKRI_MMatrix = vecMmat.getDVector();

  int    proc, csize;
  double dmean, dstd;
  psVector vecCommBuf;
  vecCommBuf.setLength(nInputs_+1);
  double *commBuffer = vecCommBuf.getDVector();

  //**/ ----------------------------------------------------
  //**/ call optimizer 
  //**/ ----------------------------------------------------
  //**/ use pLevel=4444 to tell bobyqa_ to use this function
  //**/ for now set the initial guess to mid point
  for (kk = 0; kk < nSamOpt; kk++)
  {
    if (kk % nprocs_ == mypid_)
    {
      if (outputLevel_ >= 1) 
        printf("%4d: PKriging multi-start optimization: start = %d (%d)\n",
               mypid_, kk+1, nSamOpt);
      PKRI_iter = 0;
      pLevel = 4444;
      for (ii = 0; ii < nInputs_; ii++) 
        TValues[ii] = samInputs[kk*nInputs_+ii];
#ifdef HAVE_BOBYQA
      PKRI_noProgressCnt = 0;
      kbobyqa_(&nInputs_,&nPts,TValues,vecTLowers.getDVector(),
               vecTUppers.getDVector(),&rhobeg,&rhoend,
               &pLevel, &maxfun, work);
#else
      printf("PKriging ERROR: Bobyqa optimizer not installed.\n");
      exit(1);
#endif
      if (outputLevel_ >= 3) 
      {
        printf("%4d: multi-start optimization: iteration = %d (%d)\n",
               mypid_, kk+1, nSamOpt);
        for (ii = 0; ii < nInputs_; ii++) 
          printf("%4d: Input %4d final length scale = %e\n",
                 mypid_, ii+1,TValues[ii]);
        printf("%4d: final objective value = %e\n", mypid_, PKRI_currY);
      }
      if (PKRI_OptY < optY)
      {
        optY = PKRI_OptY;
        for (ii = 0; ii < nInputs_; ii++)
          optThetas[ii] = PKRI_OptThetas[ii];
      }
      fp = fopen("psuade_stop", "r");
      if (fp != NULL)
      {
        fclose(fp);
        fp = NULL;
        printf("PKriging: psuade_stop file found - terminating ....\n");
        break;
      }
    }
  }
  for (kk = 0; kk < nSamOpt; kk++)
  {
    if (mypid_ == 0 && (kk % nprocs_ != mypid_))
    {
      proc = kk % nprocs_;
      csize = nInputs_ + 1; 
      psCommMgr_->recv((void *) commBuffer,csize,DOUBLE,kk,proc);
      if (commBuffer[nInputs_] < optY)
      {
        for (ii = 0; ii < nInputs_; ii++) optThetas[ii] = commBuffer[ii];
        optY = commBuffer[nInputs_];
      }
    }
    else if (mypid_ != 0 && kk % nprocs_ == mypid_)
    {
      for (ii = 0; ii < nInputs_; ii++) commBuffer[ii] = optThetas[ii];
      commBuffer[nInputs_] = optY;
      csize = nInputs_ + 1; 
      psCommMgr_->send((void *) optThetas,csize,DOUBLE,kk,0);
    }
  }
  if (mypid_ == 0)
  {
    for (ii = 0; ii < nInputs_; ii++) commBuffer[ii] = optThetas[ii];
    commBuffer[nInputs_] = optY;
  }
  csize = nInputs_ + 1;
  psCommMgr_->bcast((void *) commBuffer, csize, DOUBLE, 0);
  for (ii = 0; ii < nInputs_; ii++) optThetas[ii] = commBuffer[ii]; 
  optY = commBuffer[nInputs_];

  //**/ ----------------------------------------------------
  //**/ display results and clean up
  //**/ ----------------------------------------------------
  for (ii = 0; ii < nInputs_; ii++) VecThetas_[ii] = optThetas[ii];
  if (mypid_ == 0) 
  {
    printAsterisks(PL_INFO,0);
    for (ii = 0; ii < nInputs_; ii++) 
      printf("PKriging: Input %4d optimal length scale = %e\n",
             ii+1,VecThetas_[ii]);
    printf("PKriging: optimal objective value = %e\n",PKRI_OptY);
    printAsterisks(PL_INFO,0);
  }
  if (mypid_ == 0) 
  {
    fp = fopen("psuade_kriging_optdata","w");
    for (ii = 0; ii < nInputs_; ii++) 
      fprintf(fp, "%16.8e\n", VecThetas_[ii]);
    printf("PKriging: length scales are saved in 'psuade_kriging_optdata'.\n");
    fclose(fp);
  }
  PKRI_OptThetas = NULL;
  PKRI_XDists = NULL;
  PKRI_X = NULL;
  PKRI_Y = NULL;
  PKRI_Ytmp = NULL;
  PKRI_SMatrix = NULL;
  PKRI_FMatrix = NULL;
  PKRI_FMatTmp = NULL;
  PKRI_MMatrix = NULL;
  return 0.0;
}

// ************************************************************************
// compute distances between all pairs of inputs
// ------------------------------------------------------------------------
int PKriging::computeDistances(psVector &vecXDists)
{
  int    ii, jj, kk, count;
  double dist;

  vecXDists.setLength((nSamples_*(nSamples_-1)/2)*nInputs_);
  count = 0;
  for (jj = 0; jj < nSamples_; jj++)
  {
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs_; ii++)
      {
        vecXDists[count*nInputs_+ii] = VecNormalX_[jj*nInputs_+ii] - 
                              VecNormalX_[kk*nInputs_+ii];
        if (vecXDists[count*nInputs_+ii] < 0) 
          vecXDists[count*nInputs_+ii] = - vecXDists[count*nInputs_+ii]; 
        dist += pow(vecXDists[count*nInputs_+ii], 2.0);
      }
      if (dist == 0.0)
      {
        printf("PKriging ERROR: repeated sample points.\n");
        printf("               Prune repeated points and re-run.\n");
        printf("Sample %d : (with sample point %d)\n", kk+1, jj+1);
        for (ii = 0; ii < nInputs_; ii++)
          printf("   Input %d : %e\n",ii+1,
            VecNormalX_[kk*nInputs_+ii]*VecXStds_[ii] +VecXMeans_[ii]);
        exit(1);
      }
      count++;
    }
  }
  return 0;
}

