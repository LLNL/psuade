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
// Functions for the class HKriging
// AUTHOR : CHARLES TONG
// DATE   : 2019
// ************************************************************************
#ifdef WINDOWS
#include <windows.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "HKriging.h"
#include "sysdef.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
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
  void newuoa_(int *,int *,double *,double *,double *,int *,
               int *,double*);
}

// ************************************************************************
// ************************************************************************
// internal global variables
// ------------------------------------------------------------------------
int    HKRI_outputLevel=0;
int    HKRI_iter=-1;
int    HKRI_nInputs=-1;
int    HKRI_nSamples=-1;
double *HKRI_XDists=NULL;
double *HKRI_SMatrix=NULL;
double *HKRI_FMatrix=NULL;
double *HKRI_FMatTmp=NULL;
double *HKRI_MMatrix=NULL;
double *HKRI_X=NULL;
double *HKRI_Y=NULL;
double HKRI_currY=0.0;
double *HKRI_Ytmp=NULL;
double HKRI_OptY=0.0;
double HKRI_OptTheta=0;
double HKRI_Exponent=2.0;
double HKRI_YStd=1.0;
double HKRI_nugget=0.0;
int    HKRI_terminate=0;
int    HKRI_noProgressCnt=0;

// ************************************************************************
// function to perform evaluation 
// ************************************************************************
#ifdef __cplusplus
extern "C"
{
#endif
  void *hkrinewuoaevalfunc_(int *nInps, double *XValues, double *YValue)
  {
    int    nBasis, count, Cleng, ii, jj, kk, status, iOne=1;
    double dist, ddata, ddata2;
    char   uplo='L';
    FILE   *fp=NULL;

    //**/ =======================================================
    //**/ termination signal has been detected
    //**/ =======================================================
    if (XValues[0] < 0) 
    {
      (*YValue) = PSUADE_UNDEFINED;
      return NULL;
    }
    if (HKRI_noProgressCnt >= 200)
    {
      if (HKRI_outputLevel >= 1)
      {
        //HKRI_outputLevel = 0;
        printf("\t*** no progress for more than 200 iterations.\n");
      }
      (*YValue) = PSUADE_UNDEFINED;
      return NULL;
    }
    //**/ =======================================================
    //**/ fill matrix
    //**/ =======================================================
    //**/ figure out the size of the matrix
    HKRI_iter++;
    nBasis = 1;
    Cleng = HKRI_nSamples + nBasis;

    //**/ fill in the S matrix
    count = 0;
    for (jj = 0; jj < HKRI_nSamples; jj++)
    {
      HKRI_SMatrix[jj*HKRI_nSamples+jj] = 1.0 + 1e-15 * HKRI_nSamples;
      if (HKRI_nugget != 0.0) 
        HKRI_SMatrix[jj*HKRI_nSamples+jj] += HKRI_nugget;

      for (kk = jj+1; kk < HKRI_nSamples; kk++)
      {
        dist = 0.0;
        for (ii = 0; ii < HKRI_nInputs; ii++)
          dist += pow(HKRI_XDists[count*HKRI_nInputs+ii]/XValues[0],
                      HKRI_Exponent);
        dist = exp(-dist);
        if (dist < 1.0e-16) dist = 0;
        HKRI_SMatrix[jj*HKRI_nSamples+kk] = dist;
        HKRI_SMatrix[kk*HKRI_nSamples+jj] = dist;
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
    for (jj = 0; jj < HKRI_nSamples; jj++)
    {
      HKRI_FMatrix[jj] = 1.0;
      HKRI_FMatTmp[jj] = 1.0;
    }
    //**/ -------------------------------------------------------
    //**/ S^{-1} F ==> FMatTmp
    //**/ -------------------------------------------------------
    dpotrf_(&uplo,&HKRI_nSamples,HKRI_SMatrix,&HKRI_nSamples,&status);
    kk = nBasis;
    dpotrs_(&uplo, &HKRI_nSamples, &kk, HKRI_SMatrix, &HKRI_nSamples,
            HKRI_FMatTmp, &HKRI_nSamples, &status);
    //**/ -------------------------------------------------------
    //**/ M = F^T S^{-1} F = F^T FMatTmp, create inv(M)
    //**/ -------------------------------------------------------
    for (jj = 0; jj < nBasis; jj++)
    {
      for (kk = 0; kk < nBasis; kk++)
      {
        ddata = 0.0;
        for (ii = 0; ii < HKRI_nSamples; ii++)
          ddata += HKRI_FMatrix[ii+jj*HKRI_nSamples]*
                   HKRI_FMatTmp[ii+kk*HKRI_nSamples];
        HKRI_MMatrix[jj+kk*nBasis] = ddata;
      }
    }
    if (nBasis > 1) 
      dpotrf_(&uplo,&nBasis,HKRI_MMatrix,&nBasis,&status);
    //**/ -------------------------------------------------------
    //**/ Ytmp = S^{-1} Y (Y is the output vector)
    //**/ -------------------------------------------------------
    for (jj = 0; jj < HKRI_nSamples; jj++) HKRI_Ytmp[jj] = HKRI_Y[jj];
    kk = 1;
    dpotrs_(&uplo,&HKRI_nSamples,&kk,HKRI_SMatrix,&HKRI_nSamples,
            HKRI_Ytmp, &HKRI_nSamples, &status);
    //**/ -------------------------------------------------------
    //**/ Ytmp = F^T S^{-1} Y - X = F^T Ytmp (X = 0)
    //**/ -------------------------------------------------------
    for (jj = 0; jj < nBasis; jj++)
    {
      ddata = 0.0;
      for (ii = 0; ii < HKRI_nSamples; ii++)
        ddata += HKRI_FMatrix[ii+jj*HKRI_nSamples]*HKRI_Ytmp[ii];
      HKRI_Ytmp[HKRI_nSamples+jj] = ddata;
    }
    //**/ -------------------------------------------------------
    //**/ Ytmp = (F^T S^{-1} F)^{-1} (F^T S^{-1} Y - X) = M^{-1} Ytmp
    //**/ beta is now in Ytmp
    //**/ -------------------------------------------------------
    if (nBasis == 1)
    {
      if (HKRI_MMatrix[0] == 0)
      {
        printf("HKriging ERROR: divide by 0.\n");
        exit(1);
      }
      HKRI_Ytmp[HKRI_nSamples] /= HKRI_MMatrix[0];
    }
    else
    {
      kk = 1;
      dpotrs_(&uplo,&kk,&kk,HKRI_MMatrix,&nBasis,
              &HKRI_Ytmp[HKRI_nSamples], &kk, &status);
    }
    //**/ -------------------------------------------------------
    //**/ Ytmp = (Y - F beta)
    //**/ -------------------------------------------------------
    for (jj = 0; jj < HKRI_nSamples; jj++)
    {
      ddata = 0.0;
      for (ii = 0; ii < nBasis; ii++)
        ddata += HKRI_FMatrix[jj+ii*HKRI_nSamples]*
                 HKRI_Ytmp[ii+HKRI_nSamples];
      HKRI_Ytmp[jj] = HKRI_Y[jj] - ddata;
    }
    //**/ -------------------------------------------------------
    //**/ alpha = S^{-1} (Y - F beta) = S^{-1} Ytmp
    //**/ now everything is in Ytmp
    //**/ -------------------------------------------------------
    kk = 1;
    dpotrs_(&uplo,&HKRI_nSamples,&kk,HKRI_SMatrix,&HKRI_nSamples,
            HKRI_Ytmp,&HKRI_nSamples, &status);

    //**/ =======================================================
    //**/ compute metric
    //**/ =======================================================
    ddata = 0.0;
    for (jj = 0; jj < HKRI_nSamples; jj++)
      ddata += HKRI_Ytmp[jj] * HKRI_Y[jj];
    ddata /= (double) HKRI_nSamples;

    for (jj = 0; jj < HKRI_nSamples; jj++) 
    {
      ddata2 = HKRI_SMatrix[HKRI_nSamples*jj+jj];
      if (ddata2 < 0.0) ddata2 = - ddata2;
      ddata *= pow(ddata2, 2.0/(double) HKRI_nSamples);
    }

    (*YValue) = HKRI_currY = ddata;

    if (ddata < HKRI_OptY || HKRI_iter == 1)
    {
      HKRI_noProgressCnt = 0;
      HKRI_OptY = ddata;
      HKRI_OptTheta = XValues[0];
      if (HKRI_outputLevel > 1)
      {
        printf("\t HKriging : iteration %d\n",HKRI_iter);
        printf("\t    Current best theta = %e\n",XValues[0]);
        printf("\t    Current best objective value = %e\n",ddata);
      }
    }
    else HKRI_noProgressCnt++;

    return NULL;
  }
#ifdef __cplusplus
}
#endif
// ************************************************************************
// Constructor for object class HKriging
// ------------------------------------------------------------------------
HKriging::HKriging(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  int  ii;
  char pString[101], winput[500], equal[500], *strPtr;

  //**/ =======================================================
  // initialize variables and parameters
  //**/ =======================================================
  faID_ = PSUADE_RS_HKR;
  pOrder_  = 0;
  initFlag_ = 0;
  optTolerance_ = 1.0e-4;
  fastMode_ = 3;
  Theta_ = 0.01;

  //**/ =======================================================
  // display banner and additonal information
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printf("*                HKriging Analysis\n");
    printf("* Set printlevel to 1-4 to see Kriging details.\n");
    printf("* Turn on rs_expert mode to set slow or fast mode.\n");
    printf("*  + Fast mode: no optimization of hyperparameters.\n");
    printf("*      - turn on rs_expert to set hyperparameters.\n");
    printf("*      - default values = 1.0\n");
    printf("*  + Slow mode : hyperparameters are optimized.\n");
    printf("*      - to change optimization parameters, turn\n");
    printf("*        rs_expert mode.\n");
    printf("*  + Snail mode (DEFAULT): use multi-start optimization.\n");
    printf("*      - to change optimization parameters, turn\n");
    printf("*        rs_expert mode.\n");
    printf("* Create 'psuade_stop' file to gracefully terminate.\n");
    printEquals(PL_INFO, 0);
  }

  //**/ =======================================================
  // if configure file is not used, ask for parameters if
  // response surface expert mode is on 
  //**/ =======================================================
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("There are two modes available: \n");
    printf("(1) slow mode with optimization on theta\n");
    printf("(2) very slow mode with multi-start optimization\n");
    snprintf(pString,100,"Please select mode (1 - 2) : ");
    fastMode_ = getInt(1,2,pString);
    snprintf(pString,100,"RS_HKRI_mode = %d",fastMode_);
    psConfig_.putParameter(pString);

    snprintf(pString,100,"Enter optimization tolerance (default = 1e-4): ");
    optTolerance_ = getDouble(pString);
    if (optTolerance_ <= 0 || optTolerance_ > 0.5)
    {
      printf("HKriging INFO: optimization tol should be in (0,0.5]).\n");
      printf("               Tolerance set to default = 1.0e-4.\n");
      optTolerance_ = 1.0e-4;
    }
    snprintf(pString,100,"RS_HKRI_tol = %e",optTolerance_);
    psConfig_.putParameter(pString);

    printf("HKriging: Current initial length scales = %e\n",Theta_);
    snprintf(pString,100,"Change initial theta? (y or n) ");
    getString(pString, winput);
    if (winput[0] == 'y')
    {
      snprintf(pString,100,"Enter new theta : ");
      Theta_ = 0;
      while (Theta_ <= 0)
      {
        Theta_ = getDouble(pString);
        if (Theta_ <= 0.0) printf("ERROR: theta has to be > 0.\n");
      }
      snprintf(pString,100,"RS_HKRI_theta = %e",Theta_);
      psConfig_.putParameter(pString);
    }

    if (psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
    {
      snprintf(pString,100,"Add nugget? (y or n) ");
      getString(pString, winput);
      if (winput[0] == 'y') 
      {
        HKRI_nugget = 1.0;
        while (HKRI_nugget >= 1.0 || HKRI_nugget < 0.0)
        {
          snprintf(pString,100,"Enter nugget ([0,1)) : ");
          HKRI_nugget = getDouble(pString);
        }
        snprintf(pString,100,"RS_HKRI_nugget = %e",HKRI_nugget);
        psConfig_.putParameter(pString);
      }
    }
  }
  else
  {
    //**/ get mode
    strPtr = psConfig_.getParameter("RS_HKRI_mode");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %d", winput, equal, &fastMode_);
      if (fastMode_ < 1 || fastMode_ > 2)
      {
        printf("HKriging INFO: mode from config not valid.\n");
        printf("               mode set to %d.\n", fastMode_);
      }
      printf("HKriging INFO: mode from config = %d.\n",fastMode_);
    }

    //**/ get tolerance
    strPtr = psConfig_.getParameter("RS_HKRI_tol");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %lg", winput, equal, &optTolerance_);
      if (optTolerance_ <= 0 || optTolerance_ > 0.5)
      {
        printf("HKriging INFO: optimization tol should be in (0,0.5]).\n");
        printf("               Tolerance set to default = 1.0e-4.\n");
        optTolerance_ = 1.0e-4;
      }
      printf("HKriging INFO: optimization tolerance from config = %e.\n",
             optTolerance_);
    }

    //**/ get length scale
    strPtr = psConfig_.getParameter("RS_HKRI_theta");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %lg", winput, equal, &Theta_);
      printf("HKriging INFO: length scale from config = %e.\n",Theta_);
    }

    //**/ get nugget 
    strPtr = psConfig_.getParameter("RS_HKRI_nugget");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %lg", winput, equal, &HKRI_nugget);
      printf("HKriging INFO: nugget from config = %e.\n",HKRI_nugget);
    }
  }
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
HKriging::~HKriging()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int HKriging::initialize(double *X, double *Y)
{
  //**/ ---------------------------------------------------------------
  //**/ generate the hyperparameters
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("HKriging training begins....\n");
  train(X,Y);
  if (outputLevel_ >= 1) printf("HKriging training completed.\n");
  return 0;
}
  
// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int HKriging::genNDGridData(double *XIn, double *YIn, int *NOut,
                            double **XOut, double **YOut)
{
  int totPts;

  //**/ ---------------------------------------------------------------
  //**/ generate the hyperparameters
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("HKriging training begins....\n");
  train(XIn,YIn);
  if (outputLevel_ >= 1) printf("HKriging training completed.\n");
  if ((*NOut) == -999) return 0;
 
  //**/ ---------------------------------------------------------------
  //**/ generating regular grid data
  //**/ ---------------------------------------------------------------
  genNDGrid(NOut, XOut);
  if ((*NOut) == 0) return 0;
  totPts = (*NOut);

  //**/ ---------------------------------------------------------------
  //**/ generate the data points 
  //**/ ---------------------------------------------------------------
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();

  if (outputLevel_ >= 1) printf("HKriging interpolation begins....\n");
  predict(totPts, *XOut, *YOut, NULL);
  if (outputLevel_ >= 1) printf("HKriging interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int HKriging::gen1DGridData(double *XIn,double *YIn,int ind1,
                            double *settings,int *NOut, double **XOut, 
                            double **YOut)
{
  int    ii, kk, totPts;
  double HX;
  psVector vecXT;

  //**/ ---------------------------------------------------------------
  //**/ generate the hyperparameters
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("HKriging training begins....\n");
  train(XIn,YIn);
  if (outputLevel_ >= 1) printf("HKriging training completed.\n");
 
  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ---------------------------------------------------------------
  psVector vecXOut;
  vecXOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (kk = 0; kk < nInputs_; kk++) 
      vecXT[ii*nInputs_+kk] = settings[kk]; 
    vecXT[ii*nInputs_+ind1] = HX * ii + VecLBs_[ind1];
    (*XOut)[ii] = HX * ii + VecLBs_[ind1];
  }
    
  //**/ ---------------------------------------------------------------
  //**/ interpolate
  //**/ ---------------------------------------------------------------
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  if (outputLevel_ >= 1) printf("HKriging interpolation begins....\n");
  predict(totPts, vecXT.getDVector(), *YOut, NULL);
  if (outputLevel_ >= 1) printf("HKriging interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 2D results for display
// ------------------------------------------------------------------------
int HKriging::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                            double *settings, int *NOut, double **XOut, 
                            double **YOut)
{
  int ii, jj, kk, totPts, index;
  psVector vecHX, vecXT;

  //**/ ---------------------------------------------------------------
  //**/ generate the hyperparameters
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("HKriging training begins....\n");
  train(XIn, YIn);
  if (outputLevel_ >= 1) printf("HKriging training completed.\n");
  
  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ---------------------------------------------------------------
  psVector vecXOut;
  vecXOut.setLength(totPts*2);
  (*XOut) = vecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      index = ii * nPtsPerDim_ + jj;
      for (kk = 0; kk < nInputs_; kk++) 
        vecXT[index*nInputs_+kk] = settings[kk]; 
      vecXT[index*nInputs_+ind1]  = vecHX[0] * ii + VecLBs_[ind1];
      vecXT[index*nInputs_+ind2]  = vecHX[1] * jj + VecLBs_[ind2];
      (*XOut)[index*2]   = vecHX[0] * ii + VecLBs_[ind1];
      (*XOut)[index*2+1] = vecHX[1] * jj + VecLBs_[ind2];
    }
  }
    
  //**/ ---------------------------------------------------------------
  //**/ interpolate
  //**/ ---------------------------------------------------------------
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  if (outputLevel_ >= 1) printf("HKriging interpolation begins....\n");
  predict(totPts, vecXT.getDVector(), *YOut, NULL);
  if (outputLevel_ >= 1) printf("HKriging interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int HKriging::gen3DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                            int ind3, double *settings, int *NOut, 
                            double **XOut, double **YOut)
{
  int ii, jj, ll, kk, totPts, index;
  psVector vecHX, vecXT;

  //**/ ---------------------------------------------------------------
  //**/ generate the hyperparameters
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("HKriging training begins....\n");
  train(XIn, YIn);
  if (outputLevel_ >= 1) printf("HKriging training completed.\n");
 
  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ---------------------------------------------------------------
  psVector vecXOut;
  vecXOut.setLength(totPts*3);
  (*XOut) = vecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      for (ll = 0; ll < nPtsPerDim_; ll++) 
      {
        index = ii * nPtsPerDim_ * nPtsPerDim_ + jj * nPtsPerDim_ + ll;
        for (kk = 0; kk < nInputs_; kk++) 
          vecXT[index*nInputs_+kk] = settings[kk]; 
        vecXT[index*nInputs_+ind1] = vecHX[0] * ii + VecLBs_[ind1];
        vecXT[index*nInputs_+ind2] = vecHX[1] * jj + VecLBs_[ind2];
        vecXT[index*nInputs_+ind3] = vecHX[2] * ll + VecLBs_[ind3];
        (*XOut)[index*3]   = vecHX[0] * ii + VecLBs_[ind1];
        (*XOut)[index*3+1] = vecHX[1] * jj + VecLBs_[ind2];
        (*XOut)[index*3+2] = vecHX[2] * ll + VecLBs_[ind3];
      }
    }
  }
   
  //**/ ---------------------------------------------------------------
  //**/ interpolate
  //**/ ---------------------------------------------------------------
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  if (outputLevel_ >= 1) printf("HKriging interpolation begins....\n");
  predict(totPts, vecXT.getDVector(), *YOut, NULL);
  if (outputLevel_ >= 1) printf("HKriging interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int HKriging::gen4DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                            int ind3, int ind4, double *settings, 
                            int *NOut, double **XOut, double **YOut)
{
  int ii, jj, ll, mm, kk, totPts, index;
  psVector vecHX, vecXT;

  //**/ ---------------------------------------------------------------
  //**/ generate the hyperparameters
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("HKriging training begins....\n");
  train(XIn, YIn);
  if (outputLevel_ >= 1) printf("HKriging training completed.\n");
  
  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ---------------------------------------------------------------
  psVector vecXOut;
  vecXOut.setLength(totPts*4);
  (*XOut) = vecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      for (ll = 0; ll < nPtsPerDim_; ll++) 
      {
        for (mm = 0; mm < nPtsPerDim_; mm++) 
        {
          index = ii*nPtsPerDim_*nPtsPerDim_*nPtsPerDim_ +
                  jj*nPtsPerDim_*nPtsPerDim_ + ll*nPtsPerDim_ + mm;
          for (kk = 0; kk < nInputs_; kk++) 
            vecXT[index*nInputs_+kk] = settings[kk]; 
          vecXT[index*nInputs_+ind1] = vecHX[0] * ii + VecLBs_[ind1];
          vecXT[index*nInputs_+ind2] = vecHX[1] * jj + VecLBs_[ind2];
          vecXT[index*nInputs_+ind3] = vecHX[2] * ll + VecLBs_[ind3];
          vecXT[index*nInputs_+ind4] = vecHX[3] * mm + VecLBs_[ind4];
          (*XOut)[index*4]   = vecHX[0] * ii + VecLBs_[ind1];
          (*XOut)[index*4+1] = vecHX[1] * jj + VecLBs_[ind2];
          (*XOut)[index*4+2] = vecHX[2] * ll + VecLBs_[ind3];
          (*XOut)[index*4+3] = vecHX[3] * mm + VecLBs_[ind4];
        }
      }
    }
  }
    
  //**/ ---------------------------------------------------------------
  //**/ interpolate
  //**/ ---------------------------------------------------------------
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  if (outputLevel_ >= 1) printf("HKriging interpolation begins....\n");
  predict(totPts, vecXT.getDVector(), *YOut, NULL);
  if (outputLevel_ >= 1) printf("HKriging interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double HKriging::evaluatePoint(double *X)
{
  int    iOne=1;
  double Y=0.0;
  predict(iOne, X, &Y, NULL);
  return Y;
}

// ************************************************************************
// Evaluate a number of points 
// ------------------------------------------------------------------------
double HKriging::evaluatePoint(int npts, double *X, double *Y)
{
  predict(npts, X, Y, NULL);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point, return also the standard deviation 
// ------------------------------------------------------------------------
double HKriging::evaluatePointFuzzy(double *X, double &Ystd)
{
  int    iOne=1;
  double Y=0.0;
  predict(iOne, X, &Y, &Ystd);
  return Y;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double HKriging::evaluatePointFuzzy(int npts, double *X, double *Y, 
                                    double *Ystds)
{
  predict(npts, X, Y, Ystds);
  return 0.0;
}

// ************************************************************************
// training 
// ------------------------------------------------------------------------
double HKriging::train(double *X, double *Y)
{
  int    ii, jj, kk, count, status, nDists, iOne=1, iZero=0;
  double *XDists, dist, ddata, TValue;
  FILE   *fp;

  //**/ ============================================================
  // normalize the input and outputs 
  //**/ ============================================================
  vecNormalX_.setLength(nSamples_*nInputs_);
  for (ii = 0; ii < nSamples_; ii++)
  {
    for (jj = 0; jj < nInputs_; jj++)
      vecNormalX_[ii*nInputs_+jj] = (X[ii*nInputs_+jj] - VecLBs_[jj])/
                                    (VecUBs_[jj] - VecLBs_[jj]);
  }
  for (jj = 0; jj < nInputs_; jj++)
  {
    VecXMeans_[jj] = VecLBs_[jj];
    VecXStds_[jj] = VecUBs_[jj] - VecLBs_[jj];
  }
  vecNormalY_.setLength(nSamples_);
  initOutputScaling(Y, vecNormalY_.getDVector());
  HKRI_YStd = YStd_;

  //**/ ============================================================
  // compute distances between all pairs of inputs (nDists, XDists)
  //**/ needed for creating the correlation matrix
  //**/ ============================================================
  status = computeDistances(&XDists, &nDists);
  if (status != 0)
  {
    printf("HKriging INFO: since there are repeated sample points.\n");
    printf("    Please prune the sample and do it again.\n");
  }
  if (HKRI_nugget != 0.0 && outputLevel_ > 0) 
    printf("HKriging INFO: nugget = %e\n",HKRI_nugget);

  //**/ ============================================================
  // optimize
  //**/ ============================================================
  int    nBasis, Cleng, maxfun, pLevel, nPts, nSamOpt;
  double rhobeg=1.0, rhoend=1.0e-4, optTheta;
  double optY=PSUADE_UNDEFINED;

  //**/ ----------------------------------------------------
  //**/ if slow mode, need to make sure the psuade_stop
  //**/ file has been cleaned up
  //**/ ----------------------------------------------------
  fp = fopen("psuade_stop", "r");
  if (fp != NULL)
  {
    printf("HKriging ERROR: remove the 'psuade_stop' file\n");
    printf("                first and re-do.\n");
    fclose(fp);
    exit(1);
  }

  //**/ ----------------------------------------------------
  //**/ get number of basis 
  //**/ ----------------------------------------------------
  if (nSamples_ <= nInputs_ + 1) pOrder_ = 0;
  if (pOrder_ == 1) nBasis = nInputs_ + 1;
  else              nBasis = 1;
  Cleng = nSamples_ + nBasis;

  //**/ ----------------------------------------------------
  //**/ prepare for optimization 
  //**/ ----------------------------------------------------
  double   TUpper=30.0, TLower=0.1;
  psVector vecWT, vecSamOpt;
  if (VecXMeans_[0] == 0 && VecXStds_[0] == 1.0)
  {
    TUpper = 30.0 * (VecUBs_[0]-VecLBs_[0]);
    TLower =  0.1 * (VecUBs_[0]-VecLBs_[0]);
  }
  rhobeg = TUpper - TLower;
  rhobeg *= 0.5;
  rhoend = rhobeg * optTolerance_;
  nPts = (iOne + 1) * (iOne + 2) / 2;
  jj = (nPts+13) * (nPts+iOne) + 3*iOne*(iOne+3)/2;
  kk = (nPts+5)*(nPts+iOne)+3*iOne*(iOne+5)/2+1;
  if (jj > kk) vecWT.setLength(jj);
  else         vecWT.setLength(kk);
  if (outputLevel_ > 0)
  {
    printEquals(PL_INFO, 0);
    printf("* HKriging optimization tolerance = %e\n", rhoend);
  }

  //**/ ----------------------------------------------------
  //**/ generate a sample
  //**/ ----------------------------------------------------
  if (fastMode_ == 1)
  {
    if (outputLevel_ >= 1) 
      printf("HKriging training (2) begins.... (order = %d)\n",pOrder_);
    nSamOpt = 1;
    vecSamOpt.setLength(1);
    vecSamOpt[0] = Theta_;
  }
  else
  {
    if (outputLevel_ >= 1) 
      printf("HKriging training (3) begins.... (order = %d)\n",pOrder_);
    nSamOpt = 10;
    vecSamOpt.setLength(10);
    for (ii = 0; ii < nSamOpt; ii++)
      vecSamOpt[ii] = (TUpper - TLower) / (nSamOpt - 1.0) * ii + TLower;
  }

  //**/ ----------------------------------------------------
  //**/ pass data to global variables for external calls
  //**/ ----------------------------------------------------
  HKRI_XDists = XDists;
  HKRI_nSamples = nSamples_;
  HKRI_nInputs = nInputs_;
  HKRI_X = vecNormalX_.getDVector();
  HKRI_Y = vecNormalY_.getDVector();
  psVector vecYtmp;
  vecYtmp.setLength(Cleng);
  HKRI_Ytmp = vecYtmp.getDVector();
  HKRI_OptTheta = Theta_;
  maxfun = 5000;
  HKRI_OptY = PSUADE_UNDEFINED;
  HKRI_outputLevel = outputLevel_;
  optTheta = 0.0;

  psVector vecSMat, vecFMat, vecFTmp, vecMMat, vecTValSave;
  vecSMat.setLength(nSamples_*nSamples_);
  vecFMat.setLength(nSamples_*nBasis);
  vecFTmp.setLength(nSamples_*nBasis);
  vecMMat.setLength(nBasis*nBasis);
  HKRI_SMatrix = vecSMat.getDVector();
  HKRI_FMatrix = vecFMat.getDVector();
  HKRI_FMatTmp = vecFTmp.getDVector();
  HKRI_MMatrix = vecMMat.getDVector();
  vecTValSave.setLength(nSamOpt);
  double dmean, dstd;
  int    stopFlag = 0;

  //**/ ----------------------------------------------------
  //**/ call optimizer 
  //**/ ----------------------------------------------------
  //**/ use pLevel=4444 to tell newuoa to use this function
  //**/ for now set the initial guess to mid point
  for (kk = 0; kk < nSamOpt; kk++)
  {
    if (outputLevel_ > 1) 
      printf("HKriging multi-start optimization: start = %d (%d)\n",
             kk+1, nSamOpt);
    HKRI_iter = 0;
    TValue = vecSamOpt[kk];
    if (outputLevel_ > 1) 
    {
      printf("HKriging: initial length scale = %e\n", TValue);
    }
    HKRI_noProgressCnt = 0;

    pLevel = -1;
#ifdef HAVE_BOBYQA
    pLevel = 1113;
#endif
#ifdef HAVE_NEWUOA
    pLevel = 4444;
#endif
#ifdef HAVE_BOBYQA
    if (pLevel == 1113)
      kbobyqa_(&iOne,&nPts,&TValue,&TLower,&TUpper,&rhobeg,
               &rhoend,&pLevel,&maxfun,vecWT.getDVector());
#endif
#ifdef HAVE_NEWUOA
    if (pLevel == 4444)
      newuoa_(&iOne,&nPts,&TValue,&rhobeg,&rhoend,&pLevel,&maxfun,
              vecWT.getDVector());
#endif
    if (pLevel == -1)
    {
      printf("ERROR: No optimizer unavailable.\n");
      exit(1);
    }
    if (outputLevel_ > 1) 
    {
      printf("HKriging multi-start optimization: iteration = %d (%d)\n",
             kk+1, nSamOpt);
      printf("HKriging: final length scale = %e\n", TValue);
      printf("HKriging final objective value = %e (ref = %e)\n", 
             HKRI_currY, rhoend);
    }
    if (HKRI_OptY < optY)
    {
      optY = HKRI_OptY;
      optTheta = HKRI_OptTheta;
      if (optY < rhoend)
      {
        if (outputLevel_ > 2) 
          printf("HKriging INFO: termination (sufficiently accurate)\n");
        break;
      }
    }
    fp = fopen("psuade_stop", "r");
    if (fp != NULL)
    {
      fclose(fp);
      fp = NULL;
      printf("HKriging: psuade_stop file found - terminating ....\n");
      break;
    }
    vecTValSave[kk] = TValue;
    if (kk >= 4)
    {
      dmean = 0.0;
      for (jj = 1; jj < kk+1; jj++) dmean += vecTValSave[jj];
      dmean /= (double) kk; 
      dstd = 0.0;
      for (jj = 1; jj < kk+1; jj++)
        dstd += pow(vecTValSave[jj]-dmean,2.0);
      dstd = sqrt(dstd/(double) (kk-1));
      if (dstd/dmean < 0.01) stopFlag++;
      if (outputLevel_ > 1) 
        printf("Opt convergence check: input %d: mean=%e, std=%e\n",
               ii+1,dmean,dstd);
    }
    if (stopFlag == nInputs_) 
    {
      if (outputLevel_ >= 1) 
      {
        printf("HKriging INFO: same optimum after %d iterations.\n",
               kk+1);
        printf("              Stop further processing.\n");
      }
      break;
    }  
  }
  //**/ ----------------------------------------------------
  //**/ display results and clean up
  //**/ ----------------------------------------------------
  Theta_ = optTheta;
  if (outputLevel_ > 1) 
  {
    printf("HKriging: optimal length scale = %e\n", Theta_);
    printf("HKriging: optimal objective value = %e\n",HKRI_OptY);
  }
  HKRI_XDists = NULL;
  HKRI_X = NULL;
  HKRI_Y = NULL;
  HKRI_Ytmp = NULL;
  HKRI_SMatrix = NULL;
  HKRI_FMatrix = NULL;
  HKRI_FMatTmp = NULL;
  HKRI_MMatrix = NULL;
  if (outputLevel_ > 0)
  {
    if (fastMode_ == 2)
         printf("HKriging training (2) ends.\n");
    else printf("HKriging training (3) ends.\n");
  }

  //**/ ============================================================
  //**/ compute output variance 
  //**/    ||inv(R)Y - RF inv([RF]^T RF) [RF]^T inv(R)Y||/nBasis
  //**/ where RF = inv(R) F
  //**/ At the end, this section will produce
  //**/ Rmatrix has inverse of R (Cholesky decomposition)
  //**/ Mmatrix has inverse of F^T R^-1 F (Cholesky decomposition)
  //**/ ============================================================
  //**/ -------------------------------------------------------
  //**/ 1. Cholesky decomposition for the covarance matrix 
  //**/ -------------------------------------------------------
  char uplo='L';
  char trans='N';
  char diag='N';
  count = 0;
  vecRMat_.setLength(nSamples_*nSamples_);
  for (jj = 0; jj < nSamples_; jj++)
  {
    vecRMat_[jj*nSamples_+jj] = 1.0 + 1e-15 * nSamples_;
    if (HKRI_nugget != 0.0) vecRMat_[jj*nSamples_+jj] += HKRI_nugget;

    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs_; ii++)
        dist += pow(XDists[count*nInputs_+ii]/Theta_,HKRI_Exponent);
      dist = exp(-dist);
      if (dist < 1.0e-16) dist = 0.0;
      vecRMat_[jj*nSamples_+kk] = dist;
      vecRMat_[kk*nSamples_+jj] = dist;
      count++;
    }
  }
  if (outputLevel_ > 4)
  {
    printf("HKriging: covariance matrix for Cholesky decompositon.\n");
    for (jj = 0; jj < nSamples_; jj++)
    {
      for (kk = 0; kk < nSamples_; kk++)
        printf("%e ", vecRMat_[jj*nSamples_+kk]);
      printf("\n");
    }
  }
  dpotrf_(&uplo, &nSamples_,vecRMat_.getDVector(), &nSamples_, &status);
  if (status != 0) 
  {
    printf("HKriging ERROR: Cholesky decomposition not successful.\n");
    exit(1);
  }
  //**/ -------------------------------------------------------
  //**/ 2. compute CY = inv(C) Y where R = C C^T
  //**/ -------------------------------------------------------
  int    inc=1;
  psVector vecCY;
  vecCY.setLength(nSamples_);
  for (jj = 0; jj < nSamples_; jj++) vecCY[jj] = vecNormalY_[jj];
  dtrsv_(&uplo,&trans,&diag,&nSamples_,vecRMat_.getDVector(),&nSamples_,
         vecCY.getDVector(), &inc);
  //**/ -------------------------------------------------------
  //**/ 3. Create F (nSamples x nBasis)
  //**/ -------------------------------------------------------
  psVector vecCFMat, vecRFMat;
  vecCFMat.setLength(nBasis*nSamples_);
  vecRFMat.setLength(nBasis*nSamples_);
  for (jj = 0; jj < nSamples_; jj++)
  {
    vecCFMat[jj] = vecRFMat[jj] = 1.0;
    if (pOrder_ == 1)
    {
      for (kk = 1; kk < nInputs_+1; kk++)
      {
        ddata = vecNormalX_[jj*nInputs_+kk-1];
        vecCFMat[kk*nSamples_+jj] = vecRFMat[kk*nSamples_+jj] = ddata;
      }
    }
  }
  //**/ -------------------------------------------------------
  //**/ 4. F = inv(C) * F
  //**/ -------------------------------------------------------
  double *cfMat = vecCFMat.getDVector();
  for (ii = 0; ii < nBasis; ii++)
  {
    dtrsv_(&uplo,&trans,&diag,&nSamples_,vecRMat_.getDVector(),
           &nSamples_, &cfMat[ii*nSamples_], &inc);
  } 
  //**/ -------------------------------------------------------
  //**/ 5. compute V1 = [inv(C) F]^T inv(C)Y 
  //**/ -------------------------------------------------------
  vecV1_.setLength(nBasis);
  for (jj = 0; jj < nBasis; jj++)
  {
    ddata = 0.0;
    for (ii = 0; ii < nSamples_; ii++)
      ddata += vecCFMat[ii+jj*nSamples_]*vecCY[ii];
    vecV1_[jj] = ddata;
  }
  //**/ -------------------------------------------------------
  //**/ 6. compute M = [inv(C) F]^T [inv(C) F] = F^T inv(R) F 
  //**/ -------------------------------------------------------
  vecMMat_.setLength(nBasis*nBasis);
  for (jj = 0; jj < nBasis; jj++)
  {
    for (kk = jj; kk < nBasis; kk++)
    {
      ddata = 0.0;
      for (ii = 0; ii < nSamples_; ii++)
        ddata += vecCFMat[ii+jj*nSamples_]*vecCFMat[ii+kk*nSamples_];
      vecMMat_[jj+kk*nBasis] = vecMMat_[kk+jj*nBasis] = ddata;
    }
  }
  //**/ -------------------------------------------------------
  //**/ 7. compute V1 = inv(M) F^T inv(R) Y 
  //**/ -------------------------------------------------------
  dpotrf_(&uplo, &nBasis, vecMMat_.getDVector(), &nBasis, &status);
  if (status != 0) 
  {
    printf("HKriging ERROR: Cholesky decomposition not successful.\n");
    exit(1);
  }
  dpotrs_(&uplo, &nBasis,&inc,vecMMat_.getDVector(), &nBasis, 
          vecV1_.getDVector(), &nBasis, &status);
  if (status != 0) 
  {
    printf("HKriging ERROR: LU solve not successful.\n");
    exit(1);
  }
  //**/ -------------------------------------------------------
  //**/ 8. compute CY = inv(C)Y - inv(C)F inv(M) CF^T inv(C) Y 
  //**/ -------------------------------------------------------
  for (ii = 0; ii < nBasis; ii++)
  {
    for (jj = 0; jj < nSamples_; jj++) 
      vecCY[jj] -= vecCFMat[jj+ii*nSamples_] * vecV1_[ii];
  } 
  HKrigingVariance_ = 0.0;
  for (jj = 0; jj < nSamples_; jj++) 
    HKrigingVariance_ += vecCY[jj] * vecCY[jj];
  HKrigingVariance_ /= (double) nSamples_;
  HKrigingVariance_ *= YStd_ * YStd_;
  if (outputLevel_ > 0) 
    printf("HKriging variance = %e\n",HKrigingVariance_);
  //**/ -------------------------------------------------------
  //**/ 9. compute V2 = inv(C) [inv(C) Y - inv(C) CF V1
  //**/               = inv(R) Y - inv(R) F V1
  //**/ -------------------------------------------------------
  vecV2_.setLength(nSamples_);
  for (ii = 0; ii < nSamples_; ii++) vecV2_[ii] = vecCY[ii];
  trans = 'T';
  dtrsv_(&uplo,&trans,&diag,&nSamples_,vecRMat_.getDVector(),&nSamples_,
         vecV2_.getDVector(), &inc);
  //**/ -------------------------------------------------------
  //**/ 10. clean up
  //**/ -------------------------------------------------------
  delete [] XDists;
  //**/ ============================================================
  //**  END
  //**/ ============================================================
  return 0.0;
}

// ************************************************************************
// predict 
// ------------------------------------------------------------------------
double HKriging::predict(int length, double *X, double *Y, double *YStds)
{
  int    ii, jj, kk, ll, status, nRows, nBasis, leng,maxLeng=500;
  int    offset;
  double ddata, dist, mean, stdev;
  char   uplo='L';

  //**/ -------------------------------------------------------
  //**/ normalize the test set inputs and put into vecWX
  //**/ -------------------------------------------------------
  nRows = nSamples_ + 1;
  if (pOrder_ == 1) nRows = nRows + nInputs_;
  nBasis = nRows - nSamples_;
  if (vecWork_.length() == 0)
  {
    vecWork_.setLength(2*nSamples_*maxLeng+maxLeng);
    vecWX_.setLength(maxLeng*nInputs_+2*maxLeng*nBasis);
  }
  //**/ -------------------------------------------------------
  //**/ compute correlation with sample inputs ==> r
  //**/ -------------------------------------------------------
  for (ll = 0; ll < length; ll+=maxLeng)
  {
    leng = maxLeng;
    if (ll+leng > length) leng = length - ll;
    //**/ normalize the inputs
    for (ii = 0; ii < nInputs_; ii++)
    {
      mean  = VecXMeans_[ii];
      stdev = VecXStds_[ii];
      for (kk = 0; kk < leng; kk++)
        vecWX_[kk*nInputs_+ii] = (X[(ll+kk)*nInputs_+ii]-mean)/stdev;
    }
    //**/ compute the right hand side 
    for (kk = 0; kk < leng; kk++)
    {
      for (jj = 0; jj < nSamples_; jj++)
      {
        dist = 0.0;
        for (ii = 0; ii < nInputs_; ii++)
        {
          ddata = vecNormalX_[jj*nInputs_+ii]-vecWX_[kk*nInputs_+ii];
          dist += ddata * ddata / (Theta_ * Theta_);
        }
        ddata = exp(-dist);
        if (ddata < 1.0e-16) ddata = 0.0;
        vecWork_[kk*nSamples_+jj] = ddata;
        vecWork_[nSamples_*leng+kk*nSamples_+jj] = ddata;
      }
    }
    //**/ -------------------------------------------------------
    //**/ Recall V1 = inv(F^T inv(R) F) F^T inv(R) Y 
    //**/ Recall V2 = inv(R) Y - inv(R)F inv(M) F^T inv(R) Y 
    //**/           = inv(R) Y - inv(C) F V1
    //**/ estimate y = [1 x]' * V1 + r' * V2
    //**/ -------------------------------------------------------
    for (kk = 0; kk < leng; kk++)
    {
      ddata = 1.0 * vecV1_[0];
      if (pOrder_ == 1)
        for (ii = 1; ii <= nInputs_; ii++) 
          ddata += vecWX_[kk*nInputs_+ii-1]*vecV1_[ii];
      for (jj = 0; jj < nSamples_; jj++) 
        ddata += vecWork_[nSamples_*leng+kk*nSamples_+jj] * vecV2_[jj];
      Y[ll+kk] = ddata * YStd_ + YMean_;
    }
    if (YStds != NULL)
    {
      //**/ ----------------------------------------------------
      //**/ compute s_k = inv(R) r_k for all k
      //**/ (vecWork_[leng*nSamples])
      //**/ ----------------------------------------------------
      double *workArray = vecWork_.getDVector();
      dpotrs_(&uplo, &nSamples_,&leng,vecRMat_.getDVector(),&nSamples_,
              &workArray[leng*nSamples_], &nSamples_, &status);
      //**/ ----------------------------------------------------
      //**/ compute r_k^T inv(R) r_k for all k
      //**/ (vecWork[2*leng*nSamples])
      //**/ ----------------------------------------------------
      for (kk = 0; kk < leng; kk++)
      {
        ddata = 0.0;
        for (jj = 0; jj < nSamples_; jj++)
          ddata += vecWork_[kk*nSamples_+jj] *
                   vecWork_[leng*nSamples_+kk*nSamples_+jj];
        vecWork_[2*leng*nSamples_+kk] = ddata; 
      }
      //**/ ----------------------------------------------------
      //**/ compute u = F^T s_k - f (=> vecWX_[leng*nInputs])
      //**/ ----------------------------------------------------
      for (kk = 0; kk < leng; kk++)
      {
        ddata = 0.0;
        for (jj = 0; jj < nSamples_; jj++)
           ddata += vecWork_[leng*nSamples_+kk*nSamples_+jj]; 
        vecWX_[leng*nInputs_+kk*nBasis] = ddata - 1.0; 
        vecWX_[leng*nInputs_+leng*nBasis+kk*nBasis] = ddata - 1.0; 
        for (ii = 1; ii < nBasis; ii++)
        {
          ddata = 0.0;
          for (jj = 0; jj < nSamples_; jj++)
            ddata += vecWX_[kk*nInputs_+ii-1] *
                     vecWork_[leng*nSamples_+kk*nSamples_+jj]; 
          ddata -= vecWX_[kk*nInputs_+ii-1];
          vecWX_[leng*nInputs_+kk*nBasis+ii] = ddata; 
          vecWX_[leng*nInputs_+leng*nBasis+kk*nBasis+ii] = ddata; 
        }
      }
      //**/ ----------------------------------------------------
      //**/ compute v = inv(M) u
      //**/ ----------------------------------------------------
      double *workX = vecWX_.getDVector();
      dpotrs_(&uplo, &nBasis, &leng, vecMMat_.getDVector(), &nSamples_,
              &workX[leng*nInputs_+leng*nBasis], &nBasis, &status);

      //**/ ----------------------------------------------------
      //**/ compute final standard dev = sig(1+ u' * v - r'inv(R)r)
      //**/ ----------------------------------------------------
      for (kk = 0; kk < leng; kk++)
      {
        ddata = 0.0;
        for (ii = 0; ii < nBasis; ii++)
          ddata += vecWX_[leng*nInputs_+kk*nBasis+ii] *
                   vecWX_[leng*nInputs_+(leng+kk)*nBasis+ii];
        ddata = HKrigingVariance_*
                (1.0 + ddata-vecWork_[2*leng*nSamples_+kk]);
        if (ddata < 0.0)
        {
          printf("HKriging WARNING: prediction variance < 0\n");
          ddata = 0.0;
        }
        YStds[ll+kk] = sqrt(ddata);
      }
    }
  }
  return 0.0;
}

// ************************************************************************
// compute distances between all pairs of inputs
// ------------------------------------------------------------------------
int HKriging::computeDistances(double **XDists, int *length)
{
  int    ii, jj, kk, count, error=0;
  double dist;
  psVector vecLDists;

  vecLDists.setLength((nSamples_*(nSamples_-1)/2)*nInputs_);
  count = 0;
  for (jj = 0; jj < nSamples_; jj++)
  {
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs_; ii++)
      {
        vecLDists[count*nInputs_+ii] = vecNormalX_[jj*nInputs_+ii] - 
                            vecNormalX_[kk*nInputs_+ii];
        if (vecLDists[count*nInputs_+ii] < 0) 
          vecLDists[count*nInputs_+ii] = - vecLDists[count*nInputs_+ii]; 
        dist += pow(vecLDists[count*nInputs_+ii], 2.0);
      }
      if (dist == 0.0)
      {
        printf("HKriging ERROR: repeated sample points.\n");
        printf("                Prune repeated points and re-run.\n");
        printf("Sample %d : (with sample point %d)\n", kk+1, jj+1);
        error = 1;
      }
      count++;
    }
  }
  (*length) = count;
  (*XDists) = vecLDists.takeDVector();
  return error;
}

