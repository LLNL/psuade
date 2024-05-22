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
// Functions for the class GP3
// AUTHOR : CHARLES TONG
// DATE   : 2017
// ************************************************************************
#ifdef WINDOWS
#include <windows.h>
#undef ERROR  //Windows already has a macro defined as ERROR
#undef IS_ERROR
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "GP3.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"
#include "Psuade.h"
#include "Sampling.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// ************************************************************************
// external functions
// ------------------------------------------------------------------------
extern "C" {
  void newuoa_(int *,int *,double *,double *,double *,int *,int *,double*);
#ifdef HAVE_LBFGS
#include "../../External/L-BFGS-B-C/src/lbfgsb.h"
#endif
}
GP3 *GP3Obj=NULL;
psVector GP3_OptX;
double   GP3_OptY=1e35;

// ************************************************************************
// resident function perform evaluation 
// ------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" 
{
#endif
  void *gp3newuoaevalfunc_(int *nInps, double *XValues, double *YValue)
  {
    int status;
    psVector VecG;
    if (GP3Obj == NULL)
    {
      printf("GP3 ERROR: no GP3 object in function evalution\n");
      exit(1);
    }
    psVector VecP;
    VecP.load(*nInps, XValues);
    (*YValue) = GP3Obj->computeGradients(VecP, VecG, status);
    if ((*YValue) < GP3_OptY)
    {
      GP3_OptY = *YValue;
      GP3_OptX = VecP;
    }
    return NULL;
  }    
#ifdef __cplusplus
}
#endif

// ************************************************************************
// Constructor for object class GP3
// ------------------------------------------------------------------------
GP3::GP3(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  //**/ linear terms have not been implemented yet
  optLinTerm_ = 0;
  faID_ = PSUADE_RS_GP3;
  expPower_ = 2.0; // but some prefers 1.9 (more stable?)
  nStarts_  = 5;   // number of starts in optimization
  strcpy(hyperParamFile_, ".psuade_gp3");
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
         "*           Gaussian Process Model\n");
    printOutTS(PL_INFO,
         "* Default exponential degree    = %e\n",expPower_);
    printOutTS(PL_INFO,
         "* Number of optimization starts = %d\n",nStarts_);
    printEquals(PL_INFO, 0);
  }
  if (psConfig_.InteractiveIsOn() && psConfig_.RSExpertModeIsOn())
  {
    char pString[101];
    printOutTS(PL_INFO,
         "* Default exponential degree = %e\n",expPower_);
    snprintf(pString,100,"Set the exponential degree to? [1.5,2.0] ");
    expPower_ = 1;
    while (expPower_ < 1.5 || expPower_ > 2) 
      expPower_ = getDouble(pString);
    printf("GP3: Exponential degree = %e\n", expPower_);
    snprintf(pString,100,"RS_GP_power = %e", expPower_);
    psConfig_.putParameter(pString);
    printOutTS(PL_INFO,
         "* Default optimization choice = %d starts\n",nStarts_);
    snprintf(pString,100,
             "Set the number of optimization starts ? [1 - 20] ");
    nStarts_ = getInt(1, 20, pString);
    printf("GP3: opt num_starts = %d\n", nStarts_);
    snprintf(pString,100,"RS_GP_nstarts = %d", nStarts_);
    psConfig_.putParameter(pString);
  }
  else
  {
    char winput1[1000], winput2[1000];
    char *cString = psConfig_.getParameter("RS_GP_power");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %lg", winput1, winput2, &expPower_);
      if (expPower_ < 1.5 || expPower_ > 2) 
      {
        printOutTS(PL_INFO,
           "GP3 exponential (%e) invalid - reset to 2\n",expPower_);
        expPower_ = 2.0;
      }
      printOutTS(PL_INFO,"GP3 INFO: exponential degree set to %e\n",
                   expPower_);
    }
    cString = psConfig_.getParameter("RS_GP_nstarts");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nStarts_);
      printOutTS(PL_INFO,
         "GP3 INFO: Number of optimization starts = %d\n",nStarts_);
    } 
  }
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
GP3::~GP3()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int GP3::initialize(double *XIn, double *YIn)
{
  //**/ ----------------------------------------------------------------
  //**/ normalize sample values
  //**/ ----------------------------------------------------------------
  VecXDataN_.setLength(nSamples_*nInputs_);
  initInputScaling(XIn, VecXDataN_.getDVector(), 1);
  VecYDataN_.setLength(nSamples_);
  initOutputScaling(YIn, VecYDataN_.getDVector());
   
  //**/ ----------------------------------------------------------------
  //**/ generate the Gaussian hyperparameters
  //**/ ----------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
    printf("GP3 training begins....\n");
  train();
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
    printf("GP3 training completed.\n");
  if (psConfig_.RSCodeGenIsOn()) genRSCode();
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int GP3::genNDGridData(double *XIn, double *YIn, int *NOut, double **XOut, 
                      double **YOut)
{
  int    totPts;
  double *XX;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ---------------------------------------------------------------
  //**/ generating regular grid data
  //**/ ---------------------------------------------------------------
  genNDGrid(NOut, &XX);
  if ((*NOut) == 0) return 0;
  totPts = (*NOut);

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points and generate them
  //**/ ---------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("GP3 interpolation begins....\n");
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  evaluatePoint(totPts, XX, (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("GP3 interpolation completed.\n");
  (*XOut) = XX;
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int GP3::gen1DGridData(double *XIn, double *YIn, int ind1,double *settings, 
                       int *NOut, double **XOut, double **YOut)
{
  int    totPts, ii, kk, ndim=1;
  double HX;
  psVector vecXT;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  (*NOut) = nPtsPerDim_;
  totPts = nPtsPerDim_;

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts*ndim);
  (*XOut) = VecXOut.takeDVector();
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (kk = 0; kk < nInputs_; kk++) 
      vecXT[ii*nInputs_+kk] = settings[kk]; 
    vecXT[ii*nInputs_+ind1] = HX * ii + VecLBs_[ind1];
    (*XOut)[ii] = HX * ii + VecLBs_[ind1];
  }
   
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("GP3 interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("GP3 interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 2D results for display
// ------------------------------------------------------------------------
int GP3::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                  double *settings,int *NOut,double **XOut,double **YOut)
{
  int totPts, ii, jj, kk, index, ndim=2;
  psVector vecXT, vecHX;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  vecHX.setLength(ndim);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  totPts = nPtsPerDim_ * nPtsPerDim_;
  (*NOut) = totPts;

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts*ndim);
  (*XOut) = VecXOut.takeDVector();
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
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
    
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("GP3 interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("GP3 interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int GP3::gen3DGridData(double *XIn, double *YIn,int ind1,int ind2,int ind3,
                  double *settings,int *NOut,double **XOut,double **YOut)
{
  int totPts, ii, jj, kk, ll, index, ndim=3;
  psVector vecXT, vecHX;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  vecHX.setLength(ndim);
  vecHX[0] = (VecUBs_[ind1]-VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2]-VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3]-VecLBs_[ind3]) / (nPtsPerDim_ - 1); 
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  (*NOut) = totPts;

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts*ndim);
  (*XOut) = VecXOut.takeDVector();
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
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
        vecXT[index*nInputs_+ind1]  = vecHX[0] * ii + VecLBs_[ind1];
        vecXT[index*nInputs_+ind2]  = vecHX[1] * jj + VecLBs_[ind2];
        vecXT[index*nInputs_+ind3]  = vecHX[2] * ll + VecLBs_[ind3];
        (*XOut)[index*3]   = vecHX[0] * ii + VecLBs_[ind1];
        (*XOut)[index*3+1] = vecHX[1] * jj + VecLBs_[ind2];
        (*XOut)[index*3+2] = vecHX[2] * ll + VecLBs_[ind3];
      }
    }
  }
    
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("GP3 interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("GP3 interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int GP3::gen4DGridData(double *XIn,double *YIn,int ind1,int ind2, int ind3,
                       int ind4, double *settings, int *NOut,double **XOut, 
                       double **YOut)
{
  int totPts, ii, jj, kk, ll, mm, index, ndim=4;
  psVector vecXT, vecHX;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
 
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1]-VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2]-VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3]-VecLBs_[ind3]) / (nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4]-VecLBs_[ind4]) / (nPtsPerDim_ - 1); 
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  (*NOut) = totPts;

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts*ndim);
  (*XOut) = VecXOut.takeDVector();
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
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
          vecXT[index*nInputs_+ind1]  = vecHX[0] * ii + VecLBs_[ind1];
          vecXT[index*nInputs_+ind2]  = vecHX[1] * jj + VecLBs_[ind2];
          vecXT[index*nInputs_+ind3]  = vecHX[2] * ll + VecLBs_[ind3];
          vecXT[index*nInputs_+ind4]  = vecHX[3] * mm + VecLBs_[ind4];
          (*XOut)[index*4]   = vecHX[0] * ii + VecLBs_[ind1];
          (*XOut)[index*4+1] = vecHX[1] * jj + VecLBs_[ind2];
          (*XOut)[index*4+2] = vecHX[2] * ll + VecLBs_[ind3];
          (*XOut)[index*4+3] = vecHX[3] * mm + VecLBs_[ind4];
        }
      }
    }
  }
    
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("GP3 interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("GP3 interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double GP3::evaluatePoint(double *X)
{
  int    ii, iOne=1;
  double Y=0.0;
  psVector vecXT;
  if (VecXMeans_.length() == 0)
  {
    printf("GP3 ERROR : not initialized yet.\n");
    exit(1);
  }
  vecXT.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    vecXT[ii] = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
  interpolate(iOne, vecXT.getDVector(), &Y, NULL);
  Y = Y * YStd_ + YMean_;
  return Y;
}

// ************************************************************************
// Evaluate a number of points 
// ------------------------------------------------------------------------
double GP3::evaluatePoint(int npts, double *X, double *Y)
{
  int      ii, jj;
  psVector VecXX;

  if (VecXMeans_.length() == 0)
  {
    printf("GP3 ERROR : Not initialized yet.\n");
    exit(1);
  }
  VecXX.setLength(npts*nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    for (jj = 0; jj < npts; jj++)
      VecXX[jj*nInputs_+ii] = (X[jj*nInputs_+ii]-VecXMeans_[ii])/
                              VecXStds_[ii];
  interpolate(npts, VecXX.getDVector(), Y, NULL);
  for (jj = 0; jj < npts; jj++) Y[jj] = Y[jj] * YStd_ + YMean_;
  return 0.0;
}

// ************************************************************************
// Evaluate a given point, return also the standard deviation 
// ------------------------------------------------------------------------
double GP3::evaluatePointFuzzy(double *X, double &std)
{
  int      ii, iOne=1;
  double   Y=0.0;
  psVector VecXX;

  if (VecXMeans_.length() == 0)
  {
    printf("GP3 ERROR : Not initialized yet.\n");
    exit(1);
  }
  VecXX.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    VecXX[ii] = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
  interpolate(iOne, VecXX.getDVector(), &Y, &std);
  Y = Y * YStd_ + YMean_;
  if (std < 0) printf("GP3 ERROR: variance (%e) < 0\n",std);
  else         std = sqrt(std) * YStd_;
  return Y;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double GP3::evaluatePointFuzzy(int npts,double *X, double *Y,double *Ystds)
{
  int      ii, jj;
  psVector VecXX;

  if (VecXMeans_.length() == 0)
  {
    printf("GP3 ERROR : Not initialized yet.\n");
    exit(1);
  }
  VecXX.setLength(npts*nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    for (jj = 0; jj < npts; jj++)
      VecXX[jj*nInputs_+ii] = (X[jj*nInputs_+ii]-VecXMeans_[ii])/
                              VecXStds_[ii];
  interpolate(npts, VecXX.getDVector(), Y, Ystds);
  for (int ii = 0; ii < npts; ii++)
  {
    Y[ii] = Y[ii] * YStd_ + YMean_;
    if (Ystds[ii] < 0) 
         printf("GP3 ERROR: variance (%e) < 0\n", Ystds[ii]);
    else Ystds[ii] = sqrt(Ystds[ii]) * YStd_;
  }
  return 0.0;
}

// ************************************************************************
// train to get lengthScales, magnitude, and noise components 
// ------------------------------------------------------------------------
int GP3::train()
{
  int    ii, jj, kk, count;
  double dmax, dmin, dtemp;

  //**/ ----------------------------------------------------------
  //**/ allocate and initialize length scales
  //**/ ----------------------------------------------------------
  int nhypers = nInputs_ + 4;
  if (optLinTerm_) nhypers += nInputs_;

  VecHypers_.setLength(nhypers);
  for (ii = 0; ii < nInputs_; ii++)
  {
    dmax = -PSUADE_UNDEFINED;
    dmin =  PSUADE_UNDEFINED;
    for (kk = 0; kk < nSamples_; kk++)
    {
      if (VecXDataN_[kk*nInputs_+ii] > dmax) 
        dmax = VecXDataN_[kk*nInputs_+ii];
      if (VecXDataN_[kk*nInputs_+ii] < dmin) 
        dmin = VecXDataN_[kk*nInputs_+ii];
    }
    VecHypers_[ii] = 0.25 * (dmax - dmin);
    if (VecHypers_[ii] == 0)
    {
      printf("GP3 ERROR: Input %d is a constant.\n", ii+1);
      printf("           Prune this input first (use idelete).\n");
      exit(1);
    }
    VecHypers_[ii] = log(VecHypers_[ii]);
  }
  //**/ multiplier to the exponential term 
  dtemp = 0.0;
  for (kk = 0; kk < nSamples_; kk++) 
    if (PABS(VecYDataN_[kk]) > dtemp) dtemp = PABS(VecYDataN_[kk]);
  VecHypers_[nInputs_] = dtemp;
    
  //**/ constant added to all entries
  VecHypers_[nInputs_+1] = log(1e-12);
  //**/ constant term (YM)
  VecHypers_[nInputs_+2] = 0;
  //**/ constant added to diagonals
  VecHypers_[nInputs_+3] = log(1e-10);
  count = nInputs_ + 4;
  if (optLinTerm_) 
  {
    for (ii = 0; ii < nInputs_; ii++) 
      VecHypers_[count++] = 0.0;
  }
  //**/if (outputLevel_ > 0) 
  //**/{
  //**/  printf("GP3 initial hyperparameter values:\n");
  //**/  for (ii = 0; ii < VecHypers_.length(); ii++) 
  //**/    printf("   Initial VecHypers_ %2d  = %e\n",ii+1,
  //**/           VecHypers_[ii]);
  //**/}

  //**/ ----------------------------------------------------------
  //**/ get options and hyperparameters from users
  //**/ ----------------------------------------------------------
  //**/ keep optimizeFlag = 1 especially if OpenMP is used
  int    optimizeFlag=1, errFlag;
  double ddata;
  char   pString[101], winput[1000], *inStr;
  //**/ optimizeFlag = NEWUOA but since it is not as good, turn
  //**/ off this option for now
  //inStr = psConfig_.getParameter("GP3_optimize2");
  //if (inStr != NULL) optimizeFlag = 2;

  //**/ if parameter file exists, see if it is acceptable.
  errFlag = 0;
  FILE *fp = fopen(hyperParamFile_, "r");
  if (fp != NULL)
  {
    printf("GP3 INFO: Found a hyperparameter file %s.\n",
           hyperParamFile_);
    fscanf(fp, "%d", &ii);
    if (ii == 0) 
    {
      printf("GP3 ERROR: When reading %s\n",hyperParamFile_);
      printf("           The first line should be > 0.\n");
      printf("           This first line is the hyperparameter size.\n");
      errFlag = 1;
    }
    else
    {
      VecHypers_.setLength(ii);
      for (ii = 0; ii < VecHypers_.length(); ii++)
      {
        fscanf(fp, "%lg", &ddata);
        VecHypers_[ii] = ddata;
      } 
      fscanf(fp, "%d %d", &jj, &kk);
      //**/ make sure nSamples and nInputs match
      if (jj != nSamples_ || kk != nInputs_) 
      {
        errFlag = 1;
        printf("GP3 INFO: Unable to read hyperparameters from %s.\n",
               hyperParamFile_);
        printf("          Either nSamples or nInputs does not match.\n");
      }
    }
    fclose(fp);
    //**/ if read successfully, set optimize flag = 0
    if (errFlag == 0) 
    {
      printf("GP3 INFO: Hyperparameters has been read from %s.\n",
             hyperParamFile_);
      printf("GP3 INFO: If this is not what you have in mind,");
      printf(" exit, delete the\n");
      printf("          hyperparameter file, and restart.\n");
      optimizeFlag = 0;
    }   
    if (errFlag == 1)
    {
      printf("GP3 INFO: Hyperparameter file is found but ");
      printf("it is not valid.\n");
      printf("          Will perform optimization to obtain ");
      printf("Hyperparameters.\n");
      //snprintf(pString,100,"Enter hyperparameters manually ? (y or n) ");
      //getString(pString, winput);
      //if (winput[0] == 'y')
      //{
      //  for (ii = 0; ii < VecHypers_.length(); ii++)
      //  {
      //    snprintf(pString,100,"Enter hyperparameter %d : ",ii+1);
      //    VecHypers_[ii] = getDouble(pString);
      //  }
      //  optimizeFlag = 0;
      //}
    }
  }

#ifndef HAVE_LBFGS
  //**/ if LBFGS not available, do not optimize
  if (optimizeFlag == 1)
  {
    optimizeFlag = 0;
    for (ii = 0; ii <= nInputs_; ii++) VecHypers_[ii] = 1.0;
    for (ii = nInputs_+1; ii < VecHypers_.length(); ii++)
      VecHypers_[ii] = 0.0;
    printf("GP3 INFO: Since LBFGS is unavailable, no ");
    printf("optimization will be done.\n");
  }
#endif

  //**/ ----------------------------------------------------------
  //**/ compute pairwise distance
  //**/ ----------------------------------------------------------
  int status = computeDistances();
  if (status != 0 and optimizeFlag != 0)
  {
    printf("GP3 INFO: since there are repeated sample points. ");
    printf("GP will continue\n");
    printf("    without optimizing the length scales but instead ");
    printf("set them to 1\n");
    printf("    (GP can combine the repeated sample points but we ");
    printf("will let users\n");
    printf("    hanndle it themselves). This may not give a good ");
    printf("quality response\n"); 
    printf("    surface. So if you want to prune the sample first ");
    printf("and do it again,\n");
    printf("    terminate now. You are given 30 seconds to decide. ");
    printf("Otherwise, it\n");
    printf("    continue with optimization.\n");
    optimizeFlag = 0;
    for (ii = 0; ii < nInputs_; ii++) VecHypers_[ii] = 1.0;
    for (ii = nInputs_; ii < VecHypers_.length(); ii++)
      VecHypers_[ii] = 0.0;
#ifdef WINDOWS
    Sleep(20000);
#else
    sleep(20);
#endif
    printf("    10 more seconds.\n");
#ifdef WINDOWS
    Sleep(10000);
#else
    sleep(10);
#endif
  }

  //**/ ----------------------------------------------------------
  //**/ tracking optimization history 
  //**/ ----------------------------------------------------------
  VecXLows_.setLength(VecHypers_.length());
  VecXHighs_.setLength(VecHypers_.length());
  for (ii = 0; ii < VecHypers_.length(); ii++) 
  {
    VecXLows_[ii]  = +PSUADE_UNDEFINED;
    VecXHighs_[ii] = -PSUADE_UNDEFINED;
  }

  //**/ ----------------------------------------------------------
  //**/ initialize variables for optimization
  //**/ ----------------------------------------------------------
  int       nSamp=1000, fail, nHist, maxHist=10, its;
  double    minVal=1e35, dmean, dstd, FValue;
  psVector  VecYT, VecLB, VecUB, VecPVals, VecGrads, VecXS, VecYS;
  psVector  VecW, VecHist;
  psIVector VecIS;
  Sampling  *sampler=NULL;
  if (optimizeFlag == 1)
  {
    nhypers = VecHypers_.length();
    VecPVals.setLength(nhypers);
    //**/ Notes (Mar, 2017): these bounds have been carefully
    //**/    tuned to give good results. Do not change them.
    //**/    Since the inputs and outputs have been normalized,
    //**/    these bounds should work on most problems.
    //**/    (Ymean in [-1, 1] gives a little worse results)
    VecLB.setLength(nhypers);
    VecUB.setLength(nhypers);
    //**/ the ranges
    for (ii = 0; ii < nInputs_; ii++) 
    {
      if (VecXMeans_[ii] == 0 && VecXStds_[ii] == 1.0)
      {
        VecUB[ii] = 10.00 * (VecUBs_[ii]-VecLBs_[ii]);
        VecLB[ii] =  0.01 * (VecUBs_[ii]-VecLBs_[ii]);
      }
      else
      {
        VecUB[ii] = 10.0;
        VecLB[ii] = 0.01;
      }
    }
//  for (ii = 0; ii < nInputs_; ii++) 
//  {
//    VecLB[ii] = -4.0;
//    VecUB[ii] =  4.0;
//  }
    //**/ vertical magnitude
    VecLB[nInputs_] = -3.0;
    VecUB[nInputs_] = 10.0;
    //**/ nugget for the entire covariance matrix
    VecLB[nInputs_+1] = -28;
    VecUB[nInputs_+1] = - 3;
    //**/ Ymean 
    VecLB[nInputs_+2] = -0.5;
    VecUB[nInputs_+2] =  0.5;
    //**/ nugget for diagonal of the covariance matrix
    //**/ if nInputs == 2, nugget should be larger to
    //**/ avoid oscillations
    VecLB[nInputs_+3] = -16;
    if (nInputs_ <= 2) VecLB[nInputs_+3] = log(1e-12);
    VecUB[nInputs_+3] = -2;
    count = nInputs_ + 4;
    if (optLinTerm_) 
    {
      for (ii = 0; ii < nInputs_; ii++) 
      {
        VecLB[count] = - 5*VecHypers_[nInputs_]/VecHypers_[ii];
        VecUB[count++] = 5*VecHypers_[nInputs_]/VecHypers_[ii];
      }
    }
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 2)
    {
      for (ii = 0; ii < nhypers; ii++)
        printf("   Hyperparameter bounds %3d = %12.4e %12.4e\n",ii+1,
               VecLB[ii],VecUB[ii]);
    }

    //**/ create a sample to be used as multi-starts
    nSamp = nStarts_;
    if (nhypers > 51)
      sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
    else
      sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
    sampler->setInputBounds(nhypers,VecLB.getDVector(),
                            VecUB.getDVector());
    sampler->setOutputParams(1);
    sampler->setSamplingParams(nSamp, 1, 0);
    sampler->initialize(0);
    VecIS.setLength(nSamp);
    VecXS.setLength(nSamp*nhypers);
    VecYS.setLength(nSamp);
    sampler->getSamples(nSamp,nhypers,1,VecXS.getDVector(),
                        VecYS.getDVector(), VecIS.getIVector());
    delete sampler;
    //for (ii = 0; ii < nhypers; ii++) VecXS[ii] = VecHypers_[ii];
 
#ifdef HAVE_LBFGS
    //**/ set up for LBFGS
    integer nInps, iprint=0, itask, *task=&itask, lsave[4], isave[44];
    integer *iwork, nCorr=5, *nbds, csave[60];
    int     nLBFGS=nSamp;
    double  factr, pgtol, dsave[29];

    nInps = nhypers;
    VecGrads.setLength(nInps);
    nbds = new integer[nInps];
    for (ii = 0; ii < nInps; ii++) nbds[ii] = 2;
    factr = 1e7;
    pgtol = 1e-8;
    kk = (2 * nCorr + 5) * nInps + 11 * nCorr * nCorr + 8 * nCorr;
    VecW.setLength(kk);
    iwork = new integer[3*nInps];
    its   = 0;
    VecHist.setLength(maxHist);

    if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
      printf("GP3 INFO: Performing %d hyperparameter searches.\n",
             nLBFGS);
    for (ii = 0; ii < nLBFGS; ii++)
    {
      //**/ on-the-fly modification of output print level
      fp = fopen("psuade_print", "r");
      if (fp != NULL)
      {
        fscanf(fp, "%d", &outputLevel_);
        fclose(fp);
        fp = NULL;
        if (outputLevel_ > 0 && outputLevel_ <= 5)
          printf("GP3: Output level set to %d.\n", outputLevel_);
        else
        {
          outputLevel_ = 4;
          printf("GP3: Output level set to %d.\n", outputLevel_);
        }
      }

      //**/ display initial information for current iteration
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
        printf("GP3 LBFGS #%d (%d)\n",ii+1, nLBFGS);
      for (jj = 0; jj < nInps; jj++) VecPVals[jj] = VecXS[ii*nInps+jj];
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
      {
        printf("   GP3 initial hyperparameter values:\n");
        for (jj = 0; jj < VecHypers_.length(); jj++) 
          printf("   Hyperparameter %2d = %e\n",jj+1,
                 VecPVals[jj]);
      }

      //**/ initialization for optimization
      nHist = 0;
      fail = 0;
      its = 0;
      *task = (integer) START;
      while (1 && its < 1000)
      {
        its++;
        setulb(&nInps,&nCorr,VecPVals.getDVector(),VecLB.getDVector(),
               VecUB.getDVector(),nbds,&FValue,VecGrads.getDVector(),
               &factr,&pgtol,VecW.getDVector(),iwork,task,&iprint,
               csave,lsave,isave,dsave);
        if (IS_FG(*task))
        {
          //if (outputLevel_ > 2) 
          //  for (kk = 0; kk < nInps; kk++)
          //    printf("     Hyperparameter %5d = %e\n",kk+1,VecPVals[kk]);

          FValue = computeGradients(VecPVals, VecGrads, status);
          if (FValue == 1e35) 
          {
            fail = 1;
            break;
          }
          dtemp = 0.0;
          for (kk = 0; kk < nInps; kk++)
            dtemp += pow(VecGrads[kk], 2.0);
          dtemp = sqrt(dtemp);

          if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
            printf("   LBFGS Current FValue = %e (its=%d, |grad|=%e)\n",
                   FValue,its,dtemp);
          if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
            for (kk = 0; kk < nInps; kk++)
              printf("         Gradient %5d = %e\n",kk+1,VecGrads[kk]);
          if (FValue < minVal && status == 0)
          {
            for (jj = 0; jj < nhypers; jj++) 
              VecHypers_[jj] = VecPVals[jj];
            minVal = FValue;
          }
          if (nHist < maxHist && FValue != 1e35) VecHist[nHist++] = FValue;
          else if (FValue != 1e35)
          {
            for (kk = 0; kk < maxHist-1; kk++) VecHist[kk] = VecHist[kk+1];
            VecHist[maxHist-1] = FValue;
          }
          if (nHist >= maxHist && FValue != 1e35)
          {
            dmean = 0.0;
            for (kk = 0; kk < maxHist; kk++) dmean += VecHist[kk];
            dmean /= (double) maxHist;
            dstd = 0.0;
            for (kk = 0; kk < maxHist; kk++) 
              dstd += pow(VecHist[kk]-dmean,2.0);
            dstd = sqrt(dstd/(maxHist-1.0));
            if (dmean == 0) dmean = 1;
            if (PABS(dstd/dmean) < 1e-3)
            {
              *task = STOP_ITER;
              if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
              {
                printf("INFO: PSUADE issues a stop (converged))\n");
                printf("INFO: Current iteration   = %d\n", its);
                printf("INFO: Current best FValue = %e (%e)\n",minVal,
                       PABS(dstd/dmean));
                printf("INFO: Current grad norm   = %e\n",dtemp);
              }
            }
            else
            {
              if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
                printf("INFO: Convergence check: %e < 1e-3?\n",
                       PABS(dstd/dmean));
            }
          }
          if (isave[33] >= 300) 
          {
            *task = STOP_ITER;
            if (outputLevel_ > 3) 
            {
              printf("INFO: PSUADE issues a stop (> 300 iterations)\n");
              printf("INFO: Current best FValue = %e\n", minVal);
            }
          }
        }
        else if (*task == NEW_X)
        {
          if (isave[33] >= 300) 
          {
            *task = STOP_ITER;
            if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
              printf("INFO: PSUADE issues a stop (> 300 iterations)\n");
          }
        }
        else
        {
          if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
          {
            printf("INFO: LBFGS issues a stop (its = %d)\n",its);
            printf("INFO: Current best FValue = %e\n", minVal);
          }
          break;
        }
      }
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
      {
        printf("   GP3 final hyperparameter values:\n");
        for (jj = 0; jj < VecHypers_.length(); jj++) 
          printf("   Hyperparameter %2d = %e\n",jj+1,
                 VecHypers_[jj]);
        printf("   Sample %5d: Fvalue (final) = %e\n", ii+1, minVal);
      }
      if (ii == (nLBFGS-1) && (fail == 1)) nLBFGS++;
    }
    delete [] iwork;
    delete [] nbds;
#endif

    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    {
      printf("GP3 very final hyperparameter values:\n");
      for (ii = 0; ii < VecHypers_.length(); ii++) 
        printf("   Hyperparameter %2d = %24.16e\n",ii+1,
               VecHypers_[ii]);
    }
  }
  else if (optimizeFlag == 2)
  {
    printf("GP3: You are choosing optimization option 2.\n");
    printf("     This isn't working as good as option 1.\n");
    nhypers = VecHypers_.length();
    VecPVals.setLength(nhypers);
    VecLB.setLength(nhypers);
    VecUB.setLength(nhypers);
    //**/ scale parameter ranges
    for (ii = 0; ii < nInputs_; ii++) 
    {
      VecLB[ii] = -4.0;
      VecUB[ii] =  4.0;
    }
    //**/ vertical magnitude
    VecLB[nInputs_] = -3.0;
    VecUB[nInputs_] = 10.0;
    //**/ nugget for the entire covariance matrix
    VecLB[nInputs_+1] = -28;
    VecUB[nInputs_+1] = - 3;
    //**/ Ymean 
    VecLB[nInputs_+2] = -0.5;
    VecUB[nInputs_+2] =  0.5;
    //**/ nugget for diagonal of the covariance matrix
    //**/ if nInputs == 2, nugget should be larger to
    //**/ avoid oscillations
    VecLB[nInputs_+3] = -24;
    if (nInputs_ <= 2) VecLB[nInputs_+3] = log(1e-6);
    VecUB[nInputs_+3] = - 2;
    count = nInputs_ + 4;
    if (optLinTerm_) 
    {
      for (ii = 0; ii < nInputs_; ii++) 
      {
        VecLB[count] = - 5*VecHypers_[nInputs_]/VecHypers_[ii];
        VecUB[count++] = + 5*VecHypers_[nInputs_]/VecHypers_[ii];
      }
    }
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    {
      for (ii = 0; ii < nhypers; ii++)
        printf("   Hyperparameter bounds %3d = %12.4e %12.4e\n",ii+1,
               VecLB[ii],VecUB[ii]);
    }
    nSamp = 5;
    if (nhypers > 51)
         sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
    else sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
    sampler->setInputBounds(nhypers,VecLB.getDVector(),VecUB.getDVector());
    sampler->setOutputParams(1);
    sampler->setSamplingParams(nSamp, 1, 0);
    sampler->initialize(0);
    VecIS.setLength(nSamp);
    VecXS.setLength(nSamp*nhypers);
    VecYS.setLength(nSamp);
    sampler->getSamples(nSamp,nhypers,1,VecXS.getDVector(),
                        VecYS.getDVector(), VecIS.getIVector());
    delete sampler;
    //**/ pre-set the first start
    for (ii = 0; ii < nhypers; ii++) VecXS[ii] = VecHypers_[ii];
 
    //**/ set up for least squares
    int      nUOA=nSamp, nPts, pLevel=5555, maxfun=1000;
    double   tol=1e-5, rhobeg, rhoend;

    nPts = (nhypers + 1) * (nhypers + 2) / 2;
    jj = (nPts+13) * (nPts+nhypers) + 3*nhypers*(nhypers+3)/2;
    kk = (nPts+5)*(nPts+nhypers)+3*nInputs_*(nhypers+5)/2+1;
    if (jj > kk) VecW.setLength(jj);
    else         VecW.setLength(kk);
    rhobeg = VecUB[0] - VecLB[0];
    for (ii = 1; ii < nhypers; ii++)
    {
      dtemp = VecUB[ii] - VecLB[ii];
      if (dtemp < rhobeg) rhobeg = dtemp;
    }
    rhobeg *= 0.5;
    rhoend = rhobeg * tol;

    its = 0;
    GP3_OptY = 1e35;
    if (GP3Obj != NULL) delete GP3Obj;
    GP3Obj = new GP3(nInputs_, nSamples_);
    GP3Obj->VecXDataN_ = VecXDataN_;
    GP3Obj->VecYDataN_ = VecYDataN_;
    GP3Obj->VecXDists_ = VecXDists_;

    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
      printf("GP3 NEWUOA optimization (%d):\n",nUOA);
    for (ii = 0; ii < nUOA; ii++)
    {
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 3 && 
          nUOA > 1 && ii > 0) 
        printf("GP3 WLS sample %d (%d)\n",ii+1, nUOA);
      for (jj = 0; jj < nhypers; jj++) VecPVals[jj] = VecXS[ii*nhypers+jj];
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
      {
        printf("   GP3 initial hyperparameter values:\n");
        for (jj = 0; jj < VecPVals.length(); jj++) 
          printf("   Hyperparameter %2d = %e\n",jj+1,VecPVals[jj]);
      }
#ifdef HAVE_NEWUOA
      newuoa_(&nhypers,&nPts,VecPVals.getDVector(),&rhobeg,&rhoend,&pLevel,
              &maxfun, VecW.getDVector());
#else
      printf("ERROR: NEWUOA optimizer unavailable.\n");
      exit(1);
#endif 
      VecHypers_ = GP3_OptX;
      //THIS NEEDS DEBUGGED
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
      {
        printf("   GP3 final hyperparameter values:\n");
        for (jj = 0; jj < VecPVals.length(); jj++) 
          printf("   Hyperparameter %2d = %e\n",jj+1,
                 VecPVals[jj]);
        printf("   Best objective function so far = %e\n",GP3_OptY);
      }
    }
    delete GP3Obj;
    GP3Obj = NULL;
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    {
      printf("GP3 very final hyperparameter values:\n");
      for (ii = 0; ii < VecHypers_.length(); ii++) 
        printf("   Hyperparameter %2d = %24.16e\n",ii+1,
               VecHypers_[ii]);
      printf("   Best objective function = %e\n",GP3_OptY);
    }
  }
  if (psConfig_.RSExpertModeIsOn())
  {
    FILE *fileOut = fopen(hyperParamFile_, "w");
    if (fileOut != NULL)
    {
      fprintf(fileOut, "%d\n", VecHypers_.length());
      for (ii = 0; ii < VecHypers_.length(); ii++) 
        fprintf(fileOut, "%24.16e\n", VecHypers_[ii]);
      fprintf(fileOut, "%d %d\n", nSamples_, nInputs_);
      fclose(fileOut);
      printf("GP3 INFO: hyperparameters have been stored in %s.\n",
             hyperParamFile_);
    }
  }

  //**/ =======================================================
  //**/ create and factorize the final K matrix
  //**/ =======================================================
  constructCMatrix(CMatrix_, VecHypers_);
  CMatrix_.LUDecompose();
  VecCInvY_.setLength(nSamples_);
  VecYT.setLength(nSamples_);
  for (jj = 0; jj < nSamples_; jj++) 
    VecYT[jj] = VecYDataN_[jj] - VecHypers_[nInputs_+2];
  CMatrix_.LUSolve(VecYT, VecCInvY_);

  //Cannot do because inverse seems to have problems 3/2017
  //double likelihood = 0.0;
  //for (jj = 0; jj < nSamples_; jj++) 
  //  likelihood += VecCInvY_[jj] * VecYT[jj];
  //likelihood *= 0.5;
  //for (jj = 0; jj < nSamples_; jj++) 
  //{
  //  dtemp = CMatrix.getEntry(jj, jj);
  //  likelihood += log(dtemp*sqrt2pi+1e-20);
  //}
  //if (outputLevel_ > 0) 
  //  printf("GP3 Best likelihood = %e\n", likelihood);
  if (psConfig_.GMModeIsOn())
  { 
    psMatrix CMatrix;
    constructCMatrix(CMatrix, VecHypers_);
    fp = fopen("Cmat.m", "w");
    fprintf(fp, "A = [\n");
    for (ii = 0; ii < nSamples_; ii++) 
    {
      for (jj = 0; jj < nSamples_; jj++) 
        fprintf(fp, "%16.8e ", CMatrix.getEntry(ii,jj));
      fprintf(fp, "\n");
    }
    fprintf(fp,"];\n");
    fclose(fp);
    printf("*** Final Cmatrix given in Cmat.m\n");
  }
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3 && optimizeFlag == 1)
  {
    printf("Optimization summary: \n");
    for (ii = 0; ii < nInputs_+4; ii++) 
      printf("Input %2d search bounds = %12.4e %12.4e\n",ii+1,
             VecXLows_[ii], VecXHighs_[ii]);
  }
  return 0;
}

// ************************************************************************
// compute log likelihood value (called directly from external)
// ------------------------------------------------------------------------
double GP3::computeLikelihood(psVector VecXD,psVector VecYN,psVector VecP)
{
  int    ii, jj, kk, status, count;
  double likelihood, sqrt2pi=sqrt(2*3.1415928), ddata, fudge, *arrayC;
  double dist;
  psMatrix MatC, MatCI;
  psVector VecY, VecCIY;

  //**/ =======================================================
  //**/ fill in the covariance matrix
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
  {
    printf("computeLikelihood:\n");
    for (jj = 0; jj < VecP.length(); jj++)
      printf("  hyperparameter %3d = %e\n", jj+1, VecP[jj]);
  }
  MatC.setDim(nSamples_, nSamples_);
  arrayC = MatC.getMatrix1D();
  count = 0;
  fudge = sqrt(nSamples_);
  for (jj = 0; jj < nSamples_; jj++)
  {
    arrayC[jj*nSamples_+jj] = exp(VecP[nInputs_]) +
                fudge*exp(VecP[nInputs_+1]) + exp(VecP[nInputs_+3]);
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs_; ii++)
        dist += pow(VecXD[count*nInputs_+ii],expPower_)/
                exp(expPower_*VecP[ii]);
      dist *= 0.5;
      dist = exp(VecP[nInputs_]) * exp(-dist);
      if (dist < 1.0e-50) dist = 0;
      arrayC[jj*nSamples_+kk] = dist + exp(VecP[nInputs_+1]);
      arrayC[kk*nSamples_+jj] = dist + exp(VecP[nInputs_+1]);
      count++;
    }
  }

  //**/ =======================================================
  //**/ Solve the linear system (VecCIY = inv(MatC) * VecY)
  //**/ =======================================================
  status = MatC.computeInverse(MatCI);
  if (status != 0)
  {
    printf("GP3 ERROR (1): failed in matrix inversion (%d).\n",status);
    if (psConfig_.InteractiveIsOn()) 
    {
      for (jj = 0; jj < VecP.length(); jj++)
        printf("  hyperparameter %3d = %e\n", jj+1, VecP[jj]);
    }
    return 1e35;
  }
  VecY.setLength(nSamples_);
  for (jj = 0; jj < nSamples_; jj++) 
    VecY[jj] = VecYN[jj] - VecP[nInputs_+2];
  VecCIY.setLength(nSamples_);
  MatCI.matvec(VecY, VecCIY, 0);

  //**/ =======================================================
  //**/ Solve the linear system (YT = inv(C) * YY)
  //**/ =======================================================
  int errCount=0;
  psMatrix MatEVecs;
  psVector VecEVals;
  MatC.eigenSolve(MatEVecs, VecEVals, 1);
  likelihood = 0.0;
  for (jj = 0; jj < nSamples_; jj++)
  {
    ddata = VecEVals[jj];
    if (ddata <= 0) errCount++;
    if (ddata != 0)
      likelihood += log(sqrt(PABS(ddata))*sqrt2pi+1e-50);
  }
  if (errCount > 0)
  {
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3)
      printf("GP3 WARNING: Covariance Matrix not positive definite\n");
  }
  for (jj = 0; jj < nSamples_; jj++)
    likelihood += 0.5 * (VecCIY[jj] * VecY[jj]);
  if (likelihood < 0) likelihood = - likelihood;
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    printf("   computeLikelihood: Likelihood = %e\n", likelihood);
  return likelihood;
}

// ************************************************************************
// compute gradients of log likelihood 
// ------------------------------------------------------------------------
double GP3::computeGradients(psVector VecHypers, psVector &VecGrads, 
                             int &retstat)
{
  int    jj, kk, ii, ll, status, count;
  double ddata, likelihood, sqrt2pi=sqrt(2*3.1415928), dist, fudge;
  psMatrix CMatrix, CInverse, CPrime, CT;
  psVector YVec, CInvY, YT;

  //**/ =======================================================
  //**/ fill in the C matrix
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
  {
    printf("computeGradients:\n");
    for (jj = 0; jj < VecHypers.length(); jj++)
      printf("  hyperparameter %3d = %e\n", jj+1, VecHypers[jj]);
  }
  constructCMatrix(CMatrix, VecHypers);

  //**/ =======================================================
  //**/ tracking input history 
  //**/ =======================================================
  if (VecXHighs_.length() == VecHypers.length())
  {
    for (ii = 0; ii < VecHypers.length(); ii++)
    {
      if (VecHypers[ii] > VecXHighs_[ii]) 
        VecXHighs_[ii] = VecHypers[ii]; 
      if (VecHypers[ii] < VecXLows_[ii])  
        VecXLows_[ii]  = VecHypers[ii]; 
    }
  }

  //**/ =======================================================
  //**/ Solve the linear system (CInvY = inv(C) * Y)
  //**/ =======================================================
  VecGrads.setLength(VecHypers.length());
  status = CMatrix.computeInverse(CInverse);
  if (status != 0)
  {
    printf("GP3 ERROR (1): failed in matrix inversion (%d).\n",status);
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    {
      for (jj = 0; jj < VecHypers.length(); jj++)
        printf("  hyperparameter %3d = %e\n", jj+1, VecHypers[jj]);
    }
    for (jj = 0; jj < VecHypers.length(); jj++) VecGrads[jj] = 100;
    return 1e35;
  }
  YVec.setLength(nSamples_);
  for (jj = 0; jj < nSamples_; jj++) 
    YVec[jj] = VecYDataN_[jj] - VecHypers[nInputs_+2];
  CInvY.setLength(nSamples_);
  CInverse.matvec(YVec, CInvY, 0);

  //**/ =======================================================
  //**/ compute negative log likelihood wrt length scales
  //**/ =======================================================
  YT.setLength(nSamples_);
  CPrime.setDim(nSamples_, nSamples_);
  double *TMat = CPrime.getMatrix1D();
  //**/ create derivative of C with respect to length scales
  for (ll = 0; ll < nInputs_; ll++)
  {
    //**/ create derivative of C with respect to length scales
    count = 0;
    for (jj = 0; jj < nSamples_; jj++)
    {
      TMat[jj*nSamples_+jj] = 0;
      for (kk = jj+1; kk < nSamples_; kk++)
      {
        dist = 0.0;
        for (ii = 0; ii < nInputs_; ii++)
          dist += pow(VecXDists_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecHypers[ii]);
        dist *= 0.5;
        dist = exp(VecHypers[nInputs_]) * exp(-dist);
        ddata = pow(VecXDists_[count*nInputs_+ll],expPower_)/
                exp((1.0+expPower_)*VecHypers[ll])*0.5*expPower_;
        dist *= ddata;
        TMat[jj*nSamples_+kk] = dist;
        TMat[kk*nSamples_+jj] = dist;
        count++;
      }
    }
    //**/ compute: C' * C^{-1} (Y - mu)
    CPrime.matvec(CInvY, YT, 0);
    //**/ now compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
    ddata = 0.0;
    for (jj = 0; jj < nSamples_; jj++) ddata += YT[jj] * CInvY[jj];
    VecGrads[ll] = - ddata;
    ddata = 0.0;
    //**/ now compute trace(C^{-1} C')
    for (ii = 0; ii < nSamples_; ii++)
      for (jj = 0; jj < nSamples_; jj++) 
        ddata += CPrime.getEntry(ii, jj) * CInverse.getEntry(ii, jj);
    //**/ now: trace(C^{-1} C') - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
    VecGrads[ll] += ddata;
    //**/ finally: 0.5[tr(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)]
    VecGrads[ll] *= 0.5;
  }

  //**/ =======================================================
  //**/ compute negative log likelihood wrt theta_1 
  //**/ =======================================================
  //**/ create derivative of C with respect to theta_1
  count = 0;
  for (jj = 0; jj < nSamples_; jj++)
  {
    TMat[jj*nSamples_+jj] = 1;
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs_; ii++)
        dist += pow(VecXDists_[count*nInputs_+ii],expPower_)/
                exp(expPower_*VecHypers[ii]);
      dist = exp(-0.5 * dist);
      if (dist < 1.0e-50) dist = 0;
      TMat[jj*nSamples_+kk] = dist;
      TMat[kk*nSamples_+jj] = dist;
      count++;
    }
  }
  //**/ compute: C' * C^{-1} (Y - mu)
  CPrime.matvec(CInvY, YT, 0);
  //**/ compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  ddata = 0.0;
  for (jj = 0; jj < nSamples_; jj++) ddata += YT[jj] * CInvY[jj];
  VecGrads[nInputs_] = - ddata;
  //**/ compute: trace(C^{-1} C') - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  ddata = 0.0;
  for (ii = 0; ii < nSamples_; ii++) 
    for (jj = 0; jj < nSamples_; jj++) 
      ddata += CInverse.getEntry(ii, jj) * CPrime.getEntry(ii, jj);
  //**/ compute: [tr(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)]
  VecGrads[nInputs_] += ddata;
  //**/ compute: 0.5*exp(theta_1)*[tr(C^{-1}C')-(Y-mu)'C^{-1}C'*C^{-1}(Y-mu)]
  VecGrads[nInputs_] *= 0.5 * exp(VecHypers[nInputs_]);

  //**/ =======================================================
  //**/ compute negative log likelihood wrt theta_2 (offset to C)
  //**/ =======================================================
  //**/ compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  //**/          when C' is a all-1 matrix
  fudge = sqrt(1.0 * nSamples_);
  ddata = 0.0;
  for (jj = 0; jj < nSamples_; jj++) ddata += CInvY[jj];
  VecGrads[nInputs_+1] = - ddata * ddata - (fudge - 1.0) * ddata;
  //**/ compute: trace(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)
  ddata = 0.0;
  for (ii = 0; ii < nSamples_; ii++)
    for (jj = 0; jj < nSamples_; jj++) 
      ddata += CInverse.getEntry(ii, jj);
  VecGrads[nInputs_+1] += ddata;
  //**/ 0.5*exp(theta_2)*[tr(C^{-1}C')-(Y-mu)'C^{-1}C'*C^{-1}(Y-mu)]
  ddata = exp(VecHypers[nInputs_+1]);
  VecGrads[nInputs_+1] *= 0.5 * ddata;

  //**/ =======================================================
  //**/ compute negative log likelihood with respect to mean
  //**/ =======================================================
  //*/ compute: - 1' * C' * C^{-1} (Y - mu)
  ddata = 0.0;
  for (jj = 0; jj < nSamples_; jj++) ddata += CInvY[jj];
  VecGrads[nInputs_+2] = - ddata;

  //**/ =======================================================
  //**/ compute negative log likelihood wrt theta_3 
  //**/ =======================================================
  //**/ compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  //  //**/          since CPrime is a diagonal matrix of [dist]
  ddata = 0.0;
  for (jj = 0; jj < nSamples_; jj++) ddata += CInvY[jj] * CInvY[jj];
  VecGrads[nInputs_+3] = - ddata;
  //**/ compute: trace(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)
  ddata = 0.0;
  for (ii = 0; ii < nSamples_; ii++) 
    ddata += CInverse.getEntry(ii, ii);
  VecGrads[nInputs_+3] += ddata;
  //**/ 0.5*exp(theta_3)*[tr(C^{-1}C')-(Y-mu)'C^{-1}C'*C^{-1}(Y-mu)]
  VecGrads[nInputs_+3] *= 0.5 * exp(VecHypers[nInputs_+3]);

  //**/ =======================================================
  //**/ compute negative log likelihood 
  //**/ use eigenSolve to detect non-positive definiteness of C
  //**/ =======================================================
  int errCount=0;
  CMatrix.eigenSolve(CT, YT, 1);
  likelihood = 0.0;
  for (jj = 0; jj < nSamples_; jj++) 
  {
    ddata = YT[jj];
    if (ddata <= 0) errCount++;
    if (ddata != 0)
      likelihood += log(sqrt(PABS(ddata))*sqrt2pi+1e-50);
  }
  retstat = 0;
  if (errCount > 0)
  {
    retstat = 1;
    if (outputLevel_ > 3)
    {
      printf("GP3 WARNING: CMatrix non-positive definite (%d)\n",
             errCount);
      for (jj = 0; jj < VecHypers.length(); jj++)
        printf("   hyperparameter %2d = %e\n",jj+1,VecHypers[jj]);
    }
    if (psConfig_.GMModeIsOn())
    { 
      printf("INFO: non-PD covariance matrix, terminate in GM mode.\n");
      printf("      diagnostics information is stored in errmat.m\n");
      FILE *fp = fopen("errmat.m", "w");
      fprintf(fp, "n = %d;\n", nSamples_);
      fprintf(fp, "f = %e;\n", sqrt(1.0*nSamples_));
      fprintf(fp, "A = [\n");
      for (ii = 0; ii < nSamples_; ii++) 
      {
        for (jj = 0; jj < nSamples_; jj++) 
          fprintf(fp, "%16.8e ", CMatrix.getEntry(ii,jj));
        fprintf(fp, "\n");
      }
      fprintf(fp, "];\n");
      fprintf(fp, "L = [\n");
      for (ii = 0; ii < nSamples_*(nSamples_-1)/2*nInputs_; ii++) 
        fprintf(fp, "%24.16e\n", VecXDists_[ii]);
      fprintf(fp, "];\n");
      fprintf(fp, "H = [\n");
      for (jj = 0; jj < nInputs_; jj++)
        fprintf(fp, "%e\n", exp(2.0*VecHypers[jj]));
      fprintf(fp, "%e\n", exp(VecHypers[nInputs_]));
      fprintf(fp, "%e\n", exp(VecHypers[nInputs_+1]));
      fprintf(fp, "%e\n", VecHypers[nInputs_+2]);
      fprintf(fp, "%e\n", exp(VecHypers[nInputs_+3]));
      fprintf(fp, "];\n");
      fprintf(fp, "count = 0;\n");
      fprintf(fp, "for ii = 1 : n\n");
      fprintf(fp, "  B(ii,ii) = H(%d) + ll*H(%d)+H(%d);\n",nInputs_+1,
              nInputs_+2,nInputs_+4);
      fprintf(fp, "  for jj = ii+1 : n \n");
      fprintf(fp, "    dist = 0;\n");
      fprintf(fp, "    for kk = 1 : %d\n",nInputs_);
      fprintf(fp, "      dist = dist + L(count*3+kk)^2 / H(kk);\n");
      fprintf(fp, "    end;\n");
      fprintf(fp, "    ddata = H(%d) * exp(-0.5*dist);\n",nInputs_+1);
      fprintf(fp, "    B(ii,jj) = ddata + H(%d);\n",nInputs_+2);
      fprintf(fp, "    B(jj,ii) = ddata + H(%d);\n",nInputs_+2);
      fprintf(fp, "    count = count + 1;\n");
      fprintf(fp, "  end;\n");
      fprintf(fp, "end;\n");
      fprintf(fp, "Bmin = min(eig(B))\n");
      fclose(fp);
      exit(1);
    }
  }
  for (jj = 0; jj < nSamples_; jj++) 
    likelihood += 0.5 * (CInvY[jj] * YVec[jj]);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    printf("   computeGradients: Likelihood = %e\n", likelihood);
  return likelihood;
}

// ************************************************************************
// compute pairwise distances 
// ------------------------------------------------------------------------
int GP3::computeDistances()
{
  int    ii, jj, kk, count, error=0;
  double *LDists, dist;
  psVector VecLDists;

  VecLDists.setLength((nSamples_*(nSamples_-1)/2)*nInputs_);
  LDists = VecLDists.getDVector();
  count = 0;
  for (jj = 0; jj < nSamples_; jj++)
  {
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs_; ii++)
      {
        LDists[count*nInputs_+ii] = VecXDataN_[jj*nInputs_+ii] -
                                    VecXDataN_[kk*nInputs_+ii];
        if (LDists[count*nInputs_+ii] < 0)
           LDists[count*nInputs_+ii] = - LDists[count*nInputs_+ii];
        dist += pow(LDists[count*nInputs_+ii], 2.0);
      }
      if (dist == 0.0)
      {
        printf("GP3 ERROR: repeated sample points.\n");
        printf("           Prune repeated points and re-run.\n");
        printf("Sample %d : (with sample point %d)\n", kk+1, jj+1);
        for (ii = 0; ii < nInputs_; ii++)
          printf("   Input %d : %e\n",ii+1,
                 VecXDataN_[kk*nInputs_+ii]*VecXStds_[ii] +VecXMeans_[ii]);
        error = 1;
      }
      count++;
    }
  }
  VecXDists_.load(count*nInputs_, LDists);
  return error;
}

// ************************************************************************
// interpolation 
// ------------------------------------------------------------------------
int GP3::interpolate(int npts, double *XX, double *Y, double *Ystds)
{
  int    ii, kk, nn, iOne=1, status;
  double dist, expn, ddata;
  psVector YT, YT2;

  YT.setLength(nSamples_);
  YT2.setLength(nSamples_);
  Y[0] = 0;
  for (nn = 0; nn < npts; nn++)
  {
    for (kk = 0; kk < nSamples_; kk++)
    {
      expn = 0.0;
      for (ii = 0; ii < nInputs_; ii++)
      {
        dist = VecXDataN_[kk*nInputs_+ii] - XX[nn*nInputs_+ii];
        if (dist < 0) dist = - dist;
        expn += pow(dist,expPower_)/exp(expPower_*VecHypers_[ii]);
      }
      expn *= 0.5;
      YT[kk] = exp(VecHypers_[nInputs_]) * exp(-expn);
    }
    Y[nn] = VecHypers_[nInputs_+2];
    for (kk = 0; kk < nSamples_; kk++)
      Y[nn] += YT[kk] * VecCInvY_[kk];
    if (Ystds != NULL)
    {
      Ystds[nn] = 0.0;
      ddata = exp(VecHypers_[nInputs_]) + 
              exp(VecHypers_[nInputs_+1]) +
              exp(VecHypers_[nInputs_+3]);
      CMatrix_.LUSolve(YT, YT2);
      for (kk = 0; kk < nSamples_; kk++)
        ddata -= YT[kk] * YT2[kk];
      //**/ a kludge for now to handle negative variance
      if (ddata < 0) ddata = - ddata;
      Ystds[nn] = ddata;
    }
  }
  return 0;
}

// ************************************************************************
// construct C matrix
// ------------------------------------------------------------------------
void GP3::constructCMatrix(psMatrix &CMatrix, psVector VecP)
{
  int    ii, jj, kk, count;
  double dist, ddata, fudge;

  //**/ =======================================================
  //**/ fill in the K matrix
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
  {
    printf("Construct CMatrix:\n");
    for (jj = 0; jj < VecP.length(); jj++)
      printf("  hyperparameter %3d = %e\n", jj+1, VecP[jj]);
  }
  CMatrix.setDim(nSamples_, nSamples_);
  double *TMat = CMatrix.getMatrix1D();
  //**/ This fudge is used to avoid matrix indefiniteness
  //**/ Probably not very helpful if theta_1 is huge
  fudge = sqrt(nSamples_);
  count = 0;
  for (jj = 0; jj < nSamples_; jj++)
  {
    TMat[jj*nSamples_+jj] = exp(VecP[nInputs_]) + 
                fudge*exp(VecP[nInputs_+1]) + exp(VecP[nInputs_+3]);
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs_; ii++)
        dist += pow(VecXDists_[count*nInputs_+ii],expPower_)/
                exp(expPower_*VecP[ii]);
      dist *= 0.5;
      dist = exp(VecP[nInputs_]) * exp(-dist);
      if (dist < 1.0e-50) dist = 0;
      TMat[jj*nSamples_+kk] = dist + exp(VecP[nInputs_+1]);
      TMat[kk*nSamples_+jj] = dist + exp(VecP[nInputs_+1]);
      count++;
    }
  }
  return;
}

// ************************************************************************
// save context 
// ------------------------------------------------------------------------
void GP3::saveFA()
{
  int ii, jj;
  FILE *fp = fopen(".psuadeGP3", "w");
  if (fp == NULL) return;
  fprintf(fp, "%d %d\n", nSamples_, nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    fprintf(fp, "%24.16e %24.16e\n", VecXMeans_[ii], VecXStds_[ii]);
  fprintf(fp, "%24.16e %24.16e\n", YMean_, YStd_);
  fprintf(fp, "%d\n", VecHypers_.length());
  for (ii = 0; ii < VecHypers_.length(); ii++)
    fprintf(fp, "%24.16e \n", VecHypers_[ii]);
  for (jj = 0; jj < nSamples_; jj++)
  {
    for (ii = 0; ii < nInputs_; ii++)
      fprintf(fp, "%24.16e ", VecXDataN_[jj*nInputs_+ii]);
    fprintf(fp, "\n");
  }
  for (jj = 0; jj < nSamples_; jj++)
    fprintf(fp, "%24.16e\n", VecCInvY_[jj]);
  for (jj = 0; jj < nSamples_; jj++)
  {
    for (ii = 0; ii < nSamples_; ii++)
      fprintf(fp, "%24.16e ", CMatrix_.getEntry(jj,ii));
    fprintf(fp, "\n");
  }
  for (jj = 0; jj < nSamples_; jj++)
    fprintf(fp, "%d\n", CMatrix_.pivots_[jj]);
  fprintf(fp, "%24.16e\n", expPower_);
  fclose(fp);
}

// ************************************************************************
// retrieve context 
// ------------------------------------------------------------------------
void GP3::retrieveFA()
{
  int    ii, jj;
  double ddata1, ddata2;

  FILE *fp = fopen(".psuadeGP3", "r");
  if (fp == NULL) 
  {
    printf("GP3 ERROR: .psuadeGP3 file not found.\n");
    return;
  }
  fscanf(fp, "%d %d", &nSamples_, &nInputs_);
  VecXMeans_.setLength(nInputs_);
  VecXStds_.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
  {
    fscanf(fp, "%lg %lg", &ddata1, &ddata2);
    VecXMeans_[ii] = ddata1;
    VecXStds_[ii] = ddata2;
  }
  fscanf(fp, "%lg %lg", &YMean_, &YStd_);
  fscanf(fp, "%d", &ii);
  VecHypers_.setLength(ii);
  for (ii = 0; ii < VecHypers_.length(); ii++)
  {
    fscanf(fp, "%lg", &ddata1);
    VecHypers_[ii] = ddata1;
  }
  VecXDataN_.setLength(nSamples_*nInputs_);
  for (jj = 0; jj < nSamples_; jj++)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      fscanf(fp, "%lg", &ddata1);
      VecXDataN_[jj*nInputs_+ii] = ddata1;
    }
  }
  VecCInvY_.setLength(nSamples_);
  for (jj = 0; jj < nSamples_; jj++)
  {
    fscanf(fp, "%lg", &ddata1);
    VecCInvY_[jj] = ddata1;
  }
  CMatrix_.setDim(nSamples_, nSamples_);
  for (jj = 0; jj < nSamples_; jj++)
  {
    for (ii = 0; ii < nSamples_; ii++)
    {
      fscanf(fp, "%lg", &ddata1);
      CMatrix_.setEntry(jj,ii,ddata1);
    }
  }
  psIVector vecIT;
  vecIT.setLength(nSamples_); 
  CMatrix_.pivots_ = vecIT.getIVector();
  for (jj = 0; jj < nSamples_; jj++)
  {
    fscanf(fp, "%d", &ii);
    CMatrix_.pivots_[jj] = ii;
  }
  fscanf(fp, "%lg", &expPower_);
  fclose(fp);
}

// ************************************************************************
// generate C code
// ------------------------------------------------------------------------
void GP3::genRSCode()
{
  int ii, jj;
  FILE *fp = fopen("psuade_rs.info", "w");
  if (fp == NULL) return;
  fprintf(fp,"This file contains information to re-construct GP\n");
  fprintf(fp,"response surface offline. Follow the steps below:\n");
  fprintf(fp,"1. Search for the keywords 'SPLIT HERE' in this file.\n");
  fprintf(fp,"2. Store the lines below keywords into main.c\n");
  fprintf(fp,"3. Add the to-be-evaluated points inside main() in main.c\n");
  fprintf(fp,"   (or, embed all except main() in your own program).\n");
  fprintf(fp,"4. Compile main.c (cc -o main main.c -lm) and run\n");
  fprintf(fp,"   (Do not change the psuade_rs.info file.\n");
  fprintf(fp,"Note: if std dev is desired, uncomment the corresponding\n");
  fprintf(fp,"      section and compile with: cc main.c -llapack -lm\n");
  fprintf(fp,"\n");
  fprintf(fp,"PSUADE_BEGIN\n");
  fprintf(fp, "%d %d\n", nSamples_, nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    fprintf(fp, "%24.16e %24.16e\n", VecXMeans_[ii], VecXStds_[ii]);
  fprintf(fp, "%24.16e %24.16e\n", YMean_, YStd_);
  fprintf(fp, "%d\n", VecHypers_.length());
  for (jj = 0; jj < VecHypers_.length(); jj++)
    fprintf(fp, "%24.16e \n", VecHypers_[jj]);
  for (jj = 0; jj < nSamples_; jj++)
  {
    for (ii = 0; ii < nInputs_; ii++)
      fprintf(fp, "%24.16e ", VecXDataN_[jj*nInputs_+ii]);
    fprintf(fp, "\n");
  }
  for (jj = 0; jj < nSamples_; jj++)
    fprintf(fp, "%24.16e\n", VecCInvY_[jj]);
  for (jj = 0; jj < nSamples_; jj++)
  {
    for (ii = 0; ii < nSamples_; ii++)
      fprintf(fp, "%24.16e ", CMatrix_.getEntry(jj,ii));
    fprintf(fp, "\n");
  }
  for (jj = 0; jj < nSamples_; jj++)
    fprintf(fp, "%d\n", CMatrix_.pivots_[jj]);
  fprintf(fp,"==================== SPLIT HERE =====================\n");
  fprintf(fp,"/* ***********************************************/ \n");
  fprintf(fp,"/* GP interpolator from PSUADE.                  */ \n");
  fprintf(fp,"/* To estimate prediction uncertainty uncomment, */ \n");
  fprintf(fp,"/* dgetrs and the corresponding code segment.    */ \n");
  fprintf(fp,"/* ==============================================*/ \n");
  fprintf(fp,"#include <math.h>\n");
  fprintf(fp,"#include <stdlib.h>\n");
  fprintf(fp,"#include <stdio.h>\n");
  fprintf(fp,"#include <string.h>\n");
  fprintf(fp,"/*void dgetrs_(char*,int*,int*,double*,int*,int*,\n");
  fprintf(fp,"               double*,int*,int*);*/\n");
  fprintf(fp,"int initialize();\n");
  fprintf(fp,"int finalize();\n");
  fprintf(fp,"int interpolate(int,double*,double*,double*);\n");
  fprintf(fp,"main(int argc, char **argv)\n{\n");
  fprintf(fp,"  int    i, iOne=1, nInps;\n");
  fprintf(fp,"  double X[%d], Y, Std;\n",nInputs_);
  fprintf(fp,"  FILE   *fIn=NULL, *fOut=NULL;\n");
  fprintf(fp,"  if (argc < 2) \n  {\n");
  fprintf(fp,"    printf(\"ERROR: not enough argument.\\n\");\n");
  fprintf(fp,"    exit(1);\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  fIn = fopen(argv[1], \"r\");\n");
  fprintf(fp,"  if (fIn == NULL)\n  {\n");
  fprintf(fp,"    printf(\"ERROR: cannot open input file.\\n\");\n");
  fprintf(fp,"    exit(1);\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  fscanf(fIn, \"%%d\", &nInps);\n");
  fprintf(fp,"  if (nInps != %d)\n  {\n", nInputs_);
  fprintf(fp,"    printf(\"ERROR - wrong nInputs.\\n\");\n");
  fprintf(fp,"    exit(1);\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  for (i=0; i<%d; i++) fscanf(fIn, \"%%lg\", &X[i]);\n",
          nInputs_);
  fprintf(fp,"  fclose(fIn);\n");
  fprintf(fp,"  initialize();\n");
  fprintf(fp,"  interpolate(iOne, X, &Y, &Std);\n");
  fprintf(fp,"  printf(\"Y = %%e (stdev = %%e)\\n\", Y, Std);\n");
  fprintf(fp,"  finalize();\n");
  fprintf(fp,"  if (argc >= 3) \n  {\n");
  fprintf(fp,"    fOut = fopen(argv[2], \"w\");\n");
  fprintf(fp,"    if (fOut == NULL) {\n");
  fprintf(fp,"      printf(\"ERROR: cannot open output file.\\n\");\n");
  fprintf(fp,"      exit(1);\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    fprintf(fOut,\" %%e\\n\", Y);\n");
  fprintf(fp,"    fclose(fOut);\n  }\n");
  fprintf(fp,"}\n");
  fprintf(fp,"/* ==========================================*/\n");
  fprintf(fp,"/* Regression interpolation function         */\n");
  fprintf(fp,"/* X[0], X[1],   .. X[m-1]   - first point\n");
  fprintf(fp," * X[m], X[m+1], .. X[2*m-1] - second point\n");
  fprintf(fp," * ... */\n");
  fprintf(fp,"/* ==========================================*/\n");
  fprintf(fp,"int    nInps, nSamples, nParams, *pivs=NULL;\n");
  fprintf(fp,"double *XMeans=NULL,*XStds=NULL,YMean,YStd,*Thetas=NULL;\n");
  fprintf(fp,"double *CMat=NULL,*CInvY=NULL, *XN=NULL; \n");
  fprintf(fp,"int interpolate(int npts,double *X,double *Y,");
  fprintf(fp,"double *YStds) \n{\n");
  fprintf(fp,"  int    ss, ii, jj, kk;\n");
  fprintf(fp,"  double expn, *xt, *yt, *zt, dist, ddata;\n");
  fprintf(fp,"  xt = (double *) malloc(nInps*sizeof(double));\n");
  fprintf(fp,"  yt = (double *) malloc(nSamples*sizeof(double));\n");
  fprintf(fp,"  zt = (double *) malloc(nSamples*sizeof(double));\n");
  fprintf(fp,"  for (ss = 0; ss < npts; ss++)\n  {\n");
  fprintf(fp,"    for (ii = 0; ii < nInps; ii++)\n");
  fprintf(fp,"      xt[ii] = (X[ss*nInps+ii]-XMeans[ii])/XStds[ii];\n");
  fprintf(fp,"    for (kk = 0; kk < nSamples; kk++)\n    {\n");
  fprintf(fp,"      expn = 0.0;\n");
  fprintf(fp,"      for (ii = 0; ii < nInps; ii++)\n      {\n");
  fprintf(fp,"        dist = XN[kk*nInps+ii] - xt[ii];\n");
  fprintf(fp,"        expn += pow(dist,%e)/exp(%e*Thetas[ii]);\n      }\n",
          expPower_,expPower_);
  fprintf(fp,"      expn *= 0.5;\n");
  fprintf(fp,"      yt[kk] = exp(Thetas[nInps]) * exp(-expn);\n    }\n");
  fprintf(fp,"    Y[ss] = Thetas[nInps+2];\n");
  fprintf(fp,"    for (kk = 0; kk < nSamples; kk++)\n");
  fprintf(fp,"      Y[ss] += yt[kk] * CInvY[kk];\n");
  fprintf(fp,"    Y[ss] = Y[ss] * YStd + YMean;\n");
  fprintf(fp,"    YStds[ss] = 0.0;\n");
  fprintf(fp,"    /* ==== if need to compute std dev. ===== \n");
  fprintf(fp,"    ddata = exp(Thetas[nInps])+exp(Thetas[nInps+1])+");
  fprintf(fp,"exp(Thetas[nInps+3]);\n");
  fprintf(fp,"    LUSolve(yt, zt);\n");
  fprintf(fp,"    for (kk = 0; kk < nSamples; kk++)\n");
  fprintf(fp,"      ddata -= yt[kk] * zt[kk];\n");
  fprintf(fp,"    if (ddata < 0) ddata = - ddata;\n");
  fprintf(fp,"    YStds[ss] = sqrt(ddata);\n");
  fprintf(fp,"    */\n  }\n");
  fprintf(fp,"  free(xt); free(yt); free(zt);\n}\n");
  fprintf(fp,"int initialize()\n{\n");
  fprintf(fp,"  int    ii, jj;\n");
  fprintf(fp,"  double ddata;\n");
  fprintf(fp,"  char   line[1001], word[1001];\n");
  fprintf(fp,"  FILE *fp = fopen(\"psuade_rs.info\", \"r\");\n");
  fprintf(fp,"  if (fp == NULL)\n  {\n");
  fprintf(fp,"    printf(\"Data file (psuade_rs.info) not found.\\n\");\n");
  fprintf(fp,"    exit(1);\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  while (1) \n  {\n");
  fprintf(fp,"    fgets(line, 500, fp);\n");
  fprintf(fp,"    sscanf(line, \"%%s\",word);\n");
  fprintf(fp,"    if (!strcmp(word, \"PSUADE_BEGIN\")) break;\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  fscanf(fp, \"%%d %%d\", &nSamples, &nInps);\n");
  fprintf(fp,"  XMeans = (double *) malloc(nInps*sizeof(double));\n");
  fprintf(fp,"  XStds  = (double *) malloc(nInps*sizeof(double));\n");
  fprintf(fp,"  for (ii = 0; ii < nInps; ii++) \n  {\n");
  fprintf(fp,"    fscanf(fp, \"%%lg\", &ddata);\n");
  fprintf(fp,"    XMeans[ii] = ddata;\n");
  fprintf(fp,"    fscanf(fp, \"%%lg\", &ddata);\n");
  fprintf(fp,"    XStds[ii] = ddata;\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  fscanf(fp, \"%%lg\", &YMean);\n");
  fprintf(fp,"  fscanf(fp, \"%%lg\", &YStd);\n");
  fprintf(fp,"  fscanf(fp, \"%%d\", &nParams);\n");
  fprintf(fp,"  Thetas = (double *) malloc(nParams*sizeof(double));\n");
  fprintf(fp,"  for (ii = 0; ii < nParams; ii++) \n  {\n");
  fprintf(fp,"    fscanf(fp, \"%%lg\", &ddata);\n");
  fprintf(fp,"    Thetas[ii] = ddata;\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  XN = (double *) malloc(nSamples*nInps*sizeof(double));\n");
  fprintf(fp,"  for (jj = 0; jj < nSamples; jj++) \n");
  fprintf(fp,"    for (ii = 0; ii < nInps; ii++) \n");
  fprintf(fp,"      fscanf(fp, \"%%lg\", &XN[jj*nInps+ii]);\n");
  fprintf(fp,"  CInvY = (double *) malloc(nSamples*sizeof(double));\n");
  fprintf(fp,"  for (ii = 0; ii < nSamples; ii++)\n");
  fprintf(fp,"    fscanf(fp, \"%%lg\", &CInvY[ii]);\n");
  fprintf(fp,"  CMat=(double*) malloc(nSamples*nSamples*sizeof(double));\n");
  fprintf(fp,"  for (jj = 0; jj < nSamples; jj++) \n");
  fprintf(fp,"    for (ii = 0; ii < nSamples; ii++)\n");
  fprintf(fp,"      fscanf(fp, \"%%lg \", &CMat[jj+ii*nSamples]);\n");
  fprintf(fp,"  pivs = (int *) malloc(nSamples*sizeof(int));\n");
  fprintf(fp,"  for (ii = 0; ii < nSamples; ii++)\n");
  fprintf(fp,"    fscanf(fp, \"%%d\", &pivs[ii]);\n");
  fprintf(fp,"}\n");
  fprintf(fp,"/* ==========================================*/\n");
  fprintf(fp,"int finalize()\n{\n");
  fprintf(fp,"  if (XMeans != NULL) free(XMeans);\n");
  fprintf(fp,"  if (XStds  != NULL) free(XStds);\n");
  fprintf(fp,"  if (Thetas != NULL) free(Thetas);\n");
  fprintf(fp,"  if (CInvY  != NULL) free(CInvY);\n");
  fprintf(fp,"  if (CMat   != NULL) free(CMat);\n");
  fprintf(fp,"  if (XN     != NULL) free(XN);\n");
  fprintf(fp,"  if (pivs   != NULL) free(pivs);\n");
  fprintf(fp,"}\n");
  fclose(fp);
  printf("FILE psuade_rs.info contains the GP interpolator.\n");
}

// ************************************************************************
// get hyperparameters 
// ------------------------------------------------------------------------
void GP3::getHyperparameters(psVector &inhyper)
{
  inhyper = VecHypers_;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double GP3::setParams(int targc, char **targv)
{
  int    ii;
  double mmax, range;
  char   pString[101];
  FILE   *fp=NULL;

  if (targc > 0 && !strcmp(targv[0], "rank"))
  {
    psVector VecLScales;
    VecLScales.setLength(nInputs_);
    mmax = 0.0;
    for (ii = 0; ii < nInputs_; ii++)
    {
      VecLScales[ii] = 1.0 / VecHypers_[ii];
      if (VecXMeans_[ii] == 0 && VecXStds_[ii] == 1)
      {
        range = VecUBs_[ii] - VecLBs_[ii];
        VecLScales[ii] *= range;
      }
      if (VecLScales[ii] > mmax) mmax = VecLScales[ii];
    }
    for (ii = 0; ii < nInputs_; ii++)
      VecLScales[ii] = VecLScales[ii] / mmax * 100.0;

    if (plotScilab())
         fp = fopen("scilabgp3sa.sci", "w");
    else fp = fopen("matlabgp3sa.m", "w");
    if (fp == NULL)
    {
      printf("GP3 ERROR: something wrong with opening a write file.\n");
    }
    else
    {
      fprintf(fp, "n = %d;\n", nInputs_);
      fprintf(fp, "Y = [\n");
      for (ii = 0; ii < nInputs_; ii++)
        fprintf(fp, "%24.16e \n", PABS(VecLScales[ii]));
      fprintf(fp, "]; \n");
      fprintf(fp, "ymax = max(Y);\n");
      fprintf(fp, "ymin = 0;\n");
      fprintf(fp, "if (ymax == ymin)\n");
      fprintf(fp, "   ymax = ymax * 0.1;\n");
      fprintf(fp, "end;\n");
      fwritePlotCLF(fp);
      fprintf(fp, "bar(Y,0.8);\n");
      fwritePlotAxes(fp);
      snprintf(pString,100,"GP3 Ranking");
      fwritePlotTitle(fp, pString);
      snprintf(pString,100,"Input Numbers");
      fwritePlotXLabel(fp, pString);
      snprintf(pString,100,"GP3 Measure");
      fwritePlotYLabel(fp, pString);
      if (plotScilab())
      {
        fprintf(fp,"a=gca();\n");
        fprintf(fp,"a.data_bounds=[0,ymin; n+1,ymax+0.01*(ymax-ymin)];\n");
      }
      else
      {
        fprintf(fp,"axis([0 n+1 ymin ymax+0.01*(ymax-ymin)])\n");
      }
      fclose(fp);
      if (plotScilab())
           printf("GP3 ranking in file scilabgp3sa.sci\n");
      else printf("GP3 ranking in file matlabgp3sa.m\n");
    }
    psIVector VecIA;
    VecIA.setLength(nInputs_);
    for (ii = 0; ii < nInputs_; ii++) VecIA[ii] = ii;
    sortDbleList2a(nInputs_, VecLScales.getDVector(), VecIA.getIVector());
    printAsterisks(PL_INFO, 0);
    printf("* GP3 screening rankings\n");
    printAsterisks(PL_INFO, 0);
    for (ii = nInputs_-1; ii >= 0; ii--)
      printf("*  Rank %3d : Input = %4d (score = %5.1f) (ref = %e)\n",
             nInputs_-ii, VecIA[ii]+1, VecLScales[ii], 
             0.01*VecLScales[ii]*mmax);
    printAsterisks(PL_INFO, 0);
  }
  else if (targc == 2 && !strcmp(targv[0], "hyperparam_file"))
  {
    strcpy(hyperParamFile_, targv[1]);  
    if (psConfig_.InteractiveIsOn())
      printf("GP3 INFO: Trained hyperparameters are to be stored in %s\n",
             hyperParamFile_);
  }
  else if (targc == 2 && !strcmp(targv[0], "GP_power"))
  {
    expPower_ = *(double *) targv[1];
    printf("GP3 INFO: exponential degree = %e\n",expPower_);
  }
  else if (targc == 2 && !strcmp(targv[0], "GP_nstarts"))
  {
    nStarts_ = *(int *) targv[1];
    printf("GP3 INFO: number of optimization starts = %d\n",nStarts_);
  }
  return 0.0;
}

