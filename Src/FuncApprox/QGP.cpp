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
// Functions for the class quantile Homogeneous GP
// AUTHOR : CHARLES TONG
// DATE   : 2019
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

#include "QGP.h"
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
#ifdef HAVE_LBFGS
#include "../../External/L-BFGS-B-C/src/lbfgsb.h"
#endif
}

// ************************************************************************
// Constructor for object class QGP
// ------------------------------------------------------------------------
QGP::QGP(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  faID_ = PSUADE_RS_QGP;
  expPower_ = 2;
  nQuantiles_ = 0;
  meanIndex_ = -1;
  homogeneous_ = 1; 
  numLBFGS_ = 4;
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
         "*      Quantile Gaussian Process Analysis\n");
    printDashes(PL_INFO, 0);
    printOutTS(PL_INFO,"* This method requires that the last ");
    printOutTS(PL_INFO,"input be the quantile input - \n");
    printOutTS(PL_INFO,
         "* that is, the last input needs to be in [0,1].\n");
    printOutTS(PL_INFO,"* Let say the number of samples   = N\n");
    printOutTS(PL_INFO,"*         the number of quantiles = m\n");
    printOutTS(PL_INFO,"* The sample needs to be in N/m groups of ");
    printOutTS(PL_INFO,"m with each group\n");
    printOutTS(PL_INFO,"* corresponding to a unique point in ");
    printOutTS(PL_INFO,"the first nInputs-1 space, and\n");
    printOutTS(PL_INFO,
         "* each group should have the same quantiles.\n");
    printOutTS(PL_INFO,
         "* In addition, one of the quantile should be 0.5 (mean).\n");
    printOutTS(PL_INFO, "* There are two options: \n");
    printOutTS(PL_INFO, 
         "* 1. Homogeneous : All inputs have same bounds.\n");
    printOutTS(PL_INFO, 
         "* 2. Otherwise   : Some inputs have different bounds.\n");
    printEquals(PL_INFO, 0);
  }

  //**/ get internal parameters
  int  ii;
  char pString[100], winput[100], equal[100], *strPtr;
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  { 
    printf("Option to set number of optimization starts ");
    printf("(if optimization is done).\n");
    snprintf(pString,100,
             "How many optimization starts (1 - 10, default = 4) : ");
    numLBFGS_ = getInt(1,10,pString);
    snprintf(pString,100,"RS_QGP_nStarts = %d", numLBFGS_);
    psConfig_.putParameter(pString);
  }
  else
  {
    strPtr = psConfig_.getParameter("RS_QGP_nStarts");
    if (strPtr != NULL)
    { 
      sscanf(strPtr, "%s %s %d", winput, equal, &ii);
      if (ii < 1 || ii > 10)
      { 
        printf("QGP INFO: Invalid number of optimization starts.\n");
        printf("          It is kept at %d.\n", numLBFGS_);
      } 
      else
      {   
        numLBFGS_ = ii;
        printf("QGP INFO: number of optimization starts from config = %d.\n",
               numLBFGS_);
      }     
    }     
  }
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
QGP::~QGP()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int QGP::initialize(double *XIn, double *YIn)
{
#ifndef HAVE_LBFGS
  printf("QGP ERROR: LBFGS has not been installed.\n");
  exit(1);
#endif
  //**/ ----------------------------------------------------------
  //**/ check whether all except quantile variable have same bounds
  //**/ ----------------------------------------------------------
  int ii, kk, status = 0;
  for (ii = 1; ii < nInputs_-1; ii++)
  {
    if (VecLBs_[ii] != VecLBs_[ii-1]) status++;
    if (VecUBs_[ii] != VecUBs_[ii-1]) status++;
  }
  if (status != 0)
  {
    if (psConfig_.InteractiveIsOn())
      for (ii = 0; ii < nInputs_-1; ii++)
        printf("Input %4d: bounds = %12.4e %12.4e\n", ii+1, 
               VecLBs_[ii], VecUBs_[ii]);
    printf("QGP INFO: Not all inputs (except last) have same range.\n");
    printf("          == Assume non-homogeneity.\n");
    homogeneous_ = 0; 
  } 
  if (homogeneous_) printf("QGP INFO: Assume input homogeneity.\n");
  else              printf("QGP INFO: Assume input non-homogeneity.\n");

  //**/ ----------------------------------------------------------
  //**/ check for how many quantiles and extract levels
  //**/ ----------------------------------------------------------
  for (kk = 1; kk < nSamples_; kk++)
    if (XIn[(kk+1)*nInputs_-1] == XIn[nInputs_-1]) break;
  nQuantiles_ = kk;
  for (kk = nQuantiles_; kk < nSamples_; kk+= nQuantiles_)
  {
    for (ii = 0; ii < nQuantiles_; ii++)
      if (XIn[(kk+ii+1)*nInputs_-1] != XIn[(ii+1)*nInputs_-1]) break;
    if (ii != nQuantiles_) break;
  }
  if (kk != nSamples_ || nQuantiles_ == 1)
  {
    printf("QGP ERROR: no quantile pattern is detected in the sample.\n");
    exit(1);
  }
  else
  {
    //**/ extract quantile levels and mean index
    VecQLevels_.setLength(nQuantiles_);
    for (kk = 0; kk < nQuantiles_; kk++)
    {
      VecQLevels_[kk] = XIn[(kk+1)*nInputs_-1];
      if (PABS(VecQLevels_[kk]-0.5) < 1e-14) meanIndex_ = kk;
    }
    if (meanIndex_ < 0)
    {
      printf("QGP ERROR: no mean (quantile=0.5) in the sample\n");
      exit(1);
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ scaling inputs and outputs except the last input
  //**/ ----------------------------------------------------------------
  VecXDataN_.setLength(nSamples_*nInputs_);
  initInputScaling(XIn, VecXDataN_.getDVector(), 1);
  for (ii = 0; ii < nInputs_; ii++)
  {
    if (VecXStds_[ii] == 0)
    {
      printf("QGP ERROR: input %d does not vary.\n",ii+1);
      exit(1);
    }
  }
  for (kk = 0; kk < nSamples_; kk++)
    VecXDataN_[(ii+1)*nInputs_-1] *= VecXStds_[nInputs_-1] + 
                                     VecXMeans_[nInputs_-1];
  VecXMeans_[nInputs_-1] = 0;
  VecXStds_[nInputs_-1] = 1;
  VecYDataN_.setLength(nSamples_);
  initOutputScaling(YIn, VecYDataN_.getDVector());
  if (YStd_ == 0)
  {
    printf("QGP ERROR: output does not vary.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ generate the Gaussian hyperparameters
  //**/ ----------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("QGP training begins....\n");
  if (train() < 0) return -1;
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("QGP training completed.\n");
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int QGP::genNDGridData(double *XIn, double *YIn, int *NOut, double **XOut, 
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
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("QGP interpolation begins....\n");
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  evaluatePoint(totPts, XX, (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("QGP interpolation completed.\n");
  (*XOut) = XX;
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int QGP::gen1DGridData(double *XIn,double *YIn,int ind1,double *settings, 
                        int *NOut, double **XOut, double **YOut)
{
  int    ii, kk, ndim=1;
  double HX;
  psVector VecXT;

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

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  psVector VecXOut, VecYOut;
  VecXOut.setLength(nPtsPerDim_*ndim);
  (*XOut) = VecXOut.takeDVector();
  VecYOut.setLength(nPtsPerDim_);
  (*YOut) = VecYOut.takeDVector();
  VecXT.setLength(nPtsPerDim_*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (kk = 0; kk < nInputs_; kk++) 
      VecXT[ii*nInputs_+kk] = settings[kk]; 
    VecXT[ii*nInputs_+ind1] = HX * ii + VecLBs_[ind1];
    (*XOut)[ii] = HX * ii + VecLBs_[ind1];
  }
   
  //**/ ----------------------------------------------------------------
  //**/ interpolate
  //**/ ----------------------------------------------------------------
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("QGP interpolation begins....\n");
  evaluatePoint(nPtsPerDim_, VecXT.getDVector(), *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("QGP interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 2D results for display
// ------------------------------------------------------------------------
int QGP::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
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
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("QGP interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("QGP interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int QGP::gen3DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3,
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
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("QGP interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("QGP interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int QGP::gen4DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3,
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
  vecHX.setLength(ndim);
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
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("QGP interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("QGP interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double QGP::evaluatePoint(double *X)
{
  int    ii, iOne=1;
  double Y=0.0;
  psVector vecXT;
  if (VecXMeans_.length() == 0)
  {
    printf("QGP ERROR : not initialized yet.\n");
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
double QGP::evaluatePoint(int npts, double *X, double *Y)
{
  int      ii, jj;
  psVector vecXT;

  if (VecXMeans_.length() == 0)
  {
    printf("QGP ERROR : not initialized yet.\n");
    exit(1);
  }
  vecXT.setLength(npts*nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    for (jj = 0; jj < npts; jj++)
      vecXT[jj*nInputs_+ii] = (X[jj*nInputs_+ii]-VecXMeans_[ii])/
                              VecXStds_[ii];
  interpolate(npts, vecXT.getDVector(), Y, NULL);
  for (jj = 0; jj < npts; jj++) Y[jj] = Y[jj] * YStd_ + YMean_;
  return 0.0;
}

// ************************************************************************
// Evaluate a given point, return also the standard deviation 
// ------------------------------------------------------------------------
double QGP::evaluatePointFuzzy(double *X, double &std)
{
  int      ii, iOne=1;
  double   Y=0.0;
  psVector vecXT;

  if (VecXMeans_.length() == 0)
  {
    printf("QGP ERROR : not initialized yet.\n");
    exit(1);
  }
  vecXT.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    vecXT[ii] = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
  interpolate(iOne, vecXT.getDVector(), &Y, &std);
  Y = Y * YStd_ + YMean_;
  if (std < 0) printf("QGP ERROR: variance (%e) < 0\n",std);
  else         std = sqrt(std) * YStd_;
  return Y;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double QGP::evaluatePointFuzzy(int npts,double *X, double *Y,
                                double *Ystds)
{
  int      ii, jj;
  psVector vecXT;

  if (VecXMeans_.length() == 0)
  {
    printf("QGP ERROR : not initialized yet.\n");
    exit(1);
  }
  vecXT.setLength(npts*nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    for (jj = 0; jj < npts; jj++)
      vecXT[jj*nInputs_+ii] = (X[jj*nInputs_+ii]-VecXMeans_[ii])/
                              VecXStds_[ii];
  interpolate(npts, vecXT.getDVector(), Y, Ystds);
  for (int ii = 0; ii < npts; ii++)
  {
    Y[ii] = Y[ii] * YStd_ + YMean_;
    if (Ystds[ii] < 0) 
         printf("QGP ERROR: variance (%e) < 0\n", Ystds[ii]);
    else Ystds[ii] = sqrt(Ystds[ii]) * YStd_;
  }
  return 0.0;
}

// ************************************************************************
// train to get lengthScales, magnitude, and noise components 
// ------------------------------------------------------------------------
int QGP::train()
{
  int    ii, kk, nhypers=5, optimizeFlag = 1;
  double dtmp;
  char   configCmd[101], *cString, pString[101], winput[1000];

  //**/ ----------------------------------------------------------
  //**/ obtain hyperparameters (thetas) from user, if any
  //**/ ----------------------------------------------------------
  nhypers = 4;
  if (homogeneous_) nhypers++;
  else              nhypers += nInputs_ - 1;
  VecHypers_.setLength(nhypers);
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    snprintf(pString,100,"Use user-provided hyperparameters? (y or n) ");
    getString(pString, winput);
    if (winput[0] == 'y')
    {
      printf("Hyperparameter 1: output scale\n");
      printf("Hyperparameter 2: added to all entries in C\n");
      printf("Hyperparameter 3: Ymean (constant)\n");
      printf("Hyperparameter 4: added to diagonal of C\n");
      printf("Hyperparameter 5: input 1 scale\n");
      printf("Hyperparameter 6: input 2 scale (if non-homogeneous)\n");
      printf(".... \n");
      for (ii = 0; ii < VecHypers_.length(); ii++)
      {
        snprintf(pString,100,"Enter hyperparameter %d : ",ii+1);
        VecHypers_[ii] = getDouble(pString);
      }
      optimizeFlag = 0;
    }
  }

  //**/ ----------------------------------------------------------
  //**/ If user does not provide hyperparameters, see if it can
  //**/ be obtained from configure object. 
  //**/ ----------------------------------------------------------
  if (optimizeFlag == 1)
  {
    for (ii = 0; ii < nhypers; ii++)
    {
      snprintf(configCmd,100,"RS_QGP%d", ii+1);
      cString = psConfig_.getParameter(configCmd);
      if (cString == NULL) break;
      else
      {
        sscanf(cString,"%s %lg", winput, &dtmp);
        VecHypers_[ii] = dtmp;
        if (psConfig_.InteractiveIsOn())
          printf("QGP hyperparameter %d = %e\n",ii+1,dtmp);
      }
    }
    if (ii == nhypers) 
    {
      optimizeFlag = 0;
      printf("QGP INFO: hyperparameters info from configure object.\n");
      printf("          ==> no optimization.\n");
    }
    else
    {
      printf("QGP INFO: No hyperparameter info from configure object\n");
    }
  }

  //**/ ----------------------------------------------------------
  //**/ Next, compute pairwise distance
  //**/ ----------------------------------------------------------
  int status = computeDistances();
  if (status != 0)
  {
    printf("QGP ERROR: There are repeated sample points.\n");
    printf("           Prune sample and re-run. BYE.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------
  //**/ if optimization is to be performed
  //**/ first set initial value for input scale parameter
  //**/ ----------------------------------------------------------
  double dmax = -PSUADE_UNDEFINED;
  double dmin =  PSUADE_UNDEFINED;
  if (optimizeFlag == 1)
  {
    //**/ set multiplier to the exponential term 
    dtmp = 0.0;
    for (kk = 0; kk < nSamples_; kk+=nQuantiles_) 
      if (PABS(VecYDataN_[kk+meanIndex_]) > dtmp) 
        dtmp = PABS(VecYDataN_[kk+meanIndex_]);
    VecHypers_[0] = dtmp;

    //**/ constant added to all entries
    VecHypers_[1] = log(1e-12);

    //**/ mean term (YM)
    VecHypers_[2] = 0;

    //**/ constant added to diagonals
    VecHypers_[3] = log(1e-10);

    //**/ input scales
    for (kk = 0; kk < nSamples_; kk+=nQuantiles_)
    {
      for (ii = 0; ii < nInputs_-1; ii++)
      {
        if (VecXDataN_[kk*nInputs_+ii] > dmax) 
          dmax = VecXDataN_[kk*nInputs_+ii];
        if (VecXDataN_[kk*nInputs_+ii] < dmin) 
          dmin = VecXDataN_[kk*nInputs_+ii];
      }
    }
    if (homogeneous_)
    {
      VecHypers_[4] = 0.25 * (dmax - dmin);
      VecHypers_[4] = log(VecHypers_[4]);
    }
    else
    {
      for (ii = 0; ii < nInputs_-1; ii++)
      {
        VecHypers_[ii+4] = 0.25 * (dmax - dmin);
        VecHypers_[ii+4] = log(VecHypers_[ii+4]);
      }
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
    int       its, nHist, maxHist=10, fail;
    double    dmean, dstd, minVal=1e35, FValue;
    psVector  VecPVals,VecGrads,VecW,VecXS,VecYS,VecHist,VecLB,VecUB;
    psIVector VecIS;
    Sampling *sampler=NULL;

    VecPVals.setLength(nhypers);
    //**/ Notes (Mar, 2017): these bounds have been carefully
    //**/    tuned to give good results. Do not change them.
    //**/    Since the inputs and outputs have been normalized,
    //**/    these bounds should work on most problems.
    //**/    (Ymean in [-1, 1] gives a little worse results)
    VecLB.setLength(nhypers);
    VecUB.setLength(nhypers);
    //**/ vertical magnitude
    VecLB[0] = -3.0;
    VecUB[0] = 10.0;
    //**/ nugget for the entire covariance matrix
    VecLB[1] = -28;
    VecUB[1] = - 3;
    //**/ Ymean 
    VecLB[2] = -0.5;
    VecUB[2] =  0.5;
    //**/ nugget for diagonal of the covariance matrix
    VecLB[3] = -24;
    VecUB[3] = - 2;
    //**/ the ranges
    if (homogeneous_)
    {
      VecLB[4] = -5;
      VecUB[4] =  5;
    }
    else
    {
      for (ii = 0; ii < nInputs_-1; ii++)
      {
        VecLB[ii+4] = -5;
        VecUB[ii+4] =  5;
      }
    }
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 2) 
    {
      for (ii = 0; ii < nhypers; ii++)
        printf("   Hyperparameter bounds %3d = %12.4e %12.4e\n",ii+1,
               VecLB[ii],VecUB[ii]);
    }
    if (nhypers > 51)
      sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
    else
      sampler = SamplingCreateFromID(PSUADE_SAMP_LPTAU);
    sampler->setInputBounds(nhypers,VecLB.getDVector(),VecUB.getDVector());
    sampler->setOutputParams(1);
    sampler->setSamplingParams(numLBFGS_, 1, 0);
    sampler->initialize(0);
    VecIS.setLength(numLBFGS_);
    VecXS.setLength(numLBFGS_*nhypers);
    VecYS.setLength(numLBFGS_);
    sampler->getSamples(numLBFGS_,nhypers,1,VecXS.getDVector(),
                        VecYS.getDVector(), VecIS.getIVector());
    delete sampler;
    for (ii = 0; ii < nhypers; ii++) VecXS[ii] = VecHypers_[ii];
 
    //**/ set up for LBFGS
#ifdef HAVE_LBFGS
    integer nInps, iprint=0, itask, *task=&itask, lsave[4], isave[44];
    integer *iwork, nCorr=5, *nbds, csave[60];
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

    for (ii = 0; ii < numLBFGS_; ii++)
    {
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 3 && 
          numLBFGS_ > 1 && ii > 0) 
        printf("QGP LBFGS sample %d (%d)\n",ii+1, numLBFGS_);
      for (kk = 0; kk < nhypers; kk++) VecPVals[kk] = VecXS[ii*nInps+kk];
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 3 && ii > 0) 
      {
        printf("   QGP initial hyperparameter values:\n");
        for (ii = 0; ii < VecHypers_.length(); ii++) 
          printf("   Hyperparameter %2d = %e\n",ii+1,
                 VecPVals[ii]);
      }
      nHist = 0;
      fail = 0;
      its = 0;
      *task = (integer) START;
      while (1 && its < 5000)
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
          dtmp = 0.0;
          for (kk = 0; kk < nInps; kk++)
            dtmp += pow(VecGrads[kk], 2.0);
          dtmp = sqrt(dtmp);
          if (psConfig_.InteractiveIsOn() && outputLevel_ > 2) 
            printf("   LBFGS Current FValue = %e (its=%d, |grad|=%e)\n",
                   FValue,its,dtmp);
          if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
            for (kk = 0; kk < nInps; kk++)
              printf("         Gradient %5d = %e\n",kk+1,VecGrads[kk]);
          if (FValue < minVal && status == 0)
          {
            for (kk = 0; kk < nhypers; kk++) 
              VecHypers_[kk] = VecPVals[kk];
            minVal = FValue;
          }
          if (nHist < maxHist && FValue != 1e35) 
            VecHist[nHist++] = FValue;
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
                printf("INFO: current iteration   = %d\n", its);
                printf("INFO: current best FValue = %e (%e)\n",minVal,
                       PABS(dstd/dmean));
                printf("INFO: current grad norm   = %e\n",dtmp);
              }
            }
            else
            {
              if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
                printf("INFO: convergence check: %e < 1e-3?\n",
                       PABS(dstd/dmean));
            }
          }
          if (isave[33] >= 300) 
          {
            *task = STOP_ITER;
            if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
            {
              printf("INFO: PSUADE issues a stop (> 300 iterations)\n");
              printf("INFO: current best FValue = %e\n", minVal);
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
            printf("INFO: current best FValue = %e\n", minVal);
          }
          break;
        }
      }
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
      {
        printf("   QGP final hyperparameter values:\n");
        for (kk = 0; kk < VecHypers_.length(); kk++) 
          printf("   Hyperparameter %2d = %e\n",kk+1,
                 VecHypers_[kk]);
        printf("   Sample %5d: Fvalue (final) = %e\n", ii+1, minVal);
      }
      if (ii == (numLBFGS_-1) && (fail == 1)) numLBFGS_++;
    }
    delete [] iwork;
    delete [] nbds;
#endif

    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    {
      printf("QGP very final hyperparameter values:\n");
      for (ii = 0; ii < VecHypers_.length(); ii++) 
        printf("   Hyperparameter %2d = %24.16e\n",ii+1,
               VecHypers_[ii]);
    }
    for (ii = 0; ii < VecHypers_.length(); ii++)
    {
      snprintf(configCmd,100,"RS_QGP%d %e",ii+1,VecHypers_[ii]);
      psConfig_.putParameter(configCmd);
    }
  }

  //**/ =======================================================
  //**/ create and factorize the final K matrix
  //**/ =======================================================
  psVector vecYT, vecCT;
  constructCMatrix(CMatrix_, VecHypers_);
  CMatrix_.LUDecompose();
  vecYT.setLength(nSamples_/nQuantiles_);
  vecCT.setLength(nSamples_/nQuantiles_);
  MatCInvYs_.setDim(nQuantiles_, nSamples_/nQuantiles_);
  for (ii = 0; ii < nQuantiles_; ii++)
  {
    for (kk = 0; kk < nSamples_; kk+=nQuantiles_) 
      vecYT[kk/nQuantiles_] = VecYDataN_[kk+ii]-VecHypers_[2];
    CMatrix_.LUSolve(vecYT, vecCT);
    MatCInvYs_.loadRow(ii, nSamples_/nQuantiles_, vecCT.getDVector());
  }
  if (psConfig_.GMModeIsOn())
  { 
    psMatrix CMatrix;
    constructCMatrix(CMatrix, VecHypers_);
    FILE *fp = fopen("Cmat.m", "w");
    fprintf(fp, "A = [\n");
    for (ii = 0; ii < nSamples_/nQuantiles_; ii++) 
    {
      for (kk = 0; kk < nSamples_/nQuantiles_; kk++) 
        fprintf(fp, "%16.8e ", CMatrix.getEntry(ii,kk));
      fprintf(fp, "\n");
    }
    fprintf(fp,"];\n");
    fclose(fp);
    printf("*** Final Cmatrix given in Cmat.m\n");
  }
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3 && 
      optimizeFlag == 1)
  {
    printf("Optimization summary: \n");
    for (ii = 0; ii < 5; ii++) 
      printf("Input %2d search bounds = %12.4e %12.4e\n",ii+1,
             VecXLows_[ii], VecXHighs_[ii]);
  }
  return 0;
}

// ************************************************************************
// compute gradients of log likelihood 
// ------------------------------------------------------------------------
double QGP::computeGradients(psVector VecHypers, psVector &VecGrads, 
                              int &retstat)
{
  int    jj, kk, ii, ll, status, count;
  double ddata, likelihood,sqrt2pi=sqrt(2*3.1415928), dist, fudge;
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
  int nInpsM1 = nInputs_ - 1;
  int nSamsDQ = nSamples_ / nQuantiles_;
  VecGrads.setLength(VecHypers.length());
  status = CMatrix.computeInverse(CInverse);
  if (status != 0)
  {
    printf("QGP ERROR (1): failed in matrix inversion (%d).\n",status);
    for (jj = 0; jj < VecHypers.length(); jj++)
      printf("  hyperparameter %3d = %e\n", jj+1, VecHypers[jj]);
    for (jj = 0; jj < VecHypers.length(); jj++) VecGrads[jj] = 100;
    return 1e35;
  }
  YVec.setLength(nSamsDQ);
  for (jj = 0; jj < nSamsDQ; jj++) 
    YVec[jj] = VecYDataN_[jj*nQuantiles_+meanIndex_]-VecHypers[2];
  CInvY.setLength(nSamsDQ);
  CInverse.matvec(YVec, CInvY, 0);

  YT.setLength(nSamsDQ);
  CPrime.setDim(nSamsDQ,nSamsDQ);
  double *TMat = CPrime.getMatrix1D();

  //**/ =======================================================
  //**/ compute negative log likelihood wrt theta_1 
  //**/ =======================================================
  //**/ create derivative of C with respect to theta_1
  count = 0;
  for (jj = 0; jj < nSamsDQ; jj++)
  {
    TMat[jj*nSamsDQ+jj] = 1;
    for (kk = jj+1; kk < nSamsDQ; kk++)
    {
      dist = 0.0;
      if (homogeneous_)
      {
        for (ii = 0; ii < nInpsM1; ii++)
          dist += pow(VecXDists_[count*nInpsM1+ii],expPower_)/
                  exp(expPower_*VecHypers[4]);
      }
      else
      {
        for (ii = 0; ii < nInpsM1; ii++)
          dist += pow(VecXDists_[count*nInpsM1+ii],expPower_)/
                  exp(expPower_*VecHypers[ii+4]);
      }
      dist = exp(-0.5 * dist);
      if (dist < 1.0e-50) dist = 0;
      TMat[jj*nSamsDQ+kk] = dist;
      TMat[kk*nSamsDQ+jj] = dist;
      count++;
    }
  }
  //**/ compute: C' * C^{-1} (Y - mu)
  CPrime.matvec(CInvY, YT, 0);
  //**/ compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  ddata = 0.0;
  for (jj = 0; jj < nSamsDQ; jj++) ddata += YT[jj] * CInvY[jj];
  VecGrads[0] = - ddata;
  //**/ compute: trace(C^{-1} C') - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  ddata = 0.0;
  for (ii = 0; ii < nSamsDQ; ii++) 
    for (jj = 0; jj < nSamsDQ; jj++) 
      ddata += CInverse.getEntry(ii, jj) * CPrime.getEntry(ii, jj);
  //**/ compute: [tr(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)]
  VecGrads[0] += ddata;
  //**/ compute: 0.5*exp(theta_1)*[tr(C^{-1}C')-(Y-mu)'C^{-1}C'*C^{-1}(Y-mu)]
  VecGrads[0] *= 0.5 * exp(VecHypers[0]);

  //**/ =======================================================
  //**/ compute negative log likelihood wrt theta_2 (offset to C)
  //**/ =======================================================
  //**/ compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  //**/          when C' is a all-1 matrix
  fudge = sqrt(1.0 * nSamsDQ);
  ddata = 0.0;
  for (jj = 0; jj < nSamsDQ; jj++) ddata += CInvY[jj];
  VecGrads[1] = - ddata * ddata - (fudge - 1.0) * ddata;
  //**/ compute: trace(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)
  ddata = 0.0;
  for (ii = 0; ii < nSamsDQ; ii++)
    for (jj = 0; jj < nSamsDQ; jj++) 
      ddata += CInverse.getEntry(ii, jj);
  VecGrads[1] += ddata;
  //**/ 0.5*exp(theta_2)*[tr(C^{-1}C')-(Y-mu)'C^{-1}C'*C^{-1}(Y-mu)]
  ddata = exp(VecHypers[1]);
  VecGrads[1] *= 0.5 * ddata;

  //**/ =======================================================
  //**/ compute negative log likelihood with respect to mean
  //**/ =======================================================
  //*/ compute: - 1' * C' * C^{-1} (Y - mu)
  ddata = 0.0;
  for (jj = 0; jj < nSamsDQ; jj++) ddata += CInvY[jj];
  VecGrads[2] = - ddata;

  //**/ =======================================================
  //**/ compute negative log likelihood wrt theta_3 
  //**/ =======================================================
  //**/ compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  //  //**/          since CPrime is a diagonal matrix of [dist]
  ddata = 0.0;
  for (jj = 0; jj < nSamsDQ; jj++) ddata += CInvY[jj] * CInvY[jj];
  VecGrads[3] = - ddata;
  //**/ compute: trace(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)
  ddata = 0.0;
  for (ii = 0; ii < nSamsDQ; ii++) 
    ddata += CInverse.getEntry(ii, ii);
  VecGrads[3] += ddata;
  //**/ 0.5*exp(theta_3)*[tr(C^{-1}C')-(Y-mu)'C^{-1}C'*C^{-1}(Y-mu)]
  VecGrads[3] *= 0.5 * exp(VecHypers[3]);

  //**/ =======================================================
  //**/ compute negative log likelihood wrt length scales
  //**/ =======================================================
  //**/ create derivative of C with respect to length scales
  if (homogeneous_)
  {
    count = 0;
    for (jj = 0; jj < nSamsDQ; jj++)
    {
      TMat[jj*nSamsDQ+jj] = 0;
      for (kk = jj+1; kk < nSamsDQ; kk++)
      {
        dist = 0.0;
        for (ii = 0; ii < nInpsM1; ii++)
          dist += pow(VecXDists_[count*nInpsM1+ii],expPower_)/
                      exp(expPower_*VecHypers[4]);
        dist = exp(VecHypers[0]) * exp(-0.5*dist);
        ddata = 0.0;
        for (ii = 0; ii < nInpsM1; ii++)
          ddata += pow(VecXDists_[count*nInpsM1+ii],2.0)/
                   exp((1.0+expPower_)*VecHypers[4])*0.5*expPower_;
        dist *= ddata;
        TMat[jj*nSamsDQ+kk] = dist;
        TMat[kk*nSamsDQ+jj] = dist;
        count++;
      }
    }
    //**/ compute: C' * C^{-1} (Y - mu)
    CPrime.matvec(CInvY, YT, 0);
    //**/ now compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
    ddata = 0.0;
    for (jj = 0; jj < nSamsDQ; jj++) ddata += YT[jj] * CInvY[jj];
    VecGrads[4] = - ddata;
    ddata = 0.0;
    //**/ now compute trace(C^{-1} C')
    for (ii = 0; ii < nSamsDQ; ii++)
      for (jj = 0; jj < nSamsDQ; jj++) 
        ddata += CPrime.getEntry(ii, jj) * CInverse.getEntry(ii, jj);
    //**/ now: trace(C^{-1} C') - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
    VecGrads[4] += ddata;
    //**/ finally: 0.5[tr(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)]
    VecGrads[4] *= 0.5;
  }
  else
  {
    for (ll = 0; ll < nInputs_-1; ll++)
    {
      count = 0;
      for (jj = 0; jj < nSamsDQ; jj++)
      {
        TMat[jj*nSamsDQ+jj] = 0;
        for (kk = jj+1; kk < nSamsDQ; kk++)
        {
          dist = 0.0;
          for (ii = 0; ii < nInpsM1; ii++)
            dist += pow(VecXDists_[count*nInpsM1+ii],expPower_)/
                        exp(expPower_*VecHypers[ii+4]);
          dist = exp(VecHypers[0]) * exp(-0.5*dist);
          ddata = pow(VecXDists_[count*nInpsM1+ll],2.0)/
                 exp((1.0+expPower_)*VecHypers[ll+4])*0.5*expPower_;
          dist *= ddata;
          TMat[jj*nSamsDQ+kk] = dist;
          TMat[kk*nSamsDQ+jj] = dist;
          count++;
        }
      }
      //**/ compute: C' * C^{-1} (Y - mu)
      CPrime.matvec(CInvY, YT, 0);
      //**/ now compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
      ddata = 0.0;
      for (jj = 0; jj < nSamsDQ; jj++) ddata += YT[jj] * CInvY[jj];
      VecGrads[ll+4] = - ddata;
      ddata = 0.0;
      //**/ now compute trace(C^{-1} C')
      for (ii = 0; ii < nSamsDQ; ii++)
        for (jj = 0; jj < nSamsDQ; jj++) 
          ddata += CPrime.getEntry(ii, jj) * CInverse.getEntry(ii, jj);
      //**/ now: trace(C^{-1} C') - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
      VecGrads[ll+4] += ddata;
      //**/ finally: 0.5[tr(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)]
      VecGrads[ll+4] *= 0.5;
    }
  }

  //**/ =======================================================
  //**/ compute negative log likelihood 
  //**/ use eigenSolve to detect non-positive definiteness of C
  //**/ =======================================================
  int errCount=0;
  CMatrix.eigenSolve(CT, YT, 1);
  likelihood = 0.0;
  for (jj = 0; jj < nSamsDQ; jj++) 
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
      printf("QGP WARNING: CMatrix non-positive definite (%d)\n",
             errCount);
      for (ii = 0; ii < VecHypers.length(); ii++)
        printf("   hyperparameter %d = %e\n",ii+1,VecHypers[ii]);
    }
  }
  for (jj = 0; jj < nSamsDQ; jj++) 
    likelihood += 0.5 * (CInvY[jj] * YVec[jj]);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    printf("   computeGradients: Likelihood = %e\n", likelihood);
  return likelihood;
}

// ************************************************************************
// compute pairwise distances 
// ------------------------------------------------------------------------
int QGP::computeDistances()
{
  int    ii, jj, kk, count, error=0, length, nSamsDQ, nInpsM1;
  double dist;
  psVector VecLDists;

  nSamsDQ = nSamples_ / nQuantiles_;
  nInpsM1 = nInputs_ - 1;
  length = (nSamsDQ*(nSamsDQ-1)/2)*nInpsM1;
  if (length < 0)
  {
    printf("QGP::computeDistances ERROR - vector length <= 0\n");
    printf("      This may be due to sample size or nInputs too large\n");
    printf("      which causes overflow.\n");
    printf("      nSamples = %d\n", nSamsDQ);
    printf("      nInputs  = %d\n", nInpsM1);
    printf("      length   = nSamples*(nSamples-1)/2*nInputs = %d\n", 
           length);
    exit(1);
  }
  VecLDists.setLength(length);
  double *LDists = VecLDists.getDVector();
  count = 0;
  for (jj = 0; jj < nSamsDQ; jj++)
  {
    for (kk = jj+1; kk < nSamsDQ; kk++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInpsM1; ii++)
      {
        LDists[count*nInpsM1+ii] = VecXDataN_[jj*nQuantiles_*nInputs_+ii] -
                                   VecXDataN_[kk*nQuantiles_*nInputs_+ii];
        if (LDists[count*nInpsM1+ii] < 0)
          LDists[count*nInpsM1+ii] = - LDists[count*nInpsM1+ii];
        dist += pow(LDists[count*nInpsM1+ii], 2.0);
      }
      if (dist == 0.0)
      {
        printf("QGP ERROR: repeated sample points.\n");
        printf("           Prune repeated points and re-run.\n");
        printf("Sample %d : (with sample point %d)\n", kk+1, jj+1);
        for (ii = 0; ii < nInpsM1; ii++)
          printf("   Input %d : %e\n",ii+1,
                 VecXDataN_[kk*nQuantiles_*nInputs_+ii]*
                 VecXStds_[ii] +VecXMeans_[ii]);
        error = 1;
      }
      count++;
    }
  }
  VecXDists_.load(count*nInpsM1, LDists);
  return error;
}

// ************************************************************************
// interpolation 
// ------------------------------------------------------------------------
int QGP::interpolate(int npts, double *XX, double *Y, double *Ystds)
{
  int    ii, kk, nn, ind, iOne=1, status, nSamsDQ, nInpsM1, iq;
  double dist, expn, ddata;
  psVector YT, YT2;

  nInpsM1 = nInputs_ - 1;
  nSamsDQ = nSamples_ / nQuantiles_;
  YT.setLength(nSamsDQ);
  YT2.setLength(nSamsDQ);
  Y[0] = 0;
  for (nn = 0; nn < npts; nn++)
  {
    for (kk = 0; kk < nSamsDQ; kk++)
    {
      expn = 0.0;
      if (homogeneous_)
      {
        for (ii = 0; ii < nInpsM1; ii++)
        {
          dist = VecXDataN_[kk*nQuantiles_*nInputs_+ii]-XX[nn*nInputs_+ii];
          expn += pow(dist,expPower_)/exp(expPower_*VecHypers_[4]);
        }
      }
      else
      {
        for (ii = 0; ii < nInpsM1; ii++)
        {
          dist = VecXDataN_[kk*nQuantiles_*nInputs_+ii]-XX[nn*nInputs_+ii];
          expn += pow(dist,expPower_)/exp(expPower_*VecHypers_[ii+4]);
        }
      }
      YT[kk] = exp(VecHypers_[0]) * exp(-0.5*expn);
    }
    Y[nn] = VecHypers_[2];
    ind = (nn + 1) * nInputs_ - 1;
    if      (XX[ind] <= VecQLevels_[0]) iq = 0;
    else if (XX[ind] >  VecQLevels_[nQuantiles_-1]) 
      iq = nQuantiles_ - 1;
    else
    {
      for (iq = 1; iq < VecQLevels_.length(); iq++)
        if (XX[ind] >  VecQLevels_[iq-1] &&
            XX[ind] <= VecQLevels_[iq]) break;
    }
    for (ii = 0; ii < nSamsDQ; ii++)
      Y[nn] += YT[ii] * MatCInvYs_.getEntry(iq,ii);

    if (Ystds != NULL)
    {
      Ystds[nn] = 0.0;
      ddata = exp(VecHypers_[0]) + exp(VecHypers_[1]) +
              exp(VecHypers_[3]);
      CMatrix_.LUSolve(YT, YT2);
      for (kk = 0; kk < nSamsDQ; kk++)
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
void QGP::constructCMatrix(psMatrix &CMatrix, psVector VecP)
{
  int    ii, jj, kk, count, nSamsDQ, nInpsM1;
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
  nInpsM1 = nInputs_ - 1;
  nSamsDQ = nSamples_ / nQuantiles_;
  CMatrix.setDim(nSamsDQ, nSamsDQ);
  double *TMat = CMatrix.getMatrix1D();
  //**/ This fudge is used to avoid matrix indefiniteness
  //**/ Probably not very helpful if theta_1 is huge
  fudge = sqrt(nSamsDQ);
  count = 0;
  for (jj = 0; jj < nSamsDQ; jj++)
  {
    TMat[jj*nSamsDQ+jj] = exp(VecP[0]) + 
                fudge*exp(VecP[1]) + exp(VecP[3]);
    for (kk = jj+1; kk < nSamsDQ; kk++)
    {
      dist = 0.0;
      if (homogeneous_)
      {
        for (ii = 0; ii < nInpsM1; ii++)
          dist += pow(VecXDists_[count*nInpsM1+ii],expPower_)/
                  exp(expPower_*VecP[4]);
      }
      else
      {
        for (ii = 0; ii < nInpsM1; ii++)
          dist += pow(VecXDists_[count*nInpsM1+ii],expPower_)/
                  exp(expPower_*VecP[ii+4]);
      }
      dist = exp(VecP[0]) * exp(-0.5*dist);
      if (dist < 1.0e-50) dist = 0;
      TMat[jj*nSamsDQ+kk] = dist + exp(VecP[1]);
      TMat[kk*nSamsDQ+jj] = dist + exp(VecP[1]);
      count++;
    }
  }
  return;
}

// ************************************************************************
// get hyperparameters 
// ------------------------------------------------------------------------
void QGP::getHyperparameters(psVector &inhyper)
{
  inhyper = VecHypers_;
}

