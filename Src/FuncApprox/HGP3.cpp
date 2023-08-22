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
// Functions for the class Homogeneous GP3
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

#include "HGP3.h"
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
// Constructor for object class HGP3
// ------------------------------------------------------------------------
HGP3::HGP3(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  char pString[101], winput1[1000], winput2[1000], *cString=NULL;

  faID_ = PSUADE_RS_HGP3;
  haveQuantiles_ = 0;
  expPower_ = 2;
  noChecking_ = 1;
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
         "*      Homogeneous Gaussian Process Analysis\n");
    printOutTS(PL_INFO,"* Default exponential degree = %e\n",expPower_);
    printEquals(PL_INFO, 0);
  }
  if (psConfig_.InteractiveIsOn() && psConfig_.RSExpertModeIsOn())
  {
    printOutTS(PL_INFO,"* Default exponential degree = %e\n",expPower_);
    snprintf(pString,100,"Set the exponential degree to? [1.5,2.0] ");
    while (expPower_ < 1.5 || expPower_ > 2)
      expPower_ = getDouble(pString);
    printf("Exponential degree = %e\n", expPower_);
    snprintf(pString,100,"GP_power = %e", expPower_);
    psConfig_.putParameter(pString);
    snprintf(pString,100,
            "Is the last input variable quantile variable? (y or n) ");
    getString(pString, winput1);
    if (winput1[0] == 'y') 
    {
      haveQuantiles_ = 1;
      snprintf(pString,100,"HGP3_haveQuantiles");
      psConfig_.putParameter(pString);
    }
  }
  else
  {
    cString = psConfig_.getParameter("GP_power");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %lg", winput1, winput2, &expPower_);
      if (expPower_ < 1.5 || expPower_ > 2)
      {
        printOutTS(PL_INFO,
             "HGP3 exponential (%d) invalid - reset to 2\n",expPower_);
        expPower_ = 2.0;
      }
      else if (psConfig_.InteractiveIsOn())
        printOutTS(PL_INFO,
                   "HGP3 exponential degree set to %e\n",expPower_);
    }
    cString = psConfig_.getParameter("HGP3_haveQuantiles");
    if (cString != NULL) haveQuantiles_ = 1;
  }
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
HGP3::~HGP3()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int HGP3::initialize(double *XIn, double *YIn)
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
    printf("HGP3 training begins....\n");
  if (train() < 0) return -1;
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("HGP3 training completed.\n");
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int HGP3::genNDGridData(double *XIn, double *YIn, int *NOut, double **XOut, 
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
    printf("HGP3 interpolation begins....\n");
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  evaluatePoint(totPts, XX, (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("HGP3 interpolation completed.\n");
  (*XOut) = XX;
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int HGP3::gen1DGridData(double *XIn,double *YIn,int ind1,double *settings, 
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
    printf("HGP3 interpolation begins....\n");
  evaluatePoint(nPtsPerDim_, VecXT.getDVector(), *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("HGP3 interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 2D results for display
// ------------------------------------------------------------------------
int HGP3::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
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
    printf("HGP3 interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("HGP3 interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int HGP3::gen3DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3,
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
    printf("HGP3 interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("HGP3 interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int HGP3::gen4DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3,
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
    printf("HGP3 interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
    printf("HGP3 interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double HGP3::evaluatePoint(double *X)
{
  int    ii, iOne=1;
  double Y=0.0;
  psVector vecXT;
  if (VecXMeans_.length() == 0)
  {
    printf("HGP3 ERROR : not initialized yet.\n");
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
double HGP3::evaluatePoint(int npts, double *X, double *Y)
{
  int      ii, jj;
  psVector vecXT;

  if (VecXMeans_.length() == 0)
  {
    printf("HGP3 ERROR : not initialized yet.\n");
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
double HGP3::evaluatePointFuzzy(double *X, double &std)
{
  int      ii, iOne=1;
  double   Y=0.0;
  psVector vecXT;

  if (VecXMeans_.length() == 0)
  {
    printf("HGP3 ERROR : not initialized yet.\n");
    exit(1);
  }
  vecXT.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    vecXT[ii] = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
  interpolate(iOne, vecXT.getDVector(), &Y, &std);
  Y = Y * YStd_ + YMean_;
  if (std < 0) printf("HGP3 ERROR: variance (%e) < 0\n",std);
  else         std = sqrt(std) * YStd_;
  return Y;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double HGP3::evaluatePointFuzzy(int npts,double *X, double *Y,
                                double *Ystds)
{
  int      ii, jj;
  psVector vecXT;

  if (VecXMeans_.length() == 0)
  {
    printf("HGP3 ERROR : not initialized yet.\n");
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
         printf("HGP3 ERROR: variance (%e) < 0\n", Ystds[ii]);
    else Ystds[ii] = sqrt(Ystds[ii]) * YStd_;
  }
  return 0.0;
}

// ************************************************************************
// train to get lengthScales, magnitude, and noise components 
// ------------------------------------------------------------------------
int HGP3::train()
{
  //**/ ----------------------------------------------------------
  //**/ check whether it should be quantile method
  //**/ ----------------------------------------------------------
  int status = 0, ii, kk;
  char pString[1000], winput[1000];
  for (ii = 1; ii < nInputs_; ii++)
  {
    if (VecLBs_[ii] != VecLBs_[ii-1]) status += ii;
    if (VecUBs_[ii] != VecUBs_[ii-1]) status += ii;
  }
  if (status == 2*(nInputs_-1))
  {
    printf("HGP3 INFO: use quantile method on the last input.\n");
    haveQuantiles_ = 1;
  }
  else if (status != 0)
  {
    if (noChecking_)
    {
      if (psConfig_.InteractiveIsOn())
      {
        printf("HGP3 INFO: no homogeneity checking\n");
        printf("     Homogeneity is assumed for the first %d inputs\n",
               nInputs_-1);
        if (psConfig_.MasterModeIsOn())
        {
          snprintf(pString,100,"Is the last input a quantile variable? ");
          getString(pString, winput);
          if (winput[0] == 'y') haveQuantiles_ = 1;
        }
      }
      else 
      {
        haveQuantiles_ = 0;
        printf("HGP3 INFO: Not all inputs have same ranges.\n");
        printf("           Will assume homogeneity.\n");
      }
    }
    else
    {
      if (psConfig_.InteractiveIsOn())
        for (ii = 0; ii < nInputs_; ii++)
          printf("Input %4d: bounds = %12.4e %12.4e\n", ii+1, 
                 VecLBs_[ii], VecUBs_[ii]);
      printf("HGP3 INFO: Not all inputs have same ranges.\n");
      printf("           Will assume homogeneity.\n");
      haveQuantiles_ = 0;
    } 
  } 

  //**/ ----------------------------------------------------------
  //**/ allocate and initialize length scales
  //**/ theta1, theta2, theta3
  //**/ If available from configure object, turn off optimization
  //**/ ----------------------------------------------------------
  int    nhypers = 5, optimizeFlag = 1;
  double dtmp;
  char   configCmd[101], *cString;
  if (haveQuantiles_) nhypers++;
  VecHypers_.setLength(nhypers);

  //**/ ask users to provide hyperparameters
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    snprintf(pString,100,"Use user-provided hyperparameters? (y or n) ");
    getString(pString, winput);
    if (winput[0] == 'y')
    {
      printf("Hyperparameter 1: input scale\n");
      printf("Hyperparameter 2: output scale\n");
      printf("Hyperparameter 3: added to all entries in C\n");
      printf("Hyperparameter 4: Ymean (constant)\n");
      printf("Hyperparameter 5: added to diagonal of C\n");
      if (haveQuantiles_)
        printf("Hyperparameter 6: quantile variable\n");
      for (ii = 0; ii < VecHypers_.length(); ii++)
      {
        snprintf(pString,100,"Enter hyperparameter %d : ",ii+1);
        VecHypers_[ii] = getDouble(pString);
      }
      optimizeFlag = 0;
    }
  }
  //**/ if not user inputs, search the configure object
  if (optimizeFlag == 1)
  {
    for (ii = 0; ii < nhypers; ii++)
    {
      snprintf(configCmd,100,"HGP_Hyperparam%d", ii+1);
      cString = psConfig_.getParameter(configCmd);
      if (cString == NULL) break;
      else
      {
        sscanf(cString,"%s %lg", winput, &dtmp);
        VecHypers_[ii] = dtmp;
      }
    }
    if (ii == nhypers) 
    {
      optimizeFlag = 0;
      printf("HGP3 INFO: got hyperparameters from configure object.\n");
    }
  }

  //**/ ----------------------------------------------------------
  //**/ compute pairwise distance
  //**/ ----------------------------------------------------------
  status = computeDistances();
  if (status != 0)
  {
    printf("HGP3 ERROR: there are repeated sample points.\n");
    printf("     GP will not handle this case. BYE.\n");
  }

  //**/ ----------------------------------------------------------
  //**/ if optimization is to be performed
  //**/ first set initial value for input scale parameter
  //**/ ----------------------------------------------------------
  double dmax = -PSUADE_UNDEFINED;
  double dmin =  PSUADE_UNDEFINED;
  if (optimizeFlag == 1)
  {
    for (kk = 0; kk < nSamples_; kk++)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        if (VecXDataN_[kk*nInputs_+ii] > dmax) 
          dmax = VecXDataN_[kk*nInputs_+ii];
        if (VecXDataN_[kk*nInputs_+ii] < dmin) 
          dmin = VecXDataN_[kk*nInputs_+ii];
      }
    }
    VecHypers_[0] = 0.25 * (dmax - dmin);
    if (VecHypers_[0] == 0)
    {
      printf("HGP3 ERROR: All inputs are constants.\n");
      exit(1);
    }
    VecHypers_[0] = log(VecHypers_[0]);

    //**/ set multiplier to the exponential term 
    dtmp = 0.0;
    for (kk = 0; kk < nSamples_; kk++) 
      if (PABS(VecYDataN_[kk]) > dtmp) dtmp = PABS(VecYDataN_[kk]);
    VecHypers_[1] = dtmp;
    
    //**/ constant added to all entries
    VecHypers_[2] = log(1e-12);
    //**/ mean term (YM)
    VecHypers_[3] = 0;
    //**/ constant added to diagonals
    VecHypers_[4] = log(1e-10);
    //**/ set quantile variable initial condition
    if (haveQuantiles_)
    {
      VecHypers_[5] = 0.25 * (dmax - dmin);
      VecHypers_[5] = log(VecHypers_[5]);
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
    int       its, nLBFGS=2, nHist, maxHist=10, fail;
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
    //**/ the ranges
    VecLB[0] = -5;
    VecUB[0] =  5;
    //**/ vertical magnitude
    VecLB[1] = -3.0;
    VecUB[1] = 10.0;
    //**/ nugget for the entire covariance matrix
    VecLB[2] = -28;
    VecUB[2] = - 3;
    //**/ Ymean 
    VecLB[3] = -0.5;
    VecUB[3] =  0.5;
    //**/ nugget for diagonal of the covariance matrix
    VecLB[4] = -24;
    VecUB[4] = - 2;
    //**/ range for quantile variable
    if (haveQuantiles_)
    {
      VecLB[5] = -5;
      VecUB[5] =  5;
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
    sampler->setSamplingParams(nLBFGS, 1, 0);
    sampler->initialize(0);
    VecIS.setLength(nLBFGS);
    VecXS.setLength(nLBFGS*nhypers);
    VecYS.setLength(nLBFGS);
    sampler->getSamples(nLBFGS,nhypers,1,VecXS.getDVector(),
                        VecYS.getDVector(), VecIS.getIVector());
    delete sampler;
    for (ii = 0; ii < nhypers; ii++) VecXS[ii] = VecHypers_[ii];
 
    //**/ set up for LBFGS
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

    for (ii = 0; ii < nLBFGS; ii++)
    {
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 3 && 
          nLBFGS > 1 && ii > 0) 
        printf("HGP3 LBFGS sample %d (%d)\n",ii+1, nLBFGS);
      for (kk = 0; kk < nhypers; kk++) VecPVals[kk] = VecXS[ii*nInps+kk];
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 3 && ii > 0) 
      {
        printf("   HGP3 initial hyperparameter values:\n");
        for (ii = 0; ii < VecHypers_.length(); ii++) 
          printf("   Hyperparameter %2d = %e\n",ii+1,
                 VecPVals[ii]);
      }
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
        printf("   HGP3 final hyperparameter values:\n");
        for (kk = 0; kk < VecHypers_.length(); kk++) 
          printf("   Hyperparameter %2d = %e\n",kk+1,
                 VecHypers_[kk]);
        printf("   Sample %5d: Fvalue (final) = %e\n", ii+1, minVal);
      }
      if (ii == (nLBFGS-1) && (fail == 1)) nLBFGS++;
    }
    delete [] iwork;
    delete [] nbds;

    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    {
      printf("HGP3 very final hyperparameter values:\n");
      for (ii = 0; ii < VecHypers_.length(); ii++) 
        printf("   Hyperparameter %2d = %24.16e\n",ii+1,
               VecHypers_[ii]);
    }
    for (ii = 0; ii < VecHypers_.length(); ii++)
    {
      snprintf(configCmd,100,"HGP_Hyperparam%d %e",ii+1,VecHypers_[ii]);
      psConfig_.putParameter(configCmd);
    }
  }

  //**/ =======================================================
  //**/ create and factorize the final K matrix
  //**/ =======================================================
  psVector VecYT;
  constructCMatrix(CMatrix_, VecHypers_);
  CMatrix_.LUDecompose();
  VecCInvY_.setLength(nSamples_);
  VecYT.setLength(nSamples_);
  for (kk = 0; kk < nSamples_; kk++) 
    VecYT[kk] = VecYDataN_[kk] - VecHypers_[3];
  CMatrix_.LUSolve(VecYT, VecCInvY_);

  if (psConfig_.GMModeIsOn())
  { 
    psMatrix CMatrix;
    constructCMatrix(CMatrix, VecHypers_);
    FILE *fp = fopen("Cmat.m", "w");
    fprintf(fp, "A = [\n");
    for (ii = 0; ii < nSamples_; ii++) 
    {
      for (kk = 0; kk < nSamples_; kk++) 
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
double HGP3::computeGradients(psVector VecHypers, psVector &VecGrads, 
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
  VecGrads.setLength(VecHypers.length());
  status = CMatrix.computeInverse(CInverse);
  if (status != 0)
  {
    printf("HGP3 ERROR (1): failed in matrix inversion (%d).\n",status);
    for (jj = 0; jj < VecHypers.length(); jj++)
      printf("  hyperparameter %3d = %e\n", jj+1, VecHypers[jj]);
    for (jj = 0; jj < VecHypers.length(); jj++) VecGrads[jj] = 100;
    return 1e35;
  }
  YVec.setLength(nSamples_);
  for (jj = 0; jj < nSamples_; jj++) 
    YVec[jj] = VecYDataN_[jj] - VecHypers[3];
  CInvY.setLength(nSamples_);
  CInverse.matvec(YVec, CInvY, 0);

  //**/ =======================================================
  //**/ compute negative log likelihood wrt length scales
  //**/ =======================================================
  YT.setLength(nSamples_);
  CPrime.setDim(nSamples_, nSamples_);
  double *TMat = CPrime.getMatrix1D();

  //**/ create derivative of C with respect to length scales
  count = 0;
  for (jj = 0; jj < nSamples_; jj++)
  {
    TMat[jj*nSamples_+jj] = 0;
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      if (haveQuantiles_)
      {
        for (ii = 0; ii < nInputs_-1; ii++)
          dist += pow(VecXDists_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecHypers[0]);
        dist += pow(VecXDists_[count*nInputs_+nInputs_-1],expPower_)/
                exp(expPower_*VecHypers[5]);
      }
      else
      {
        for (ii = 0; ii < nInputs_; ii++)
          dist += pow(VecXDists_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecHypers[0]);
      }
      dist *= 0.5;
      dist = exp(VecHypers[1]) * exp(-dist);
      ddata = 0.0;
      if (haveQuantiles_)
      {
        for (ii = 0; ii < nInputs_-1; ii++)
          ddata += pow(VecXDists_[count*nInputs_+ii],expPower_)/
                   exp((1.0+expPower_)*VecHypers[0])*0.5*expPower_;
      }
      else
      {
        for (ii = 0; ii < nInputs_; ii++)
          ddata += pow(VecXDists_[count*nInputs_+ii],2.0)/
                   exp((1.0+expPower_)*VecHypers[0])*0.5*expPower_;
      }
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
  VecGrads[0] = - ddata;
  ddata = 0.0;
  //**/ now compute trace(C^{-1} C')
  for (ii = 0; ii < nSamples_; ii++)
    for (jj = 0; jj < nSamples_; jj++) 
      ddata += CPrime.getEntry(ii, jj) * CInverse.getEntry(ii, jj);
  //**/ now: trace(C^{-1} C') - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  VecGrads[0] += ddata;
  //**/ finally: 0.5[tr(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)]
  VecGrads[0] *= 0.5;

  if (haveQuantiles_)
  {
    count = 0;
    for (jj = 0; jj < nSamples_; jj++)
    {
      TMat[jj*nSamples_+jj] = 0;
      for (kk = jj+1; kk < nSamples_; kk++)
      {
        dist = 0.0;
        for (ii = 0; ii < nInputs_-1; ii++)
          dist += pow(VecXDists_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecHypers[0]);
        dist += pow(VecXDists_[count*nInputs_+nInputs_-1],expPower_)/
                exp(expPower_*VecHypers[5]);
        dist *= 0.5;
        dist = exp(VecHypers[1]) * exp(-dist);
        ddata = pow(VecXDists_[count*nInputs_+nInputs_-1],expPower_)/
                exp((1+expPower_)*VecHypers[5])*0.5*expPower_;
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
    VecGrads[5] = - ddata;
    ddata = 0.0;
    //**/ now compute trace(C^{-1} C')
    for (ii = 0; ii < nSamples_; ii++)
      for (jj = 0; jj < nSamples_; jj++) 
        ddata += CPrime.getEntry(ii, jj) * CInverse.getEntry(ii, jj);
    //**/ now: trace(C^{-1} C') - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
    VecGrads[5] += ddata;
    //**/ finally: 0.5[tr(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)]
    VecGrads[5] *= 0.5;
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
      if (haveQuantiles_)
      {
        for (ii = 0; ii < nInputs_-1; ii++)
          dist += pow(VecXDists_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecHypers[0]);
        dist += pow(VecXDists_[count*nInputs_+nInputs_-1],expPower_)/
                exp(expPower_*VecHypers[5]);
      }
      else
      {
        for (ii = 0; ii < nInputs_; ii++)
          dist += pow(VecXDists_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecHypers[0]);
      }
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
  VecGrads[1] = - ddata;
  //**/ compute: trace(C^{-1} C') - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  ddata = 0.0;
  for (ii = 0; ii < nSamples_; ii++) 
    for (jj = 0; jj < nSamples_; jj++) 
      ddata += CInverse.getEntry(ii, jj) * CPrime.getEntry(ii, jj);
  //**/ compute: [tr(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)]
  VecGrads[1] += ddata;
  //**/ compute: 0.5*exp(theta_1)*[tr(C^{-1}C')-(Y-mu)'C^{-1}C'*C^{-1}(Y-mu)]
  VecGrads[1] *= 0.5 * exp(VecHypers[1]);

  //**/ =======================================================
  //**/ compute negative log likelihood wrt theta_2 (offset to C)
  //**/ =======================================================
  //**/ compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  //**/          when C' is a all-1 matrix
  fudge = sqrt(1.0 * nSamples_);
  ddata = 0.0;
  for (jj = 0; jj < nSamples_; jj++) ddata += CInvY[jj];
  VecGrads[2] = - ddata * ddata - (fudge - 1.0) * ddata;
  //**/ compute: trace(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)
  ddata = 0.0;
  for (ii = 0; ii < nSamples_; ii++)
    for (jj = 0; jj < nSamples_; jj++) 
      ddata += CInverse.getEntry(ii, jj);
  VecGrads[2] += ddata;
  //**/ 0.5*exp(theta_2)*[tr(C^{-1}C')-(Y-mu)'C^{-1}C'*C^{-1}(Y-mu)]
  ddata = exp(VecHypers[2]);
  VecGrads[2] *= 0.5 * ddata;

  //**/ =======================================================
  //**/ compute negative log likelihood with respect to mean
  //**/ =======================================================
  //*/ compute: - 1' * C' * C^{-1} (Y - mu)
  ddata = 0.0;
  for (jj = 0; jj < nSamples_; jj++) ddata += CInvY[jj];
  VecGrads[3] = - ddata;

  //**/ =======================================================
  //**/ compute negative log likelihood wrt theta_3 
  //**/ =======================================================
  //**/ compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  //  //**/          since CPrime is a diagonal matrix of [dist]
  ddata = 0.0;
  for (jj = 0; jj < nSamples_; jj++) ddata += CInvY[jj] * CInvY[jj];
  VecGrads[4] = - ddata;
  //**/ compute: trace(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)
  ddata = 0.0;
  for (ii = 0; ii < nSamples_; ii++) 
    ddata += CInverse.getEntry(ii, ii);
  VecGrads[4] += ddata;
  //**/ 0.5*exp(theta_3)*[tr(C^{-1}C')-(Y-mu)'C^{-1}C'*C^{-1}(Y-mu)]
  VecGrads[4] *= 0.5 * exp(VecHypers[4]);

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
      printf("HGP3 WARNING: CMatrix non-positive definite (%d)\n",
             errCount);
      for (ii = 0; ii < VecHypers.length(); ii++)
        printf("   hyperparameter %d = %e\n",ii+1,VecHypers[ii]);
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
int HGP3::computeDistances()
{
  int    ii, jj, kk, count, error=0, length;
  double dist;
  psVector VecLDists;

  length = (nSamples_*(nSamples_-1)/2)*nInputs_;
  if (length < 0)
  {
    printf("HGP3::computeDistances ERROR - vector length <= 0\n");
    printf("      This may be due to sample size or nInputs too large\n");
    printf("      which causes overflow.\n");
    printf("      nSamples = %d\n", nSamples_);
    printf("      nInputs  = %d\n", nInputs_);
    printf("      length   = nSamples*(nSamples-1)/2*nInputs = %d\n", 
           length);
    exit(1);
  }
  VecLDists.setLength((nSamples_*(nSamples_-1)/2)*nInputs_);
  double *LDists = VecLDists.getDVector();
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
        printf("HGP3 ERROR: repeated sample points.\n");
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
int HGP3::interpolate(int npts, double *XX, double *Y, double *Ystds)
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
      if (haveQuantiles_)
      {
        for (ii = 0; ii < nInputs_-1; ii++)
        {
          dist = VecXDataN_[kk*nInputs_+ii] - XX[nn*nInputs_+ii];
          expn += pow(dist,expPower_)/exp(expPower_*VecHypers_[0]);
        }
        dist = VecXDataN_[kk*nInputs_+nInputs_-1]-XX[nn*nInputs_+nInputs_-1];
        expn += pow(dist,expPower_)/exp(expPower_*VecHypers_[5]);
        expn *= 0.5;
      }
      else
      {
        for (ii = 0; ii < nInputs_; ii++)
        {
          dist = VecXDataN_[kk*nInputs_+ii] - XX[nn*nInputs_+ii];
          expn += pow(dist,expPower_)/exp(expPower_*VecHypers_[0]);
        }
        expn *= 0.5;
      }
      YT[kk] = exp(VecHypers_[1]) * exp(-expn);
    }
    Y[nn] = VecHypers_[3];
    for (kk = 0; kk < nSamples_; kk++)
      Y[nn] += YT[kk] * VecCInvY_[kk];
    if (Ystds != NULL)
    {
      Ystds[nn] = 0.0;
      ddata = exp(VecHypers_[1]) + 
              exp(VecHypers_[2]) +
              exp(VecHypers_[4]);
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
void HGP3::constructCMatrix(psMatrix &CMatrix, psVector VecP)
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
    TMat[jj*nSamples_+jj] = exp(VecP[1]) + 
                fudge*exp(VecP[2]) + exp(VecP[4]);
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      if (haveQuantiles_)
      {
        for (ii = 0; ii < nInputs_-1; ii++)
          dist += pow(VecXDists_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecP[0]);
        dist += pow(VecXDists_[count*nInputs_+nInputs_-1],expPower_)/
                exp(expPower_*VecP[5]);
      }
      else
      {
        for (ii = 0; ii < nInputs_; ii++)
          dist += pow(VecXDists_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecP[0]);
      }
      dist *= 0.5;
      dist = exp(VecP[1]) * exp(-dist);
      if (dist < 1.0e-50) dist = 0;
      TMat[jj*nSamples_+kk] = dist + exp(VecP[2]);
      TMat[kk*nSamples_+jj] = dist + exp(VecP[2]);
      count++;
    }
  }
  return;
}

// ************************************************************************
// get hyperparameters 
// ------------------------------------------------------------------------
void HGP3::getHyperparameters(psVector &inhyper)
{
  inhyper = VecHypers_;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double HGP3::setParams(int targc, char **targv)
{
  if (targc > 0)
  {
    if (!strcmp(targv[0], "no_checking")) noChecking_ = 1;
  }
  return 0;
}

// ************************************************************************
// save context 
// ------------------------------------------------------------------------
void HGP3::saveFA()
{
  int ii, jj;
  FILE *fp = fopen(".psuadeHGP3", "w");
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
  fprintf(fp, "%d\n", haveQuantiles_);
  fprintf(fp, "%d\n", noChecking_);
  fclose(fp);
}

// ************************************************************************
// retrieve context 
// ------------------------------------------------------------------------
void HGP3::retrieveFA()
{
  int    ii, jj;
  double ddata1, ddata2;

  FILE *fp = fopen(".psuadeHGP3", "r");
  if (fp == NULL) 
  {
    printf("HGP3 ERROR: .psuadeHGP3 file not found.\n");
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
  fscanf(fp, "%d", &haveQuantiles_);
  fscanf(fp, "%d", &noChecking_);
  fclose(fp);
}
