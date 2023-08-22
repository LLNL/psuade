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
// Functions for the class GP1
// AUTHOR : CHARLES TONG
// DATE   : 2005
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "GP1.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"
#include "psVector.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

#ifdef HAVE_TPROS
extern "C" 
{
  void TprosTrain(int nInputs, int nTrains, double *trainInputs,
                  double *trainOutput, int, double *, double *);

  void TprosInterp(int nTests, double *inputs, double *output, double *stds);

  void TprosGetLengthScales(int nInputs, double *lengthScales);
}
#endif

// ************************************************************************
// Constructor for object class GP1
// ------------------------------------------------------------------------
GP1::GP1(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  faID_ = PSUADE_RS_GP1;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
GP1::~GP1()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int GP1::initialize(double *XIn, double *YIn)
{
#ifdef HAVE_TPROS
  int    ss, ii;
  char   pString[101], response[500], *cString;
  psVector VecStds, VecX, VecY;

  //**/ ----------------------------------------------------------------
  //**/ users have the option to normalize the output
  //**/ ----------------------------------------------------------------
  response[0] = 'n';
  if (!psConfig_.RSExpertModeIsOn())
  {
    cString = psConfig_.getParameter("normalize_outputs");
    if (cString != NULL) response[0] = 'y';
  }
  if (psConfig_.RSExpertModeIsOn())
  {
    snprintf(pString,100,"GP1: normalize output? (y or n) ");
    getString(pString, response);
  }
   
  //**/ ----------------------------------------------------------------
  //**/ prescale
  //**/ ----------------------------------------------------------------
  VecX.setLength(nSamples_*nInputs_);
  initInputScaling(XIn, VecX.getDVector(), 0);
  VecY.setLength(nSamples_);
  if (response[0] == 'y')
       initOutputScaling(YIn, VecY.getDVector());
  else VecY.load(nSamples_, YIn);
   
  //**/ ----------------------------------------------------------------
  //**/ generate the Gaussian hyperparameters
  //**/ ----------------------------------------------------------------
  psVector VecLScales;
  VecStds.setLength(nSamples_);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("GP1 training begins....\n");
  TprosTrain(nInputs_,nSamples_,VecX.getDVector(),VecY.getDVector(),
             0,NULL,VecStds.getDVector());
  for (ss = 0; ss < nSamples_; ss++) VecStds[ss] = 0.0;
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("GP1 training completed.\n");
  if (psConfig_.RSExpertModeIsOn())
  {
    VecLScales.setLength(nInputs_);
    TprosGetLengthScales(nInputs_, VecLScales.getDVector());
    printf("GP1 training information: \n");
    for (ii = 0; ii < nInputs_; ii++)
       printf("Input %d mean,std,length scale = %e %e %e\n",
              ii+1, VecXMeans_[ii], VecXStds_[ii], VecLScales[ii]); 
  }
  if (psConfig_.RSCodeGenIsOn())
    printf("GP1 INFO: response surface stand-alone code not available.\n");
  return 0;
#else
  printf("PSUADE ERROR : GP1 not installed.\n");
  return -1;
#endif
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int GP1::genNDGridData(double *XIn, double *YIn, int *NOut, double **XOut, 
                      double **YOut)
{
#ifdef HAVE_TPROS
  int    totPts, ii, jj;
  double *XX;
  psVector VecXT;

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
    printf("GP1 interpolation begins....\n");
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  VecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
  {
    for (jj = 0; jj < totPts; jj++)
      VecXT[jj*nInputs_+ii] = (XX[jj*nInputs_+ii] - VecXMeans_[ii]) /
                              VecXStds_[ii];
  } 
  TprosInterp(totPts, VecXT.getDVector(), *YOut, NULL);
  for (ii = 0; ii < totPts; ii++)
    (*YOut)[ii] = ((*YOut)[ii] * YStd_) + YMean_;
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("GP1 interpolation completed.\n");
  (*XOut) = XX;
#else
  printf("PSUADE ERROR : GP1 not installed.\n");
  return -1;
#endif
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int GP1::gen1DGridData(double *XIn, double *YIn, int ind1,double *settings, 
                       int *NOut, double **XOut, double **YOut)
{
#ifdef HAVE_TPROS
  int    totPts, ii, jj, kk;
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
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  psVector VecXOut;
  VecXOut.setLength(totPts);
  (*XOut) = VecXOut.takeDVector();
  VecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (kk = 0; kk < nInputs_; kk++) 
      VecXT[ii*nInputs_+kk] = settings[kk]; 
    VecXT[ii*nInputs_+ind1] = HX * ii + VecLBs_[ind1];
    (*XOut)[ii] = HX * ii + VecLBs_[ind1];
  }
    
  //**/ ----------------------------------------------------------------
  //**/ scale and interpolate
  //**/ ----------------------------------------------------------------
  for (ii = 0; ii < nInputs_; ii++)
  {
    for (jj = 0; jj < totPts; jj++)
      VecXT[jj*nInputs_+ii] = (VecXT[jj*nInputs_+ii] - VecXMeans_[ii]) /
                               VecXStds_[ii];
  } 
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("GP1 interpolation begins....\n");
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  TprosInterp(totPts, VecXT.getDVector(), *YOut, NULL);
  for (ii = 0; ii < totPts; ii++) 
    (*YOut)[ii] = ((*YOut)[ii] * YStd_) + YMean_;
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("GP1 interpolation completed.\n");
  (*NOut) = totPts;
#else
  printf("PSUADE ERROR : GP1 not installed.\n");
#endif
  return 0;
}

// ************************************************************************
// Generate 2D results for display
// ------------------------------------------------------------------------
int GP1::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                 double *settings, int *NOut, double **XOut, double **YOut)
{
#ifdef HAVE_TPROS
  int totPts, ii, jj, kk, index;
  psVector VecHX, VecXT;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_;
  VecHX.setLength(2);
  VecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  VecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  psVector VecXOut;
  VecXOut.setLength(totPts*2);
  (*XOut) = VecXOut.takeDVector();
  VecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      index = ii * nPtsPerDim_ + jj;
      for (kk = 0; kk < nInputs_; kk++) 
        VecXT[index*nInputs_+kk] = settings[kk]; 
      VecXT[index*nInputs_+ind1]  = VecHX[0] * ii + VecLBs_[ind1];
      VecXT[index*nInputs_+ind2]  = VecHX[1] * jj + VecLBs_[ind2];
      (*XOut)[index*2]   = VecHX[0] * ii + VecLBs_[ind1];
      (*XOut)[index*2+1] = VecHX[1] * jj + VecLBs_[ind2];
    }
  }
    
  //**/ ----------------------------------------------------------------
  //**/ scale and interpolate
  //**/ ----------------------------------------------------------------
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  for (ii = 0; ii < nInputs_; ii++)
  {
    for (jj = 0; jj < totPts; jj++)
      VecXT[jj*nInputs_+ii] = (VecXT[jj*nInputs_+ii] - VecXMeans_[ii]) /
                              VecXStds_[ii];
  } 
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("GP1 interpolation begins....\n");
  TprosInterp(totPts, VecXT.getDVector(), *YOut, NULL);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("GP1 interpolation completed.\n");
  for (ii = 0; ii < totPts; ii++)
    (*YOut)[ii] = ((*YOut)[ii] * YStd_) + YMean_;
  (*NOut) = totPts;
#else
  printf("PSUADE ERROR : GP1 not installed.\n");
#endif
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int GP1::gen3DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3,
              double *settings, int *NOut, double **XOut, double **YOut)
{
#ifdef HAVE_TPROS
  int totPts, ii, jj, kk, ll, index;
  psVector VecHX, VecXT;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  VecHX.setLength(3);
  VecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  VecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  VecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  VecXT.setLength(totPts*nInputs_);
  psVector VecXOut;
  VecXOut.setLength(totPts*3);
  (*XOut) = VecXOut.takeDVector();
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      for (ll = 0; ll < nPtsPerDim_; ll++) 
      {
        index = ii * nPtsPerDim_ * nPtsPerDim_ + jj * nPtsPerDim_ + ll;
        for (kk = 0; kk < nInputs_; kk++) 
          VecXT[index*nInputs_+kk] = settings[kk]; 
        VecXT[index*nInputs_+ind1] = VecHX[0] * ii + VecLBs_[ind1];
        VecXT[index*nInputs_+ind2] = VecHX[1] * jj + VecLBs_[ind2];
        VecXT[index*nInputs_+ind3] = VecHX[2] * ll + VecLBs_[ind3];
        (*XOut)[index*3]   = VecHX[0] * ii + VecLBs_[ind1];
        (*XOut)[index*3+1] = VecHX[1] * jj + VecLBs_[ind2];
        (*XOut)[index*3+2] = VecHX[2] * ll + VecLBs_[ind3];
      }
    }
  }
    
  //**/ ----------------------------------------------------------------
  //**/ scale and interpolate
  //**/ ----------------------------------------------------------------
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  for (ii = 0; ii < nInputs_; ii++)
  {
    for (jj = 0; jj < totPts; jj++)
      VecXT[jj*nInputs_+ii] = (VecXT[jj*nInputs_+ii] - VecXMeans_[ii]) /
                              VecXStds_[ii];
  } 
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("GP1 interpolation begins....\n");
  TprosInterp(totPts, VecXT.getDVector(), *YOut, NULL);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("GP1 interpolation completed.\n");
  for (ii = 0; ii < totPts; ii++)
    (*YOut)[ii] = ((*YOut)[ii] * YStd_) + YMean_;
  (*NOut) = totPts;
#else
  printf("PSUADE ERROR : GP1 not installed.\n");
#endif
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int GP1::gen4DGridData(double *XIn,double *YIn,int ind1,int ind2, int ind3,
                       int ind4,double *settings, int *NOut, double **XOut, 
                       double **YOut)
{
#ifdef HAVE_TPROS
  int totPts, ii, jj, kk, ll, mm, index;
  psVector VecXT, VecHX;

  //**/ ----------------------------------------------------------------
  //**/ initialize
  //**/ ----------------------------------------------------------------
  initialize(XIn, YIn);
  if ((*NOut) == -999 || XOut == NULL || YOut == NULL) return 0;
  
  //**/ ----------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ----------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  VecHX.setLength(4);
  VecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  VecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  VecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 
  VecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4]) / (nPtsPerDim_ - 1); 

  //**/ ----------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ----------------------------------------------------------------
  VecXT.setLength(totPts*nInputs_);
  psVector VecXOut;
  VecXOut.setLength(totPts*4);
  (*XOut) = VecXOut.takeDVector();
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
            VecXT[index*nInputs_+kk] = settings[kk]; 
          VecXT[index*nInputs_+ind1] = VecHX[0] * ii + VecLBs_[ind1];
          VecXT[index*nInputs_+ind2] = VecHX[1] * jj + VecLBs_[ind2];
          VecXT[index*nInputs_+ind3] = VecHX[2] * ll + VecLBs_[ind3];
          VecXT[index*nInputs_+ind4] = VecHX[3] * mm + VecLBs_[ind4];
          (*XOut)[index*4]   = VecHX[0] * ii + VecLBs_[ind1];
          (*XOut)[index*4+1] = VecHX[1] * jj + VecLBs_[ind2];
          (*XOut)[index*4+2] = VecHX[2] * ll + VecLBs_[ind3];
          (*XOut)[index*4+3] = VecHX[3] * mm + VecLBs_[ind4];
        }
      }
    }
  }
    
  //**/ ----------------------------------------------------------------
  //**/ scale and interpolate
  //**/ ----------------------------------------------------------------
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  for (ii = 0; ii < nInputs_; ii++)
  {
    for (jj = 0; jj < totPts; jj++)
      VecXT[jj*nInputs_+ii] = (VecXT[jj*nInputs_+ii] - VecXMeans_[ii]) /
                              VecXStds_[ii];
  } 
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("GP1 interpolation begins....\n");
  TprosInterp(totPts, VecXT.getDVector(), *YOut, NULL);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1) 
    printf("GP1 interpolation completed.\n");
  for (ii = 0; ii < totPts; ii++)
    (*YOut)[ii] = ((*YOut)[ii] * YStd_) + YMean_;
  (*NOut) = totPts;
#else
  printf("PSUADE ERROR : GP1 not installed.\n");
#endif
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double GP1::evaluatePoint(double *X)
{
  double Y=0.0;
#ifdef HAVE_TPROS
  int    ii, iOne=1;
  psVector VecXT;
  if (VecXMeans_.length() == 0)
  {
    printf("PSUADE ERROR : not initialized yet.\n");
    exit(1);
  }
  VecXT.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    VecXT[ii] = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
  TprosInterp(iOne, VecXT.getDVector(), &Y, NULL);
  Y = Y * YStd_ + YMean_;
#else
  printf("PSUADE ERROR : GP1 not installed.\n");
#endif
  return Y;
}

// ************************************************************************
// Evaluate a number of points 
// ------------------------------------------------------------------------
double GP1::evaluatePoint(int npts, double *X, double *Y)
{
#ifdef HAVE_TPROS
  int ii, jj;
  psVector VecXT;
  if (VecXMeans_.length() == 0)
  {
    printf("PSUADE ERROR : not initialized yet.\n");
    exit(1);
  }
  VecXT.setLength(npts*nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    for (jj = 0; jj < npts; jj++)
      VecXT[jj*nInputs_+ii] = (X[jj*nInputs_+ii]-VecXMeans_[ii])/VecXStds_[ii];
  TprosInterp(npts, VecXT.getDVector(), Y, NULL);
  for (jj = 0; jj < npts; jj++)
    Y[jj] = Y[jj] * YStd_ + YMean_;
#else
  printf("PSUADE ERROR : GP1 not installed.\n");
#endif
  return 0.0;
}

// ************************************************************************
// Evaluate a given point, return also the standard deviation 
// ------------------------------------------------------------------------
double GP1::evaluatePointFuzzy(double *X, double &std)
{
  double Y=0.0;
#ifdef HAVE_TPROS
  int ii;
  psVector VecXT;
  if (VecXMeans_.length() == 0)
  {
    printf("PSUADE ERROR : not initialized yet.\n");
    exit(1);
  }
  VecXT.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    VecXT[ii] = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
  TprosInterp(1, VecXT.getDVector(), &Y, &std);
  Y = Y * YStd_ + YMean_;
  if (std < 0) printf("GP1 ERROR: variance < 0\n");
  else         std = std * YStd_;
#else
  printf("PSUADE ERROR : GP1 not installed.\n");
#endif
  return Y;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double GP1::evaluatePointFuzzy(int npts,double *X, double *Y, double *Ystd)
{
#ifdef HAVE_TPROS
  int    ii, jj;
  psVector VecXT;
  if (VecXMeans_.length() == 0)
  {
    printf("PSUADE ERROR : not initialized yet.\n");
    exit(1);
  }
  VecXT.setLength(npts*nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    for (jj = 0; jj < npts; jj++)
      VecXT[jj*nInputs_+ii] = (X[jj*nInputs_+ii]-VecXMeans_[ii])/VecXStds_[ii];
  TprosInterp(npts, VecXT.getDVector(), Y, Ystd);
  for (int ii = 0; ii < npts; ii++)
  {
    Y[ii] = Y[ii] * YStd_ + YMean_;
    if (Ystd[ii] < 0) printf("GP1 ERROR: variance < 0\n");
    else              Ystd[ii] = Ystd[ii] * YStd_;
  }
#else
  printf("PSUADE ERROR : GP1 not installed.\n");
#endif
  return 0.0;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double GP1::setParams(int targc, char **targv)
{
  int    ii, ind;
  double mmax, ddata=0.0, range;
  char   pString[101];
  FILE   *fp;
  psVector  VecLScales;
  psIVector VecIT;
                                                                                
  if (targc > 0 && !strcmp(targv[0], "rank"))
  {
    VecLScales.setLength(nInputs_);
#ifdef HAVE_TPROS
    TprosGetLengthScales(nInputs_, VecLScales.getDVector());
#else
    printf("PSUADE ERROR : GP1 not installed.\n");
    return 0.0;
#endif
    mmax = 0.0;
    for (ii = 0; ii < nInputs_; ii++)
    {
      VecLScales[ii] = 1.0/VecLScales[ii];
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
         fp = fopen("scilabgpsa.sci", "w");
    else fp = fopen("matlabgpsa.m", "w");
    if (fp == NULL)
    {
      printf("GP1 ERROR: something wrong with opening a write file.\n");
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
      snprintf(pString,100,"GP Ranking");
      fwritePlotTitle(fp, pString);
      snprintf(pString,100,"Input Numbers");
      fwritePlotXLabel(fp, pString);
      snprintf(pString,100,"GP Measure");
      fwritePlotYLabel(fp, pString);
      if (plotScilab())
      {
        fprintf(fp,"a=gca();\n");
        fprintf(fp,
           "a.data_bounds=[0, ymin; n+1, ymax+0.01*(ymax-ymin)];\n");
      }
      else
      {
        fprintf(fp,"axis([0 n+1 ymin ymax+0.01*(ymax-ymin)])\n");
      }
      fclose(fp);
      if (plotScilab())
           printf("GP ranking in file scilabgpsa.sci\n");
      else printf("GP ranking in file matlabgpsa.m\n");
    }
    VecIT.setLength(nInputs_);
    for (ii = 0; ii < nInputs_; ii++) VecIT[ii] = ii;
    sortDbleList2a(nInputs_, VecLScales.getDVector(), VecIT.getIVector());
    if (targc == 1)
    {
      printAsterisks(PL_INFO, 0);
      printf("* GP1 screening rankings \n");
      printAsterisks(PL_INFO, 0);
      for (ii = nInputs_-1; ii >= 0; ii--)
        printf("*  Rank %3d : Input = %3d (score = %5.1f) (ref = %e)\n", 
               nInputs_-ii, VecIT[ii]+1, VecLScales[ii], 
               VecLScales[ii]*mmax*0.01);
      printAsterisks(PL_INFO, 0);
    }
    if (targc > 1)
    {
      ind = *(int *) targv[1];
      if (ind >= 0 && ind < nInputs_) ddata = VecLScales[ind];
    }
  }
  return ddata;
}

