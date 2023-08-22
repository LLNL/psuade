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
// Functions for the class Hybrid Legendre-GP
// AUTHOR : CHARLES TONG
// DATE   : 2019
// ************************************************************************
//**/ Aug 2019 - experiments show that using Legendre coefficients
//**/            as hyperparameters causes too much fluctuations
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

#include "HybridGP.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"
#include "Psuade.h"
#include "Sampling.h"
#include "HomLegendreRegression.h"

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
// Constructor for object class HybridGP
// ------------------------------------------------------------------------
HybridGP::HybridGP(int nInputs,int nSamples):FuncApprox(nInputs,nSamples)
{
  faID_ = PSUADE_RS_HYGP;
  expPower_ = 2.0;
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
         "*         Hybrid Legendre-Gaussian Process Analysis\n");
    printOutTS(PL_INFO,"* Default exponential degree = %e\n",expPower_);
    printEquals(PL_INFO, 0);
  }
  polyOrder_ = 2;
  homRegressor_ = NULL;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
HybridGP::~HybridGP()
{
  if (homRegressor_ != NULL) delete homRegressor_;
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int HybridGP::initialize(double *XIn, double *YIn)
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
    printf("HybridGP training begins....\n");
  train();
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
    printf("HybridGP training completed.\n");
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int HybridGP::genNDGridData(double *XIn, double *YIn, int *NOut, 
                            double **XOut, double **YOut)
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
    printf("HybridGP interpolation begins....\n");
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  evaluatePoint(totPts, XX, (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("HybridGP interpolation completed.\n");
  (*XOut) = XX;
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int HybridGP::gen1DGridData(double *XIn, double *YIn, int ind1,
                 double *settings, int *NOut, double **XOut, double **YOut)
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
    printf("HybridGP interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("HybridGP interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 2D results for display
// ------------------------------------------------------------------------
int HybridGP::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
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
    printf("HybridGP interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("HybridGP interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int HybridGP::gen3DGridData(double *XIn, double *YIn,int ind1,int ind2,
                 int ind3, double *settings, int *NOut, double **XOut,
                 double **YOut)
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
    printf("HybridGP interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("HybridGP interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int HybridGP::gen4DGridData(double *XIn,double *YIn,int ind1,int ind2, 
                   int ind3, int ind4, double *settings, int *NOut,
                   double **XOut, double **YOut)
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
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("HybridGP interpolation begins....\n");
  evaluatePoint(totPts, vecXT.getDVector(), (*YOut));
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 1)
    printf("HybridGP interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double HybridGP::evaluatePoint(double *X)
{
  int    ii, iOne=1;
  double Y=0.0;
  psVector vecXT;
  if (VecXMeans_.length() == 0)
  {
    printf("HybridGP ERROR : not initialized yet.\n");
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
double HybridGP::evaluatePoint(int npts, double *X, double *Y)
{
  int      ii, jj;
  psVector VecXX;

  if (VecXMeans_.length() == 0)
  {
    printf("HybridGP ERROR : not initialized yet.\n");
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
double HybridGP::evaluatePointFuzzy(double *X, double &std)
{
  int      ii, iOne=1;
  double   Y=0.0;
  psVector VecXX;

  if (VecXMeans_.length() == 0)
  {
    printf("HybridGP ERROR : not initialized yet.\n");
    exit(1);
  }
  VecXX.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    VecXX[ii] = (X[ii] - VecXMeans_[ii]) / VecXStds_[ii];
  interpolate(iOne, VecXX.getDVector(), &Y, &std);
  Y = Y * YStd_ + YMean_;
  if (std < 0) printf("HybridGP ERROR: variance (%e) < 0\n",std);
  else         std = sqrt(std) * YStd_;
  return Y;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double HybridGP::evaluatePointFuzzy(int npts,double *X, double *Y,
                                    double *Ystds)
{
  int      ii, jj;
  psVector VecXX;

  if (VecXMeans_.length() == 0)
  {
    printf("HybridGP ERROR : not initialized yet.\n");
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
      printf("HybridGP ERROR: variance (%e) < 0\n", Ystds[ii]);
    else               Ystds[ii] = sqrt(Ystds[ii]) * YStd_;
  }
  return 0.0;
}

// ************************************************************************
// train to get lengthScales, magnitude, and noise components 
// ------------------------------------------------------------------------
int HybridGP::train()
{
  int  ii, jj, kk, ss;

  //**/ ----------------------------------------------------------
  //**/ hyperparameters: VecLegendreCoeffs_
  //**/ hyperparameters: VecHyperPs_
  //**/ 1 : for all input variables
  //**/ 2 : Ymean (copy from VecLegendreCoeffs)
  //**/ 3 : Y magnitude 
  //**/ 4 : added to all entries in covariance matrix
  //**/ 5 : added to diagonal entries in covariance matrix
  //**/ 6 : for quantile variable (depends on sample)
  //**/ ----------------------------------------------------------

  //**/ ----------------------------------------------------------
  //**/ first check to see if this there are quantiles 
  //**/ if so, hasQuantiles = 1 and sampleIncrement != 1
  //**/ else   hasQuantiles = 0 and sampleIncrement == 1
  //**/ ----------------------------------------------------------
  int    sampleIncrement, nSamplesUnique;
  double dtmp, daccu;
  char   configCmd[101], winput[1000], *cString, *tArgv[1];
  psVector VecXT,VecYT,VecLocalX,VecLocalY,VecRegCoefs,VecRegStds;

  if (psConfig_.InteractiveIsOn())
    printf("HybridGP: Check whether sample has a quantile variable\n");
  for (ii = 1; ii < nSamples_; ii++)
    if (VecXDataN_[ii*nInputs_+nInputs_-1] == VecXDataN_[nInputs_-1])
      break;
  sampleIncrement = ii;
  nSamplesUnique = nSamples_ / sampleIncrement;
  if (sampleIncrement > 1)
  {
    for (ii = 0; ii < nSamples_; ii+=sampleIncrement)
    {
      for (jj = 1; jj < sampleIncrement; jj++)
      {
        if (ii+jj < nSamples_)
        {
          for (kk = 0; kk < nInputs_-1; kk++)
            if (VecXDataN_[(ii+jj)*nInputs_+kk] !=
                VecXDataN_[ii*nInputs_+kk])
              break;
          if (kk != (nInputs_-1)) break;
        }
      }
      if (jj != sampleIncrement) break;
    }
    if (ii != nSamples_) sampleIncrement = 1;
  }

  //**/ if no quantile variable or if there is quantile but the
  //**/ sample size is not consisten, check further
  if (sampleIncrement == 1 ||
      (nSamplesUnique * sampleIncrement != nSamples_))
  {
    if (psConfig_.InteractiveIsOn())
    {
     printf("Sample does not have consistent quantile information. As\n");
     printf("such, HybridGP expects one of the following 2 scenarios:\n");
     printf("(1) If all inputs are homogeneous, treat the sample as\n");
     printf("    containing only the means (not quantiles).\n");
     printf("(2) If the last input looks like a quantile variable,\n");
     printf("    look for pre-computed Legendre information (because\n");
     printf("    otherwise they cannot be computed from the means).\n");
    }
    for (ii = 1; ii < nInputs_; ii++)
    {
      if (VecLBs_[ii] != VecLBs_[0]) break;
      if (VecUBs_[ii] != VecUBs_[0]) break;
    }
    if (ii == nInputs_)
    {
      if (psConfig_.InteractiveIsOn())
      {
       printf("(1) Inputs are homogeneous (same bounds) - 2 scenarios:\n");
       printf("    (a) Check Legendre/hyperparameters in configure object\n");
       printf("    (b) Compute Legendre coefs and hyperparameters\n");
      }
      hasQuantiles_ = 0;
      sampleIncrement = 1;
      nSamplesUnique = nSamples_ / sampleIncrement;
    }
    else
    {
      for (ii = 1; ii < nInputs_-1; ii++)
      {
        if (VecLBs_[ii] != VecLBs_[0]) break;
        if (VecUBs_[ii] != VecUBs_[0]) break;
      }
      if (ii == (nInputs_-1))
      {
        if (psConfig_.InteractiveIsOn())
          printf("(2) The last input looks like quantile variable.\n");
        hasQuantiles_ = 1;
        sampleIncrement = 1;
      }
      else
      {
        printf("ERROR: Inputs not homogeneous (bounds not the same).\n");
        printf("HybridGP does not apply in this case. BYE.\n");
        exit(1);
      }
    }
  }
  else
  {
    //**/ checking indeed it has quantile information
    if (psConfig_.InteractiveIsOn())
    {
      printf("Sample has consistent quantile information: 2 scenarios\n");
      printf("(1) Check hyperparameters in the configure object.\n");
      printf("(2) Compute hyperparameters if not found\n");
    }
    hasQuantiles_ = 1;
  }

  //**/ ----------------------------------------------------------
  //**/ read Legendre coefficients from configuration table, if any
  //**/ ----------------------------------------------------------
  int numTerms=1, iOne=1;
  if      (polyOrder_ == 1) numTerms = 2;
  else if (polyOrder_ == 2) numTerms = 4;
  int useConfig=0;
  for (ii = 0; ii < numTerms; ii++)
  {
    snprintf(configCmd,100,"HYGP_LegendreCoeffs%d", ii+1);
    cString = psConfig_.getParameter(configCmd);
    if (cString == NULL) break;
    else
    {
      if (ii == 0) 
      {
        VecLegendreCoefs_.setLength(numTerms);
        VecLegendreStds_.setLength(numTerms);
      }
      sscanf(cString,"%s %lg %lg", winput, &dtmp, &daccu);
      //printf("HybridGP: Legendre coef from config %d = %e\n",ii+1,dtmp);
      VecLegendreCoefs_[ii] = dtmp;
      VecLegendreStds_[ii] = daccu;
      useConfig++;
    } 
  }
  if (useConfig > 0 && useConfig != numTerms)
  {
    printf("HybridGP ERROR: unable to read all Legendre coefficients ");
    printf("   from the configure object. Ask developers for help.\n"); 
    exit(1);
  }
  else if (useConfig == numTerms)
  {
    if (hasQuantiles_ == 0)
      homRegressor_ = new HomLegendreRegression(nInputs_,nSamples_);
    else
      homRegressor_ = new HomLegendreRegression(nInputs_-1,nSamples_);
    tArgv[0] = (char *) &polyOrder_;
    homRegressor_->setParams(iOne, tArgv);
    homRegressor_->setBounds(VecLBs_.getDVector(),
                             VecUBs_.getDVector());
  }

  //**/ ----------------------------------------------------------
  //**/ read GP hyperparameters, if any
  //**/ ----------------------------------------------------------
  int nhypers = 5;
  snprintf(configCmd,100,"HYGP_Hyperparam6");
  cString = psConfig_.getParameter(configCmd);
  if (cString != NULL) 
  {
    nhypers = 6;
    hasQuantiles_ = 1;
  }
  if (hasQuantiles_) nhypers = 6;
  VecHyperPs_.setLength(nhypers);
  for (ii = 0; ii < nhypers; ii++)
  {
    snprintf(configCmd,100,"HYGP_Hyperparam%d", ii+1);
    cString = psConfig_.getParameter(configCmd);
    if (cString == NULL) break;
    else
    {
      sscanf(cString,"%s %lg", winput, &dtmp);
      VecHyperPs_[ii] = dtmp;
      //printf("HybridGP: hyperparam from config %d = %e\n",ii+1,dtmp);
    }
  }
  int optimizeFlag=1;
  if (ii == nhypers) optimizeFlag = 0;
  else               optimizeFlag = 1;
  if (psConfig_.InteractiveIsOn())
  {
    printf("HybridGP summary:\n");
    if (hasQuantiles_) printf("- Has quantiles.\n");
    else               printf("- No quantiles used.\n");
    if (VecLegendreCoefs_.length() > 0) 
         printf("- Legendre coefficients retrieved from configure.\n");
    else printf("- Legendre coefficients to be computed.\n");
    if (optimizeFlag == 0) 
         printf("- GP hyperparameters retrieved from configure.\n");
    else printf("- GP hyperparameters to be computed.\n");
  }

  //**/ ----------------------------------------------------------
  //**/ VecLegendre length = 0 and the sample has quantile 
  //**/ information but the sample is not consisten ==> error
  //**/ (since means cannot be computed)
  //**/ ----------------------------------------------------------
  if (VecLegendreCoefs_.length() == 0 && 
      sampleIncrement == 1 && hasQuantiles_ == 1)
  {
    printf("HybridGP ERROR: cannot compute hyperparameters. BYE\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------
  //**/ if Legendre polynomial has not been set up, extract means
  //**/ (or use sample outputs directly if it does not have
  //**/ quantile information), and then compute Legendre coefficients
  //**/ ----------------------------------------------------------
  int nInputs2, foundMean;
  double xVal;
  if (VecLegendreCoefs_.length() == 0)
  {
    if (hasQuantiles_ == 0) nInputs2 = nInputs_;
    else                    nInputs2 = nInputs_ - 1;
    VecLocalX.setLength(nSamplesUnique*nInputs2);
    VecLocalY.setLength(nSamplesUnique);
    //**/ See if the mean can be found. If so, just use it.
    for (ii = 0; ii < nSamplesUnique; ii++)
    {
      for (jj = 0; jj < nInputs2; jj++)
        VecLocalX[ii*nInputs2+jj] = 
               VecXDataN_[ii*nInputs_*sampleIncrement+jj];
      VecLocalY[ii] = 0;
      foundMean = 0;
      for (jj = 0; jj < sampleIncrement; jj++)
      {
        xVal = VecXDataN_[(ii*sampleIncrement+jj)*nInputs_+nInputs_-1];
        if (xVal == 0.5)
        {
          VecLocalY[ii] = VecYDataN_[ii*sampleIncrement+jj];
          foundMean = 1;
        }
        else VecLocalY[ii] += VecYDataN_[ii*sampleIncrement+jj];
      }
      if (foundMean == 0) VecLocalY[ii] /= sampleIncrement;
    }

    //**/ ----------------------------------------------------------
    //**/ train HomLegendreRegression
    //**/ ----------------------------------------------------------
    int interactiveSave=0;
    if (psConfig_.InteractiveIsOn()) 
    {
      interactiveSave = 1;
      psConfig_.InteractiveOff();
    }
    homRegressor_ = new HomLegendreRegression(nInputs2,nSamplesUnique);
    tArgv[0] = (char *) &polyOrder_;
    homRegressor_->setParams(iOne, tArgv);
    homRegressor_->setBounds(VecLBs_.getDVector(),
                             VecUBs_.getDVector());
    homRegressor_->initialize(VecLocalX.getDVector(), 
                              VecLocalY.getDVector());
    homRegressor_->getRegressionCoefficients(VecRegCoefs,VecRegStds);
    double *dPtr = VecLocalX.getDVector();
     
    if (psConfig_.InteractiveIsOn())
    {
      for (ss = 0; ss < nSamplesUnique; ss++)
      {
        homRegressor_->evaluatePoint(iOne,&dPtr[ss*nInputs2],&dtmp,
                                     VecRegCoefs.getDVector());
        printf("Compare %5d = %12.4e %12.4e(true)\n",ss+1,dtmp,VecLocalY[ss]);
      } 
    } 
    if (interactiveSave == 1) psConfig_.InteractiveOn();
    VecLegendreCoefs_ = VecRegCoefs;
    VecLegendreStds_ = VecRegStds;

    winput[0] = 'n';
    if (psConfig_.MasterModeIsOn())
    {
      printf("HybridGP: store Legendre coefficients to configure? (y or n) ");
      scanf("%s", winput);
    }
    if (winput[0] == 'y')
    {
      snprintf(configCmd,100,"HYGP_LegendreCoeffs");
      psConfig_.removeParameter(configCmd);
      for (ii = 0; ii < numTerms; ii++)
      {
        VecLegendreStds_[ii] = 0.01 * PABS(VecLegendreCoefs_[ii]);
        snprintf(configCmd,100,"HYGP_LegendreCoeffs%d %e %e",ii+1,
                VecLegendreCoefs_[ii], VecLegendreStds_[ii]);
        psConfig_.putParameter(configCmd);
      }
    }
    //**/ check HomLegendre 
    VecXT.setLength(nInputs_);
    if (psConfig_.InteractiveIsOn())
    {
      daccu = 0.0;
      for (ii = 0; ii < nSamplesUnique; ii++)
      {
        for (jj = 0; jj < nInputs2; jj++) 
          VecXT[jj] = VecLocalX[ii*nInputs2+jj];
        homRegressor_->evaluatePoint(iOne, VecXT.getDVector(), &dtmp,
                                     VecLegendreCoefs_.getDVector());
        daccu += pow(VecLocalY[ii] - dtmp, 2.0);
      }
      printf("HybridGP: HomLegendre check rms error = %e\n",
             sqrt(daccu/nSamplesUnique));
    }
  }

  //**/ ----------------------------------------------------------
  //**/ compute pairwise distance 
  //**/ ----------------------------------------------------------
  int status = computeDistances();
  if (status != 0)
  {
    printf("HybridGP ERROR: There are repeated sample points.\n");
    printf("                Prune the sample and do it again,\n");
    return -1;
  }

  //**/ ----------------------------------------------------------
  //**/ If optimization is needed, first prescribe initial guess 
  //**/ for hyperparameters
  //**/ ----------------------------------------------------------
  double dmax = -PSUADE_UNDEFINED;
  double dmin =  PSUADE_UNDEFINED;
  if (optimizeFlag == 1)
  {
    //**/ homogeneous input hyperparameter
    for (kk = 0; kk < nSamples_*nInputs_; kk++)
    {
      if (VecXDataN_[kk] > dmax) dmax = VecXDataN_[kk];
      if (VecXDataN_[kk] < dmin) dmin = VecXDataN_[kk];
    }
    VecHyperPs_[0] = 0.25 * (dmax - dmin);
    VecHyperPs_[0] = log(VecHyperPs_[0]);

    //**/ Ymean
    VecHyperPs_[1] = 0;

    //**/ multiplier to the exponential term 
    dmax = -PSUADE_UNDEFINED;
    for (ii = 0; ii < nSamples_; ii++) 
      if (PABS(VecYDataN_[ii]) > dmax) dmax = PABS(VecYDataN_[ii]);
    VecHyperPs_[2] = 0.5 * dmax;
    
    //**/ constant added to all entries
    VecHyperPs_[3] = log(1e-12);

    //**/ constant added to diagonals
    VecHyperPs_[4] = log(1e-10);

    //**/ quantile variable is in [0,1]
    if (hasQuantiles_)
    {
      VecHyperPs_[5] = 0.5;
      VecHyperPs_[5] = log(VecHyperPs_[5]);
    }

    //**/ ----------------------------------------------------------
    //**/ tracking optimization history 
    //**/ ----------------------------------------------------------
    VecXLows_.setLength(VecHyperPs_.length());
    VecXHighs_.setLength(VecHyperPs_.length());
    for (ii = 0; ii < VecHyperPs_.length(); ii++) 
    {
      VecXLows_[ii]  = +PSUADE_UNDEFINED;
      VecXHighs_[ii] = -PSUADE_UNDEFINED;
    }

    //**/ --------------------------------------------------------
    //**/ initialize variables for optimization
    //**/ --------------------------------------------------------
    psVector  VecLB, VecUB, VecPVals, VecXS, VecYS;
    psIVector VecIS;

    VecPVals.setLength(nhypers);
    VecLB.setLength(nhypers);
    VecUB.setLength(nhypers);
    //*/ input common variable
    VecLB[0] = -5;
    VecUB[0] =  5;
    //**/ YMean
    VecLB[1] = -0.5;
    VecUB[1] =  0.5;
    //**/ vertical magnitude
    VecLB[2] = -3.0;
    VecUB[2] = 10.0;
    //**/ nugget for the entire covariance matrix
    VecLB[3] = -28;
    VecUB[3] = - 3;
    //**/ nugget for diagonal of the covariance matrix
    VecLB[4] = -24;
    VecUB[4] = - 2;
    //*/ quantile variable
    if (hasQuantiles_ == 1)
    {
      VecLB[5] = -5;
      VecUB[5] =  5;
    }
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 2)
    {
      for (ii = 0; ii < VecLB.length(); ii++)
        printf("   Hyperparameter bounds %3d = %12.4e %12.4e\n",ii+1,
               VecLB[ii],VecUB[ii]);
    }

    int nLBFGS = 3;
    Sampling *sampler;
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
    for (ii = 0; ii < nhypers; ii++) VecXS[ii] = VecHyperPs_[ii];

    //**/ set up for LBFGS
    integer nInps, iprint=0, itask, *task=&itask, lsave[4], isave[44];
    integer *iwork, nCorr=5, *nbds, csave[60];
    int     nHist, maxHist=10, fail;
    double  factr, pgtol, dsave[29], dmean, dstd, dtemp, FValue, minVal=1e35;
    psVector VecHist, VecGrads, VecW;

    nInps = nhypers;
    VecGrads.setLength(nInps);
    nbds = new integer[nInps];
    for (ii = 0; ii < nInps; ii++) nbds[ii] = 2;
    factr = 1e7;
    pgtol = 1e-8;
    ii = (2 * nCorr + 5) * nInps + 11 * nCorr * nCorr + 8 * nCorr;
    VecW.setLength(ii);
    iwork = new integer[3*nInps];
    int its = 0;
    VecHist.setLength(maxHist);

    for (ii = 0; ii < nLBFGS; ii++)
    {
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 3 &&
          nLBFGS > 1 && ii > 0) 
        printf("HybridGP LBFGS sample %d (%d)\n",ii+1, nLBFGS);
      for (jj = 0; jj < nInps; jj++) VecPVals[jj] = VecXS[ii*nInps+jj];
      if (psConfig_.InteractiveIsOn() && outputLevel_ > 3 && ii > 0) 
      {
        printf("   HybridGP initial hyperparameter values:\n");
        for (jj = 0; jj < VecPVals.length(); jj++) 
          printf("   Hyperparameter %2d = %e\n",jj+1,
                 VecPVals[jj]);
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
          if (outputLevel_ > 2) 
            for (jj = 0; jj < nInps; jj++)
              printf("     Hyperparameter %5d = %e\n",jj+1,VecPVals[jj]);

          FValue = computeGradients(VecPVals, VecGrads, status);
          if (FValue == 1e35) 
          {
            fail = 1;
            break;
          }
          dtemp = 0.0;
          for (jj = 0; jj < nInps; jj++)
            dtemp += pow(VecGrads[jj], 2.0);
          dtemp = sqrt(dtemp);

          if (psConfig_.InteractiveIsOn() && outputLevel_ > 1) 
            printf("   LBFGS Current FValue = %e (its=%d, |grad|=%e)\n",
                   FValue,its,dtemp);
          if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
            for (jj = 0; jj < nInps; jj++)
              printf("         Gradient %5d = %e\n",jj+1,VecGrads[jj]);
          if (FValue < minVal && status == 0)
          {
            for (jj = 0; jj < nhypers; jj++) 
              VecHyperPs_[jj] = VecPVals[jj];
            minVal = FValue;
          }
          if (nHist < maxHist && FValue != 1e35) VecHist[nHist++] = FValue;
          else if (FValue != 1e35)
          {
            for (jj = 0; jj < maxHist-1; jj++) VecHist[jj] = VecHist[jj+1];
            VecHist[maxHist-1] = FValue;
          }
          if (nHist >= maxHist && FValue != 1e35)
          {
            dmean = 0.0;
            for (jj = 0; jj < maxHist; jj++) dmean += VecHist[jj];
            dmean /= (double) maxHist;
            dstd = 0.0;
            for (jj = 0; jj < maxHist; jj++) 
              dstd += pow(VecHist[jj]-dmean,2.0);
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
                printf("INFO: current grad norm   = %e\n",dtemp);
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
            if (outputLevel_ > 3) 
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
        printf("   HybridGP final hyperparameter values:\n");
        for (jj = 0; jj < VecHyperPs_.length(); jj++) 
          printf("   Hyperparameter %2d = %e\n",jj+1, VecHyperPs_[jj]);
        printf("   Sample %5d: Fvalue (final) = %e\n", ii+1, minVal);
      }
      if (ii == (nLBFGS-1) && (fail == 1)) nLBFGS++;
    }
    delete [] iwork;
    delete [] nbds;

    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    {
      printf("HybridGP very final hyperparameter values:\n");
      for (ii = 0; ii < VecHyperPs_.length(); ii++) 
        printf("   Hyperparameter %2d = %24.16e\n",ii+1,VecHyperPs_[ii]);
    }
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3)
    {
      printf("Optimization summary: \n");
      for (ii = 0; ii < VecHyperPs_.length(); ii++) 
        printf("Input %2d search bounds = %12.4e %12.4e\n",ii+1,
               VecXLows_[ii], VecXHighs_[ii]);
    }
    winput[0] = 'n';
    if (psConfig_.MasterModeIsOn())
    {
      printf("HybridGP: store hyperparameters to configure? (y or n) ");
      scanf("%s", winput);
    }
    if (winput[0] == 'y')
    {
      for (ii = 0; ii < VecHyperPs_.length(); ii++)
      {
        snprintf(configCmd,100,"HYGP_Hyperparam%d %e",ii+1,VecHyperPs_[ii]);
        psConfig_.putParameter(configCmd);
      }
    }
  }

  //**/ =======================================================
  //**/ create and factorize the final K matrix
  //**/ =======================================================
  if (hasQuantiles_ == 0) nInputs2 = nInputs_;
  else                    nInputs2 = nInputs_ - 1;
  constructCMatrix(CMatrix_, VecHyperPs_);
  CMatrix_.LUDecompose();
  VecCInvY_.setLength(nSamples_);
  VecXT.setLength(nInputs2);
  VecYT.setLength(nSamples_);
  for (jj = 0; jj < nSamples_; jj++) 
  {
    for (ii = 0; ii < nInputs2; ii++) 
      VecXT[ii] = VecXDataN_[jj*nInputs_+ii];
    homRegressor_->evaluatePoint(iOne, VecXT.getDVector(), &dtmp,
                                 VecLegendreCoefs_.getDVector());
    VecYT[jj] = VecYDataN_[jj] - dtmp - VecHyperPs_[1];
  }
  CMatrix_.LUSolve(VecYT, VecCInvY_);

  return 0;
}

// ************************************************************************
// compute gradients of log likelihood 
// ------------------------------------------------------------------------
double HybridGP::computeGradients(psVector VecHypers, psVector &VecGrads, 
                                  int &retstat)
{
  int    jj, kk, ii, ll, status, count, iOne=1;
  double ddata, likelihood,sqrt2pi=sqrt(2*3.1415928), dist, fudge;
  psMatrix MatC, MatCInv, MatCPrime, MatCT;
  psVector VecX, VecY, VecCInvY, VecYT;

  //**/ =======================================================
  //**/ fill in the C matrix
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
  {
    printf("computeGradients:\n");
    for (jj = 0; jj < VecHypers.length(); jj++)
      printf("  hyperparameter %3d = %e\n", jj+1, VecHypers[jj]);
  }
  constructCMatrix(MatC, VecHypers);

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
  status = MatC.computeInverse(MatCInv);
  if (status != 0)
  {
    printf("HybridGP ERROR (1): failed in matrix inversion (%d).\n",
           status);
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    {
      for (jj = 0; jj < VecHypers.length(); jj++)
        printf("  hyperparameter %3d = %e\n", jj+1, VecHypers[jj]);
    }
    for (jj = 0; jj < VecHypers.length(); jj++) VecGrads[jj] = 100;
    return 1e35;
  }
  VecX.setLength(nInputs_);
  VecY.setLength(nSamples_);
  for (jj = 0; jj < nSamples_; jj++) 
  {
    for (ii = 0; ii < nInputs_; ii++) 
      VecX[ii] = VecXDataN_[jj*nInputs_+ii];
    homRegressor_->evaluatePoint(iOne,VecX.getDVector(), &ddata,
                                 VecLegendreCoefs_.getDVector()); 
    VecY[jj] = VecYDataN_[jj] - ddata - VecHypers[1];
  }
  VecCInvY.setLength(nSamples_);
  MatCInv.matvec(VecY, VecCInvY, 0);

  //**/ =======================================================
  //**/ compute negative log likelihood wrt first length scale 
  //**/ =======================================================
  VecYT.setLength(nSamples_);
  MatCPrime.setDim(nSamples_, nSamples_);
  double *TMat = MatCPrime.getMatrix1D();
  //**/ create derivative of C with respect to first length scales
  count = 0;
  for (jj = 0; jj < nSamples_; jj++)
  {
    TMat[jj*nSamples_+jj] = 0;
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      if (hasQuantiles_)
      {
        for (ii = 0; ii < nInputs_-1; ii++)
          dist += pow(VecXDistances_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecHypers[0]);
        dist += pow(VecXDistances_[count*nInputs_+nInputs_-1],expPower_)/
                exp(expPower_*VecHypers[5]);
      }
      else
      {
        for (ii = 0; ii < nInputs_; ii++)
          dist += pow(VecXDistances_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecHypers[0]);
      }
      dist *= 0.5;
      dist = exp(VecHypers[2]) * exp(-dist);
      if (hasQuantiles_)
      {
        for (ii = 0; ii < nInputs_-1; ii++)
          ddata = pow(VecXDistances_[count*nInputs_],expPower_)/
            exp((1.0+expPower_)*VecHypers[0])*0.5*expPower_;
      }
      else
      {
        for (ii = 0; ii < nInputs_; ii++)
          ddata = pow(VecXDistances_[count*nInputs_],expPower_)/
            exp((1.0+expPower_)*VecHypers[0])*0.5*expPower_;
      }
      dist *= ddata;
      TMat[jj*nSamples_+kk] = dist;
      TMat[kk*nSamples_+jj] = dist;
      count++;
    }
  }
  //**/ compute: C' * C^{-1} (Y - mu)
  MatCPrime.matvec(VecCInvY, VecYT, 0);
  //**/ now compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  ddata = 0.0;
  for (jj = 0; jj < nSamples_; jj++) ddata += VecYT[jj] * VecCInvY[jj];
  VecGrads[0] = - ddata;
  ddata = 0.0;
  //**/ now compute trace(C^{-1} C')
  for (ii = 0; ii < nSamples_; ii++)
    for (jj = 0; jj < nSamples_; jj++) 
      ddata += MatCPrime.getEntry(ii, jj) * MatCInv.getEntry(ii, jj);
  //**/ now: trace(C^{-1} C') - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  VecGrads[0] += ddata;
  //**/ finally: 0.5[tr(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)]
  VecGrads[0] *= 0.5;

  //**/ =======================================================
  //*/ compute: - 1' * C' * C^{-1} (Y - homLeg - mu)
  //**/ =======================================================
  ddata = 0.0;
  for (jj = 0; jj < nSamples_; jj++) ddata += VecCInvY[jj];
  VecGrads[1] = - ddata;

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
      if (hasQuantiles_)
      {
        for (ii = 0; ii < nInputs_-1; ii++)
          dist += pow(VecXDistances_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecHypers[0]);
        dist += pow(VecXDistances_[count*nInputs_+ii],expPower_)/
                exp(expPower_*VecHypers[5]);
      }
      else
      {
        for (ii = 0; ii < nInputs_; ii++)
          dist += pow(VecXDistances_[count*nInputs_+ii],expPower_)/
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
  MatCPrime.matvec(VecCInvY, VecYT, 0);
  //**/ compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  ddata = 0.0;
  for (jj = 0; jj < nSamples_; jj++) ddata += VecYT[jj] * VecCInvY[jj];
  VecGrads[2] = - ddata;
  //**/ compute: trace(C^{-1} C') - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  ddata = 0.0;
  for (ii = 0; ii < nSamples_; ii++) 
    for (jj = 0; jj < nSamples_; jj++) 
      ddata += MatCInv.getEntry(ii, jj) * MatCPrime.getEntry(ii, jj);
  //**/ compute: [tr(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)]
  VecGrads[2] += ddata;
  //**/ compute: 0.5*exp(theta_1)*[tr(C^{-1}C')-(Y-mu)'C^{-1}C'*C^{-1}(Y-mu)]
  VecGrads[2] *= 0.5 * exp(VecHypers[2]);

  //**/ =======================================================
  //**/ compute negative log likelihood wrt theta_2 (offset to C)
  //**/ =======================================================
  //**/ compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  //**/          when C' is a all-1 matrix
  fudge = sqrt(1.0 * nSamples_);
  ddata = 0.0;
  for (jj = 0; jj < nSamples_; jj++) ddata += VecCInvY[jj];
  VecGrads[3] = - ddata * ddata - (fudge - 1.0) * ddata;
  //**/ compute: trace(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)
  ddata = 0.0;
  for (ii = 0; ii < nSamples_; ii++)
    for (jj = 0; jj < nSamples_; jj++) 
      ddata += MatCInv.getEntry(ii, jj);
  VecGrads[3] += ddata;
  //**/ 0.5*exp(theta_2)*[tr(C^{-1}C')-(Y-mu)'C^{-1}C'*C^{-1}(Y-mu)]
  VecGrads[3] *= 0.5 * exp(VecHypers[3]);

  //**/ =======================================================
  //**/ compute negative log likelihood wrt theta_3 
  //**/ =======================================================
  //**/ compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
  //  //**/          since CPrime is a diagonal matrix of [dist]
  ddata = 0.0;
  for (jj = 0; jj < nSamples_; jj++) 
    ddata += VecCInvY[jj] * VecCInvY[jj];
  VecGrads[4] = - ddata;
  //**/ compute: trace(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)
  ddata = 0.0;
  for (ii = 0; ii < nSamples_; ii++) 
    ddata += MatCInv.getEntry(ii, ii);
  VecGrads[4] += ddata;
  //**/ 0.5*exp(theta_3)*[tr(C^{-1}C')-(Y-mu)'C^{-1}C'*C^{-1}(Y-mu)]
  VecGrads[4] *= 0.5 * exp(VecHypers[4]);

  //**/ =======================================================
  //**/ compute negative log likelihood wrt quantile length scale 
  //**/ =======================================================
  if (hasQuantiles_)
  {
    VecYT.setLength(nSamples_);
    MatCPrime.setDim(nSamples_, nSamples_);
    TMat = MatCPrime.getMatrix1D();
    //**/ create derivative of C with respect to quantile length scale 
    count = 0;
    for (jj = 0; jj < nSamples_; jj++)
    {
      TMat[jj*nSamples_+jj] = 0;
      for (kk = jj+1; kk < nSamples_; kk++)
      {
        dist = 0.0;
        for (ii = 0; ii < nInputs_-1; ii++)
          dist += pow(VecXDistances_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecHypers[0]);
        dist += pow(VecXDistances_[count*nInputs_+nInputs_-1],expPower_)/
                exp(expPower_*VecHypers[5]);
        dist *= 0.5;
        dist = exp(VecHypers[2]) * exp(-dist);
        ddata = pow(VecXDistances_[count*nInputs_+nInputs_-1],expPower_)/
                  exp((1+expPower_)*VecHypers[5])*0.5*expPower_;
        dist *= ddata;
        TMat[jj*nSamples_+kk] = dist;
        TMat[kk*nSamples_+jj] = dist;
        count++;
      }
    }
    //**/ compute: C' * C^{-1} (Y - mu)
    MatCPrime.matvec(VecCInvY, VecYT, 0);
    //**/ now compute: - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
    ddata = 0.0;
    for (jj = 0; jj < nSamples_; jj++) ddata += VecYT[jj] * VecCInvY[jj];
    VecGrads[5] = - ddata;
    ddata = 0.0;
    //**/ now compute trace(C^{-1} C')
    for (ii = 0; ii < nSamples_; ii++)
      for (jj = 0; jj < nSamples_; jj++) 
        ddata += MatCPrime.getEntry(ii, jj) * MatCInv.getEntry(ii, jj);
    //**/ now: trace(C^{-1} C') - (Y - mu)' C^{-1} C' * C^{-1} (Y - mu)
    VecGrads[5] += ddata;
    //**/ finally: 0.5[tr(C^{-1} C')-(Y-mu)' C^{-1} C' * C^{-1} (Y-mu)]
    VecGrads[5] *= 0.5;
  }

  //**/ =======================================================
  //**/ compute negative log likelihood 
  //**/ use eigenSolve to detect non-positive definiteness of C
  //**/ =======================================================
  int errCount=0;
  MatC.eigenSolve(MatCT, VecYT, 1);
  likelihood = 0.0;
  for (jj = 0; jj < nSamples_; jj++) 
  {
    ddata = VecYT[jj];
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
      printf("HybridGP WARNING: CMatrix non-positive definite (%d)\n",
             errCount);
      for (jj = 0; jj < VecHypers.length(); jj++)
        printf("   hyperparameter %2d = %e\n",jj+1,VecHypers[jj]);
    }
  }
  for (jj = 0; jj < nSamples_; jj++) 
    likelihood += 0.5 * (VecCInvY[jj] * VecY[jj]);
  if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
    printf("   computeGradients: Likelihood = %e\n", likelihood);
  return likelihood;
}

// ************************************************************************
// compute pairwise distances 
// ------------------------------------------------------------------------
int HybridGP::computeDistances()
{
  int    ii, jj, kk, count, error=0;
  double dist;
  psVector VecLDists;

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
        printf("HybridGP ERROR: repeated sample points.\n");
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
  VecXDistances_.load(count*nInputs_, LDists);
  return error;
}

// ************************************************************************
// interpolation 
// ------------------------------------------------------------------------
int HybridGP::interpolate(int npts, double *XX, double *Y, double *Ystds)
{
  int    ii, kk, nn, iOne=1;
  double dist, expn, ddata;
  psVector VecYT, VecYT2;

  VecYT.setLength(nSamples_);
  VecYT2.setLength(nSamples_);
  for (nn = 0; nn < npts; nn++)
  {
    for (kk = 0; kk < nSamples_; kk++)
    {
      expn = 0.0;
      if (hasQuantiles_)
      {
        for (ii = 0; ii < nInputs_-1; ii++)
        {
          dist = VecXDataN_[kk*nInputs_+ii] - XX[nn*nInputs_+ii];
          if (dist < 0) dist = - dist;
          expn += pow(dist,expPower_)/exp(expPower_*VecHyperPs_[0]);
        }
        dist = VecXDataN_[kk*nInputs_+nInputs_-1] - 
               XX[nn*nInputs_+nInputs_-1];
        expn += pow(dist,expPower_)/exp(expPower_*VecHyperPs_[5]);
      }
      else
      {
        for (ii = 0; ii < nInputs_; ii++)
        {
          dist = VecXDataN_[kk*nInputs_+ii] - XX[nn*nInputs_+ii];
          if (dist < 0) dist = - dist;
          expn += pow(dist,expPower_)/exp(expPower_*VecHyperPs_[0]);
        }
      }
      expn *= 0.5;
      VecYT[kk] = exp(VecHyperPs_[2]) * exp(-expn);
    }
    homRegressor_->evaluatePoint(iOne, &(XX[nn*nInputs_]), &Y[nn],
                                 VecLegendreCoefs_.getDVector());
    Y[nn] += VecHyperPs_[1];
    for (kk = 0; kk < nSamples_; kk++)
      Y[nn] += VecYT[kk] * VecCInvY_[kk];
    if (Ystds != NULL)
    {
      Ystds[nn] = 0.0;
      ddata = exp(VecHyperPs_[2]) + exp(VecHyperPs_[3]) +
              exp(VecHyperPs_[4]);
      CMatrix_.LUSolve(VecYT, VecYT2);
      for (kk = 0; kk < nSamples_; kk++)
        ddata -= VecYT[kk] * VecYT2[kk];
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
void HybridGP::constructCMatrix(psMatrix &CMatrix, psVector VecP)
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
    TMat[jj*nSamples_+jj] = exp(VecP[2]) + 
         fudge*exp(VecP[3]) + exp(VecP[4]);
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      if (hasQuantiles_)
      {
        for (ii = 0; ii < nInputs_-1; ii++)
          dist += pow(VecXDistances_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecP[0]);
        dist += pow(VecXDistances_[count*nInputs_+nInputs_-1],expPower_)/
                exp(expPower_*VecP[5]);
      }
      else
      {
        for (ii = 0; ii < nInputs_; ii++)
          dist += pow(VecXDistances_[count*nInputs_+ii],expPower_)/
                  exp(expPower_*VecP[0]);
      }
      dist *= 0.5;
      dist = exp(VecP[2]) * exp(-dist);
      if (dist < 1.0e-50) dist = 0;
      TMat[jj*nSamples_+kk] = dist + exp(VecP[3]);
      TMat[kk*nSamples_+jj] = dist + exp(VecP[3]);
      count++;
    }
  }
  return;
}

