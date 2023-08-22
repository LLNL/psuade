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
// Functions for the class RBF
// AUTHOR : CHARLES TONG
// DATE   : 2014
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "RBF.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"
#include "PrintingTS.h"

extern "C" {
#if 0
   void dgetrf_(int *, int *, double *, int *, int *, int *);
   void dgetrs_(char *,int *,int*,double*,int*,int*,double*,int*,int*);
#endif
   void dgesvd_(char *, char *, int *, int *, double *, int *, double *,
               double *, int *, double *, int *, double *, int *, int *);
}

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

#define PS_RBF1

// ************************************************************************
// Constructor for object class RBF
// ------------------------------------------------------------------------
RBF::RBF(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  int  kernel;
  char pString[101], *cString, winput1[1000], winput2[1000];

  faID_ = PSUADE_RS_RBF;
  //**/ =======================================================
  //**/ set identifier
  //**/ 0:  multiquadratic, 1: inverse multiquadratic, 
  //**/ 2:  Gaussian, 3: thin plate spline
  //**/ =======================================================
  type_ = 0;

  //**/ =======================================================
  //**/ set internal parameters and initialize stuff
  //**/ =======================================================
  svdThresh_ = 1e-15;
  gaussScale_ = 1;

  //**/ =======================================================
  // display banner and additonal information
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,
         "*           Radial Basis Function (RBF) Analysis\n");
    printOutTS(PL_INFO,"* Set printlevel to 1-4 to see RBF details.\n");
    printOutTS(PL_INFO,"* Default kernel    = multi-quadratic \n");
    printOutTS(PL_INFO,
         "* Default threshold = 1.0e-15 (for SVD truncation)\n");
    printOutTS(PL_INFO,"* Turn on rs_expert mode to make changes.\n");
    printEquals(PL_INFO, 0);
  }
   
  //**/ =======================================================
  //**/ user adjustable parameters
  //**/ =======================================================
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("In the following you have the option to select the kernel\n");
    printf("0. multi-quadratic\n");
    printf("1. inverse multi-quadratic\n");
    printf("2. Gaussian\n");
    printf("3. thin plate spline\n");
    snprintf(pString,100,"Enter your choice (0 - 3) : ");
    type_ = getInt(0, 3, pString);
    if (type_ == 2)
    {
      snprintf(pString,100,
         "Enter scaling factor for Gaussian kernel (default=1) : ");
      gaussScale_ = getDouble(pString);
    }
    printOutTS(PL_INFO,
      "The RBF matrix to be constructed may be near-singular.\n");
    printOutTS(PL_INFO,
      "Currently, singular values < max(svd)*1e-15 are truncated.\n");
    printOutTS(PL_INFO,
      "You have the option to change this threshold (1e-15).\n");
    printOutTS(PL_INFO,
      "NOTE: truncating singular values may lead to erroneous results.\n");
    snprintf(pString,100,"Enter new threshold for SVD (> 0 but << 1) : ");
    svdThresh_ = getDouble(pString);

    snprintf(pString,100,"RBF_kernel = %d", type_);
    psConfig_.putParameter(pString);
    snprintf(pString,100,"RBF_scale = %e", gaussScale_);
    psConfig_.putParameter(pString);
    snprintf(pString,100,"RBF_thresh = %e", svdThresh_);
    psConfig_.putParameter(pString);
  }
  else
  {
    cString = psConfig_.getParameter("RBF_kernel");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &type_);
      if (type_ < 0 || type_ > 3) type_ = 0;
    }

    cString = psConfig_.getParameter("RBF_scale");
    if (cString != NULL)
      sscanf(cString, "%s %s %lg", winput1, winput2, &gaussScale_);

    cString = psConfig_.getParameter("RBF_thresh");
    if (cString != NULL)
      sscanf(cString, "%s %s %lg", winput1, winput2, &svdThresh_);
  }
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
RBF::~RBF()
{
}

// ************************************************************************
// initialize 
// ------------------------------------------------------------------------
int RBF::initialize(double *X, double *Y)
{
  int    ii, kk, ss, ss2, nSubSamples, count, nSamp1;
  double range, ddata;
  char   pString[501];
  FILE   *fp;
  psVector vecDmat;

  //**/ ---------------------------------------------------------------
  //**/ normalize the inputs and output (just to offset by its mean)
  //**/ ---------------------------------------------------------------
  if (VecLBs_.length() == 0)
  {
    printOutTS(PL_ERROR,
       "RBF initialize ERROR: sample bounds not set yet.\n");
    return -1;
  }
  VecNormalX_.setLength(nSamples_*nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
  {
    range = 1.0 / (VecUBs_[ii] - VecLBs_[ii]);
    for (ss = 0; ss < nSamples_; ss++)
      VecNormalX_[ss*nInputs_+ii] = 
         (X[ss*nInputs_+ii] - VecLBs_[ii]) * range;
  }
  VecNormalY_.setLength(nSamples_);
  initOutputScaling(Y, VecNormalY_.getDVector());
  for (ii = 0; ii < nSamples_; ii++) VecNormalY_[ii] = Y[ii] - YMean_;
  VecRanges_.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    VecRanges_[ii] = 1.0 / (VecUBs_[ii] - VecLBs_[ii]);
 
  //**/ ---------------------------------------------------------------
  //**/ construct the covariance matrix
  //**/ ---------------------------------------------------------------
#ifdef PS_RBF1
  nSamp1 = nSamples_ + 1;
#else
  nSamp1 = nSamples_;
#endif
  vecDmat.setLength(nSamp1*nSamp1);
  switch(type_) 
  {
    case 0: 
       if (outputLevel_ > 0) 
         printOutTS(PL_INFO,"Kernel = multi-quadratic\n");
       for (ss = 0; ss < nSamples_; ss++)
       {
         vecDmat[ss*nSamp1+ss] = 1.0; 
         for (ss2 = ss+1; ss2 < nSamples_; ss2++)
         {
           ddata = 0.0;
           for (ii = 0; ii < nInputs_; ii++)
             ddata += pow((VecNormalX_[ss*nInputs_+ii]-
                           VecNormalX_[ss2*nInputs_+ii]),2.0);
           vecDmat[ss*nSamp1+ss2] = 
                vecDmat[ss2*nSamp1+ss] = sqrt(ddata+1.0);
         }
       }
       break;

    case 1: 
       if (outputLevel_ > 0) 
          printOutTS(PL_INFO,"Kernel = inverse multi-quadratic\n");
       for (ss = 0; ss < nSamples_; ss++)
       {
         vecDmat[ss*nSamp1+ss] = 1.0; 
         for (ss2 = ss+1; ss2 < nSamples_; ss2++)
         {
           ddata = 0.0;
           for (ii = 0; ii < nInputs_; ii++)
             ddata += pow((VecNormalX_[ss*nInputs_+ii]-
                           VecNormalX_[ss2*nInputs_+ii]),2.0);
           vecDmat[ss*nSamp1+ss2] = 
                vecDmat[ss2*nSamp1+ss] = 1.0/sqrt(ddata+1.0);
         }
       }
       break;

    case 2: 
       if (outputLevel_ > 0) 
         printOutTS(PL_INFO,"Kernel = Gaussian\n");
       for (ss = 0; ss < nSamples_; ss++)
       {
         vecDmat[ss*nSamp1+ss] = 1.0; 
         for (ss2 = ss+1; ss2 < nSamples_; ss2++)
         {
           ddata = 0.0;
           for (ii = 0; ii < nInputs_; ii++)
             ddata += pow((VecNormalX_[ss*nInputs_+ii]-
                           VecNormalX_[ss2*nInputs_+ii]),2.0);
           vecDmat[ss*nSamp1+ss2] = 
                vecDmat[ss2*nSamp1+ss] = exp(-gaussScale_*ddata/2.0);
         }
       }
       break;

    case 3: 
       if (outputLevel_ > 0) 
         printOutTS(PL_INFO,"Kernel = thin plate spline\n");
       for (ss = 0; ss < nSamples_; ss++)
       {
         vecDmat[ss*nSamp1+ss] = 0.0; 
         for (ss2 = ss+1; ss2 < nSamples_; ss2++)
         {
           ddata = 0.0;
           for (ii = 0; ii < nInputs_; ii++)
             ddata += pow((VecNormalX_[ss*nInputs_+ii]-
                           VecNormalX_[ss2*nInputs_+ii]),2.0);
           vecDmat[ss*nSamp1+ss2] = 
             vecDmat[ss2*nSamp1+ss] = (ddata+1.0)*log(sqrt(ddata+1.0));
         }
       }
       break;
  }
#ifdef PS_RBF1
  for (ss = 0; ss < nSamples_; ss++)
    vecDmat[ss*nSamp1+nSamples_] = vecDmat[nSamples_*nSamp1+ss] = 1.0; 
  vecDmat[nSamples_*nSamp1+nSamples_] = 0.0;
#endif

  //**/ ---------------------------------------------------------------
  //**/ solve D c = YN to get the coefficients c
  //**/ ---------------------------------------------------------------
  //**/ use SVD with truncation to maintain stability
  int    info, cnt=0;
  int    wlen = 5 * nSamp1;
  char   jobu = 'A', jobvt = 'A';
  psVector vecSS, vecUU, vecVV, vecWW;
  vecSS.setLength(nSamp1);
  vecUU.setLength(nSamp1*nSamp1);
  vecVV.setLength(nSamp1*nSamp1);
  vecWW.setLength(wlen);
  dgesvd_(&jobu,&jobvt,&nSamp1,&nSamp1,vecDmat.getDVector(),&nSamp1,
          vecSS.getDVector(),vecUU.getDVector(),&nSamp1,vecVV.getDVector(), 
          &nSamp1,vecWW.getDVector(), &wlen,&info);
  if (info != 0) 
  {
    printOutTS(PL_WARN,"RBF ERROR: dgesvd returns error %d.\n",info);
    return -1;
  }
  VecRegCoeffs_.setLength(nSamp1);
  for (ii = 0; ii < nSamples_; ii++) 
    VecRegCoeffs_[ii] = VecNormalY_[ii];
#ifdef PS_RBF1
  VecRegCoeffs_[nSamples_] = 0.0;
#endif
  for (ss = 1; ss < nSamp1; ss++)
  {
    if (vecSS[ss]/vecSS[0] < svdThresh_)
    {
      vecSS[ss] = 0;
      cnt++;
    }
  }
  if (cnt > 0 && psConfig_.InteractiveIsOn() && outputLevel_ > 0) 
  {
    printOutTS(PL_WARN,
         "WARNING: RBF matrix is near-singular. Small singular values\n");
    printOutTS(PL_WARN,
         "         (%d out of %d) are truncated.\n",cnt,nSamp1);
    printOutTS(PL_WARN,"         Approximation may be inaccurate.\n");
  }
  for (ss = 0; ss < nSamp1; ss++)
  {
    vecWW[ss] = 0.0;
    for (ss2 = 0; ss2 < nSamp1; ss2++)
      vecWW[ss] += vecUU[ss*nSamp1+ss2] * VecRegCoeffs_[ss2];
  }
  for (ss = 0; ss < nSamp1; ss++) 
  {
    if (vecSS[ss] != 0) vecWW[ss] /= vecSS[ss];
    else                vecWW[ss] = 0;
  }
  for (ss = 0; ss < nSamp1; ss++)
  {
    VecRegCoeffs_[ss] = 0.0;
    for (ss2 = 0; ss2 < nSamp1; ss2++) 
      VecRegCoeffs_[ss] += vecVV[ss*nSamp1+ss2] * vecWW[ss2];
  }

  //**/ ---------------------------------------------------------------
  //**/ generate code
  //**/ ---------------------------------------------------------------
  if (!psConfig_.RSCodeGenIsOn()) return 0;
  genRSCode();
  return 0;
}
  
// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int RBF::genNDGridData(double *X,double *Y,int *NOut,double **XOut,
                       double **YOut)
{
  int totPts;
  psVector vecYOut;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(X,Y);

  //**/ ---------------------------------------------------------------
  //**/ if requested not to create mesh, just return
  //**/ ---------------------------------------------------------------
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
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  evaluatePoint(totPts, *XOut, *YOut);

  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int RBF::gen1DGridData(double *X, double *Y, int ind1, double *settings, 
                       int *NOut, double **XOut, double **YOut)
{
  int    ii, ss, totPts;
  double HX;
  psVector vecXT, vecXOut, vecYOut;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(X,Y);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  vecXOut.setLength(totPts);
  vecYOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  (*YOut) = vecYOut.takeDVector();
  (*NOut) = totPts;

  //**/ allocate local storage for the data points
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii]; 
    
  //**/ generate the data points 
  for (ss = 0; ss < totPts; ss++) 
  {
    vecXT[ss*nInputs_+ind1] = HX * ss + VecLBs_[ind1];
    (*XOut)[ss] = HX * ss + VecLBs_[ind1];
    (*YOut)[ss] = 0.0;
  }

  //**/ evaluate 
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int RBF::gen2DGridData(double *X, double *Y, int ind1, int ind2, 
                double *settings, int *NOut, double **XOut, double **YOut)
{
  int ii, ss, jj, index, totPts;
  psVector vecXT, vecXOut, vecYOut, vecHX;
 
  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(X,Y);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  vecXOut.setLength(totPts*2);
  vecYOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  (*YOut) = vecYOut.takeDVector();
  (*NOut) = totPts;

  //**/ allocate local storage for the data points
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii]; 
    
  //**/ generate the data points 
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++)
    {
      index = ii * nPtsPerDim_ + jj;
      vecXT[index*nInputs_+ind1] = vecHX[0] * ii + VecLBs_[ind1];
      vecXT[index*nInputs_+ind2] = vecHX[1] * jj + VecLBs_[ind2];
      (*XOut)[index*2]   = vecHX[0] * ii + VecLBs_[ind1];
      (*XOut)[index*2+1] = vecHX[1] * jj + VecLBs_[ind2];
    }
  }

  //**/ evaluate 
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int RBF::gen3DGridData(double *X, double *Y, int ind1, int ind2, int ind3, 
                 double *settings, int *NOut, double **XOut, double **YOut)
{
  int ii, ss, jj, ll, index, totPts;
  psVector vecXT, vecXOut, vecYOut, vecHX;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(X,Y);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  vecXOut.setLength(totPts*3);
  vecYOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  (*YOut) = vecYOut.takeDVector();
  (*NOut) = totPts;

  //**/ allocate local storage for the data points
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++)
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii];

  //**/ generate the data points 
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++)
    {
      for (ll = 0; ll < nPtsPerDim_; ll++)
      {
        index = ii * nPtsPerDim_ * nPtsPerDim_ + jj * nPtsPerDim_ + ll;
        vecXT[index*nInputs_+ind1]  = vecHX[0] * ii + VecLBs_[ind1];
        vecXT[index*nInputs_+ind2]  = vecHX[1] * jj + VecLBs_[ind2];
        vecXT[index*nInputs_+ind3]  = vecHX[2] * ll + VecLBs_[ind3];
        (*XOut)[index*3]   = vecHX[0] * ii + VecLBs_[ind1];
        (*XOut)[index*3+1] = vecHX[1] * jj + VecLBs_[ind2];
        (*XOut)[index*3+2] = vecHX[2] * ll + VecLBs_[ind3];
      }
    }
  }

  //**/ evaluate 
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int RBF::gen4DGridData(double *X, double *Y, int ind1, int ind2, int ind3, 
                       int ind4, double *settings, int *NOut, double **XOut, 
                       double **YOut)
{
  int ii, ss, jj, ll, mm, index, totPts;
  psVector vecXT, vecXOut, vecYOut, vecHX;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(X,Y);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  vecXOut.setLength(totPts*3);
  vecYOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  (*YOut) = vecYOut.takeDVector();
  (*NOut) = totPts;

  //**/ allocate local storage for the data points
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++)
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii];
    
  //**/ generate the data points 
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++)
    {
      for (ll = 0; ll < nPtsPerDim_; ll++)
      {
        for (mm = 0; mm < nPtsPerDim_; mm++)
        {
          index = ii*nPtsPerDim_*nPtsPerDim_ * nPtsPerDim_ +
                  jj*nPtsPerDim_*nPtsPerDim_ + ll*nPtsPerDim_ + mm;
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

  //**/ evaluate 
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  return 0;
}

// ************************************************************************
// Evaluate a point
// ------------------------------------------------------------------------
double RBF::evaluatePoint(double *X)
{
  int    iOne=1;
  double Y;
  evaluatePoint(1, X, &Y);
  return Y;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double RBF::evaluatePoint(int nPts, double *X, double *Y)
{
  int    ss, ss2, ii;
  double dist, Yt, ddata;

  for (ss = 0; ss < nPts; ss++) 
  {
    Yt = 0.0;
    for (ss2 = 0; ss2 < nSamples_; ss2++) 
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs_; ii++) 
      {
        ddata = X[ss*nInputs_+ii];
        ddata = (ddata - VecLBs_[ii]) * VecRanges_[ii];
        ddata -= VecNormalX_[ss2*nInputs_+ii];
        dist += ddata * ddata;
      }
      switch (type_)
      {
        case 0: dist = sqrt(dist + 1.0); break;
        case 1: dist = 1.0/sqrt(dist + 1.0); break;
        case 2: dist = exp(-0.5*dist*gaussScale_); break;
        case 3: dist = (dist+1)*log(sqrt(dist+1)); break;
      }
      Yt += dist * VecRegCoeffs_[ss2];
    }
#ifdef PS_RBF1
    Y[ss] = Yt + YMean_ + VecRegCoeffs_[nSamples_];
#else
    Y[ss] = Yt + YMean_;
#endif
  }
  return 0.0;
}

// ************************************************************************
// Evaluate a given point with standard deviation
// ------------------------------------------------------------------------
double RBF::evaluatePointFuzzy(double *X, double &std)
{
  int    iOne=1;
  double Y=0.0;
  evaluatePoint(iOne, X, &Y);
  std = 0.0;
  return Y;
}

// ************************************************************************
// Evaluate a number of points with standard deviations
// ------------------------------------------------------------------------
double RBF::evaluatePointFuzzy(int npts, double *X, double *Y, double *Ystd)
{
  evaluatePoint(npts, X, Y);
  for (int ss = 0; ss < npts; ss++) Ystd[ss] = 0.0;
  return 0.0;
}

// ************************************************************************
// generate C and Python codes
// ------------------------------------------------------------------------
void RBF::genRSCode()
{
  int  ii, ss;
  FILE *fp = fopen("psuade_rs.info", "w");
#ifdef PS_RBF1
  int nSamp1 = nSamples_ + 1;
#else
  int nSamp1 = nSamples_;
#endif
  if (fp != NULL)
  {
    fprintf(fp,"/* *********************************************/\n");
    fprintf(fp,"/* RBF interpolator from PSUADE.       */\n");
    fprintf(fp,"/* ============================================*/\n");
    fprintf(fp,"/* This file contains information for interpolation\n");
    fprintf(fp,"   using response surface. Follow the steps below:\n");
    fprintf(fp,"   1. move this file to *.c file (e.g. main.c)\n");
    fprintf(fp,"   2. Compile main.c (cc -o main main.c -lm) \n");
    fprintf(fp,"   3. run: main input output\n");
    fprintf(fp,"          where input has the number of inputs and\n");
    fprintf(fp,"          the input values\n");
    fprintf(fp,"*/\n");
    fprintf(fp,"/* ==========================================*/\n");
    fprintf(fp,"int nSamples = %d;\n",nSamples_);
    fprintf(fp,"int nInps = %d;\n",nInputs_);
    fprintf(fp,"static double\n");
    fprintf(fp,"LBs[%d] = \n", nInputs_);
    fprintf(fp,"{\n");
    for (ii = 0; ii < nInputs_; ii++)
      fprintf(fp,"  %24.16e ,\n", VecLBs_[ii]);
    fprintf(fp,"};\n");
    fprintf(fp,"static double\n");
    fprintf(fp,"UBs[%d] = \n", nInputs_);
    fprintf(fp,"{\n");
    for (ii = 0; ii < nInputs_; ii++)
      fprintf(fp,"  %24.16e ,\n", VecUBs_[ii]);
    fprintf(fp,"};\n");
    fprintf(fp,"static double\n");
    fprintf(fp,"Coefs[%d] = \n", nSamp1);
    fprintf(fp,"{\n");
    for (ss = 0; ss < nSamples_; ss++)
      fprintf(fp,"  %24.16e ,\n", VecRegCoeffs_[ss]);
#ifdef PS_RBF1
    fprintf(fp,"  %24.16e };\n", VecRegCoeffs_[nSamples_]);
#endif
    fprintf(fp,"static double\n");
    fprintf(fp,"XN[%d][%d] = \n", nSamples_, nInputs_);
    fprintf(fp,"{\n");
    for (ss = 0; ss < nSamples_; ss++)
    {
      fprintf(fp,"   { ");
      for (ii = 0; ii < nInputs_-1; ii++)
        fprintf(fp,"  %24.16e ,", VecNormalX_[ss*nInputs_+ii]);
      fprintf(fp,"  %24.16e },\n", VecNormalX_[ss*nInputs_+nInputs_-1]);
    }
    fprintf(fp,"};\n");
    fprintf(fp,"double YMean = %e;\n",YMean_);
    fprintf(fp,"/* *********************************************/\n");
    fprintf(fp,"/* RBF interpolator from PSUADE.       */\n");
    fprintf(fp,"/* ==========================================*/\n");
    fprintf(fp,"#include <math.h>\n");
    fprintf(fp,"#include <stdlib.h>\n");
    fprintf(fp,"#include <stdio.h>\n");
    fprintf(fp,"int interpolate(int,double*,double*);\n");
    fprintf(fp,"main(int argc, char **argv) {\n");
    fprintf(fp,"  int    i, iOne=1, nInps;\n");
    fprintf(fp,"  double X[%d], Y, S;\n",nInputs_);
    fprintf(fp,"  FILE   *fIn=NULL, *fOut=NULL;\n");
    fprintf(fp,"  if (argc < 3) {\n");
    fprintf(fp,"     printf(\"ERROR: not enough argument.\\n\");\n");
    fprintf(fp,"     exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fIn = fopen(argv[1], \"r\");\n");
    fprintf(fp,"  if (fIn == NULL) {\n");
    fprintf(fp,"     printf(\"ERROR: cannot open input file.\\n\");\n");
    fprintf(fp,"     exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fscanf(fIn, \"%%d\", &nInps);\n");
    fprintf(fp,"  if (nInps != %d) {\n", nInputs_);
    fprintf(fp,"    printf(\"ERROR - wrong nInputs.\\n\");\n");
    fprintf(fp,"    exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  for (i=0; i<nInps; i++) fscanf(fIn, \"%%lg\", &X[i]);\n");
    fprintf(fp,"  fclose(fIn);\n");
    fprintf(fp,"  interpolate(iOne, X, &Y);\n");
    fprintf(fp,"  fOut = fopen(argv[2], \"w\");\n");
    fprintf(fp,"  if (fOut == NULL) {\n");
    fprintf(fp,"     printf(\"ERROR: cannot open output file.\\n\");\n");
    fprintf(fp,"     exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fprintf(fOut,\" %%e\\n\", Y);\n");
    fprintf(fp,"  fclose(fOut);\n");
    fprintf(fp,"}\n\n");
    fprintf(fp,"/* *************************************/\n");
    fprintf(fp,"/*  interpolation function             */\n");
    fprintf(fp,"/* X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp," * X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp," * ... */\n");
    fprintf(fp,"/* ====================================*/\n");
    fprintf(fp,"int interpolate(int npts,double *X,double *Y){\n");
    fprintf(fp,"  int    ss, ss2, ii;\n");
    fprintf(fp,"  double dist, dd;\n");
    fprintf(fp,"  for (ss2 = 0; ss2 < npts; ss2++) {\n");
    fprintf(fp,"    Y[ss2] = 0;\n");
    fprintf(fp,"    for (ss = 0; ss < nSamples; ss++) {\n");
    fprintf(fp,"      dist = 0.0;\n");
    fprintf(fp,"      for (ii = 0; ii < nInps; ii++) {\n");
    fprintf(fp,"        dd = X[ii+ss2*nInps];\n");
    fprintf(fp,"        dd = (dd-LBs[ii])/(UBs[ii]-LBs[ii]);\n");
    fprintf(fp,"        dd = dd - XN[ss][ii];\n");
    fprintf(fp,"        dist += dd * dd;\n");
    fprintf(fp,"      }\n");
    switch (type_) 
    {
      case 0: 
         fprintf(fp,"      dist = sqrt(dist+1);\n");
         break;
      case 1: 
         fprintf(fp,"      dist = 1.0/sqrt(dist+1);\n");
         break;
      case 2: 
         fprintf(fp,"      dist = exp(-0.5*dist*%e);\n",gaussScale_);
         break;
      case 3: 
         fprintf(fp,"      dist = (dist+1)*log(sqrt(dist+1));\n");
         break;
    }
    fprintf(fp,"      Y[ss2] += dist * Coefs[ss];\n");
    fprintf(fp,"    }\n");
#ifdef PS_RBF1
    fprintf(fp,"    Y[ss2] = Y[ss2] + YMean + Coefs[nSamples];\n");
#else
    fprintf(fp,"    Y[ss2] = Y[ss2] + YMean;\n");
#endif
    fprintf(fp,"  }\n");
    fprintf(fp,"  return 0;\n");
    fprintf(fp,"}\n");
    fclose(fp);
  }
  fp = fopen("psuade_rs.py", "w");
  if (fp != NULL)
  {
    fwriteRSPythonHeader(fp);
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"# RBF regression interpolation\n");
    fprintf(fp,"#==================================================\n");
    fwriteRSPythonCommon(fp);
    fprintf(fp,"nSamples = %d;\n",nSamples_);
    fprintf(fp,"nInps = %d;\n",nInputs_);
    fprintf(fp,"LBs = [\n");
    for (ii = 0; ii < nInputs_; ii++)
      fprintf(fp," %24.16e ,\n", VecLBs_[ii]);
    fprintf(fp,"]\n");
    fprintf(fp,"UBs = [\n");
    for (ii = 0; ii < nInputs_; ii++)
      fprintf(fp," %24.16e ,\n", VecUBs_[ii]);
    fprintf(fp,"]\n");
    fprintf(fp,"Coefs = [\n");
    for (ss = 0; ss <= nSamples_; ss++)
    {
      fprintf(fp,"  %24.16e , \n", VecRegCoeffs_[ss]);
    }
    fprintf(fp,"]\n");
    fprintf(fp,"XN = [\n");
    for (ss = 0; ss < nSamples_; ss++)
    {
      fprintf(fp,"   [ ");
      for (ii = 0; ii < nInputs_-1; ii++)
        fprintf(fp,"  %24.16e ,", VecNormalX_[ss*nInputs_+ii]);
      fprintf(fp,"  %24.16e ],\n", VecNormalX_[ss*nInputs_+nInputs_-1]);
    }
    fprintf(fp,"]\n");
    fprintf(fp,"YMean = %e\n", YMean_);
    fprintf(fp,"###################################################\n");
    fprintf(fp,"# interpolation function  \n");
    fprintf(fp,"# X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp,"# X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp,"# ... \n");
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"def interpolate(XX): \n");
    fprintf(fp,"  npts = int(len(XX) / nInps + 1.0e-8)\n");
    fprintf(fp,"  Ys = (2 * npts) * [0.0]\n");
    fprintf(fp,"  X  = nInps * [0.0]\n");
    fprintf(fp,"  for nn in range(npts) : \n");
    fprintf(fp,"    for ii in range(nInps) : \n");
    fprintf(fp,"      X[ii] = XX[nn*nInps+ii]\n");
    fprintf(fp,"    for ss in range(nSamples) : \n");
    fprintf(fp,"      dist = 0.0\n");
    fprintf(fp,"      for ii in range(nInps) : \n");
    fprintf(fp,"        dd = (X[ii]-LBs[ii])/(UBs[ii]-LBs[ii])\n");
    fprintf(fp,"        dd = dd - XN[ss][ii]\n");
    fprintf(fp,"        dist += dd * dd\n");
    switch (type_) 
    {
      case 0: 
         fprintf(fp,"      dist = math.sqrt(dist+1)\n");
         break;
      case 1: 
         fprintf(fp,"      dist = 1.0/math.sqrt(dist+1)\n");
         break;
      case 2: 
         fprintf(fp,"      dist = math.exp(-0.5*dist*%e)\n",gaussScale_);
         break;
      case 3: 
         fprintf(fp,"      dist = (dist+1)*math.log(math.sqrt(dist+1))\n");
         break;
    }
    fprintf(fp,"      Ys[nn] = Ys[nn] + Coefs[ss] * dist\n");
#ifdef PS_RBF1
    fprintf(fp,"    Ys[nn] = Ys[nn] + YMean + Coefs[nSamples]\n");
#else
    fprintf(fp,"    Ys[nn] = Ys[nn] + YMean\n");
#endif
    fprintf(fp,"  return Ys\n");
    fprintf(fp,"###################################################\n");
    fprintf(fp,"# main program\n");
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"infileName  = sys.argv[1]\n");
    fprintf(fp,"outfileName = sys.argv[2]\n");
    fprintf(fp,"inputs = getInputData(infileName)\n");
    fprintf(fp,"outputs = interpolate(inputs)\n");
    fprintf(fp,"genOutputFile(outfileName, outputs)\n");
    fprintf(fp,"###################################################\n");
    printf("FILE psuade_rs.py contains the final RBF functional form.\n");
    fclose(fp);
  }
}

