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
// Functions for the class KNN
// AUTHOR : CHARLES TONG
// DATE   : 2013
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "KNN.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// Constructor for object class KNN
// ------------------------------------------------------------------------
KNN::KNN(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  char pString[501];

  //**/ set identifier
  faID_ = PSUADE_RS_KNN;
  //**/ 0: linear combination, 1: mode
  mode_ = 0;

  //**/ set number of nearest neighbors
  k_     = 0;
  kmax_  = (nSamples < 10) ? nSamples : 10;
  kfold_ = (nSamples < 100) ? nSamples : 100;
  VecDistances_.setLength(kmax_);
  VecYStored_.setLength(kmax_);

  //**/ =======================================================
  // display banner and additonal information
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printf("*           K-nearest neighbors Analysis\n");
    printf("* Set printlevel to 1-4 to see KNN details.\n");
    printf("* Default number of neighbors = to be searched based\n");
    printf("*         on %d-fold cross validation.\n",kfold_);
    printf("* Turn on rs_expert mode to set internal parameters.\n");
    printEquals(PL_INFO, 0);
  }
   
  //**/ =======================================================
  //**/ user adjustable parameters
  //**/ =======================================================
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("In the following you have the option to select K. But if\n");
    printf("you want it to be selected automatically, enter option 0.\n");
    sprintf(pString,"Enter number of nearest neighbors (>= 0, <= %d): ",
            kmax_);
    k_ = getInt(0, kmax_, pString);
    if (k_ == 0)
    {
      sprintf(pString,"Maximum K to be searched (>1,<=%d): ",kmax_);
      kmax_ = getInt(2, kmax_, pString);
      sprintf(pString,"How many fold cross validation (10-%d)? ",kfold_);
      kfold_ = getInt(10, kfold_, pString);
    }
    printf("There are two options for interpolation: \n");
    printf("0: linear weightings of neighboring points.\n");
    printf("1: mode (for integer labels of outputs, classification.\n");
    sprintf(pString,"Enter desired mode (0 or 1): ");
    mode_ = getInt(0, 1, pString);
    sprintf(pString, "KNN_mode = %d", mode_);
    psConfig_.putParameter(pString);
  }
  else
  {
    char keyword[1000], equalSign[100];
    char *cString = psConfig_.getParameter("KNN_mode");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", keyword, equalSign, &mode_);
      if (mode_ != 0 && mode_ != 1) mode_ = 0;
    }
  }
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
KNN::~KNN()
{
}

// ************************************************************************
// initialize 
// ------------------------------------------------------------------------
int KNN::initialize(double *X, double *Y)
{
  int    ii, kk, ss, ss2, nSubSamples, count;
  double errMin, error, range;
  psVector VecXX, VecX2, VecYY, VecY2;
  psIVector VecIT;

  //**/ ---------------------------------------------------------------
  //**/ normalize
  //**/ ---------------------------------------------------------------
  if (VecLBs_.length() == 0)
  {
    printf("KNN initialize ERROR: sample bounds not set yet.\n");
    return -1;
  }
  VecNormalX_.setLength(nSamples_*nInputs_);
  VecNormalY_.setLength(nSamples_*nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
  {
    range = 1.0 / (VecUBs_[ii] - VecLBs_[ii]);
    for (ss = 0; ss < nSamples_; ss++)
      VecNormalX_[ss*nInputs_+ii] = 
         (X[ss*nInputs_+ii] - VecLBs_[ii]) * range;
  }
  for (ss = 0; ss < nSamples_; ss++) VecNormalY_[ss] = Y[ss];
  VecRanges_.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    VecRanges_[ii] = 1.0 / (VecUBs_[ii] - VecLBs_[ii]);

  //**/ ---------------------------------------------------------------
  //**/ searching for best k, if it has not been selected 
  //**/ ---------------------------------------------------------------
  if (k_ == 0)
  {
    VecIT.setLength(nSamples_);
    VecXX.setLength(nSamples_*nInputs_);
    VecYY.setLength(nSamples_);
    double *arrayXX = VecXX.getDVector();
    double *arrayYY = VecYY.getDVector();
    generateRandomIvector(nSamples_, VecIT.getIVector());
    for (ii = 0; ii < nInputs_; ii++)
    {
      for (ss = 0; ss < nSamples_; ss++)
        VecXX[VecIT[ss]*nInputs_+ii] = VecNormalX_[ss*nInputs_+ii];
    }
    for (ss = 0; ss < nSamples_; ss++) VecYY[VecIT[ss]] = Y[ss];

    //**/ get ready for search
    VecX2.setLength(nSamples_*nInputs_);
    VecY2.setLength(nSamples_);
    nSubSamples = nSamples_ / kfold_;
    if (nSubSamples == 0) nSubSamples = 1;
    errMin = 1e35;
    for (kk = 1; kk <= kmax_; kk++)
    {
      error = 0.0;
      //**/ for each of the k folds
      for (ss = 0; ss < nSamples_; ss+=nSubSamples)
      {
        //**/ put the training set into (VecX2, VecY2)
        for (ss2 = 0; ss2 < ss*nInputs_; ss2++) VecX2[ss2] = VecXX[ss2];
        for (ss2 = 0; ss2 < ss; ss2++) VecY2[ss2] = VecYY[ss2];
        count = ss;
        for (ss2 = ss+nSubSamples; ss2 < nSamples_; ss2++)
        {
          for (ii = 0; ii < nInputs_; ii++)
            VecX2[(ss2-nSubSamples)*nInputs_+ii] = VecXX[ss2*nInputs_+ii];
          count++;
        }
        //**/ train and test it on the hold out
        error += train(count, VecX2.getDVector(), VecY2.getDVector(), kk, 
                       nSamples_-count, &arrayXX[ss*nInputs_],&arrayYY[ss]);
      }
      if (error < errMin)
      {
        k_ = kk;
        errMin = error;
      }
      if (psConfig_.InteractiveIsOn())
        printf("K=%d (%d) : error = %e\n", kk, kmax_, error);
    }
    if (psConfig_.InteractiveIsOn()) printf("KNN: K selected = %d\n", k_);
  }
  if (!psConfig_.RSCodeGenIsOn()) return 0;
  genRSCode();
  return 0;
}
  
// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int KNN::genNDGridData(double *XIn,double *YIn,int *NOut,double **XOut,
                       double **YOut)
{
  int totPts;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

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
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  evaluatePoint(totPts, *XOut, *YOut);

  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int KNN::gen1DGridData(double *XIn, double *YIn,int ind1,double *settings, 
                       int *NOut, double **XOut, double **YOut)
{
  int    ii, ss, totPts;
  double HX;
  psVector VecXT;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts);
  VecYOut.setLength(totPts);
  (*XOut) = VecXOut.takeDVector();
  (*YOut) = VecYOut.takeDVector();
  (*NOut) = totPts;

  //**/ allocate local storage for the data points
  VecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) 
      VecXT[ss*nInputs_+ii] = settings[ii]; 
    
  //**/ generate the data points 
  for (ss = 0; ss < totPts; ss++) 
  {
    VecXT[ss*nInputs_+ind1]  = HX * ss + VecLBs_[ind1];
    (*XOut)[ss] = HX * ss + VecLBs_[ind1];
    (*YOut)[ss] = 0.0;
  }

  //**/ evaluate 
  evaluatePoint(totPts, VecXT.getDVector(), (*YOut));

  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int KNN::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                       double *settings, int *NOut, double **XOut, 
                       double **YOut)
{
  int ii, ss, jj, index, totPts;
  psVector VecXT, VecHX;
 
  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_;
  VecHX.setLength(2);
  VecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  VecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts*2);
  VecYOut.setLength(totPts);
  (*XOut) = VecXOut.takeDVector();
  (*YOut) = VecYOut.takeDVector();
  (*NOut) = totPts;

  //**/ allocate local storage for the data points
  VecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) VecXT[ss*nInputs_+ii] = settings[ii]; 
    
  //**/ generate the data points 
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++)
    {
      index = ii * nPtsPerDim_ + jj;
      VecXT[index*nInputs_+ind1] = VecHX[0] * ii + VecLBs_[ind1];
      VecXT[index*nInputs_+ind2] = VecHX[1] * jj + VecLBs_[ind2];
      (*XOut)[index*2]   = VecHX[0] * ii + VecLBs_[ind1];
      (*XOut)[index*2+1] = VecHX[1] * jj + VecLBs_[ind2];
    }
  }

  //**/ evaluate 
  evaluatePoint(totPts, VecXT.getDVector(), *YOut);

  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int KNN::gen3DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3, 
                       double *settings, int *NOut, double **XOut, 
                       double **YOut)
{
  int ii, ss, jj, ll, index, totPts;
  psVector VecXT, VecHX;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  VecHX.setLength(3);
  VecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  VecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  VecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts*3);
  VecYOut.setLength(totPts);
  (*XOut) = VecXOut.takeDVector();
  (*YOut) = VecYOut.takeDVector();
  (*NOut) = totPts;

  //**/ allocate local storage for the data points
  VecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++)
    for (ii = 0; ii < nInputs_; ii++) 
      VecXT[ss*nInputs_+ii] = settings[ii];

  //**/ generate the data points 
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++)
    {
      for (ll = 0; ll < nPtsPerDim_; ll++)
      {
        index = ii * nPtsPerDim_ * nPtsPerDim_ + jj * nPtsPerDim_ + ll;
        VecXT[index*nInputs_+ind1] = VecHX[0] * ii + VecLBs_[ind1];
        VecXT[index*nInputs_+ind2] = VecHX[1] * jj + VecLBs_[ind2];
        VecXT[index*nInputs_+ind3] = VecHX[2] * ll + VecLBs_[ind3];
        (*XOut)[index*3]   = VecHX[0] * ii + VecLBs_[ind1];
        (*XOut)[index*3+1] = VecHX[1] * jj + VecLBs_[ind2];
        (*XOut)[index*3+2] = VecHX[2] * ll + VecLBs_[ind3];
      }
    }
  }

  //**/ evaluate 
   evaluatePoint(totPts, VecXT.getDVector(), *YOut);

  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int KNN::gen4DGridData(double *XIn,double *YIn,int ind1,int ind2,int ind3, 
                       int ind4,double *settings, int *NOut, double **XOut, 
                       double **YOut)
{
  int    ii, ss, jj, ll, mm, index, totPts;
  psVector VecXT, VecHX;

  //**/ ---------------------------------------------------------------
  //**/ initialization
  //**/ ---------------------------------------------------------------
  initialize(XIn,YIn);

  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  VecHX.setLength(4);
  VecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  VecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  VecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 
  VecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  psVector VecXOut, VecYOut;
  VecXOut.setLength(totPts*4);
  VecYOut.setLength(totPts);
  (*XOut) = VecXOut.takeDVector();
  (*YOut) = VecYOut.takeDVector();
  (*NOut) = totPts;

  //**/ allocate local storage for the data points
  VecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) 
      VecXT[ss*nInputs_+ii] = settings[ii]; 
    
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
          VecXT[index*nInputs_+ind1] = VecHX[0]*ii + VecLBs_[ind1];
          VecXT[index*nInputs_+ind2] = VecHX[1]*jj + VecLBs_[ind2];
          VecXT[index*nInputs_+ind3] = VecHX[2]*ll + VecLBs_[ind3];
          VecXT[index*nInputs_+ind4] = VecHX[3]*mm + VecLBs_[ind4];
          (*XOut)[index*4]   = VecHX[0] * ii + VecLBs_[ind1];
          (*XOut)[index*4+1] = VecHX[1] * jj + VecLBs_[ind2];
          (*XOut)[index*4+2] = VecHX[2] * ll + VecLBs_[ind3];
          (*XOut)[index*4+3] = VecHX[3] * mm + VecLBs_[ind4];
        }
      }
    }
  }

  //**/ evaluate 
  evaluatePoint(totPts, VecXT.getDVector(), *YOut);
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double KNN::evaluatePoint(int nSamp, double *XN, double *YN, double *X,
                          int normFlag,int knn)
{
  int    ss, ii, kk, count=0, cnt, maxcnt;
  double dist, sumDist, ddata, Y=0.0, ylabel;

  for (ss = 0; ss < nSamp; ss++) 
  {
    dist = 0.0;
    for (ii = 0; ii < nInputs_; ii++) 
    {
      ddata = X[ii];
      if (normFlag == 1)
        ddata = (ddata - VecLBs_[ii]) * VecRanges_[ii];
      ddata -= XN[ss*nInputs_+ii];
      dist += ddata * ddata;
      if (count > 0 && count < knn && dist > VecDistances_[count-1]) break; 
    }
    if (dist == 0.0)
    {
      Y = YN[ss];
      return Y;
    }
    if (count < knn)
    {
      VecDistances_[count] = dist;
      VecYStored_[count] = YN[ss];
      count++;
      for (kk = count-1; kk > 0; kk--) 
      {
        if (VecDistances_[kk] < VecDistances_[kk-1])
        {
          ddata = VecDistances_[kk];
          VecDistances_[kk] = VecDistances_[kk-1];
          VecDistances_[kk-1] = ddata;
          ddata = VecYStored_[kk];
          VecYStored_[kk] = VecYStored_[kk-1];
          VecYStored_[kk-1] = ddata;
        } 
      } 
    }
    else
    {
      if (dist < VecDistances_[count-1])
      {
        VecDistances_[count-1] = dist;
        VecYStored_[count-1] = YN[ss];
        for (kk = count-1; kk > 0; kk--) 
        {
          if (VecDistances_[kk] < VecDistances_[kk-1])
          {
            ddata = VecDistances_[kk];
            VecDistances_[kk] = VecDistances_[kk-1];
            VecDistances_[kk-1] = ddata;
            ddata = VecYStored_[kk];
            VecYStored_[kk] = VecYStored_[kk-1];
            VecYStored_[kk-1] = ddata;
          } 
        } 
      } 
    } 
  }
  if (mode_ == 0)
  {
    Y = sumDist = 0.0;
    for (ss = 0; ss < count; ss++) 
    {
      Y += VecYStored_[ss] / VecDistances_[ss];
      sumDist += 1.0 / VecDistances_[ss];
    }
    Y /= sumDist;
  }
  else
  {
    sortDbleList(count, VecYStored_.getDVector());
    maxcnt = 0;
    cnt = 1;
    for (ss = 1; ss < count; ss++) 
    {
      if (VecYStored_[ss] == VecYStored_[ss-1]) cnt++;
      else
      {
        if (cnt > maxcnt)
        {
          maxcnt = cnt;
          cnt = 1;
          ylabel = VecYStored_[ss-1];
        }
      } 
    }
    Y = ylabel;
  }
  return Y;
}

// ************************************************************************
// Evaluate a point
// ------------------------------------------------------------------------
double KNN::evaluatePoint(double *X)
{
  return evaluatePoint(nSamples_, VecNormalX_.getDVector(), 
                       VecNormalY_.getDVector(), X, 1, k_);
}

// ************************************************************************
// Evaluate a number of points
// ------------------------------------------------------------------------
double KNN::evaluatePoint(int npts, double *X, double *Y)
{
  for (int ss = 0; ss < npts; ss++)
    Y[ss] = evaluatePoint(nSamples_, VecNormalX_.getDVector(), 
                    VecNormalY_.getDVector(), &(X[ss*nInputs_]), 1, k_);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point with standard deviation
// ------------------------------------------------------------------------
double KNN::evaluatePointFuzzy(double *X, double &std)
{
  double Y=0.0;
  Y = evaluatePoint(nSamples_, VecNormalX_.getDVector(), 
                    VecNormalY_.getDVector(), X, 1, k_);
  std = 0.0;
  return Y;
}

// ************************************************************************
// Evaluate a number of points with standard deviations
// ------------------------------------------------------------------------
double KNN::evaluatePointFuzzy(int npts, double *X, double *Y, double *Ystd)
{
  evaluatePoint(npts, X, Y);
  for (int ss = 0; ss < npts; ss++) Ystd[ss] = 0.0;
  return 0.0;
}

// ************************************************************************
// initialize the trees
// ------------------------------------------------------------------------
double KNN::train(int nSamp, double *X, double *Y, int knn, int nTests, 
                  double *XTest, double *YTest)
{
  int    ii;
  double ddata, YEst, error;

  error = 0.0;
  for (ii = 0; ii < nTests; ii++)
  {
    YEst = evaluatePoint(nSamp, X, Y, &XTest[ii*nInputs_],0,knn);
    ddata = YTest[ii] - YEst;
    error += ddata * ddata;
  }
  return error;
}

// ************************************************************************
// generate codes
// ------------------------------------------------------------------------
void KNN::genRSCode()
{
  int  ii, ss;
  FILE *fp = fopen("psuade_rs.info", "w");
  if (fp != NULL)
  {
    fprintf(fp,"/* *************************************/\n");
    fprintf(fp,"/* KNN interpolator from PSUADE.       */\n");
    fprintf(fp,"/* ====================================*/\n");
    fprintf(fp,"/* This file contains information for interpolation\n");
    fprintf(fp,"   using response surface. Follow the steps below:\n");
    fprintf(fp,"   1. move this file to *.c file (e.g. main.c)\n");
    fprintf(fp,"   2. Modify the main.c program\n");
    fprintf(fp,"      a. replace func with your user-defined function\n"); 
    fprintf(fp,"   3. Compile main.c (cc -o main main.c -lm) \n");
    fprintf(fp,"   4. run: main input output\n");
    fprintf(fp,"          where input has the number of inputs and\n");
    fprintf(fp,"          the input values\n");
    fprintf(fp,"*/\n");
    fprintf(fp,"/* ==========================================*/\n");
    fprintf(fp,"int nSamples = %d;\n",nSamples_);
    fprintf(fp,"int nInps = %d;\n",nInputs_);
    fprintf(fp,"int K = %d;\n",k_);
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
    fprintf(fp,"Sample[%d][%d] = \n", nSamples_, nInputs_);
    fprintf(fp,"{\n");
    for (ss = 0; ss < nSamples_; ss++)
    {
      fprintf(fp," { %24.16e ", VecNormalX_[ss*nInputs_]);
      for (ii = 1; ii < nInputs_; ii++)
        fprintf(fp,", %24.16e ", VecNormalX_[ss*nInputs_+ii]);
      fprintf(fp,"},\n");
    }
    fprintf(fp,"};\n");
    fprintf(fp,"static double\n");
    fprintf(fp,"SamOut[%d] = \n", nSamples_);
    fprintf(fp,"{\n");
    for (ss = 0; ss < nSamples_; ss++)
      fprintf(fp,"  %24.16e ,\n", VecNormalY_[ss]);
    fprintf(fp,"};\n");
    fprintf(fp,"/* *************************************/\n");
    fprintf(fp,"/* KNN interpolator from PSUADE.       */\n");
    fprintf(fp,"/* ====================================*/\n");
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
    fprintf(fp,"  for (i=0; i<%d; i++) fscanf(fIn, \"%%lg\", &X[i]);\n",
            nInputs_);
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
    fprintf(fp,"  int    ss, ss2, ii, cnt, kk;\n");
    fprintf(fp,"  double y, dist, dd, *Dists=NULL, *YT=NULL;\n");
    fprintf(fp,"  Dists = (double *) malloc(nSamples*sizeof(double));\n");
    fprintf(fp,"  YT = (double *) malloc(2*K*sizeof(double));\n");
    fprintf(fp,"  for (ss2 = 0; ss2 < npts; ss2++) {\n");
    fprintf(fp,"    cnt = 0;\n");
    fprintf(fp,"    for (ss = 0; ss < nSamples; ss++) {\n");
    fprintf(fp,"      dist = 0.0;\n");
    fprintf(fp,"      for (ii = 0; ii < nInps; ii++) {\n");
    fprintf(fp,"        dd = X[ii];\n");
    fprintf(fp,"        dd = (dd-LBs[ii])/(UBs[ii]-LBs[ii]);\n");
    fprintf(fp,"        dd = dd - Sample[ss][ii];\n");
    fprintf(fp,"        dist += dd * dd;\n");
    fprintf(fp,"        if (cnt>0 & cnt<K & dist>Dists[cnt-1]) break;\n");
    fprintf(fp,"      }\n");
    fprintf(fp,"      if (dist == 0.0) \n");
    fprintf(fp,"        Y[ss] = SamOut[ss];\n");
    fprintf(fp,"      else {\n"); 
    fprintf(fp,"        if (cnt < K) {\n");
    fprintf(fp,"          Dists[cnt] = dist;\n");
    fprintf(fp,"          YT[cnt] = SamOut[ss];\n");
    fprintf(fp,"          cnt++;\n");
    fprintf(fp,"          for (kk=cnt-1; kk>0; kk--) {\n");
    fprintf(fp,"            if (Dists[kk] < Dists[kk-1]) {\n");
    fprintf(fp,"              dd = Dists[kk];\n");
    fprintf(fp,"              Dists[kk] = Dists[kk-1];\n");
    fprintf(fp,"              Dists[kk-1] = dd;\n");
    fprintf(fp,"              dd = YT[kk];\n");
    fprintf(fp,"              YT[kk] = YT[kk-1];\n");
    fprintf(fp,"              YT[kk-1] = dd;\n");
    fprintf(fp,"            }\n");
    fprintf(fp,"          }\n");
    fprintf(fp,"        }\n");
    fprintf(fp,"        else {\n"); 
    fprintf(fp,"          if (dist < Dists[cnt-1]) {\n"); 
    fprintf(fp,"            Dists[cnt-1] = dist;\n"); 
    fprintf(fp,"            YT[cnt-1] = SamOut[ss];\n"); 
    fprintf(fp,"            for (kk=cnt-1; kk>0; kk--) {\n");
    fprintf(fp,"              if (Dists[kk] < Dists[kk-1]) {\n");
    fprintf(fp,"                dd = Dists[kk];\n");
    fprintf(fp,"                Dists[kk] = Dists[kk-1];\n");
    fprintf(fp,"                Dists[kk-1] = dd;\n");
    fprintf(fp,"                dd = YT[kk];\n");
    fprintf(fp,"                YT[kk] = YT[kk-1];\n");
    fprintf(fp,"                YT[kk-1] = dd;\n");
    fprintf(fp,"              }\n");
    fprintf(fp,"            }\n");
    fprintf(fp,"          }\n");
    fprintf(fp,"        }\n");
    fprintf(fp,"      }\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    dd = 0.0;\n");
    fprintf(fp,"    Y[ss2] = 0.0;\n");
    fprintf(fp,"    for (ss = 0; ss < cnt; ss++) {\n");
    fprintf(fp,"      Y[ss2] += YT[ss] / Dists[ss];\n");
    fprintf(fp,"      dd += 1.0 / Dists[ss];\n");
    fprintf(fp,"    }\n");
    fprintf(fp,"    Y[ss2] /= dd;\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  if (Dists != NULL) free(Dists);\n");
    fprintf(fp,"  if (YT != NULL) free(YT);\n");
    fprintf(fp,"  return 0;\n");
    fprintf(fp,"}\n");
    fclose(fp);
  }
  fp = fopen("psuade_rs.py", "w");
  if (fp != NULL)
  {
    fwriteRSPythonHeader(fp);
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"# KNN Regression interpolation\n");
    fprintf(fp,"#==================================================\n");
    fwriteRSPythonCommon(fp);
    fprintf(fp,"nSamples = %d;\n",nSamples_);
    fprintf(fp,"nInps = %d;\n",nInputs_);
    fprintf(fp,"K = %d;\n",k_);
    fprintf(fp,"LBs = [\n");
    for (ii = 0; ii < nInputs_; ii++)
      fprintf(fp," %24.16e ,\n", VecLBs_[ii]);
    fprintf(fp,"]\n");
    fprintf(fp,"UBs = [\n");
    for (ii = 0; ii < nInputs_; ii++)
      fprintf(fp," %24.16e ,\n", VecUBs_[ii]);
    fprintf(fp,"]\n");
    fprintf(fp,"Sample = [\n");
    for (ss = 0; ss < nSamples_; ss++)
    {
      fprintf(fp," [ %24.16e ", VecNormalX_[ss*nInputs_]);
      for (ii = 1; ii < nInputs_; ii++)
        fprintf(fp,", %24.16e ", VecNormalX_[ss*nInputs_+ii]);
      fprintf(fp,"],\n");
    }
    fprintf(fp,"]\n");
    fprintf(fp,"SamOut = [\n");
    for (ss = 0; ss < nSamples_; ss++)
      fprintf(fp," %24.16e ,\n", VecNormalY_[ss]);
    fprintf(fp,"]\n");
    fprintf(fp,"###################################################\n");
    fprintf(fp,"# interpolation function  \n");
    fprintf(fp,"# X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp,"# X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp,"# ... \n");
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"def interpolate(XX): \n");
    fprintf(fp,"  nSamp = int(len(XX) / %d + 1.0e-8)\n", nInputs_);
    fprintf(fp,"  X  = %d * [0.0]\n", nInputs_);
    fprintf(fp,"  Ys = 2 * nSamp * [0.0]\n");
    fprintf(fp,"  Dists = nSamples * [0.0]\n");
    fprintf(fp,"  YT    = (2 * K) * [0.0]\n");
    fprintf(fp,"  for ss2 in range(nSamp) : \n");
    fprintf(fp,"    for ii in range(%d) : \n", nInputs_);
    fprintf(fp,"      X[ii] = XX[ss2*%d+ii]\n",nInputs_);
    fprintf(fp,"    cnt = 0;\n");
    fprintf(fp,"    for ss in range(nSamples) : \n");
    fprintf(fp,"      dist = 0.0\n");
    fprintf(fp,"      for ii in range(nInps) : \n");
    fprintf(fp,"        dd = X[ii]\n");
    fprintf(fp,"        dd = (dd-LBs[ii])/(UBs[ii]-LBs[ii])\n");
    fprintf(fp,"        dd = dd - Sample[ss][ii]\n");
    fprintf(fp,"        dist += dd * dd\n");
    fprintf(fp,"        if (cnt>0 and cnt<K and dist>Dists[cnt-1]) : \n");
    fprintf(fp,"          break\n");
    fprintf(fp,"      if (dist < 1.0e-16) :\n");
    fprintf(fp,"        YT[0] = SamOut[ss]\n");
    fprintf(fp,"        cnt = 1\n"); 
    fprintf(fp,"        Dists[0] = 1.0\n"); 
    fprintf(fp,"        break\n");
    fprintf(fp,"      if (cnt < K) :\n");
    fprintf(fp,"        Dists[cnt] = dist\n");
    fprintf(fp,"        YT[cnt] = SamOut[ss]\n");
    fprintf(fp,"        cnt = cnt + 1\n");
    fprintf(fp,"        for kk2 in range(cnt-1) : \n");
    fprintf(fp,"          kk = cnt - 1 - kk2\n");
    fprintf(fp,"          if (Dists[kk] < Dists[kk-1]) :\n");
    fprintf(fp,"            dd = Dists[kk]\n");
    fprintf(fp,"            Dists[kk] = Dists[kk-1]\n");
    fprintf(fp,"            Dists[kk-1] = dd\n");
    fprintf(fp,"            dd = YT[kk]\n");
    fprintf(fp,"            YT[kk] = YT[kk-1]\n");
    fprintf(fp,"            YT[kk-1] = dd\n");
    fprintf(fp,"      else :\n"); 
    fprintf(fp,"        if (dist < Dists[cnt-1]) :\n"); 
    fprintf(fp,"          Dists[cnt-1] = dist\n"); 
    fprintf(fp,"          YT[cnt-1] = SamOut[ss]\n"); 
    fprintf(fp,"          for kk2 in range(cnt-1) : \n");
    fprintf(fp,"            kk = cnt - 1 - kk2\n");
    fprintf(fp,"            if (Dists[kk] < Dists[kk-1]) :\n");
    fprintf(fp,"              dd = Dists[kk];\n");
    fprintf(fp,"              Dists[kk] = Dists[kk-1]\n");
    fprintf(fp,"              Dists[kk-1] = dd\n");
    fprintf(fp,"              dd = YT[kk]\n");
    fprintf(fp,"              YT[kk] = YT[kk-1]\n");
    fprintf(fp,"              YT[kk-1] = dd\n");
    fprintf(fp,"    dd = 0.0\n");
    fprintf(fp,"    Y  = 0.0\n");
    fprintf(fp,"    for ss in range(cnt) : \n");
    fprintf(fp,"      Y += YT[ss] / Dists[ss]\n");
    fprintf(fp,"      dd += 1.0 / Dists[ss]\n");
    fprintf(fp,"    Y /= dd\n");
    fprintf(fp,"    Ys[ss2*2] = Y\n");
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
    printf("FILE psuade_rs.py contains the final Legendre polynomial\n");
    printf("     functional form.\n");
    fclose(fp);
  }
  return;
}

