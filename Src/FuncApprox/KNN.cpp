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
#include "Sampling.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

psVector KNN_VecX_;
psVector KNN_VecY_;
psVector KNN_VecSigmas_;
int      KNN_nSamples_;
int      KNN_kfold_;
double   KNN_currY_;
double   KNN_kmax_;
int      KNN_currK_;
psVector KNN_VecYStore_;
psVector KNN_VecDists_;
int      KNN_numIts_; 

// ************************************************************************
// ************************************************************************
// external functions
// ------------------------------------------------------------------------
extern "C" {
  void kbobyqa_(int *,int *, double *, double *, double *, double *,
                double *, int *, int *, double*);
}

// ************************************************************************
// resident function to perform evaluation
// This function performs k-fold cross validation
// ------------------------------------------------------------------------
#ifdef __cplusplus
extern "C"
{
#endif
  void *knnevalfunc_(int *nInps, double *SValues, double *YValue)
  {
    int    ii, kk, ss, ss2, ss3, nInputs, nSubSamples, nTrain, kcnt;
    double error, ddata, dist, sumDist, YEst;
    psVector vecX2, vecY2;

    KNN_numIts_++; 
    //**/ search for neighbors if this is empty
    int searchFlag=0;
    if (KNN_VecYStore_.length() == 0) 
    {
      searchFlag = 1;
      KNN_VecDists_.setLength(KNN_nSamples_*KNN_currK_);
      KNN_VecYStore_.setLength(KNN_nSamples_*KNN_currK_);
    }

    //**/ get ready for search
    nInputs = KNN_VecX_.length() / KNN_VecY_.length();
    vecX2.setLength(KNN_nSamples_*nInputs);
    vecY2.setLength(KNN_nSamples_);
    nSubSamples = KNN_nSamples_ / KNN_kfold_;
    if (nSubSamples == 0) nSubSamples = 1;

    //**/ for each fold do the following
    error = 0.0;
    for (ss = 0; ss < KNN_nSamples_; ss+=nSubSamples)
    {
      //**/ put a training set into (vecX2, vecY2)
      for (ss2 = 0; ss2 < ss*nInputs; ss2++) 
        vecX2[ss2] = KNN_VecX_[ss2];
      for (ss2 = 0; ss2 < ss; ss2++) vecY2[ss2] = KNN_VecY_[ss2];
      nTrain = ss;
      for (ss2 = ss+nSubSamples; ss2 < KNN_nSamples_; ss2++)
      {
        for (ii = 0; ii < nInputs; ii++)
          vecX2[(ss2-nSubSamples)*nInputs+ii] = 
                   KNN_VecX_[ss2*nInputs+ii];
        vecY2[ss2-nSubSamples] = KNN_VecY_[ss2];
        nTrain++;
      }

      //**/ now we have separated into training and test sets, compute
      //**/ test error (test set in KNN_VecX_[ss2*nInputs] on)
      for (ss2 = ss; ss2 < ss+KNN_nSamples_-nTrain; ss2++)
      {
        //**/ compute Y of test point ss2 wrt to training points
        //**/ but first identify neighbors of point ss2
        if (searchFlag == 1)
        {
          kcnt = 0;
          for (ss3 = 0; ss3 < nTrain; ss3++) 
          {
            dist = 0.0;
            //**/ find distance between training and test sample point 
            for (ii = 0; ii < nInputs; ii++) 
            {
              ddata = KNN_VecX_[ss2*nInputs+ii];
              ddata -= vecX2[ss3*nInputs+ii];
              dist += ddata * ddata;
            }
            //**/ If test coincides with training sample, use sample 
            //**/ output of training sample
            if (dist == 0.0) 
            {
              KNN_VecDists_[ss2*KNN_currK_] = 0;
              KNN_VecYStore_[ss2*KNN_currK_] = vecY2[ss3];
              for (kk = 1; kk < KNN_currK_; kk++)
              {
                KNN_VecDists_[ss2*KNN_currK_+kk] = 0;
                KNN_VecYStore_[ss2*KNN_currK_+kk] = 0;
              }
              break;
            }
            else
            {
              //**/ compile k neighbors
              if (kcnt < KNN_currK_)
              {
                KNN_VecDists_[ss2*KNN_currK_+kcnt] = dist;
                KNN_VecYStore_[ss2*KNN_currK_+kcnt] = vecY2[ss3];
                kcnt++;
                for (kk = kcnt-1; kk > 0; kk--)
                {
                  if (KNN_VecDists_[ss2*KNN_currK_+kk] < 
                      KNN_VecDists_[ss2*KNN_currK_+kk-1])
                  {
                    ddata = KNN_VecDists_[ss2*KNN_currK_+kk];
                    KNN_VecDists_[ss2*KNN_currK_+kk] = 
                             KNN_VecDists_[ss2*KNN_currK_+kk-1];
                    KNN_VecDists_[ss2*KNN_currK_+kk-1] = ddata;
                    ddata = KNN_VecYStore_[ss2*KNN_currK_+kk];
                    KNN_VecYStore_[ss2*KNN_currK_+kk] = 
                             KNN_VecYStore_[ss2*KNN_currK_+kk-1];
                    KNN_VecYStore_[ss2*KNN_currK_+kk-1] = ddata;
                  }
                }
              }
              else
              {
                if (dist < KNN_VecDists_[ss2*KNN_currK_+kcnt-1])
                {
                  KNN_VecDists_[ss2*KNN_currK_+kcnt-1] = dist;
                  KNN_VecYStore_[ss2*KNN_currK_+kcnt-1] = KNN_VecY_[ss3];
                  for (kk = kcnt-1; kk > 0; kk--)
                  {
                    if (KNN_VecDists_[ss2*KNN_currK_+kk] < 
                        KNN_VecDists_[ss2*KNN_currK_+kk-1])
                    {
                      ddata = KNN_VecDists_[ss2*KNN_currK_+kk];
                      KNN_VecDists_[ss2*KNN_currK_+kk] = 
                              KNN_VecDists_[ss2*KNN_currK_+kk-1];
                      KNN_VecDists_[ss2*KNN_currK_+kk-1] = ddata;
                      ddata = KNN_VecYStore_[ss2*KNN_currK_+kk];
                      KNN_VecYStore_[ss2*KNN_currK_+kk] = 
                             KNN_VecYStore_[ss2*KNN_currK_+kk-1];
                      KNN_VecYStore_[ss2*KNN_currK_+kk-1] = ddata;
                    }
                  }
                }
              }
            }
          } /* ss3 */
        }
        //**/ KNN_VecDists_ and KNN_VecYStore_ have been 
        //**/ constructed, next is to perform interpolation
        YEst = sumDist = 0.0;
        for (kk = 0; kk < KNN_currK_; kk++) 
        {
          ddata = pow(KNN_VecDists_[ss2*KNN_currK_+kk]/SValues[kk],2.0);
          ddata = exp(-ddata);
          YEst += KNN_VecYStore_[ss2*KNN_currK_+kk] * ddata;
          sumDist += ddata;
        }
        YEst = YEst / sumDist;
        error += pow(YEst-KNN_VecY_[ss2], 2.0);
      }
    }
    error = error / (double) KNN_nSamples_;
    error = sqrt(error);
    YValue[0] = error;
    if (error < KNN_currY_)
    {
      KNN_currY_ = error;
      for (kk = 0; kk < KNN_currK_; kk++) 
        KNN_VecSigmas_[kk] = SValues[kk]; 
    }
    return NULL;
  }
#ifdef __cplusplus
}
#endif

// ************************************************************************
// Constructor for object class KNN
// ------------------------------------------------------------------------
KNN::KNN(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  char pString[101];

  //**/ set identifier
  faID_ = PSUADE_RS_KNN;
  //**/ 0: linear combination, 1: Gaussian kernel, 2: classification
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
    snprintf(pString,100,"Enter number of nearest neighbors (>= 0, <= %d): ",
             kmax_);
    k_ = getInt(0, kmax_, pString);
    snprintf(pString,100,"RS_KNN_k = %d", k_);
    psConfig_.putParameter(pString);
    if (k_ == 0)
    {
      snprintf(pString,100,"Maximum K to be searched (>1,<=%d): ",kmax_);
      kmax_ = getInt(2, kmax_, pString);
      snprintf(pString,100,"RS_KNN_kmax = %d", kmax_);
      psConfig_.putParameter(pString);
      snprintf(pString,100,"How many fold cross validation (10-%d)? ",kfold_);
      kfold_ = getInt(10, kfold_, pString);
      snprintf(pString,100,"RS_KNN_kfold = %d", kfold_);
      psConfig_.putParameter(pString);
    }
    printf("There are two options for interpolation: \n");
    printf("0: Linear interpolation from neighboring points.\n");
    printf("1: Exponential interpolation from neighboring points.\n");
    printf("2: Mode of neigbors (for classification: with integer outputs.\n");
    snprintf(pString,100,"Enter desired mode (0, 1, or 2): ");
    mode_ = getInt(0, 2, pString);
    snprintf(pString,100,"RS_KNN_mode = %d", mode_);
    psConfig_.putParameter(pString);
  }
  else
  {
    char keyword[1000], equalSign[100];
    char *cString = psConfig_.getParameter("RS_KNN_mode");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", keyword, equalSign, &mode_);
      if (mode_ < 0 || mode_ > 2) mode_ = 0;
      if (outputLevel_ > 1) printf("KNN_mode = %d\n", mode_);
    }
    cString = psConfig_.getParameter("RS_KNN_k");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", keyword, equalSign, &k_);
      if (k_ < 0) k_ = 0;
      if (outputLevel_ > 1) printf("KNN_k = %d\n", k_);
    }
    cString = psConfig_.getParameter("RS_KNN_kfold");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", keyword, equalSign, &kfold_);
      if (kfold_ < 0) kfold_ = 0;
      if (outputLevel_ > 1) printf("KNN_kfold = %d\n", kfold_);
    }
    cString = psConfig_.getParameter("RS_KNN_kmax");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", keyword, equalSign, &kmax_);
      if (kmax_ < k_) kmax_ = k_;
      if (outputLevel_ > 1) printf("KNN_kmax = %d\n", kmax_);
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
  VecNormalY_.setLength(nSamples_);
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
  if (mode_ == 1)
  {
    //**/ randomize the order of the training sample ==> VecXX
    //**/ this is needed for cross validation to ensure randomness
    VecIT.setLength(nSamples_);
    VecXX.setLength(nSamples_*nInputs_);
    VecYY.setLength(nSamples_);
    generateRandomIvector(nSamples_, VecIT.getIVector());
    for (ii = 0; ii < nInputs_; ii++)
    {
      for (ss = 0; ss < nSamples_; ss++)
        VecXX[VecIT[ss]*nInputs_+ii] = VecNormalX_[ss*nInputs_+ii];
    }
    for (ss = 0; ss < nSamples_; ss++) VecYY[VecIT[ss]] = Y[ss];
    trainGaussian(VecXX, VecYY);
  }
  else
  {
    //**/ randomize the order of the training sample ==> VecXX
    //**/ this is needed for cross validation to ensure randomness
    VecIT.setLength(nSamples_);
    VecXX.setLength(nSamples_*nInputs_);
    VecYY.setLength(nSamples_);
    generateRandomIvector(nSamples_, VecIT.getIVector());
    for (ii = 0; ii < nInputs_; ii++)
    {
      for (ss = 0; ss < nSamples_; ss++)
        VecXX[VecIT[ss]*nInputs_+ii] = VecNormalX_[ss*nInputs_+ii];
    }
    for (ss = 0; ss < nSamples_; ss++) VecYY[VecIT[ss]] = Y[ss];

    //**/ set up to search k or just use one k_
    //**/ if k_=0, search k = 1 to kmax
    int kbeg, kend;
    if (k_ == 0)
    {
      kbeg = 1;
      kend = kmax_;
    }
    else
    {
      kbeg = k_;
      kend = k_;
    }

    //**/ get ready for search for mode=0,2 (k-fold cross validation)
    //**/ VecX2 and VecY2 are for selecting training points
    double *arrayXX = VecXX.getDVector();
    double *arrayYY = VecYY.getDVector();
    VecX2.setLength(nSamples_*nInputs_);
    VecY2.setLength(nSamples_);
    nSubSamples = nSamples_ / kfold_;
    if (nSubSamples == 0) nSubSamples = 1;
    errMin = 1e35;
    for (kk = kbeg; kk <= kend; kk++)
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
          VecY2[ss2-nSubSamples] = VecYY[ss2];
          count++;
        }
        //**/ train and test it on the hold out
        error += computeTestError(count,VecX2.getDVector(),VecY2.getDVector(),
                    kk,nSamples_-count, &arrayXX[ss*nInputs_],&arrayYY[ss]);
      }
      if (error < errMin)
      {
        k_ = kk;
        errMin = error;
      }
      if (psConfig_.InteractiveIsOn() && kbeg != kend)
        printf("   K=%d (%d) : error = %e\n", kk, kend, error);
    }
    if (psConfig_.InteractiveIsOn() && kbeg != kend) 
      printf("KNN: K selected = %d\n", k_);
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

  //**/ construct a nearest neighbor list for X
  for (ss = 0; ss < nSamp; ss++) 
  {
    dist = 0.0;
    //**/ find distance between X and training sample point ss
    for (ii = 0; ii < nInputs_; ii++) 
    {
      ddata = X[ii];
      if (normFlag == 1)
        ddata = (ddata - VecLBs_[ii]) * VecRanges_[ii];
      ddata -= XN[ss*nInputs_+ii];
      dist += ddata * ddata;
    }
    //**/ If X coincides with training sample s, return sample output of s
    if (dist == 0.0)
    {
      Y = YN[ss];
      return Y;
    }
    //**/ if less than knn so far, add training sample point s to the list
    if (count < knn)
    {
      VecDistances_[count] = dist;
      VecYStored_[count] = YN[ss];
      count++;
      //**/ update VecDistances so that distance[kk] >= distance[kk-1]
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
    //**/ if the list has knn element and the current point ss is nearer
    //**/ than some point on the list, swap the points
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

  //**/ Now VecDistances and VecYStored have been constructed, next is
  //**/ to perform interpolation
  //**/ linear kernel
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
  //**/ Gaussian kernel
  else if (mode_ == 1)
  {
    Y = sumDist = 0.0;
    for (ss = 0; ss < count; ss++) 
    {
      ddata = exp(-VecDistances_[ss]*VecDistances_[ss] /
                  (VecSigmas_[ss]*VecSigmas_[ss]));
      Y += VecYStored_[ss] * ddata;
      sumDist += ddata;
    }
    Y /= sumDist;
  }
  //**/ For classification: search for label with largest frequency
  else
  {
    //**/ sort in ascending order
    sortDbleList(count, VecYStored_.getDVector());
    maxcnt = 0;
    cnt = 1;
    //**/ search for label with maxcnt and set Y to that label
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
// Train Gaussian kernel (incoming vecX has been randomized)
// ------------------------------------------------------------------------
double KNN::trainGaussian(psVector &vecX, psVector &vecY)
{
  int    ii, jj, kk, pLevel=1112, kbeg, kend;
  double errMin = 1e35, ddata;

  //**/ set up for optimization
  psVector vecUBs, vecLBs;
  vecUBs.setLength(kmax_);
  vecLBs.setLength(kmax_);
  for (ii = 0; ii < kmax_; ii++) 
  { 
    vecLBs[ii] = 0.1;
    vecUBs[ii] = 3.0;
  }
  if (k_ == 0) 
  {
    kbeg = 1; 
    kend = kmax_;
  }
  else kbeg = kend = k_;

  int    maxfun=5000;
  double rhobeg, rhoend;
  psVector vecTVals, vecW;
  vecTVals.setLength(kmax_+1);
  double *TValues = vecTVals.getDVector();
  int nPts = (kmax_ + 1) * (kmax_ + 2) / 2;
  jj = (nPts+13) * (nPts+kmax_) + 3*kmax_*(kmax_+3)/2;
  kk = (nPts+5)*(nPts+kmax_)+3*kmax_*(kmax_+5)/2+1;
  if (jj > kk) vecW.setLength(jj);
  else         vecW.setLength(kk);

  //**/ to facilitate callling external optimizer
  KNN_VecX_ = vecX;
  KNN_VecY_ = vecY;
  KNN_nSamples_ = nSamples_;
  KNN_kfold_ = kfold_;
  KNN_kmax_ = kmax_;

  //**/ create a number of iniital guesses 
  int nSamp = 5;
  Sampling *sampler = SamplingCreateFromID(PSUADE_SAMP_LHS);
  sampler->setInputBounds(kmax_,vecLBs.getDVector(),vecUBs.getDVector());
  sampler->setOutputParams(1);
  sampler->setSamplingParams(nSamp, 1, 0);
  sampler->initialize(0);
  psVector  vecXS, vecYS;
  psIVector vecIS;
  vecIS.setLength(nSamp);
  vecXS.setLength(nSamp*kmax_);
  vecYS.setLength(nSamp);
  sampler->getSamples(nSamp,kmax_,1,vecXS.getDVector(),
                      vecYS.getDVector(), vecIS.getIVector());
  delete sampler;

  //**/ search for each k
  int ss;
  psVector VecBestSigmas;
  for (kk = kbeg; kk <= kend; kk++)
  {
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
      printf("KNN INFO: Search for %d nearest neighbors.\n",kk);
    //**/ send callback function some initial information
    KNN_currK_ = kk;
    KNN_currY_ = 1e35;
    //**/ for storing best sigmas for this optimizaiton 
    KNN_VecSigmas_.setLength(kk);
    //**/ optimizer parameter setting
    nPts = (kk + 1) * (kk + 2) / 2;

    //**/ these two arrays are for speeding up the callback function
    KNN_VecDists_.clean();
    KNN_VecYStore_.clean();

    //**/ optimization using some initial guesses
    for (ss = 0; ss < nSamp; ss++) 
    {
      for (ii = 0; ii < kmax_; ii++) vecTVals[ii] = vecXS[ss*kmax_+ii];
      if (outputLevel_ > 1)
      {
        for (ii = 0; ii < kk; ii++) 
          printf("Initial guess %d = %e\n",ii+1,vecTVals[ii]);
      }
      //**/ tell BOBYQA to call back 
      pLevel = 1112;

      double rhobeg = vecUBs[0] - vecLBs[0];
      for (ii = kbeg; ii < kend; ii++)
      {
        ddata = vecUBs[ii] - vecLBs[ii];
        if (ddata < rhobeg) rhobeg = ddata;
      }
      rhobeg *= 0.5;
      rhoend = rhobeg * 1.0e-8;
#ifdef HAVE_BOBYQA
      KNN_numIts_ = 0;
      kbobyqa_(&kk,&nPts,vecTVals.getDVector(),vecLBs.getDVector(),
               vecUBs.getDVector(),&rhobeg,&rhoend,&pLevel,&maxfun, 
               vecW.getDVector());
#else
      printf("KNN ERROR: BOBYQA optimizer not installed.\n");
      exit(1);
#endif

      if (KNN_currY_ < errMin)
      {
        k_ = kk;
        errMin = KNN_currY_;
        VecBestSigmas = KNN_VecSigmas_; 
      }
      printf("K=%d, sample=%d : Number of iterations = %d\n",kk,ss+1,
             KNN_numIts_); 
    }
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 1)
      printf("K=%d (%d) : error = %e\n", kk, kend, KNN_currY_);
  }
  if (psConfig_.InteractiveIsOn() && kbeg != kend) 
    printf("KNN: K selected = %d\n", k_);
  VecSigmas_ = VecBestSigmas;
  return 0;
}

// ************************************************************************
// compute error on test set
// ------------------------------------------------------------------------
double KNN::computeTestError(int nSamp, double *X, double *Y, int knn, 
                             int nTests, double *XTest, double *YTest)
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

